"""
tests/test_behavioral.py
Fase 5: geração (sementes, decodificação explícita), extração de
candidatos e indicadores, taxas, dominante empírico, perguntas do
relatório, amostra de validação e concordância.
"""

import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import analyze
import check_results
import run_behavioral
from behavior.extraction import Extractor
from behavior.generation import Generator, effective_config, generation_configs, prompt_seed
from behavior.validation import agreement, cohen_kappa, stratified_sample, wilson
from config.experiment_config import ExperimentConfig
from config.loader import (
    CONFIG_DIR, UnconfirmedError, load_behavioral, load_experiment, load_facts,
    parse_behavioral, require_confirmed, validate_behavioral,
)
from conftest import WordTokenizer, build_tiny_model
from logit_lens.store import behavioral_dir
from scripts import fill_empirical_dominant as fill_ed

FACTS = load_facts()
EXP   = load_experiment()
B     = load_behavioral()
BC    = dataclasses.replace(B, status="confirmed")
CHAT  = lambda tok, text: f"<user>{text}<assistant>"  # noqa: E731
G     = {**B.generation, "max_new_tokens": 8, "samples": 3}


def _raw():
    return yaml.safe_load((CONFIG_DIR / "behavioral.yaml").read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════

def test_config_valid_and_pending():
    assert validate_behavioral(EXP, FACTS, B) == []
    assert B.status == "confirmed"
    g = B.generation
    assert (g["greedy"], g["samples"], g["temperature"], g["top_p"], g["top_k"],
            g["max_new_tokens"]) == (1, 10, 0.7, 1.0, 0, 200)
    with pytest.raises(UnconfirmedError, match="formas de superfície"):
        require_confirmed({}, [], behavioral=dataclasses.replace(B, status="proposed"))
    require_confirmed({}, [], behavioral=B)


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r["surface_forms"]["telefone"].pop("reis"), "sem o candidato 'reis'"),
    (lambda r: r["surface_forms"]["telefone"]["meucci"]["latin"].append("Bell"), "em bell e meucci"),
    (lambda r: r["surface_forms"]["telefone"]["reis"]["exclude"].update(pt=["Rey"]), "não é forma"),
    (lambda r: r["report"]["precedence"][0].update(first="gray"), "desconhecido"),
    (lambda r: r["generation"].update(temperature=0), "temperature"),
    (lambda r: r["surface_forms"]["aviao"]["wright"]["latin"].append("Wríght"), None),   # mesmo dono: ok
])
def test_validation_catches(mutate, msg):
    raw = copy.deepcopy(_raw())
    mutate(raw)
    errs = validate_behavioral(EXP, FACTS, parse_behavioral(raw))
    assert (errs == []) if msg is None else any(msg in e for e in errs), errs


# ══════════════════════════════════════════════════════════════
# Geração
# ══════════════════════════════════════════════════════════════

def _gen(model=None, g=G):
    model = model or build_tiny_model("llama", vocab=4096)
    return Generator(model, WordTokenizer(), CHAT, g, seed=42), model


def _qa(fact="telefone", lang="it", par=1):
    from config.loader import prompts_for, load_prompts
    return prompts_for(load_prompts(), fact, lang, "QA")[par - 1]


def test_decoding_ignores_model_defaults():
    model = build_tiny_model("llama")
    gc = model.generation_config
    gc.top_p, gc.top_k, gc.min_p, gc.repetition_penalty, gc.typical_p = 0.5, 7, 0.3, 1.3, 0.8
    greedy, sample = generation_configs(model, WordTokenizer(), B.generation)
    eff = effective_config(model, sample)
    assert eff["do_sample"] and eff["temperature"] == 0.7
    assert (eff["top_p"], eff["top_k"], eff["min_p"], eff["typical_p"],
            eff["repetition_penalty"]) == (1.0, 0, 0.0, 1.0, 1.0)
    assert eff["max_new_tokens"] == 200
    assert not effective_config(model, greedy)["do_sample"]


def test_answers_structure_seed_and_determinism():
    gen, model = _gen()
    p = _qa()
    a = gen.answers(p, "llama")
    assert [r["sample_id"] for r in a] == [0, 1, 2, 3]
    assert [r["kind"] for r in a] == ["greedy", "sample", "sample", "sample"]
    assert {r["seed"] for r in a[1:]} == {prompt_seed(42, "llama", "telefone", "it", 1)}
    assert all(r["n_new_tokens"] <= 8 for r in a)
    b = _gen(model)[0].answers(p, "llama")
    assert [r["text"] for r in a] == [r["text"] for r in b]          # sementes fixas
    assert prompt_seed(42, "llama", "telefone", "it", 1) != prompt_seed(42, "llama", "telefone", "it", 2)


def test_greedy_is_argmax_decoding():
    gen, model = _gen(g={**G, "samples": 0})
    p = _qa()
    tok = gen.tok
    ids = tok(CHAT(tok, p.text), add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids])
    new = []
    with torch.no_grad():
        for _ in range(8):
            nxt = model(input_ids=x).logits[0, -1].argmax().item()
            if nxt == model.generation_config.eos_token_id:
                break
            new.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    assert gen.answers(p, "llama")[0]["text"] == tok.decode(new).strip()


def test_run_behavioral_resumes(tmp_path):
    from test_run_pipeline import tiny_config
    cfg = tiny_config()
    b = dataclasses.replace(B, generation=G)
    model = build_tiny_model("llama", vocab=4096)
    out = behavioral_dir(str(tmp_path), "llama")
    m = run_behavioral.run_model("llama", model, WordTokenizer(), cfg, out, CHAT, behavioral=b,
                                 log=lambda *a: None)
    n_qa = sum(p.type == "QA" for p in cfg.prompt_specs)
    a = pd.read_parquet(out / "answers.parquet")
    assert len(a) == n_qa * 4 and m["units_done"] == m["units_expected"]
    assert m["effective_sample"]["top_p"] == 1.0
    m = run_behavioral.run_model("llama", model, WordTokenizer(), cfg, out, CHAT, behavioral=b,
                                 log=lambda *a: None)
    assert m["sessions"][-1]["units_run"] == 0
    assert run_behavioral.behavioral_units(cfg) == [f"{f}__{l}" for f in cfg.facts for l in cfg.languages]


# ══════════════════════════════════════════════════════════════
# Extração
# ══════════════════════════════════════════════════════════════

EX = Extractor(FACTS, BC, EXP["languages"])


@pytest.mark.parametrize("fact,lang,text,primary,order", [
    ("telefone", "it", "Fu Antonio Meucci, anche se Bell brevettò.", "meucci", ["meucci", "bell"]),
    ("telefone", "it", "Una bella invenzione di Campbell.", "none", []),
    ("telefone", "pt", "Os reis apoiaram Graham Bell.", "bell", ["bell"]),              # 'reis' excluído
    ("telefone", "pt", "Philipp Reis antes de Bell.", "reis", ["reis", "bell"]),
    ("telefone", "de", "Johann Philipp Reis, nicht Bell.", "reis", ["reis", "bell"]),
    ("aviao", "fr", "SANTOS-DUMONT a volé avant les frères Wright", "santos_dumont", ["santos_dumont", "wright"]),
    ("aviao", "pt", "os irmãos Wright e Santos Dumont", "wright", ["wright", "santos_dumont"]),
    ("cinema", "fr", "la lumière du projecteur; Edison", "edison_cinema", ["edison_cinema"]),
    ("cinema", "fr", "les frères Lumière ont inventé", "lumiere", ["lumiere"]),
    ("cinema", "en", "the Lumiere brothers", "lumiere", ["lumiere"]),                    # sem acento
    ("fotografia", "en", "Daguerre; Florence in Italy", "daguerre", ["daguerre"]),        # cidade
    ("fotografia", "pt", "Hércules Florence, em Campinas", "florence", ["florence"]),
    ("fotografia", "en", "Nicephore Niepce", "niepce", ["niepce"]),
])
def test_candidate_extraction(fact, lang, text, primary, order):
    c = EX.candidates(text, fact, lang)
    assert c["primary"] == primary and c["order"] == order and c["also_cited"] == order[1:]


def test_indicator_extraction():
    rows = EX.indicators("Os Wright, mas o 14-bis voou em Bagatelle.", "aviao")
    hit = {(r["candidate"], r["indicator"]) for r in rows if r["mentioned"]}
    assert ("santos_dumont", "14-bis") in hit
    assert all(r["indicator_type"] in ("place", "artifact") for r in rows)


# ══════════════════════════════════════════════════════════════
# Taxas, dominante empírico, relatório
# ══════════════════════════════════════════════════════════════

def _answers(spec):
    """spec: [(model, fact, lang, texto)] -> tabela answers (sample_id sequencial)."""
    rows = []
    for i, (m, f, l, t) in enumerate(spec):
        rows.append({"model": m, "fact": f, "lang": l, "prompt_type": "QA", "paraphrase": 1,
                     "sample_id": i, "kind": "greedy" if i % 2 == 0 else "sample",
                     "question": "?", "text": t, "n_new_tokens": 5, "finished": True, "seed": 1})
    return pd.DataFrame(rows)


def test_behavioral_metrics():
    spec = ([("m1", "telefone", "en", "Bell")] * 3 + [("m1", "telefone", "en", "Meucci")]
            + [("m1", "telefone", "en", "nobody")] + [("m2", "telefone", "en", "Meucci then Bell")] * 2
            + [("m1", "telefone", "de", "Reis vor Meucci")] * 2 + [("m1", "telefone", "de", "Meucci, dann Reis")]
            + [("m1", "telefone", "de", "Bell")]
            + [("m1", "aviao", l, t) for l, t in [("fr", "Santos-Dumont"), ("fr", "Wright"),
                                                 ("en", "Wright"), ("de", "Wright"), ("it", "Santos Dumont")]])
    res = analyze.compute_behavioral(_answers(spec), ExperimentConfig(), BC)
    r = res["behavioral_rates"]
    m1 = r[(r.kind == "all") & (r.model == "m1") & (r.fact == "telefone") & (r.lang == "en")].set_index("candidate")
    assert m1.loc["bell", "primary_rate"] == pytest.approx(3 / 5)
    assert m1.loc["meucci", "mention_rate"] == pytest.approx(1 / 5)

    ed = res["empirical_dominant"].set_index(["scope", "fact"])
    pooled = ed.loc[("pooled", "telefone")]
    # pooled: bell 3, meucci 3 -> empate; "nobody" fora do denominador
    assert pooled.empirical_dominant == "tie" and pooled.n_none == 1 and pooled.n_named == 6
    assert ed.loc[("m1", "telefone")].empirical_dominant == "bell"
    assert ed.loc[("m1", "telefone")].agrees
    assert ed.loc[("m2", "telefone")].empirical_dominant == "meucci"

    rep = res["behavioral_report"]
    pre = rep[(rep.question == "precedence:telefone:reis<meucci") & (rep.scope == "pooled")
              & (rep.lang == "de")].iloc[0]
    assert pre.target_lang and pre.n_citing_either == 3 and pre.value == pytest.approx(2 / 3)
    assert pre.value_among_both == pytest.approx(2 / 3)
    men = rep[(rep.question == "mention_rate:aviao:santos_dumont") & (rep.scope == "pooled")].set_index("lang")
    assert men.loc["fr", "value"] == pytest.approx(0.5)
    assert men.loc["fr", "compare_mean"] == pytest.approx((0 + 0 + 1) / 3)

    ir = res["indicator_rates"]
    assert set(ir["given_primary"]) >= {"any", "bell"}


def test_analyze_skips_behavioral_when_unconfirmed(tmp_path, capsys, monkeypatch):
    import config.loader
    monkeypatch.setattr(config.loader, "load_behavioral",
                        lambda *a, **k: dataclasses.replace(B, status="proposed"))
    out = behavioral_dir(str(tmp_path), "m1")
    out.mkdir(parents=True)
    _answers([("m1", "telefone", "en", "Bell")]).to_parquet(out / "answers.parquet")
    assert analyze.main(["--results-dir", str(tmp_path), "--output-dir", str(tmp_path / "metrics")]) == 0
    assert "comportamental pulado" in capsys.readouterr().out
    with pytest.raises(ValueError):
        analyze._guard_output(str(tmp_path / "behavioral" / "x"), str(tmp_path))


def test_fill_empirical_dominant_edits_only_that_field():
    text = ("facts:\n  aviao:\n    kind: disputed\n    empirical_dominant: null\n"
            "  telefone:\n    empirical_dominant: null   # comentário\n")
    new = fill_ed.fill(text, {"telefone": "bell"})
    assert new == text.replace("empirical_dominant: null   # comentário",
                               "empirical_dominant: bell   # comentário")


# ══════════════════════════════════════════════════════════════
# Validação manual
# ══════════════════════════════════════════════════════════════

def test_stratified_sample():
    spec = [(m, "telefone", l, "Bell") for m in ("m1", "m2") for l in ("pt", "de") for _ in range(30)]
    ans = _answers(spec)
    attr = pd.DataFrame(Extractor(FACTS, BC, EXP["languages"]).extract(ans.to_dict("records"))["attribution"])
    s = stratified_sample(ans, attr, 13, 42, {"telefone": "bell=Bell"})
    assert s.groupby(["model", "lang"]).size().tolist() == [13] * 4
    assert (s["auto_primary"] == "bell").all() and (s["human_primary"] == "").all()
    s2 = stratified_sample(ans, attr, 13, 42, {"telefone": "bell=Bell"})
    assert s["uid"].tolist() == s2["uid"].tolist()


def test_kappa_and_agreement():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    lo, hi = wilson(9, 10)
    assert lo < 0.9 < hi
    df = pd.DataFrame({
        "fact": ["telefone"] * 4, "lang": ["pt", "pt", "de", "de"],
        "auto_primary": ["bell", "meucci", "none", "reis"],
        "auto_cited":   ["bell", "meucci,bell", "", "reis"],
        "human_primary": ["bell", "meucci", "bell", ""],             # última sem rótulo
        "human_cited":   ["bell", "meucci", "bell", ""]})
    res = agreement(df, {"telefone": ["bell", "meucci", "reis"]})
    assert res["n_labelled"] == 3
    assert res["primary"]["agreement"] == pytest.approx(2 / 3)
    assert res["primary_by_lang"]["de"]["agreement"] == 0.0
    c = res["cited"]
    assert c["n_pairs"] == 9 and c["precision"] == pytest.approx(2 / 3) and c["recall"] == pytest.approx(2 / 3)


def test_export_script_keeps_existing_sample(tmp_path, capsys):
    # Amostra existente: mantida, com mensagem, e código 0 (não interrompe run_all_server.sh)
    from scripts import export_validation_sample as ev
    out = tmp_path / "s.csv"
    out.write_text("labels")
    assert ev.main(["--output", str(out), "--results-dir", str(tmp_path)]) == 0
    assert out.read_text() == "labels" and "mantida" in capsys.readouterr().out


def test_check_behavioral(tmp_path):
    from test_run_pipeline import tiny_config
    cfg = tiny_config()
    assert "sem respostas" in check_results.check_behavioral("llama", str(tmp_path), cfg)[0]


def test_report_when_a_candidate_is_never_cited():
    # Ninguém cita Reis: a pergunta de precedência não pode quebrar
    spec = [("m1", "telefone", "de", "Meucci")] * 2 + [("m1", "telefone", "en", "Bell")]
    res = analyze.compute_behavioral(_answers(spec), ExperimentConfig(), BC)
    rep = res["behavioral_report"]
    pre = rep[(rep.question == "precedence:telefone:reis<meucci") & (rep.scope == "pooled")
              & (rep.lang == "de")].iloc[0]
    assert pre.n_citing_either == 2 and pre.value == 0.0 and pre.n_citing_both == 0


def test_bell_is_case_sensitive():
    # Decisão da autora: "Bell" só com B maiúsculo, em todo idioma
    assert B.case_sensitive_for("telefone", "bell") == ["Bell"]
    for lang in ("en", "pt", "de", "it", "fr"):
        assert EX.candidates("the bell rang, then Meucci", "telefone", lang)["order"] == ["meucci"]
        assert EX.candidates("Bell, then Meucci", "telefone", lang)["order"] == ["bell", "meucci"]
    assert EX.candidates("graham bell", "telefone", "en")["order"] == ["bell"]   # forma completa
    # posição exata mesmo com ß (casefold muda o comprimento)
    assert EX.candidates("Straße, Meucci, dann Bell", "telefone", "de")["order"] == ["meucci", "bell"]
    assert EX.candidates("Straßeß Bell, dann Meucci", "telefone", "de")["order"] == ["bell", "meucci"]
