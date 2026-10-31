"""
tests/test_token_gaps.py
Correções da auditoria de tokens (2026-10-30, reports/token_gap_audit.md):
separador da resposta QA tirado do chat template (o Mistral era medido no
token errado), títulos / ligações / marcação forçados, top-20 gravado nas
passadas forçadas, nomes novos sem colisão, "Mendeleyew" no
comportamental, limite de tempo por job na fila.
"""

import dataclasses
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest

from config.experiment_config import ExperimentConfig
from config.loader import Prompt, load_forcing, load_models, parse_forcing
from conftest import WordTokenizer, build_tiny_model
from logit_lens.confound import ConfoundTester
from logit_lens.entity_tokens import first_token_after, format_prompt
from logit_lens.measure import LensRunner
from models.chat_template import ChatFormatter, TemplateSeparatorError, answer_separator
from test_confound import CHAT, P_PT2, P_QA, _fact, _main_with_top

ROOT = Path(__file__).resolve().parent.parent
REG = load_models()
CFG = ExperimentConfig()
FORCING = load_forcing()


# ══════════════════════════════════════════════════════════════
# Separador da resposta QA, em todo tokenizer registrado
# ══════════════════════════════════════════════════════════════

QUESTION = "Who invented the telephone?"
PROBES = ["Alexander", "Bell", "The", "Nicéphore", "Sir", "**"]


def _cached_tokenizer(key):
    from transformers import AutoTokenizer
    spec = REG[key]
    try:
        return AutoTokenizer.from_pretrained(spec.id, revision=spec.revision, local_files_only=True)
    except Exception as e:                                   # sem cache local (CI, máquina nova)
        pytest.skip(f"tokenizer de {key} não está no cache do Hugging Face: {type(e).__name__}")


@pytest.mark.parametrize("key", list(REG.models))
def test_qa_measures_the_token_the_template_emits(key):
    """
    O token medido para cada candidato no QA tem que ser o primeiro token
    da resposta que o próprio chat template produz. Com o separador ""
    fixo de antes, falhava no Mistral ('Alex' medido, '▁Alexander' emitido).
    """
    tok = _cached_tokenizer(key)
    if REG[key].kind != "instruct":
        assert "qa" not in REG[key].formats                   # base: sem QA, nada a medir
        if not getattr(tok, "chat_template", None):
            return
    fmt  = ChatFormatter.for_tokenizer(tok, key)
    text = fmt(tok, QUESTION)
    g    = tok.encode(text, add_special_tokens=False)
    for w in PROBES:
        full = tok.apply_chat_template([{"role": "user", "content": QUESTION},
                                        {"role": "assistant", "content": w + " x"}], tokenize=False)
        f = tok.encode(full, add_special_tokens=False)
        assert f[:len(g)] == g, f"{key}: a conversa não começa pelo prompt de geração"
        measured = first_token_after(tok, text, w, sep=fmt.answer_sep, add_special_tokens=False)["token_id"]
        assert measured == f[len(g)], (
            f"{key}: {w!r} medido como {tok.convert_ids_to_tokens([measured])} mas a resposta "
            f"começa por {tok.convert_ids_to_tokens([f[len(g)]])} (separador {fmt.answer_sep!r})")


def test_mistral_separator_is_a_space():
    tok = _cached_tokenizer("mistral")
    assert answer_separator(tok) == " "
    for k in ("llama", "qwen", "gemma", "gemma27"):
        assert answer_separator(_cached_tokenizer(k)) == ""


class _TemplTok(WordTokenizer):
    """Tokenizer de brinquedo com chat template (com ou sem espaço antes da resposta)."""

    def __init__(self, space=True, broken=False):
        super().__init__()
        self.space, self.broken = space, broken
        self.chat_template = "toy"

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<{m['role']}>{m['content']}" if m["role"] == "user" else
                      ("<assistant>" + (" " if self.space else "") + m["content"] + "<end>")
                      for m in messages)
        if self.broken and not add_generation_prompt:
            out = "<sys>" + out                       # conversa completa não começa pelo prompt de geração
        return out + ("<assistant>" if add_generation_prompt else "")


def test_answer_separator_toy_templates():
    assert answer_separator(_TemplTok(space=True)) == " "
    assert answer_separator(_TemplTok(space=False)) == ""
    with pytest.raises(TemplateSeparatorError):
        answer_separator(_TemplTok(broken=True))


def test_format_prompt_uses_formatter_separator():
    tok = _TemplTok(space=True)
    fmt = ChatFormatter.for_tokenizer(tok, "toy")
    text, sep, add = format_prompt(P_QA, tok, fmt)
    assert sep == " " and add is False and text.endswith("<assistant>")
    assert format_prompt(P_QA, tok, CHAT)[1] == ""                     # lambda simples: ""
    assert format_prompt(P_PT2, tok, fmt)[1] == " "                    # completion inalterado


# ══════════════════════════════════════════════════════════════
# Título, palavra de ligação, marcação
# ══════════════════════════════════════════════════════════════

class _PunctTok(WordTokenizer):
    """Separa "**" das letras, como os tokenizers reais ("**Wright" -> "**", "Wright")."""

    def _pieces(self, text):
        import re
        out = []
        for chunk in re.split(r"(<[^>]+>)", text):
            if chunk.startswith("<") and chunk.endswith(">"):
                out.append(chunk)
                continue
            out += re.findall(r"\s?\*\*|\s?[^\s*]+", chunk)
        return out


def _tester(chat=None, tok=None):
    runner = LensRunner(build_tiny_model("llama", vocab=512), tok or WordTokenizer(), chat_formatter=chat)
    return runner, ConfoundTester(runner, FORCING)


def _top(runner, prompt, fact, word, sep=" ", add_special=True):
    res = runner.measure(prompt, fact)
    text, _, _ = runner.prepare(prompt)
    tid = first_token_after(runner.tokenizer, text, word, sep=sep, add_special_tokens=add_special)["token_id"]
    res["layers"][-1]["top_ids"] = [tid] + res["layers"][-1]["top_ids"][1:]
    return res


def test_title_sir_forced_and_names_measured_next():
    runner, tester = _tester()
    main = _top(runner, P_PT2, _fact(), "Sir")
    ps = tester.run(P_PT2, _fact(), main)
    assert [p["pass_id"] for p in ps] == ["title:Sir"]
    st = ps[0]["steps"]
    assert len(st) == 1 and st[0]["forced_text"] == P_PT2.text + " Sir"
    assert st[0]["result"]["token_sets"]["sets"]["wright"]["words"][0]["token"] == " Wright"


def test_william_only_in_raios_x():
    runner, tester = _tester()
    aviao = _fact()
    raios = dataclasses.replace(_fact(), key="raios_x")
    assert tester.run(P_PT2, aviao, _top(runner, P_PT2, aviao, "William")) == []
    ps = tester.run(P_PT2, raios, _top(runner, P_PT2, raios, "William"))
    assert [p["pass_id"] for p in ps] == ["title:William"]


def test_no_title_pass_when_top_is_a_candidate_token():
    runner, tester = _tester()
    f = _fact()
    f.candidates[1].names["pt"]["nom"] = f.candidates[1].names["pt"]["nom"] + ["Sir"]
    assert tester.run(P_PT2, f, _top(runner, P_PT2, f, "Sir")) == []


def test_lead_word_como_then_article(monkeypatch):
    runner, tester = _tester()
    main = _top(runner, P_PT2, _fact(), "como")
    orig = runner.measure_words

    def fake(text, sep, *a, **k):
        res = orig(text, sep, *a, **k)
        if text.endswith(" como"):                                      # depois de "como", top-1 = "os"
            tid = first_token_after(runner.tokenizer, text, "os")["token_id"]
            res["layers"][-1]["top_ids"] = [tid] + res["layers"][-1]["top_ids"][1:]
        return res
    monkeypatch.setattr(runner, "measure_words", fake)
    ps = tester.run(P_PT2, _fact(), main)
    assert ps[0]["pass_id"] == "lead:como"
    assert [s["forced_text"] for s in ps[0]["steps"]] == [P_PT2.text + " como", P_PT2.text + " como os"]
    assert ps[0]["steps"][1]["suffix"] == " como os"
    P_IT = Prompt("aviao", "it", "F2", 1, "L'inventore dell'aereo è considerato")
    f = _fact()
    for c in f.candidates:
        c.names["it"] = {"nom": list(c.names["pt"]["nom"])}
    assert tester.run(P_IT, f, _top(runner, P_IT, f, "come"))[0]["pass_id"] == "lead:come"
    assert tester.run(P_PT2, _fact(), _top(runner, P_PT2, _fact(), "come")) == []   # "come" só em it


def test_format_bold_then_names_without_space():
    runner, tester = _tester(chat=CHAT, tok=_PunctTok())
    main = _main_with_top(runner, P_QA, "**", sep="", add_special=False)
    ps = tester.run(P_QA, _fact(), main)
    assert ps[0]["pass_id"] == "format:**"
    st = ps[0]["steps"][0]
    assert st["forced_text"] == CHAT(None, P_QA.text) + "**"
    words = st["result"]["token_sets"]["sets"]["wright"]["words"]
    assert words[0]["token"] == "Wright"                                 # colado depois de "**"


def test_forcing_config_new_fields_and_validation():
    assert FORCING.titles_for("de", "aviao") == ["Sir"]
    assert FORCING.titles_for("it", "raios_x") == ["Sir", "William"]
    assert FORCING.lead_words == {"pt": ["como"], "it": ["come"]} and FORCING.format_tokens == ["**"]
    from config.loader import validate_forcing
    bad = parse_forcing({"status": "confirmed", "articles": {l: ["x"] for l in CFG.languages},
                         "plural_nouns": {l: ["y"] for l in CFG.languages}, "contractions": {},
                         "fact_titles": {"lampada": ["Thomas"]}, "titles": {"ru": ["Sir"]}})
    errs = validate_forcing(CFG.experiment, bad, CFG.prompt_specs)
    assert any("lampada" in e for e in errs) and any("'ru'" in e for e in errs)


# ══════════════════════════════════════════════════════════════
# Top-20 das passadas forçadas
# ══════════════════════════════════════════════════════════════

def test_forced_passes_store_top_tokens():
    import run_lens
    runner, tester = _tester()
    tables = run_lens.run_fact_unit(runner, tester, CFG, "aviao", "pt", "llama", ("completion",))
    L = pd.DataFrame(tables["layers"])
    F = pd.DataFrame(tables["forced"])
    if not len(F):
        pytest.skip("o modelo aleatório não disparou passadas forçadas neste fato")
    fl = L[L.source == "forced"]
    key = ["prompt_type", "paraphrase", "pass_id", "force_step"]
    assert set(map(tuple, F[key].drop_duplicates().values)) == set(map(tuple, fl[key].drop_duplicates().values))
    assert fl.groupby(key).layer.max().eq(fl.n_layers.iloc[0]).all()
    assert {"top_ids", "top_tokens", "top_probs", "forced_kind"} <= set(fl.columns)


# ══════════════════════════════════════════════════════════════
# Nomes novos e forma comportamental
# ══════════════════════════════════════════════════════════════

def test_new_given_names_present():
    names = lambda f, c, l: CFG.fact_specs[f].candidate(c).names_for(l)  # noqa: E731
    for l in CFG.languages:
        assert "Joseph" in names("fotografia", "niepce", l)
        assert {"Dimitri", "Dmitry"} <= set(names("tabela_periodica", "mendeleev", l))
        assert "Conrad" in names("raios_x", "rontgen", l)
        assert "Timothy" in names("www", "berners_lee", l)
        assert "Julius" in names("tabela_periodica", "meyer", l)
        assert "William" not in names("raios_x", "rontgen", l)          # ambíguo: forçado, não somado
    for f in ("telefone", "penicilina"):
        cid = "bell" if f == "telefone" else "fleming"
        assert "Alexandre" in names(f, cid, "pt") and "Alexandre" in names(f, cid, "fr")
        assert "Alessandro" in names(f, cid, "it") and "Alessandro" not in names(f, cid, "en")


def test_no_name_shared_by_two_candidates_of_a_fact():
    for k, f in CFG.fact_specs.items():
        for l in CFG.languages:
            seen = {}
            for c in f.candidates:
                for w in c.names_for(l):
                    assert seen.setdefault(w.casefold(), c.id) == c.id, \
                        f"{k}/{l}: '{w}' em {seen[w.casefold()]} e {c.id}"


def test_mendeleyew_extracted():
    from behavior.extraction import Extractor
    from config.loader import load_behavioral
    ex = Extractor(CFG.fact_specs, load_behavioral(), CFG.languages)
    c = ex.candidates("Das Periodensystem wurde von Dmitri Mendeleyew entwickelt.", "tabela_periodica", "de")
    assert c["primary"] == "mendeleev"


# ══════════════════════════════════════════════════════════════
# Fila: [timeout=]
# ══════════════════════════════════════════════════════════════

def _queue(tmp_path, lines, gpus="0", env=None, timeout=60):
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("\n".join(lines) + "\n")
    t0 = time.time()
    r = subprocess.run(["bash", str(ROOT / "scripts" / "run_queue.sh"), str(jobs), gpus],
                       capture_output=True, text=True, cwd=tmp_path, timeout=timeout,
                       env={**os.environ, "QUEUE_STATE": str(tmp_path / "state"), "QUEUE_POLL": "0.2",
                            "QUEUE_TIMEOUT_POLL": "0.2", "QUEUE_KILL_GRACE": "1", **(env or {})})
    return r, time.time() - t0


def _status(tmp_path):
    return {int(l.split("\t")[0]): int(l.split("\t")[1])
            for l in (tmp_path / "state" / "status").read_text().strip().splitlines()}


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_timeout_kills_job_and_children(tmp_path):
    lines = ["[timeout=2] bash -c 'sleep 30 & echo $! > child; sleep 30'",
             "[timeout=1m] echo ok > fast"]
    r, dt = _queue(tmp_path, lines)
    assert r.returncode == 1 and dt < 20, r.stdout + r.stderr
    st = _status(tmp_path)
    assert st == {1: 124, 2: 0}
    assert "TEMPO ESGOTADO job 1" in r.stdout
    assert not _alive(int((tmp_path / "child").read_text()))            # filho morto junto
    assert "tempo esgotado" in (tmp_path / "state" / "job1.log").read_text()


def test_timeout_sigkill_after_grace(tmp_path):
    r, dt = _queue(tmp_path, ["[timeout=1] bash -c 'trap \"\" TERM; sleep 30'"])
    assert _status(tmp_path) == {1: 124} and dt < 20
    assert "SIGKILL" in (tmp_path / "state" / "job1.log").read_text()


def test_default_timeout_and_validation(tmp_path):
    r, dt = _queue(tmp_path, ["sleep 30"], env={"QUEUE_TIMEOUT": "1"})
    assert _status(tmp_path) == {1: 124} and dt < 20
    for bad in ("[timeout=abc] echo x", "[timeout=0] echo x", "[timeout=5d] echo x"):
        r, _ = _queue(tmp_path, [bad])
        assert r.returncode == 2 and "timeout" in r.stderr


# ══════════════════════════════════════════════════════════════
# Comportamental: Lumière com caixa em fr, grafias de Mendeleev,
# tabela descritiva de não candidatos (2026-10-30)
# ══════════════════════════════════════════════════════════════

def _extractor():
    from behavior.extraction import Extractor
    from config.loader import load_behavioral
    return Extractor(CFG.fact_specs, load_behavioral(), CFG.languages)


def test_lumiere_capital_l_only_in_french():
    from config.loader import load_behavioral
    b, ex = load_behavioral(), _extractor()
    assert b.case_sensitive_for("cinema", "lumiere", "fr") == ["Lumière"]
    assert b.case_sensitive_for("cinema", "lumiere", "en") == []
    assert b.case_sensitive_for("telefone", "bell", "it") == ["Bell"]              # lista = todos os idiomas
    assert ex.candidates("c'est le Lumière frères qui", "cinema", "fr")["primary"] == "lumiere"
    assert ex.candidates("les frères **Lumière, Auguste et Louis**", "cinema", "fr")["primary"] == "lumiere"
    assert ex.candidates("la lumière du projecteur, puis Edison", "cinema", "fr")["primary"] == "edison_cinema"
    assert ex.candidates("the lumière brothers", "cinema", "en")["primary"] == "lumiere"


@pytest.mark.parametrize("spelling", ["Mendeleew", "Mendeleews", "Mendel'eev", "Mendel’eev", "Mendel'ev", "Mendel’ev", "Mendelev"])
def test_mendeleev_spellings(spelling):
    assert _extractor().candidates(f"Dmitrij Ivanovič {spelling}, un chimico", "tabela_periodica", "it")["primary"] == "mendeleev"


def test_noncandidate_table():
    import analyze
    from config.loader import load_behavioral
    b = load_behavioral()
    k = dict(paraphrase=1, kind="sample")
    ans = pd.DataFrame([
        {"model": "llama", "fact": "aviao", "lang": "fr", "sample_id": 1, "text": "Clément Ader et Otto Lilienthal.", **k},
        {"model": "llama", "fact": "aviao", "lang": "fr", "sample_id": 2, "text": "L'Éole de Clement Ader.", **k},
        {"model": "llama", "fact": "aviao", "lang": "fr", "sample_id": 3, "text": "Il n'y a pas un seul inventeur.", **k},
        {"model": "llama", "fact": "aviao", "lang": "fr", "sample_id": 4, "text": "Les frères Wright.", **k},
        {"model": "llama", "fact": "dinamite", "lang": "pt", "sample_id": 1, "text": "Inventada por Einar Anker.", **k}])
    attr = pd.DataFrame(_extractor().extract(ans.to_dict("records"))["attribution"])
    T = analyze.noncandidate_table(ans, attr, b)
    g = T[(T.fact == "aviao")].set_index("person")
    assert g.loc["ader", "n_answers"] == 2 and g.loc["lilienthal", "n_answers"] == 1
    assert g.loc[analyze.NOBODY_LISTED, "n_answers"] == 1
    assert (g["n_no_candidate"] == 3).all() and (g["n_answers_cell"] == 4).all()     # Wright não entra
    d = T[T.fact == "dinamite"].iloc[0]
    assert d.person == "einar_anker" and d.status == "fabricated"


def test_noncandidate_statuses_and_schoenbein():
    from config.loader import NON_CANDIDATE_STATUS, load_behavioral
    b = load_behavioral()
    people = {x["id"]: x for ps in b.non_candidates.values() for x in ps}
    assert NON_CANDIDATE_STATUS == {"related", "misattributed", "fabricated"}
    assert {x["status"] for x in people.values()} <= NON_CANDIDATE_STATUS
    # toda classificação que não é "related" diz o porquê
    assert all(x.get("note") for x in people.values() if x["status"] != "related")
    s = people["von_schoenbein"]
    assert s["status"] == "fabricated" and "Albert von Schönbein" in s["forms"]
    assert people["lewis"]["status"] == "misattributed" and people["lilienthal"]["status"] == "related"


def test_non_candidate_validation():
    import copy
    from config.loader import load_behavioral, validate_non_candidates
    b = copy.deepcopy(load_behavioral())
    assert validate_non_candidates(CFG.fact_specs, b) == []
    b.non_candidates["aviao"].append({"id": "x", "label": "x", "status": "maybe", "forms": ["Dumont"]})
    b.non_candidates["lampada"] = []
    errs = validate_non_candidates(CFG.fact_specs, b)
    assert any("status 'maybe'" in e for e in errs) and any("forma de candidato" in e for e in errs)
    assert any("lampada" in e for e in errs)
