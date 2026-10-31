"""
tests/test_fingerprint.py
Impressão digital por unidade: o que entra (nomes, prompts, forçamento,
lens) e o que não entra (dominante); run_lens --facts refaz só as
unidades com entradas novas, arquiva as velhas e nunca as mistura;
unidades antigas sem impressão são refeitas; check_results acusa
unidades velhas; scripts/stamp_fingerprints.py carimba só o verificado.
"""

import copy
import dataclasses

import pandas as pd
import pytest

import check_results
import run_lens
from conftest import WordTokenizer, build_tiny_model
from logit_lens.fingerprint import fingerprint, lens_identity
from logit_lens.store import ChunkStore, lens_dir
from test_run_pipeline import CHAT, tiny_config


def _with_name(cfg, fact, cid, word):
    new = copy.deepcopy(cfg)
    c = new.fact_specs[fact].candidate(cid)
    for lang in c.names:
        c.names[lang]["nom"] = c.names[lang]["nom"] + [word]
    return new


def test_what_changes_the_fingerprint():
    cfg = tiny_config()
    units = run_lens.lens_units(cfg)
    fp = {u: fingerprint(u, cfg, "logit_lens") for u in units}
    # dominante não entra
    dom = copy.deepcopy(cfg)
    for c in dom.fact_specs["aviao"].candidates:
        c.dominant = not c.dominant
    assert all(fingerprint(u, dom, "logit_lens") == fp[u] for u in units)
    # nome novo: unidades do fato + linha de base
    new = _with_name(cfg, "aviao", "santos_dumont", "Alberto2")
    changed = {u for u in units if fingerprint(u, new, "logit_lens") != fp[u]}
    assert changed == {u for u in units if u.startswith(("aviao__", "baseline__"))}
    # palavra forçada nova em pt: unidades de fato em pt, não a linha de base
    f = copy.deepcopy(cfg)
    f.forcing_spec = dataclasses.replace(
        f.forcing_spec, articles={**f.forcing_spec.articles, "pt": f.forcing_spec.articles["pt"] + ["Sir"]})
    changed = {u for u in units if fingerprint(u, f, "logit_lens") != fp[u]}
    assert changed == {u for u in units if u.endswith("__pt") and not u.startswith("baseline")}
    # outro lens ou outra versão do código: tudo
    assert all(fingerprint(u, cfg, "tuned_lens:abc") != fp[u] for u in units)
    assert all(fingerprint(u, cfg, "logit_lens", measure_version=99) != fp[u] for u in units)


def test_lens_identity():
    class T:
        name, source_sha256 = "tuned_lens", "f" * 64
    assert lens_identity(None) == "logit_lens" and lens_identity(T()) == "tuned_lens:" + "f" * 64


def test_parse_facts():
    cfg = tiny_config()
    assert run_lens.parse_facts(None, cfg) is None
    assert run_lens.parse_facts("aviao, penicilina", cfg) == {"aviao", "penicilina"}
    with pytest.raises(SystemExit):
        run_lens.parse_facts("aviao,lampada", cfg)


@pytest.fixture
def fresh(tmp_path):
    cfg = tiny_config()
    out = lens_dir(str(tmp_path), "llama")
    model, tok = build_tiny_model("llama", vocab=4096), WordTokenizer()
    run_lens.run_model("llama", model, tok, cfg, out, model_id="tiny/llama",
                       chat_formatter=CHAT, log=lambda *a: None)
    return {"cfg": cfg, "out": out, "model": model, "tok": tok, "root": tmp_path}


def _run(f, cfg, facts=None):
    logs = []
    m = run_lens.run_model("llama", f["model"], f["tok"], cfg, f["out"], model_id="tiny/llama",
                           chat_formatter=CHAT, facts=facts, log=logs.append)
    return m, logs


def test_rerun_only_changed_units_with_facts(fresh):
    f = fresh
    store = ChunkStore(f["out"])
    before = {u: (f["out"] / "chunks" / u / "DONE").stat().st_mtime_ns for u in store.done_units()}
    assert all(store.fingerprint(u) for u in before)                  # gravadas com impressão
    m, logs = _run(f, f["cfg"])                                        # nada mudou: nada roda
    assert m["sessions"][-1]["units_run"] == 0
    new = _with_name(f["cfg"], "aviao", "santos_dumont", "Alberto2")
    m, logs = _run(f, new, facts={"aviao"})
    rerun = set(m["sessions"][-1]["units_rerun_stale"])
    assert rerun == {u for u in before if u.startswith(("aviao__", "baseline__"))}
    for u in before:                                                   # o resto não foi tocado
        if u not in rerun:
            assert (f["out"] / "chunks" / u / "DONE").stat().st_mtime_ns == before[u]
    stale_dir = next((f["out"] / "chunks_stale").iterdir())
    assert {p.name for p in stale_dir.iterdir()} == rerun              # antigas arquivadas, não apagadas
    assert m["units_stale"] == [] and m["units_done"] == len(before)
    ts = pd.read_parquet(f["out"] / "token_sets.parquet")
    assert "Alberto2" in set(ts[(ts.fact == "aviao")].word)            # tabelas com as novas


def test_stale_units_of_other_facts_left_out_of_tables(fresh):
    f = fresh
    new = _with_name(f["cfg"], "penicilina", "fleming", "Alexandre")
    m, logs = _run(f, new, facts={"aviao"})                            # penicilina não foi pedida
    assert any("velhas" in l and "penicilina" in l for l in logs)
    assert set(m["units_stale"]) == {u for u in run_lens.lens_units(new) if u.startswith("penicilina__")}
    probs = pd.read_parquet(f["out"] / "probs.parquet")
    assert "penicilina" not in set(probs.fact)                         # nunca misturadas
    probs_cr = check_results.check_model("llama", str(f["root"]), new)
    assert any("faltando" in p or "entradas antigas" in p or "sem probs" in p for p in probs_cr)


def test_legacy_units_without_fingerprint_are_redone(fresh):
    f = fresh
    store = ChunkStore(f["out"])
    for u in store.done_units():
        (f["out"] / "chunks" / u / "FINGERPRINT").unlink()
    assert any("sem impressão digital" in p for p in check_results.check_model("llama", str(f["root"]), f["cfg"]))
    m, _ = _run(f, f["cfg"])
    assert m["sessions"][-1]["units_run"] == len(run_lens.lens_units(f["cfg"]))


def test_stamp_only_done_and_unstamped(fresh):
    f = fresh
    store = ChunkStore(f["out"])
    u = store.done_units()[0]
    with pytest.raises(ValueError):
        store.stamp(u, "x", "test")                                    # já tem impressão
    (f["out"] / "chunks" / u / "FINGERPRINT").unlink()
    fp = fingerprint(u, f["cfg"], "logit_lens")
    store.stamp(u, fp, "test")
    assert store.current(u, fp) and (f["out"] / "chunks" / u / "FINGERPRINT_SOURCE").read_text() == "test"
    with pytest.raises(ValueError):
        store.stamp("nao_existe__pt", fp, "test")


def test_stamp_script_verifies_manifest(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, "scripts")
    import stamp_fingerprints as sf
    man = {"config_sha256": {"facts.yaml": "a"}, "sessions": [{"started": "2026-10-29T00:00:00"}]}
    assert sf.verify(tmp_path, man, {"facts.yaml": "a"}, "logit_lens", "llama") is None
    assert "difere" in sf.verify(tmp_path, man, {"facts.yaml": "b"}, "logit_lens", "llama")
    assert "mudou" in sf.verify(tmp_path, {**man, "config_changes": [{}]}, {"facts.yaml": "a"}, "logit_lens", "llama")
    assert sf.verify(tmp_path, {}, {}, "logit_lens", "llama") == "sem manifest"


def test_nothing_to_do_only_when_every_unit_is_current(fresh):
    f = fresh
    assert run_lens.nothing_to_do(f["out"], f["cfg"], "llama", "logit")
    new = _with_name(f["cfg"], "aviao", "santos_dumont", "Alberto2")
    assert not run_lens.nothing_to_do(f["out"], new, "llama", "logit")        # entrada nova
    (f["out"] / "probs.parquet").unlink()
    assert not run_lens.nothing_to_do(f["out"], f["cfg"], "llama", "logit")   # sem tabelas
    assert not run_lens.nothing_to_do(f["root"] / "vazio", f["cfg"], "llama", "tuned",
                                      lens_root=f["root"] / "sem_lentes")      # sem lente


def _no_load(monkeypatch):
    import models.model_loader
    monkeypatch.setattr(models.model_loader, "load_model",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("carregou o modelo")))


def test_run_lens_main_skips_model_load_when_nothing_to_do(monkeypatch, tmp_path, capsys):
    _no_load(monkeypatch)
    monkeypatch.setattr(run_lens, "nothing_to_do", lambda *a, **k: True)
    assert run_lens.main(["--model", "llama", "--lens", "logit", "--results-dir", str(tmp_path)]) == 0
    assert "nada a refazer" in capsys.readouterr().out


def test_run_behavioral_main_skips_model_load_when_all_done(monkeypatch, tmp_path, capsys):
    import run_behavioral
    from logit_lens.store import ChunkStore
    _no_load(monkeypatch)
    monkeypatch.setattr(ChunkStore, "done", lambda self, u: True)
    out = run_behavioral.behavioral_dir(str(tmp_path), "llama")
    out.mkdir(parents=True)
    (out / "answers.parquet").write_bytes(b"x")
    assert run_behavioral.main(["--model", "llama", "--results-dir", str(tmp_path)]) == 0
    assert "nada a refazer" in capsys.readouterr().out


def test_sanity_skipped_when_already_passed(monkeypatch, tmp_path, capsys):
    import json
    from config.loader import load_models
    from models.runtime import precision_tag
    spec = load_models()["llama"]
    rep = {"passed": True, "revision": spec.revision, "precision_tag": precision_tag(spec)}
    monkeypatch.setattr(run_lens, "SANITY_DIR", tmp_path)
    p = run_lens.sanity_path("llama", spec, tmp_path)
    p.write_text(json.dumps(rep))
    assert run_lens.sanity_already_passed("llama", spec, tmp_path)
    p.write_text(json.dumps({**rep, "passed": False}))
    assert not run_lens.sanity_already_passed("llama", spec, tmp_path)
    p.write_text(json.dumps({**rep, "revision": "outro"}))
    assert not run_lens.sanity_already_passed("llama", spec, tmp_path)
    p.write_text(json.dumps(rep))
    _no_load(monkeypatch)
    monkeypatch.setattr(run_lens, "sanity_already_passed", lambda m, s: True)
    assert run_lens.main(["--model", "llama", "--sanity-only"]) == 0
    assert "sanidade já aprovada" in capsys.readouterr().out
