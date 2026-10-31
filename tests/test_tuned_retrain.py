"""
tests/test_tuned_retrain.py
Correção do Tuned Lens (2026-10-30): corte do gradiente, hash da receita,
checkpoint de outra receita recusado, teste de aceitação, lente antiga
arquivada (nunca apagada) e retreinada, lente reprovada fora da remedição
e listada no PACK_INFO.json, e a fila marcando o job reprovado como falho.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

import check_results
import train_tuned_lens as ttl
from logit_lens.lens_training import RecipeMismatch, acceptance, recipe_sha256, train
from logit_lens.tuned_lens import LensRejected, TunedLens
from test_tuned_lens import FACTS, LensTok, _model, _sources, _tiny_config, tiny_tl

ROOT = Path(__file__).resolve().parents[1]
QUIET = dict(log=lambda *a: None)


# ══════════════════════════════════════════════════════════════
# Corte do gradiente e receita
# ══════════════════════════════════════════════════════════════

def test_gradient_is_clipped_before_each_step(tmp_path):
    cfg = tiny_tl(num_steps=1, max_grad_norm=1e-3)
    lens = TunedLens(_model())
    res = train(lens, lens.model, LensTok(), _sources(), cfg, FACTS, tmp_path, **QUIET)
    assert res["grad_norms_preclip"][0] > 1e-3                 # o corte agiu
    assert res["max_grad_norm"] == 1e-3 and res["recipe_sha256"] == recipe_sha256(cfg)
    # Primeiro passo de Nesterov a partir de zero: Δθ = lr · (1 + momento) · g_cortado
    t = cfg["training"]
    delta = torch.sqrt(sum(p.detach().pow(2).sum() for p in lens.translators.parameters())).item()
    assert delta <= t["lr"] * (1 + t["momentum"]) * 1e-3 * 1.01


def test_recipe_hash_ignores_memory_settings_only():
    a = tiny_tl()
    assert recipe_sha256(a) == recipe_sha256(tiny_tl(loss_chunk_tokens=4, checkpoint_every=3))
    assert recipe_sha256(a) != recipe_sha256(tiny_tl(max_grad_norm=0.5))
    assert recipe_sha256(a) != recipe_sha256(tiny_tl(tokens_per_step=128))


def test_checkpoint_of_another_recipe_is_refused(tmp_path):
    lens = TunedLens(_model())
    train(lens, lens.model, LensTok(), _sources(), tiny_tl(), FACTS, tmp_path, max_steps=2, **QUIET)
    other = TunedLens(_model())
    with pytest.raises(RecipeMismatch, match="outra receita"):
        train(other, other.model, LensTok(), _sources(), tiny_tl(max_grad_norm=0.5), FACTS, tmp_path, **QUIET)


# ══════════════════════════════════════════════════════════════
# Teste de aceitação
# ══════════════════════════════════════════════════════════════

def _rows(tuned, logit):
    return [{"lang": l, "layer": k + 1, "kl_tuned": t, "kl_logit": g}
            for l in ("pt", "en") for k, (t, g) in enumerate(zip(tuned, logit))]


def test_acceptance_rule():
    L = 11                                         # camadas traduzidas 1..10; zona 30% = 8..10
    logit = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ok = acceptance(_rows([5, 5, 5, 4, 4, 3, 3, 2.9, 2.0, 1.02], logit), L, 0.30, 0.05)
    assert ok["passed"] and ok["zone_layers"] == [8, 9, 10] and ok["last_layer"] == 10
    bad = acceptance(_rows([5, 5, 5, 4, 4, 3, 3, 2.5, 2.5, 5.75], logit), L, 0.30, 0.05)
    assert not bad["passed"] and len(bad["reasons"]) == 2
    assert [f["layer"] for f in bad["failing_layers"]] == [9, 10]
    assert bad["last_kl_tuned"] == 5.75 and bad["last_kl_logit"] == 1
    # NaN reprova (nunca passa em silêncio); zona sem camada traduzida: só a última conta
    assert not acceptance(_rows([5] * 9 + [float("nan")], logit), L, 0.30, 0.05)["passed"]
    tiny = acceptance(_rows([1, 0.5], [2, 0.6]), 3, 0.30, 0.05)
    assert tiny["passed"] and tiny["zone_kl_tuned"] is None and tiny["zone_layers"] == []


def _run(tmp_path, tol, **kw):
    cfg, model, tok = _tiny_config(), _model(), LensTok()
    s = ttl.run("llama", model, tok, cfg, tiny_tl(acceptance_tol=tol), _sources(),
                lens_root=tmp_path / "lenses", report_dir=tmp_path / "rep", **QUIET, **kw)
    return s, model


def test_accepted_lens_writes_eval_status_and_recipe(tmp_path):
    s, model = _run(tmp_path, 1e9)
    rep, lens = tmp_path / "rep", tmp_path / "lenses" / "llama"
    assert s["acceptance"]["passed"] and (rep / "llama_eval.csv").exists()
    assert not (rep / "llama_eval_rejected.csv").exists()
    st = json.loads((rep / "llama_status.json").read_text())
    meta = json.loads((lens / "config.json").read_text())["meta"]
    sha = recipe_sha256(tiny_tl(acceptance_tol=1e9))
    assert st["status"] == "accepted" and st["recipe_sha256"] == sha == meta["recipe_sha256"]
    assert meta["max_grad_norm"] == 1.0 and meta["recipe_version"] == 2 and meta["acceptance"]["passed"]
    assert json.loads((lens / "RECIPE").read_text())["recipe_sha256"] == sha
    TunedLens.load(model, lens)                                 # aprovada: carrega


def test_rejected_lens_is_not_remeasured(tmp_path, capsys):
    msgs = []
    cfg, model, tok = _tiny_config(), _model(), LensTok()
    s = ttl.run("llama", model, tok, cfg, tiny_tl(acceptance_tol=-1e9), _sources(),
                lens_root=tmp_path / "lenses", report_dir=tmp_path / "rep", log=msgs.append)
    rep, lens = tmp_path / "rep", tmp_path / "lenses" / "llama"
    assert not s["acceptance"]["passed"]
    assert (rep / "llama_eval_rejected.csv").exists() and not (rep / "llama_eval.csv").exists()
    assert json.loads((rep / "llama_status.json").read_text())["status"] == "rejected"
    text = "\n".join(msgs)
    assert "REPROVADO" in text and "llama" in text and "camada" in text and "tuned" in text and "logit" in text
    with pytest.raises(LensRejected):                          # run_lens --lens tuned não usa
        TunedLens.load(model, lens)
    TunedLens.load(model, lens, allow_rejected=True)           # --eval-only ainda pode


def test_main_exit_code_when_rejected(monkeypatch):
    import models.model_loader
    monkeypatch.setattr(ttl, "skip_reason", lambda *a, **k: None)
    monkeypatch.setattr(ttl.TrainingLock, "__enter__", lambda self: self)
    monkeypatch.setattr(ttl.TrainingLock, "__exit__", lambda self, *a: False)
    monkeypatch.setattr(models.model_loader, "load_model", lambda *a, **k: (None, None))
    monkeypatch.setattr(ttl, "hf_sources", lambda *a, **k: {})
    monkeypatch.setattr(ttl, "run", lambda *a, **k: {"done": True, "acceptance": {"passed": False}})
    assert ttl.main(["--model", "llama"]) == ttl.EXIT_REJECTED == 4
    monkeypatch.setattr(ttl, "run", lambda *a, **k: {"done": True, "acceptance": {"passed": True}})
    assert ttl.main(["--model", "llama"]) == 0


# ══════════════════════════════════════════════════════════════
# Lente antiga: arquivada (nunca apagada) e retreinada; mesma receita: pula
# ══════════════════════════════════════════════════════════════

def _old_lens(tmp_path, key="llama"):
    """Lente como as de 2026-10-29: sem RECIPE, sem hash no config.json."""
    lens, rep = tmp_path / "lenses" / key, tmp_path / "reports" / "tuned_lens"
    lens.mkdir(parents=True)
    rep.mkdir(parents=True)
    (lens / "translators.pt").write_bytes(b"old")
    (lens / "checkpoint.pt").write_bytes(b"old-ckpt")
    (lens / "config.json").write_text(json.dumps({"meta": {"steps": 250}}))
    (rep / f"{key}_eval.csv").write_text("old")
    (rep / f"{key}_train.json").write_text("{}")
    res = tmp_path / "results" / "lens" / "tuned_lens" / key
    res.mkdir(parents=True)
    (res / "manifest.json").write_text("{}")
    return lens, rep, res


def test_old_recipe_lens_is_archived_then_retrained(tmp_path):
    lens, rep, res = _old_lens(tmp_path)
    msgs = []
    tl = tiny_tl()
    why = ttl.skip_reason("llama", tl, lens_root=tmp_path / "lenses", report_dir=rep,
                          results_dir=tmp_path / "results", log=msgs.append)
    assert why is None                                         # treina
    assert not lens.exists() and not (rep / "llama_eval.csv").exists() and not res.exists()
    stale = next((tmp_path / "lenses_stale").iterdir())
    assert (stale / "llama" / "translators.pt").read_bytes() == b"old"       # nada apagado
    assert (stale / "llama" / "checkpoint.pt").exists()
    assert (stale / "llama_results_tuned_lens" / "manifest.json").exists()
    rstale = next((tmp_path / "reports" / "tuned_lens_stale").iterdir())
    assert {p.name for p in rstale.iterdir()} == {"llama_eval.csv", "llama_train.json"}
    assert "sem hash de receita" in msgs[0] and "Nada foi apagado" in msgs[0]
    assert "llama" in (stale / "README.txt").read_text()


def test_same_recipe_skips_and_rejected_stays_rejected(tmp_path):
    tl = tiny_tl()
    sha = recipe_sha256(tl)
    lens, rep = tmp_path / "lenses" / "llama", tmp_path / "rep"
    lens.mkdir(parents=True)
    rep.mkdir()
    (lens / "translators.pt").write_bytes(b"new")
    (lens / "RECIPE").write_text(json.dumps({"recipe_sha256": sha}))
    (rep / "llama_eval.csv").write_text("x")
    (rep / "llama_status.json").write_text(json.dumps({"status": "accepted"}))
    kw = dict(lens_root=tmp_path / "lenses", report_dir=rep, **QUIET)
    msg, rc = ttl.skip_reason("llama", tl, **kw)
    assert rc == 0 and "aprovada" in msg and lens.exists()
    (rep / "llama_eval.csv").unlink()
    (rep / "llama_status.json").write_text(json.dumps({"status": "rejected",
                                                       "acceptance": {"reasons": ["x"]}}))
    msg, rc = ttl.skip_reason("llama", tl, **kw)
    assert rc == 4 and "REPROVADA" in msg and "--tokens-per-step 262144" in msg
    # outra receita (2^18): a reprovada é arquivada e o treino roda
    assert ttl.skip_reason("llama", tiny_tl(tokens_per_step=128), **kw) is None
    assert not lens.exists() and (tmp_path / "lenses_stale").exists()


def test_lock_prevents_archiving(tmp_path):
    lens, rep, _ = _old_lens(tmp_path)
    (lens / "TRAINING.lock").write_text("pid 1")
    assert ttl.skip_reason("llama", tiny_tl(), lens_root=tmp_path / "lenses", report_dir=rep, **QUIET) is None
    assert (lens / "translators.pt").exists()                   # outro processo treinando: nada movido


# ══════════════════════════════════════════════════════════════
# check_results, PACK_INFO.json e a fila
# ══════════════════════════════════════════════════════════════

def _statuses(rep):
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "llama_status.json").write_text(json.dumps({"status": "accepted", "recipe_sha256": "a" * 64,
                                                       "max_grad_norm": 1.0, "tokens_per_step": 65536}))
    (rep / "mistral_status.json").write_text(json.dumps({
        "status": "rejected", "acceptance": {"passed": False, "reasons": ["última camada"],
                                             "failing_layers": [{"layer": 31}]}}))


def test_check_results_lists_lenses_not_accepted(tmp_path):
    _statuses(tmp_path / "rep")
    p = check_results.check_tuned_status(tmp_path / "rep", ["llama", "mistral", "qwen"])
    assert len(p) == 2 and any("mistral" in x and "reprovada" in x for x in p)
    assert any("qwen" in x and "sem qwen_status.json" in x for x in p)


def test_pack_info_lists_failed_lenses(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import pack_results
    _statuses(tmp_path / "reports" / "tuned_lens")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "x.txt").write_text("x")
    _, info = pack_results.pack(tmp_path, tmp_path / "packs", check=(False, "report"))
    assert info["tuned_lenses_failed"] == ["gemma", "mistral", "qwen"]
    assert info["tuned_lenses"]["llama"]["status"] == "accepted"
    assert info["tuned_lenses"]["llama"]["max_grad_norm"] == 1.0
    assert info["tuned_lenses"]["mistral"]["acceptance"]["failing_layers"] == [{"layer": 31}]


def test_queue_marks_rejected_training_failed_and_skips_remeasure(tmp_path):
    from test_token_gaps import _queue, _status
    (tmp_path / "reports" / "tuned_lens").mkdir(parents=True)
    py = sys.executable
    # treino reprovado: sai com 4, só escreve _eval_rejected.csv; treino aprovado: escreve _eval.csv
    rej = (f"{py} -c \"open('reports/tuned_lens/a_eval_rejected.csv','w'); raise SystemExit(4)\"; "
           f"test -f reports/tuned_lens/a_eval.csv && touch remeasured_a")
    acc = (f"{py} -c \"open('reports/tuned_lens/b_eval.csv','w')\"; "
           f"test -f reports/tuned_lens/b_eval.csv && touch remeasured_b")
    r, _ = _queue(tmp_path, [rej, acc])
    assert r.returncode == 1, r.stdout + r.stderr                # a fila acusa a falha
    assert _status(tmp_path) == {1: 1, 2: 0}
    assert not (tmp_path / "remeasured_a").exists() and (tmp_path / "remeasured_b").exists()


def test_run_all_server_handles_the_retrain():
    import re
    s = re.sub(r"[ \t]+", " ", (ROOT / "run_all_server.sh").read_text())    # colunas alinhadas
    assert "\nunset HF_HUB_OFFLINE" in s
    assert "[gpu=0] [timeout=8h] python train_tuned_lens.py --model gemma;" in s
    for m in ("llama", "qwen", "mistral"):
        assert f"[timeout=5h] python train_tuned_lens.py --model {m};" in s
        assert f"test -f reports/tuned_lens/{m}_eval.csv && python run_lens.py --model {m}" in s
    assert "[after=reports/tuned_lens/mistral_status.json]" in s and "mistral_eval.csv] [mem" not in s
    assert "pack_results.py --include-lenses" in s
    assert os.access(ROOT / "run_all_server.sh", os.X_OK)
    r = subprocess.run(["bash", "-n", str(ROOT / "run_all_server.sh")], capture_output=True)
    assert r.returncode == 0


def test_rejected_eval_csv_is_never_read_as_accepted(tmp_path):
    # tuned_eval (fig6c) lê só *_eval.csv: uma lente reprovada não entra na figura
    import make_figures as mf
    (tmp_path / "llama_eval.csv").write_text("lang,layer,kl_tuned,kl_logit\npt,1,1,2\n")
    (tmp_path / "qwen_eval_rejected.csv").write_text("lang,layer,kl_tuned,kl_logit\npt,1,9,2\n")
    ev = mf.tuned_eval(tmp_path)
    assert set(ev["model"]) == {"llama"}
    assert isinstance(ev, pd.DataFrame)
