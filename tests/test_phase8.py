"""
tests/test_phase8.py
Fase 8: opções dos pontos de entrada (--model, --gpus, --dtype,
--sanity-only), retomada protegida, fila de jobs, cópia anônima e o
dry run do pipeline inteiro.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

import run_lens
import run_pipeline_dry
import train_tuned_lens as ttl
from config.loader import load_models, load_tuned_lens
from models import runtime
from scripts import make_anonymous_copy as anon

ROOT = Path(__file__).resolve().parent.parent
REG = load_models()


# ══════════════════════════════════════════════════════════════
# Opções comuns
# ══════════════════════════════════════════════════════════════

def test_apply_gpus(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert runtime.apply_gpus("1 2") == "1,2"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert runtime.apply_gpus("3") == "3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    assert runtime.apply_gpus(None) == "5"          # sem --gpus: vale o ambiente (fila)
    with pytest.raises(ValueError):
        runtime.apply_gpus("gpu0")


def test_dtype_override():
    r = runtime.registry_with_dtype(REG, "gemma27", "bf16")
    assert r["gemma27"].precision == "bfloat16" and r["gemma27"].quantization is None
    assert REG["gemma27"].quantization == "nf4"                    # original intacto
    r = runtime.registry_with_dtype(REG, "llama", "nf4")
    assert r["llama"].quantization == "nf4"
    assert runtime.registry_with_dtype(REG, "llama", None) is REG
    assert runtime.precision_tag(REG["gemma27"]) == "nf4"
    assert runtime.precision_tag(REG["llama"]) == "bfloat16"


def test_resume_refuses_other_precision_or_commit(tmp_path):
    spec = REG["gemma27"]
    runtime.check_resume({}, spec, tmp_path)
    runtime.check_resume({"precision_tag": "nf4", "registry": {"revision": spec.revision}}, spec, tmp_path)
    with pytest.raises(RuntimeError, match="precisão"):
        runtime.check_resume({"precision_tag": "bfloat16"}, spec, tmp_path)
    with pytest.raises(RuntimeError, match="commit"):
        runtime.check_resume({"registry_revision": "a" * 40}, spec, tmp_path)


def test_entry_points_use_model_flag(monkeypatch):
    import models.model_loader
    monkeypatch.setattr(models.model_loader, "load_model",
                        lambda *a, **k: pytest.fail("não deveria carregar"))
    with pytest.raises(SystemExit):
        run_lens.main(["llama"])                               # posicional não existe mais
    with pytest.raises(SystemExit):
        run_lens.main(["--model", "gpt"])


def test_sanity_report_file_name(tmp_path):
    spec = REG["gemma27"]
    p = run_lens.write_sanity({"passed": True}, "gemma27", spec, tmp_path)
    assert p.name == "gemma27.json"
    bf = runtime.registry_with_dtype(REG, "gemma27", "bf16")["gemma27"]
    assert run_lens.write_sanity({"passed": True}, "gemma27", bf, tmp_path).name == "gemma27_bfloat16.json"


def test_train_skip_rules(tmp_path):
    tl = load_tuned_lens()
    assert tl["pretrained"] == {}
    assert ttl.skip_reason("llama", tl, tmp_path / "l", tmp_path / "r") is None
    (tmp_path / "l" / "llama").mkdir(parents=True)
    (tmp_path / "l" / "llama" / "translators.pt").write_text("x")
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "llama_eval.csv").write_text("x")
    assert "já treinada" in ttl.skip_reason("llama", tl, tmp_path / "l", tmp_path / "r")
    assert ttl.skip_reason("llama", tl, tmp_path / "l", tmp_path / "r", force=True) is None
    assert "pré-treinada" in ttl.skip_reason("llama", {**tl, "pretrained": {"llama": "x"}})


# ══════════════════════════════════════════════════════════════
# Fila de jobs
# ══════════════════════════════════════════════════════════════

def test_run_queue(tmp_path):
    jobs = tmp_path / "jobs.txt"
    lines = ["# comentário", ""]
    for i in range(6):
        lines.append(f'echo "$CUDA_VISIBLE_DEVICES" > {tmp_path}/out{i}; sleep 0.2'
                     + ("; exit 7" if i == 3 else ""))
    jobs.write_text("\n".join(lines) + "\n")
    r = subprocess.run(["bash", str(ROOT / "scripts" / "run_queue.sh"), str(jobs), "0 1,2"],
                       capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "QUEUE_STATE": str(tmp_path / "state")}, timeout=60)
    assert r.returncode == 1                                        # um job falhou
    got = sorted((tmp_path / f"out{i}").read_text().strip() for i in range(6))
    assert set(got) <= {"0", "1,2"} and len(got) == 6               # cada job rodou uma vez
    status = (tmp_path / "state" / "status").read_text().strip().splitlines()
    assert len(status) == 6 and sum(l.split("\t")[1] != "0" for l in status) == 1
    assert "job 4" in r.stdout                                       # linha 4 da fila (i = 3)


def test_run_queue_usage():
    r = subprocess.run(["bash", str(ROOT / "scripts" / "run_queue.sh")], capture_output=True, text=True)
    assert r.returncode == 2 and "uso" in r.stderr


# ══════════════════════════════════════════════════════════════
# Cópia anônima
# ══════════════════════════════════════════════════════════════

def test_scrub_text():
    terms = ["Ana Souza", "ana@uni.edu", "asouza", "Ana", "Souza"]
    t, n = anon.scrub_text(
        "Autora: Ana Souza (ana@uni.edu), github.com/asouza/repo, token hf_ABCDEFGHIJKLMNOPQRSTUVWX, "
        "/Users/asouza/x, Souza-Dumont não é nome, Anatomia fica.", terms, "asouza/repo")
    assert "Ana" not in t.replace("Anonymous", "").replace("Anatomia", "")
    assert "ana@uni.edu" not in t and "hf_REDACTED" in t and "~/x" in t
    assert "asouza" not in t and "github.com/anonymous/repository" in t
    assert "Anatomia" in t and n >= 6
    t2, _ = anon.scrub_text("https://github.com/asouza/repo.git", terms, "asouza/repo")
    assert t2 == "anonymous repository"


def test_make_copy_on_temp_repo(tmp_path):
    src = tmp_path / "repo"
    (src / "sub").mkdir(parents=True)
    (src / "README.md").write_text("Por Ana Souza <ana@uni.edu>\n")
    (src / "sub" / "code.py").write_text("# autor: asouza\nx = 1\n")
    (src / ".env").write_text("HF_TOKEN=hf_ABCDEFGHIJKLMNOPQRSTUVWX\n")
    (src / "img.bin").write_bytes(b"\0\x01binary Ana Souza")
    git = lambda *a: subprocess.run(["git", *a], cwd=src, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=x", "-c", "user.email=x@y", "commit", "-qm", "init")
    out = tmp_path / "anon"
    stats = anon.make_copy(out, ["Ana Souza", "ana@uni.edu", "asouza", "Ana", "Souza"], src=src,
                           log=lambda *a: None)
    assert not (out / ".git").exists() and not (out / ".env").exists()
    assert stats["skipped_secret"] == [".env"]
    assert "Anonymous" in (out / "README.md").read_text()
    assert "asouza" not in (out / "sub" / "code.py").read_text()
    assert stats["problems"] == ["img.bin"]                         # binário com nome é acusado
    with pytest.raises(FileExistsError):
        anon.make_copy(out, ["Ana"], src=src)


def test_anonymous_copy_of_this_repo(tmp_path):
    terms = anon.identity_terms()
    if not terms:
        pytest.skip("sem git config")
    stats = anon.make_copy(tmp_path / "anon", terms, log=lambda *a: None)
    assert stats["problems"] == [] and ".env" in stats["skipped_secret"]
    assert not (tmp_path / "anon" / ".env").exists()


# ══════════════════════════════════════════════════════════════
# Dry run
# ══════════════════════════════════════════════════════════════

def test_dry_run_quick(tmp_path):
    out = tmp_path / "dry"
    assert run_pipeline_dry.main(["--quick", "--out", str(out)]) == 0
    rep = json.loads((out / "dry_run_report.json").read_text())
    assert rep["ok"] and rep["unconfirmed_values_used"]
    assert set(rep["steps"]) == {"1_sanity", "2_logit_lens", "3_behavioral", "4_tuned_lens",
                                 "5_analysis_statistics", "6_figures_tables", "7_check_results",
                                 "8_validation"}
    assert rep["checks"]["check_results_problems"] == []
    assert rep["checks"]["tables"] == ["results_compact.tex", "results_full.tex"]
    w = rep["workload"]["tiny_instruct"]
    assert w["logit_lens"]["forwards_prompts"] > 0 and w["behavioral"]["answers"] > 0
    assert (out / "figures" / "fig2_delta_cs.pdf").exists()


def test_training_lock(tmp_path):
    with ttl.TrainingLock(tmp_path / "llama") as lock:
        assert lock.path.exists() and "pid" in lock.path.read_text()
        with pytest.raises(RuntimeError, match="outro processo"):
            ttl.TrainingLock(tmp_path / "llama").__enter__()
    assert not (tmp_path / "llama" / "TRAINING.lock").exists()
    with ttl.TrainingLock(tmp_path / "llama"):                       # liberado: pode de novo
        pass
