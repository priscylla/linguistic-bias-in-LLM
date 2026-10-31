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
    r = runtime.registry_with_dtype(REG, "gemma27", "nf4")
    assert r["gemma27"].precision == "bfloat16" and r["gemma27"].quantization == "nf4"
    assert REG["gemma27"].quantization is None                     # original intacto
    r = runtime.registry_with_dtype(REG, "gemma27", "fp32")
    assert r["gemma27"].precision == "float32" and r["gemma27"].quantization is None
    r = runtime.registry_with_dtype(REG, "llama", "nf4")
    assert r["llama"].quantization == "nf4"
    assert runtime.registry_with_dtype(REG, "llama", None) is REG
    assert runtime.precision_tag(REG["gemma27"]) == "bfloat16"
    assert runtime.precision_tag(REG["llama"]) == "bfloat16"


def test_resume_refuses_other_precision_or_commit(tmp_path):
    spec = REG["gemma27"]
    runtime.check_resume({}, spec, tmp_path)
    runtime.check_resume({"precision_tag": "bfloat16", "registry": {"revision": spec.revision}}, spec, tmp_path)
    with pytest.raises(RuntimeError, match="precisão"):
        runtime.check_resume({"precision_tag": "nf4"}, spec, tmp_path)   # resultados antigos em 4 bits
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
    q = runtime.registry_with_dtype(REG, "gemma27", "nf4")["gemma27"]
    assert run_lens.write_sanity({"passed": True}, "gemma27", q, tmp_path).name == "gemma27_nf4.json"


def test_train_skip_rules(tmp_path):
    import json
    from logit_lens.lens_training import recipe_sha256
    tl = load_tuned_lens()
    assert tl["pretrained"] == {}
    quiet = dict(log=lambda *a: None)
    assert ttl.skip_reason("llama", tl, tmp_path / "l", tmp_path / "r", **quiet) is None   # sem lente
    lens = tmp_path / "l" / "llama"
    lens.mkdir(parents=True)
    (lens / "translators.pt").write_text("x")
    (lens / "RECIPE").write_text(json.dumps({"recipe_sha256": recipe_sha256(tl)}))
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "llama_eval.csv").write_text("x")
    (tmp_path / "r" / "llama_status.json").write_text(json.dumps({"status": "accepted"}))
    msg, rc = ttl.skip_reason("llama", tl, tmp_path / "l", tmp_path / "r", **quiet)
    assert rc == 0 and "já treinada e aprovada" in msg                  # mesma receita, aprovada
    assert ttl.skip_reason("llama", tl, tmp_path / "l", tmp_path / "r", force=True, **quiet) is None
    assert not lens.exists() and (tmp_path / "l_stale").exists()        # --force arquiva, não apaga
    msg, rc = ttl.skip_reason("llama", {**tl, "pretrained": {"llama": "x"}})
    assert rc == 0 and "pré-treinada" in msg


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
    # Termos do próprio repositório (autores dos commits, dono do remote),
    # não do git config de quem roda: mesmo resultado em qualquer máquina
    terms = anon.identity_terms(from_config=False)
    if not terms:
        pytest.skip("não é um checkout git com histórico")
    stats = anon.make_copy(tmp_path / "anon", terms, log=lambda *a: None)
    assert stats["problems"] == [] and ".env" in stats["skipped_secret"]
    assert not (tmp_path / "anon" / ".env").exists()


def test_leftovers_whole_word_only():
    assert anon.leftovers("Leonardo da Vinci".encode(), ["Leo"]) == []      # dentro de outra palavra
    assert anon.leftovers("by Leo, 2026".encode(), ["Leo"]) == ["Leo"]
    assert anon.leftovers("SILVA et al.".encode(), ["Silva"]) == ["Silva"]  # sem caixa


def test_identity_terms_independent_of_git_config(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "a.txt").write_text("x\n")
    git = lambda *a: subprocess.run(["git", *a], cwd=src, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Ana Souza", "-c", "user.email=ana@uni.edu", "commit", "-qm", "init")
    git("remote", "add", "origin", "https://github.com/asouza/proj.git")
    terms = anon.identity_terms(src=src, from_config=False)
    assert {"Ana Souza", "ana@uni.edu", "asouza", "Ana", "Souza"} <= set(terms)


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


def test_tuned_lens_training_not_blocked_by_native_review(monkeypatch):
    # Fatos confirmados bastam: a revisão nativa pendente não bloqueia o treino
    import models.model_loader
    calls = []
    monkeypatch.setattr(models.model_loader, "load_model",
                        lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(RuntimeError("parou aqui")))
    monkeypatch.setattr(ttl, "skip_reason", lambda *a, **k: None)
    monkeypatch.setattr(ttl.TrainingLock, "__enter__", lambda self: self)
    monkeypatch.setattr(ttl.TrainingLock, "__exit__", lambda self, *a: False)
    with pytest.raises(RuntimeError, match="parou aqui"):
        ttl.main(["--model", "llama"])
    assert calls                                     # passou pelos guardas e foi carregar o modelo


def test_indicator_cell_skipped_when_fewer_than_two_valid_candidates(tmp_path):
    import pandas as pd
    from config.loader import Candidate, Fact, Prompt
    from devtools.tiny import WordTokenizer, build_tiny_model
    from logit_lens.confound import ConfoundTester
    from logit_lens.measure import LensRunner
    from config.loader import load_forcing

    def cand(cid, dom, place):
        return Candidate(id=cid, label=cid, dominant=dom, stakes=[], names={"pt": {"nom": [cid.title()]}},
                         indicators={"place": {"pt": place}})
    # Os dois candidatos com o mesmo indicador de um token só ("Paris"):
    # nem o prefixo forçado resolve -> nenhum candidato medido
    fact = Fact(key="radio", kind="disputed", status="confirmed", verb="invented",
                keyword={"pt": "rádio"}, candidates=[cand("a", True, ["Paris"]), cand("b", False, ["Paris"])])
    prompts = [Prompt("radio", "pt", "place", i, t) for i, t in
               enumerate(["O rádio foi inventado em", "O rádio foi testado pela primeira vez em",
                          "O lugar onde o rádio foi inventado é"], start=1)]
    prompts += [Prompt("radio", "pt", "F1", 2, "Quem inventou o rádio foi")]
    cfg = type("C", (), {"fact_specs": {"radio": fact}, "prompt_specs": tuple(prompts)})()
    runner = LensRunner(build_tiny_model("llama", vocab=512), WordTokenizer())
    t = run_lens.run_fact_unit(runner, ConfoundTester(runner, load_forcing()), cfg, "radio", "pt",
                               "tiny", ("completion",))
    assert {r["prompt_type"] for r in t["probs"]} == {"F1"}           # place pulado inteiro
    sk = pd.DataFrame(t["skipped"])
    assert set(sk["paraphrase"]) == {1, 2, 3} and set(sk["prompt_type"]) == {"place"}
    assert (sk["valid_candidates"] == "").all()
    assert t["prefix_forcing"] == []          # célula pulada não é medida (motivo em skipped)

    # São Tomé / São Paulo: " São" compartilhado, resolvido por prefixo -> célula mantida
    fact2 = Fact(key="radio", kind="disputed", status="confirmed", verb="invented",
                 keyword={"pt": "rádio"},
                 candidates=[cand("a", True, ["São Tomé"]), cand("b", False, ["São Paulo"])])
    cfg2 = type("C", (), {"fact_specs": {"radio": fact2}, "prompt_specs": tuple(prompts)})()
    t2 = run_lens.run_fact_unit(runner, ConfoundTester(runner, load_forcing()), cfg2, "radio", "pt",
                                "tiny", ("completion",))
    assert t2["skipped"] == [] and {r["prompt_type"] for r in t2["probs"]} == {"place", "F1"}
    pf = pd.DataFrame(t2["prefix_forcing"])
    assert set(pf["status"]) == {"combined"} and set(pf["shared_token"]) == {" São"}
