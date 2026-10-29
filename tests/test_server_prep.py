"""
tests/test_server_prep.py
Preparação para o servidor do coautor: token (placeholder no .env
ignorado), verificação de acesso sem download, scripts/check_env.py,
scripts/pack_results.py, condições da fila ([gpu=], [after=], [mem=])
e Gemma 2 27B em bf16 no registro.
"""

import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from config.loader import load_models
from models import model_loader, runtime
from scripts import check_env, pack_results

ROOT = Path(__file__).resolve().parent.parent
REG = load_models()
REAL = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz01234567"
OTHER = "hf_" + "ZyXwVuTsRqPoNmLkJiHgFeDcBa98765432"


# ══════════════════════════════════════════════════════════════
# Token
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tok,ok", [
    (REAL, True), ("hf_tok…", False), ("hf_seu_token", False), ("hf_your_token_here", False),
    ("hf_" + "x" * 34, False), ("hf_YOURTOKENHEREyourtokenhereYOURTOKEN", False),
    ("", False), (None, False), (REAL[:20], False), ("sk-" + REAL[3:], False),
])
def test_looks_like_token(tok, ok):
    assert model_loader.looks_like_token(tok) is ok


@pytest.fixture
def token_env(tmp_path, monkeypatch):
    """.env e arquivo do `hf auth login` isolados em tmp_path."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    login = tmp_path / "hf_home" / "token"
    monkeypatch.setattr(model_loader, "_token_files", lambda: [login])
    model_loader._warned.clear()
    return SimpleNamespace(env=tmp_path / ".env", login=login)


def test_placeholder_in_dotenv_falls_back_to_login(token_env, capsys):
    token_env.env.write_text("HF_TOKEN=hf_tok…\n")
    before = token_env.env.read_bytes()
    token_env.login.parent.mkdir()
    token_env.login.write_text(OTHER + "\n")
    tok, src = model_loader.find_hf_token(token_env.env)
    assert tok == OTHER and "hf auth login" in src
    assert "placeholder" in capsys.readouterr().err
    assert token_env.env.read_bytes() == before                     # .env intocado
    assert "HF_TOKEN" not in os.environ                              # placeholder não vai ao ambiente


def test_real_dotenv_token_used(token_env):
    token_env.env.write_text(f"HF_TOKEN={REAL}\n")
    assert model_loader.find_hf_token(token_env.env) == (REAL, "HF_TOKEN (.env)")
    assert os.environ["HF_TOKEN"] == REAL                            # como load_dotenv


def test_env_placeholder_removed_env_real_wins(token_env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_your_token_here")
    token_env.env.write_text(f"HF_TOKEN={REAL}\n")
    assert model_loader.find_hf_token(token_env.env)[0] == REAL
    monkeypatch.setenv("HF_TOKEN", OTHER)
    assert model_loader.find_hf_token(token_env.env) == (OTHER, "HF_TOKEN (ambiente)")


def test_no_token_anywhere(token_env):
    token_env.env.write_text("HF_TOKEN=hf_seu_token\n")
    assert model_loader.find_hf_token(token_env.env) == (None, None)


def _raise(exc):
    def f(url, token=None):
        raise exc
    return f


@pytest.mark.parametrize("exc,expect", [
    (None, "access ok"),
    ("GatedRepoError", "accept the license at https://huggingface.co/org/m"),
    ("RevisionNotFoundError", "pinned commit"),
    ("RepositoryNotFoundError", "repository not found"),
    ("RemoteEntryNotFoundError", "not found at commit"),
    (ConnectionError("offline"), "ConnectionError"),
])
def test_check_repo_access(monkeypatch, exc, expect):
    import huggingface_hub
    from huggingface_hub import errors
    seen = {}
    if exc is None:
        fake = lambda url, token=None: seen.setdefault("url", url)          # noqa: E731
    elif isinstance(exc, str):
        resp = httpx.Response(404, request=httpx.Request("HEAD", "https://hf.co"))
        fake = _raise(getattr(errors, exc)("x", response=resp))
    else:
        fake = _raise(exc)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", fake)
    ok, detail = model_loader.check_repo_access("org/m", "a" * 40, "tok")
    assert ok is (exc is None) and expect in detail
    if exc is None:
        assert "a" * 40 in seen["url"] and seen["url"].endswith("config.json")


# ══════════════════════════════════════════════════════════════
# check_env.py
# ══════════════════════════════════════════════════════════════

def test_check_python():
    assert check_env.check_python((3, 10, 14))[0].status == "FAIL"
    assert check_env.check_python((3, 11, 0))[0].status == "OK"
    assert check_env.check_python((3, 12, 7))[0].status == "OK"


def test_check_packages(tmp_path):
    req = tmp_path / "req.txt"
    req.write_text("torch==2.11.0\nnumpy==2.4.4   # x\nfoo==1.0; platform_system == 'Nowhere'\n"
                   "pandas==3.0.2\n# comment\n")
    from importlib.metadata import PackageNotFoundError
    have = {"torch": "2.11.0+cu128", "numpy": "2.4.4", "pandas": "2.2.0"}

    def ver(n):
        if n not in have:
            raise PackageNotFoundError(n)
        return have[n]
    out = check_env.check_packages(req, ver)
    assert [c.status for c in out] == ["WARN"] and "pandas 2.2.0" in out[0].detail   # +cu128 ok, foo fora
    del have["numpy"]
    assert [c.status for c in check_env.check_packages(req, ver)] == ["FAIL", "WARN"]
    assert check_env.read_pins(ROOT / "requirements.txt")["torch"] == "2.11.0"


def _torch(gpus, cuda="13.0", available=True):
    """torch falso: gpus = [(nome, GB, (major, minor))]."""
    props = [SimpleNamespace(name=n, total_memory=gb * 1e9) for n, gb, _ in gpus]
    return SimpleNamespace(
        __version__="2.11.0", version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(is_available=lambda: available, device_count=lambda: len(gpus),
                             get_device_properties=lambda i: props[i],
                             get_device_capability=lambda i: gpus[i][2]))


def _by_name(checks):
    return {c.name: c for c in checks}


def test_check_gpus_server():
    out = _by_name(check_env.check_gpus(_torch([("NVIDIA A100", 96, (8, 0))] * 4), driver="580.65"))
    assert {out[f"gpu {i}"].status for i in range(4)} == {"OK"}
    assert "bf16 ok" in out["gpu 0"].detail and "96.0 GB" in out["gpu 0"].detail
    assert out["gemma27 bf16"].status == "OK" and out["tuned lens"].status == "OK"
    assert out["two jobs per GPU"].status == "OK" and "580.65" in out["torch"].detail


def test_check_gpus_small_and_old():
    out = _by_name(check_env.check_gpus(_torch([("Tesla T4", 16, (7, 5))]), driver=""))
    assert out["gpu 0"].status == "FAIL" and "bf16" in out["gpu 0"].detail
    assert out["gemma27 bf16"].status == "FAIL" and out["tuned lens"].status == "FAIL"
    two = _by_name(check_env.check_gpus(_torch([("L40S", 48, (8, 9))] * 2), driver=""))
    assert two["gemma27 bf16"].status == "WARN" and "two GPUs" in two["gemma27 bf16"].detail
    assert two["two jobs per GPU"].status == "WARN"


def test_check_gpus_no_cuda_hints():
    out = check_env.check_gpus(_torch([], cuda="13.0", available=False), driver="550.54")
    assert out[-1].status == "FAIL" and "cu128" in out[-1].detail and "580" in out[-1].detail
    out = check_env.check_gpus(_torch([], cuda=None, available=False), driver="")
    assert "CPU-only" in out[-1].detail


def test_check_disk_same_filesystem_adds_up(tmp_path):
    hub = tmp_path / "hub"
    blobs = hub / "models--org--m" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "b1").write_bytes(b"0" * 1000)
    free = {"gb": 260}
    usage = lambda p: SimpleNamespace(free=free["gb"] * 1e9)                 # noqa: E731
    work = {"results": tmp_path / "results", "lenses": tmp_path / "lenses"}  # ainda não existem
    out = check_env.check_disk(hub, work, ["org/m"], usage=usage)
    assert len(out) == 1 and out[0].status == "FAIL" and "needs 280 GB" in out[0].detail
    free["gb"] = 300
    assert check_env.check_disk(hub, work, ["org/m"], usage=usage)[0].status == "OK"


def test_check_hub(monkeypatch):
    from config.experiment_config import ExperimentConfig
    from config.loader import load_tuned_lens
    cfg, tl = ExperimentConfig(), load_tuned_lens()
    calls = []

    def access(rid, rev, token, repo_type="model"):
        calls.append((rid, repo_type))
        return (rid != "google/gemma-2-27b-it"), "detail"
    out = check_env.check_hub(cfg, tl, find_token=lambda: (REAL, "x"), access=access)
    by = _by_name(out)
    assert by["hf token"].status == "OK" and REAL[:6] not in by["hf token"].detail   # token não aparece
    assert by["model gemma27"].status == "FAIL" and by["model llama"].status == "OK"
    assert len([c for c in calls if c[1] == "model"]) == len(cfg.registry.models)
    assert {r for r, t in calls if t == "dataset"} == {"HuggingFaceFW/fineweb-2", "HuggingFaceFW/fineweb"}
    out = check_env.check_hub(cfg, tl, find_token=lambda: (None, None), access=lambda *a, **k: (True, ""))
    assert _by_name(out)["hf token"].status == "FAIL"                # há modelos gated


def test_check_env_main_exit_code(monkeypatch, tmp_path):
    C = check_env.Check
    monkeypatch.setattr(check_env, "run_checks", lambda **k: [C("OK", "a", ""), C("WARN", "b", "")])
    assert check_env.main(["--report", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["counts"] == {"OK": 1, "WARN": 1, "FAIL": 0}
    monkeypatch.setattr(check_env, "run_checks", lambda **k: [C("FAIL", "a", "")])
    assert check_env.main(["--report", str(tmp_path / "r.json")]) == 1


# ══════════════════════════════════════════════════════════════
# pack_results.py
# ══════════════════════════════════════════════════════════════

def _tree(root: Path):
    files = {
        "results/lens/logit_lens/llama/manifest.json": "{}",
        "results/lens/logit_lens/llama/probs.parquet": "p",
        "results/behavioral/llama/manifest.json": "{}",
        "results/.gitkeep": "",
        "reports/sanity/llama.json": "{}",
        "reports/tuned_lens/llama_eval.csv": "a,b",
        "reports/check_env.json": "{}",
        "reports/phase1_review.md": "não vai",
        "logs/queue/main-1/job1.log": "ok",
        "logs/__pycache__/x.pyc": "no",
        "lenses/llama/translators.pt": "T",
        "lenses/llama/config.json": "{}",
        "lenses/llama/checkpoint.pt": "C",
        "lenses/llama/TRAINING.lock": "L",
        "config/models.yaml": "não vai",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def test_pack_contents(tmp_path):
    _tree(tmp_path)
    path, info = pack_results.pack(tmp_path, tmp_path / "packs", check=(False, "1 PROBLEMA"))
    with tarfile.open(path) as tar:
        names = set(tar.getnames())
        meta = json.load(tar.extractfile("PACK_INFO.json"))
    assert names == {"PACK_INFO.json", "results/lens/logit_lens/llama/manifest.json",
                     "results/lens/logit_lens/llama/probs.parquet", "results/behavioral/llama/manifest.json",
                     "reports/sanity/llama.json", "reports/tuned_lens/llama_eval.csv",
                     "reports/check_env.json", "logs/queue/main-1/job1.log"}
    assert meta["check_results_ok"] is False and meta["check_results_report"] == "1 PROBLEMA"
    assert meta["n_files"] == 7 and not meta["include_lenses"]
    sha = Path(str(path) + ".sha256").read_text().split()[0]
    assert sha == info["sha256"] and len(sha) == 64

    path2, _ = pack_results.pack(tmp_path, tmp_path / "packs2", include_lenses=True, check=(True, ""))
    with tarfile.open(path2) as tar:
        lens = {n for n in tar.getnames() if n.startswith("lenses/")}
    assert lens == {"lenses/llama/translators.pt", "lenses/llama/config.json"}   # sem checkpoint/lock


def test_pack_main_require_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(pack_results, "run_check_results", lambda: (False, "faltam unidades"))
    out = tmp_path / "out"
    assert pack_results.main(["--require-complete", "--out-dir", str(out)]) == 1
    assert not out.exists()
    monkeypatch.setattr(pack_results, "ROOT", tmp_path)
    _tree(tmp_path)
    assert pack_results.main(["--out-dir", str(out)]) == 0                    # incompleto, mas empacota
    assert len(list(out.glob("results-*.tar.gz"))) == 1


def test_human_sizes():
    assert pack_results.human(512) == "512 B"
    assert pack_results.human(3 * 1024 ** 2) == "3.0 MB"
    assert pack_results.human(5 * 1024 ** 3) == "5.0 GB"


# ══════════════════════════════════════════════════════════════
# Fila: [gpu=], [after=], [mem=]
# ══════════════════════════════════════════════════════════════

def _queue(tmp_path, lines, gpus, env=None, timeout=60):
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("\n".join(lines) + "\n")
    return subprocess.run(["bash", str(ROOT / "scripts" / "run_queue.sh"), str(jobs), gpus],
                          capture_output=True, text=True, cwd=tmp_path, timeout=timeout,
                          env={**os.environ, "QUEUE_STATE": str(tmp_path / "state"),
                               "QUEUE_POLL": "0.2", **(env or {})})


def test_queue_gpu_pin(tmp_path):
    lines = [f'[gpu=1,2] echo "$CUDA_VISIBLE_DEVICES" > out{i}' for i in range(3)]
    lines += [f'[gpu=0] echo "$CUDA_VISIBLE_DEVICES" > zero{i}' for i in range(2)]
    r = _queue(tmp_path, lines, "0 1,2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert {(tmp_path / f"out{i}").read_text().strip() for i in range(3)} == {"1,2"}
    assert {(tmp_path / f"zero{i}").read_text().strip() for i in range(2)} == {"0"}


def test_queue_after_waits_without_blocking(tmp_path):
    lines = ['[after=flag] test -e flag && echo "$CUDA_VISIBLE_DEVICES" > waited',
             "sleep 0.6; touch flag",
             "echo free > other"]
    r = _queue(tmp_path, lines, "0 1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "waited").exists() and (tmp_path / "other").exists()
    assert "aguarda: esperando flag" in r.stdout


def test_queue_mem_check(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    smi = bin_ / "nvidia-smi"
    # memória livre em MiB por GPU: 0 → 10 GB, 1 → 80 GB
    smi.write_text('#!/usr/bin/env bash\nwhile [ $# -gt 0 ]; do [ "$1" = "-i" ] && g=$2; shift; done\n'
                   'case "$g" in 0) echo 10240 ;; 1) echo 81920 ;; esac\n')
    smi.chmod(0o755)
    r = _queue(tmp_path, ['[mem=60] echo "$CUDA_VISIBLE_DEVICES" > big', 'echo x > small'], "0 1",
               env={"PATH": f"{bin_}:{os.environ['PATH']}"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "big").read_text().strip() == "1"
    assert "esperando 60 GB livres na GPU 0" in r.stdout


def test_queue_rejects_bad_conditions(tmp_path):
    r = _queue(tmp_path, ["[gpu=5] echo x"], "0 1")
    assert r.returncode == 2 and "[gpu=5]" in r.stderr
    r = _queue(tmp_path, ["[cpu=1] echo x"], "0")
    assert r.returncode == 2 and "desconhecida" in r.stderr
    r = _queue(tmp_path, ["[mem=lots] echo x"], "0")
    assert r.returncode == 2
    r = _queue(tmp_path, ["[ -d . ] && echo plain > plain"], "0")        # teste do bash não é condição
    assert r.returncode == 0 and (tmp_path / "plain").exists()


# ══════════════════════════════════════════════════════════════
# Gemma 2 27B em bf16 (2026-10-29)
# ══════════════════════════════════════════════════════════════

def test_gemma27_bf16_by_default_nf4_still_available():
    g = REG["gemma27"]
    assert g.precision == "bfloat16" and g.quantization is None
    assert runtime.precision_tag(g) == "bfloat16"
    assert "nf4" in REG.quantization                                   # --dtype nf4 ainda funciona
    q = runtime.registry_with_dtype(REG, "gemma27", "nf4")["gemma27"]
    assert q.quantization == "nf4" and runtime.precision_tag(q) == "nf4"


def test_tuned_lens_full_precision_settings():
    from config.loader import load_tuned_lens
    t = load_tuned_lens()["training"]
    assert t["translator_dtype"] == "float32" and t["seq_len"] == 2048
