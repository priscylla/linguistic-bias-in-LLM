"""
scripts/check_env.py
Checks the machine before any run (first step of the run guides).

    python scripts/check_env.py                 # everything
    python scripts/check_env.py --skip-hub      # no network checks

Checks, each one printed as OK / WARN / FAIL:
    - Python >= 3.11 (the pinned numpy, pandas, scipy, pyarrow need it)
    - installed packages against the pins in requirements.txt
    - torch, its CUDA build, CUDA available
    - each GPU: name, memory, bf16 support (compute capability >= 8.0),
      and whether the large jobs fit (Gemma 2 27B in bf16 ~55 GB on one
      GPU; Gemma 2 9B Tuned Lens training ~27 GB)
    - free disk at the Hugging Face cache (~230 GB for all models, minus
      what is already cached) and at the results / lenses directories
    - Hugging Face token (a placeholder HF_TOKEN in .env is ignored) and,
      without downloading, access to every model at its pinned commit
      (gated: license accepted) and to the Tuned Lens corpus datasets

Exits with 1 if any check FAILs. Also writes reports/check_env.json
(packed by scripts/pack_results.py with the results).
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

# Same GPU numbering as nvidia-smi (and scripts/run_queue.sh)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_PYTHON = (3, 11)
MODELS_GB = 230            # all 11 models in the cache (gemma_base is stored in fp32)
WORK_GB = {"results": 20,  # lens and behavioral tables (~5 GB), logs, the results pack
           "lenses": 30}   # 4 lenses (~8 GB) + training checkpoints with momentum (~17 GB)
GEMMA27_GB = 64            # Gemma 2 27B bf16: ~55 GB weights + activations, one GPU
TUNED_LENS_GB = 32         # Gemma 2 9B Tuned Lens training peak ~27 GB
SHARED_GPU_GB = 80         # two jobs per GPU (RUNNING_server.md)
REPORT = ROOT / "reports" / "check_env.json"


@dataclass
class Check:
    status: str            # OK | WARN | FAIL
    name: str
    detail: str


def _gb(n_bytes) -> float:
    return n_bytes / 1e9


# ── Python and packages ──

def check_python(version=None):
    v = tuple(version or sys.version_info[:3])
    s = ".".join(map(str, v))
    if v[:2] < MIN_PYTHON:
        return [Check("FAIL", "python", f"{s}: needs >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} "
                                        f"(pinned numpy/pandas/scipy/pyarrow); make a fresh venv")]
    return [Check("OK", "python", f"{s} ({sys.executable})")]


def read_pins(path):
    """{package: version} from requirements.txt, only lines whose marker applies here."""
    from packaging.markers import Marker
    pins = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        spec, _, marker = line.partition(";")
        if marker.strip() and not Marker(marker.strip()).evaluate():
            continue
        name, ver = (x.strip() for x in spec.split("=="))
        pins[name] = ver
    return pins


def check_packages(requirements=ROOT / "requirements.txt", version_of=metadata.version):
    out, missing, differ = [], [], []
    for name, want in read_pins(requirements).items():
        try:
            have = version_of(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if have.split("+")[0] != want:          # torch 2.11.0+cu128 counts as 2.11.0
            differ.append(f"{name} {have} (pinned {want})")
    if missing:
        out.append(Check("FAIL", "packages", "not installed: " + ", ".join(missing)
                         + "  → pip install -r requirements.txt in a fresh venv"))
    if differ:
        out.append(Check("WARN", "packages", "versions differ from requirements.txt: " + "; ".join(differ)))
    if not missing and not differ:
        out.append(Check("OK", "packages", "all pins of requirements.txt installed"))
    return out


# ── torch, CUDA, GPUs ──

def _driver_version():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def check_gpus(torch=None, driver=None):
    if torch is None:
        try:
            import torch
        except ImportError as e:
            return [Check("FAIL", "torch", f"cannot import torch: {e}")]
    driver = driver if driver is not None else _driver_version()
    cuda_build = torch.version.cuda
    out = [Check("OK", "torch", f"{torch.__version__}, CUDA build {cuda_build or 'none (CPU-only)'}"
                 + (f", NVIDIA driver {driver}" if driver else ""))]
    if not torch.cuda.is_available():
        hint = "CPU-only torch: reinstall it with CUDA" if not cuda_build else (
            f"torch is built for CUDA {cuda_build}; the driver may be too old "
            f"(CUDA 13.0 needs driver >= 580). Either update the driver or install a CUDA 12 build: "
            f"pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128")
        return out + [Check("FAIL", "cuda", "no GPU visible to torch. " + hint)]

    n = torch.cuda.device_count()
    mems = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        major, minor = torch.cuda.get_device_capability(i)
        mem = _gb(p.total_memory)
        mems.append(mem)
        bf16 = (major, minor) >= (8, 0)
        out.append(Check("OK" if bf16 else "FAIL", f"gpu {i}",
                         f"{p.name}, {mem:.1f} GB, compute {major}.{minor}, "
                         + ("bf16 ok" if bf16 else "no native bf16 (needs compute >= 8.0, e.g. A100)")))
    if not mems:
        return out + [Check("FAIL", "gpus", "torch sees CUDA but 0 devices")]
    top = max(mems)
    if top >= GEMMA27_GB:
        out.append(Check("OK", "gemma27 bf16", f"fits on one GPU (needs ~{GEMMA27_GB} GB)"))
    elif sum(sorted(mems)[-2:]) >= GEMMA27_GB and n >= 2:
        out.append(Check("WARN", "gemma27 bf16", f"no single GPU has {GEMMA27_GB} GB: "
                                                f"split it over two GPUs (--gpus \"a b\" or a \"a,b\" queue worker)"))
    else:
        out.append(Check("FAIL", "gemma27 bf16", f"needs ~{GEMMA27_GB} GB of GPU memory (one or two GPUs)"))
    if top >= TUNED_LENS_GB:
        small = [i for i, m in enumerate(mems) if m < TUNED_LENS_GB]
        out.append(Check("WARN" if small else "OK", "tuned lens",
                         "Gemma 2 9B training (~27 GB peak, float32 translators, seq_len 2048) fits"
                         + (f"; not on GPU(s) {small}" if small else " on every GPU")))
    else:
        out.append(Check("FAIL", "tuned lens", f"no GPU with {TUNED_LENS_GB} GB: Gemma 2 9B training "
                                               "(~27 GB peak) does not fit with the registered settings"))
    shared = all(m >= SHARED_GPU_GB for m in mems)
    out.append(Check("OK" if shared else "WARN", "two jobs per GPU",
                     f"{'every' if shared else 'not every'} GPU has >= {SHARED_GPU_GB} GB "
                     f"(RUNNING_server.md runs a Tuned Lens training and a second job together)"))
    return out


# ── Disk ──

def _existing(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    while not path.exists():
        path = path.parent
    return path


def _cached_bytes(hub_cache: Path, repo_ids):
    total = 0
    for rid in repo_ids:
        d = Path(hub_cache) / ("models--" + rid.replace("/", "--")) / "blobs"
        if d.is_dir():
            total += sum(f.stat().st_size for f in d.iterdir() if f.is_file())
    return total


def check_disk(hub_cache, work_dirs, repo_ids, models_gb=MODELS_GB, work_gb=None,
               usage=shutil.disk_usage):
    """
    Free space where the models go (minus what is already cached) and
    where results and lenses go (work_dirs: {label: path}, needs in
    work_gb). Directories on the same filesystem add up their needs.
    """
    work_gb = work_gb or WORK_GB
    cached = _gb(_cached_bytes(hub_cache, repo_ids))
    needs = [("hf cache", Path(hub_cache), max(0.0, models_gb - cached),
              f"{models_gb} GB for all models, {cached:.0f} GB already cached")]
    needs += [(label, Path(d), float(work_gb.get(label, 0)), "") for label, d in work_dirs.items()]
    by_dev = {}
    for label, d, gb, note in needs:
        e = _existing(d)
        by_dev.setdefault(e.stat().st_dev, []).append((label, d, e, gb, note))
    out = []
    for items in by_dev.values():
        need = sum(it[3] for it in items)
        free = _gb(usage(items[0][2]).free)
        names = ", ".join(f"{it[0]} ({it[1]})" for it in items)
        notes = "; ".join(it[4] for it in items if it[4])
        out.append(Check("OK" if free >= need else "FAIL", "disk",
                         f"{names}: {free:.0f} GB free, needs {need:.0f} GB" + (f" ({notes})" if notes else "")))
    return out


# ── Hugging Face ──

def check_hub(config, tuned_cfg, find_token=None, access=None):
    from models.model_loader import check_repo_access, find_hf_token
    find_token = find_token or find_hf_token
    access = access or check_repo_access
    out = []
    token, source = find_token()
    gated = [k for k, m in config.registry.models.items() if m.gated]
    if token is None:
        out.append(Check("FAIL" if gated else "OK", "hf token",
                         "none found: run `hf auth login` (= huggingface-cli login) or export HF_TOKEN"
                         if gated else "none (no gated models)"))
    else:
        out.append(Check("OK", "hf token", f"found: {source}"))
    for key, spec in config.registry.models.items():
        ok, detail = access(spec.id, spec.revision, token)
        out.append(Check("OK" if ok else "FAIL", f"model {key}",
                         f"{spec.id}{' (gated)' if spec.gated else ''}: {detail}"))
    seen = set()
    for lang, c in tuned_cfg["corpus"].items():
        k = (c["dataset"], c["revision"])
        if k in seen:
            continue
        seen.add(k)
        ok, detail = access(c["dataset"], c["revision"], token, repo_type="dataset")
        out.append(Check("OK" if ok else "FAIL", "tuned lens corpus", f"{c['dataset']}: {detail}"))
    return out


# ── Main ──

def run_checks(skip_hub=False, results_dir=None):
    from config.experiment_config import ExperimentConfig
    from config.loader import load_tuned_lens
    config = ExperimentConfig()
    tuned = load_tuned_lens()
    checks = check_python() + check_packages() + check_gpus()
    try:
        from huggingface_hub import constants
        hub_cache = constants.HF_HUB_CACHE
    except ImportError:
        hub_cache = os.path.expanduser("~/.cache/huggingface/hub")
    work = {"results": ROOT / (results_dir or config.results_dir),
            "lenses": ROOT / tuned.get("lens_dir", "lenses")}
    checks += check_disk(hub_cache, work, [m.id for m in config.registry.models.values()])
    if skip_hub:
        checks.append(Check("WARN", "hugging face", "skipped (--skip-hub)"))
    else:
        checks += check_hub(config, tuned)
    return checks


def print_checks(checks):
    for c in checks:
        print(f"  {c.status:<4}  {c.name:<18} {c.detail}")
    n = {s: sum(c.status == s for c in checks) for s in ("OK", "WARN", "FAIL")}
    print(f"\n  {n['OK']} OK, {n['WARN']} WARN, {n['FAIL']} FAIL")
    if n["FAIL"]:
        print("  Fix every FAIL before running (see RUNNING_server.md, step 0).")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check the machine before running the experiments.")
    ap.add_argument("--skip-hub", action="store_true", help="skip the Hugging Face checks (network)")
    ap.add_argument("--results-dir", default=None, help="results directory (default: config)")
    ap.add_argument("--report", default=str(REPORT), help="where to write the JSON report")
    args = ap.parse_args(argv)

    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    checks = run_checks(skip_hub=args.skip_hub, results_dir=args.results_dir)
    n = print_checks(checks)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"host_os": platform.platform(), "python": platform.python_version(),
                                  "counts": n, "checks": [asdict(c) for c in checks]},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Report: {report.relative_to(ROOT) if report.is_relative_to(ROOT) else report}")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
