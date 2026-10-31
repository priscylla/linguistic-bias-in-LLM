"""
scripts/pack_results.py
Packs everything the runs produced into one tarball to send back (last
step of the run guides).

    python scripts/pack_results.py                     # results, reports, logs
    python scripts/pack_results.py --include-lenses    # + trained Tuned Lens translators (~8 GB)
    python scripts/pack_results.py --require-complete  # refuse if check_results.py finds problems

First runs check_results.py (every expected prompt, unit and answer
present). If it finds problems the pack is still made (partial runs can
be shared) and PACK_INFO.json says so; --require-complete refuses
instead.

The tarball (packs/results-<date>-<commit>.tar.gz, paths relative to
the repository root) holds:
    results/                     lens and behavioral tables, chunks, manifest.json per model
    reports/sanity/              sanity check per model
    reports/tuned_lens/          held-out evaluation and training summary per model
    reports/validation/          behavioral validation sample (to label)
    reports/tokenization/, reports/tokenization_summary.md, verification_report.json
    reports/check_env.json       the machine (GPUs, versions)
    logs/                        queue logs (logs/queue/), background run logs
    lenses/                      only with --include-lenses: translators.pt and
                                 config.json (training checkpoints and locks left out)
    PACK_INFO.json               git commit, local changes, check_results report, file list,
                                 Tuned Lens per model (status, recipe hash, max_grad_norm,
                                 tokens per step, acceptance test) and tuned_lenses_failed

Unpack at the repository root: tar -xzf results-....tar.gz
"""

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INCLUDE = ["results", "reports/sanity", "reports/tuned_lens", "reports/validation",
           "reports/tokenization", "reports/tokenization_summary.md", "verification_report.json",
           "reports/check_env.json", "logs"]
LENS_FILES = ("translators.pt", "config.json", "RECIPE")
SKIP_NAMES = {"__pycache__", ".DS_Store", ".gitkeep", "chunks_stale"}   # unidades velhas ficam no servidor


def _git(*args) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def collect(root: Path, include_lenses: bool, lens_dir="lenses"):
    """Files to pack, as paths relative to `root`, sorted."""
    files = []
    for entry in INCLUDE:
        p = root / entry
        if p.is_file():
            files.append(Path(entry))
        elif p.is_dir():
            for f in p.rglob("*"):
                rel = f.relative_to(root)
                if f.is_file() and not (SKIP_NAMES & set(rel.parts)):
                    files.append(rel)
    if include_lenses and (root / lens_dir).is_dir():
        for f in (root / lens_dir).glob("*/*"):
            if f.is_file() and f.name in LENS_FILES:
                files.append(f.relative_to(root))
    return sorted(set(files))


def tuned_lens_info(root: Path, lens_dir="lenses", report_dir="reports/tuned_lens") -> dict:
    """
    Per Tuned Lens model (config/tuned_lens.yaml models): status from
    reports/tuned_lens/{m}_status.json (accepted | rejected | missing) and the
    recipe from lenses/{m}/config.json, for the methods section.
    """
    try:
        from config.loader import load_tuned_lens
        models = load_tuned_lens()["models"]
    except Exception:                                    # config ilegível: o que houver
        models = sorted({p.name.split("_status.json")[0]
                         for p in (root / report_dir).glob("*_status.json")})
    out = {}
    for m in models:
        st, meta = {}, {}
        try:
            st = json.loads((root / report_dir / f"{m}_status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        try:
            meta = json.loads((root / lens_dir / m / "config.json").read_text(encoding="utf-8")).get("meta", {})
        except (OSError, ValueError):
            pass
        acc = st.get("acceptance") or meta.get("acceptance") or {}
        out[m] = {"status": st.get("status", "missing"),
                  "recipe_sha256": st.get("recipe_sha256") or meta.get("recipe_sha256"),
                  "recipe_version": st.get("recipe_version") or meta.get("recipe_version"),
                  "max_grad_norm": st.get("max_grad_norm") or meta.get("max_grad_norm"),
                  "tokens_per_step": st.get("tokens_per_step") or meta.get("tokens_per_step"),
                  "steps": meta.get("steps"),
                  "acceptance": {k: acc.get(k) for k in ("passed", "reasons", "last_layer", "last_kl_tuned",
                                                         "last_kl_logit", "zone_kl_tuned", "zone_kl_logit",
                                                         "failing_layers", "tolerance_nats")} if acc else None}
    return out


def run_check_results():
    """check_results.check_all_results(), with its report captured. Returns (ok, text)."""
    buf = io.StringIO()
    try:
        from check_results import check_all_results
        with redirect_stdout(buf):
            ok = bool(check_all_results())
    except Exception as e:                               # sem resultados, config inválido
        print(f"check_results failed: {type(e).__name__}: {e}", file=buf)
        ok = False
    return ok, buf.getvalue()


def pack(root: Path, out_dir: Path, include_lenses=False, check=None, now=None):
    """
    Writes the tarball and returns (path, info). `check` = (ok, text) from
    check_results (run here if None).
    """
    ok, report = check if check is not None else run_check_results()
    files = collect(root, include_lenses)
    if not files:
        raise SystemExit("nothing to pack: no results/, reports/ or logs/ found")
    commit = _git("rev-parse", "HEAD")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results-{stamp}-{commit[:7] or 'nogit'}.tar.gz"
    info = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "git_commit": commit,
        "git_local_changes": [l for l in _git("status", "--porcelain").splitlines() if l],
        "check_results_ok": ok,
        "check_results_report": report,
        "include_lenses": include_lenses,
        "tuned_lenses": (tl := tuned_lens_info(root)),
        "tuned_lenses_failed": sorted(m for m, v in tl.items() if v["status"] != "accepted"),
        "python": platform.python_version(),
        "n_files": len(files),
        "bytes_uncompressed": sum((root / f).stat().st_size for f in files),
        "files": [str(f) for f in files],
    }
    with tarfile.open(path, "w:gz") as tar:
        data = json.dumps(info, ensure_ascii=False, indent=2).encode("utf-8")
        ti = tarfile.TarInfo("PACK_INFO.json")
        ti.size, ti.mtime = len(data), int(time.time())
        tar.addfile(ti, io.BytesIO(data))
        for f in files:
            tar.add(root / f, arcname=str(f), recursive=False)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n")
    info["sha256"] = digest
    return path, info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pack results, reports and logs into one tarball.")
    ap.add_argument("--include-lenses", action="store_true",
                    help="also pack lenses/*/translators.pt and config.json (~2 GB per model)")
    ap.add_argument("--require-complete", action="store_true",
                    help="refuse to pack if check_results.py finds problems")
    ap.add_argument("--out-dir", default=str(ROOT / "packs"))
    args = ap.parse_args(argv)

    print("Running check_results.py ...")
    ok, report = run_check_results()
    print(report.rstrip())
    if not ok:
        if args.require_complete:
            print("\n❌ check_results.py found problems; not packing (--require-complete).")
            return 1
        print("\n⚠️  check_results.py found problems: packing anyway, marked incomplete in PACK_INFO.json.")

    path, info = pack(ROOT, Path(args.out_dir), args.include_lenses, check=(ok, report))
    size = os.path.getsize(path)
    print(f"\n📦 {path}")
    print(f"   {info['n_files']} files, {human(info['bytes_uncompressed'])} → {human(size)} compressed")
    print(f"   sha256 {info['sha256']}  ({path.name}.sha256)")
    print(f"   results complete: {'yes' if ok else 'NO (see PACK_INFO.json)'}"
          + ("" if args.include_lenses else "; lenses not included (--include-lenses)"))
    if info["tuned_lenses_failed"]:
        print(f"   ⚠️  Tuned Lens not accepted: {', '.join(info['tuned_lenses_failed'])} (see PACK_INFO.json)")
    if info["git_local_changes"]:
        print(f"   local changes recorded: {len(info['git_local_changes'])} file(s)")
    print("   Unpack at the repository root: tar -xzf " + path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
