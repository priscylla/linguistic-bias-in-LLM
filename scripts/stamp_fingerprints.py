"""
scripts/stamp_fingerprints.py
Grava a impressão digital (logit_lens/fingerprint.py) das unidades
gravadas antes de 2026-10-30, que não têm uma. Sem impressão, run_lens.py
trata a unidade como velha e a refaz; com ela, só refaz as unidades cujas
entradas mudaram (ex.: nomes novos de um fato).

    python scripts/stamp_fingerprints.py --config-ref fa31f65            # verifica e grava
    python scripts/stamp_fingerprints.py --config-ref fa31f65 --dry-run  # só verifica

A impressão é calculada com o config do commit `--config-ref` (o que
produziu os resultados), e só depois de verificar, para cada modelo e
lens, que:
    - o sha256 de cada arquivo de config gravado no manifest é igual ao do
      arquivo nesse commit (e o config não mudou entre sessões);
    - Tuned Lens: os tradutores em lenses/{modelo}/translators.pt são
      anteriores à primeira sessão de medição (a lente medida é esta).
Um modelo que não passa na verificação não é carimbado (será refeito).
"""

import argparse
import hashlib
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.experiment_config import ExperimentConfig                     # noqa: E402
from config.loader import (load_baseline, load_experiment, load_facts,    # noqa: E402
                           load_forcing, load_models, load_prompts, load_tuned_lens)
from logit_lens.fingerprint import LEGACY_MEASURE_VERSION, file_sha256, fingerprint  # noqa: E402
from logit_lens.store import ChunkStore, read_manifest                    # noqa: E402


def git(*args) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def config_at(ref: str, dest: Path) -> dict:
    """Escreve config/*.yaml do commit `ref` em dest; devolve {arquivo: sha256}."""
    hashes = {}
    for name in git("ls-tree", "--name-only", f"{ref}:config").decode().split():
        if name.endswith(".yaml"):
            data = git("show", f"{ref}:config/{name}")
            (dest / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def load_config(d: Path) -> ExperimentConfig:
    d = str(d)
    return ExperimentConfig(experiment=load_experiment(d), fact_specs=load_facts(d),
                            prompt_specs=load_prompts(d), baseline_spec=load_baseline(d),
                            forcing_spec=load_forcing(d), registry=load_models(d))


def verify(out: Path, manifest: dict, ref_hashes: dict, lens: str, model: str):
    """None se ok; senão o motivo."""
    if not manifest:
        return "sem manifest"
    rec = manifest.get("config_sha256") or {}
    diff = [f for f, h in rec.items() if ref_hashes.get(f) != h]
    if diff:
        return f"config do manifest difere do commit em {diff}"
    if manifest.get("config_changes"):
        return "config mudou entre sessões"
    if lens == "tuned_lens":
        t = ROOT / load_tuned_lens()["lens_dir"] / model / "translators.pt"
        if not t.exists():
            return f"sem {t}"
        first = min(s["started"] for s in manifest.get("sessions", [{"started": "9999"}]))
        mtime = datetime.fromtimestamp(t.stat().st_mtime, timezone.utc).isoformat()
        if mtime > first:
            return f"tradutores ({mtime}) mais novos que a medição ({first})"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Carimba unidades antigas com a impressão digital verificada.")
    ap.add_argument("--config-ref", required=True, help="commit cujo config produziu os resultados")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        ref_hashes = config_at(args.config_ref, Path(tmp))
        old = load_config(Path(tmp))
    stamped, refused = 0, []
    for out in sorted((ROOT / args.results_dir / "lens").glob("*/*")):
        lens, model = out.parent.name, out.name
        why = verify(out, read_manifest(out), ref_hashes, lens, model)
        if why:
            refused.append(f"{lens}/{model}: {why}")
            continue
        lens_id = "logit_lens" if lens == "logit_lens" else \
            f"tuned_lens:{file_sha256(ROOT / load_tuned_lens()['lens_dir'] / model / 'translators.pt')}"
        store = ChunkStore(out)
        todo = [u for u in store.done_units() if store.fingerprint(u) is None]
        for u in todo:
            if not args.dry_run:
                store.stamp(u, fingerprint(u, old, lens_id, LEGACY_MEASURE_VERSION),
                            f"stamped {datetime.now(timezone.utc).isoformat()} from config at "
                            f"{args.config_ref} (manifest config_sha256 verified)")
        stamped += len(todo)
        print(f"  {'(dry run) ' if args.dry_run else ''}{lens}/{model}: {len(todo)} unidades carimbadas")
    for r in refused:
        print(f"  ❌ não carimbado (será refeito): {r}")
    print(f"\n{stamped} unidades {'a carimbar' if args.dry_run else 'carimbadas'}; {len(refused)} recusados")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
