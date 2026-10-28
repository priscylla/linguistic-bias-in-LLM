"""
check_results.py
Verificação de integridade dos resultados brutos (results/lens/*/).

    python check_results.py

Para cada modelo do config confere:
    - manifest presente e todas as unidades esperadas feitas
    - cada prompt esperado (formatos do manifest) está em probs, com
      n_layers linhas por candidato, e cada prompt neutro em baseline
    - probabilidades em [0, 1]
    - KL até a final ≈ 0 na última camada (o lens reproduz o modelo)
    - manifest com o commit do registro
E soma as GPU-horas registradas por modelo.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.experiment_config import ExperimentConfig
from logit_lens.store import ChunkStore, lens_dir, read_manifest
from run_lens import lens_units

KL_LAST_TOL = 1e-3
PROMPT_KEYS = ["fact", "lang", "prompt_type", "paraphrase"]


def _expected(prompts, formats, facts=None):
    return {(p.fact, p.lang, p.type, p.paraphrase) for p in prompts
            if p.format in formats and (facts is None or p.fact in facts)}


def check_model(model_key: str, results_dir: str, config: ExperimentConfig) -> list:
    """Lista de problemas do modelo (vazia se tudo ok)."""
    out = lens_dir(results_dir, model_key)
    man = read_manifest(out)
    if not man:
        return [f"{model_key}: sem manifest ({out})"]
    formats  = tuple(man.get("formats", ["completion"]))
    problems = []
    spec = config.registry.models.get(model_key)
    if spec is not None:
        if formats != spec.formats:
            problems.append(f"{model_key}: formatos {formats} != registro {spec.formats}")
        if man.get("revision") not in (None, spec.revision):
            problems.append(f"{model_key}: commit {man.get('revision')} != registro {spec.revision}")

    store   = ChunkStore(out)
    missing = [u for u in lens_units(config, formats) if not store.done(u)]
    if missing:
        problems.append(f"{model_key}: {len(missing)} unidades faltando (ex.: {missing[:3]})")

    if not (out / "probs.parquet").exists():
        return problems + [f"{model_key}: sem probs.parquet (rode --consolidate-only)"]
    probs  = pd.read_parquet(out / "probs.parquet")
    layers = pd.read_parquet(out / "layers.parquet")
    base   = pd.read_parquet(out / "baseline.parquet") if (out / "baseline.parquet").exists() \
        else pd.DataFrame(columns=PROMPT_KEYS + ["prob"])

    got  = set(map(tuple, probs[PROMPT_KEYS].drop_duplicates().values))
    lack = _expected(config.prompt_specs, formats) - got
    if lack:
        problems.append(f"{model_key}: {len(lack)} prompts sem probs (ex.: {sorted(lack)[:3]})")

    got_b  = {(None, l, t, p) for l, t, p in
              base[["lang", "prompt_type", "paraphrase"]].drop_duplicates().itertuples(index=False)}
    lack_b = _expected(config.baseline_spec.prompts, formats) - got_b
    if lack_b:
        problems.append(f"{model_key}: {len(lack_b)} prompts neutros sem baseline")

    per = probs.groupby(PROMPT_KEYS + ["format", "candidate"]).size()
    n_l = man.get("n_layers")
    if n_l and (per != n_l).any():
        problems.append(f"{model_key}: {(per != n_l).sum()} conjuntos sem {n_l} camadas")

    for name, df in (("probs", probs), ("baseline", base)):
        if len(df) and ((df["prob"] < -1e-6) | (df["prob"] > 1 + 1e-6)).any():
            problems.append(f"{model_key}: {name} com probabilidade fora de [0, 1]")

    last = layers[layers["layer"] == layers["n_layers"]]
    if (last["kl_to_final"] > KL_LAST_TOL).any():
        problems.append(f"{model_key}: KL da última camada > {KL_LAST_TOL} "
                        f"(máx {last['kl_to_final'].max():.2e})")
    return problems


def check_all_results(results_dir: str = None, config: ExperimentConfig = None) -> bool:
    config      = config or ExperimentConfig()
    results_dir = results_dir or config.results_dir

    print("=" * 60)
    print("VERIFICAÇÃO DOS RESULTADOS (results/lens/)")
    print("=" * 60)
    problems, gpu_h = [], {}
    for model in config.models:
        p = check_model(model, results_dir, config)
        man = read_manifest(lens_dir(results_dir, model))
        if man:
            gpu_h[model] = man.get("gpu_hours", 0.0)
        hours = f"  {gpu_h[model]:.2f} GPU-h" if model in gpu_h else ""
        print(f"  {'✅' if not p else '❌'} {model}{hours}"
              + ("" if not p else f"  ({len(p)} problema(s))"))
        problems += p
    print(f"\n  GPU-horas registradas: {sum(gpu_h.values()):.2f}")

    if not problems:
        print("\n✅ TODOS OS RESULTADOS COMPLETOS E VÁLIDOS")
        print("   Próximo passo: python analyze.py && python make_figures.py")
        return True
    print(f"\n❌ {len(problems)} PROBLEMA(S):")
    for p in problems:
        print(f"   → {p}")
    print("\n   Rerun: python run_lens.py [modelo]  (unidades feitas são puladas)")
    return False


if __name__ == "__main__":
    sys.exit(0 if check_all_results() else 1)
