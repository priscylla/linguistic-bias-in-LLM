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
from logit_lens.store import ChunkStore, behavioral_dir, lens_dir, read_manifest
from run_lens import lens_units

KL_LAST_TOL = 1e-3
PROMPT_KEYS = ["fact", "lang", "prompt_type", "paraphrase"]


def _expected(prompts, formats, facts=None):
    return {(p.fact, p.lang, p.type, p.paraphrase) for p in prompts
            if p.format in formats and (facts is None or p.fact in facts)}


def check_model(model_key: str, results_dir: str, config: ExperimentConfig,
                lens: str = "logit_lens") -> list:
    """Lista de problemas do modelo (vazia se tudo ok)."""
    out = lens_dir(results_dir, model_key, lens)
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
    # Células de indicador puladas pela regra dos dois candidatos não faltam
    skipped = set()
    if (out / "skipped.parquet").exists():
        sk = pd.read_parquet(out / "skipped.parquet")
        skipped = set(map(tuple, sk[PROMPT_KEYS].drop_duplicates().values))
    lack = _expected(config.prompt_specs, formats) - got - skipped
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


def check_behavioral(model_key: str, results_dir: str, config: ExperimentConfig,
                     behavioral=None) -> list:
    """Respostas comportamentais: todo prompt QA com greedy + amostras."""
    from config.loader import load_behavioral
    out = behavioral_dir(results_dir, model_key)
    if not read_manifest(out):
        return [f"{model_key}: sem respostas comportamentais ({out})"]
    if not (out / "answers.parquet").exists():
        return [f"{model_key}: sem answers.parquet (rode run_behavioral.py --consolidate-only)"]
    g = (behavioral or load_behavioral()).generation
    per = g["greedy"] + g["samples"]
    a = pd.read_parquet(out / "answers.parquet")
    exp = {(p.fact, p.lang, p.paraphrase) for p in config.prompt_specs if p.type == "QA"}
    got = a.groupby(["fact", "lang", "paraphrase"]).size()
    problems = []
    lack = exp - set(got.index)
    if lack:
        problems.append(f"{model_key}: {len(lack)} prompts QA sem respostas")
    if (got != per).any():
        problems.append(f"{model_key}: {(got != per).sum()} prompts sem {per} respostas")
    if a["text"].isna().any():
        problems.append(f"{model_key}: respostas sem texto")
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
    # Tuned Lens: só os modelos que já têm resultados com ele
    for model in config.models:
        if read_manifest(lens_dir(results_dir, model, "tuned_lens")):
            p = check_model(model, results_dir, config, "tuned_lens")
            man = read_manifest(lens_dir(results_dir, model, "tuned_lens"))
            gpu_h[f"{model}/tuned"] = man.get("gpu_hours", 0.0)
            print(f"  {'✅' if not p else '❌'} {model} (tuned lens)  "
                  f"{gpu_h[f'{model}/tuned']:.2f} GPU-h")
            problems += [f"[tuned] {x}" for x in p]
    # Comportamental: modelos instruct
    for model, spec in config.registry.models.items():
        if spec.kind != "instruct":
            continue
        p = check_behavioral(model, results_dir, config)
        man = read_manifest(behavioral_dir(results_dir, model))
        if man:
            gpu_h[f"{model}/behavioral"] = man.get("gpu_hours", 0.0)
        print(f"  {'✅' if not p else '❌'} {model} (comportamental)")
        problems += [f"[comportamental] {x}" for x in p]
    print(f"\n  GPU-horas registradas: {sum(gpu_h.values()):.2f}")

    if not problems:
        print("\n✅ TODOS OS RESULTADOS COMPLETOS E VÁLIDOS")
        print("   Próximo passo: python analyze.py && python make_figures.py")
        return True
    print(f"\n❌ {len(problems)} PROBLEMA(S):")
    for p in problems:
        print(f"   → {p}")
    print("\n   Rerun: python run_lens.py --model <modelo> [--lens tuned] / "
          "python run_behavioral.py --model <modelo>  (unidades feitas são puladas)")
    return False


if __name__ == "__main__":
    sys.exit(0 if check_all_results() else 1)
