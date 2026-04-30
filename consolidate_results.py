"""
consolidate_results.py
Consolida os JSONs de todos os modelos em metrics_comparison.csv.

Executar apos todos os modelos terminarem:
    python consolidate_results.py

Tambem pode ser executado parcialmente -- consolida
apenas os modelos cujo JSON ja existe em results/.
"""

import json
import os
import sys
import pandas as pd
from pathlib import Path

from config.experiment_config import ExperimentConfig
from logit_lens.normalizer import build_comparison_table, has_cultural_stake, normalize_metric


def consolidate(results_dir: str = "results") -> pd.DataFrame:

    config = ExperimentConfig()

    print("=" * 50)
    print("CONSOLIDANDO RESULTADOS")
    print("=" * 50)

    all_results = {}

    for model_key in ["llama", "mistral", "qwen", "gemma"]:
        path = os.path.join(results_dir, f"{model_key}_results.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                all_results[model_key] = json.load(f)
            print(f"  carregado: {model_key}")
        else:
            print(f"  nao encontrado (pulando): {path}")

    if not all_results:
        print("\nNenhum resultado encontrado em results/")
        print("Execute primeiro: python run_single_model.py [modelo]")
        sys.exit(1)

    df = build_comparison_table(all_results, config)

    output = os.path.join(results_dir, "metrics_comparison.csv")
    df.to_csv(output, index=False)

    print(f"\n  {len(df)} runs consolidados")
    print(f"  Modelos:  {df['model'].unique().tolist()}")
    print(f"  Colunas:  {list(df.columns)}")
    print(f"\n  Salvo em: {output}")

    # Preview das metricas principais
    print("\n  Preview -- Commitment Score final por idioma/fato (F1, LLaMA):")
    preview = df[
        (df["formulation"] == "F1") & (df["model"] == df["model"].iloc[0])
    ][["fact", "language", "iv_local", "iv_competing", "commitment_final"]]
    print(preview.to_string(index=False))

    return df


if __name__ == "__main__":
    consolidate()
