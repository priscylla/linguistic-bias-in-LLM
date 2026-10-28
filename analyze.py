"""
analyze.py
Única implementação do cálculo de métricas a partir dos resultados
brutos. Substitui compute_metrics.py, recalculate_metrics.py,
consolidate_results.py e o recálculo dentro de generate_visualizations.py.

Os resultados brutos (results/{modelo}_results.json) são apenas lidos;
nada aqui os modifica.

Uso:
    python analyze.py
    python analyze.py --results-dir results --output-dir results

Saídas:
    {output-dir}/metrics_computed.json  -- métricas por run
                                            {modelo: {fato: {idioma: {form: ...}}}}
    {output-dir}/metrics_computed.csv   -- tabela plana (uma linha por run)
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.experiment_config import ExperimentConfig
from logit_lens.analyzer import LogitLensAnalyzer

RAW_SUFFIX = "_results.json"


def raw_results_path(results_dir: str, model: str) -> str:
    return os.path.join(results_dir, f"{model}{RAW_SUFFIX}")


def _guard_output(path: str):
    """Impede que uma saída sobrescreva um arquivo de resultados brutos."""
    if path.endswith(RAW_SUFFIX):
        raise ValueError(
            f"Saída '{path}' tem o nome de um resultado bruto "
            f"(*{RAW_SUFFIX}); analyze.py nunca modifica resultados brutos."
        )


def load_raw_results(results_dir: str, config: ExperimentConfig) -> Dict:
    """Lê os JSONs brutos de todos os modelos do config que existirem."""
    data = {}
    for model in config.models:
        path = raw_results_path(results_dir, model)
        if not os.path.exists(path):
            print(f"  [{model}] não encontrado — pulando")
            continue
        with open(path, encoding="utf-8") as f:
            data[model] = json.load(f)
    return data


def compute_all(raw: Dict, config: ExperimentConfig) -> Tuple[Dict, pd.DataFrame]:
    """
    Calcula as métricas de todos os runs sem tocar nos dados brutos.

    Returns:
        (computed, df): dict espelhando a estrutura dos JSONs e tabela plana
    """
    analyzer = LogitLensAnalyzer()
    computed = {}
    rows     = []

    for model, data in raw.items():
        computed[model] = {}
        n_runs = 0
        for fact in config.facts:
            computed[model][fact] = {}
            for lang in config.languages:
                computed[model][fact][lang] = {}
                for form in config.formulations:
                    run = data.get(fact, {}).get(lang, {}).get(form, {})

                    if "last_token" not in run:
                        computed[model][fact][lang][form] = {"error": "no last_token"}
                        continue

                    metrics_last = analyzer.compute_all_metrics(run["last_token"])
                    metrics_kw = None
                    if run.get("keyword_token"):
                        metrics_kw = analyzer.compute_all_metrics(run["keyword_token"])

                    n_layers = run.get("n_layers", len(run["last_token"]))
                    computed[model][fact][lang][form] = {
                        "prompt":          run.get("prompt", ""),
                        "response":        run.get("response", ""),
                        "n_layers":        n_layers,
                        "metrics_last":    metrics_last,
                        "metrics_keyword": metrics_kw,
                    }
                    n_runs += 1

                    rows.append({
                        "model":            model,
                        "fact":             fact,
                        "language":         lang,
                        "formulation":      form,
                        "has_stake":        config.stake(fact, lang),
                        "n_layers":         n_layers,
                        "prompt":           run.get("prompt", ""),
                        "response_snippet": run.get("response", "")[:80],

                        # Ultima camada
                        "iv_local_final":   metrics_last.get("iv_local_final"),
                        "iv_competing":     metrics_last.get("iv_competing"),
                        "commitment_final": metrics_last.get("commitment_final"),

                        # Pico
                        "iv_local_peak":    metrics_last.get("iv_local_peak"),
                        "peak_layer":       metrics_last.get("peak_layer"),
                        "peak_layer_norm":  metrics_last.get("peak_layer_norm"),
                        "cs_at_peak":       metrics_last.get("cs_at_peak"),

                        # Supressao e convergencia
                        "suppression":      metrics_last.get("suppression"),
                        "pcc":              metrics_last.get("pcc"),
                        "pcc_norm":         metrics_last.get("pcc_norm"),
                    })
        print(f"  [{model}] {n_runs} runs calculados")

    return computed, pd.DataFrame(rows)


def main(argv=None):
    config = ExperimentConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=config.results_dir)
    parser.add_argument("--output-dir",  default=config.results_dir)
    args = parser.parse_args(argv)

    out_json = os.path.join(args.output_dir, "metrics_computed.json")
    out_csv  = os.path.join(args.output_dir, "metrics_computed.csv")
    for p in (out_json, out_csv):
        _guard_output(p)

    print("Calculando métricas (resultados brutos não são alterados)...")
    raw = load_raw_results(args.results_dir, config)
    if not raw:
        print(f"Nenhum resultado em {args.results_dir}/. "
              f"Rode antes: python run_single_model.py [modelo]")
        return 1

    computed, df = compute_all(raw, config)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(computed, f, ensure_ascii=False, indent=2)
    df.to_csv(out_csv, index=False)
    print(f"\n  JSON: {out_json}\n  CSV:  {out_csv}  ({len(df)} linhas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
