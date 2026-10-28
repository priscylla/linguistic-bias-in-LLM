"""
compute_metrics.py
Calcula as metricas a partir dos JSONs de resultados
e salva em um arquivo separado, sem alterar os JSONs originais.

Uso:
    python compute_metrics.py
    python compute_metrics.py --results-dir results --output metrics_computed.csv

Saidas:
    results/metrics_computed.json  -- metricas por run (estrutura espelhada dos JSONs)
    results/metrics_computed.csv   -- tabela plana para analise e visualizacao
"""

import json
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.experiment_config import ExperimentConfig
from logit_lens.analyzer import LogitLensAnalyzer


def compute_all(results_dir: str = "results") -> dict:
    """
    Le os JSONs originais e calcula as metricas sem modificar nada.

    Returns:
        Dict com estrutura: {model: {fact: {lang: {form: {metrics}}}}}
    """
    config  = ExperimentConfig()
    models  = ["llama", "mistral", "qwen", "gemma"]
    facts   = ["aviao", "telefone"]
    langs   = ["pt", "en", "de", "it"]
    forms   = ["F1", "F2", "F3"]

    computed  = {}
    rows      = []

    for model in models:
        path = os.path.join(results_dir, f"{model}_results.json")
        if not os.path.exists(path):
            print(f"  [{model}] nao encontrado — pulando")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        computed[model] = {}
        n_runs = 0

        for fact in facts:
            computed[model][fact] = {}
            analyzer  = LogitLensAnalyzer(config.expected_entities[fact])
            competing = config.competing_entities[fact]

            for lang in langs:
                computed[model][fact][lang] = {}

                for form in forms:
                    run = data.get(fact, {}).get(lang, {}).get(form, {})

                    if "last_token" not in run:
                        computed[model][fact][lang][form] = {"error": "no last_token"}
                        continue

                    # Calcular metricas — sem tocar no JSON original
                    metrics_last = analyzer.compute_all_metrics(
                        run["last_token"], lang,
                        competing_entities=competing
                    )
                    metrics_kw = None
                    if run.get("keyword_token"):
                        metrics_kw = analyzer.compute_all_metrics(
                            run["keyword_token"], lang,
                            competing_entities=competing
                        )

                    computed[model][fact][lang][form] = {
                        "prompt":          run.get("prompt", ""),
                        "response":        run.get("response", ""),
                        "metrics_last":    metrics_last,
                        "metrics_keyword": metrics_kw,
                    }
                    n_runs += 1

                    # Linha para o CSV
                    stake = {"aviao": "pt", "telefone": "it"}[fact]
                    rows.append({
                        "model":             model,
                        "fact":              fact,
                        "language":          lang,
                        "formulation":       form,
                        "has_stake":         lang == stake,
                        "prompt":            run.get("prompt", ""),
                        "response_snippet":  run.get("response", "")[:80],

                        # Metricas da ultima camada
                        "iv_local_final":    metrics_last.get("iv_local_final", 0),
                        "iv_competing":      metrics_last.get("iv_competing", 0),
                        "commitment_final":  metrics_last.get("commitment_final", 0),

                        # Metricas do pico
                        "iv_local_peak":     metrics_last.get("iv_local_peak", 0),
                        "peak_layer":        metrics_last.get("peak_layer"),
                        "peak_layer_norm":   metrics_last.get("peak_layer_norm"),
                        "cs_at_peak":        metrics_last.get("cs_at_peak", 0),

                        # Supressao gramatical
                        "suppression":       metrics_last.get("suppression", 0),

                        # Convergencia cultural
                        "pcc":               metrics_last.get("pcc"),
                        "pcc_norm":          metrics_last.get("pcc_norm"),
                    })

        print(f"  [{model}] {n_runs} runs calculados")

    return computed, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-json", default="results/metrics_computed.json")
    parser.add_argument("--output-csv",  default="results/metrics_computed.csv")
    args = parser.parse_args()

    print("Calculando metricas (JSONs originais nao serao alterados)...")
    computed, df = compute_all(args.results_dir)

    # Salvar JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(computed, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON salvo: {args.output_json}")

    # Salvar CSV
    df.to_csv(args.output_csv, index=False)
    print(f"  CSV salvo:  {args.output_csv}")
    print(f"  Linhas:     {len(df)}")

    # Verificacao rapida
    print("\n=== VERIFICACAO: cs_at_peak (F1) ===")
    f1 = df[df["formulation"] == "F1"]
    for fact in ["aviao", "telefone"]:
        print(f"\n[{fact.upper()}]")
        pivot = f1[f1["fact"] == fact].pivot_table(
            index="model", columns="language",
            values="cs_at_peak", aggfunc="first"
        ).round(3)
        print(pivot.to_string())


if __name__ == "__main__":
    main()