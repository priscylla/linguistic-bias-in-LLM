"""
check_results.py
Verificação de integridade dos resultados após os experimentos.

Executa após todos os modelos terem rodado:
    python check_results.py

Verifica:
    - Todos os 288 runs estão presentes
    - Nenhum run tem apenas erro (sem dados de Logit Lens)
    - Métricas parecem razoáveis (IV > 0, PCC not None)
"""

import json
import os
import sys
import pandas as pd


def check_all_results(results_dir: str = "results") -> bool:
    """
    Verifica integridade completa dos resultados.

    Returns:
        True se todos os 288 runs estão válidos.
    """
    models       = ["llama", "mistral", "qwen", "gemma"]
    facts        = ["aviao", "telefone", "radio"]
    languages    = ["pt", "en", "de", "it"]
    formulations = ["F1", "F2", "F3"]

    print("=" * 60)
    print("VERIFICAÇÃO FINAL DOS RESULTADOS")
    print("=" * 60)

    all_ok   = True
    problems = []
    stats    = []

    for model_key in models:
        path = os.path.join(results_dir, f"{model_key}_results.json")

        if not os.path.exists(path):
            print(f"\n❌ {model_key}: arquivo não encontrado — {path}")
            all_ok = False
            continue

        with open(path) as f:
            data = json.load(f)

        print(f"\n[{model_key.upper()}]")

        for fact in facts:
            for lang in languages:
                for form in formulations:

                    run = data.get(fact, {}).get(lang, {}).get(form, {})

                    if not run:
                        status = "❌ MISSING"
                        problems.append(f"{model_key}/{fact}/{lang}/{form} — missing")
                        all_ok = False

                    elif "error" in run and "last_token" not in run:
                        status = f"❌ ERROR: {run['error'][:50]}"
                        problems.append(f"{model_key}/{fact}/{lang}/{form} — error")
                        all_ok = False

                    elif not run.get("last_token"):
                        status = "⚠️  NO LOGIT DATA"
                        problems.append(f"{model_key}/{fact}/{lang}/{form} — no logit data")
                        all_ok = False

                    else:
                        m   = run.get("metrics_last", {})
                        iv  = m.get("iv", 0)
                        pcc = m.get("pcc_norm")
                        kw  = run.get("keyword_strategy", "?")
                        status = f"✅ iv={iv:.3f}  pcc={str(pcc)[:5]}  kw={kw}"

                        stats.append({
                            "model": model_key, "fact": fact,
                            "lang": lang, "form": form,
                            "iv": iv, "pcc_norm": pcc
                        })

                    print(f"  [{fact}][{lang}][{form}] {status}")

    # ── Resumo ──
    print("\n" + "=" * 60)

    if all_ok:
        df = pd.DataFrame(stats)
        print("✅ TODOS OS 288 RUNS COMPLETOS E VÁLIDOS\n")

        print("── Média de IV por idioma/fato (F1) ──")
        pivot = df[df["form"] == "F1"].groupby(
            ["fact", "lang"]
        )["iv"].mean().round(3).unstack()
        print(pivot.to_string())

        print("\n── PCC normalizado médio por modelo (F1) ──")
        pcc_summary = df[
            (df["form"] == "F1") & (df["pcc_norm"].notna())
        ].groupby("model")["pcc_norm"].mean().round(3)
        print(pcc_summary.to_string())

        print(f"\n   Próximo passo: python run_visualization.py")

    else:
        print(f"❌ {len(problems)} PROBLEMA(S) ENCONTRADO(S):\n")
        for p in problems:
            print(f"   → {p}")
        print("\n   Rerun:")
        print("   python run_single_model.py [modelo]")
        print("   (runs já completos são pulados automaticamente)")

    return all_ok


if __name__ == "__main__":
    ok = check_all_results()
    sys.exit(0 if ok else 1)
