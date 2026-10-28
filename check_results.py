"""
check_results.py
Verificação de integridade dos resultados após os experimentos.

    python check_results.py

O número de runs esperado vem do config
(modelos × fatos × idiomas × formulações). Verifica:
    - todos os runs esperados estão presentes
    - nenhum run tem apenas erro (sem dados de Logit Lens)
    - cada camada tem as probabilidades exatas (entity_probs) e a
      auditoria dos token IDs (entity_tokens) está presente
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze import raw_results_path
from config.experiment_config import ExperimentConfig


def check_run(run: dict):
    """Retorna None se o run está válido, senão a descrição do problema."""
    if not run:
        return "missing"
    if "error" in run and "last_token" not in run:
        return f"error: {run['error'][:50]}"
    layers = run.get("last_token")
    if not layers:
        return "no logit data"
    if any("entity_probs" not in l for l in layers):
        return "old format (no entity_probs)"
    if not run.get("entity_tokens"):
        return "no entity_tokens"
    return None


def check_all_results(results_dir: str = None, config: ExperimentConfig = None) -> bool:
    """
    Verifica integridade completa dos resultados.

    Returns:
        True se todos os runs esperados estão presentes e válidos.
    """
    config      = config or ExperimentConfig()
    results_dir = results_dir or config.results_dir
    expected    = config.expected_cells()

    print("=" * 60)
    print(f"VERIFICAÇÃO DOS RESULTADOS — {expected} runs esperados")
    print("=" * 60)

    problems = []
    valid    = 0

    for model in config.models:
        path = raw_results_path(results_dir, model)
        if not os.path.exists(path):
            print(f"\n❌ {model}: arquivo não encontrado — {path}")
            n = len(config.facts) * len(config.languages) * len(config.formulations)
            problems.append(f"{model} — arquivo ausente ({n} runs)")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        print(f"\n[{model.upper()}]")
        for fact in config.facts:
            for lang in config.languages:
                for form in config.formulations:
                    run = data.get(fact, {}).get(lang, {}).get(form, {})
                    problem = check_run(run)
                    cell = f"{model}/{fact}/{lang}/{form}"
                    if problem:
                        problems.append(f"{cell} — {problem}")
                        print(f"  ❌ [{fact}][{lang}][{form}] {problem}")
                    else:
                        valid += 1
                        m = run.get("metrics_last", {})
                        print(f"  ✅ [{fact}][{lang}][{form}] "
                              f"CS_final={m.get('commitment_final', float('nan')):+.3f}")

    print("\n" + "=" * 60)
    print(f"  {valid}/{expected} runs válidos")
    if not problems:
        print("✅ TODOS OS RUNS COMPLETOS E VÁLIDOS")
        print("   Próximo passo: python analyze.py && python make_figures.py")
        return True

    print(f"❌ {len(problems)} PROBLEMA(S):")
    for p in problems:
        print(f"   → {p}")
    print("\n   Rerun: python run_single_model.py [modelo]"
          "  (runs completos são pulados)")
    return False


if __name__ == "__main__":
    sys.exit(0 if check_all_results() else 1)
