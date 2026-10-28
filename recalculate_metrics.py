import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.experiment_config import ExperimentConfig
from logit_lens.analyzer import LogitLensAnalyzer
from logit_lens.normalizer import build_comparison_table

config  = ExperimentConfig()
models  = ["llama", "mistral", "qwen", "gemma"]
facts   = ["aviao", "telefone"]
langs   = ["pt", "en", "de", "it"]
forms   = ["F1", "F2", "F3"]

print("Recalculando metricas...")
all_results = {}

for model in models:
    path = f"results/{model}_results.json"
    if not os.path.exists(path):
        print(f"  [{model}] nao encontrado — pulando")
        continue

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    all_results[model] = data
    updated = 0

    for fact in facts:
        analyzer  = LogitLensAnalyzer(config.expected_entities[fact])
        competing = config.competing_entities[fact]

        for lang in langs:
            for form in forms:
                run = data.get(fact, {}).get(lang, {}).get(form, {})
                if "last_token" not in run:
                    continue

                run["metrics_last"] = analyzer.compute_all_metrics(
                    run["last_token"], lang,
                    competing_entities=competing
                )
                if run.get("keyword_token"):
                    run["metrics_keyword"] = analyzer.compute_all_metrics(
                        run["keyword_token"], lang,
                        competing_entities=competing
                    )
                updated += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  [{model}] {updated} runs atualizados")

# Regenerar CSV
df = build_comparison_table(all_results, config)
df.to_csv("results/metrics_comparison.csv", index=False)
print(f"\n  CSV salvo: results/metrics_comparison.csv")

# Verificar resultado
print("\n=== VERIFICACAO: iv_local_peak (F1) ===")
for model in [m for m in models if os.path.exists(f"results/{m}_results.json")]:
    with open(f"results/{model}_results.json") as f:
        d = json.load(f)
    print(f"[{model}]")
    for fact in facts:
        for lang in ["pt", "en", "it"]:
            m = d[fact][lang]["F1"].get("metrics_last", {})
            pk = m.get("iv_local_peak", "MISSING")
            cs = m.get("cs_at_peak",    "MISSING")
            if pk != "MISSING" and float(str(pk)) > 0.01:
                print(f"  [{fact}][{lang}] peak={pk:.4f}  cs_peak={cs:.4f}")