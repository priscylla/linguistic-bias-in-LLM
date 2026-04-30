import sys
import os
import json
import torch
from pathlib import Path

from config.experiment_config import ExperimentConfig
from data.prompts import COMPLETION_PROMPTS
from models.model_loader import load_model
from logit_lens.extractor import LogitLensExtractor
from logit_lens.analyzer import LogitLensAnalyzer
from logit_lens.normalizer import normalize_layer_results


VALID_MODELS = ["llama", "mistral", "qwen", "gemma"]


def run_model(model_key: str):
    config = ExperimentConfig()
    Path(config.results_dir).mkdir(exist_ok=True)

    output_path = os.path.join(
        config.results_dir, f"{model_key}_results.json"
    )

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            model_results = json.load(f)
        print(f"Retomando de checkpoint: {output_path}")
    else:
        model_results = {}

    model_name = config.models[model_key]
    model, tokenizer = load_model(model_name, model_key)
    extractor = LogitLensExtractor(model, tokenizer, config.top_k_tokens)

    facts        = ["aviao", "telefone"]
    languages    = ["pt", "en", "de", "it"]
    formulations = ["F1", "F2", "F3"]

    total   = len(facts) * len(languages) * len(formulations)
    current = 0
    errors  = []

    for fact in facts:
        if fact not in model_results:
            model_results[fact] = {}

        analyzer  = LogitLensAnalyzer(config.expected_entities[fact])
        competing = config.competing_entities[fact]

        for lang in languages:
            if lang not in model_results[fact]:
                model_results[fact][lang] = {}

            keyword = config.keyword_tokens[fact][lang]

            for formulation in formulations:
                current += 1

                existing = model_results[fact][lang].get(formulation, {})
                if "response" in existing and "last_token" in existing:
                    print(
                        f"  [{current:3d}/{total}] "
                        f"[{fact}][{lang}][{formulation}] -- checkpoint"
                    )
                    continue

                prompt = COMPLETION_PROMPTS[fact][lang][formulation]

                print(f"\n  [{current:3d}/{total}] [{fact}][{lang}][{formulation}]")
                print(f"     Prompt: \"{prompt}\"")

                try:
                    # ── 1. Resposta comportamental ──
                    # Mesmo prompt completion usado diretamente
                    # sem chat template — modelo completa a frase
                    response = extractor.get_final_response(prompt)
                    print(f"     Resposta: {response[:80]}...")

                    # ── 2. Logit Lens ──
                    # Mesmo prompt — analisa ativacoes internas
                    # na posicao do ultimo token da completion
                    logit_data = extractor.extract(
                        prompt=prompt,
                        keyword=keyword
                    )

                    logit_data["last_token"] = normalize_layer_results(
                        logit_data["last_token"], model_key
                    )
                    if logit_data["keyword_token"]:
                        logit_data["keyword_token"] = normalize_layer_results(
                            logit_data["keyword_token"], model_key
                        )

                    metrics_last = analyzer.compute_all_metrics(
                        logit_data["last_token"], lang,
                        competing_entities=competing
                    )
                    metrics_kw = None
                    if logit_data["keyword_token"]:
                        metrics_kw = analyzer.compute_all_metrics(
                            logit_data["keyword_token"], lang,
                            competing_entities=competing
                        )

                    model_results[fact][lang][formulation] = {
                        "prompt":          prompt,
                        "response":        response,
                        "keyword_strategy": logit_data.get("keyword_strategy", "N/A"),
                        "keyword_pos":     logit_data["keyword_pos"],
                        "n_layers":        logit_data["n_layers"],
                        "last_token":      logit_data["last_token"],
                        "keyword_token":   logit_data["keyword_token"],
                        "metrics_last":    metrics_last,
                        "metrics_keyword": metrics_kw
                    }

                    m = metrics_last
                    print(
                        f"     P(local):   {m['iv_local']:.4f}\n"
                        f"     P(compet.): {m['iv_competing']:.4f}\n"
                        f"     CS_final:   {m['commitment_final']:+.4f}"
                    )

                except Exception as e:
                    print(f"     ERRO: {e}")
                    import traceback; traceback.print_exc()
                    errors.append({
                        "fact": fact, "lang": lang,
                        "form": formulation, "error": str(e)
                    })
                    model_results[fact][lang][formulation] = {
                        "error": str(e)
                    }

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(model_results, f, ensure_ascii=False, indent=2)

    completed = sum(
        1 for f in model_results.values()
        for l in f.values()
        for form in l.values()
        if "response" in form
    )

    print(f"\n{'='*50}")
    print(f"  {model_key} completo: {completed}/{total} runs")
    if errors:
        print(f"  Erros: {len(errors)}")
        for e in errors:
            print(f"    [{e['fact']}][{e['lang']}][{e['form']}]: {e['error'][:60]}")
    print(f"  Resultados: {output_path}")
    print(f"{'='*50}")

    del model, tokenizer, extractor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in VALID_MODELS:
        print(f"Uso: python run_single_model.py [{' | '.join(VALID_MODELS)}]")
        sys.exit(1)
    run_model(sys.argv[1])