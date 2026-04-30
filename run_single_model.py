"""
run_single_model.py
Roda os experimentos para um unico modelo.

Uso:
    python run_single_model.py llama
    python run_single_model.py mistral
    python run_single_model.py qwen
    python run_single_model.py gemma

Arquitetura de dois prompts:
    1. CHAT PROMPT    → get_final_response() → analise comportamental
    2. COMPLETION PROMPT → extractor.extract() → Logit Lens por camada

O Completion Prompt ("O aviao foi inventado por") tem o ultimo token
como preposicao, forcando o modelo a prever o nome do inventor.
Isso replica a metodologia do TalkTuner (2024).
"""

import sys
import os
import json
import torch
from pathlib import Path

from config.experiment_config import ExperimentConfig
from data.prompts import PROMPTS, COMPLETION_PROMPTS
from models.model_loader import load_model
from models.chat_template import apply_template
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

    # Retomar de checkpoint se existir
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

            keyword             = config.keyword_tokens[fact][lang]
            completion_prompt   = COMPLETION_PROMPTS[fact][lang]

            for formulation in formulations:
                current += 1

                existing = model_results[fact][lang].get(formulation, {})
                if "response" in existing and "last_token" in existing:
                    print(
                        f"  [{current:3d}/{total}] "
                        f"[{fact}][{lang}][{formulation}] -- checkpoint"
                    )
                    continue

                print(f"\n  [{current:3d}/{total}] [{fact}][{lang}][{formulation}]")

                try:
                    # ── 1. Resposta comportamental (chat template) ──
                    chat_prompt = PROMPTS[fact][lang][formulation]
                    formatted_chat, _ = apply_template(
                        tokenizer, model_key, chat_prompt
                    )
                    response = extractor.get_final_response(formatted_chat)
                    print(f"     Resposta: {response[:80]}...")

                    # ── 2. Logit Lens (completion-style prompt) ──
                    # Sem chat template — ultimo token e a preposicao
                    # que forca a predicao do nome do inventor
                    logit_data = extractor.extract(
                        prompt=completion_prompt,
                        keyword=keyword
                        # last_relevant_pos=None → usa len-1 automaticamente
                        # que e o ultimo token da completion ("por","by","von","da")
                    )

                    # Normalizar indices de camada
                    logit_data["last_token"] = normalize_layer_results(
                        logit_data["last_token"], model_key
                    )
                    if logit_data["keyword_token"]:
                        logit_data["keyword_token"] = normalize_layer_results(
                            logit_data["keyword_token"], model_key
                        )

                    # Calcular metricas
                    metrics_last = analyzer.compute_all_metrics(
                        logit_data["last_token"],
                        lang,
                        competing_entities=competing
                    )
                    metrics_kw = None
                    if logit_data["keyword_token"]:
                        metrics_kw = analyzer.compute_all_metrics(
                            logit_data["keyword_token"],
                            lang,
                            competing_entities=competing
                        )

                    model_results[fact][lang][formulation] = {
                        # Prompts usados
                        "prompt_raw":         chat_prompt,
                        "completion_prompt":  completion_prompt,
                        "prompt_formatted":   formatted_chat,

                        # Resultados comportamentais
                        "response":           response,

                        # Metadados do Logit Lens
                        "keyword_strategy":   logit_data.get("keyword_strategy", "N/A"),
                        "keyword_pos":        logit_data["keyword_pos"],
                        "n_layers":           logit_data["n_layers"],

                        # Dados do Logit Lens por camada
                        "last_token":         logit_data["last_token"],
                        "keyword_token":      logit_data["keyword_token"],

                        # Metricas calculadas
                        "metrics_last":       metrics_last,
                        "metrics_keyword":    metrics_kw
                    }

                    m = metrics_last
                    print(
                        f"     P(local):    {m['iv_local']:.4f}\n"
                        f"     P(compet.):  {m['iv_competing']:.4f}\n"
                        f"     CS_final:    {m['commitment_final']:+.4f}"
                    )

                except Exception as e:
                    print(f"     ERRO: {e}")
                    import traceback
                    traceback.print_exc()
                    errors.append({
                        "fact": fact, "lang": lang,
                        "form": formulation, "error": str(e)
                    })
                    model_results[fact][lang][formulation] = {
                        "error": str(e)
                    }

                # Checkpoint apos cada run
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(model_results, f, ensure_ascii=False, indent=2)

    completed = sum(
        1 for fact in model_results.values()
        for lang in fact.values()
        for form in lang.values()
        if "response" in form
    )

    print(f"\n{'='*50}")
    print(f"  {model_key} completo: {completed}/{total} runs")
    if errors:
        print(f"  Erros: {len(errors)}")
        for e in errors:
            print(f"    [{e['fact']}][{e['lang']}][e['form']]: {e['error'][:60]}")
    print(f"  Resultados: {output_path}")
    print(f"{'='*50}")

    del model, tokenizer, extractor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in VALID_MODELS:
        print(f"Uso: python run_single_model.py [{' | '.join(VALID_MODELS)}]")
        sys.exit(1)
    run_model(sys.argv[1])