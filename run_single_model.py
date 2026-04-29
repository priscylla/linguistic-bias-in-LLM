"""
run_single_model.py
Roda os experimentos para um único modelo.

Uso:
    python run_single_model.py llama
    python run_single_model.py mistral
    python run_single_model.py qwen
    python run_single_model.py gemma

Características:
    - Checkpoint automático: salva após cada run, retoma de onde parou
    - 36 runs por modelo (3 fatos × 4 idiomas × 3 formulações)
    - Loga progresso e métricas em tempo real
"""

import sys
import os
import json
import torch
from pathlib import Path

from config.experiment_config import ExperimentConfig
from data.prompts import PROMPTS
from models.model_loader import load_model
from models.chat_template import apply_template
from logit_lens.extractor import LogitLensExtractor
from logit_lens.analyzer import LogitLensAnalyzer
from logit_lens.normalizer import normalize_layer_results


VALID_MODELS = ["llama", "mistral", "qwen", "gemma"]


def run_model(model_key: str):
    """
    Executa todos os 36 runs para um modelo específico.

    Usa checkpoint incremental — se o arquivo de resultados
    já existir, pula os runs já completados e continua
    a partir do ponto de interrupção.

    Args:
        model_key: um de "llama", "mistral", "qwen", "gemma"
    """
    config = ExperimentConfig()
    Path(config.results_dir).mkdir(exist_ok=True)

    output_path = os.path.join(
        config.results_dir, f"{model_key}_results.json"
    )

    # ── Retomar de checkpoint se existir ──
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            model_results = json.load(f)
        print(f"♻️  Retomando de checkpoint: {output_path}")
    else:
        model_results = {}

    # ── Carregar modelo ──
    model_name = config.models[model_key]
    model, tokenizer = load_model(model_name, model_key)
    extractor = LogitLensExtractor(model, tokenizer, config.top_k_tokens)

    facts        = ["aviao", "telefone", "radio"]
    languages    = ["pt", "en", "de", "it"]
    formulations = ["F1", "F2", "F3"]

    total   = len(facts) * len(languages) * len(formulations)
    current = 0
    errors  = []

    for fact in facts:
        if fact not in model_results:
            model_results[fact] = {}

        analyzer = LogitLensAnalyzer(config.expected_entities[fact])

        for lang in languages:
            if lang not in model_results[fact]:
                model_results[fact][lang] = {}

            keyword = config.keyword_tokens[fact][lang]

            for formulation in formulations:
                current += 1

                # ── Pular se já completado ──
                existing = model_results[fact][lang].get(formulation, {})
                if "response" in existing and "last_token" in existing:
                    print(
                        f"  ⏭️  [{current:3d}/{total}] "
                        f"[{fact}][{lang}][{formulation}] — checkpoint"
                    )
                    continue

                print(
                    f"\n  🔄 [{current:3d}/{total}] "
                    f"[{fact}][{lang}][{formulation}]"
                )

                try:
                    prompt_raw = PROMPTS[fact][lang][formulation]

                    # Aplicar chat template
                    formatted, last_pos = apply_template(
                        tokenizer, model_key, prompt_raw
                    )

                    # Ground truth — resposta final
                    response = extractor.get_final_response(formatted)
                    print(f"     Resposta: {response[:80]}...")

                    # Keyword finding via keyword_finder robusto
                    from data.keyword_finder import find_keyword_position_robust
                    input_ids = tokenizer.encode(
                        formatted, add_special_tokens=False
                    )
                    kw_pos, kw_strategy = find_keyword_position_robust(
                        tokenizer, input_ids, keyword, model_key
                    )

                    # Logit Lens
                    logit_data = extractor.extract(
                        prompt=formatted,
                        keyword=keyword,
                        last_relevant_pos=last_pos,
                        keyword_pos_override=kw_pos
                    )

                    # Normalizar índices de camada para 0-1
                    logit_data["last_token"] = normalize_layer_results(
                        logit_data["last_token"], model_key
                    )
                    if logit_data["keyword_token"]:
                        logit_data["keyword_token"] = normalize_layer_results(
                            logit_data["keyword_token"], model_key
                        )

                    # Calcular métricas
                    metrics_last = analyzer.compute_all_metrics(
                        logit_data["last_token"], lang
                    )
                    metrics_kw = None
                    if logit_data["keyword_token"]:
                        metrics_kw = analyzer.compute_all_metrics(
                            logit_data["keyword_token"], lang
                        )

                    # Armazenar resultado completo
                    model_results[fact][lang][formulation] = {
                        "prompt_raw":        prompt_raw,
                        "prompt_formatted":  formatted,
                        "response":          response,
                        "keyword_strategy":  kw_strategy,
                        "keyword_pos":       kw_pos,
                        "last_relevant_pos": last_pos,
                        "n_layers":          logit_data["n_layers"],
                        "last_token":        logit_data["last_token"],
                        "keyword_token":     logit_data["keyword_token"],
                        "metrics_last":      metrics_last,
                        "metrics_keyword":   metrics_kw
                    }

                    # Log das métricas
                    m = metrics_last
                    print(
                        f"     PCC: {m['pcc']} ({m['pcc_norm']})  "
                        f"IV: {m['iv']:.4f}  "
                        f"KW: {kw_strategy}"
                    )

                except Exception as e:
                    print(f"     ❌ ERRO: {e}")
                    errors.append({
                        "fact": fact, "lang": lang,
                        "form": formulation, "error": str(e)
                    })
                    # Registrar erro mas continuar
                    model_results[fact][lang][formulation] = {
                        "error": str(e)
                    }

                # ── Checkpoint após cada run ──
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        model_results, f,
                        ensure_ascii=False, indent=2
                    )

    # ── Resumo final ──
    completed = sum(
        1 for fact in model_results.values()
        for lang in fact.values()
        for form in lang.values()
        if "response" in form
    )

    print(f"\n{'='*50}")
    print(f"✅ {model_key} completo")
    print(f"   Runs concluídos: {completed}/{total}")
    if errors:
        print(f"   Erros: {len(errors)}")
        for e in errors:
            print(f"     [{e['fact']}][{e['lang']}][{e['form']}]: {e['error'][:60]}")
    print(f"   Resultados: {output_path}")
    print(f"{'='*50}")

    del model, tokenizer, extractor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in VALID_MODELS:
        print(f"Uso: python run_single_model.py [{' | '.join(VALID_MODELS)}]")
        sys.exit(1)

    run_model(sys.argv[1])
