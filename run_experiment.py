import os
import json
import torch
import pandas as pd
from pathlib import Path

from config.experiment_config import ExperimentConfig
from data.prompts import PROMPTS
from models.model_loader import load_model
from models.chat_template import apply_template, verify_templates
from logit_lens.extractor import LogitLensExtractor
from logit_lens.analyzer import LogitLensAnalyzer
from logit_lens.normalizer import (
    normalize_layer_results,
    build_comparison_table
)
from data.keyword_finder import (
    find_keyword_position_robust,
    verify_keyword_finding
)


def run_verification_phase(config: ExperimentConfig):
    """
    FASE 0 — Verificação antes de rodar os experimentos.
    Carrega apenas os tokenizers (sem os modelos completos)
    para verificar templates e keyword finding.
    """
    from transformers import AutoTokenizer
    
    print("\n" + "="*60)
    print("FASE 0: VERIFICAÇÃO")
    print("="*60)
    
    tokenizers = {}
    for model_key, model_name in config.models.items():
        tokenizers[model_key] = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    
    # Verificar chat templates
    verify_templates(config.models, tokenizers)
    
    # Verificar keyword finding
    verify_keyword_finding(tokenizers, PROMPTS)
    
    print("\n✅ Verificação completa. Prosseguir com experimentos? (s/n)")
    response = input().strip().lower()
    if response != "s":
        print("Experimento cancelado.")
        exit(0)


def run_single_experiment(
    extractor: LogitLensExtractor,
    analyzer: LogitLensAnalyzer,
    model_key: str,
    fact: str,
    lang: str,
    formulation: str,
    config: ExperimentConfig
) -> Dict:
    """
    Roda um único experimento (1 dos 288 runs).
    """
    prompt_raw = PROMPTS[fact][lang][formulation]
    keyword = config.keyword_tokens[fact][lang]
    
    # Aplicar chat template — Questão A
    formatted_prompt, last_relevant_pos = apply_template(
        extractor.tokenizer, model_key, prompt_raw
    )
    
    # Ground truth — resposta final
    response = extractor.get_final_response(formatted_prompt)
    
    # Tokenizar para keyword finding — Questão C
    input_ids = extractor.tokenizer.encode(
        formatted_prompt,
        add_special_tokens=False
    )
    
    keyword_pos, kw_strategy = find_keyword_position_robust(
        extractor.tokenizer, input_ids, keyword, model_key
    )
    
    # Logit Lens — passando posições corretas
    logit_lens_data = extractor.extract(
        prompt=formatted_prompt,
        keyword=keyword,
        last_relevant_pos=last_relevant_pos,
        keyword_pos_override=keyword_pos  # Questão C
    )
    
    # Normalizar camadas — Questão B
    if logit_lens_data["last_token"]:
        logit_lens_data["last_token"] = normalize_layer_results(
            logit_lens_data["last_token"], model_key
        )
    if logit_lens_data["keyword_token"]:
        logit_lens_data["keyword_token"] = normalize_layer_results(
            logit_lens_data["keyword_token"], model_key
        )
    
    # Calcular métricas
    metrics_last = analyzer.compute_all_metrics(
        logit_lens_data["last_token"], lang
    )
    
    metrics_kw = None
    if logit_lens_data["keyword_token"]:
        metrics_kw = analyzer.compute_all_metrics(
            logit_lens_data["keyword_token"], lang
        )
    
    return {
        "prompt_raw":       prompt_raw,
        "prompt_formatted": formatted_prompt,
        "response":         response,
        "keyword_strategy": kw_strategy,
        "keyword_pos":      keyword_pos,
        "last_relevant_pos":last_relevant_pos,
        "n_layers":         logit_lens_data["n_layers"],
        "last_token":       logit_lens_data["last_token"],
        "keyword_token":    logit_lens_data["keyword_token"],
        "metrics_last":     metrics_last,
        "metrics_keyword":  metrics_kw
    }


def run_all_experiments():
    
    config = ExperimentConfig()
    Path(config.results_dir).mkdir(exist_ok=True)
    
    # Fase 0 — Verificação
    run_verification_phase(config)
    
    all_results = {}
    
    for model_key, model_name in config.models.items():
        
        print(f"\n{'='*60}")
        print(f"MODELO: {model_key.upper()}")
        print(f"{'='*60}")
        
        model, tokenizer = load_model(model_name, model_key)
        extractor = LogitLensExtractor(model, tokenizer, config.top_k_tokens)
        
        model_results = {}
        
        for fact in ["aviao", "telefone", "radio"]:
            model_results[fact] = {}
            analyzer = LogitLensAnalyzer(config.expected_entities[fact])
            
            for lang in ["pt", "en", "de", "it"]:
                model_results[fact][lang] = {}
                
                for formulation in ["F1", "F2", "F3"]:
                    
                    print(f"\n  [{fact}][{lang}][{formulation}]")
                    
                    result = run_single_experiment(
                        extractor, analyzer,
                        model_key, fact, lang, formulation,
                        config
                    )
                    
                    model_results[fact][lang][formulation] = result
                    
                    # Log resumido
                    m = result["metrics_last"]
                    print(f"  Resposta:  {result['response'][:80]}...")
                    print(f"  KW strategy: {result['keyword_strategy']}")
                    print(f"  PCC: camada {m['pcc']} ({m.get('pcc_norm', 'N/A'):.2f})")
                    print(f"  IV:  {m['iv']:.4f}")
        
        all_results[model_key] = model_results
        
        # Salvar resultados por modelo
        output_path = os.path.join(
            config.results_dir, f"{model_key}_results.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, ensure_ascii=False, indent=2)
        
        del model, tokenizer, extractor
        torch.cuda.empty_cache()
    
    # Salvar tabela comparativa consolidada
    comparison_df = build_comparison_table(all_results, config)
    comparison_df.to_csv(
        os.path.join(config.results_dir, "metrics_comparison.csv"),
        index=False
    )
    
    print(f"\n{'='*60}")
    print("EXPERIMENTO COMPLETO")
    print(f"Resultados em: {config.results_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_experiments()