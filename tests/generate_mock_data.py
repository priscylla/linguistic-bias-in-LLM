import json
import numpy as np
from pathlib import Path

np.random.seed(42)

def generate_mock_layer_results(
    n_layers: int,
    lang: str,
    fact: str,
    model_key: str,
    bias_strength: float = 0.3
) -> list:
    """
    Gera dados sintéticos de Logit Lens por camada.
    Simula o comportamento esperado:
    - Camadas iniciais: tokens genéricos
    - Camadas médias:  entidades começam a emergir
    - Camadas finais:  entidade cultural dominante
    """
    
    # Entidades por idioma e fato
    local_entities = {
        ("aviao", "pt"): ["Santos", "Dumont", "brasileiro"],
        ("aviao", "en"): ["Wright", "Brothers", "Orville"],
        ("aviao", "de"): ["Wright", "Brothers", "Gebrüder"],
        ("aviao", "it"): ["Wright", "Brothers", "fratelli"],
        ("telefone", "it"): ["Meucci", "Antonio", "italiano"],
        ("telefone", "pt"): ["Bell", "Graham", "Alexander"],
        ("telefone", "en"): ["Bell", "Graham", "Alexander"],
        ("telefone", "de"): ["Bell", "Graham", "Telefon"],
        ("radio", "it"):    ["Marconi", "Guglielmo", "italiano"],
        ("radio", "pt"):    ["Marconi", "Tesla", "Nikola"],
        ("radio", "en"):    ["Tesla", "Nikola", "Marconi"],
        ("radio", "de"):    ["Marconi", "Tesla", "Hertz"],
    }
    
    generic_tokens = [
        "the", "a", "inventor", "was", "who",
        "person", "man", "first", "created", "made"
    ]
    
    entities = local_entities.get((fact, lang), ["Wright", "inventor"])
    
    layer_results = []
    
    for layer_idx in range(1, n_layers + 1):
        
        # Progresso normalizado (0→1)
        progress = (layer_idx - 1) / (n_layers - 1)
        
        # Probabilidade da entidade local cresce com a profundidade
        # com curva sigmoide — simula convergência gradual
        entity_prob = bias_strength / (
            1 + np.exp(-10 * (progress - 0.5))
        )
        entity_prob += np.random.normal(0, 0.02)
        entity_prob = np.clip(entity_prob, 0, 1)
        
        # Construir top-10 tokens
        top_tokens = []
        top_probs  = []
        
        # Token da entidade principal
        top_tokens.append(entities[0])
        top_probs.append(entity_prob)
        
        # Tokens secundários da entidade
        remaining = 1 - entity_prob
        for i, entity in enumerate(entities[1:], 1):
            prob = remaining * (0.4 / i) + np.random.normal(0, 0.01)
            prob = np.clip(prob, 0, remaining)
            top_tokens.append(entity)
            top_probs.append(prob)
            remaining -= prob
        
        # Preencher com tokens genéricos
        while len(top_tokens) < 10:
            token = np.random.choice(generic_tokens)
            prob  = remaining / (10 - len(top_tokens))
            prob += np.random.normal(0, 0.005)
            prob  = np.clip(prob, 0, 0.1)
            top_tokens.append(token)
            top_probs.append(prob)
        
        # Normalizar para somar ~1
        total = sum(top_probs)
        top_probs = [p / total for p in top_probs]
        
        layer_results.append({
            "layer":      layer_idx,
            "layer_norm": progress,
            "top_tokens": top_tokens,
            "top_probs":  top_probs
        })
    
    return layer_results


def generate_mock_results() -> dict:
    """
    Gera estrutura completa de resultados sintéticos
    compatível com o pipeline de visualização.
    """
    
    models = {
        "llama":   {"n_layers": 32, "bias_mult": 1.0},
        "mistral": {"n_layers": 32, "bias_mult": 0.85},
        "qwen":    {"n_layers": 28, "bias_mult": 0.70},
        "gemma":   {"n_layers": 42, "bias_mult": 0.90},
    }
    
    facts     = ["aviao", "telefone", "radio"]
    languages = ["pt", "en", "de", "it"]
    forms     = ["F1", "F2", "F3"]
    
    # Bias por idioma/fato — simula stake cultural
    bias_map = {
        ("aviao",    "pt"): 0.45,  # alto stake
        ("aviao",    "en"): 0.40,
        ("aviao",    "de"): 0.30,  # baixo stake
        ("aviao",    "it"): 0.28,  # baixo stake
        ("telefone", "it"): 0.42,  # alto stake
        ("telefone", "pt"): 0.32,
        ("telefone", "en"): 0.38,
        ("telefone", "de"): 0.25,
        ("radio",    "it"): 0.44,  # alto stake
        ("radio",    "pt"): 0.33,
        ("radio",    "en"): 0.35,
        ("radio",    "de"): 0.27,
    }
    
    # F3 amplifica o bias em idiomas com stake
    formulation_mult = {
        "F1": 1.00,
        "F2": 1.05,
        "F3": 1.15,  # F3 mais forte
    }
    
    all_results = {}
    
    for model_key, model_info in models.items():
        all_results[model_key] = {}
        n_layers = model_info["n_layers"]
        
        for fact in facts:
            all_results[model_key][fact] = {}
            
            for lang in languages:
                all_results[model_key][fact][lang] = {}
                
                base_bias = bias_map.get((fact, lang), 0.30)
                base_bias *= model_info["bias_mult"]
                
                for form in forms:
                    
                    bias = base_bias * formulation_mult[form]
                    
                    last_token = generate_mock_layer_results(
                        n_layers, lang, fact,
                        model_key, bias
                    )
                    
                    keyword_token = generate_mock_layer_results(
                        n_layers, lang, fact,
                        model_key, bias * 0.7
                    )
                    
                    # Métricas sintéticas
                    # PCC: camada onde prob > 0.10 pela primeira vez
                    pcc = None
                    for i, layer in enumerate(last_token):
                        if layer["top_probs"][0] >= 0.10:
                            pcc = layer["layer"]
                            break
                    
                    pcc_norm = (pcc - 1) / (n_layers - 1) if pcc else None
                    iv = last_token[-1]["top_probs"][0]
                    
                    all_results[model_key][fact][lang][form] = {
                        "prompt_raw":    f"[mock] {fact} {lang} {form}",
                        "response":      f"[mock response for {fact}/{lang}/{form}]",
                        "keyword_strategy": "exact",
                        "keyword_pos":   10,
                        "n_layers":      n_layers,
                        "last_token":    last_token,
                        "keyword_token": keyword_token,
                        "metrics_last": {
                            "pcc":      pcc,
                            "pcc_norm": pcc_norm,
                            "ed":       pcc + 3 if pcc else None,
                            "iv":       iv
                        },
                        "metrics_keyword": {
                            "pcc":      pcc,
                            "pcc_norm": pcc_norm,
                            "ed":       None,
                            "iv":       iv * 0.7
                        }
                    }
    
    return all_results


if __name__ == "__main__":
    
    Path("results").mkdir(exist_ok=True)
    
    print("Gerando dados sintéticos...")
    results = generate_mock_results()
    
    # Salvar por modelo (mesma estrutura dos dados reais)
    for model_key in ["llama", "mistral", "qwen", "gemma"]:
        path = f"results/{model_key}_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {model_key: results[model_key]}
                if model_key in results
                else results,
                f, ensure_ascii=False, indent=2
            )
        print(f"✅ {path}")
    
    # Gerar metrics_comparison.csv
    import pandas as pd
    from logit_lens.normalizer import build_comparison_table
    from config.experiment_config import ExperimentConfig
    
    config = ExperimentConfig()
    df = build_comparison_table(results, config)
    df.to_csv("results/metrics_comparison.csv", index=False)
    print("✅ results/metrics_comparison.csv")
    
    print("\n✅ Dados sintéticos prontos!")
    print("   Rodar: python run_visualization.py")