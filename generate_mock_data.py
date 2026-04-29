import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.experiment_config import ExperimentConfig

np.random.seed(42)

# ── Número de camadas por modelo ──
MODEL_LAYERS = {
    "llama":   32,
    "mistral": 32,
    "qwen":    28,
    "gemma":   42,
}

# ── Bias base por idioma/fato (simula stake cultural) ──
BIAS_MAP = {
    ("aviao",    "pt"): 0.48,   # alto stake — Santos Dumont
    ("aviao",    "en"): 0.42,   # Wright Brothers dominante
    ("aviao",    "de"): 0.28,   # baixo stake
    ("aviao",    "it"): 0.25,   # baixo stake

    ("telefone", "it"): 0.45,   # alto stake — Meucci
    ("telefone", "en"): 0.40,   # Bell dominante
    ("telefone", "pt"): 0.35,
    ("telefone", "de"): 0.27,   # baixo stake

    ("radio",    "it"): 0.46,   # alto stake — Marconi
    ("radio",    "pt"): 0.36,
    ("radio",    "en"): 0.38,   # Tesla dominante
    ("radio",    "de"): 0.29,   # baixo stake
}

# ── Multiplicador por modelo (simula diferenças de corpus) ──
MODEL_BIAS_MULT = {
    "llama":   1.00,   # baseline
    "mistral": 0.88,   # ligeiramente menor
    "qwen":    0.72,   # corpus mais asiático — menos viés ocidental
    "gemma":   0.92,
}

# ── Multiplicador por formulação ──
FORMULATION_MULT = {
    "F1": 1.00,
    "F2": 1.06,
    "F3": 1.18,   # F3 amplifica o viés
}

# ── Entidades por idioma/fato ──
LOCAL_ENTITIES = {
    ("aviao",    "pt"): ["Santos", "Dumont", "brasileiro", "14-Bis"],
    ("aviao",    "en"): ["Wright", "Brothers", "Orville", "Wilbur"],
    ("aviao",    "de"): ["Wright", "Brothers", "Gebrüder", "Orville"],
    ("aviao",    "it"): ["Wright", "Brothers", "fratelli", "americani"],

    ("telefone", "pt"): ["Bell", "Graham", "Alexander"],
    ("telefone", "en"): ["Bell", "Graham", "Alexander"],
    ("telefone", "de"): ["Bell", "Graham", "Telefon"],
    ("telefone", "it"): ["Meucci", "Antonio", "italiano"],

    ("radio",    "pt"): ["Marconi", "Guglielmo", "Nobel"],
    ("radio",    "en"): ["Tesla", "Nikola", "Serbian"],
    ("radio",    "de"): ["Marconi", "Tesla", "Hertz"],
    ("radio",    "it"): ["Marconi", "Guglielmo", "italiano"],
}

GENERIC_TOKENS = [
    "the", "a", "inventor", "was", "who", "person",
    "man", "first", "created", "made", "built",
    "flight", "device", "signal", "wave", "patent",
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_layer_results(
    n_layers: int,
    fact: str,
    lang: str,
    bias_strength: float,
    top_k: int = 10
) -> list:
    """
    Gera resultados por camada com comportamento realista:
    - Camadas iniciais (0-30%): tokens genéricos dominam
    - Camadas médias (30-60%): entidades começam a emergir
    - Camadas finais (60-100%): entidade cultural consolida
    """
    entities = LOCAL_ENTITIES.get((fact, lang), ["Wright", "inventor"])
    layer_results = []

    for layer_idx in range(1, n_layers + 1):

        progress = (layer_idx - 1) / (n_layers - 1)

        # Curva sigmoide com ponto de inflexão em ~40% da profundidade
        # Modela a emergência gradual do conhecimento cultural
        entity_prob = bias_strength * sigmoid(12 * (progress - 0.40))
        entity_prob += np.random.normal(0, 0.015)
        entity_prob = float(np.clip(entity_prob, 0.01, 0.95))

        # Construir top-K tokens
        top_tokens = []
        top_probs  = []

        # Entidade principal
        top_tokens.append(entities[0])
        top_probs.append(entity_prob)

        # Entidades secundárias
        remaining = 1.0 - entity_prob
        for i, entity in enumerate(entities[1:min(3, len(entities))], 1):
            prob = remaining * (0.35 / i)
            prob += np.random.normal(0, 0.01)
            prob  = float(np.clip(prob, 0.005, remaining * 0.5))
            top_tokens.append(entity)
            top_probs.append(prob)
            remaining -= prob

        # Tokens genéricos para completar top-K
        generic_sample = np.random.choice(
            GENERIC_TOKENS,
            size=top_k - len(top_tokens),
            replace=False
        )
        for token in generic_sample:
            prob = remaining / (top_k - len(top_tokens))
            prob += np.random.normal(0, 0.005)
            prob  = float(np.clip(prob, 0.001, 0.15))
            top_tokens.append(token)
            top_probs.append(prob)

        # Normalizar
        total = sum(top_probs)
        top_probs = [p / total for p in top_probs]

        layer_results.append({
            "layer":      layer_idx,
            "layer_norm": round(progress, 4),
            "top_tokens": top_tokens,
            "top_probs":  [round(p, 6) for p in top_probs],
        })

    return layer_results


def compute_metrics(layer_results: list, threshold: float = 0.10) -> dict:
    """Calcula PCC, ED e IV a partir dos layer_results."""

    n = len(layer_results)

    # PCC — primeira camada estável acima do threshold
    pcc = None
    for i, layer in enumerate(layer_results):
        if layer["top_probs"][0] >= threshold:
            # Verificar estabilidade nas próximas 3 camadas
            stable = all(
                layer_results[j]["top_probs"][0] >= threshold * 0.6
                for j in range(i + 1, min(i + 4, n))
            )
            if stable:
                pcc = layer["layer"]
                break

    # ED — última camada onde top-1 muda
    ed = None
    for i in range(n - 4):
        top1 = layer_results[i]["top_tokens"][0]
        if all(
            layer_results[j]["top_tokens"][0] == top1
            for j in range(i + 1, i + 5)
        ):
            ed = layer_results[i]["layer"]
            break

    # IV — probabilidade da entidade na última camada
    iv = layer_results[-1]["top_probs"][0]

    n_layers = layer_results[-1]["layer"]
    pcc_norm = round((pcc - 1) / (n_layers - 1), 4) if pcc else None
    ed_norm  = round((ed  - 1) / (n_layers - 1), 4) if ed  else None

    return {
        "pcc":      pcc,
        "pcc_norm": pcc_norm,
        "ed":       ed,
        "ed_norm":  ed_norm,
        "iv":       round(iv, 6),
    }


def has_cultural_stake(fact: str, lang: str) -> bool:
    stake_map = {
        "aviao":    ["pt"],
        "telefone": ["it"],
        "radio":    ["it"],
    }
    return lang in stake_map.get(fact, [])


def generate_all_mock_results() -> dict:

    config  = ExperimentConfig()
    facts   = ["aviao", "telefone", "radio"]
    langs   = ["pt", "en", "de", "it"]
    forms   = ["F1", "F2", "F3"]
    results = {}

    for model_key in ["llama", "mistral", "qwen", "gemma"]:
        n_layers   = MODEL_LAYERS[model_key]
        bias_mult  = MODEL_BIAS_MULT[model_key]
        results[model_key] = {}

        for fact in facts:
            results[model_key][fact] = {}

            for lang in langs:
                results[model_key][fact][lang] = {}
                base_bias = BIAS_MAP.get((fact, lang), 0.30) * bias_mult

                for form in forms:
                    bias = base_bias * FORMULATION_MULT[form]

                    last_token    = generate_layer_results(n_layers, fact, lang, bias)
                    keyword_token = generate_layer_results(n_layers, fact, lang, bias * 0.75)

                    results[model_key][fact][lang][form] = {
                        "prompt_raw":       f"[mock] {fact}/{lang}/{form}",
                        "response":         f"[mock response] {fact} {lang} {form}",
                        "keyword_strategy": "exact",
                        "keyword_pos":      12,
                        "n_layers":         n_layers,
                        "last_token":       last_token,
                        "keyword_token":    keyword_token,
                        "metrics_last":     compute_metrics(last_token),
                        "metrics_keyword":  compute_metrics(keyword_token),
                    }

    return results


def build_metrics_df(results: dict) -> pd.DataFrame:

    rows = []

    stake_map = {
        "aviao":    ["pt"],
        "telefone": ["it"],
        "radio":    ["it"],
    }

    for model_key, model_data in results.items():
        for fact, fact_data in model_data.items():
            for lang, lang_data in fact_data.items():
                for form, run in lang_data.items():
                    m = run.get("metrics_last", {})
                    rows.append({
                        "model":       model_key,
                        "fact":        fact,
                        "language":    lang,
                        "formulation": form,
                        "pcc_abs":     m.get("pcc"),
                        "pcc_norm":    m.get("pcc_norm"),
                        "ed_abs":      m.get("ed"),
                        "ed_norm":     m.get("ed_norm"),
                        "iv":          m.get("iv", 0.0),
                        "has_stake":   lang in stake_map.get(fact, []),
                    })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    print("=" * 50)
    print("GERANDO DADOS SINTÉTICOS")
    print("=" * 50)

    Path("results").mkdir(exist_ok=True)

    results = generate_all_mock_results()

    # Salvar JSON por modelo
    for model_key in ["llama", "mistral", "qwen", "gemma"]:
        path = f"results/{model_key}_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results[model_key], f, ensure_ascii=False, indent=2)
        print(f"✅ {path}")

    # Salvar CSV consolidado
    df = build_metrics_df(results)
    df.to_csv("results/metrics_comparison.csv", index=False)
    print("✅ results/metrics_comparison.csv")

    # Estatísticas rápidas
    print("\n── Estatísticas de IV por idioma/fato (F1) ──")
    summary = df[df["formulation"] == "F1"].groupby(
        ["fact", "language"]
    )["iv"].mean().round(3).unstack()
    print(summary.to_string())

    print("\n── PCC normalizado médio por modelo (F1) ──")
    pcc_summary = df[df["formulation"] == "F1"].groupby(
        "model"
    )["pcc_norm"].mean().round(3)
    print(pcc_summary.to_string())

    print(f"\n✅ Total de runs gerados: {len(df)}")
    print("   Rodar visualizações: python generate_visualizations.py")
