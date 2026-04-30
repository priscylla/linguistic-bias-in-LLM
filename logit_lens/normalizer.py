"""
logit_lens/normalizer.py
Normalizacao de indices de camada para escala 0-1,
permitindo comparacao entre modelos com diferentes
numeros de camadas.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


MODEL_LAYERS = {
    "llama":   32,
    "mistral": 32,
    "qwen":    28,
    "gemma":   42,
}

# Idiomas com inventor local relevante para cada fato
# aviao:    PT tem Santos Dumont
# telefone: IT tem Meucci
STAKE_MAP = {
    "aviao":    ["pt"],
    "telefone": ["it"],
}


def normalize_layer(layer_idx: int, model_key: str) -> float:
    n_layers = MODEL_LAYERS[model_key]
    return round((layer_idx - 1) / (n_layers - 1), 4)


def normalize_layer_results(
    layer_results: List[Dict],
    model_key: str
) -> List[Dict]:
    for entry in layer_results:
        entry["layer_norm"] = normalize_layer(entry["layer"], model_key)
    return layer_results


def normalize_metric(
    metric_value: Optional[int],
    model_key: str
) -> Optional[float]:
    if metric_value is None:
        return None
    return normalize_layer(metric_value, model_key)


def has_cultural_stake(fact: str, lang: str) -> bool:
    return lang in STAKE_MAP.get(fact, [])


def build_comparison_table(
    all_results: Dict,
    config
) -> pd.DataFrame:
    """
    Constroi tabela comparativa de metricas normalizadas
    entre todos os modelos, fatos e idiomas.

    Compativel com resultados antigos (campo 'iv') e novos
    (campos 'iv_local', 'iv_competing', 'commitment_final').
    """
    rows = []

    for model_key, model_data in all_results.items():
        for fact, fact_data in model_data.items():
            for lang, lang_data in fact_data.items():
                for formulation, run_data in lang_data.items():

                    # Pular runs com erro
                    if "error" in run_data and "metrics_last" not in run_data:
                        continue

                    metrics = run_data.get("metrics_last", {})

                    # Compatibilidade retroativa:
                    # resultados antigos usam 'iv'
                    # resultados novos usam 'iv_local'
                    iv_local = metrics.get(
                        "iv_local",
                        metrics.get("iv", 0.0)
                    )
                    iv_competing    = metrics.get("iv_competing", 0.0)
                    commitment_final = metrics.get(
                        "commitment_final",
                        iv_local   # fallback: sem concorrente conhecido
                    )

                    pcc = metrics.get("pcc")
                    ed  = metrics.get("ed")

                    rows.append({
                        "model":             model_key,
                        "fact":              fact,
                        "language":          lang,
                        "formulation":       formulation,

                        # Convergencia
                        "pcc_abs":           pcc,
                        "pcc_norm":          normalize_metric(pcc, model_key),
                        "ed_abs":            ed,
                        "ed_norm":           normalize_metric(ed, model_key),

                        # Probabilidades na ultima camada
                        "iv_local":          round(iv_local, 6),
                        "iv_competing":      round(iv_competing, 6),

                        # Commitment Score -- metrica central
                        # CS > 0: favorece local
                        # CS ~= 0: ambiguo
                        # CS < 0: favorece concorrente
                        "commitment_final":  round(commitment_final, 6),

                        # Stake cultural
                        "has_stake":         has_cultural_stake(fact, lang),
                    })

    return pd.DataFrame(rows)