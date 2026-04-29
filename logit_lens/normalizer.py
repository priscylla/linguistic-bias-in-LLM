"""
logit_lens/normalizer.py
Normalização de índices de camada para escala 0-1,
permitindo comparação entre modelos com diferentes
números de camadas.

Problema:
    LLaMA 3.1 8B → 32 camadas
    Mistral 7B   → 32 camadas
    Qwen 2.5 7B  → 28 camadas
    Gemma 2 9B   → 42 camadas

    PCC na camada 18 do LLaMA (18/32 = 56%) é comparável
    a PCC na camada 25 do Gemma (25/42 = 59%), mas os
    índices absolutos são enganosos.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# Número de camadas por modelo — fixo e documentado
MODEL_LAYERS = {
    "llama":   32,
    "mistral": 32,
    "qwen":    28,
    "gemma":   42,
}

# Stake cultural: idiomas com inventor local relevante para cada fato
STAKE_MAP = {
    "aviao":    ["pt"],           # Santos Dumont
    "telefone": ["it"],           # Meucci
    "radio":    ["it"],           # Marconi
}


def normalize_layer(
    layer_idx: int,
    model_key: str
) -> float:
    """
    Normaliza o índice de camada para escala 0-1.

    Usa normalização min-max com:
        min = 1 (primeira camada transformer)
        max = n_layers (última camada)

    Args:
        layer_idx: índice absoluto da camada (1-based)
        model_key: chave do modelo

    Returns:
        Valor normalizado entre 0.0 e 1.0
    """
    n_layers = MODEL_LAYERS[model_key]
    return round((layer_idx - 1) / (n_layers - 1), 4)


def normalize_layer_results(
    layer_results: List[Dict],
    model_key: str
) -> List[Dict]:
    """
    Adiciona campo 'layer_norm' (0-1) a cada entrada
    da lista de resultados por camada.

    Modifica a lista in-place E retorna para encadeamento.

    Args:
        layer_results: lista de dicts com campo 'layer'
        model_key:     chave do modelo

    Returns:
        Mesma lista com campo 'layer_norm' adicionado.
    """
    for entry in layer_results:
        entry["layer_norm"] = normalize_layer(entry["layer"], model_key)
    return layer_results


def normalize_metric(
    metric_value: Optional[int],
    model_key: str
) -> Optional[float]:
    """
    Normaliza uma métrica que representa índice de camada
    (PCC, ED) para escala 0-1.

    Args:
        metric_value: valor absoluto (índice de camada) ou None
        model_key:    chave do modelo

    Returns:
        Valor normalizado, ou None se metric_value for None.
    """
    if metric_value is None:
        return None
    return normalize_layer(metric_value, model_key)


def align_layers_across_models(
    results_by_model: Dict[str, List[Dict]],
    value_key: str = "entity_prob",
    n_points: int = 100
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Interpola os resultados de diferentes modelos para uma
    grade comum de n_points pontos entre 0 e 1.

    Permite comparação direta e plotagem sobreposta entre
    modelos com números diferentes de camadas.

    Args:
        results_by_model: dict {model_key: layer_results}
                          onde layer_results tem campo 'layer_norm'
                          e o campo indicado por value_key
        value_key:        nome do campo numérico a interpolar
        n_points:         número de pontos na grade comum

    Returns:
        Tupla (common_grid, aligned_dict) onde:
        - common_grid: array de n_points valores entre 0 e 1
        - aligned_dict: dict {model_key: array interpolado}
    """
    common_grid = np.linspace(0, 1, n_points)
    aligned = {}

    for model_key, layer_results in results_by_model.items():
        x = np.array([r["layer_norm"] for r in layer_results])
        y = np.array([r.get(value_key, 0.0) for r in layer_results])

        # Interpolação linear para a grade comum
        y_interp = np.interp(common_grid, x, y)
        aligned[model_key] = y_interp

    return common_grid, aligned


def has_cultural_stake(fact: str, lang: str) -> bool:
    """
    Indica se o idioma tem stake cultural direto no fato.

    Usado para a análise de Neutralidade Condicional:
    idiomas com stake → espera-se convergência mais cedo (PCC menor)
    e maior intensidade (IV maior).

    Args:
        fact: "aviao", "telefone" ou "radio"
        lang: "pt", "en", "de" ou "it"

    Returns:
        True se o idioma tem inventor local relevante para esse fato.
    """
    return lang in STAKE_MAP.get(fact, [])


def build_comparison_table(
    all_results: Dict,
    config
) -> pd.DataFrame:
    """
    Constrói tabela comparativa de métricas normalizadas
    entre todos os modelos, fatos e idiomas.

    Estrutura esperada de all_results:
        {model_key: {fact: {lang: {formulation: {
            metrics_last: {pcc, pcc_norm, ed, ed_norm, iv}
        }}}}}

    Args:
        all_results: dict completo de resultados do experimento
        config:      ExperimentConfig (não usado diretamente,
                     mantido para compatibilidade de interface)

    Returns:
        DataFrame com colunas:
        model, fact, language, formulation,
        pcc_abs, pcc_norm, ed_abs, ed_norm, iv, has_stake
    """
    rows = []

    for model_key, model_data in all_results.items():
        for fact, fact_data in model_data.items():
            for lang, lang_data in fact_data.items():
                for formulation, run_data in lang_data.items():

                    metrics = run_data.get("metrics_last", {})

                    pcc_abs = metrics.get("pcc")
                    ed_abs  = metrics.get("ed")
                    iv      = metrics.get("iv", 0.0)

                    rows.append({
                        "model":       model_key,
                        "fact":        fact,
                        "language":    lang,
                        "formulation": formulation,
                        "pcc_abs":     pcc_abs,
                        "pcc_norm":    normalize_metric(pcc_abs, model_key),
                        "ed_abs":      ed_abs,
                        "ed_norm":     normalize_metric(ed_abs, model_key),
                        "iv":          iv,
                        "has_stake":   has_cultural_stake(fact, lang),
                    })

    return pd.DataFrame(rows)
