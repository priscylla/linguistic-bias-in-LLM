import numpy as np
from typing import Dict, List, Optional, Tuple


# Número de camadas por modelo — fixo e documentado
MODEL_LAYERS = {
    "llama":   32,
    "mistral": 32,
    "qwen":    28,
    "gemma":   42
}


def normalize_layer(
    layer_idx: int,
    model_key: str,
    n_buckets: int = 10
) -> float:
    """
    Normaliza o índice de camada para escala 0-1.
    
    Args:
        layer_idx: índice absoluto da camada (1-based)
        model_key: chave do modelo
        n_buckets: número de buckets para discretização
                   (usado nas visualizações)
    
    Returns:
        Valor normalizado entre 0.0 e 1.0
    """
    n_layers = MODEL_LAYERS[model_key]
    return (layer_idx - 1) / (n_layers - 1)


def normalize_layer_results(
    layer_results: List[Dict],
    model_key: str
) -> List[Dict]:
    """
    Adiciona campo 'layer_norm' (0-1) a cada entrada
    da lista de resultados por camada.
    
    Args:
        layer_results: lista de dicts com campo 'layer'
        model_key:     chave do modelo
    
    Returns:
        Mesma lista com campo 'layer_norm' adicionado
    """
    n_layers = MODEL_LAYERS[model_key]
    
    for entry in layer_results:
        entry["layer_norm"] = normalize_layer(
            entry["layer"], model_key
        )
    
    return layer_results


def normalize_metric(
    metric_value: Optional[int],
    model_key: str
) -> Optional[float]:
    """
    Normaliza uma métrica que representa um índice de camada
    (PCC, ED) para escala 0-1.
    
    Args:
        metric_value: valor da métrica em índice absoluto
        model_key:    chave do modelo
    
    Returns:
        Valor normalizado, ou None se métrica for None
    """
    if metric_value is None:
        return None
    return normalize_layer(metric_value, model_key)


def align_layers_across_models(
    results_by_model: Dict[str, List[Dict]],
    n_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Interpola os resultados de diferentes modelos para
    uma grade comum de n_points pontos entre 0 e 1.
    
    Isso permite comparação direta entre modelos com
    números diferentes de camadas.
    
    Args:
        results_by_model: dict {model_key: layer_results}
                          onde layer_results tem campo 'layer_norm'
                          e algum valor numérico de interesse
        n_points:         número de pontos na grade comum
    
    Returns:
        dict {model_key: array interpolado de n_points valores}
    """
    common_grid = np.linspace(0, 1, n_points)
    aligned = {}
    
    for model_key, layer_results in results_by_model.items():
        # Extrair x (layer_norm) e y (valor de interesse)
        # Aqui usando IV como exemplo — adaptar conforme necessidade
        x = np.array([r["layer_norm"] for r in layer_results])
        y = np.array([r.get("entity_prob", 0.0) for r in layer_results])
        
        # Interpolação linear para a grade comum
        y_interp = np.interp(common_grid, x, y)
        aligned[model_key] = y_interp
    
    return common_grid, aligned


def compute_pcc_normalized(
    layer_results: List[Dict],
    model_key: str,
    language: str,
    expected_entities: Dict,
    threshold: float = 0.10
) -> Tuple[Optional[int], Optional[float]]:
    """
    Calcula PCC e retorna tanto o valor absoluto
    quanto o normalizado.
    
    Returns:
        Tupla (pcc_absoluto, pcc_normalizado)
    """
    from logit_lens.analyzer import LogitLensAnalyzer
    
    analyzer = LogitLensAnalyzer(expected_entities)
    pcc_abs = analyzer.compute_pcc(layer_results, language, threshold)
    pcc_norm = normalize_metric(pcc_abs, model_key)
    
    return pcc_abs, pcc_norm


def build_comparison_table(
    all_results: Dict,
    config
) -> "pd.DataFrame":
    """
    Constrói tabela comparativa de métricas normalizadas
    entre todos os modelos, fatos e idiomas.
    
    Estrutura do all_results:
        {model_key: {fact: {lang: {formulation: {metrics_last: {...}}}}}}
    
    Returns:
        DataFrame pandas com colunas:
        model, fact, lang, formulation,
        pcc_abs, pcc_norm, ed_abs, ed_norm, iv
    """
    import pandas as pd
    
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
                        
                        # Stake cultural — usado nas visualizações
                        "has_stake":   _has_cultural_stake(fact, lang)
                    })
    
    return pd.DataFrame(rows)


def _has_cultural_stake(fact: str, lang: str) -> bool:
    """
    Indica se o idioma tem stake cultural direto no fato.
    Usada para a análise de Neutralidade Condicional.
    """
    stake_map = {
        "aviao":    ["pt"],
        "telefone": ["it"],
        "radio":    ["it"]
    }
    return lang in stake_map.get(fact, [])