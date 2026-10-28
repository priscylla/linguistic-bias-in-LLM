"""
logit_lens/normalizer.py
Normalizacao de indices de camada para escala 0-1,
permitindo comparacao entre modelos com diferentes
numeros de camadas.

O numero de camadas vem do proprio modelo (n_layers do forward),
nunca de uma tabela fixa no codigo.
"""

from typing import Dict, List, Optional


def normalize_layer(layer_idx: int, n_layers: int) -> float:
    """Camada 1-based -> profundidade relativa em [0, 1]."""
    return round((layer_idx - 1) / (n_layers - 1), 4)


def normalize_layer_results(
    layer_results: List[Dict],
    n_layers: int
) -> List[Dict]:
    for entry in layer_results:
        entry["layer_norm"] = normalize_layer(entry["layer"], n_layers)
    return layer_results


def normalize_metric(
    metric_value: Optional[int],
    n_layers: int
) -> Optional[float]:
    if metric_value is None:
        return None
    return normalize_layer(metric_value, n_layers)
