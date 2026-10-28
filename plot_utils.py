"""
plot_utils.py
Suavização única usada por todas as figuras.

Média móvel centrada com min_periods=1: nas bordas a média usa só os
vizinhos que existem (sem preenchimento com zeros, que puxava as
últimas camadas para 0 e podia fabricar um "colapso antes da saída").

A suavização é só para os gráficos. Todas as métricas usam os valores
brutos.
"""

from typing import Sequence

import numpy as np
import pandas as pd


def smooth_curve(values: Sequence[float], window: int) -> np.ndarray:
    """
    Média móvel centrada.

    Args:
        values: curva por camada (valores brutos)
        window: tamanho da janela, ímpar (1 = sem suavização)

    Returns:
        np.ndarray do mesmo tamanho de `values`
    """
    if not isinstance(window, int) or window < 1 or window % 2 == 0:
        raise ValueError(
            f"Janela de suavização {window!r} inválida: use um inteiro "
            f"ímpar >= 1 (uma janela par não pode ser centrada)"
        )
    arr = np.asarray(values, dtype=float)
    if window == 1:
        return arr.copy()
    return (
        pd.Series(arr)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def smoothing_suffix(window: int) -> str:
    """Sufixo do arquivo da figura: '_raw' sem suavização, '_w3' etc."""
    return "_raw" if window == 1 else f"_w{window}"
