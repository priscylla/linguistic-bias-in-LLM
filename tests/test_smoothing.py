"""
tests/test_smoothing.py
Fase 0, correção 6: suavização centrada única, sem artefato de borda.
"""

import numpy as np
import pytest

from plot_utils import smooth_curve, smoothing_suffix


def test_constant_series_has_no_edge_drop():
    y = smooth_curve([0.4] * 10, 3)
    assert np.allclose(y, 0.4)          # np.convolve(mode="same") dava 0.267 nas bordas


def test_edges_average_available_neighbors_only():
    y = smooth_curve([1.0, 2.0, 3.0, 4.0], 3)
    assert y[0] == pytest.approx(1.5)   # (1+2)/2
    assert y[1] == pytest.approx(2.0)   # (1+2+3)/3
    assert y[-1] == pytest.approx(3.5)  # (3+4)/2


def test_centered_peak_stays_in_place():
    raw = [0.0, 0.1, 0.4, 0.9, 0.4, 0.1, 0.0]
    y = smooth_curve(raw, 3)
    assert int(np.argmax(y)) == 3       # janela antiga (não centrada) deslocava o pico
    assert y[2] == pytest.approx(y[4])  # simétrico
    assert y[1] == pytest.approx(y[5])


def test_window_one_is_raw():
    raw = [0.1, 0.5, 0.2]
    y = smooth_curve(raw, 1)
    assert np.array_equal(y, np.array(raw))


@pytest.mark.parametrize("w", [0, 2, 4, -1, 3.0])
def test_invalid_window_raises(w):
    with pytest.raises(ValueError):
        smooth_curve([1, 2, 3], w)


def test_same_length_short_series():
    assert len(smooth_curve([0.2, 0.3], 3)) == 2


def test_suffix():
    assert smoothing_suffix(1) == "_raw"
    assert smoothing_suffix(3) == "_w3"


def test_config_windows():
    from config.experiment_config import ExperimentConfig
    assert ExperimentConfig().smoothing_windows == [3, 1]
