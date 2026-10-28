"""
tests/test_figures_smoke.py
As figuras de curva saem em duas versões (_w3 e _raw) e o heatmap
colore por token ID, com dados sintéticos no formato novo.
"""

import json
import sys

import numpy as np

import generate_paper_figures as gpf
import generate_visualizations as gv


def _run(n_layers, seed):
    rng = np.random.default_rng(seed)
    p_loc  = rng.uniform(0, 0.5, n_layers)
    p_comp = rng.uniform(0, 0.5, n_layers)
    layers = [{
        "layer": i + 1, "layer_norm": i / (n_layers - 1),
        "top_tokens": [" Bell", " the", " Me", " a", " il"],
        "top_ids":    [10, 2, 20, 3, 4],
        "top_probs":  [0.3, 0.2, 0.1, 0.05, 0.05],
        "entity_probs": {"local": float(a), "competing": float(b)},
    } for i, (a, b) in enumerate(zip(p_loc, p_comp))]
    cs = list(p_loc - p_comp)
    return {
        "prompt": "p", "response": "Alexander Graham Bell",
        "last_token": layers,
        "entity_tokens": {"sets": {"local": {"ids": [10]},
                                   "competing": {"ids": [20]}}},
        "metrics_last": {
            "commitment_curve": cs, "commitment_final": cs[-1],
            "cs_at_peak": max(cs), "iv_local_peak": float(p_loc.max()),
            "iv_local_final": float(p_loc[-1]),
        },
    }


def _data():
    data = {}
    for k, model in enumerate(["llama", "mistral", "qwen", "gemma"]):
        n = gv.MODEL_LAYERS[model]
        data[model] = {fact: {lang: {"F1": _run(n, k * 100 + j)}
                              for j, lang in enumerate(["pt", "en", "de", "it"])}
                       for fact in ["aviao", "telefone"]}
    return data


def test_generate_visualizations_curve_figures(tmp_path):
    data = _data()
    for w in [3, 1]:
        sfx = gv.smoothing_suffix(w)
        gv.fig1_divergence_map(data, str(tmp_path / f"fig1{sfx}.png"), window=w)
        gv.fig5_convergence(data, "aviao", str(tmp_path / f"fig5{sfx}.png"),
                            use_absolute=True, window=w)
    gv.fig3_logit_lens_heatmap(data, "telefone", str(tmp_path / "fig3.png"))
    gv.fig4_behavioral(data, str(tmp_path / "fig4.png"))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["fig1_raw.png", "fig1_w3.png", "fig3.png", "fig4.png",
                     "fig5_raw.png", "fig5_w3.png"]


def test_generate_paper_figures_main_writes_both_versions(tmp_path, monkeypatch):
    computed = tmp_path / "metrics_computed.json"
    computed.write_text(json.dumps(_data()))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--computed", str(computed),
                                      "--output-dir", str(out), "--fig", "fig1"])
    gpf.main()
    assert sorted(p.name for p in out.iterdir()) == ["fig1_raw.png", "fig1_w3.png"]
