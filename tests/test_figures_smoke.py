"""
tests/test_figures_smoke.py
Pipeline analyze.py -> make_figures.py com resultados sintéticos no
formato novo: figuras de curva em duas versões (_w3 e _raw), heatmap
colorido por token ID, número de camadas lido dos dados.
"""

import json

import numpy as np

import analyze
import make_figures as mf
from config.experiment_config import ExperimentConfig

# Camadas diferentes por modelo, para garantir que nada usa tabela fixa
N_LAYERS = {"llama": 6, "mistral": 6, "qwen": 5, "gemma": 7}


def _run(n_layers, seed):
    rng = np.random.default_rng(seed)
    p_loc  = rng.uniform(0, 0.5, n_layers)
    p_comp = rng.uniform(0, 0.5, n_layers)
    layers = [{
        "layer": i + 1, "layer_norm": round(i / (n_layers - 1), 4),
        "top_tokens": [" Bell", " the", " Me", " a", " il"],
        "top_ids":    [10, 2, 20, 3, 4],
        "top_probs":  [0.3, 0.2, 0.1, 0.05, 0.05],
        "entity_probs": {"local": float(a), "competing": float(b)},
    } for i, (a, b) in enumerate(zip(p_loc, p_comp))]
    return {
        "prompt": "p", "response": "Alexander Graham Bell",
        "n_layers": n_layers, "last_token": layers, "keyword_token": None,
        "entity_tokens": {"sets": {"local": {"ids": [10]},
                                   "competing": {"ids": [20]}}},
    }


def write_raw(results_dir, config):
    for k, model in enumerate(config.models):
        data = {fact: {lang: {form: _run(N_LAYERS[model], k * 100 + j)
                              for form in config.formulations}
                       for j, lang in enumerate(config.languages)}
                for fact in config.facts}
        (results_dir / f"{model}_results.json").write_text(json.dumps(data))


def test_analyze_then_make_figures(tmp_path):
    config = ExperimentConfig()
    res, figs = tmp_path / "results", tmp_path / "figures"
    res.mkdir()
    write_raw(res, config)

    assert analyze.main(["--results-dir", str(res), "--output-dir", str(res)]) == 0
    computed = json.loads((res / "metrics_computed.json").read_text())
    assert computed["qwen"]["aviao"]["pt"]["F1"]["n_layers"] == 5
    assert len(computed["gemma"]["telefone"]["it"]["F3"]
               ["metrics_last"]["commitment_curve"]) == 7

    assert mf.main(["--results-dir", str(res), "--output-dir", str(figs)]) == 0
    names = {p.name for p in figs.iterdir()}
    assert {"fig1_w3.png", "fig1_raw.png", "fig6.png",
            "fig3_logit_lens_aviao.png", "fig3_logit_lens_telefone_abs.png",
            "fig3_logit_lens_aviao_llama.png",
            "fig3_logit_lens_telefone_gemma_it_abs.png"} <= names
    n_fig3 = len(config.facts) * 2 * (1 + len(config.models) * (1 + len(config.languages)))
    assert len(names) == 3 + n_fig3


def test_model_layers_from_data():
    data = {"qwen": {"aviao": {"pt": {"F1": {"n_layers": 28}}}}}
    assert mf.model_layers(data, "qwen") == 28
    assert mf.model_layers(data, "llama") is None


def test_decision_start_uses_config():
    assert mf.decision_start(40) == int(round(40 * (1 - mf.CONFIG.decision_zone)))
