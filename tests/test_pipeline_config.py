"""
tests/test_pipeline_config.py
Fase 0, correções 7, 10 e consolidação: prompts com acentos, contagem
esperada derivada do config, analyze.py nunca modifica resultados
brutos, camadas normalizadas pelo número de camadas do modelo.
"""

import hashlib
import json

import pytest

import analyze
import check_results
from config.experiment_config import ExperimentConfig
from data.prompts import COMPLETION_PROMPTS
from logit_lens.normalizer import normalize_layer, normalize_layer_results
from test_figures_smoke import write_raw

CONFIG = ExperimentConfig()


# ── Correção 7: prompts ──

def test_prompts_cover_config():
    for fact in CONFIG.facts:
        for lang in CONFIG.languages:
            for form in CONFIG.formulations:
                assert COMPLETION_PROMPTS[fact][lang][form].strip()


@pytest.mark.parametrize("bad", ["aviao", " e stato", " e considerad", "Gebruder"])
def test_no_unaccented_forms(bad):
    for fact in COMPLETION_PROMPTS.values():
        for lang in fact.values():
            for prompt in lang.values():
                assert bad not in prompt, prompt


def test_keyword_appears_in_every_prompt():
    for fact in CONFIG.facts:
        for lang in CONFIG.languages:
            kw = CONFIG.keyword_tokens[fact][lang]
            for form in CONFIG.formulations:
                assert kw in COMPLETION_PROMPTS[fact][lang][form]


# ── Correção 10: contagem esperada vem do config ──

def test_expected_cells_from_config():
    assert CONFIG.expected_cells() == (len(CONFIG.models) * len(CONFIG.facts)
                                       * len(CONFIG.languages) * len(CONFIG.formulations))
    assert CONFIG.expected_cells() == 96


def test_check_results_all_valid(tmp_path):
    write_raw(tmp_path, CONFIG)
    assert check_results.check_all_results(str(tmp_path), CONFIG)


def test_check_results_flags_missing_and_old_format(tmp_path):
    write_raw(tmp_path, CONFIG)
    path = tmp_path / "llama_results.json"
    data = json.loads(path.read_text())
    del data["aviao"]["pt"]["F1"]
    for l in data["aviao"]["en"]["F2"]["last_token"]:
        del l["entity_probs"]
    path.write_text(json.dumps(data))
    assert not check_results.check_all_results(str(tmp_path), CONFIG)
    assert check_results.check_run({}) == "missing"
    assert "old format" in check_results.check_run(data["aviao"]["en"]["F2"])


# ── Consolidação: analyze.py não toca nos brutos ──

def _digest(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.glob("*_results.json"))}


def test_analyze_never_modifies_raw(tmp_path):
    write_raw(tmp_path, CONFIG)
    before = _digest(tmp_path)
    assert analyze.main(["--results-dir", str(tmp_path),
                         "--output-dir", str(tmp_path)]) == 0
    assert _digest(tmp_path) == before
    assert (tmp_path / "metrics_computed.csv").exists()


def test_analyze_refuses_raw_output_name():
    with pytest.raises(ValueError):
        analyze._guard_output("results/llama_results.json")


def test_analyze_csv_has_stake_from_config(tmp_path):
    write_raw(tmp_path, CONFIG)
    analyze.main(["--results-dir", str(tmp_path), "--output-dir", str(tmp_path)])
    import pandas as pd
    df = pd.read_csv(tmp_path / "metrics_computed.csv")
    assert len(df) == CONFIG.expected_cells()
    stake = df[df["has_stake"]][["fact", "language"]].drop_duplicates()
    assert set(map(tuple, stake.values)) == {
        (f, l) for f, ls in CONFIG.stake_languages.items() for l in ls}


# ── Normalização usa o número de camadas do modelo ──

def test_normalize_layer_uses_n_layers():
    assert normalize_layer(1, 28) == 0.0
    assert normalize_layer(28, 28) == 1.0
    layers = normalize_layer_results([{"layer": k} for k in (1, 21, 41)], 41)
    assert [l["layer_norm"] for l in layers] == [0.0, 0.5, 1.0]
