"""
tests/test_paper_numbers.py
analysis/paper_numbers.py: profundidades de decisão, vitórias na última
camada, robustez do coeficiente de nacionalidade, correlação interno ×
comportamental, pico e queda, marcadores de narrativa.
"""

import numpy as np
import pandas as pd
import pytest

from analysis import paper_numbers as pn
from config.experiment_config import ExperimentConfig

CFG = ExperimentConfig()
F = CFG.fact_specs


def test_depths_rules():
    layers = np.array([1, 2, 3, 4])
    P = np.array([[0.001, 0.000],      # nada ≥ 0.01
                  [0.020, 0.030],      # emergência; topo = b
                  [0.200, 0.050],      # metade do máximo final (0.4); topo = a
                  [0.400, 0.100]])     # vencedor a
    d = pn.depths(layers, 4, ["a", "b"], P)
    assert d == {"emergence": 0.5, "half": 0.75, "decision": 0.75, "winner": "a"}
    never = pn.depths(layers, 4, ["a", "b"], P * 0.001)
    assert np.isnan(never["emergence"]) and never["decision"] == 0.75


def _curves(p_alt, p_dom, fact="aviao", alt="santos_dumont", model="m1", lang="pt", pt="F1"):
    L = len(p_alt)
    return pd.DataFrame({"model": model, "lens": "logit_lens", "fact": fact, "format": "completion",
                         "prompt_type": pt, "token_set": "names", "candidate": alt, "lang": lang,
                         "layer": range(1, L + 1), "n_layers": L, "prob": p_alt, "p_dom": p_dom})


def test_decision_depth_and_final_wins():
    C = pd.concat([_curves([0.0, 0.3, 0.6, 0.2], [0.0, 0.1, 0.2, 0.7]),              # Wright vence
                   _curves([0.0, 0.05, 0.3, 0.5], [0.001, 0.02, 0.1, 0.2], lang="en")])  # Santos-Dumont vence
    T = pn.decision_depth(C, F).set_index("lang")
    assert T.loc["pt", "winner"] == "wright" and T.loc["pt", "winner_is_dominant"]
    assert T.loc["pt", "decision"] == 1.0 and T.loc["pt", "emergence"] == 0.5
    assert T.loc["en", "winner"] == "santos_dumont" and T.loc["en", "decision"] == 0.5
    S = pn.decision_depth_summary(T.reset_index())
    assert set(S["model"]) == {"m1", "all"}
    W = pn.final_layer_wins(T.reset_index(), F).set_index(["candidate", "lang"])["win_share"]
    assert W[("wright", "pt")] == 1.0 and W[("santos_dumont", "en")] == 1.0 and W[("santos_dumont", "pt")] == 0.0


def test_nationality_robustness_variants():
    from test_statistics import _D
    D = _D(0.3)
    for m in ("cs_adj_zone", "cs_final", "cs_star"):
        D[f"d_{m}"] = D["d_cs_zone"]
    R = pn.nationality_robustness(D, F, CFG.registry, "en")
    assert set(R["variant_type"]) == {"metric", "leave_one_fact_out", "formulation"}
    assert set(R[R.variant_type == "formulation"]["variant"]) == {"F1", "F2", "F3", "QA"}
    n_facts = D[~D["dominant"]]["fact"].nunique()
    assert (R.variant_type == "leave_one_fact_out").sum() == n_facts
    assert (R["coef"] - 0.3).abs().max() < 0.06 and R["converged"].all()


def test_internal_behavioral_corr():
    J = pd.DataFrame({"model": ["m"] * 4, "lens": "logit_lens", "fact": "aviao", "lang": list("abcd"),
                      "candidate": ["x", "x", "y", "y"], "dominant": False,
                      "cs_zone": [1.0, 3.0, 5.0, 7.0], "cs_zone_completion": [7.0, 5.0, 3.0, 1.0],
                      "primary_rate": [1.0, 2.0, 3.0, 4.0], "mention_rate": [3.0, 1.0, 7.0, 5.0]})
    R = pn.internal_behavioral_corr(J).set_index(["measure", "rate"])["spearman_rho"]
    assert R[("qa_cs_zone", "primary_rate")] == pytest.approx(1.0)
    assert R[("completion_cs_zone", "primary_rate")] == pytest.approx(-1.0)
    # dentro do candidato: x: cs ↑, menção ↓; y: cs ↑, menção ↓
    assert R[("qa_cs_zone_within_candidate", "mention_rate")] == pytest.approx(-1.0)


def test_peak_drop_roles():
    M = pd.DataFrame({"token_set": "names", "lens": "logit_lens",
                      "kind": ["disputed", "disputed", "disputed", "control", "disputed"],
                      "dominant": [False, False, True, True, False], "stake": [True, False, False, False, False],
                      "p_peak": [0.2, 0.2, 0.2, 0.2, 0.005], "p_final": [0.05, 0.15, 0.05, 0.2, 0.0]})
    R = pn.peak_drop(M).set_index("role")
    assert R.loc["local", "n_prompts"] == 2 and R.loc["local", "drop_share"] == 0.5     # p_peak < 0.01 fora
    assert R.loc["local_stake_language", "drop_share"] == 1.0
    assert R.loc["dominant", "drop_share"] == 1.0 and R.loc["control", "drop_share"] == 0.0


def test_narrative_markers_pool_answers():
    rates = pd.DataFrame({"model": ["a", "b", "a"], "fact": "aviao", "lang": "pt", "candidate": "wright",
                          "indicator_type": "place", "indicator": "Kitty Hawk",
                          "n_answers": [33, 132, 33], "mention_rate": [1.0, 0.5, 0.0],
                          "given_primary": ["any", "any", "wright"]})
    T = pn.narrative_markers(rates).iloc[0]
    assert T.n_answers == 165 and T.n_mentioned == 99 and T.mention_rate == pytest.approx(0.6)


def test_control_fact_has_one_candidate_column():
    # Controle: o dominante (Fleming) aparece como candidato E em p_dom; uma coluna só
    C = _curves([0.0, 0.2, 0.6], [0.0, 0.2, 0.6], fact="penicilina", alt="fleming")
    T = pn.decision_depth(C, F).iloc[0]
    assert T.winner == "fleming" and T.winner_is_dominant and T.kind == "control"
