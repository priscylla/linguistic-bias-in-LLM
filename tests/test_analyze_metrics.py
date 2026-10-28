"""
tests/test_analyze_metrics.py
Fase 2: métricas de analyze.py em tabelas sintéticas com valores
conferíveis à mão (CS_e, k*, CS*, supressão, zona de decisão, linha de
base, veredito do artigo, ΔCS sobre médias de paráfrases).
"""

import numpy as np
import pandas as pd
import pytest

import analyze
from config.experiment_config import ExperimentConfig

CFG = ExperimentConfig()
L   = 4


def _rows(curves, lang="pt", paraphrase=1, n_ids=None, **extra):
    """curves: {candidato: [P por camada]}; wright é o dominante."""
    n_ids = n_ids or {}
    return [{"model": "tiny", "lens": "logit_lens", "fact": "aviao", "lang": lang,
             "format": "completion", "prompt_type": "F1", "paraphrase": paraphrase,
             "slot_case": "nom", "article_blocked": False, "token_set": "names",
             "candidate": c, "dominant": c == "wright", "layer": k + 1,
             "depth": (k + 1) / L, "n_layers": L, "prob": p, "n_ids": n_ids.get(c, 1), **extra}
            for c, ps in curves.items() for k, p in enumerate(ps)]


SD = [0.10, 0.50, 0.20, 0.05]     # pico na camada 2
WR = [0.10, 0.10, 0.30, 0.60]


def _metrics(probs, baseline=None, forced=None):
    tables = {"probs": pd.DataFrame(probs)}
    if baseline is not None:
        tables["baseline"] = pd.DataFrame(baseline)
    if forced is not None:
        tables["forced"] = pd.DataFrame(forced)
    return analyze.compute_all(tables, CFG)


def _one(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    assert m.sum() == 1, kw
    return df[m].iloc[0]


def test_cs_peak_suppression_zone():
    res = _metrics(_rows({"wright": WR, "santos_dumont": SD}) +
                   _rows({"wright": WR, "santos_dumont": SD}, lang="en"))
    C = res["curves"]
    sd = C[(C.candidate == "santos_dumont") & (C.lang == "pt")].sort_values("layer")
    assert sd["cs"].tolist() == pytest.approx([0.0, 0.4, -0.1, -0.55])
    assert C[C.candidate == "wright"]["cs"].isna().all()          # dominante não tem CS

    m = _one(res["metrics"], candidate="santos_dumont", lang="pt")
    assert m.k_star == 2 and m.k_star_depth == pytest.approx(0.5)
    assert m.p_peak == pytest.approx(0.5) and m.p_final == pytest.approx(0.05)
    assert m.suppression == pytest.approx(0.45)
    assert m.cs_star == pytest.approx(0.4) and m.cs_final == pytest.approx(-0.55)
    # zona de decisão 0.30 com L=4: camadas k > 2.8 -> 3 e 4
    assert m.cs_zone == pytest.approx((-0.1 - 0.55) / 2)
    assert m.stake and m.kind == "disputed"
    assert not _one(res["metrics"], candidate="santos_dumont", lang="en").stake
    # dominante: pico e supressão próprios (controles usam isso)
    w = _one(res["metrics"], candidate="wright", lang="pt")
    assert w.k_star == 4 and w.suppression == pytest.approx(0.0) and np.isnan(w.cs_final)


def test_peak_is_first_max():
    res = _metrics(_rows({"wright": WR, "santos_dumont": [0.1, 0.4, 0.4, 0.0]}))
    assert _one(res["metrics"], candidate="santos_dumont").k_star == 2


def test_cs_undefined_when_a_set_is_empty():
    res = _metrics(_rows({"wright": [0.0] * 4, "santos_dumont": SD}, n_ids={"wright": 0}))
    assert res["curves"]["cs"].isna().all()
    res = _metrics(_rows({"wright": WR, "santos_dumont": [0.0] * 4}, n_ids={"santos_dumont": 0}))
    assert res["curves"]["cs"].isna().all()


def test_cs_undefined_without_dominant_indicators():
    rows = [r for r in _rows({"wright": WR, "santos_dumont": SD}) if r["candidate"] != "wright"]
    res  = _metrics(rows)
    assert res["curves"]["cs"].isna().all()
    assert _one(res["metrics"], candidate="santos_dumont").p_peak == pytest.approx(0.5)


def test_baseline_adjusted_difference():
    base = _rows({"wright": [0.01] * 4, "santos_dumont": [0.04] * 4})
    for r in base:
        r["article_blocked"] = False
    res = _metrics(_rows({"wright": WR, "santos_dumont": SD}), baseline=base)
    sd  = res["curves"][res["curves"].candidate == "santos_dumont"].sort_values("layer")
    assert sd["b"].tolist() == pytest.approx([0.04] * 4)
    assert sd["p_adj"].tolist() == pytest.approx([p - 0.04 for p in SD])
    assert sd["cs_adj"].tolist() == pytest.approx([(s - 0.04) - (w - 0.01) for s, w in zip(SD, WR)])
    m = _one(res["metrics"], candidate="santos_dumont")
    assert m.cs_adj_final == pytest.approx((0.05 - 0.04) - (0.60 - 0.01))


def _forced(pass_id, kind, steps):
    """steps: lista de {candidato: P na última camada}, um por passo."""
    rows = []
    for i, final in enumerate(steps, start=1):
        for r in _rows({c: [0.0] * (L - 1) + [p] for c, p in final.items()}):
            r.update(forced_kind=kind, pass_id=pass_id, force_step=i, n_steps=len(steps),
                     forced_suffix=" os", trigger_token="os")
            rows.append(r)
    return rows


@pytest.mark.parametrize("final,verdict", [
    ({"wright": 0.7, "santos_dumont": 0.1}, "dominant_continuation"),
    ({"wright": 0.1, "santos_dumont": 0.6}, "local_recovers"),     # 0.6 >= pico 0.5
    ({"wright": 0.1, "santos_dumont": 0.3}, "suppressed"),          # abaixo do pico
])
def test_article_verdict(final, verdict):
    # O passo 1 diria outra coisa: o veredito usa o último passo
    forced = _forced("article:os", "article", [{"wright": 0.0, "santos_dumont": 0.9}, final])
    res = _metrics(_rows({"wright": WR, "santos_dumont": SD}), forced=forced)
    c = _one(res["confound"], candidate="santos_dumont")
    assert c.verdict == verdict and c.force_step == 2
    m = _one(res["metrics"], candidate="santos_dumont")
    assert m.article_verdict == verdict and m.article_trigger == "os"


def test_control_dominant_recovers_or_suppressed():
    forced = _forced("article:os", "article", [{"wright": 0.7}])
    res = _metrics(_rows({"wright": WR}), forced=forced)
    assert _one(res["confound"], candidate="wright").verdict == "local_recovers"


def test_not_triggered_and_shared_prefix_without_verdict():
    forced = _forced("shared_prefix: São", "shared_prefix", [{"wright": 0.2, "santos_dumont": 0.1}])
    res = _metrics(_rows({"wright": WR, "santos_dumont": SD}), forced=forced)
    assert res["confound"]["verdict"].isna().all()
    assert set(res["metrics"]["article_verdict"]) == {"not_triggered"}


def test_delta_cs_on_paraphrase_means():
    pt = (_rows({"wright": WR, "santos_dumont": SD}, paraphrase=1)
          + _rows({"wright": WR, "santos_dumont": [0.0, 0.1, 0.1, 0.25]}, paraphrase=2))
    en = _rows({"wright": WR, "santos_dumont": [0.0, 0.0, 0.0, 0.0]}, lang="en")
    res = _metrics(pt + en)
    d = _one(res["delta_cs"], candidate="santos_dumont", lang="pt")
    cs_pt = np.mean([0.05 - 0.6, 0.25 - 0.6])
    assert d.n_paraphrases == 2
    assert d.cs_final == pytest.approx(cs_pt)
    assert d.d_cs_final == pytest.approx(cs_pt - (0.0 - 0.6))
    assert _one(res["delta_cs"], candidate="santos_dumont", lang="en").d_cs_final == pytest.approx(0.0)
    cm = res["curves_mean"]
    last = _one(cm, candidate="santos_dumont", lang="pt", layer=L)
    assert last.d_cs == pytest.approx(cs_pt + 0.6)


def test_decision_zone_rule():
    assert analyze.in_decision_zone([1, 2, 3, 4], 4, 0.30).tolist() == [False, False, True, True]
    # L=40, zona 0.30: camadas 29..40 (12 camadas)
    assert analyze.in_decision_zone(list(range(1, 41)), 40, 0.30).sum() == 12
    assert analyze.in_decision_zone(29, 40, 0.30) and not analyze.in_decision_zone(28, 40, 0.30)


def test_guard_refuses_raw_dir(tmp_path):
    with pytest.raises(ValueError):
        analyze._guard_output(str(tmp_path / "lens" / "llama"), str(tmp_path))
    analyze._guard_output(str(tmp_path / "metrics"), str(tmp_path))
