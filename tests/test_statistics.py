"""
tests/test_statistics.py
Fase 6: IC bootstrap estratificado, teste de sinal por permutação,
Holm, teste stake × não-stake (com o espelho do dominante), modelos
mistos e junção interno × comportamental, em dados sintéticos.
"""

import numpy as np
import pandas as pd
import pytest

from analysis import statistics as st
from config.experiment_config import ExperimentConfig

CFG  = ExperimentConfig()
SCFG = {**CFG.experiment["statistics"], "bootstrap": 2000}
LANGS = CFG.languages


def _M(values, fact="aviao", model="llama", lens="logit_lens", fmt="completion",
       ts="names", dominant_vals=None):
    """values: {(lang, prompt_type): [cs por paráfrase]} para santos_dumont."""
    rows = []
    for (lang, pt), vs in values.items():
        for i, v in enumerate(vs, start=1):
            base = dict(model=model, lens=lens, fact=fact, lang=lang, format=fmt,
                        prompt_type=pt, paraphrase=i, token_set=ts)
            rows.append({**base, "candidate": "santos_dumont", "dominant": False,
                         "cs_zone": v, "cs_star": v + 0.1, "cs_final": v - 0.1, "suppression": 0.2})
            rows.append({**base, "candidate": "wright", "dominant": True,
                         "cs_zone": np.nan, "cs_star": np.nan, "cs_final": np.nan, "suppression": 0.1})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Bootstrap
# ══════════════════════════════════════════════════════════════

def test_stratified_bootstrap_pools_formulations():
    # Cada formulação constante: a média estratificada não varia entre reamostragens
    vals = {("pt", "F1"): [1, 1, 1], ("pt", "F2"): [0, 0, 0], ("pt", "F3"): [.5, .5, .5],
            ("en", "F1"): [0, 0, 0], ("en", "F2"): [0, 0, 0], ("en", "F3"): [0, 0, 0]}
    ci = st.bootstrap_ci(_M(vals), SCFG, "en")
    r = ci[(ci.prompt_group == "F1-F3") & (ci.lang == "pt") & (ci.metric == "cs_zone")
           & (ci.candidate == "santos_dumont")].iloc[0]
    assert r.n_prompts == 9
    assert r.estimate == pytest.approx(0.5) and r.ci_low == pytest.approx(0.5) and r.ci_high == pytest.approx(0.5)
    assert r.delta == pytest.approx(0.5) and r.delta_low == pytest.approx(0.5)
    per = ci[(ci.prompt_group == "F1") & (ci.lang == "pt") & (ci.metric == "cs_zone")].iloc[0]
    assert per.n_prompts == 3 and per.estimate == pytest.approx(1.0)


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    vals = {(l, t): list(rng.normal(0.2 if l == "pt" else 0, 0.1, 3))
            for l in ("pt", "en", "de") for t in ("F1", "F2", "F3", "QA")}
    M = _M(vals)
    a = st.bootstrap_ci(M, SCFG, "en")
    b = st.bootstrap_ci(M.sample(frac=1, random_state=3), SCFG, "en")   # outra ordem
    key = ["prompt_group", "lang", "metric", "candidate"]
    a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    ok = a.dropna(subset=["ci_low"])
    assert ((ok.ci_low <= ok.estimate + 1e-12) & (ok.estimate <= ok.ci_high + 1e-12)).all()
    assert set(a.metric) == set(SCFG["ci_metrics"])
    assert a[(a.lang == "en")]["delta"].isna().all()               # referência sem ΔCS
    assert a[(a.metric == "suppression")]["delta"].isna().all()     # supressão sem ΔCS
    # dominante: sem CS, só supressão
    assert set(a[a.candidate == "wright"]["metric"]) == {"suppression"}


# ══════════════════════════════════════════════════════════════
# Permutação e Holm
# ══════════════════════════════════════════════════════════════

def test_sign_flip_exact_and_monte_carlo():
    r = st.sign_flip_test(np.array([1.0, 2.0, 3.0]), 16, 1000, np.random.default_rng(0))
    assert r["method"] == "exact" and r["p"] == pytest.approx(2 / 8)
    assert st.sign_flip_test(np.array([1.0, -1.0]), 16, 10, None)["p"] == pytest.approx(1.0)
    d = np.random.default_rng(1).normal(1.0, 0.5, 30)
    mc = st.sign_flip_test(d, 16, 20000, np.random.default_rng(2))
    assert mc["method"].startswith("monte_carlo") and mc["p"] < 0.001
    assert st.sign_flip_test(np.array([]), 16, 10, None)["method"] == "none"


def test_holm():
    assert st.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert st.holm([0.5, np.nan]) == pytest.approx([0.5, np.nan], nan_ok=True)


# ══════════════════════════════════════════════════════════════
# Stake × não-stake
# ══════════════════════════════════════════════════════════════

def test_mirror_score_for_dominant():
    M = _M({("pt", "F1"): [0.4, 0.2, 0.0]})
    S = st.stake_scores(M, "cs_zone")
    w = S[S.candidate == "wright"].sort_values("paraphrase")["score"].tolist()
    assert w == pytest.approx([-0.4, -0.2, 0.0])


def _stake_data(effect):
    parts = []
    rng = np.random.default_rng(5)
    for m in ("llama", "mistral", "qwen", "gemma", "llama_base", "qwen_base"):
        vals = {}
        for l in LANGS:
            for t in ("F1", "F2", "F3"):
                mu = effect if l == "pt" else (effect / 2 if l == "fr" else 0.0)
                vals[(l, t)] = list(mu + rng.normal(0, 0.02, 3))
        parts.append(_M(vals, model=m))
    return pd.concat(parts, ignore_index=True)


def test_stake_tests_detect_effect_and_keep_types_separate():
    T = st.stake_tests(_stake_data(0.3), CFG.fact_specs, LANGS, {**SCFG, "permutation_mc": 20000})
    sd = T[(T.candidate == "santos_dumont")].set_index("stake_type")
    assert set(sd.index) == {"nationality", "place_of_event"}
    nat = sd.loc["nationality"]
    assert nat.stake_lang == "pt" and nat.n_units == 18 and nat.mean_d == pytest.approx(0.3, abs=0.02)
    assert nat.p < 0.001 and nat.method.startswith("monte_carlo")
    assert set(nat.non_stake_langs.split(",")) == {"en", "de", "it"}   # pt e fr têm stake
    assert sd.loc["place_of_event"].stake_lang == "fr"
    # dominante (Wright, stake en) testado pelo espelho
    w = T[T.candidate == "wright"].iloc[0]
    assert w.dominant and w.stake_lang == "en"
    assert {"p_holm"} <= set(T.columns) and (T.p_holm >= T.p - 1e-12).all()


def test_stake_tests_null_effect_not_significant():
    T = st.stake_tests(_stake_data(0.0), CFG.fact_specs, LANGS, SCFG)
    assert (T[T.candidate == "santos_dumont"]["p"] > 0.01).all()


def _en_heavy_data():
    """Sem efeito de stake, mas en muito comprometido com o dominante (CS baixo)."""
    parts = []
    rng = np.random.default_rng(7)
    for m in ("llama", "mistral", "qwen", "gemma", "llama_base", "qwen_base"):
        vals = {(l, t): list((-0.3 if l == "en" else 0.0) + rng.normal(0, 0.02, 3))
                for l in LANGS for t in ("F1", "F2", "F3")}
        parts.append(_M(vals, model=m))
    return pd.concat(parts, ignore_index=True)


def test_stake_tests_without_en_in_comparison():
    M = _en_heavy_data()
    full = st.stake_tests(M, CFG.fact_specs, LANGS, SCFG)
    non_en = st.stake_tests(M, CFG.fact_specs, LANGS, SCFG, exclude_from_comparison=("en",))
    sd_full = full[(full.candidate == "santos_dumont") & (full.stake_type == "nationality")].iloc[0]
    sd = non_en[(non_en.candidate == "santos_dumont") & (non_en.stake_type == "nationality")].iloc[0]
    assert set(sd.non_stake_langs.split(",")) == {"de", "it"} and sd.comparison == "non_stake_no_en"
    assert sd_full.comparison == "non_stake"
    assert sd_full.mean_d == pytest.approx(0.1, abs=0.02) and sd_full.p < 0.01   # inflado por en
    assert abs(sd.mean_d) < 0.02 and sd.p > 0.01                                 # sem en: nada
    # candidato com stake em en: o grupo já não tinha en, nada muda
    w_full = full[full.candidate == "wright"].iloc[0]
    w = non_en[non_en.candidate == "wright"].iloc[0]
    assert w.non_stake_langs == w_full.non_stake_langs and w.mean_d == pytest.approx(w_full.mean_d)
    assert (non_en.p_holm >= non_en.p - 1e-12).all()


# ══════════════════════════════════════════════════════════════
# Modelos mistos
# ══════════════════════════════════════════════════════════════

def _D(effect=0.3, seed=0):
    """Linhas célula × idioma com efeito de stake conhecido + efeitos de fato e modelo."""
    rng = np.random.default_rng(seed)
    rows = []
    disputed = [k for k, f in CFG.fact_specs.items() if f.kind == "disputed"]
    models = ["llama", "mistral", "qwen", "gemma", "llama_base", "qwen_base", "gemma_base"]
    fx = {f: rng.normal(0, 0.05) for f in disputed}
    mx = {m: rng.normal(0, 0.05) for m in models}
    for f in disputed:
        for c in CFG.fact_specs[f].alternatives:
            stakes = {s.language for s in c.stakes}
            for m in models:
                pts = ["F1", "F2", "F3"] + (["QA"] if CFG.registry[m].kind == "instruct" else [])
                for pt in pts:
                    for l in LANGS:
                        y = fx[f] + mx[m] + (effect if l in stakes else 0) + rng.normal(0, 0.05)
                        rows.append({"model": m, "lens": "logit_lens", "fact": f,
                                     "format": "qa" if pt == "QA" else "completion",
                                     "prompt_type": pt, "token_set": "names", "candidate": c.id,
                                     "lang": l, "dominant": False,
                                     "d_cs_zone": 0.0 if l == "en" else y})
    return pd.DataFrame(rows)


def test_mixed_models_recover_stake_effect():
    coefs, fits = st.mixed_models(_D(0.3), CFG.fact_specs, CFG.registry, "en", "cs_zone")
    assert set(fits["fit"]) == {"A_all", "B_completion"}
    assert fits["converged"].all()
    a = fits[fits.fit == "A_all"].iloc[0]
    assert "C(format)" in a.formula and "(1|fact)" in a.formula and "(1|model)" in a.formula
    assert "C(formulation)" in fits[fits.fit == "B_completion"].iloc[0].formula
    for fit in ("A_all", "B_completion"):
        c = coefs[(coefs.fit == fit) & coefs.term.str.contains("nationality")].iloc[0]
        assert c.coef == pytest.approx(0.3, abs=0.05) and c.p < 1e-6
    assert a.n_obs == len(_D()[lambda d: d.lang != "en"])


def test_mixed_models_drop_single_level_terms():
    D = _D(0.3)
    D = D[D.model.isin(["llama", "mistral", "qwen", "gemma"])]        # só instruct
    _, fits = st.mixed_models(D, CFG.fact_specs, CFG.registry, "en", "cs_zone")
    assert (fits["dropped_terms"].str.contains("base_vs_instruct")).all()


# ══════════════════════════════════════════════════════════════
# Interno × comportamental
# ══════════════════════════════════════════════════════════════

def test_internal_behavioral_join():
    D = pd.DataFrame([
        {"model": "llama", "lens": "logit_lens", "fact": "aviao", "lang": "pt", "candidate": "santos_dumont",
         "format": f, "prompt_type": t, "token_set": "names", "dominant": False,
         "cs_zone": v, "cs_star": v, "cs_final": v, "d_cs_zone": v, "d_cs_star": v,
         "p_peak": .5, "p_final": .1, "suppression": .4}
        for f, t, v in [("qa", "QA", 0.4), ("completion", "F1", 0.1), ("completion", "F2", 0.3),
                        ("completion", "F3", 0.2)]])
    rates = pd.DataFrame([{"model": "llama", "fact": "aviao", "lang": "pt", "candidate": "santos_dumont",
                           "kind": k, "n_answers": 33, "primary_rate": r, "mention_rate": .9,
                           "mean_rank": 1.2} for k, r in (("all", 0.6), ("greedy", 1.0))])
    J = st.internal_behavioral(D, rates, ["F1", "F2", "F3"])
    assert len(J) == 1
    r = J.iloc[0]
    assert r.cs_zone == pytest.approx(0.4) and r.cs_zone_completion == pytest.approx(0.2)
    assert r.primary_rate == pytest.approx(0.6)


def test_mixed_models_without_llama():
    D = _D(0.3)
    _, fits = st.mixed_models(D, CFG.fact_specs, CFG.registry, "en", "cs_zone", exclude_models=st.NO_LLAMA)
    a = fits[fits.fit == "A_all"].iloc[0]
    assert a.n_model == 5 and a.n_obs == len(D[(D.lang != "en") & ~D.model.isin(["llama", "llama_base"])])


def test_mixed_by_model_one_fit_per_model():
    coefs, fits = st.mixed_by_model(_D(0.3), CFG.fact_specs, CFG.registry, "en", "cs_zone")
    assert set(fits.model) == {"llama", "mistral", "qwen", "gemma", "llama_base", "qwen_base", "gemma_base"}
    assert not fits.formula.str.contains("1|model", regex=False).any()
    assert fits.formula.str.contains("1|fact", regex=False).all()
    base = fits[fits.model == "qwen_base"].iloc[0]
    assert "format" in base.dropped_terms                              # base: só completion
    nat = coefs[coefs.term.str.contains("nationality")]
    assert len(nat) == 7 and (nat.coef - 0.3).abs().max() < 0.06


def _fake_fit(monkeypatch, converges_with):
    """MixedLM.fit que só converge com os otimizadores em `converges_with`."""
    from statsmodels.regression.mixed_linear_model import MixedLM
    orig = MixedLM.fit
    calls = []

    def fit(self, *a, method=None, **k):
        calls.append(method or "default")
        r = orig(self, *a, **k)
        r.converged = (method or "default") in converges_with
        return r
    monkeypatch.setattr(MixedLM, "fit", fit)
    return calls


def test_unconverged_fit_retried_with_lbfgs_then_powell(monkeypatch):
    calls = _fake_fit(monkeypatch, {"powell"})
    _, fits = st.mixed_models(_D(0.3), CFG.fact_specs, CFG.registry, "en", "cs_zone")
    a = fits[fits.fit == "A_all"].iloc[0]
    assert a.converged and a.optimizer == "powell" and a.optimizers_tried == "default,lbfgs,powell"
    assert calls[:3] == ["default", "lbfgs", "powell"]


def test_fit_that_never_converges_says_to_report_fit_b(monkeypatch):
    _fake_fit(monkeypatch, set())
    _, fits = st.mixed_models(_D(0.3), CFG.fact_specs, CFG.registry, "en", "cs_zone")
    a = fits[fits.fit == "A_all"].iloc[0]
    assert not a.converged and "ajuste B" in a.note
    assert "ajuste B" not in fits[fits.fit == "B_completion"].iloc[0].note
