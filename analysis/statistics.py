"""
analysis/statistics.py
Estatística da Fase 6, sobre as tabelas de analyze.py. Configuração em
config/experiment.yaml (statistics); decisões confirmadas pela autora
em 2026-10-28.

Métrica principal: cs_zone (média de CS_e na zona de decisão); ΔCS =
métrica(l) − métrica(en).

1. IC bootstrap (percentil, 10.000 reamostragens) por célula
   modelo × lens × fato × formato × conjunto × grupo × candidato × idioma.
   Grupo "F1-F3": as três formulações juntas (9 prompts), reamostrando
   paráfrases dentro de cada formulação (estratificado); os demais
   grupos são um tipo de prompt (3 paráfrases; IC grosseiro, marcado
   com n_prompts = 3). Para ΔCS, idioma e referência são reamostrados
   de forma independente. Semente por célula (hash da chave): o
   resultado não depende da ordem.

2. Teste stake × não-stake por entidade, por tipo de stake
   (nationality, place_of_event separados). Score por prompt:
       candidato não dominante:  CS_e (métrica principal)
       candidato dominante:      P(dom) − média P(alternativas)
                                 = −média_alt CS_alt (espelho)
   Unidade pareada: modelo × formato × tipo de prompt (média das
   paráfrases); d = score(idioma de stake) − média score(idiomas sem
   nenhum stake da entidade). Teste de sinal por permutação, bicaudal,
   estatística = média de d: exato (2^n) com n ≤ 16, senão Monte
   Carlo (p = (k+1)/(N+1)). Holm por lens × conjunto × tipo de stake,
   entre fatos/entidades.
   CS (e não ΔCS) porque ΔCS(en) = 0 por definição e en é idioma de
   stake (Wright, Bell, Edison, Newton).
   Teste principal (stats_stake_tests, decisão da autora em 2026-10-30):
   o grupo de comparação são só os idiomas sem stake e diferentes de en
   (en é o idioma mais comprometido com o dominante e inflava d). A
   versão com en no grupo vai para o apêndice (stats_stake_tests_with_en).
   Mesmo Holm nas duas.

3. Modelos mistos (statsmodels MixedLM, efeitos aleatórios cruzados de
   fato e modelo como componentes de variância), linhas = célula ×
   idioma (médias das paráfrases), sem o idioma de referência, só
   candidatos não dominantes, um ajuste por lens × conjunto:
       A (todas):      ΔCS ~ stake_type + format + base_vs_instruct
       B (completion): ΔCS ~ stake_type + formulation + base_vs_instruct
   Termos com um único nível nos dados ficam fora (registrado).
   As linhas de en ficam fora (ΔCS(en) = 0 por definição); o coeficiente
   de stake compara idiomas de stake com idiomas sem stake ≠ en.
   Ajuste que não converge é refeito com outro otimizador (lbfgs, depois
   powell); a coluna `optimizer` diz qual convergiu e `note` registra
   quando nenhum convergiu (usar então o ajuste B).
   Sensibilidade: o mesmo sem o Llama 3.1 (base e instruct,
   stats_mixed_*_no_llama) e o ajuste A dentro de cada modelo
   (stats_mixed_*_by_model, só efeito aleatório de fato).

4. Junção métrica interna × comportamento por célula modelo × fato ×
   idioma × candidato: formato QA (mesmos prompts das respostas) e
   média de F1–F3 (completion).
"""

import hashlib
import itertools
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

KEYS = ["model", "lens", "fact", "lang", "format", "prompt_type", "paraphrase", "token_set"]
CELL = ["model", "lens", "fact", "format", "token_set", "prompt_group", "candidate"]
POOLED = "F1-F3"


def _rng(seed: int, key) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


# ══════════════════════════════════════════════════════════════
# 1. IC bootstrap
# ══════════════════════════════════════════════════════════════

def with_groups(M: pd.DataFrame, pooled_types: List[str]) -> pd.DataFrame:
    """Linhas por tipo de prompt (grupo = o tipo) + grupo agregado F1-F3."""
    per = M.assign(prompt_group=M["prompt_type"])
    pool = M[M["prompt_type"].isin(pooled_types)].assign(prompt_group=POOLED)
    return pd.concat([pool, per], ignore_index=True)


def _boot_means(strata: List[np.ndarray], B: int, rng) -> np.ndarray:
    """Média das médias dos estratos, B reamostragens (paráfrases dentro do estrato)."""
    parts = []
    for v in strata:
        idx = rng.integers(0, len(v), size=(B, len(v)))
        parts.append(v[idx].mean(axis=1))
    return np.mean(parts, axis=0)


def _strata(g: pd.DataFrame, metric: str) -> List[np.ndarray]:
    out = []
    # Ordem fixa (paráfrase) dentro do estrato: o resultado não pode
    # depender da ordem das linhas na tabela
    for _, s in g.groupby("prompt_type", sort=True):
        v = s.sort_values("paraphrase")[metric].dropna().to_numpy(dtype=float)
        if len(v):
            out.append(v)
    return out


def bootstrap_ci(M: pd.DataFrame, scfg: Dict, ref: str) -> pd.DataFrame:
    B, seed = scfg["bootstrap"], scfg["seed"]
    lo_q, hi_q = (1 - scfg["ci_level"]) / 2, 1 - (1 - scfg["ci_level"]) / 2
    G = with_groups(M, scfg["pooled_types"])
    rows = []
    for cell, g in G.groupby(CELL, sort=True):
        by_lang = dict(tuple(g.groupby("lang")))
        for metric in scfg["ci_metrics"]:
            boots = {}
            for lang, h in by_lang.items():
                st = _strata(h, metric)
                if not st:
                    continue
                est = float(np.mean([v.mean() for v in st]))
                b = _boot_means(st, B, _rng(seed, (cell, lang, metric)))
                boots[lang] = (est, b, sum(len(v) for v in st))
            for lang, (est, b, n) in boots.items():
                r = dict(zip(CELL, cell), lang=lang, metric=metric, n_prompts=n, estimate=est,
                         ci_low=float(np.quantile(b, lo_q)), ci_high=float(np.quantile(b, hi_q)))
                if metric in scfg["delta_metrics"] and lang != ref and ref in boots:
                    e0, b0, _ = boots[ref]
                    d = b - b0
                    r.update(delta=est - e0, delta_low=float(np.quantile(d, lo_q)),
                             delta_high=float(np.quantile(d, hi_q)))
                rows.append(r)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 2. Teste stake × não-stake
# ══════════════════════════════════════════════════════════════

def stake_scores(M: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Score por prompt × candidato: CS_e (alternativas) ou o espelho (dominante)."""
    alt = M[~M["dominant"]][KEYS + ["candidate", metric]]
    mirror = alt.groupby(KEYS)[metric].mean().mul(-1).rename("score").reset_index()
    dom = M[M["dominant"]][KEYS + ["candidate"]].merge(mirror, on=KEYS, how="inner")
    alt = alt.rename(columns={metric: "score"})
    S = pd.concat([alt, dom], ignore_index=True)
    return S.dropna(subset=["score"])


def sign_flip_test(d: np.ndarray, exact_max_n: int, n_mc: int, rng) -> Dict:
    """Teste de permutação por troca de sinal, bicaudal, estatística = média."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return {"p": np.nan, "method": "none"}
    obs = abs(d.mean())
    tol = 1e-12 * max(1.0, obs)
    if n <= exact_max_n:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        null = np.abs(signs @ d / n)
        return {"p": float((null >= obs - tol).mean()), "method": "exact"}
    signs = rng.choice((1.0, -1.0), size=(n_mc, n))
    k = int((np.abs(signs @ d / n) >= obs - tol).sum())
    return {"p": (k + 1) / (n_mc + 1), "method": f"monte_carlo_{n_mc}"}


def holm(p: List[float]) -> List[float]:
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m, running = len(idx), 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def stake_tests(M: pd.DataFrame, facts: Dict, languages: List[str], scfg: Dict,
                exclude_from_comparison=()) -> pd.DataFrame:
    """
    `exclude_from_comparison`: idiomas fora do grupo de comparação além
    dos de stake (ex.: ("en",): o inglês é o idioma mais comprometido com
    o dominante e puxa a média do grupo para baixo, inflando d). Para
    candidatos com stake em en, en é o próprio idioma testado e o grupo
    não muda.
    """
    S = stake_scores(M, scfg["primary_metric"])
    comparison = "non_stake" + "".join(f"_no_{l}" for l in exclude_from_comparison)
    unit = ["model", "lens", "format", "token_set", "prompt_type"]
    U = S.groupby(unit + ["fact", "candidate", "lang"])["score"].mean().reset_index()
    rows = []
    for (fact, cid), g in U.groupby(["fact", "candidate"]):
        cand = facts[fact].candidate(cid)
        any_stake = set(cand.stake_languages())
        non = [l for l in languages if l not in any_stake and l not in exclude_from_comparison]
        for s in cand.stakes:
            for (lens, ts), h in g.groupby(["lens", "token_set"]):
                w = h.pivot_table(index=unit, columns="lang", values="score")
                if s.language not in w.columns:
                    continue
                cols = [l for l in non if l in w.columns]
                if not cols:
                    continue
                d = (w[s.language] - w[cols].mean(axis=1)).dropna()
                t = sign_flip_test(d.to_numpy(), scfg["permutation_exact_max_n"],
                                   scfg["permutation_mc"],
                                   _rng(scfg["seed"], (fact, cid, s.language, s.type, lens, ts)))
                rows.append({"comparison": comparison, "fact": fact, "candidate": cid, "dominant": cand.dominant,
                             "stake_lang": s.language, "stake_type": s.type,
                             "lens": lens, "token_set": ts, "non_stake_langs": ",".join(cols),
                             "n_units": len(d), "mean_d": float(d.mean()) if len(d) else np.nan,
                             "median_d": float(d.median()) if len(d) else np.nan,
                             "frac_positive": float((d > 0).mean()) if len(d) else np.nan,
                             "p": t["p"], "method": t["method"]})
    T = pd.DataFrame(rows)
    if len(T):
        T["p_holm"] = np.nan
        for _, idx in T.groupby(["lens", "token_set", "stake_type"]).groups.items():
            T.loc[idx, "p_holm"] = holm(T.loc[idx, "p"].tolist())
    return T


# ══════════════════════════════════════════════════════════════
# 3. Modelos mistos
# ══════════════════════════════════════════════════════════════

def stake_type_of(facts: Dict, fact: str, cid: str, lang: str) -> str:
    for s in facts[fact].candidate(cid).stakes:
        if s.language == lang:
            return s.type
    return "none"


def mixed_rows(D: pd.DataFrame, facts: Dict, registry, ref: str, metric: str) -> pd.DataFrame:
    y = f"d_{metric}"
    R = D[~D["dominant"] & (D["lang"] != ref)].dropna(subset=[y]).copy()
    R["y"] = R[y]
    R["stake_type"] = [stake_type_of(facts, f, c, l)
                       for f, c, l in zip(R["fact"], R["candidate"], R["lang"])]
    R["base_vs_instruct"] = R["model"].map(
        lambda m: registry.models[m].kind if m in registry.models else "unknown")
    R["formulation"] = R["prompt_type"]
    return R


FIT_METHODS = (None, "lbfgs", "powell")     # None = padrão do statsmodels (bfgs → cg → lbfgs)


def _fit(R: pd.DataFrame, fixed: List[str], name: str) -> Dict:
    import statsmodels.formula.api as smf
    used, dropped = [], []
    for t in fixed:
        (used if R[t].nunique() > 1 else dropped).append(t)
    rhs = " + ".join(
        "C(stake_type, Treatment('none'))" if t == "stake_type" else f"C({t})" for t in used) or "1"
    formula = f"y ~ {rhs}"
    vc = {k: f"0 + C({k})" for k in ("fact", "model") if R[k].nunique() > 1}
    info = {"fit": name, "formula": formula + "".join(f" + (1|{k})" for k in vc),
            "dropped_terms": ",".join(dropped), "n_obs": len(R),
            "n_fact": R["fact"].nunique(), "n_model": R["model"].nunique()}
    if len(R) < 5:
        return {**info, "converged": False, "error": "poucas linhas", "coefs": pd.DataFrame()}
    if vc:
        md = smf.mixedlm(formula, R, groups=np.ones(len(R)), re_formula="0", vc_formula=vc)
    else:
        md = smf.mixedlm(formula, R, groups=np.ones(len(R)))
    res, w, tried, err = None, [], [], None
    # Padrão do statsmodels; se não convergir, lbfgs e depois powell
    for method in FIT_METHODS:
        with warnings.catch_warnings(record=True) as wm:
            warnings.simplefilter("always")
            try:
                r = md.fit(reml=True) if method is None else md.fit(reml=True, method=method)
            except Exception as e:      # registra e tenta o próximo (ex.: matriz singular)
                err = str(e)
                tried.append(method or "default")
                continue
        tried.append(method or "default")
        if res is None or (r.converged and not res.converged):
            res, w = r, wm
        if r.converged:
            break
    info["optimizers_tried"] = ",".join(tried)
    if res is None:
        return {**info, "converged": False, "error": err, "coefs": pd.DataFrame()}
    info["optimizer"] = tried[-1] if res.converged else None
    if not res.converged:
        info["note"] = ("não convergiu com " + "/".join(tried)
                        + (": reportar o ajuste B" if name.startswith("A") else ""))
    ci = res.conf_int()
    fe = res.fe_params.index
    coefs = pd.DataFrame({"term": fe, "coef": res.fe_params.values, "se": res.bse[fe].values,
                          "z": res.tvalues[fe].values, "p": res.pvalues[fe].values,
                          "ci_low": ci.loc[fe, 0].values, "ci_high": ci.loc[fe, 1].values})
    vcs = dict(zip(res.model.exog_vc.names, np.atleast_1d(res.vcomp))) if vc else {}
    return {**info, "converged": bool(res.converged),
            "warnings": "; ".join(sorted({str(x.message)[:120] for x in w})),
            "residual_var": float(res.scale),
            **{f"var_{k}": float(v) for k, v in vcs.items()}, "coefs": coefs}


def mixed_models(D: pd.DataFrame, facts: Dict, registry, ref: str, metric: str,
                 exclude_models=()):
    """
    (coeficientes, resumo) dos ajustes A e B por lens × conjunto.
    `exclude_models`: chaves de modelo fora do ajuste (ex.: sem o Llama 3.1).
    """
    R = mixed_rows(D, facts, registry, ref, metric)
    R = R[~R["model"].isin(list(exclude_models))]
    coefs, info = [], []
    for (lens, ts), g in R.groupby(["lens", "token_set"]):
        fits = [_fit(g, ["stake_type", "format", "base_vs_instruct"], "A_all")]
        comp = g[g["format"] == "completion"]
        if len(comp):
            fits.append(_fit(comp, ["stake_type", "formulation", "base_vs_instruct"], "B_completion"))
        for f in fits:
            c = f.pop("coefs")
            meta = {"lens": lens, "token_set": ts, "metric": f"d_{metric}"}
            info.append({**meta, **f})
            if len(c):
                coefs.append(c.assign(**meta, fit=f["fit"]))
    return (pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame(),
            pd.DataFrame(info))


def mixed_by_model(D: pd.DataFrame, facts: Dict, registry, ref: str, metric: str):
    """
    Efeito de stake por modelo: o ajuste A dentro de cada modelo
    (ΔCS ~ stake_type + format + (1|fact)); termos de um só nível saem
    (base_vs_instruct sempre, format nos modelos base, só completion).
    """
    R = mixed_rows(D, facts, registry, ref, metric)
    coefs, info = [], []
    for (model, lens, ts), g in R.groupby(["model", "lens", "token_set"]):
        f = _fit(g, ["stake_type", "format", "base_vs_instruct"], "A_all")
        c = f.pop("coefs")
        meta = {"model": model, "lens": lens, "token_set": ts, "metric": f"d_{metric}"}
        info.append({**meta, **f})
        if len(c):
            coefs.append(c.assign(**meta, fit=f["fit"]))
    return (pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame(),
            pd.DataFrame(info))


# Llama 3.1 (base e instruct): ajuste de sensibilidade sem ele
NO_LLAMA = ("llama", "llama_base")


# ══════════════════════════════════════════════════════════════
# 4. Interno × comportamental
# ══════════════════════════════════════════════════════════════

def internal_behavioral(D: pd.DataFrame, rates: pd.DataFrame, pooled_types: List[str]) -> pd.DataFrame:
    if rates is None or not len(rates):
        return pd.DataFrame()
    cols = ["cs_zone", "cs_star", "cs_final", "d_cs_zone", "d_cs_star", "p_peak", "p_final",
            "suppression"]
    names = D[D["token_set"] == "names"]
    key = ["model", "lens", "fact", "lang", "candidate"]
    qa = names[names["format"] == "qa"][key + ["dominant"] + cols]
    comp = (names[names["prompt_type"].isin(pooled_types)].groupby(key)[cols].mean()
            .add_suffix("_completion").reset_index())
    beh = rates[rates["kind"] == "all"][["model", "fact", "lang", "candidate", "n_answers",
                                         "primary_rate", "mention_rate", "mean_rank"]]
    out = qa.merge(comp, on=key, how="outer").merge(beh, on=["model", "fact", "lang", "candidate"],
                                                    how="inner")
    return out


# ══════════════════════════════════════════════════════════════
# Orquestração
# ══════════════════════════════════════════════════════════════

def run_statistics(res: Dict[str, pd.DataFrame], config) -> Dict[str, pd.DataFrame]:
    scfg = config.experiment["statistics"]
    ref  = config.reference_language
    out  = {}
    if "metrics" in res and len(res["metrics"]):
        M = res["metrics"]
        out["stats_ci"] = bootstrap_ci(M, scfg, ref)
        # Principal: comparação sem en; apêndice: com en no grupo
        out["stats_stake_tests"] = stake_tests(M, config.fact_specs, config.languages, scfg,
                                               exclude_from_comparison=(ref,))
        out["stats_stake_tests_with_en"] = stake_tests(M, config.fact_specs, config.languages, scfg)
        D, pm = res["delta_cs"], scfg["primary_metric"]
        coefs, info = mixed_models(D, config.fact_specs, config.registry, ref, pm)
        out["stats_mixed_coefs"], out["stats_mixed_fits"] = coefs, info
        coefs, info = mixed_models(D, config.fact_specs, config.registry, ref, pm, exclude_models=NO_LLAMA)
        out["stats_mixed_coefs_no_llama"], out["stats_mixed_fits_no_llama"] = coefs, info
        coefs, info = mixed_by_model(D, config.fact_specs, config.registry, ref, pm)
        out["stats_mixed_coefs_by_model"], out["stats_mixed_fits_by_model"] = coefs, info
    if "delta_cs" in res and "behavioral_rates" in res:
        out["internal_behavioral"] = internal_behavioral(res["delta_cs"], res["behavioral_rates"],
                                                         scfg["pooled_types"])
    return out
