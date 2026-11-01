"""
analysis/paper_numbers.py
Números do artigo que não são tabelas da Fase 6, calculados a partir das
tabelas de analyze.py (CPU; nenhum modelo roda). Cada um vira um CSV em
results/metrics/ (2026-11-01):

1. decision_depth          por prompt (curves_mean, nomes): entre os
   decision_depth_summary  candidatos do fato,
       emergence  primeira camada em que algum candidato tem P ≥ 0.01
       half       primeira camada em que o maior P chega à metade do seu
                  valor final
       decision   primeira camada a partir da qual o candidato do topo não
                  muda mais
       winner     candidato do topo na última camada
   (profundidades k/L). Resumo: medianas por modelo × lens × tipo de fato
   (e "all" = todos os modelos) e a fração de prompts em que o vencedor é
   o dominante. Prompt = modelo × lens × fato × formato × formulação ×
   idioma (curva média das paráfrases).
2. final_layer_wins        fração de prompts em que cada candidato vence na
                           última camada, por lens × fato × candidato × idioma
                           (todos os modelos, nomes).
3. stats_nationality_robustness  coeficiente de nacionalidade do ajuste A
   (+ dom_en) por lens, nos nomes, sob: outra métrica (d_cs_adj_zone,
   d_cs_final, d_cs_star), deixando um fato de fora, e cada formulação sozinha
   (F1, F2, F3, QA).
4. internal_behavioral_corr  ρ de Spearman entre CS interno e as taxas
   comportamentais (primária, menção) dos candidatos não dominantes, por
   lens: cs_zone do QA; média de F1–F3; e cs_zone do QA dentro do candidato
   (CS e taxa centrados por modelo × fato × candidato).
5. peak_drop               entre os prompts com p_peak ≥ 0.01, a fração com
                           p_final < 0.5 · p_peak, por papel: local (não
                           dominante de fato disputado), local no idioma de
                           stake, dominante (de fato disputado), controle.
6. narrative_markers       por indicador × idioma: taxa de menção sobre todas
                           as respostas (todos os modelos), a partir de
                           indicator_rates (já sem grafias duplicadas).
"""

import warnings
from typing import Dict

import numpy as np
import pandas as pd

PROMPT = ["model", "lens", "fact", "format", "prompt_type", "lang"]
EMERGENCE_P = 0.01


# ── 1. Profundidade de decisão ──

def prompt_trajectories(curves_mean: pd.DataFrame, facts: Dict):
    """(chave do prompt, camadas, n_layers, nomes dos candidatos, matriz P [camadas × candidatos])."""
    C = curves_mean[curves_mean["token_set"] == "names"]
    alt = C.pivot_table(index=PROMPT + ["layer", "n_layers"], columns="candidate", values="prob")
    dom = C.groupby(PROMPT + ["layer", "n_layers"])["p_dom"].first().rename("__dominant__")
    M = alt.join(dom).reset_index()
    for key, g in M.groupby(PROMPT, sort=True):
        g = g.sort_values("layer")
        names = {c.id for c in facts[key[2]].candidates}
        dom = facts[key[2]].dominant.id
        P = g.drop(columns=PROMPT + ["layer", "n_layers"])
        # Controles: o único candidato é o dominante e já tem a própria coluna
        P = (P.drop(columns="__dominant__") if dom in P.columns and P[dom].notna().any()
             else P.drop(columns=[dom], errors="ignore").rename(columns={"__dominant__": dom}))
        P = P[[c for c in P.columns if c in names and P[c].notna().any()]]
        yield key, g["layer"].to_numpy(), int(g["n_layers"].iloc[0]), list(P.columns), P.to_numpy(float)


def depths(layers, n_layers, names, P) -> Dict:
    mx = np.nanmax(P, axis=1)
    top = np.array(names)[np.nanargmax(np.nan_to_num(P, nan=-np.inf), axis=1)]
    winner = top[-1]
    emerged = mx >= EMERGENCE_P
    half = mx >= 0.5 * mx[-1]
    changed = np.where(top != winner)[0]
    dec = changed[-1] + 1 if len(changed) else 0
    return {"emergence": layers[np.argmax(emerged)] / n_layers if emerged.any() else np.nan,
            "half": layers[np.argmax(half)] / n_layers,
            "decision": layers[dec] / n_layers, "winner": winner}


def decision_depth(curves_mean: pd.DataFrame, facts: Dict) -> pd.DataFrame:
    rows = []
    for key, layers, L, names, P in prompt_trajectories(curves_mean, facts):
        d = depths(layers, L, names, P)
        dom = facts[key[2]].dominant.id
        rows.append({**dict(zip(PROMPT, key)), "kind": facts[key[2]].kind, "n_layers": L, **d,
                     "winner_is_dominant": d["winner"] == dom})
    return pd.DataFrame(rows)


def decision_depth_summary(T: pd.DataFrame) -> pd.DataFrame:
    if not len(T):
        return pd.DataFrame()
    parts = [T, T.assign(model="all")]
    S = (pd.concat(parts).groupby(["lens", "model", "kind"])
         .agg(n_prompts=("winner", "size"), n_no_emergence=("emergence", lambda s: int(s.isna().sum())),
              emergence_median=("emergence", "median"), half_median=("half", "median"),
              decision_median=("decision", "median"), winner_dominant_share=("winner_is_dominant", "mean"))
         .reset_index())
    return S


# ── 2. Vitórias na última camada ──

def final_layer_wins(T: pd.DataFrame, facts: Dict) -> pd.DataFrame:
    rows = []
    for (lens, fact, lang), g in T.groupby(["lens", "fact", "lang"]):
        for c in facts[fact].candidates:
            rows.append({"lens": lens, "fact": fact, "candidate": c.id, "dominant": c.dominant,
                         "lang": lang, "n_prompts": len(g), "win_share": float((g["winner"] == c.id).mean())})
    return pd.DataFrame(rows)


# ── 3. Robustez do coeficiente de nacionalidade ──

def nationality_robustness(D: pd.DataFrame, facts: Dict, registry, ref: str) -> pd.DataFrame:
    from analysis.statistics import _fit, dominant_is_english, mixed_rows

    def fit(R, variant_type, variant, metric):
        R = R.copy()
        R["dom_en"] = [int(dominant_is_english(facts, f)) for f in R["fact"]]
        f = _fit(R, ["stake_type", "format", "base_vs_instruct", "dom_en"], "A_all_domen")
        c = f["coefs"]
        nat = c[c["term"].str.contains("nationality")] if len(c) else c
        row = {"variant_type": variant_type, "variant": variant, "metric": f"d_{metric}",
               "n_obs": f["n_obs"], "n_fact": f["n_fact"], "n_model": f["n_model"],
               "converged": f["converged"], "optimizer": f.get("optimizer"), "formula": f["formula"]}
        if len(nat):
            r = nat.iloc[0]
            row.update(coef=r["coef"], ci_low=r["ci_low"], ci_high=r["ci_high"], p=r["p"])
        return row

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for metric in ("cs_zone", "cs_adj_zone", "cs_final", "cs_star"):
            R = mixed_rows(D, facts, registry, ref, metric)
            R = R[R["token_set"] == "names"]
            for lens, g in R.groupby("lens"):
                rows.append({"lens": lens, **fit(g, "metric", metric, metric)})
                if metric != "cs_zone":
                    continue
                for fact in sorted(g["fact"].unique()):
                    rows.append({"lens": lens, **fit(g[g["fact"] != fact], "leave_one_fact_out", fact, metric)})
                for pt in ("F1", "F2", "F3", "QA"):
                    sub = g[g["prompt_type"] == pt]
                    if len(sub):
                        rows.append({"lens": lens, **fit(sub, "formulation", pt, metric)})
    return pd.DataFrame(rows)


# ── 4. Interno × comportamental ──

def internal_behavioral_corr(J: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr
    if J is None or not len(J):
        return pd.DataFrame()
    J = J[~J["dominant"].astype(bool)]
    rows = []
    for lens, g in J.groupby("lens"):
        for measure, col in (("qa_cs_zone", "cs_zone"), ("completion_cs_zone", "cs_zone_completion")):
            for rate in ("primary_rate", "mention_rate"):
                d = g.dropna(subset=[col, rate])
                rho, p = spearmanr(d[col], d[rate]) if len(d) > 2 else (np.nan, np.nan)
                rows.append({"lens": lens, "measure": measure, "rate": rate, "n_cells": len(d),
                             "spearman_rho": rho, "p": p})
        d = g.dropna(subset=["cs_zone", "primary_rate", "mention_rate"]).copy()
        key = ["model", "fact", "candidate"]
        cs = d["cs_zone"] - d.groupby(key)["cs_zone"].transform("mean")
        for rate in ("primary_rate", "mention_rate"):
            r = d[rate] - d.groupby(key)[rate].transform("mean")
            rho, p = spearmanr(cs, r) if len(d) > 2 else (np.nan, np.nan)
            rows.append({"lens": lens, "measure": "qa_cs_zone_within_candidate", "rate": rate,
                         "n_cells": len(d), "spearman_rho": rho, "p": p})
    return pd.DataFrame(rows)


# ── 5. Pico e queda ──

def peak_drop(M: pd.DataFrame, min_peak: float = 0.01, ratio: float = 0.5) -> pd.DataFrame:
    rows = []
    N = M[(M["token_set"] == "names") & (M["p_peak"] >= min_peak)]
    for lens, g in N.groupby("lens"):
        dropped = g["p_final"] < ratio * g["p_peak"]
        disputed, dom = g["kind"] == "disputed", g["dominant"].astype(bool)
        roles = {"local": disputed & ~dom, "local_stake_language": disputed & ~dom & g["stake"].astype(bool),
                 "dominant": disputed & dom, "control": g["kind"] == "control"}
        for role, m in roles.items():
            rows.append({"lens": lens, "role": role, "n_prompts": int(m.sum()), "n_drop": int((dropped & m).sum()),
                         "drop_share": float(dropped[m].mean()) if m.any() else np.nan,
                         "rule": f"p_peak >= {min_peak}, p_final < {ratio} * p_peak"})
    return pd.DataFrame(rows)


# ── 6. Marcadores de narrativa ──

def narrative_markers(rates: pd.DataFrame) -> pd.DataFrame:
    if rates is None or not len(rates):
        return pd.DataFrame()
    I = rates[rates["given_primary"] == "any"].copy()
    I["n_mentioned"] = I["mention_rate"] * I["n_answers"]
    T = (I.groupby(["fact", "candidate", "indicator_type", "indicator", "lang"])
         .agg(n_answers=("n_answers", "sum"), n_mentioned=("n_mentioned", "sum")).reset_index())
    T["n_mentioned"] = T["n_mentioned"].round().astype(int)
    T["mention_rate"] = T["n_mentioned"] / T["n_answers"]
    return T


# ── Orquestração ──

PAPER_TABLES = ("decision_depth", "decision_depth_summary", "final_layer_wins",
                "stats_nationality_robustness", "internal_behavioral_corr", "peak_drop",
                "narrative_markers")


def run_paper_numbers(res: Dict[str, pd.DataFrame], config) -> Dict[str, pd.DataFrame]:
    out, F = {}, config.fact_specs
    if "curves_mean" in res and len(res["curves_mean"]):
        T = decision_depth(res["curves_mean"], F)
        out["decision_depth"] = T
        out["decision_depth_summary"] = decision_depth_summary(T)
        out["final_layer_wins"] = final_layer_wins(T, F)
    if "delta_cs" in res and len(res["delta_cs"]):
        out["stats_nationality_robustness"] = nationality_robustness(
            res["delta_cs"], F, config.registry, config.reference_language)
    if "internal_behavioral" in res:
        out["internal_behavioral_corr"] = internal_behavioral_corr(res["internal_behavioral"])
    if "metrics" in res and len(res["metrics"]):
        out["peak_drop"] = peak_drop(res["metrics"])
    if "indicator_rates" in res:
        out["narrative_markers"] = narrative_markers(res["indicator_rates"])
    return out
