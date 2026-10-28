"""
analyze.py
Única implementação das métricas (Fase 2). Lê as tabelas brutas de
results/lens/*/ (run_lens.py) e grava tabelas novas em
results/metrics/. Os brutos nunca são modificados.

    python analyze.py
    python analyze.py --results-dir results --output-dir results/metrics

Métricas (valores brutos, sem suavização), por prompt × candidato e:
    CS_e(k)        P(e) − P(dominante), para todo candidato não dominante,
                   em todo idioma; NaN se algum dos dois conjuntos ficou
                   sem token IDs (ex.: dominante sem indicador)
    B_e(k)         P(e) no prompt neutro espelhado (mesmo idioma, tipo,
                   paráfrase, camada)
    P_adj, CS_adj  P(e) − B_e e (P(e) − B_e) − (P(dom) − B_dom)
    k*             argmax_k P(e) (primeira ocorrência); depth = k/L
    CS*            CS_e(k*)
    suppression    P_e(k*) − P_e(final)
    cs_zone        média de CS na zona de decisão (camadas k > L·(1 − z),
                   z = config.decision_zone)
    article_*      teste do confundidor de artigo, veredito contra o pico
                   do próprio candidato:
                     dominant_continuation  P'(dom) > P'(e)  (e não dominante)
                     local_recovers         P'(e) ≥ P_e(k*)
                     suppressed             caso contrário
                   P' = última camada do último passo forçado.

ΔCS_e(l) = CS_e(l) − CS_e(en), a partir da média sobre paráfrases de
cada célula (modelo × fato × formato × tipo × conjunto × candidato);
paráfrase i de um idioma não é tradução da paráfrase i de outro.

Saídas (results/metrics/):
    curves.parquet       por camada: P, P_dom, B, CS, CS_adj
    metrics.parquet/csv  por prompt × candidato: escalares acima
    confound.parquet     uma linha por passada forçada × candidato
    curves_mean.parquet  por camada, média das paráfrases, com ΔCS
    delta_cs.parquet/csv por célula × idioma, médias e ΔCS dos escalares
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.experiment_config import ExperimentConfig
from logit_lens.store import load_tables

KEYS  = ["model", "lens", "fact", "lang", "format", "prompt_type", "paraphrase", "token_set"]
GROUP = KEYS + ["candidate"]
CELL  = ["model", "lens", "fact", "format", "prompt_type", "token_set", "candidate"]
VERDICTS = ("dominant_continuation", "local_recovers", "suppressed")


def in_decision_zone(layer, n_layers, zone: float):
    """
    Camadas da zona de decisão: k > L·(1 − zone). Mesma regra nas
    figuras. A folga de 1e-9 evita que erro de ponto flutuante
    (0.7 · 40 = 27.999…) mude a fronteira.
    """
    return np.asarray(layer) > np.asarray(n_layers) * (1 - zone) + 1e-9


def _guard_output(output_dir: str, results_dir: str):
    """Nada é gravado dentro dos resultados brutos (results/lens/)."""
    out, raw = Path(output_dir).resolve(), (Path(results_dir) / "lens").resolve()
    if out == raw or raw in out.parents:
        raise ValueError(f"Recusado: {output_dir} está dentro dos resultados brutos ({raw})")


# ══════════════════════════════════════════════════════════════
# Curvas por camada
# ══════════════════════════════════════════════════════════════

def _dominant_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    d = df[df["dominant"]][KEYS + ["layer", "prob", "n_ids"]]
    return d.rename(columns={"prob": f"{prefix}", "n_ids": f"n_ids_{prefix}"})


def curves(probs: pd.DataFrame, baseline: pd.DataFrame = None) -> pd.DataFrame:
    C = probs.merge(_dominant_cols(probs, "p_dom"), on=KEYS + ["layer"], how="left")
    if baseline is not None and len(baseline):
        B = baseline[GROUP + ["layer", "prob"]].rename(columns={"prob": "b"})
        C = C.merge(B, on=GROUP + ["layer"], how="left")
        Bd = baseline[baseline["dominant"]][KEYS + ["layer", "prob"]].rename(columns={"prob": "b_dom"})
        C = C.merge(Bd, on=KEYS + ["layer"], how="left")
    else:
        C["b"], C["b_dom"] = np.nan, np.nan

    measurable = (C["n_ids"] > 0) & (C["n_ids_p_dom"] > 0) & ~C["dominant"]
    C["cs"]     = np.where(measurable, C["prob"] - C["p_dom"], np.nan)
    C["p_adj"]  = C["prob"] - C["b"]
    C["cs_adj"] = np.where(measurable, C["p_adj"] - (C["p_dom"] - C["b_dom"]), np.nan)
    return C.sort_values(GROUP + ["layer"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# Escalares por prompt × candidato
# ══════════════════════════════════════════════════════════════

def prompt_metrics(C: pd.DataFrame, zone: float) -> pd.DataFrame:
    g    = C.groupby(GROUP, sort=False)
    peak = C.loc[g["prob"].idxmax(), GROUP + ["layer", "depth", "prob", "cs"]].rename(
        columns={"layer": "k_star", "depth": "k_star_depth", "prob": "p_peak", "cs": "cs_star"})
    last = C[C["layer"] == C["n_layers"]][GROUP + [
        "slot_case", "article_blocked", "dominant", "n_layers", "n_ids", "n_ids_p_dom",
        "prob", "p_dom", "b", "cs", "p_adj", "cs_adj"]].rename(columns={
            "prob": "p_final", "p_dom": "p_dom_final", "b": "b_final", "cs": "cs_final",
            "p_adj": "p_adj_final", "cs_adj": "cs_adj_final", "n_ids_p_dom": "n_ids_dom"})
    z = C[in_decision_zone(C["layer"], C["n_layers"], zone)]
    zm = z.groupby(GROUP)[["cs", "cs_adj"]].mean().rename(
        columns={"cs": "cs_zone", "cs_adj": "cs_adj_zone"}).reset_index()
    M = last.merge(peak, on=GROUP).merge(zm, on=GROUP, how="left")
    M["suppression"] = M["p_peak"] - M["p_final"]
    return M


def add_config_info(M: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Tipo do fato e se o idioma é stake do próprio candidato."""
    specs = config.fact_specs
    M = M.copy()
    M["kind"]  = M["fact"].map(lambda f: specs[f].kind)
    M["stake"] = [lang in specs[f].candidate(c).stake_languages()
                  for f, c, lang in zip(M["fact"], M["candidate"], M["lang"])]
    return M


# ══════════════════════════════════════════════════════════════
# Confundidor de artigo
# ══════════════════════════════════════════════════════════════

def confound(forced: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    if forced is None or not len(forced):
        return pd.DataFrame()
    F = forced.copy()
    F["max_step"] = F.groupby(KEYS + ["pass_id"])["force_step"].transform("max")
    F = F[(F["force_step"] == F["max_step"]) & (F["layer"] == F["n_layers"])]
    d = F[F["dominant"]][KEYS + ["pass_id", "prob", "n_ids"]].rename(
        columns={"prob": "p_forced_dom", "n_ids": "n_ids_forced_dom"})
    F = F.merge(d, on=KEYS + ["pass_id"], how="left").rename(
        columns={"prob": "p_forced", "n_ids": "n_ids_forced"})
    F = F.merge(M[GROUP + ["p_peak", "p_final"]], on=GROUP, how="left")

    nd        = ~F["dominant"]
    undefined = (F["n_ids_forced"] == 0) | (nd & ~(F["n_ids_forced_dom"] > 0)) | F["p_peak"].isna()
    verdict   = np.where(undefined, "undefined",
                np.where(nd & (F["p_forced_dom"] > F["p_forced"]), "dominant_continuation",
                np.where(F["p_forced"] >= F["p_peak"], "local_recovers", "suppressed")))
    # Prefixo compartilhado mede o segundo token: sem veredito
    F["verdict"] = np.where(F["forced_kind"] == "shared_prefix", None, verdict)
    cols = GROUP + ["dominant", "forced_kind", "pass_id", "force_step", "forced_suffix",
                    "trigger_token", "p_forced", "p_forced_dom", "n_ids_forced",
                    "n_ids_forced_dom", "p_peak", "p_final", "verdict"]
    return F[cols].reset_index(drop=True)


def add_article_verdict(M: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    M = M.copy()
    if len(conf):
        a = conf[conf["forced_kind"] == "article"][GROUP + [
            "trigger_token", "p_forced", "p_forced_dom", "verdict"]].rename(columns={
                "trigger_token": "article_trigger", "p_forced": "article_p_forced",
                "p_forced_dom": "article_p_forced_dom", "verdict": "article_verdict"})
        M = M.merge(a, on=GROUP, how="left")
    else:
        M[["article_trigger", "article_p_forced", "article_p_forced_dom", "article_verdict"]] = np.nan
    M["article_verdict"] = M["article_verdict"].fillna("not_triggered")
    return M


# ══════════════════════════════════════════════════════════════
# Médias de paráfrases e ΔCS
# ══════════════════════════════════════════════════════════════

def _delta(df: pd.DataFrame, by: list, cols: list, ref: str) -> pd.DataFrame:
    """Para cada coluna c: d_c = c(l) − c(ref), dentro do mesmo `by`."""
    r = df[df["lang"] == ref][by + cols].rename(columns={c: f"{c}__ref" for c in cols})
    out = df.merge(r, on=by, how="left")
    for c in cols:
        out[f"d_{c}"] = out[c] - out[f"{c}__ref"]
    return out.drop(columns=[f"{c}__ref" for c in cols])


def curves_mean(C: pd.DataFrame, ref: str) -> pd.DataFrame:
    g = C.groupby(CELL + ["lang", "layer"])
    m = g.agg(n_layers=("n_layers", "first"), depth=("depth", "first"),
              prob=("prob", "mean"), p_dom=("p_dom", "mean"), b=("b", "mean"),
              cs=("cs", "mean"), cs_adj=("cs_adj", "mean"),
              n_paraphrases=("paraphrase", "nunique")).reset_index()
    return _delta(m, CELL + ["layer"], ["cs", "cs_adj"], ref)


SCALARS = ["cs_final", "cs_star", "cs_zone", "cs_adj_final", "cs_adj_zone",
           "p_peak", "p_final", "p_adj_final", "suppression", "k_star_depth"]


def delta_cs(M: pd.DataFrame, ref: str) -> pd.DataFrame:
    g = M.groupby(CELL + ["lang"])
    m = g[SCALARS].mean()
    m["n_paraphrases"] = g["paraphrase"].nunique()
    m["dominant"] = g["dominant"].first()
    m["kind"] = g["kind"].first()
    m["stake"] = g["stake"].first()
    m = m.reset_index()
    return _delta(m, CELL, ["cs_final", "cs_star", "cs_zone", "cs_adj_final", "cs_adj_zone"], ref)


# ══════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════

def compute_all(tables: Dict[str, pd.DataFrame], config: ExperimentConfig) -> Dict[str, pd.DataFrame]:
    if "probs" not in tables:
        raise FileNotFoundError("sem probs.parquet em results/lens/*/ (rode run_lens.py)")
    C    = curves(tables["probs"], tables.get("baseline"))
    M    = add_config_info(prompt_metrics(C, config.decision_zone), config)
    conf = confound(tables.get("forced"), M)
    M    = add_article_verdict(M, conf)
    ref  = config.reference_language
    return {"curves": C, "metrics": M, "confound": conf,
            "curves_mean": curves_mean(C, ref), "delta_cs": delta_cs(M, ref)}


def main(argv=None):
    config = ExperimentConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=config.results_dir)
    parser.add_argument("--output-dir", default=None,
                        help="padrão: {results-dir}/metrics")
    args = parser.parse_args(argv)
    out = Path(args.output_dir or os.path.join(args.results_dir, "metrics"))
    _guard_output(str(out), args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Lendo {args.results_dir}/lens/*/")
    res = compute_all(load_tables(args.results_dir), config)
    for name, df in res.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"  OK  {out / name}.parquet  ({len(df)} linhas)")
    for name in ("metrics", "delta_cs"):
        res[name].to_csv(out / f"{name}.csv", index=False)
    print(f"\nPróximo passo: python make_figures.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
