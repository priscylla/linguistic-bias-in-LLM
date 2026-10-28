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
    lens_agreement.parquet          (se houver Tuned Lens) Tuned vs Logit
    lens_agreement_summary.csv      Lens por modelo × conjunto (Fase 4)

Comportamental (Fase 5; results/behavioral/*/answers.parquet, só com as
formas de superfície confirmadas):
    behavioral_attribution  por resposta: primary, ordem, also_cited
    behavioral_mentions     por resposta × candidato: citado, posição, primário
    behavioral_indicators   por resposta × indicador: citado
    behavioral_rates        por modelo × fato × idioma × kind × candidato:
                            taxa primária, de menção, posição média
    indicator_rates         taxa de menção de cada indicador, e por
                            candidato primário da resposta (ex.: respostas
                            em pt que creditam os Wright e citam o 14-bis)
    empirical_dominant.csv  maioria das atribuições primárias em inglês
                            (todas as respostas de todos os modelos; "none"
                            fora do denominador), agregada e por modelo
    behavioral_report.csv   perguntas de config/behavioral.yaml (report)
    kind = all (greedy + amostras) | greedy | sample

Estatística (Fase 6, analysis/statistics.py):
    stats_ci                IC bootstrap de cs_zone, CS*, supressão, CS final e ΔCS
    stats_stake_tests.csv   stake × não-stake por entidade (permutação, Holm)
    stats_mixed_coefs.csv   modelos mistos A (todas) e B (completion)
    stats_mixed_fits.csv    fórmula, n, convergência, variâncias
    internal_behavioral     métrica interna × taxas comportamentais por célula

Concordância Tuned Lens × Logit Lens (mesmas células):
    pico      |depth(k*_tuned) − depth(k*_logit)| por prompt × candidato
    ΔCS       sinal igual (fração) e magnitude (Pearson, Spearman,
              |diferença| média) de ΔCS na zona de decisão e em k*.
              ΔCS final não entra: a última camada não tem tradutor,
              então é igual nos dois lenses. O idioma de referência
              (ΔCS = 0 por definição) fica fora.
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


RAW_DIRS = ("lens", "behavioral")


def _guard_output(output_dir: str, results_dir: str):
    """Nada é gravado dentro dos resultados brutos (results/lens/, results/behavioral/)."""
    out = Path(output_dir).resolve()
    for d in RAW_DIRS:
        raw = (Path(results_dir) / d).resolve()
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
# Concordância Tuned Lens × Logit Lens
# ══════════════════════════════════════════════════════════════

LENS_A, LENS_B = "logit_lens", "tuned_lens"
AGREE_DELTAS   = ["d_cs_zone", "d_cs_star"]


def _pair(df: pd.DataFrame, keys: list, cols: list) -> pd.DataFrame:
    """Linhas do Logit Lens e do Tuned Lens lado a lado (sufixos _logit, _tuned)."""
    a = df[df["lens"] == LENS_A][keys + cols]
    b = df[df["lens"] == LENS_B][keys + cols]
    return a.merge(b, on=keys, suffixes=("_logit", "_tuned"))


def lens_agreement(M: pd.DataFrame, D: pd.DataFrame, ref: str):
    """
    Returns:
        (por_célula, resumo): por_célula com o pico por prompt ×
        candidato e os ΔCS por célula × idioma; resumo por modelo ×
        conjunto. (None, None) se não há os dois lenses.
    """
    if not {LENS_A, LENS_B} <= set(M["lens"]):
        return None, None
    pk = [k for k in GROUP if k != "lens"]
    peaks = _pair(M, pk, ["k_star_depth"])
    peaks["abs_d_peak_depth"] = (peaks["k_star_depth_tuned"] - peaks["k_star_depth_logit"]).abs()

    ck = [k for k in CELL if k != "lens"] + ["lang"]
    cells = _pair(D[~D["dominant"] & (D["lang"] != ref)], ck, AGREE_DELTAS)

    rows = []
    for (model, ts), g in cells.groupby(["model", "token_set"]):
        p = peaks[(peaks["model"] == model) & (peaks["token_set"] == ts)]
        r = {"model": model, "token_set": ts, "n_prompt_candidates": len(p),
             "median_abs_d_peak_depth": p["abs_d_peak_depth"].median(),
             "mean_abs_d_peak_depth": p["abs_d_peak_depth"].mean(), "n_cells": len(g)}
        for c in AGREE_DELTAS:
            x, y = g[f"{c}_logit"], g[f"{c}_tuned"]
            ok = x.notna() & y.notna()
            x, y = x[ok], y[ok]
            r[f"{c}_n"] = int(ok.sum())
            r[f"{c}_sign_agree"] = float((np.sign(x) == np.sign(y)).mean()) if len(x) else np.nan
            r[f"{c}_pearson"]  = x.corr(y) if len(x) > 2 else np.nan
            r[f"{c}_spearman"] = x.corr(y, method="spearman") if len(x) > 2 else np.nan
            r[f"{c}_mean_abs_diff"] = float((x - y).abs().mean()) if len(x) else np.nan
        rows.append(r)
    per_cell = {"peaks": peaks, "cells": cells}
    return per_cell, pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Comportamental
# ══════════════════════════════════════════════════════════════

BCELL = ["model", "fact", "lang"]


def _kinds(df: pd.DataFrame):
    """(kind, recorte): all, greedy, sample."""
    yield "all", df
    for k in ("greedy", "sample"):
        yield k, df[df["kind"] == k]


def behavioral_rates(ment: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kind, d in _kinds(ment):
        g = d.groupby(BCELL + ["candidate"])
        r = g.agg(n_answers=("mentioned", "size"), primary_rate=("primary", "mean"),
                  mention_rate=("mentioned", "mean"), mean_rank=("rank", "mean"),
                  dominant=("dominant", "first")).reset_index()
        r["kind"] = kind
        out.append(r)
    return pd.concat(out, ignore_index=True)


def none_rates(attr: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kind, d in _kinds(attr):
        r = d.assign(none=d["primary"] == "none").groupby(BCELL).agg(
            n_answers=("none", "size"), none_rate=("none", "mean")).reset_index()
        r["kind"] = kind
        out.append(r)
    return pd.concat(out, ignore_index=True)


def indicator_rates(ind: pd.DataFrame, attr: pd.DataFrame) -> pd.DataFrame:
    """Menção de cada indicador, no total e por candidato primário da resposta."""
    key = ["model", "fact", "lang", "paraphrase", "sample_id"]
    d = ind.merge(attr[key + ["primary"]], on=key, how="left")
    cols = BCELL + ["candidate", "indicator_type", "indicator"]
    tot = d.groupby(cols).agg(n_answers=("mentioned", "size"),
                              mention_rate=("mentioned", "mean")).reset_index()
    tot["given_primary"] = "any"
    by = d.groupby(cols + ["primary"]).agg(n_answers=("mentioned", "size"),
                                            mention_rate=("mentioned", "mean")).reset_index()
    by = by.rename(columns={"primary": "given_primary"})
    return pd.concat([tot, by], ignore_index=True)


def empirical_dominant(attr: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Maioria das atribuições primárias no idioma de referência (pooled e por modelo)."""
    ref = config.reference_language
    d = attr[attr["lang"] == ref]
    rows = []
    for scope, part in [("pooled", d)] + [(m, g) for m, g in d.groupby("model")]:
        for fact, g in part.groupby("fact"):
            named = g[g["primary"] != "none"]
            counts = named["primary"].value_counts()
            top = counts[counts == counts.max()].index.tolist() if len(counts) else []
            winner = top[0] if len(top) == 1 else ("tie" if top else "none")
            configured = config.fact_specs[fact].dominant.id
            rows.append({"scope": scope, "fact": fact, "lang": ref,
                         "empirical_dominant": winner, "configured_dominant": configured,
                         "agrees": winner == configured, "n_answers": len(g),
                         "n_named": len(named), "n_none": int((g["primary"] == "none").sum()),
                         "share": float(counts.max() / len(named)) if len(named) else np.nan,
                         "counts": ";".join(f"{k}={v}" for k, v in counts.items())})
    return pd.DataFrame(rows)


def behavioral_report(ment: pd.DataFrame, b) -> pd.DataFrame:
    """Perguntas de config/behavioral.yaml, por modelo e agregadas, em todos os idiomas."""
    rows = []
    key = ["model", "fact", "lang", "paraphrase", "sample_id"]
    scopes = lambda d: [("pooled", d)] + [(m, g) for m, g in d.groupby("model")]  # noqa: E731
    for q in b.report.get("precedence", []):
        d = ment[ment["fact"] == q["fact"]]
        w = d.pivot_table(index=key, columns="candidate", values="rank", aggfunc="first")
        w = w.reset_index()
        for scope, g in scopes(w):
            for lang, h in g.groupby("lang"):
                a = h[q["first"]].fillna(np.inf)
                c = h[q["second"]].fillna(np.inf)
                cited = (a < np.inf) | (c < np.inf)
                both  = (a < np.inf) & (c < np.inf)
                rows.append({"question": f"precedence:{q['fact']}:{q['first']}<{q['second']}",
                             "scope": scope, "lang": lang, "target_lang": lang == q["lang"],
                             "n_answers": len(h), "n_citing_either": int(cited.sum()),
                             "n_citing_both": int(both.sum()),
                             "value": float((a[cited] < c[cited]).mean()) if cited.any() else np.nan,
                             "value_among_both": float((a[both] < c[both]).mean()) if both.any() else np.nan})
    for q in b.report.get("mention_rate", []):
        d = ment[(ment["fact"] == q["fact"]) & (ment["candidate"] == q["candidate"])]
        for scope, g in scopes(d):
            rates = g.groupby("lang")["mentioned"].agg(["mean", "size"])
            comp = [l for l in q.get("compare", []) if l in rates.index]
            base = float(rates.loc[comp, "mean"].mean()) if comp else np.nan
            for lang in rates.index:
                rows.append({"question": f"mention_rate:{q['fact']}:{q['candidate']}",
                             "scope": scope, "lang": lang, "target_lang": lang == q["lang"],
                             "n_answers": int(rates.loc[lang, "size"]),
                             "value": float(rates.loc[lang, "mean"]),
                             "compare_langs": ",".join(comp), "compare_mean": base,
                             "diff_vs_compare": float(rates.loc[lang, "mean"]) - base})
    return pd.DataFrame(rows)


def compute_behavioral(answers: pd.DataFrame, config: ExperimentConfig, b=None) -> Dict[str, pd.DataFrame]:
    """Extração e taxas. Recusa (UnconfirmedError) sem as formas de superfície confirmadas."""
    from behavior.extraction import Extractor
    from config.loader import load_behavioral, require_confirmed
    b = b or load_behavioral()
    require_confirmed({}, [], behavioral=b)
    ex  = Extractor(config.fact_specs, b, config.languages)
    res = ex.extract(answers.to_dict("records"))
    attr, ment, ind = (pd.DataFrame(res[k]) for k in ("attribution", "mentions", "indicators"))
    return {"behavioral_attribution": attr, "behavioral_mentions": ment,
            "behavioral_indicators": ind,
            "behavioral_rates": behavioral_rates(ment), "behavioral_none": none_rates(attr),
            "indicator_rates": indicator_rates(ind, attr) if len(ind) else pd.DataFrame(),
            "empirical_dominant": empirical_dominant(attr, config),
            "behavioral_report": behavioral_report(ment, b)}


def load_answers(results_dir: str):
    parts = [pd.read_parquet(p) for p in sorted((Path(results_dir) / "behavioral").glob("*/answers.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else None


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
    D    = delta_cs(M, ref)
    out  = {"curves": C, "metrics": M, "confound": conf,
            "curves_mean": curves_mean(C, ref), "delta_cs": D}
    per_cell, summary = lens_agreement(M, D, ref)
    if summary is not None:
        out["lens_agreement_peaks"] = per_cell["peaks"]
        out["lens_agreement"] = per_cell["cells"]
        out["lens_agreement_summary"] = summary
    return out


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

    print(f"Lendo {args.results_dir}/lens/*/*/")
    tables = load_tables(args.results_dir)
    res = compute_all(tables, config) if "probs" in tables else {}
    if not res:
        print("  (sem resultados do lens)")

    answers = load_answers(args.results_dir)
    if answers is not None:
        from config.loader import UnconfirmedError
        try:
            res.update(compute_behavioral(answers, config))
        except UnconfirmedError as e:
            print(f"  ⚠️  comportamental pulado: formas de superfície não confirmadas\n{e}")

    # Estatística (Fase 6): IC bootstrap, teste de stake, modelos mistos,
    # junção interno × comportamental
    from analysis.statistics import run_statistics
    res.update(run_statistics(res, config))
    for name, df in res.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"  OK  {out / name}.parquet  ({len(df)} linhas)")
    for name in ("metrics", "delta_cs", "lens_agreement_summary", "behavioral_rates",
                 "empirical_dominant", "behavioral_report", "stats_stake_tests",
                 "stats_mixed_coefs", "stats_mixed_fits", "internal_behavioral"):
        if name in res:
            res[name].to_csv(out / f"{name}.csv", index=False)
    print("\nPróximo passo: python make_figures.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
