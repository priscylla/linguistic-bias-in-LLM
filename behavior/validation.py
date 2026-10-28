"""
behavior/validation.py
Validação manual da extração (Fase 5).

Amostra estratificada: `per_cell` respostas aleatórias (semente fixa)
por modelo × idioma, para que todo idioma e modelo apareça. O CSV tem
colunas vazias para o rótulo humano:

    human_primary  id do candidato citado primeiro, ou "none"
    human_cited    ids de todos os candidatos citados, separados por vírgula

Concordância (automático × humano), só nas linhas rotuladas:
    primário   concordância simples (IC 95% de Wilson) e kappa de Cohen
    citados    por resposta × candidato (citado ou não): kappa, precisão
               e revocação do automático contra o humano
    por idioma concordância do primário
"""

import math
from typing import Dict, List

import numpy as np
import pandas as pd

LABEL_COLS = ["human_primary", "human_cited", "notes"]


def stratified_sample(answers: pd.DataFrame, attr: pd.DataFrame, per_cell: int, seed: int,
                      candidates: Dict[str, str]) -> pd.DataFrame:
    """answers × atribuição automática, `per_cell` por modelo × idioma."""
    key = ["model", "fact", "lang", "paraphrase", "sample_id"]
    d = answers.merge(attr[key + ["primary", "order"]], on=key, how="left")
    parts = []
    for i, ((m, l), g) in enumerate(sorted(d.groupby(["model", "lang"]), key=lambda x: x[0])):
        parts.append(g.sample(n=min(per_cell, len(g)), random_state=seed + i))
    s = pd.concat(parts, ignore_index=True)
    s.insert(0, "uid", [f"{r.model}|{r.fact}|{r.lang}|{r.paraphrase}|{r.sample_id}"
                        for r in s.itertuples()])
    out = pd.DataFrame({
        "uid": s["uid"], "model": s["model"], "fact": s["fact"], "lang": s["lang"],
        "paraphrase": s["paraphrase"], "sample_id": s["sample_id"], "kind": s["kind"],
        "question": s["question"], "answer": s["text"],
        "candidates": s["fact"].map(candidates),
        "auto_primary": s["primary"], "auto_cited": s["order"].fillna("")})
    for c in LABEL_COLS:
        out[c] = ""
    return out


def _ids(x) -> set:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return set()
    return {t.strip() for t in str(x).split(",") if t.strip() and t.strip() != "none"}


def cohen_kappa(a: List, b: List) -> float:
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def agreement(df: pd.DataFrame, fact_candidates: Dict[str, List[str]]) -> Dict:
    lab = df[df["human_primary"].fillna("").astype(str).str.strip() != ""].copy()
    out = {"n_rows": len(df), "n_labelled": len(lab)}
    if not len(lab):
        return out
    auto = lab["auto_primary"].astype(str).str.strip().tolist()
    hum  = lab["human_primary"].astype(str).str.strip().tolist()
    k = sum(x == y for x, y in zip(auto, hum))
    out["primary"] = {"agreement": k / len(lab), "ci95": wilson(k, len(lab)),
                      "kappa": cohen_kappa(auto, hum)}
    out["primary_by_lang"] = {
        l: {"n": len(g), "agreement": float((g["auto_primary"].astype(str).str.strip()
                                             == g["human_primary"].astype(str).str.strip()).mean())}
        for l, g in lab.groupby("lang")}

    a_bin, h_bin = [], []
    for r in lab.itertuples():
        A, H = _ids(r.auto_cited), _ids(r.human_cited)
        for c in fact_candidates[r.fact]:
            a_bin.append(c in A)
            h_bin.append(c in H)
    a_bin, h_bin = np.array(a_bin), np.array(h_bin)
    tp = int((a_bin & h_bin).sum())
    out["cited"] = {"n_pairs": len(a_bin), "kappa": cohen_kappa(a_bin.tolist(), h_bin.tolist()),
                    "agreement": float((a_bin == h_bin).mean()),
                    "precision": tp / a_bin.sum() if a_bin.sum() else float("nan"),
                    "recall": tp / h_bin.sum() if h_bin.sum() else float("nan")}
    return out
