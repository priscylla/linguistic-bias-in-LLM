"""
make_figures.py
Única implementação das figuras e tabelas (Fase 7).

Lê as tabelas de results/metrics/ (analyze.py), e para o heatmap de
top tokens as tabelas brutas layers/token_sets de results/lens/; para a
Fig. 6, reports/tuned_lens/{modelo}_eval.csv. Nada é modificado.

    python make_figures.py                     # tudo
    python make_figures.py --fig fig2
    python make_figures.py --scope main        # só as versões do texto principal

Cada figura sai em PDF e PNG (config figures.formats), na largura de
coluna ou de texto do ACM sigconf, em duas versões: principal (modelos
de figures.main_models) e apêndice (sufixo _all, todos os modelos).
Paleta Okabe-Ito + traço por idioma. Lens das figuras: figures.lens
(Fig. 6 compara Logit e Tuned Lens). Figuras sem dados são puladas.

    fig1  Cultural Divergence Map: CS_e por camada, linhas = (fato,
          candidato), colunas = modelos, eixo y comum por linha, zona
          de decisão, suavização centrada que encolhe nas bordas
          (_w3 e _raw)
    fig2  ΔCS (zona de decisão) por fato × idioma, um ponto por modelo
          com IC bootstrap, idiomas de stake destacados (_qa: formato QA)
    fig3  Pico × camada final (dumbbell), controles ao lado dos disputados
    fig4  CS interno × taxa de atribuição primária (CS* e zona de decisão)
    fig5  Nomes × indicadores: ΔCS de nomes, lugares e artefatos
    fig6  Logit Lens × Tuned Lens: ΔCS, pico, KL held-out por camada
    figA  Top tokens por camada (qualitativo, apêndice)
    tables  tabela de resultados em LaTeX (compacta e completa) + CSV
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import in_decision_zone
from config.experiment_config import ExperimentConfig
from logit_lens.store import load_tables
from plot_utils import smooth_curve, smoothing_suffix

CONFIG = ExperimentConfig()

# Cores do heatmap de tokens (figA)
ENTITY_COLOR_DOMINANT = "#D55E00"   # candidato dominante
ENTITY_COLOR_OTHER    = "#0072B2"   # alternativas
OTHER_TOKEN_COLOR     = "#AAAAAA"
FACT_LABELS  = {}                   # vazio: rótulo vem do config (keyword en)
MODEL_LABELS = {}                   # vazio: rótulo vem do registro (família + kind)
LANG_LABELS  = {}


# ══════════════════════════════════════════════════════════════
# Estilo e utilidades
# ══════════════════════════════════════════════════════════════

def fcfg():
    return CONFIG.experiment["figures"]


def set_style():
    fs = fcfg()["font_size"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": fs, "axes.titlesize": fs + 0.5,
        "axes.labelsize": fs, "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "figure.titlesize": fs + 1,
        "legend.fontsize": fs - 1, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "grid.linestyle": "--", "grid.linewidth": 0.4,
        "lines.linewidth": 1.0, "axes.linewidth": 0.6, "pdf.fonttype": 42,
        "figure.dpi": 100, "savefig.dpi": fcfg()["dpi"], "savefig.bbox": "tight",
    })


def save(fig, base) -> list:
    """Grava em todos os formatos do config. Devolve os caminhos."""
    paths = []
    for ext in fcfg()["formats"]:
        p = f"{base}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    for p in paths:
        print(f"  OK  {p}")
    return paths


def lang_color(lang):
    return fcfg()["language_colors"].get(lang, "#666666")


def lang_dash(lang):
    d = fcfg()["language_dashes"].get(lang, [])
    return (0, tuple(d)) if d else "solid"


def model_style(model):
    """Cor e marcador estáveis por modelo (ordem do registro)."""
    keys = list(CONFIG.models)
    i = keys.index(model) if model in keys else len(keys)
    c, m = fcfg()["model_colors"], fcfg()["model_markers"]
    return c[i % len(c)], m[i % len(m)]


def _label(table, key):
    if table is MODEL_LABELS and key not in table and key in CONFIG.registry.models:
        m = CONFIG.registry[key]
        return f"{m.family}{' Instruct' if m.kind == 'instruct' else ''}"
    if table is LANG_LABELS:
        return table.get(key, key.upper())
    return table.get(key, key)


def model_label(model):
    return _label(MODEL_LABELS, model)


def _fact_label(fact):
    return FACT_LABELS.get(fact, CONFIG.fact_specs[fact].keyword.get("en", fact).capitalize())


def _cand_label(fact, cid):
    return CONFIG.fact_specs[fact].candidate(cid).label


def _short_name(fact, cid):
    """Rótulo sem o primeiro nome ('Alberto Santos-Dumont' -> 'Santos-Dumont')."""
    c = CONFIG.fact_specs[fact].candidate(cid)
    parts = c.label.split(" ")
    return c.label if c.plural or len(parts) < 2 else " ".join(parts[1:])


def _panel_title(fact, cid):
    kind = CONFIG.fact_specs[fact].kind
    return f"{_fact_label(fact)}{' (control)' if kind == 'control' else ''}\n{_cand_label(fact, cid)}"


def figure_models(scope: str, available=None) -> list:
    models = list(fcfg()["main_models"]) if scope == "main" else list(CONFIG.models)
    return [m for m in models if available is None or m in available]


def fig_panels(include_controls=False):
    """(fato, candidato): alternativas dos fatos disputados (+ dominante dos controles)."""
    out = [(k, c.id) for k, f in CONFIG.fact_specs.items() if f.kind == "disputed"
           for c in f.alternatives]
    if include_controls:
        out += [(k, f.dominant.id) for k, f in CONFIG.fact_specs.items() if f.kind == "control"]
    return out


def stake_marks(fact, cid):
    """{idioma: 'N' (nationality) | 'P' (place_of_event)}."""
    return {s.language: ("N" if s.type == "nationality" else "P")
            for s in CONFIG.fact_specs[fact].candidate(cid).stakes}


def _sel(df, **kw):
    if df is None or not len(df):
        return pd.DataFrame()
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k].isin(v) if isinstance(v, (list, tuple, set)) else (df[k] == v)
    return df[m]


def load_metrics(metrics_dir):
    """Tabelas de analyze.py: {nome: DataFrame} (as que existirem)."""
    out = {}
    for p in sorted(Path(metrics_dir).glob("*.parquet")):
        out[p.stem] = pd.read_parquet(p)
    return out


def model_layers(curves, model):
    """Número de camadas do modelo, lido dos próprios dados."""
    if curves is None or not len(curves):
        return None
    d = curves[curves["model"] == model]
    return int(d["n_layers"].iloc[0]) if len(d) else None


def decision_start(n_layers):
    """Primeira camada da zona de decisão (mesma regra de analyze.in_decision_zone)."""
    return next(k for k in range(1, n_layers + 1)
                if in_decision_zone(k, n_layers, CONFIG.decision_zone))


def _grid(n, ncols):
    return int(np.ceil(n / ncols)), ncols


def _pooled(df, cols, keys):
    """Média sobre F1–F3 (formulações agrupadas) por `keys`."""
    pooled = CONFIG.experiment["statistics"]["pooled_types"]
    d = _sel(df, prompt_type=pooled)
    return d.groupby(keys)[cols].mean().reset_index() if len(d) else d


def _lang_legend(langs, extra=()):
    h = [Line2D([0], [0], color=lang_color(l), lw=1.2, linestyle=lang_dash(l),
                label=_label(LANG_LABELS, l)) for l in langs]
    return h + list(extra)


def _model_legend(models):
    out = []
    for m in models:
        c, mk = model_style(m)
        out.append(Line2D([0], [0], marker=mk, color=c, lw=0, markersize=4, label=model_label(m)))
    return out


# ══════════════════════════════════════════════════════════════
# FIG 1 — CULTURAL DIVERGENCE MAP
# ══════════════════════════════════════════════════════════════

def fig1_divergence_map(metrics, out_base, models, window=3):
    cm = _sel(metrics.get("curves_mean"), lens=fcfg()["lens"], format="completion",
              token_set="names")
    if not len(cm):
        print("  –  fig1: sem curves_mean")
        return []
    C = _pooled(cm, ["cs", "n_layers"], ["model", "fact", "candidate", "lang", "layer"])
    models = [m for m in models if m in set(C["model"])]
    panels = [p for p in fig_panels() if len(_sel(C, fact=p[0], candidate=p[1]))]
    if not models or not panels:
        return []
    set_style()
    W = fcfg()["text_width_in"]
    fig, axes = plt.subplots(len(panels), len(models), figsize=(W, 0.85 * len(panels) + 0.7),
                             sharey="row", squeeze=False)
    for r, (fact, cid) in enumerate(panels):
        marks = stake_marks(fact, cid)
        for c, model in enumerate(models):
            ax = axes[r][c]
            d = _sel(C, model=model, fact=fact, candidate=cid)
            if not len(d):
                ax.set_visible(False)
                continue
            n_l = int(d["n_layers"].iloc[0])
            for lang in CONFIG.languages:
                g = d[d["lang"] == lang].sort_values("layer")
                if not len(g) or g["cs"].isna().all():
                    continue
                lw = 1.4 if lang in marks else 0.8
                ax.plot(g["layer"], smooth_curve(g["cs"].to_numpy(), window), color=lang_color(lang),
                        linestyle=lang_dash(lang), lw=lw)
            ax.axhline(0, color="#888", lw=0.4)
            ax.axvspan(decision_start(n_l) - 0.5, n_l + 0.5, color="#000", alpha=0.06, lw=0)
            ax.set_xlim(1, n_l)
            ax.tick_params(length=2, pad=1)
            if r == 0:
                ax.set_title(f"{model_label(model)} ({n_l}L)")
            if c == 0:
                stk = " ".join(f"{l.upper()}·{t}" for l, t in marks.items())
                ax.set_ylabel(f"{_fact_label(fact)}\n{_short_name(fact, cid)}"
                              + (f"\n{stk}" if stk else ""), fontsize=fcfg()["font_size"] - 1.5)
            if r == len(panels) - 1:
                ax.set_xlabel("Layer")
    zone = mpatches.Patch(color="#000", alpha=0.12, label=f"Decision zone (last "
                          f"{int(round(CONFIG.decision_zone * 100))}%)")
    fig.legend(handles=_lang_legend(CONFIG.languages, [zone]), loc="lower center",
               ncol=len(CONFIG.languages) + 1, frameon=False, bbox_to_anchor=(0.5, -0.01))
    smooth = "raw" if window == 1 else f"centered {window}-layer mean"
    fig.suptitle(f"CS_e = P(e) − P(dominant) per layer · F1–F3 completion (9 prompts) · "
                 f"{smooth} · thick = stake language (N nationality, P place)", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98), h_pad=0.3, w_pad=0.3)
    return save(fig, out_base)


# ══════════════════════════════════════════════════════════════
# FIG 2 — ΔCS por fato × idioma
# ══════════════════════════════════════════════════════════════

def _delta_panels(ax, d, models, langs, marks, metric_col="delta"):
    x = np.arange(len(langs))
    for j, l in enumerate(langs):
        if l in marks:
            ax.axvspan(j - 0.5, j + 0.5, color="#F0E442", alpha=0.25, lw=0)
            ax.text(j, 1.0, marks[l], transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=fcfg()["font_size"] - 1.5, fontweight="bold")
    width = 0.7 / max(1, len(models))
    for i, m in enumerate(models):
        c, mk = model_style(m)
        g = d[d["model"] == m].set_index("lang")
        for j, l in enumerate(langs):
            if l not in g.index or pd.isna(g.loc[l, metric_col]):
                continue
            xi = x[j] - 0.35 + width * (i + 0.5)
            y, lo, hi = g.loc[l, metric_col], g.loc[l, f"{metric_col}_low"], g.loc[l, f"{metric_col}_high"]
            ax.errorbar(xi, y, yerr=[[y - lo], [hi - y]], fmt=mk, color=c, ms=2.6, lw=0.6,
                        capsize=0, mec=c)
    ax.axhline(0, color="#888", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in langs])
    ax.set_xlim(-0.5, len(langs) - 0.5)


def fig2_delta_cs(metrics, out_base, models, fmt="completion", group="F1-F3"):
    metric = CONFIG.experiment["statistics"]["primary_metric"]
    ci = _sel(metrics.get("stats_ci"), lens=fcfg()["lens"], format=fmt, token_set="names",
              prompt_group=group, metric=metric)
    if not len(ci) or "delta" not in ci:
        print(f"  –  fig2 ({group}): sem stats_ci")
        return []
    models = [m for m in models if m in set(ci["model"])]
    panels = [p for p in fig_panels() if len(_sel(ci, fact=p[0], candidate=p[1]))]
    if not models or not panels:
        return []
    langs = [l for l in CONFIG.languages if l != CONFIG.reference_language]
    set_style()
    nr, nc = _grid(len(panels), 4)
    fig, axes = plt.subplots(nr, nc, figsize=(fcfg()["text_width_in"], 1.55 * nr + 0.5),
                             sharey=True, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (fact, cid) in zip(axes.flat, panels):
        _delta_panels(ax, _sel(ci, fact=fact, candidate=cid), models, langs, stake_marks(fact, cid))
        ax.set_title(_panel_title(fact, cid), pad=8)
    for r in range(nr):
        axes[r][0].set_ylabel(f"ΔCS vs {CONFIG.reference_language.upper()}")
    shade = mpatches.Patch(color="#F0E442", alpha=0.4, label="stake language (N nationality, P place)")
    fig.legend(handles=_model_legend(models) + [shade], loc="lower center", ncol=len(models) + 1,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    what = "F1–F3 completion (9 prompts)" if group == "F1-F3" else f"{group} ({fmt})"
    fig.suptitle(f"ΔCS_e (decision-zone mean) per language · {what} · 95% bootstrap CI", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return save(fig, out_base)


# ══════════════════════════════════════════════════════════════
# FIG 3 — PICO × FINAL (dumbbell), controles ao lado
# ══════════════════════════════════════════════════════════════

def fig3_peak_vs_final(metrics, out_base, models):
    D = _sel(metrics.get("delta_cs"), lens=fcfg()["lens"], format="completion", token_set="names")
    if not len(D):
        print("  –  fig3: sem delta_cs")
        return []
    P = _pooled(D, ["p_peak", "p_final"], ["model", "fact", "candidate", "lang"])
    models = [m for m in models if m in set(P["model"])]
    panels = [p for p in fig_panels(include_controls=True) if len(_sel(P, fact=p[0], candidate=p[1]))]
    if not models or not panels:
        return []
    langs = CONFIG.languages
    set_style()
    nr, nc = _grid(len(panels), 4)
    fig, axes = plt.subplots(nr, nc, figsize=(fcfg()["text_width_in"], 1.5 * nr + 0.5),
                             sharey=True, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    width = 0.7 / max(1, len(models))
    for ax, (fact, cid) in zip(axes.flat, panels):
        marks = stake_marks(fact, cid)
        for j, l in enumerate(langs):
            if l in marks:
                ax.axvspan(j - 0.5, j + 0.5, color="#F0E442", alpha=0.25, lw=0)
        for i, m in enumerate(models):
            c, mk = model_style(m)
            g = _sel(P, model=m, fact=fact, candidate=cid).set_index("lang")
            for j, l in enumerate(langs):
                if l not in g.index:
                    continue
                xi = j - 0.35 + width * (i + 0.5)
                pk, fn = g.loc[l, "p_peak"], g.loc[l, "p_final"]
                ax.plot([xi, xi], [fn, pk], color=c, lw=0.7, alpha=0.6)
                ax.plot(xi, pk, mk, color=c, ms=2.8)
                ax.plot(xi, fn, mk, color="white", mec=c, ms=2.8, mew=0.6)
        ax.set_title(_panel_title(fact, cid))
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels([l.upper() for l in langs])
        ax.set_ylim(-0.02, 1.02)
    for r in range(nr):
        axes[r][0].set_ylabel("P(e)")
    style = [Line2D([0], [0], marker="o", color="#555", lw=0, ms=3.5, label="peak k*"),
             Line2D([0], [0], marker="o", color="white", mec="#555", lw=0, ms=3.5, label="final layer"),
             mpatches.Patch(color="#F0E442", alpha=0.4, label="stake language")]
    fig.legend(handles=_model_legend(models) + style, loc="lower center", ncol=len(models) + 3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Peak vs final-layer probability of each candidate · F1–F3 completion · "
                 "controls: dominant candidate", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return save(fig, out_base)


# ══════════════════════════════════════════════════════════════
# FIG 4 — interno × comportamental
# ══════════════════════════════════════════════════════════════

def fig4_internal_behavioral(metrics, out_base, models):
    from scipy.stats import spearmanr
    J = _sel(metrics.get("internal_behavioral"), lens=fcfg()["lens"], dominant=False)
    J = _sel(J, model=models) if len(J) else J
    if not len(J):
        print("  –  fig4: sem internal_behavioral")
        return []
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(fcfg()["column_width_in"], 1.75), sharey=True)
    for ax, col, name in zip(axes, ["cs_star", "cs_zone"], ["CS* (peak)", "CS (dec. zone)"]):
        d = J.dropna(subset=[col, "primary_rate"])
        for m in models:
            c, mk = model_style(m)
            for l in CONFIG.languages:
                g = d[(d["model"] == m) & (d["lang"] == l)]
                if len(g):
                    ax.scatter(g[col], g["primary_rate"], s=7, marker=mk, color=lang_color(l),
                               alpha=0.8, lw=0)
        if len(d) > 2:
            rho, p = spearmanr(d[col], d["primary_rate"])
            ax.text(0.03, 0.97, f"ρ = {rho:.2f}\nn = {len(d)}", transform=ax.transAxes,
                    va="top", fontsize=fcfg()["font_size"] - 1)
        ax.axvline(0, color="#888", lw=0.4)
        ax.set_xlabel(f"{name}, QA")
    axes[0].set_ylabel("Primary rate of e")
    handles = [Line2D([0], [0], marker="o", color=lang_color(l), lw=0, ms=3, label=l.upper())
               for l in CONFIG.languages]
    fig.legend(handles=handles + _model_legend(models), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.12), fontsize=fcfg()["font_size"] - 1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save(fig, out_base)


# ══════════════════════════════════════════════════════════════
# FIG 5 — nomes × indicadores
# ══════════════════════════════════════════════════════════════

SETS = [("names", "F1-F3", "names (F1–F3)", "o"), ("indicators", "place", "places", "^"),
        ("indicators", "artifact", "artifacts", "s")]


def fig5_names_vs_indicators(metrics, out_base, models):
    metric = CONFIG.experiment["statistics"]["primary_metric"]
    ci = _sel(metrics.get("stats_ci"), lens=fcfg()["lens"], format="completion", metric=metric)
    ci = _sel(ci, model=models) if len(ci) else ci
    if not len(ci) or "delta" not in ci or "indicators" not in set(ci["token_set"]):
        print("  –  fig5: sem ΔCS de indicadores")
        return []
    panels = [p for p in fig_panels() if len(_sel(ci, fact=p[0], candidate=p[1], token_set="indicators"))]
    if not panels:
        return []
    langs = [l for l in CONFIG.languages if l != CONFIG.reference_language]
    set_style()
    nr, nc = _grid(len(panels), 4)
    fig, axes = plt.subplots(nr, nc, figsize=(fcfg()["text_width_in"], 1.5 * nr + 0.45),
                             sharey=True, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (fact, cid) in zip(axes.flat, panels):
        marks = stake_marks(fact, cid)
        for j, l in enumerate(langs):
            if l in marks:
                ax.axvspan(j - 0.5, j + 0.5, color="#F0E442", alpha=0.25, lw=0)
        for k, (ts, grp, _, mk) in enumerate(SETS):
            d = _sel(ci, fact=fact, candidate=cid, token_set=ts, prompt_group=grp)
            for j, l in enumerate(langs):
                v = d[d["lang"] == l]["delta"].dropna()
                if not len(v):
                    continue
                xi = j - 0.22 + 0.22 * k
                ax.plot([xi, xi], [v.min(), v.max()], color="#555", lw=0.6)   # faixa entre modelos
                ax.plot(xi, v.mean(), mk, color="#000" if k == 0 else "#0072B2", ms=3)
        ax.axhline(0, color="#888", lw=0.4)
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels([l.upper() for l in langs])
        ax.set_title(_panel_title(fact, cid), pad=8)
    for r in range(nr):
        axes[r][0].set_ylabel(f"ΔCS vs {CONFIG.reference_language.upper()}")
    handles = [Line2D([0], [0], marker=mk, color="#000" if k == 0 else "#0072B2", lw=0, ms=3.5,
                      label=lab) for k, (_, _, lab, mk) in enumerate(SETS)]
    handles.append(Line2D([0], [0], color="#555", lw=0.8, label="range across models"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Name-based vs indicator-based ΔCS (decision zone) · mean over models", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    return save(fig, out_base)


# ══════════════════════════════════════════════════════════════
# FIG 6 — Logit Lens × Tuned Lens
# ══════════════════════════════════════════════════════════════

def tuned_eval(report_dir="reports/tuned_lens"):
    parts = []
    for p in sorted(Path(report_dir).glob("*_eval.csv")):
        parts.append(pd.read_csv(p).assign(model=p.name[: -len("_eval.csv")]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fig6_lens_comparison(metrics, out_base, models, report_dir="reports/tuned_lens"):
    cells = _sel(metrics.get("lens_agreement"), token_set="names")
    if not len(cells):
        print("  –  fig6: sem resultados do Tuned Lens")
        return []
    cells = _sel(cells, model=models)
    peaks = _sel(metrics.get("lens_agreement_peaks"), token_set="names", model=models)
    ev = tuned_eval(report_dir)
    ev = _sel(ev, model=models) if len(ev) else ev
    ncols = 3 if len(ev) else 2
    set_style()
    fig, axes = plt.subplots(1, ncols, figsize=(fcfg()["text_width_in"], 2.0))
    ax = axes[0]
    lim = []
    for m in models:
        c, mk = model_style(m)
        g = cells[cells["model"] == m].dropna(subset=["d_cs_zone_logit", "d_cs_zone_tuned"])
        ax.scatter(g["d_cs_zone_logit"], g["d_cs_zone_tuned"], s=6, marker=mk, color=c, lw=0, alpha=0.8)
        lim += list(g["d_cs_zone_logit"]) + list(g["d_cs_zone_tuned"])
    if lim:
        lo, hi = min(lim), max(lim)
        ax.plot([lo, hi], [lo, hi], color="#888", lw=0.5, ls="--")
    ok = cells.dropna(subset=["d_cs_zone_logit", "d_cs_zone_tuned"])
    if len(ok) > 2:
        agree = (np.sign(ok["d_cs_zone_logit"]) == np.sign(ok["d_cs_zone_tuned"])).mean()
        r = ok["d_cs_zone_logit"].corr(ok["d_cs_zone_tuned"])
        ax.text(0.03, 0.97, f"r = {r:.2f}\nsame sign {agree:.0%}", transform=ax.transAxes, va="top")
    ax.set_xlabel("ΔCS, Logit Lens")
    ax.set_ylabel("ΔCS, Tuned Lens")
    ax.set_title("(a) ΔCS (decision zone)")

    ax = axes[1]
    data = [peaks[peaks["model"] == m]["abs_d_peak_depth"].dropna() for m in models]
    ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": "#000"})
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels([model_label(m).split(" ")[0] for m in models])
    ax.set_ylabel("|Δ peak depth| (k*/L)")
    ax.set_title("(b) Peak layer shift")

    if len(ev):
        ax = axes[2]
        for m in models:
            g = ev[ev["model"] == m].groupby("layer")[["kl_logit", "kl_tuned"]].mean().reset_index()
            if not len(g):
                continue
            c, _ = model_style(m)
            depth = g["layer"] / (g["layer"].max() + 1)
            ax.plot(depth, g["kl_logit"], color=c, ls="--", lw=0.8)
            ax.plot(depth, g["kl_tuned"], color=c, lw=0.9)
        ax.set_yscale("log")
        ax.set_xlabel("Depth k/L")
        ax.set_ylabel("Held-out KL to final (nats)")
        ax.set_title("(c) Lens quality")
        ax.legend(handles=[Line2D([0], [0], color="#555", ls="--", lw=0.8, label="Logit Lens"),
                           Line2D([0], [0], color="#555", lw=0.9, label="Tuned Lens")],
                  frameon=False, loc="upper right")
    fig.legend(handles=_model_legend(models), loc="lower center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return save(fig, out_base)


# ══════════════════════════════════════════════════════════════
# FIG A — TOP TOKENS POR CAMADA (qualitativo, apêndice)
# ══════════════════════════════════════════════════════════════

def token_color(token_id, run, fact=None, lang=None):
    """
    Cor do token pelo seu ID exato (nunca por substring): vermelho =
    candidato dominante, azul = candidatos alternativos (ids mantidos
    nos conjuntos do prompt, tabela token_sets).
    """
    if token_id in run.get("dominant_ids", set()):
        return ENTITY_COLOR_DOMINANT
    if token_id in run.get("other_ids", set()):
        return ENTITY_COLOR_OTHER
    return OTHER_TOKEN_COLOR


def heatmap_run(raw, metrics, model, fact, lang, form="F1", paraphrase=1):
    """Camadas (top-k), ids coloridos e métricas de um prompt completion."""
    layers, tsets = raw.get("layers"), raw.get("token_sets")
    if layers is None or not len(layers):
        return {}
    sel = dict(model=model, fact=fact, lang=lang, format="completion",
               prompt_type=form, paraphrase=paraphrase, lens="logit_lens")
    m = pd.Series(True, index=layers.index)
    for k, v in {**sel, "source": "prompt"}.items():
        m &= layers[k] == v
    L = layers[m].sort_values("layer")
    if not len(L):
        return {}
    dom_id = CONFIG.fact_specs[fact].dominant.id
    dom, oth = set(), set()
    if tsets is not None and len(tsets):
        t = pd.Series(True, index=tsets.index)
        for k, v in {**sel, "source": "prompt"}.items():
            t &= tsets[k] == v
        kept = tsets[t & tsets["kept"]]
        dom = set(kept[kept["candidate"] == dom_id]["token_id"].astype(int))
        oth = set(kept[kept["candidate"] != dom_id]["token_id"].astype(int))
    met = metrics.get("metrics")
    if met is not None and len(met):
        mm = met[(met["lens"] == "logit_lens") & (met["model"] == model)
                 & (met["fact"] == fact) & (met["lang"] == lang)
                 & (met["format"] == "completion") & (met["prompt_type"] == form)
                 & (met["paraphrase"] == paraphrase)]
    else:
        mm = pd.DataFrame()
    return {
        "layers": [{"layer": int(r.layer), "depth": float(r.depth),
                    "top_tokens": list(r.top_tokens), "top_ids": [int(i) for i in r.top_ids],
                    "top_probs": [float(x) for x in r.top_probs]} for r in L.itertuples()],
        "dominant_ids": dom, "other_ids": oth,
        "prompt": CONFIG.prompt(fact, lang, form, paraphrase).text,
        "cs_final": {r.candidate: r.cs_final for r in mm.itertuples() if not r.dominant},
    }


def _top_id(layer, col_i):
    ids = layer.get("top_ids", [])
    return ids[col_i] if col_i < len(ids) else None


def _render_logit_lens_ax(ax, run, fact, model, lang, use_absolute):
    """Renderiza o Logit Lens de um único run em um eixo."""
    layers = run.get("layers", [])
    if not layers:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    n_layers = len(layers)
    n_show   = min(20, n_layers)
    top_k    = 5

    idxs = np.linspace(0, n_layers - 1, n_show, dtype=int)
    sel  = [layers[i] for i in idxs]

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("auto"); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cw, ch = 1.0 / top_k, 1.0 / n_show
    layer_labels = []

    for row_i, layer in enumerate(sel):
        y = 1.0 - (row_i + 1) * ch
        layer_labels.append(f"L{layer['layer']}" if use_absolute
                            else f"{layer['depth']:.2f}")
        for col_i in range(min(top_k, len(layer["top_tokens"]))):
            token = layer["top_tokens"][col_i]
            prob  = layer["top_probs"][col_i]
            color = token_color(_top_id(layer, col_i), run, fact, lang)
            alpha = float(np.clip(prob * 6, 0.06, 0.95))
            x = col_i * cw

            ax.add_patch(mpatches.FancyBboxPatch(
                (x + 0.003, y + 0.001), cw - 0.006, ch - 0.002,
                boxstyle="round,pad=0.006", facecolor=color, alpha=alpha,
                edgecolor="white", linewidth=0.4, zorder=2))

            tc = "white" if alpha > 0.5 else "#333"
            ax.text(x + cw / 2, y + ch * 0.62, token.strip()[:8],
                    ha="center", va="center", fontsize=5.5, color=tc,
                    fontweight="bold" if color != OTHER_TOKEN_COLOR else "normal",
                    zorder=3)
            ax.text(x + cw / 2, y + ch * 0.22, f"{prob:.3f}",
                    ha="center", va="center", fontsize=4, color=tc,
                    alpha=0.85, zorder=3)

    ax.set_yticks([1.0 - (i + 0.5) * ch for i in range(n_show)])
    ax.set_yticklabels(layer_labels, fontsize=5.5)
    ax.set_xticks([])

    # Zonas early/mid/late
    for frac in [0.33, 0.67]:
        ax.axhline(1.0 - int(n_show * frac) * ch,
                   color="#888", lw=0.8, linestyle=":", alpha=0.5)

    ax.set_ylabel("Layer" if use_absolute else "Relative Depth", fontsize=7)

    stake_mark = " * stake" if CONFIG.stake(fact, lang) else ""
    cs_txt = "  ".join(f"{c}={v:+.3f}" for c, v in run.get("cs_final", {}).items()
                       if v == v) or "—"
    ax.set_title(
        "%s (%dL) | %s%s\n\"%s\"\nCS_final: %s" % (
            _label(MODEL_LABELS, model), n_layers, _label(LANG_LABELS, lang),
            stake_mark, run.get("prompt", ""), cs_txt),
        fontsize=7.5, fontweight="bold")


def _narrative_label(labels):
    return " / ".join(labels) + " tokens" if labels else "tokens"


def _logit_lens_legend(fig, fact):
    """Legenda: nomes das narrativas tirados dos conjuntos do config."""
    spec = CONFIG.fact_specs[fact]
    dom  = [spec.dominant.label]
    oth  = [c.label for c in spec.alternatives]
    leg = [
        mpatches.Patch(facecolor=ENTITY_COLOR_DOMINANT, alpha=0.8, label=_narrative_label(dom)),
        mpatches.Patch(facecolor=ENTITY_COLOR_OTHER,    alpha=0.8, label=_narrative_label(oth)),
        mpatches.Patch(facecolor=OTHER_TOKEN_COLOR,     alpha=0.5, label="Other tokens"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")


def _version(use_absolute):
    return "Absolute Layer Number" if use_absolute else "Relative Layer Depth (0->1)"


def figA_logit_lens_combined(raw, metrics, fact, output, use_absolute=False, form="F1"):
    """Modelos de config.heatmap_models × (idioma de referência, idioma(s) com stake)."""
    set_style()
    langs  = [CONFIG.reference_language] + CONFIG.stake_languages.get(fact, [])
    models = CONFIG.heatmap_models
    fig, axes = plt.subplots(len(models), len(langs),
                             figsize=(6 * len(langs), 4 * len(models)), squeeze=False)
    fig.suptitle(
        f"Logit Lens — Top Predicted Tokens per Layer\n"
        f"{_fact_label(fact)}  |  Intensity ∝ probability  |  {_version(use_absolute)}",
        fontweight="bold", fontsize=10, y=1.01)
    for r, model in enumerate(models):
        for c, lang in enumerate(langs):
            run = heatmap_run(raw, metrics, model, fact, lang, form)
            _render_logit_lens_ax(axes[r][c], run, fact, model, lang, use_absolute)
    _logit_lens_legend(fig, fact)
    return save_fig(fig, output)


def figA_logit_lens_per_model(raw, metrics, fact, model, output, use_absolute=False, form="F1"):
    """Todos os idiomas de um único modelo (1 linha × idiomas)."""
    set_style()
    langs = CONFIG.languages
    fig, axes = plt.subplots(1, len(langs), figsize=(4.5 * len(langs), 9),
                             sharey=True, squeeze=False)
    fig.suptitle(
        "Logit Lens | %s | %s | All Languages | %s | %s" % (
            _fact_label(fact), _label(MODEL_LABELS, model), form,
            _version(use_absolute)),
        fontweight="bold", fontsize=10, y=1.01)
    for ax, lang in zip(axes[0], langs):
        run = heatmap_run(raw, metrics, model, fact, lang, form)
        _render_logit_lens_ax(ax, run, fact, model, lang, use_absolute)
    _logit_lens_legend(fig, fact)
    return save_fig(fig, output)


def figA_logit_lens_single(raw, metrics, fact, model, lang, output, use_absolute=False, form="F1"):
    """Uma combinação (modelo × idioma) — para inspeção detalhada."""
    set_style()
    run = heatmap_run(raw, metrics, model, fact, lang, form)
    fig, ax = plt.subplots(1, 1, figsize=(5, 9))
    fig.suptitle(
        "Logit Lens | %s | %s | %s | %s | %s" % (
            _fact_label(fact), _label(MODEL_LABELS, model),
            _label(LANG_LABELS, lang), form, _version(use_absolute)),
        fontweight="bold", fontsize=9, y=1.01)
    _render_logit_lens_ax(ax, run, fact, model, lang, use_absolute)
    _logit_lens_legend(fig, fact)
    return save_fig(fig, output)


def figA_all(raw, metrics, output_dir):
    """Heatmaps do apêndice: config.heatmap_models, combinado por fato e por modelo."""
    have = set(raw["layers"]["model"]) if raw.get("layers") is not None else set()
    paths = []
    for fact in CONFIG.facts:
        paths += figA_logit_lens_combined(raw, metrics, fact,
                                          os.path.join(output_dir, f"figA_tokens_{fact}"))
        for model in CONFIG.heatmap_models:
            if model in have:
                paths += figA_logit_lens_per_model(
                    raw, metrics, fact, model, os.path.join(output_dir, f"figA_tokens_{fact}_{model}"))
    return paths




# ══════════════════════════════════════════════════════════════
# TABELA DE RESULTADOS (LaTeX)
# ══════════════════════════════════════════════════════════════

def _majority(rates, none):
    """Atribuição primária majoritária por modelo × fato × idioma (kind all)."""
    r = _sel(rates, kind="all")
    if not len(r):
        return {}
    out = {}
    nr = _sel(none, kind="all").set_index(["model", "fact", "lang"])["none_rate"] if len(_sel(none, kind="all")) else {}
    for key, g in r.groupby(["model", "fact", "lang"]):
        top = g.sort_values(["primary_rate", "candidate"], ascending=[False, True]).iloc[0]
        nrate = nr.get(key, 0.0) if hasattr(nr, "get") else 0.0
        out[key] = ("none", nrate) if nrate > top.primary_rate else (top.candidate, top.primary_rate)
    return out


def results_table(metrics, models, compact: bool) -> pd.DataFrame:
    """Linha = modelo × fato × idioma × candidato não dominante (F1–F3 agrupadas)."""
    D = _sel(metrics.get("delta_cs"), lens=fcfg()["lens"], format="completion", token_set="names",
             dominant=False)
    if not len(D):
        return pd.DataFrame()
    P = _pooled(D, ["k_star_depth", "cs_star", "suppression", "cs_zone", "d_cs_zone"],
                ["model", "fact", "candidate", "lang"])
    P = _sel(P, model=models)
    rates = metrics.get("behavioral_rates")
    maj = _majority(rates, metrics.get("behavioral_none"))
    prim = (_sel(rates, kind="all").set_index(["model", "fact", "lang", "candidate"])["primary_rate"]
            if rates is not None and len(rates) else pd.Series(dtype=float))
    rows = []
    for r in P.itertuples():
        marks = stake_marks(r.fact, r.candidate)
        if compact and r.lang not in set(marks) | {CONFIG.reference_language}:
            continue
        k = (r.model, r.fact, r.lang, r.candidate)
        mj = maj.get(k[:3])
        rows.append({
            "Model": model_label(r.model), "Fact": _fact_label(r.fact),
            "Candidate": _cand_label(r.fact, r.candidate),
            "Lang": r.lang.upper() + ("$^{" + marks[r.lang] + "}$" if r.lang in marks else ""),
            "Peak $k^*/L$": r.k_star_depth, "CS$^*$": r.cs_star, "Supp.": r.suppression,
            "$\\Delta$CS": r.d_cs_zone,
            "Primary $e$": prim.get(k, np.nan),
            "Majority": ("—" if mj is None else
                         ("none" if mj[0] == "none" else _cand_label(r.fact, mj[0])) + f" ({mj[1]:.0%})"),
            "_order": (list(CONFIG.models).index(r.model) if r.model in CONFIG.models else 99,
                       list(CONFIG.fact_specs).index(r.fact), r.candidate,
                       CONFIG.languages.index(r.lang)),
        })
    T = pd.DataFrame(rows)
    if len(T):
        T = T.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return T


def _tex_escape(s: str) -> str:
    if "$" in s:          # já é LaTeX (marcas de stake, cabeçalhos)
        return s
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def to_latex(T: pd.DataFrame, caption: str, label: str, longtable: bool) -> str:
    """Tabela booktabs (sem dependências); longtable para o apêndice."""
    cols = list(T.columns)
    align = "llll" + "r" * (len(cols) - 5) + "l"
    head = " & ".join(_tex_escape(c) for c in cols) + r" \\"
    signed = {"CS$^*$", "$\\Delta$CS"}

    def cell(col, v):
        if isinstance(v, float):
            if np.isnan(v):
                return "—"
            t = f"{v:+.2f}" if col in signed else f"{v:.2f}"
            return t.replace("-", "$-$", 1) if t.startswith("-") else t
        return _tex_escape(str(v))

    body = [" & ".join(cell(c, v) for c, v in zip(cols, row)) + r" \\"
            for row in T.itertuples(index=False)]
    note = (r"\multicolumn{%d}{l}{\footnotesize $^N$ nationality stake, $^P$ place-of-event stake; "
            r"F1--F3 completion prompts pooled; $\Delta$CS = decision-zone CS minus EN.}\\" % len(cols))
    if longtable:
        return "\n".join([r"\begin{small}", r"\begin{longtable}{%s}" % align,
                          r"\caption{%s}\label{%s}\\" % (caption, label), r"\toprule", head,
                          r"\midrule", r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead",
                          *body, r"\bottomrule", note, r"\end{longtable}", r"\end{small}", ""])
    return "\n".join([r"\begin{table*}[t]", r"\centering", r"\small",
                      r"\caption{%s}\label{%s}" % (caption, label),
                      r"\begin{tabular}{%s}" % align, r"\toprule", head, r"\midrule", *body,
                      r"\bottomrule", note, r"\end{tabular}", r"\end{table*}", ""])


def write_tables(metrics, tables_dir) -> list:
    out = Path(tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, models, compact, longt, cap in [
        ("results_compact", figure_models("main"), True, False,
         "Internal commitment and behavior for each alternative candidate: stake languages and EN."),
        ("results_full", figure_models("all"), False, True,
         "Internal commitment and behavior, all models, facts, languages and alternative candidates."),
    ]:
        T = results_table(metrics, models, compact)
        if not len(T):
            print(f"  –  {name}: sem dados")
            continue
        (out / f"{name}.tex").write_text(to_latex(T, cap, f"tab:{name}", longt), encoding="utf-8")
        T.to_csv(out / f"{name}.csv", index=False)
        paths += [str(out / f"{name}.tex"), str(out / f"{name}.csv")]
        print(f"  OK  {out / name}.tex  ({len(T)} linhas)")
    return paths


def save_fig(fig, path):
    """Compatível com caminhos com extensão (heatmaps): grava em todos os formatos."""
    base = os.path.splitext(str(path))[0]
    fig.tight_layout()
    return save(fig, base)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

FIGURES = ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "figA", "tables"]


def make_all(metrics, figs, out_dir, scopes, results_dir=None, tables_dir=None,
             report_dir="reports/tuned_lens") -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for scope in scopes:
        sfx = "" if scope == "main" else "_all"
        models = figure_models(scope)
        if "fig1" in figs:
            for w in CONFIG.smoothing_windows:
                paths += fig1_divergence_map(metrics, out_dir / f"fig1_divergence_map{sfx}{smoothing_suffix(w)}",
                                             models, window=w)
        if "fig2" in figs:
            paths += fig2_delta_cs(metrics, out_dir / f"fig2_delta_cs{sfx}", models)
            paths += fig2_delta_cs(metrics, out_dir / f"fig2_delta_cs_qa{sfx}", models, "qa", "QA")
        if "fig3" in figs:
            paths += fig3_peak_vs_final(metrics, out_dir / f"fig3_peak_vs_final{sfx}", models)
        if "fig4" in figs:
            paths += fig4_internal_behavioral(metrics, out_dir / f"fig4_internal_vs_behavior{sfx}", models)
        if "fig5" in figs:
            paths += fig5_names_vs_indicators(metrics, out_dir / f"fig5_names_vs_indicators{sfx}", models)
        if "fig6" in figs:
            paths += fig6_lens_comparison(metrics, out_dir / f"fig6_lens_comparison{sfx}", models, report_dir)
    if "figA" in figs and results_dir is not None:
        raw = load_tables(results_dir, ("layers", "token_sets"))
        if raw:
            paths += figA_all(raw, metrics, str(out_dir))
    if "tables" in figs:
        paths += write_tables(metrics, tables_dir or CONFIG.experiment.get("tables_dir", "tables"))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=CONFIG.results_dir)
    parser.add_argument("--metrics-dir", default=None, help="padrão: {results-dir}/metrics")
    parser.add_argument("--output-dir",  default=CONFIG.figures_dir)
    parser.add_argument("--tables-dir",  default=CONFIG.experiment.get("tables_dir", "tables"))
    parser.add_argument("--report-dir",  default="reports/tuned_lens")
    parser.add_argument("--fig", default="all", choices=FIGURES + ["all"])
    parser.add_argument("--scope", default="both", choices=["main", "all", "both"])
    args = parser.parse_args(argv)

    metrics_dir = args.metrics_dir or os.path.join(args.results_dir, "metrics")
    print(f"Carregando: {metrics_dir}")
    metrics = load_metrics(metrics_dir)
    figs = FIGURES if args.fig == "all" else [args.fig]
    scopes = ["main", "all"] if args.scope == "both" else [args.scope]
    make_all(metrics, figs, args.output_dir, scopes, args.results_dir, args.tables_dir,
             args.report_dir)
    print(f"\nFiguras em {args.output_dir}/, tabelas em {args.tables_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
