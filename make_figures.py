"""
make_figures.py
Única implementação das figuras.

Lê as métricas de results/metrics/ (analyze.py) e, para o heatmap de
top tokens, as tabelas brutas layers/token_sets de results/lens/*/.
Nada é modificado.

Fatos, candidatos, idiomas, modelos, stake, zona de decisão e janelas
de suavização vêm do config; o número de camadas vem dos dados. Cada
candidato não dominante e de um fato disputado tem o seu painel
(CS_e = P(e) − P(dominante)). Versão mínima da Fase 2; a Fase 7
refaz as figuras.

Uso:
    python make_figures.py
    python make_figures.py --fig fig1
    python make_figures.py --output-dir figures

Figuras:
    fig1  Cultural Divergence Map (uma versão por janela: _w3, _raw)
    fig3  Logit Lens — top tokens por camada (qualitativo)
    fig6  Peak vs. final layer (dumbbell)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

from analyze import in_decision_zone
from config.experiment_config import ExperimentConfig
from logit_lens.store import load_tables
from plot_utils import smooth_curve, smoothing_suffix

CONFIG = ExperimentConfig()

# ── Estilo (só apresentação; paleta definitiva na Fase 7) ─────
LANG_COLORS  = {"pt": "#2166AC", "en": "#D6604D", "de": "#4DAC26", "it": "#8E0152"}
LANG_LABELS  = {"pt": "Portuguese (BR)", "en": "English (US)", "de": "German", "it": "Italian"}
LANG_DASHES  = {"pt": [], "en": [6, 3], "de": [2, 4], "it": [8, 3, 2, 3]}
MODEL_COLORS = {"llama": "#E69F00", "mistral": "#56B4E9", "qwen": "#009E73", "gemma": "#CC79A7"}
MODEL_LABELS = {}   # vazio: rótulo vem do registro (família + kind)
FACT_LABELS  = {"aviao": "Airplane", "telefone": "Telephone"}
_FALLBACK    = ["#666666", "#999933", "#882255", "#44AA99", "#332288", "#AA4499"]

# Cores das entidades no heatmap
ENTITY_COLOR_DOMINANT = "#D6604D"   # vermelho — narrativa dominante
ENTITY_COLOR_OTHER    = "#2166AC"   # azul — a outra narrativa
OTHER_TOKEN_COLOR     = "#AAAAAA"


def _color(table, key, i=0):
    return table.get(key, _FALLBACK[i % len(_FALLBACK)])


def _label(table, key):
    if table is MODEL_LABELS and key not in table and key in CONFIG.registry.models:
        m = CONFIG.registry[key]
        return f"{m.family}{' Instruct' if m.kind == 'instruct' else ''}"
    return table.get(key, key)


def _dashes(lang):
    d = LANG_DASHES.get(lang, [])
    return (0, tuple(d)) if d else "solid"


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "grid.linestyle": "--",
        "lines.linewidth": 1.8, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def load_metrics(metrics_dir):
    """Tabelas de analyze.py: {nome: DataFrame}."""
    out = {}
    for name in ("curves_mean", "delta_cs", "metrics"):
        p = Path(metrics_dir) / f"{name}.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
    return out


# Recorte das figuras: completion, F1, nomes
FIG_FILTER = {"format": "completion", "prompt_type": "F1", "token_set": "names"}


def _select(df, **kw):
    if df is None or not len(df):
        return df
    m = pd.Series(True, index=df.index)
    for k, v in {**FIG_FILTER, **kw}.items():
        m &= df[k] == v
    return df[m]


def fig_panels():
    """(fato, candidato) de cada candidato não dominante de fatos disputados."""
    return [(k, c.id) for k, f in CONFIG.fact_specs.items() if f.kind == "disputed"
            for c in f.alternatives]


def _fact_label(fact):
    return FACT_LABELS.get(fact, CONFIG.fact_specs[fact].keyword.get("en", fact).capitalize())


def _cand_label(fact, cid):
    return CONFIG.fact_specs[fact].candidate(cid).label


def save_fig(fig, path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  OK  {path}")


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


def _stake_langs(fact, cid=None):
    if cid is None:
        return CONFIG.stake_languages.get(fact, [])
    return CONFIG.fact_specs[fact].candidate(cid).stake_languages()


# ══════════════════════════════════════════════════════════════
# FIG 1 — CULTURAL DIVERGENCE MAP
# Layout: linhas = fatos × colunas = modelos
# Eixo X: número absoluto da camada
# ══════════════════════════════════════════════════════════════

def fig1_cultural_divergence_map(metrics, output, window=3):
    set_style()
    curves, delta = metrics.get("curves_mean"), metrics.get("delta_cs")
    panels, models, langs = fig_panels(), list(CONFIG.models), CONFIG.languages
    zone_pct = int(round(CONFIG.decision_zone * 100))

    fig, axes = plt.subplots(len(panels), len(models),
                             figsize=(3.5 * len(models), 3.0 * len(panels)),
                             sharey=False, squeeze=False)
    smooth_txt = "raw" if window == 1 else f"smoothed (window {window})"
    fig.suptitle(
        "Cultural Divergence Map\n"
        "CS_e = P(e) − P(dominant)  │  F1 completion, mean of paraphrases  │  "
        f"Absolute layer number  │  {smooth_txt}",
        fontweight="bold", fontsize=10, y=1.01
    )

    for row, (fact, cid) in enumerate(panels):
        for col, model in enumerate(models):
            ax  = axes[row][col]
            n_l = model_layers(curves, model)
            if n_l is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            if row == 0:
                ax.set_title(f"{_label(MODEL_LABELS, model)}\n({n_l} layers)",
                             color=_color(MODEL_COLORS, model, col),
                             fontweight="bold", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"{_fact_label(fact)}: {_cand_label(fact, cid)}\nCS_e",
                              fontsize=8, fontweight="bold")
            if row == len(panels) - 1:
                ax.set_xlabel("Layer", fontsize=8)

            for j, lang in enumerate(langs):
                d = _select(curves, model=model, fact=fact, candidate=cid, lang=lang)
                d = d.sort_values("layer") if d is not None else d
                if d is None or not len(d) or d["cs"].isna().all():
                    continue

                x = d["layer"].to_numpy()
                y = smooth_curve(d["cs"].to_numpy(), window)
                c = _color(LANG_COLORS, lang, j)
                ax.plot(x, y, color=c, linewidth=1.8, linestyle=_dashes(lang),
                        alpha=0.9, label=_label(LANG_LABELS, lang))
                ax.fill_between(x, y, 0, color=c, alpha=0.06)

                s_ = _select(delta, model=model, fact=fact, candidate=cid, lang=lang)
                cs_peak = float(s_["cs_star"].iloc[0]) if s_ is not None and len(s_) else 0.0
                if abs(cs_peak) > 0.05:
                    ax.annotate(f"{cs_peak:+.2f}", xy=(x[-1], y[-1]),
                                xytext=(x[-1] - n_l * 0.13, y[-1]),
                                fontsize=6, color=c, fontweight="bold", va="center")

            ax.axhline(0, color="gray", lw=0.7, linestyle="--", alpha=0.6)

            d0 = decision_start(n_l)
            ax.axvspan(d0, n_l, alpha=0.05, color="gray")
            ax.text(d0 + 0.5,
                    ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0.05 else 0.05,
                    f"L{d0}", fontsize=6, color="gray", alpha=0.7, va="top")

            stakes = _stake_langs(fact, cid)
            if stakes:
                ax.text(0.02, 0.97, "stake: " + ", ".join(s.upper() for s in stakes),
                        transform=ax.transAxes, fontsize=6.5,
                        color=_color(LANG_COLORS, stakes[0]),
                        va="top", fontstyle="italic")

            step = max(4, n_l // 8)
            ax.set_xticks(range(1, n_l + 1, step))
            ax.set_xlim(1, n_l)

    leg = [Line2D([0], [0], color=_color(LANG_COLORS, l, j), lw=2,
                  linestyle=_dashes(l), label=_label(LANG_LABELS, l))
           for j, l in enumerate(langs)]
    leg += [
        mpatches.Patch(facecolor="gray", alpha=0.15,
                       label=f"Decision zone (last {zone_pct}%)"),
        Line2D([0], [0], color="gray", lw=1, linestyle="--", label="CS_e = 0"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output)


# ══════════════════════════════════════════════════════════════
# FIG 3 — LOGIT LENS: TOP TOKENS POR CAMADA (qualitativo)
# Cores por token ID exato: vermelho = narrativa dominante,
# azul = a outra narrativa, cinza = demais tokens.
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
               prompt_type=form, paraphrase=paraphrase)
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
        mm = met[(met["model"] == model) & (met["fact"] == fact) & (met["lang"] == lang)
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


def fig3_logit_lens_combined(raw, metrics, fact, output, use_absolute=False, form="F1"):
    """Modelos de config.heatmap_models × (idioma de referência, idioma(s) com stake)."""
    set_style()
    langs  = [CONFIG.reference_language] + _stake_langs(fact)
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
    save_fig(fig, output)


def fig3_logit_lens_per_model(raw, metrics, fact, model, output, use_absolute=False, form="F1"):
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
    save_fig(fig, output)


def fig3_logit_lens_single(raw, metrics, fact, model, lang, output, use_absolute=False, form="F1"):
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
    save_fig(fig, output)


def fig3_all(raw, metrics, output_dir):
    """Gera todas as variantes do heatmap (relativo e absoluto)."""
    models = set(raw["layers"]["model"]) if raw.get("layers") is not None else set()
    for fact in CONFIG.facts:
        for abs_ in (False, True):
            sfx = "_abs" if abs_ else ""
            fig3_logit_lens_combined(
                raw, metrics, fact, os.path.join(output_dir, f"fig3_logit_lens_{fact}{sfx}.png"), abs_)
            for model in CONFIG.models:
                if model not in models:
                    continue
                fig3_logit_lens_per_model(
                    raw, metrics, fact, model,
                    os.path.join(output_dir, f"fig3_logit_lens_{fact}_{model}{sfx}.png"), abs_)
                for lang in CONFIG.languages:
                    fig3_logit_lens_single(
                        raw, metrics, fact, model, lang,
                        os.path.join(output_dir,
                                     f"fig3_logit_lens_{fact}_{model}_{lang}{sfx}.png"), abs_)


# ══════════════════════════════════════════════════════════════
# FIG 6 — PEAK vs FINAL LAYER (Dumbbell chart)
#   círculo cheio = P_e(k*) (o que o modelo sabe internamente)
#   quadrado vazio = P_e(final) (o que a última camada gera)
#   linha vertical = suppression gap
#   Um painel por candidato não dominante; médias das paráfrases.
# ══════════════════════════════════════════════════════════════

def fig6_peak_vs_final(metrics, output):
    set_style()
    delta = metrics.get("delta_cs")
    panels, models, langs = fig_panels(), list(CONFIG.models), CONFIG.languages
    ncol = min(4, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.5 * nrow), squeeze=False)
    fig.suptitle("Peak vs. Final Layer — P(e)  |  F1 completion, mean of paraphrases",
                 fontweight="bold", fontsize=10, y=1.02)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    for ax, (fact, cid) in zip(axes.flat, panels):
        ax.set_title(f"{_fact_label(fact)}: {_cand_label(fact, cid)}",
                     fontweight="bold", fontsize=10)

        x      = np.arange(len(langs))
        width  = 0.72 / max(1, len(models))
        offset = -(len(models) - 1) / 2 * width

        for i, model in enumerate(models):
            mc = _color(MODEL_COLORS, model, i)
            peaks, finals = [], []
            for lang in langs:
                d = _select(delta, model=model, fact=fact, candidate=cid, lang=lang)
                ok = d is not None and len(d)
                peaks.append(float(d["p_peak"].iloc[0]) if ok else np.nan)
                finals.append(float(d["p_final"].iloc[0]) if ok else np.nan)

            xpos = x + offset + i * width
            for xi, pk, fn in zip(xpos, peaks, finals):
                if not (pk == pk and fn == fn):
                    continue
                if pk > 0.01:
                    ax.plot([xi, xi], [fn, pk], color=mc, lw=1.8, alpha=0.45, zorder=2)
                ax.plot(xi, pk, "o", color=mc, markersize=8, alpha=0.92,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=4)
                ax.plot(xi, fn, "s", color="white", markersize=7,
                        markeredgecolor=mc, markeredgewidth=2.0, zorder=5)
                sup = pk - fn
                if sup > 0.18 and pk > 0.12:
                    ax.text(xi + 0.025, (pk + fn) / 2, f"{sup:.2f}",
                            fontsize=5.5, color=mc, va="center", ha="left",
                            fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([_label(LANG_LABELS, l) for l in langs],
                           fontsize=8, rotation=12, ha="right")
        ax.set_ylabel("P(e)", fontsize=8.5)
        ax.set_ylim(-0.02, 1.08)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.4)

        for stake_lang in _stake_langs(fact, cid):
            if stake_lang in langs:
                j = langs.index(stake_lang)
                ax.axvspan(j - 0.5, j + 0.5, alpha=0.06, color="gold", zorder=0)
                ax.text(j, 1.04, "stake", ha="center", fontsize=7,
                        color="goldenrod", fontweight="bold")

    model_leg = [Line2D([0], [0], marker="o", color="w",
                        markerfacecolor=_color(MODEL_COLORS, m, i),
                        markeredgecolor="white", markersize=9,
                        label=_label(MODEL_LABELS, m))
                 for i, m in enumerate(models)]
    style_leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="white", markersize=9, label="Peak layer (filled circle)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor="#888", markeredgewidth=2, markersize=9,
               label="Final layer (open square)"),
        Line2D([0], [0], color="#888", lw=1.5, alpha=0.5, label="Suppression gap"),
        mpatches.Patch(facecolor="gold", alpha=0.25, label="Candidate's stake language"),
    ]
    fig.legend(handles=model_leg + style_leg, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.10 / nrow), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

FIGURES = ["fig1", "fig3", "fig6"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=CONFIG.results_dir)
    parser.add_argument("--metrics-dir", default=None,
                        help="padrão: {results-dir}/metrics")
    parser.add_argument("--output-dir",  default=CONFIG.figures_dir)
    parser.add_argument("--fig", default="all", choices=FIGURES + ["all"])
    args = parser.parse_args(argv)

    metrics_dir = args.metrics_dir or os.path.join(args.results_dir, "metrics")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    figs = FIGURES if args.fig == "all" else [args.fig]

    print(f"Carregando: {metrics_dir}")
    metrics = load_metrics(metrics_dir)

    if "fig1" in figs:
        # Uma versão por janela de suavização (_w3, _raw)
        for w in CONFIG.smoothing_windows:
            fig1_cultural_divergence_map(
                metrics, os.path.join(args.output_dir, f"fig1{smoothing_suffix(w)}.png"),
                window=w)
    if "fig3" in figs:
        fig3_all(load_tables(args.results_dir, ("layers", "token_sets")), metrics,
                 args.output_dir)
    if "fig6" in figs:
        fig6_peak_vs_final(metrics, os.path.join(args.output_dir, "fig6.png"))

    print(f"\nFiguras em {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
