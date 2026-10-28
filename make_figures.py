"""
make_figures.py
Única implementação das figuras (substitui generate_paper_figures.py
e generate_visualizations.py).

Lê results/metrics_computed.json (gerado por analyze.py) e, para o
heatmap de top tokens, os resultados brutos. Nada é modificado.

Fatos, idiomas, modelos, stake, zona de decisão e janelas de
suavização vêm do config; o número de camadas vem dos dados.

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
import json
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
from analyze import load_raw_results
from config.experiment_config import ExperimentConfig
from plot_utils import smooth_curve, smoothing_suffix

CONFIG = ExperimentConfig()

# ── Estilo (só apresentação; paleta definitiva na Fase 7) ─────
LANG_COLORS  = {"pt": "#2166AC", "en": "#D6604D", "de": "#4DAC26", "it": "#8E0152"}
LANG_LABELS  = {"pt": "Portuguese (BR)", "en": "English (US)", "de": "German", "it": "Italian"}
LANG_DASHES  = {"pt": [], "en": [6, 3], "de": [2, 4], "it": [8, 3, 2, 3]}
MODEL_COLORS = {"llama": "#E69F00", "mistral": "#56B4E9", "qwen": "#009E73", "gemma": "#CC79A7"}
MODEL_LABELS = {"llama": "LLaMA 3.1 8B", "mistral": "Mistral 7B",
                "qwen": "Qwen 2.5 7B", "gemma": "Gemma 2 9B"}
FACT_LABELS  = {"aviao": "Airplane", "telefone": "Telephone"}
_FALLBACK    = ["#666666", "#999933", "#882255", "#44AA99", "#332288", "#AA4499"]

# Cores das entidades no heatmap
ENTITY_COLOR_DOMINANT = "#D6604D"   # vermelho — narrativa dominante
ENTITY_COLOR_OTHER    = "#2166AC"   # azul — a outra narrativa
OTHER_TOKEN_COLOR     = "#AAAAAA"


def _color(table, key, i=0):
    return table.get(key, _FALLBACK[i % len(_FALLBACK)])


def _label(table, key):
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


def load_computed(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  OK  {path}")


def model_layers(data, model):
    """Número de camadas do modelo, lido dos próprios resultados."""
    for fact_data in data.get(model, {}).values():
        for lang_data in fact_data.values():
            for run in lang_data.values():
                if "n_layers" in run:
                    return run["n_layers"]
                if run.get("last_token"):
                    return len(run["last_token"])
    return None


def decision_start(n_layers):
    """Primeira camada da zona de decisão (config.decision_zone)."""
    return int(round(n_layers * (1 - CONFIG.decision_zone)))


def _stake_langs(fact):
    return CONFIG.stake_languages.get(fact, [])


# ══════════════════════════════════════════════════════════════
# FIG 1 — CULTURAL DIVERGENCE MAP
# Layout: linhas = fatos × colunas = modelos
# Eixo X: número absoluto da camada
# ══════════════════════════════════════════════════════════════

def fig1_cultural_divergence_map(computed, output, window=3):
    set_style()
    facts, models, langs = CONFIG.facts, list(CONFIG.models), CONFIG.languages
    zone_pct = int(round(CONFIG.decision_zone * 100))

    fig, axes = plt.subplots(len(facts), len(models),
                             figsize=(3.5 * len(models), 3.25 * len(facts)),
                             sharey=False, squeeze=False)
    smooth_txt = "raw" if window == 1 else f"smoothed (window {window})"
    fig.suptitle(
        "Cultural Divergence Map\n"
        "CS = P(expected entity) − P(competing entity)  │  F1  │  "
        f"Absolute layer number  │  {smooth_txt}",
        fontweight="bold", fontsize=10, y=1.01
    )

    for row, fact in enumerate(facts):
        for col, model in enumerate(models):
            ax  = axes[row][col]
            n_l = model_layers(computed, model)
            if n_l is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            if row == 0:
                ax.set_title(f"{_label(MODEL_LABELS, model)}\n({n_l} layers)",
                             color=_color(MODEL_COLORS, model, col),
                             fontweight="bold", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"{_label(FACT_LABELS, fact)}\nCommitment Score",
                              fontsize=8, fontweight="bold")
            if row == len(facts) - 1:
                ax.set_xlabel("Layer", fontsize=8)

            for j, lang in enumerate(langs):
                entry = computed.get(model, {}).get(fact, {}).get(lang, {}).get("F1", {})
                m_res = entry.get("metrics_last", {})
                curve = m_res.get("commitment_curve", [])
                if not curve:
                    continue

                x = np.arange(1, len(curve) + 1)
                y = smooth_curve(curve, window)
                c = _color(LANG_COLORS, lang, j)
                ax.plot(x, y, color=c, linewidth=1.8, linestyle=_dashes(lang),
                        alpha=0.9, label=_label(LANG_LABELS, lang))
                ax.fill_between(x, y, 0, color=c, alpha=0.06)

                cs_peak = m_res.get("cs_at_peak", 0)
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

            stakes = _stake_langs(fact)
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
        Line2D([0], [0], color="gray", lw=1, linestyle="--", label="CS = 0 (neutral)"),
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
    Cor do token pelo seu ID exato (nunca por substring).

    Os conjuntos de ids vêm de run["entity_tokens"]: "competing" é o
    candidato dominante e "local" são as narrativas alternativas, no
    mesmo sentido em todos os idiomas.
    """
    sets     = (run.get("entity_tokens") or {}).get("sets", {})
    other    = set(sets.get("local", {}).get("ids", []))
    dominant = set(sets.get("competing", {}).get("ids", []))
    if token_id in dominant:
        return ENTITY_COLOR_DOMINANT
    if token_id in other:
        return ENTITY_COLOR_OTHER
    return OTHER_TOKEN_COLOR


def _top_id(layer, col_i):
    ids = layer.get("top_ids", [])
    return ids[col_i] if col_i < len(ids) else None


def _render_logit_lens_ax(ax, run, fact, model, lang, use_absolute):
    """Renderiza o Logit Lens de um único run em um eixo."""
    layers = run.get("last_token", [])
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
                            else f"{layer['layer_norm']:.2f}")
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

    m_res = run.get("metrics_last", {})
    stake_mark = " * stake" if CONFIG.stake(fact, lang) else ""
    ax.set_title(
        "%s (%dL) | %s%s\n\"%s\"\nCS_final=%+.3f  peak=%.3f@L%s" % (
            _label(MODEL_LABELS, model), n_layers, _label(LANG_LABELS, lang),
            stake_mark, run.get("prompt", ""),
            m_res.get("commitment_final", 0), m_res.get("iv_local_peak", 0),
            str(m_res.get("peak_layer", "?"))),
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


def fig3_logit_lens_combined(raw, fact, output, use_absolute=False, form="F1"):
    """Modelos de config.heatmap_models × (idioma de referência, idioma(s) com stake)."""
    set_style()
    langs  = [CONFIG.reference_language] + _stake_langs(fact)
    models = CONFIG.heatmap_models
    fig, axes = plt.subplots(len(models), len(langs),
                             figsize=(6 * len(langs), 4 * len(models)), squeeze=False)
    fig.suptitle(
        f"Logit Lens — Top Predicted Tokens per Layer\n"
        f"{_label(FACT_LABELS, fact)}  |  Intensity ∝ probability  |  {_version(use_absolute)}",
        fontweight="bold", fontsize=10, y=1.01)
    for r, model in enumerate(models):
        for c, lang in enumerate(langs):
            run = raw.get(model, {}).get(fact, {}).get(lang, {}).get(form, {})
            _render_logit_lens_ax(axes[r][c], run, fact, model, lang, use_absolute)
    _logit_lens_legend(fig, fact)
    save_fig(fig, output)


def fig3_logit_lens_per_model(raw, fact, model, output, use_absolute=False, form="F1"):
    """Todos os idiomas de um único modelo (1 linha × idiomas)."""
    set_style()
    langs = CONFIG.languages
    fig, axes = plt.subplots(1, len(langs), figsize=(4.5 * len(langs), 9),
                             sharey=True, squeeze=False)
    fig.suptitle(
        "Logit Lens | %s | %s | All Languages | %s | %s" % (
            _label(FACT_LABELS, fact), _label(MODEL_LABELS, model), form,
            _version(use_absolute)),
        fontweight="bold", fontsize=10, y=1.01)
    for ax, lang in zip(axes[0], langs):
        run = raw.get(model, {}).get(fact, {}).get(lang, {}).get(form, {})
        _render_logit_lens_ax(ax, run, fact, model, lang, use_absolute)
    _logit_lens_legend(fig, fact)
    save_fig(fig, output)


def fig3_logit_lens_single(raw, fact, model, lang, output, use_absolute=False, form="F1"):
    """Uma combinação (modelo × idioma) — para inspeção detalhada."""
    set_style()
    run = raw.get(model, {}).get(fact, {}).get(lang, {}).get(form, {})
    fig, ax = plt.subplots(1, 1, figsize=(5, 9))
    fig.suptitle(
        "Logit Lens | %s | %s | %s | %s | %s" % (
            _label(FACT_LABELS, fact), _label(MODEL_LABELS, model),
            _label(LANG_LABELS, lang), form, _version(use_absolute)),
        fontweight="bold", fontsize=9, y=1.01)
    _render_logit_lens_ax(ax, run, fact, model, lang, use_absolute)
    _logit_lens_legend(fig, fact)
    save_fig(fig, output)


def fig3_all(raw, output_dir):
    """Gera todas as variantes do heatmap (relativo e absoluto)."""
    for fact in CONFIG.facts:
        for abs_ in (False, True):
            sfx = "_abs" if abs_ else ""
            fig3_logit_lens_combined(
                raw, fact, os.path.join(output_dir, f"fig3_logit_lens_{fact}{sfx}.png"), abs_)
            for model in CONFIG.models:
                if model not in raw:
                    continue
                fig3_logit_lens_per_model(
                    raw, fact, model,
                    os.path.join(output_dir, f"fig3_logit_lens_{fact}_{model}{sfx}.png"), abs_)
                for lang in CONFIG.languages:
                    fig3_logit_lens_single(
                        raw, fact, model, lang,
                        os.path.join(output_dir,
                                     f"fig3_logit_lens_{fact}_{model}_{lang}{sfx}.png"), abs_)


# ══════════════════════════════════════════════════════════════
# FIG 6 — PEAK vs FINAL LAYER (Dumbbell chart)
#   círculo cheio = iv_local_peak (o que o modelo sabe internamente)
#   quadrado vazio = iv_local_final (o que a última camada gera)
#   linha vertical = suppression gap
# ══════════════════════════════════════════════════════════════

def fig6_peak_vs_final(computed, output):
    set_style()
    facts, models, langs = CONFIG.facts, list(CONFIG.models), CONFIG.languages

    fig, axes = plt.subplots(1, len(facts), figsize=(6 * len(facts), 4.5), squeeze=False)
    fig.suptitle("Peak vs. Final Layer — Cultural Entity Probability  |  F1",
                 fontweight="bold", fontsize=10, y=1.02)

    for ax, fact in zip(axes[0], facts):
        ax.set_title(_label(FACT_LABELS, fact), fontweight="bold", fontsize=10)

        x      = np.arange(len(langs))
        width  = 0.72 / max(1, len(models))
        offset = -(len(models) - 1) / 2 * width

        for i, model in enumerate(models):
            mc = _color(MODEL_COLORS, model, i)
            peaks, finals = [], []
            for lang in langs:
                m = (computed.get(model, {}).get(fact, {}).get(lang, {})
                     .get("F1", {}).get("metrics_last", {}))
                peaks.append(m.get("iv_local_peak", 0))
                finals.append(m.get("iv_local_final", 0))

            xpos = x + offset + i * width
            for xi, pk, fn in zip(xpos, peaks, finals):
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
        ax.set_ylabel("P(cultural entity)", fontsize=8.5)
        ax.set_ylim(-0.02, 1.08)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.4)

        for stake_lang in _stake_langs(fact):
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
        mpatches.Patch(facecolor="gold", alpha=0.25, label="Cultural stake language"),
    ]
    fig.legend(handles=model_leg + style_leg, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.10), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

FIGURES = ["fig1", "fig3", "fig6"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=CONFIG.results_dir)
    parser.add_argument("--computed",    default=None,
                        help="padrão: {results-dir}/metrics_computed.json")
    parser.add_argument("--output-dir",  default=CONFIG.figures_dir)
    parser.add_argument("--fig", default="all", choices=FIGURES + ["all"])
    args = parser.parse_args(argv)

    computed_path = args.computed or os.path.join(args.results_dir, "metrics_computed.json")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    figs = FIGURES if args.fig == "all" else [args.fig]

    if {"fig1", "fig6"} & set(figs):
        print(f"Carregando: {computed_path}")
        computed = load_computed(computed_path)

    if "fig1" in figs:
        # Uma versão por janela de suavização (_w3, _raw)
        for w in CONFIG.smoothing_windows:
            fig1_cultural_divergence_map(
                computed, os.path.join(args.output_dir, f"fig1{smoothing_suffix(w)}.png"),
                window=w)
    if "fig3" in figs:
        fig3_all(load_raw_results(args.results_dir, CONFIG), args.output_dir)
    if "fig6" in figs:
        fig6_peak_vs_final(computed, os.path.join(args.output_dir, "fig6.png"))

    print(f"\nFiguras em {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
