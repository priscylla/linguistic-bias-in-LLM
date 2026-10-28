"""
generate_paper_figures.py
Gera as figuras para o artigo usando metrics_computed.json.
Os JSONs originais de resultados NAO sao alterados.

Uso:
    python generate_paper_figures.py
    python generate_paper_figures.py --fig fig1
    python generate_paper_figures.py --output-dir paper_figures
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── Constantes de estilo ──────────────────────────────────────
LANG_COLORS = {"pt": "#2166AC", "en": "#D6604D", "de": "#4DAC26", "it": "#8E0152"}
LANG_LABELS = {"pt": "Portuguese (BR)", "en": "English (US)", "de": "German", "it": "Italian"}
LANG_DASHES = {"pt": [], "en": [6, 3], "de": [2, 4], "it": [8, 3, 2, 3]}
MODEL_COLORS = {"llama": "#E69F00", "mistral": "#56B4E9", "qwen": "#009E73", "gemma": "#CC79A7"}
MODEL_LABELS = {"llama": "LLaMA 3.1 8B", "mistral": "Mistral 7B",
                "qwen": "Qwen 2.5 7B", "gemma": "Gemma 2 9B"}
MODEL_LAYERS = {"llama": 32, "mistral": 32, "qwen": 28, "gemma": 42}
FACT_LABELS  = {"aviao": "Airplane", "telefone": "Telephone"}
STAKE        = {"aviao": "pt", "telefone": "it"}

MODELS = ["llama", "mistral", "qwen", "gemma"]
LANGS  = ["pt", "en", "de", "it"]
FACTS  = ["aviao", "telefone"]


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


def smooth(arr, w=3):
    """Média móvel — janela recortada nas bordas (equivalente ao JS)."""
    result = []
    for i in range(len(arr)):
        s = max(0, i - w // 2)
        e = min(len(arr), s + w)
        result.append(sum(arr[s:e]) / (e - s))
    return np.array(result)


def load_computed(path="results/metrics_computed.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  OK  {path}")


# ══════════════════════════════════════════════════════════════
# FIG 1 — CULTURAL DIVERGENCE MAP
# Layout: 2 linhas (aviao, telefone) × 4 colunas (modelos)
# Eixo X: numero absoluto da camada
# Metricas: commitment_curve de metrics_computed.json
# ══════════════════════════════════════════════════════════════

def fig1_cultural_divergence_map(computed, output):
    set_style()

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharey=False)
    fig.suptitle(
        "Cultural Divergence Map\n"
        "CS = P(expected entity) \u2212 P(competing entity)  \u2502  F1  \u2502  Absolute layer number",
        fontweight="bold", fontsize=10, y=1.01
    )

    for row, fact in enumerate(FACTS):
        stake_lang = STAKE[fact]

        for col, model in enumerate(MODELS):
            ax  = axes[row][col]
            n_l = MODEL_LAYERS[model]

            # Titulo do subplot (apenas linha superior)
            if row == 0:
                ax.set_title(
                    f"{MODEL_LABELS[model]}\n({n_l} layers)",
                    color=MODEL_COLORS[model],
                    fontweight="bold", fontsize=8.5
                )

            # Label do eixo Y (apenas coluna esquerda)
            if col == 0:
                ax.set_ylabel(
                    f"{FACT_LABELS[fact]}\nCommitment Score",
                    fontsize=8, fontweight="bold"
                )

            # Label do eixo X (apenas linha inferior)
            if row == 1:
                ax.set_xlabel("Layer", fontsize=8)

            # Plotar curva por idioma
            for lang in LANGS:
                entry = computed.get(model, {}).get(fact, {}).get(lang, {}).get("F1", {})
                m_res = entry.get("metrics_last", {})
                curve = m_res.get("commitment_curve", [])
                if not curve:
                    continue

                # Eixo X: numero absoluto da camada (1 .. n_l)
                x = np.linspace(1, n_l, len(curve))
                y = smooth(np.array(curve), w=3)

                ax.plot(x, y,
                        color=LANG_COLORS[lang],
                        linewidth=1.8,
                        linestyle=(0, tuple(LANG_DASHES[lang])) if LANG_DASHES[lang] else "solid",
                        alpha=0.9,
                        label=LANG_LABELS[lang])
                ax.fill_between(x, y, 0,
                                color=LANG_COLORS[lang], alpha=0.06)

                # Anotacao do CS final (evitar sobreposicao)
                cs_peak = m_res.get("cs_at_peak", 0)
                if abs(cs_peak) > 0.05:
                    ax.annotate(
                        f"{cs_peak:+.2f}",
                        xy=(x[-1], y[-1]),
                        xytext=(x[-1] - n_l * 0.13, y[-1]),
                        fontsize=6, color=LANG_COLORS[lang],
                        fontweight="bold", va="center"
                    )

            # Linha CS = 0
            ax.axhline(0, color="gray", lw=0.7, linestyle="--", alpha=0.6)

            # Zona de decisao (ultimas 30%)
            decision_start = int(n_l * 0.70)
            ax.axvspan(decision_start, n_l, alpha=0.05, color="gray")
            ax.text(decision_start + 0.5,
                    ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0.05 else 0.05,
                    f"L{decision_start}", fontsize=6, color="gray",
                    alpha=0.7, va="top")

            # Marcador do idioma com stake
            ax.text(0.02, 0.97,
                    f"stake: {stake_lang.upper()}",
                    transform=ax.transAxes,
                    fontsize=6.5, color=LANG_COLORS[stake_lang],
                    va="top", fontstyle="italic")

            # Eixo X: ticks a cada 4 camadas
            step = max(4, n_l // 8)
            ax.set_xticks(range(1, n_l + 1, step))
            ax.set_xlim(1, n_l)

    # Legenda global
    leg = [
        Line2D([0], [0], color=LANG_COLORS[l], lw=2,
               linestyle=(0, tuple(LANG_DASHES[l])) if LANG_DASHES[l] else "solid",
               label=LANG_LABELS[l])
        for l in LANGS
    ]
    leg += [
        mpatches.Patch(facecolor="gray", alpha=0.15,
                       label="Decision zone (last 30%)"),
        Line2D([0], [0], color="gray", lw=1, linestyle="--",
               label="CS = 0 (neutral)")
    ]
    fig.legend(handles=leg, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output)



# ══════════════════════════════════════════════════════════════
# FIG 6 — PEAK vs FINAL LAYER (Dumbbell chart)
# Mostra a supressao gramatical:
#   circulo cheio = iv_local_peak (o que o modelo sabe internamente)
#   quadrado vazio = iv_local_final (o que a ultima camada gera)
#   linha vertical = suppression gap
# ══════════════════════════════════════════════════════════════

def fig6_peak_vs_final(computed, output):
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Peak vs. Final Layer \u2014 Cultural Entity Probability  |  F1",
        fontweight="bold", fontsize=10, y=1.02
    )

    for ax, fact in zip(axes, FACTS):
        ax.set_title(FACT_LABELS[fact], fontweight="bold", fontsize=10)

        x      = np.arange(len(LANGS))
        width  = 0.18
        offset = -(len(MODELS) - 1) / 2 * width

        for i, model in enumerate(MODELS):
            peaks, finals = [], []
            for lang in LANGS:
                m = computed.get(model, {}).get(fact, {}).get(lang, {}).get("F1", {}).get("metrics_last", {})
                peaks.append(m.get("iv_local_peak", 0))
                finals.append(m.get("iv_local_final", 0))

            xpos = x + offset + i * width

            for xi, pk, fn in zip(xpos, peaks, finals):
                # Linha vertical: suppression gap
                if pk > 0.01:
                    ax.plot([xi, xi], [fn, pk],
                            color=MODEL_COLORS[model],
                            lw=1.8, alpha=0.45, zorder=2)

                # Circulo cheio = peak
                ax.plot(xi, pk, "o",
                        color=MODEL_COLORS[model],
                        markersize=8, alpha=0.92,
                        markeredgecolor="white", markeredgewidth=0.8,
                        zorder=4,
                        label=MODEL_LABELS[model] if xi == xpos[0] else "")

                # Quadrado vazio = final
                ax.plot(xi, fn, "s",
                        color="white", markersize=7,
                        markeredgecolor=MODEL_COLORS[model],
                        markeredgewidth=2.0, zorder=5)

                # Anotar suppression significativa
                sup = pk - fn
                if sup > 0.18 and pk > 0.12:
                    ax.text(xi + 0.025, (pk + fn) / 2,
                            f"{sup:.2f}",
                            fontsize=5.5, color=MODEL_COLORS[model],
                            va="center", ha="left", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABELS[l] for l in LANGS],
                           fontsize=8, rotation=12, ha="right")
        ax.set_ylabel("P(cultural entity)", fontsize=8.5)
        ax.set_ylim(-0.02, 1.08)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.4)

        # Destacar idioma com stake
        stake_lang = STAKE[fact]
        j_stake = LANGS.index(stake_lang)
        ax.axvspan(j_stake - 0.5, j_stake + 0.5,
                   alpha=0.06, color="gold", zorder=0)
        ax.text(j_stake, 1.04, "stake",
                ha="center", fontsize=7,
                color="goldenrod", fontweight="bold")

    # Legenda modelos
    model_leg = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MODEL_COLORS[m],
               markeredgecolor="white", markersize=9,
               label=MODEL_LABELS[m])
        for m in MODELS
    ]
    # Legenda marcadores
    style_leg = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#888", markeredgecolor="white",
               markersize=9, label="Peak layer (filled circle)"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor="white", markeredgecolor="#888",
               markeredgewidth=2, markersize=9,
               label="Final layer (open square)"),
        Line2D([0], [0], color="#888", lw=1.5, alpha=0.5,
               label="Suppression gap"),
        mpatches.Patch(facecolor="gold", alpha=0.25,
                       label="Cultural stake language"),
    ]
    fig.legend(handles=model_leg + style_leg,
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.10), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

FIGURES = {
    "fig1": fig1_cultural_divergence_map,
    "fig6": fig6_peak_vs_final,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--computed",    default="results/metrics_computed.json")
    parser.add_argument("--output-dir",  default="paper_figures")
    parser.add_argument("--fig",         default="all",
                        help="Figura a gerar: fig1 | all")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    print(f"Carregando: {args.computed}")
    computed = load_computed(args.computed)
    print(f"Modelos: {list(computed.keys())}")
    print(f"Saida:   {args.output_dir}/\n")

    figs_to_run = FIGURES if args.fig == "all" else {args.fig: FIGURES[args.fig]}

    for name, func in figs_to_run.items():
        output = os.path.join(args.output_dir, f"{name}.png")
        func(computed, output)

    print(f"\n{len(figs_to_run)} figura(s) gerada(s) em {args.output_dir}/")


if __name__ == "__main__":
    main()