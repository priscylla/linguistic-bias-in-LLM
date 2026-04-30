"""
generate_visualizations.py
Gera todas as figuras do paper usando dados sintéticos ou reais.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════
# ESTILO GLOBAL
# ══════════════════════════════════════════════════════════════

LANGUAGE_COLORS = {
    "pt": "#2166AC",
    "en": "#D6604D",
    "de": "#4DAC26",
    "it": "#8E0152",
}
LANGUAGE_LABELS = {
    "pt": "Portuguese (BR)",
    "en": "English (US)",
    "de": "German",
    "it": "Italian",
}
MODEL_COLORS = {
    "llama":   "#E69F00",
    "mistral": "#56B4E9",
    "qwen":    "#009E73",
    "gemma":   "#CC79A7",
}
MODEL_LABELS = {
    "llama":   "LLaMA 3.1 8B",
    "mistral": "Mistral 7B",
    "qwen":    "Qwen 2.5 7B",
    "gemma":   "Gemma 2 9B",
}
FACT_LABELS = {
    "aviao":    "Airplane",
    "telefone": "Telephone",
    
}
FORMULATION_STYLES = {
    "F1": {"linestyle": "-",  "linewidth": 2.0, "label": "F1 — Direct"},
    "F2": {"linestyle": "--", "linewidth": 1.8, "label": "F2 — Attributive"},
    "F3": {"linestyle": ":",  "linewidth": 2.0, "label": "F3 — Disputative"},
}
ENTITY_COLORS = {
    "local":    "#2166AC",
    "dominant": "#D6604D",
    "other":    "#BBBBBB",
}

def set_style():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          9,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "figure.titlesize":   11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.5,
        "lines.linewidth":    1.8,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

ENTITY_MAP = {
    "aviao": {
        "pt": {"local": ["Santos", "Dumont", "14-Bis"], "dominant": ["Wright", "Orville", "Wilbur"]},
        "en": {"local": ["Wright", "Orville", "Wilbur"], "dominant": ["Santos", "Dumont"]},
        "de": {"local": ["Wright", "Orville"], "dominant": ["Santos", "Dumont"]},
        "it": {"local": ["Wright", "Brothers"], "dominant": ["Santos", "Dumont"]},
    },
    "telefone": {
        "pt": {"local": ["Bell", "Graham"], "dominant": ["Meucci"]},
        "en": {"local": ["Bell", "Graham"], "dominant": ["Meucci"]},
        "de": {"local": ["Bell", "Graham"], "dominant": ["Meucci"]},
        "it": {"local": ["Meucci", "Antonio"], "dominant": ["Bell", "Graham"]},
    },
}


def classify_token(token: str, fact: str, lang: str) -> str:
    t = token.strip().lower()
    emap = ENTITY_MAP.get(fact, {}).get(lang, {})
    for e in emap.get("local", []):
        if e.lower() in t:
            return "local"
    for e in emap.get("dominant", []):
        if e.lower() in t:
            return "dominant"
    return "other"


def entity_prob_curve(layer_results: list, fact: str, lang: str) -> np.ndarray:
    local_ents = ENTITY_MAP.get(fact, {}).get(lang, {}).get("local", [])
    probs = []
    for layer in layer_results:
        total = sum(
            prob for token, prob in zip(layer["top_tokens"], layer["top_probs"])
            if any(e.lower() in token.strip().lower() for e in local_ents)
        )
        probs.append(total)
    return np.array(probs)


def smooth(y: np.ndarray, window_frac: float = 0.10) -> np.ndarray:
    window = max(1, int(len(y) * window_frac))
    return np.convolve(y, np.ones(window) / window, mode="same")


def load_results(results_dir: str = "results") -> dict:
    all_results = {}
    for model_key in ["llama", "mistral", "qwen", "gemma"]:
        path = os.path.join(results_dir, f"{model_key}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                all_results[model_key] = json.load(f)
    return all_results


# ══════════════════════════════════════════════════════════════
# FIGURA 1 — LOGIT LENS HEATMAP
# ══════════════════════════════════════════════════════════════

def plot_logit_lens_heatmap(
    results: dict,
    fact: str,
    formulation: str,
    model_key: str,
    position: str = "last_token",
    n_layers_show: int = 24,
    top_k_show: int = 5,
    output_path: str = None
):
    set_style()
    languages = ["pt", "en", "de", "it"]

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(
        f"Logit Lens Heatmap — {FACT_LABELS[fact]}  |  "
        f"{MODEL_LABELS[model_key]}  |  {formulation}  |  "
        f"{'Last token' if position == 'last_token' else 'Keyword token'}",
        fontweight="bold", fontsize=10, y=1.01
    )

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.06,
                           left=0.07, right=0.99)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    for ax_idx, lang in enumerate(languages):
        ax = axes[ax_idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        run_data = results.get(model_key, {}).get(
            fact, {}
        ).get(lang, {}).get(formulation, {})

        layer_results = run_data.get(position, [])
        if not layer_results:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        n_total = len(layer_results)
        indices = np.linspace(0, n_total - 1, n_layers_show, dtype=int)
        selected = [layer_results[i] for i in indices]

        n_rows = len(selected)
        n_cols = top_k_show
        cw = 1.0 / n_cols
        ch = 1.0 / n_rows

        layer_labels = []

        for row_i, layer_data in enumerate(selected):
            y = 1.0 - (row_i + 1) * ch
            layer_labels.append(f"{layer_data['layer_norm']:.2f}")

            for col_i in range(top_k_show):
                token = layer_data["top_tokens"][col_i]
                prob  = layer_data["top_probs"][col_i]
                etype = classify_token(token, fact, lang)

                x = col_i * cw
                base_color = ENTITY_COLORS[etype]
                alpha = float(np.clip(prob * 9, 0.08, 0.95))

                rect = mpatches.FancyBboxPatch(
                    (x + 0.003, y + 0.001),
                    cw - 0.006, ch - 0.003,
                    boxstyle="round,pad=0.008",
                    facecolor=base_color,
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2
                )
                ax.add_patch(rect)

                text_color = "white" if alpha > 0.45 else "#333333"
                disp = token.strip()[:7] if len(token.strip()) > 7 else token.strip()

                ax.text(
                    x + cw / 2, y + ch * 0.60,
                    disp,
                    ha="center", va="center",
                    fontsize=5.2, color=text_color,
                    fontweight="bold" if etype != "other" else "normal",
                    zorder=3
                )
                ax.text(
                    x + cw / 2, y + ch * 0.22,
                    f"{prob:.3f}",
                    ha="center", va="center",
                    fontsize=4.0, color=text_color, alpha=0.85,
                    zorder=3
                )

        # Eixo Y — profundidade normalizada
        ax.set_yticks([
            1.0 - (i + 0.5) * ch for i in range(n_rows)
        ])
        if ax_idx == 0:
            ax.set_yticklabels(layer_labels, fontsize=5.5)
            ax.set_ylabel("Relative Layer Depth", fontsize=8)
        else:
            ax.set_yticklabels([])

        ax.set_xticks([])

        # Título colorido por idioma
        ax.set_title(
            LANGUAGE_LABELS[lang],
            color=LANGUAGE_COLORS[lang],
            fontweight="bold", fontsize=9, pad=5
        )

        # Linha horizontal marcando 50% de profundidade
        ax.axhline(
            y=0.50, color="gray",
            linestyle="--", linewidth=0.5, alpha=0.4, zorder=1
        )
        ax.text(
            0.99, 0.51, "50%",
            ha="right", va="bottom",
            fontsize=4.5, color="gray", alpha=0.6,
            transform=ax.transAxes
        )

    # Legenda
    legend_elements = [
        mpatches.Patch(
            facecolor=ENTITY_COLORS["local"],
            alpha=0.8, label="Local cultural entity"
        ),
        mpatches.Patch(
            facecolor=ENTITY_COLORS["dominant"],
            alpha=0.8, label="Dominant (anglophone) entity"
        ),
        mpatches.Patch(
            facecolor=ENTITY_COLORS["other"],
            alpha=0.5, label="Other token"
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=8, framealpha=0.95,
        edgecolor="lightgray"
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"  ✅ {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════
# FIGURA 2 — CULTURAL DIVERGENCE MAP
# ══════════════════════════════════════════════════════════════

def plot_cultural_divergence_map(
    results: dict,
    fact: str,
    formulation: str,
    position: str = "last_token",
    output_path: str = None
):
    set_style()
    models    = ["llama", "mistral", "qwen", "gemma"]
    languages = ["pt", "en", "de", "it"]

    fig, axes = plt.subplots(
        2, 2, figsize=(9, 6),
        sharey=False
    )
    axes = axes.flatten()

    fig.suptitle(
        f"Cultural Divergence Map — {FACT_LABELS[fact]}  |  {formulation}",
        fontweight="bold", fontsize=11
    )

    for m_idx, model_key in enumerate(models):
        ax = axes[m_idx]

        ax.set_title(
            MODEL_LABELS[model_key],
            color=MODEL_COLORS[model_key],
            fontweight="bold", fontsize=9
        )

        for lang in languages:
            run_data = results.get(model_key, {}).get(
                fact, {}
            ).get(lang, {}).get(formulation, {})

            layer_results = run_data.get(position, [])
            if not layer_results:
                continue

            x = np.array([r["layer_norm"] for r in layer_results])
            y = entity_prob_curve(layer_results, fact, lang)
            y_s = smooth(y, 0.12)

            ax.plot(
                x, y_s,
                color=LANGUAGE_COLORS[lang],
                label=LANGUAGE_LABELS[lang],
                alpha=0.9, zorder=3
            )
            ax.fill_between(
                x, y_s, alpha=0.07,
                color=LANGUAGE_COLORS[lang]
            )

            # Marcar PCC
            pcc_norm = run_data.get(
                "metrics_last", {}
            ).get("pcc_norm")

            if pcc_norm is not None:
                pcc_x_idx = np.argmin(np.abs(x - pcc_norm))
                pcc_y     = y_s[pcc_x_idx]
                ax.scatter(
                    [pcc_norm], [pcc_y],
                    color=LANGUAGE_COLORS[lang],
                    s=30, zorder=5,
                    edgecolors="white", linewidth=0.6
                )
                ax.axvline(
                    x=pcc_norm,
                    color=LANGUAGE_COLORS[lang],
                    linestyle=":", alpha=0.35, linewidth=0.8
                )

        # Linha de threshold
        ax.axhline(
            y=0.10, color="gray",
            linestyle="--", linewidth=0.7, alpha=0.5
        )
        ax.text(
            0.01, 0.105, "threshold",
            fontsize=6, color="gray", va="bottom"
        )

        # Região de camadas finais
        ax.axvspan(0.70, 1.0, alpha=0.04, color="gray")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)

        if m_idx in [0, 2]:
            ax.set_ylabel("Entity Probability", fontsize=8)
        if m_idx in [2, 3]:
            ax.set_xlabel("Relative Layer Depth", fontsize=8)

        # Anotação "early" / "late"
        ax.text(0.18, -0.08, "early", transform=ax.transAxes,
                fontsize=6.5, color="gray", ha="center")
        ax.text(0.82, -0.08, "late",  transform=ax.transAxes,
                fontsize=6.5, color="gray", ha="center")

    # Legenda global
    lang_lines = [
        Line2D([0], [0], color=LANGUAGE_COLORS[l],
               linewidth=2, label=LANGUAGE_LABELS[l])
        for l in languages
    ] + [
        Line2D([0], [0], color="gray", linestyle="--",
               linewidth=1, label="PCC threshold (p=0.10)"),
        Line2D([0], [0], marker="o", color="gray",
               markersize=5, linewidth=0,
               label="PCC marker", markeredgecolor="white")
    ]

    fig.legend(
        handles=lang_lines,
        loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=8, framealpha=0.95,
        edgecolor="lightgray"
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"  ✅ {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════
# FIGURA 3 — PCC COMPARISON (DOT PLOT)
# ══════════════════════════════════════════════════════════════

def plot_pcc_comparison(
    df: pd.DataFrame,
    formulation: str = "F1",
    output_path: str = None
):
    set_style()

    data     = df[df["formulation"] == formulation].dropna(subset=["pcc_norm"])
    facts    = ["aviao", "telefone"]
    models   = ["llama", "mistral", "qwen", "gemma"]
    langs    = ["pt", "en", "de", "it"]

    fig, axes = plt.subplots(
        1, 3, figsize=(10, 3.5),
        sharey=True
    )
    fig.suptitle(
        f"Point of Cultural Convergence (PCC) — {formulation}",
        fontweight="bold", fontsize=11
    )

    jitter = np.linspace(-0.18, 0.18, len(langs))
    lang_jitter = dict(zip(langs, jitter))

    for f_idx, fact in enumerate(facts):
        ax = axes[f_idx]
        fact_df = data[data["fact"] == fact]

        for m_idx, model_key in enumerate(models):
            model_df = fact_df[fact_df["model"] == model_key]

            # Linha de média
            mean_pcc = model_df["pcc_norm"].mean()
            if not np.isnan(mean_pcc):
                ax.hlines(
                    m_idx, 0, mean_pcc,
                    colors=MODEL_COLORS[model_key],
                    linewidth=3, alpha=0.25
                )

            for lang in langs:
                row = model_df[model_df["language"] == lang]
                if row.empty:
                    continue
                pcc_val = row["pcc_norm"].values[0]
                y_pos   = m_idx + lang_jitter[lang]

                ax.scatter(
                    pcc_val, y_pos,
                    color=LANGUAGE_COLORS[lang],
                    s=55, alpha=0.90, zorder=4,
                    edgecolors="white", linewidth=0.6
                )

        ax.axvline(
            x=0.5, color="gray",
            linestyle="--", linewidth=0.7, alpha=0.5
        )
        ax.set_xlim(0, 1)
        ax.set_title(FACT_LABELS[fact], fontweight="bold")
        ax.set_xlabel("Normalized Layer Depth", fontsize=8)

        # "early" / "late"
        ax.text(0.12, -0.12, "early", transform=ax.transAxes,
                fontsize=7, color="gray", ha="center")
        ax.text(0.88, -0.12, "late",  transform=ax.transAxes,
                fontsize=7, color="gray", ha="center")

    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(
        [MODEL_LABELS[m] for m in models], fontsize=8
    )
    axes[0].set_ylabel("Model", fontsize=8)

    # Legenda
    legend_items = [
        plt.scatter([], [], color=LANGUAGE_COLORS[l],
                    s=50, label=LANGUAGE_LABELS[l],
                    edgecolors="white")
        for l in langs
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=8, framealpha=0.95,
        edgecolor="lightgray"
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"  ✅ {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════
# FIGURA 4 — STAKE ANALYSIS
# ══════════════════════════════════════════════════════════════

def plot_stake_analysis(
    df: pd.DataFrame,
    output_path: str = None
):
    set_style()

    models = ["llama", "mistral", "qwen", "gemma"]

    fig, (ax_pcc, ax_iv) = plt.subplots(
        1, 2, figsize=(9, 3.5)
    )
    fig.suptitle(
        "Cultural Stake Analysis\n"
        "(High stake = language has local inventor candidate)",
        fontweight="bold", fontsize=10
    )

    width = 0.30

    for m_idx, model_key in enumerate(models):
        mdf = df[df["model"] == model_key]

        high = mdf[mdf["has_stake"] == True]
        low  = mdf[mdf["has_stake"] == False]

        x = m_idx

        # ── PCC ──
        for ax, col, label in [
            (ax_pcc, "pcc_norm", "PCC"),
            (ax_iv,  "iv",       "IV"),
        ]:
            h_vals = high[col].dropna().values
            l_vals = low[col].dropna().values

            if len(h_vals) > 0:
                ax.bar(
                    x - width / 2, np.mean(h_vals),
                    width=width,
                    color=MODEL_COLORS[model_key],
                    alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                    yerr=np.std(h_vals) if len(h_vals) > 1 else 0,
                    error_kw={"elinewidth": 1, "alpha": 0.6, "capsize": 3}
                )
            if len(l_vals) > 0:
                ax.bar(
                    x + width / 2, np.mean(l_vals),
                    width=width,
                    color=MODEL_COLORS[model_key],
                    alpha=0.30,
                    edgecolor=MODEL_COLORS[model_key],
                    linewidth=0.8,
                    yerr=np.std(l_vals) if len(l_vals) > 1 else 0,
                    error_kw={"elinewidth": 1, "alpha": 0.5, "capsize": 3}
                )

    for ax, ylabel, ylim in [
        (ax_pcc, "Normalized PCC", (0, 1)),
        (ax_iv,  "Bias Intensity (IV)", (0, 0.6)),
    ]:
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in models],
            fontsize=8, rotation=15, ha="right"
        )
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(*ylim)
        ax.axhline(
            y=(0.5 if ylabel == "Normalized PCC" else 0.3),
            color="gray", linestyle="--",
            linewidth=0.7, alpha=0.5
        )

    ax_pcc.set_title("(A) Point of Cultural Convergence")
    ax_iv.set_title("(B) Final Bias Intensity")

    legend_elements = [
        mpatches.Patch(facecolor="gray", alpha=0.85, label="High cultural stake"),
        mpatches.Patch(facecolor="gray", alpha=0.30,
                       edgecolor="gray", label="Low cultural stake"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=8.5, framealpha=0.95,
        edgecolor="lightgray"
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"  ✅ {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════
# FIGURA 5 — FORMULATION COMPARISON
# ══════════════════════════════════════════════════════════════

def plot_formulation_comparison(
    results: dict,
    fact: str,
    model_key: str,
    position: str = "last_token",
    output_path: str = None
):
    set_style()

    languages    = ["pt", "en", "de", "it"]
    formulations = ["F1", "F2", "F3"]
    markers      = {"F1": "o", "F2": "s", "F3": "^"}

    fig, axes = plt.subplots(
        2, 2, figsize=(9, 6),
        sharey=True
    )
    axes = axes.flatten()

    fig.suptitle(
        f"Formulation Effect — {FACT_LABELS[fact]}  |  "
        f"{MODEL_LABELS[model_key]}",
        fontweight="bold", fontsize=11
    )

    for l_idx, lang in enumerate(languages):
        ax = axes[l_idx]

        ax.set_title(
            LANGUAGE_LABELS[lang],
            color=LANGUAGE_COLORS[lang],
            fontweight="bold", fontsize=9
        )

        iv_vals = {}

        for form in formulations:
            run_data = results.get(model_key, {}).get(
                fact, {}
            ).get(lang, {}).get(form, {})

            layer_results = run_data.get(position, [])
            if not layer_results:
                continue

            x   = np.array([r["layer_norm"] for r in layer_results])
            y   = entity_prob_curve(layer_results, fact, lang)
            y_s = smooth(y, 0.10)

            st = FORMULATION_STYLES[form]
            ax.plot(
                x, y_s,
                color=LANGUAGE_COLORS[lang],
                linestyle=st["linestyle"],
                linewidth=st["linewidth"],
                label=st["label"],
                alpha=0.9
            )
            # Marcador no ponto final
            ax.scatter(
                [x[-1]], [y_s[-1]],
                color=LANGUAGE_COLORS[lang],
                s=30, zorder=5,
                marker=markers[form],
                edgecolors="white", linewidth=0.6
            )

            iv_vals[form] = run_data.get(
                "metrics_last", {}
            ).get("iv", 0)

        # Delta ΔIV(F3 − F1)
        if "F1" in iv_vals and "F3" in iv_vals:
            delta = iv_vals["F3"] - iv_vals["F1"]
            delta_color = (
                "#2166AC" if delta > 0.02
                else "#D6604D" if delta < -0.02
                else "gray"
            )
            ax.text(
                0.97, 0.95,
                f"ΔIV(F3−F1): {delta:+.3f}",
                transform=ax.transAxes,
                fontsize=7.5, ha="right", va="top",
                color=delta_color, fontweight="bold"
            )

        ax.axhline(
            y=0.10, color="gray",
            linestyle="--", linewidth=0.6, alpha=0.5
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)

        if l_idx in [0, 2]:
            ax.set_ylabel("Entity Probability", fontsize=8)
        if l_idx in [2, 3]:
            ax.set_xlabel("Relative Layer Depth", fontsize=8)

    # Legenda de formulações
    form_legend = [
        Line2D(
            [0], [0],
            color="gray",
            linestyle=FORMULATION_STYLES[f]["linestyle"],
            linewidth=2,
            marker=markers[f],
            markersize=5,
            label=FORMULATION_STYLES[f]["label"]
        )
        for f in formulations
    ]
    fig.legend(
        handles=form_legend,
        loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=8.5, framealpha=0.95,
        edgecolor="lightgray"
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"  ✅ {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════
# FIGURA 6 — LAST TOKEN vs KEYWORD TOKEN (COMPARAÇÃO)
# ══════════════════════════════════════════════════════════════

def plot_position_comparison(
    results: dict,
    fact: str,
    formulation: str,
    model_key: str,
    output_path: str = None
):
    """
    Compara a trajetória do Logit Lens entre
    o último token e o token da keyword,
    para cada idioma.
    """
    set_style()

    languages = ["pt", "en", "de", "it"]
    positions = {
        "last_token":    {"linestyle": "-",  "label": "Last token"},
        "keyword_token": {"linestyle": "--", "label": "Keyword token"},
    }

    fig, axes = plt.subplots(
        2, 2, figsize=(9, 5.5), sharey=True
    )
    axes = axes.flatten()

    fig.suptitle(
        f"Last Token vs. Keyword Token — "
        f"{FACT_LABELS[fact]}  |  {MODEL_LABELS[model_key]}  |  {formulation}",
        fontweight="bold", fontsize=10
    )

    for l_idx, lang in enumerate(languages):
        ax = axes[l_idx]

        ax.set_title(
            LANGUAGE_LABELS[lang],
            color=LANGUAGE_COLORS[lang],
            fontweight="bold", fontsize=9
        )

        run_data = results.get(model_key, {}).get(
            fact, {}
        ).get(lang, {}).get(formulation, {})

        for pos_key, pos_style in positions.items():
            layer_results = run_data.get(pos_key, [])
            if not layer_results:
                continue

            x   = np.array([r["layer_norm"] for r in layer_results])
            y   = entity_prob_curve(layer_results, fact, lang)
            y_s = smooth(y, 0.10)

            ax.plot(
                x, y_s,
                color=LANGUAGE_COLORS[lang],
                linestyle=pos_style["linestyle"],
                linewidth=2,
                label=pos_style["label"],
                alpha=0.9
            )
            ax.fill_between(
                x, y_s, alpha=0.05,
                color=LANGUAGE_COLORS[lang]
            )

        ax.axhline(
            y=0.10, color="gray",
            linestyle=":", linewidth=0.6, alpha=0.5
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)

        if l_idx in [0, 2]:
            ax.set_ylabel("Entity Probability", fontsize=8)
        if l_idx in [2, 3]:
            ax.set_xlabel("Relative Layer Depth", fontsize=8)

    # Legenda
    pos_legend = [
        Line2D([0], [0], color="gray",
               linestyle=s["linestyle"], linewidth=2,
               label=s["label"])
        for s in positions.values()
    ]
    fig.legend(
        handles=pos_legend,
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9, framealpha=0.95,
        edgecolor="lightgray"
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"  ✅ {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def generate_all_figures(
    results_dir: str = "results",
    figures_dir: str = "figures"
):
    Path(figures_dir).mkdir(exist_ok=True)

    # Gerar metrics_comparison.csv automaticamente se nao existir.
    # Isso permite rodar run_visualization.py diretamente apos
    # os experimentos, sem precisar chamar consolidate_results.py
    # manualmente.
    csv_path = os.path.join(results_dir, "metrics_comparison.csv")
    if not os.path.exists(csv_path):
        print("metrics_comparison.csv nao encontrado -- consolidando...")
        from consolidate_results import consolidate
        consolidate(results_dir)

    print("Carregando resultados...")
    results = load_results(results_dir)

    if not results:
        print("Nenhum resultado encontrado em results/")
        print("Execute primeiro: python run_single_model.py [modelo]")
        return

    df = pd.read_csv(csv_path)
    print(f"Modelos carregados: {list(results.keys())}")
    print(f"Runs no DataFrame: {len(df)}")


    # ── Fig 1: Heatmap ──
    print("[Fig 1] Logit Lens Heatmap...")
    for position in ["last_token", "keyword_token"]:
        plot_logit_lens_heatmap(
            results, fact="aviao", formulation="F1",
            model_key="llama", position=position,
            output_path=os.path.join(
                figures_dir, f"fig1_heatmap_aviao_F1_llama_{position}.png"
            )
        )

    # ── Fig 2: Cultural Divergence Map ──
    print("\n[Fig 2] Cultural Divergence Map...")
    for fact in ["aviao", "telefone"]:
        for form in ["F1", "F3"]:
            plot_cultural_divergence_map(
                results, fact=fact, formulation=form,
                output_path=os.path.join(
                    figures_dir, f"fig2_divergence_{fact}_{form}.png"
                )
            )

    # ── Fig 3: PCC Comparison ──
    print("\n[Fig 3] PCC Comparison...")
    for form in ["F1", "F2", "F3"]:
        plot_pcc_comparison(
            df, formulation=form,
            output_path=os.path.join(
                figures_dir, f"fig3_pcc_{form}.png"
            )
        )

    # ── Fig 4: Stake Analysis ──
    print("\n[Fig 4] Stake Analysis...")
    plot_stake_analysis(
        df,
        output_path=os.path.join(figures_dir, "fig4_stake.png")
    )

    # ── Fig 5: Formulation Comparison ──
    print("\n[Fig 5] Formulation Comparison...")
    for fact in ["aviao", "telefone"]:
        for model_key in ["llama", "mistral", "qwen", "gemma"]:
            plot_formulation_comparison(
                results, fact=fact, model_key=model_key,
                output_path=os.path.join(
                    figures_dir,
                    f"fig5_formulation_{fact}_{model_key}.png"
                )
            )

    # ── Fig 6: Position Comparison ──
    print("\n[Fig 6] Position Comparison (last vs keyword token)...")
    for fact in ["aviao", "telefone"]:
        plot_position_comparison(
            results, fact=fact,
            formulation="F1", model_key="llama",
            output_path=os.path.join(
                figures_dir,
                f"fig6_position_{fact}_llama_F1.png"
            )
        )

    total = len(list(Path(figures_dir).glob("*.png")))
    print(f"\n✅ {total} figuras geradas em: {figures_dir}/")


if __name__ == "__main__":
    generate_all_figures()