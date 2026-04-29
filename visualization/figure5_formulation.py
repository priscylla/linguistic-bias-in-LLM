import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
from .style import (
    set_ieee_style, LANGUAGE_COLORS, LANGUAGE_LABELS,
    MODEL_LABELS, FACT_LABELS, FORMULATION_STYLES,
    FIGURE_SIZES
)


def plot_formulation_comparison(
    results: Dict,
    fact: str,
    model_key: str,
    expected_entities: Dict[str, List[str]],
    position: str = "last_token",
    output_path: Optional[str] = None
):
    """
    Figura 5 — Comparação entre Formulações (F1, F2, F3).
    
    Layout: 4 subplots (um por idioma).
    Cada subplot: 3 curvas sobrepostas (F1, F2, F3).
    
    Mostra como a carga semântica da formulação
    afeta a trajetória de convergência cultural.
    """
    set_ieee_style()
    
    languages  = ["pt", "en", "de", "it"]
    formulations = ["F1", "F2", "F3"]
    
    fig, axes = plt.subplots(
        2, 2,
        figsize=FIGURE_SIZES["double_column_tall"],
        sharey=True
    )
    axes = axes.flatten()
    
    fig.suptitle(
        f"Formulation Effect — {FACT_LABELS[fact]} | "
        f"{MODEL_LABELS[model_key]}",
        fontweight="bold"
    )
    
    for lang_idx, lang in enumerate(languages):
        ax = axes[lang_idx]
        
        ax.set_title(
            LANGUAGE_LABELS[lang],
            color=LANGUAGE_COLORS[lang],
            fontweight="bold",
            fontsize=9
        )
        
        entities = expected_entities.get(lang, [])
        
        for formulation in formulations:
            
            run_data = results.get(model_key, {}).get(
                fact, {}
            ).get(lang, {}).get(formulation, {})
            
            layer_results = run_data.get(position, [])
            if not layer_results:
                continue
            
            x = np.array([
                r.get("layer_norm", r["layer"] / len(layer_results))
                for r in layer_results
            ])
            
            y = np.array([
                sum(
                    prob for token, prob in zip(
                        r["top_tokens"], r["top_probs"]
                    )
                    if any(
                        e.lower() in token.strip().lower()
                        for e in entities
                    )
                )
                for r in layer_results
            ])
            
            # Suavização
            window = max(1, len(y) // 8)
            y_smooth = np.convolve(
                y, np.ones(window) / window, mode="same"
            )
            
            style = FORMULATION_STYLES[formulation]
            
            ax.plot(
                x, y_smooth,
                color=LANGUAGE_COLORS[lang],
                linestyle=style["linestyle"],
                label=style["label"],
                alpha=0.9
            )
            
            # Marcar ponto final (IV)
            ax.scatter(
                [x[-1]], [y_smooth[-1]],
                color=LANGUAGE_COLORS[lang],
                s=20, zorder=5,
                marker=["o", "s", "^"][
                    formulations.index(formulation)
                ]
            )
        
        ax.axhline(
            y=0.10, color="gray",
            linestyle="--", linewidth=0.6, alpha=0.5
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        
        if lang_idx in [0, 2]:
            ax.set_ylabel("Entity Probability", fontsize=8)
        if lang_idx in [2, 3]:
            ax.set_xlabel("Relative Layer Depth", fontsize=8)
        
        # Anotação de F3 vs F1 delta
        f1_data = results.get(model_key, {}).get(
            fact, {}
        ).get(lang, {}).get("F1", {})
        
        f3_data = results.get(model_key, {}).get(
            fact, {}
        ).get(lang, {}).get("F3", {})
        
        f1_iv = f1_data.get("metrics_last", {}).get("iv", 0)
        f3_iv = f3_data.get("metrics_last", {}).get("iv", 0)
        delta = f3_iv - f1_iv
        
        delta_color = (
            "#2166AC" if delta > 0.02
            else "#D6604D" if delta < -0.02
            else "gray"
        )
        
        ax.text(
            0.98, 0.95,
            f"ΔIV(F3-F1): {delta:+.3f}",
            transform=ax.transAxes,
            fontsize=7,
            ha="right", va="top",
            color=delta_color,
            fontweight="bold"
        )
    
    # Legenda de formulações
    from matplotlib.lines import Line2D
    
    form_legend = [
        Line2D(
            [0], [0],
            color="gray",
            linestyle=FORMULATION_STYLES[f]["linestyle"],
            linewidth=2,
            label=FORMULATION_STYLES[f]["label"]
        )
        for f in formulations
    ]
    
    fig.legend(
        handles=form_legend,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=8,
        framealpha=0.95
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        print(f"✅ Figura salva: {output_path}")
    
    return fig