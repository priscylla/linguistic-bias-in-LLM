import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional
from .style import (
    set_ieee_style, MODEL_COLORS, MODEL_LABELS,
    FACT_LABELS, FIGURE_SIZES
)


def plot_stake_analysis(
    metrics_df: pd.DataFrame,
    output_path: Optional[str] = None
):
    """
    Figura 4 — Análise de Stake Cultural.
    
    Testa a hipótese de Neutralidade Condicional:
    idiomas com stake cultural ativam entidades locais
    mais cedo (PCC menor) e com maior intensidade (IV maior).
    
    Layout: 2 painéis lado a lado.
    Painel A: PCC por stake (alto vs. baixo)
    Painel B: IV por stake (alto vs. baixo)
    """
    set_ieee_style()
    
    fig, (ax_pcc, ax_iv) = plt.subplots(
        1, 2,
        figsize=FIGURE_SIZES["double_column"],
        sharey=False
    )
    
    fig.suptitle(
        "Cultural Stake Analysis",
        fontweight="bold"
    )
    
    models = ["llama", "mistral", "qwen", "gemma"]
    
    # ── Painel A: PCC ──
    ax_pcc.set_title("(A) Point of Cultural Convergence")
    
    for model_idx, model_key in enumerate(models):
        model_df = metrics_df[metrics_df["model"] == model_key]
        
        high_stake = model_df[
            model_df["has_stake"] == True
        ]["pcc_norm"].dropna()
        
        low_stake = model_df[
            model_df["has_stake"] == False
        ]["pcc_norm"].dropna()
        
        x_base = model_idx
        
        # Box plots compactos
        bp_high = ax_pcc.boxplot(
            high_stake,
            positions=[x_base - 0.2],
            widths=0.25,
            patch_artist=True,
            boxprops=dict(
                facecolor=MODEL_COLORS[model_key],
                alpha=0.8
            ),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(
                color=MODEL_COLORS[model_key],
                alpha=0.6
            ),
            capprops=dict(
                color=MODEL_COLORS[model_key],
                alpha=0.6
            ),
            flierprops=dict(
                marker="o",
                markersize=3,
                alpha=0.4,
                markerfacecolor=MODEL_COLORS[model_key]
            ),
            manage_ticks=False
        )
        
        bp_low = ax_pcc.boxplot(
            low_stake,
            positions=[x_base + 0.2],
            widths=0.25,
            patch_artist=True,
            boxprops=dict(
                facecolor=MODEL_COLORS[model_key],
                alpha=0.3
            ),
            medianprops=dict(
                color=MODEL_COLORS[model_key],
                linewidth=2
            ),
            whiskerprops=dict(
                color=MODEL_COLORS[model_key],
                alpha=0.4
            ),
            capprops=dict(
                color=MODEL_COLORS[model_key],
                alpha=0.4
            ),
            flierprops=dict(
                marker="o",
                markersize=3,
                alpha=0.3,
                markerfacecolor=MODEL_COLORS[model_key]
            ),
            manage_ticks=False
        )
    
    ax_pcc.set_xticks(range(len(models)))
    ax_pcc.set_xticklabels(
        [MODEL_LABELS[m] for m in models],
        fontsize=7.5, rotation=15, ha="right"
    )
    ax_pcc.set_ylabel("Normalized PCC", fontsize=8)
    ax_pcc.set_ylim(0, 1)
    
    # Linha de referência — ponto médio
    ax_pcc.axhline(
        y=0.5, color="gray",
        linestyle="--", linewidth=0.7, alpha=0.5
    )
    ax_pcc.text(
        3.6, 0.51, "mid", fontsize=6.5, color="gray"
    )
    
    # ── Painel B: IV ──
    ax_iv.set_title("(B) Bias Intensity (IV)")
    
    for model_idx, model_key in enumerate(models):
        model_df = metrics_df[metrics_df["model"] == model_key]
        
        high_stake = model_df[
            model_df["has_stake"] == True
        ]["iv"].dropna()
        
        low_stake = model_df[
            model_df["has_stake"] == False
        ]["iv"].dropna()
        
        x_base = model_idx
        
        ax_iv.boxplot(
            high_stake,
            positions=[x_base - 0.2],
            widths=0.25,
            patch_artist=True,
            boxprops=dict(
                facecolor=MODEL_COLORS[model_key],
                alpha=0.8
            ),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(
                color=MODEL_COLORS[model_key], alpha=0.6
            ),
            capprops=dict(
                color=MODEL_COLORS[model_key], alpha=0.6
            ),
            flierprops=dict(
                marker="o", markersize=3, alpha=0.4,
                markerfacecolor=MODEL_COLORS[model_key]
            ),
            manage_ticks=False
        )
        
        ax_iv.boxplot(
            low_stake,
            positions=[x_base + 0.2],
            widths=0.25,
            patch_artist=True,
            boxprops=dict(
                facecolor=MODEL_COLORS[model_key],
                alpha=0.3
            ),
            medianprops=dict(
                color=MODEL_COLORS[model_key], linewidth=2
            ),
            whiskerprops=dict(
                color=MODEL_COLORS[model_key], alpha=0.4
            ),
            capprops=dict(
                color=MODEL_COLORS[model_key], alpha=0.4
            ),
            flierprops=dict(
                marker="o", markersize=3, alpha=0.3,
                markerfacecolor=MODEL_COLORS[model_key]
            ),
            manage_ticks=False
        )
    
    ax_iv.set_xticks(range(len(models)))
    ax_iv.set_xticklabels(
        [MODEL_LABELS[m] for m in models],
        fontsize=7.5, rotation=15, ha="right"
    )
    ax_iv.set_ylabel("Bias Intensity (IV)", fontsize=8)
    ax_iv.set_ylim(0, 1)
    
    # ── Legenda compartilhada ──
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor="gray", alpha=0.8, label="High cultural stake"),
        Patch(facecolor="gray", alpha=0.3, label="Low cultural stake"),
    ]
    
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=8,
        framealpha=0.95
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        print(f"✅ Figura salva: {output_path}")
    
    return fig