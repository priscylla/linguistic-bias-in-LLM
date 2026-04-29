import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from typing import Dict, Optional
from .style import (
    set_ieee_style, LANGUAGE_COLORS, LANGUAGE_LABELS,
    MODEL_COLORS, MODEL_LABELS, FACT_LABELS,
    FIGURE_SIZES
)


def plot_pcc_comparison(
    metrics_df: pd.DataFrame,
    formulation: str = "F1",
    output_path: Optional[str] = None
):
    """
    Figura 3 — Comparação de PCC normalizado entre modelos.
    
    Layout: dot plot com modelos no eixo Y e PCC normalizado
    no eixo X. Agrupado por fato. Colorido por idioma.
    
    Args:
        metrics_df:   DataFrame da build_comparison_table()
        formulation:  formulação a plotar
        output_path:  path para salvar
    """
    set_ieee_style()
    
    df = metrics_df[metrics_df["formulation"] == formulation].copy()
    df = df.dropna(subset=["pcc_norm"])
    
    facts     = ["aviao", "telefone", "radio"]
    models    = ["llama", "mistral", "qwen", "gemma"]
    languages = ["pt", "en", "de", "it"]
    
    fig, axes = plt.subplots(
        1, 3,
        figsize=FIGURE_SIZES["double_column"],
        sharey=True
    )
    
    fig.suptitle(
        f"Point of Cultural Convergence (PCC) — {formulation}",
        fontweight="bold"
    )
    
    # Jitter para evitar sobreposição
    jitter_range = 0.12
    lang_offsets = {
        lang: offset
        for lang, offset in zip(
            languages,
            np.linspace(-jitter_range, jitter_range, len(languages))
        )
    }
    
    for fact_idx, fact in enumerate(facts):
        ax = axes[fact_idx]
        
        fact_df = df[df["fact"] == fact]
        
        for model_idx, model_key in enumerate(models):
            model_df = fact_df[fact_df["model"] == model_key]
            
            for lang in languages:
                lang_df = model_df[model_df["language"] == lang]
                
                if lang_df.empty:
                    continue
                
                pcc_val = lang_df["pcc_norm"].values[0]
                
                # Y position: modelo + jitter por idioma
                y_pos = model_idx + lang_offsets[lang]
                
                ax.scatter(
                    pcc_val, y_pos,
                    color=LANGUAGE_COLORS[lang],
                    s=40,
                    alpha=0.85,
                    zorder=4,
                    edgecolors="white",
                    linewidth=0.5
                )
        
        # Linhas horizontais por modelo
        for model_idx in range(len(models)):
            # Calcular média do PCC para este modelo e fato
            model_pcc_vals = df[
                (df["fact"] == fact) &
                (df["model"] == models[model_idx])
            ]["pcc_norm"].dropna()
            
            if not model_pcc_vals.empty:
                mean_pcc = model_pcc_vals.mean()
                ax.hlines(
                    model_idx,
                    xmin=0, xmax=mean_pcc,
                    colors=MODEL_COLORS[models[model_idx]],
                    linewidth=2,
                    alpha=0.3,
                    zorder=2
                )
        
        # Linha vertical — ponto médio (50% da profundidade)
        ax.axvline(
            x=0.5,
            color="gray",
            linestyle="--",
            linewidth=0.7,
            alpha=0.5
        )
        
        ax.set_xlim(0, 1)
        ax.set_title(FACT_LABELS[fact], fontweight="bold")
        ax.set_xlabel("Normalized Layer Depth", fontsize=8)
        
        # Anotação: "early" vs "late"
        ax.text(
            0.15, -0.7, "early",
            fontsize=7, color="gray", ha="center"
        )
        ax.text(
            0.85, -0.7, "late",
            fontsize=7, color="gray", ha="center"
        )
    
    # Eixo Y — nomes dos modelos (apenas no primeiro subplot)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(
        [MODEL_LABELS[m] for m in models],
        fontsize=8
    )
    axes[0].set_ylabel("Model", fontsize=8)
    
    # ── Legendas ──
    from matplotlib.lines import Line2D
    
    lang_legend = [
        plt.scatter(
            [], [],
            color=LANGUAGE_COLORS[lang],
            s=40,
            label=LANGUAGE_LABELS[lang]
        )
        for lang in languages
    ]
    
    fig.legend(
        handles=lang_legend,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=7.5,
        framealpha=0.95
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        print(f"✅ Figura salva: {output_path}")
    
    return fig