import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from typing import Dict, List, Optional
from .style import (
    set_ieee_style, LANGUAGE_COLORS, LANGUAGE_LABELS,
    MODEL_COLORS, MODEL_LABELS, FACT_LABELS,
    FORMULATION_STYLES, FIGURE_SIZES
)


def _compute_entity_prob_curve(
    layer_results: List[Dict],
    entities: List[str]
) -> np.ndarray:
    """
    Computa curva de probabilidade acumulada das entidades
    ao longo das camadas.
    """
    probs = []
    
    for layer_data in layer_results:
        total_prob = 0.0
        for token, prob in zip(
            layer_data["top_tokens"],
            layer_data["top_probs"]
        ):
            for entity in entities:
                if entity.lower() in token.strip().lower():
                    total_prob += prob
                    break
        probs.append(total_prob)
    
    return np.array(probs)


def plot_cultural_divergence_map(
    results: Dict,
    fact: str,
    formulation: str,
    expected_entities: Dict[str, List[str]],
    position: str = "last_token",
    show_pcc: bool = True,
    output_path: Optional[str] = None
):
    """
    Figura 2 — Cultural Divergence Map.
    
    Layout: 2×2 grid (um subplot por modelo).
    Cada subplot: curvas de probabilidade por idioma
                  ao longo das camadas normalizadas.
    
    Args:
        results:            dict completo de resultados
        fact:               fato histórico
        formulation:        formulação a plotar
        expected_entities:  dict {lang: [entidades]}
        position:           "last_token" ou "keyword_token"
        show_pcc:           marcar o ponto de convergência
        output_path:        path para salvar
    """
    set_ieee_style()
    
    models    = ["llama", "mistral", "qwen", "gemma"]
    languages = ["pt", "en", "de", "it"]
    
    fig = plt.figure(figsize=FIGURE_SIZES["double_column_tall"])
    fig.suptitle(
        f"Cultural Divergence Map — {FACT_LABELS[fact]} | "
        f"{formulation}",
        fontweight="bold"
    )
    
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.38, wspace=0.28,
        left=0.09, right=0.98,
        top=0.91, bottom=0.12
    )
    
    axes = [
        fig.add_subplot(gs[i // 2, i % 2])
        for i in range(4)
    ]
    
    for model_idx, model_key in enumerate(models):
        ax = axes[model_idx]
        
        ax.set_title(
            MODEL_LABELS[model_key],
            color=MODEL_COLORS[model_key],
            fontweight="bold",
            fontsize=9
        )
        
        pcc_markers = []
        
        for lang in languages:
            
            run_data = results.get(model_key, {}).get(
                fact, {}
            ).get(lang, {}).get(formulation, {})
            
            layer_results = run_data.get(position, [])
            if not layer_results:
                continue
            
            # Eixo X: camadas normalizadas
            x = np.array([
                r.get("layer_norm", r["layer"] / len(layer_results))
                for r in layer_results
            ])
            
            # Eixo Y: probabilidade acumulada das entidades
            entities = expected_entities.get(lang, [])
            y = _compute_entity_prob_curve(layer_results, entities)
            
            # Suavização leve com média móvel
            window = max(1, len(y) // 10)
            y_smooth = np.convolve(
                y,
                np.ones(window) / window,
                mode="same"
            )
            
            # Plotar curva
            ax.plot(
                x, y_smooth,
                color=LANGUAGE_COLORS[lang],
                label=LANGUAGE_LABELS[lang],
                alpha=0.9,
                zorder=3
            )
            
            # Área sob a curva (mais suave visualmente)
            ax.fill_between(
                x, y_smooth,
                alpha=0.08,
                color=LANGUAGE_COLORS[lang]
            )
            
            # Marcar PCC
            if show_pcc:
                pcc_layer = run_data.get(
                    "metrics_last", {}
                ).get("pcc")
                
                if pcc_layer is not None:
                    pcc_norm = run_data.get(
                        "metrics_last", {}
                    ).get("pcc_norm", pcc_layer / len(layer_results))
                    
                    # Encontrar y correspondente ao PCC
                    pcc_x_idx = np.argmin(np.abs(x - pcc_norm))
                    pcc_y = y_smooth[pcc_x_idx]
                    
                    ax.axvline(
                        x=pcc_norm,
                        color=LANGUAGE_COLORS[lang],
                        linestyle=":",
                        alpha=0.5,
                        linewidth=0.8,
                        zorder=2
                    )
                    
                    ax.scatter(
                        [pcc_norm], [pcc_y],
                        color=LANGUAGE_COLORS[lang],
                        s=25, zorder=5,
                        edgecolors="white",
                        linewidth=0.5
                    )
                    
                    pcc_markers.append((lang, pcc_norm, pcc_y))
        
        # Anotação dos PCCs
        for lang, px, py in pcc_markers:
            ax.annotate(
                f"PCC\n{px:.2f}",
                xy=(px, py),
                xytext=(px + 0.05, py + 0.03),
                fontsize=5.5,
                color=LANGUAGE_COLORS[lang],
                arrowprops=dict(
                    arrowstyle="->",
                    color=LANGUAGE_COLORS[lang],
                    lw=0.7
                )
            )
        
        # Linha de threshold
        ax.axhline(
            y=0.10,
            color="gray",
            linestyle="--",
            linewidth=0.6,
            alpha=0.5,
            label="Threshold (0.10)"
        )
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
        
        # Labels
        if model_idx in [0, 2]:  # coluna esquerda
            ax.set_ylabel("Entity Probability", fontsize=8)
        if model_idx in [2, 3]:  # linha de baixo
            ax.set_xlabel("Relative Layer Depth", fontsize=8)
        
        # Região de interesse — últimas 30% de camadas
        ax.axvspan(
            0.7, 1.0,
            alpha=0.04,
            color="gray",
            label="Late layers"
        )
    
    # ── Legenda global ──
    legend_elements = [
        Line2D([0], [0],
               color=LANGUAGE_COLORS[lang],
               linewidth=2,
               label=LANGUAGE_LABELS[lang])
        for lang in languages
    ] + [
        Line2D([0], [0],
               color="gray",
               linestyle="--",
               linewidth=1,
               label="PCC threshold (p=0.10)"),
        plt.scatter([], [],
                    marker="o",
                    color="gray",
                    s=25,
                    label="PCC marker")
    ]
    
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=7.5,
        framealpha=0.95,
        edgecolor="lightgray"
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        print(f"✅ Figura salva: {output_path}")
    
    return fig