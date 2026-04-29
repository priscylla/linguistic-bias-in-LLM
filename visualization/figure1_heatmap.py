import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
from .style import (
    set_ieee_style, LANGUAGE_COLORS, LANGUAGE_LABELS,
    FACT_LABELS, FIGURE_SIZES, ENTITY_HIGHLIGHT_COLORS
)


def _classify_token(
    token: str,
    local_entities: List[str],
    dominant_entities: List[str]
) -> str:
    """
    Classifica um token como 'local', 'dominant' ou 'other'
    para colorização no heatmap.
    """
    token_lower = token.strip().lower()
    
    for entity in local_entities:
        if entity.lower() in token_lower:
            return "local"
    
    for entity in dominant_entities:
        if entity.lower() in token_lower:
            return "dominant"
    
    return "other"


def plot_logit_lens_heatmap(
    results: Dict,
    fact: str,
    formulation: str,
    model_key: str,
    local_entities: Dict[str, List[str]],
    dominant_entities: Dict[str, List[str]],
    position: str = "last_token",
    n_layers_to_show: int = 20,
    top_k_show: int = 5,
    output_path: Optional[str] = None
):
    """
    Figura 1 — Logit Lens Heatmap.
    
    Layout: 4 subplots lado a lado (um por idioma).
    Cada subplot: matriz [camadas × top-K tokens].
    Células coloridas por tipo de entidade.
    Intensidade da cor = probabilidade.
    
    Args:
        results:           dict completo de resultados
        fact:              "aviao", "telefone" ou "radio"
        formulation:       "F1", "F2" ou "F3"
        model_key:         chave do modelo
        local_entities:    dict {lang: [entidades locais]}
        dominant_entities: dict {lang: [entidades dominantes]}
        position:          "last_token" ou "keyword_token"
        n_layers_to_show:  número de camadas a mostrar
                           (selecionadas uniformemente)
        top_k_show:        número de top tokens a mostrar
        output_path:       path para salvar a figura
    """
    set_ieee_style()
    
    languages = ["pt", "en", "de", "it"]
    
    fig = plt.figure(figsize=FIGURE_SIZES["double_column_tall"])
    fig.suptitle(
        f"Logit Lens — {FACT_LABELS[fact]} | "
        f"{model_key.upper()} | {formulation}",
        fontweight="bold", y=1.02
    )
    
    # GridSpec com espaçamento fino entre subplots
    gs = gridspec.GridSpec(
        1, 4,
        figure=fig,
        wspace=0.08,
        left=0.08, right=0.98
    )
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    for ax_idx, lang in enumerate(languages):
        ax = axes[ax_idx]
        
        # Dados desta língua
        run_data = results[model_key][fact][lang][formulation]
        layer_results = run_data.get(position, [])
        
        if not layer_results:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        
        n_layers_total = len(layer_results)
        
        # Selecionar camadas uniformemente distribuídas
        layer_indices = np.linspace(
            0, n_layers_total - 1, n_layers_to_show, dtype=int
        )
        selected_layers = [layer_results[i] for i in layer_indices]
        
        # Construir matriz de células
        # Cada célula: (token_text, probability, entity_type)
        local_ents    = local_entities.get(lang, [])
        dominant_ents = dominant_entities.get(lang, [])
        
        cell_data = []
        layer_labels = []
        
        for layer_data in selected_layers:
            row = []
            for k in range(top_k_show):
                token = layer_data["top_tokens"][k]
                prob  = layer_data["top_probs"][k]
                etype = _classify_token(token, local_ents, dominant_ents)
                row.append((token.strip(), prob, etype))
            cell_data.append(row)
            
            # Label da camada — usar valor normalizado
            layer_norm = layer_data.get("layer_norm", 
                                        layer_data["layer"] / n_layers_total)
            layer_labels.append(f"{layer_norm:.2f}")
        
        # ── Renderizar o heatmap como tabela de texto ──
        # Cada célula é um retângulo colorido com texto
        
        n_rows = len(cell_data)      # camadas
        n_cols = top_k_show          # top-K tokens
        
        cell_width  = 1.0 / n_cols
        cell_height = 1.0 / n_rows
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        
        for row_idx, row in enumerate(cell_data):
            # Y de cima para baixo (camadas iniciais no topo)
            y = 1.0 - (row_idx + 1) * cell_height
            
            for col_idx, (token, prob, etype) in enumerate(row):
                x = col_idx * cell_width
                
                # Cor base por tipo de entidade
                base_color = ENTITY_HIGHLIGHT_COLORS[etype]
                
                # Alpha proporcional à probabilidade
                # (mínimo 0.1 para visibilidade)
                alpha = max(0.1, min(1.0, prob * 8))
                
                # Retângulo colorido
                rect = mpatches.FancyBboxPatch(
                    (x + 0.002, y + 0.001),
                    cell_width - 0.004,
                    cell_height - 0.002,
                    boxstyle="round,pad=0.01",
                    facecolor=base_color,
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=0.3
                )
                ax.add_patch(rect)
                
                # Texto do token
                # Cor do texto: branco para alta probabilidade,
                # preto para baixa
                text_color = "white" if alpha > 0.5 else "black"
                
                # Truncar token longo
                display_token = token[:8] if len(token) > 8 else token
                
                ax.text(
                    x + cell_width / 2,
                    y + cell_height / 2,
                    display_token,
                    ha="center", va="center",
                    fontsize=5.5,
                    color=text_color,
                    fontweight="bold" if etype != "other" else "normal"
                )
                
                # Probabilidade em fonte menor
                ax.text(
                    x + cell_width / 2,
                    y + cell_height * 0.2,
                    f"{prob:.2f}",
                    ha="center", va="center",
                    fontsize=4,
                    color=text_color,
                    alpha=0.8
                )
        
        # Labels do eixo Y (camadas normalizadas)
        ax.set_yticks(
            [1.0 - (i + 0.5) * cell_height for i in range(n_rows)]
        )
        ax.set_yticklabels(
            layer_labels,
            fontsize=5.5
        )
        
        # Remover ticks do eixo X
        ax.set_xticks([])
        
        # Título do subplot = idioma
        ax.set_title(
            LANGUAGE_LABELS[lang],
            color=LANGUAGE_COLORS[lang],
            fontweight="bold",
            fontsize=9,
            pad=4
        )
        
        # Label do eixo Y apenas no primeiro subplot
        if ax_idx == 0:
            ax.set_ylabel("Relative Layer Depth", fontsize=8)
        else:
            ax.set_yticklabels([])
    
    # ── Legenda global ──
    legend_elements = [
        mpatches.Patch(
            facecolor=ENTITY_HIGHLIGHT_COLORS["local"],
            label="Local cultural entity",
            alpha=0.8
        ),
        mpatches.Patch(
            facecolor=ENTITY_HIGHLIGHT_COLORS["dominant"],
            label="Dominant (anglophone) entity",
            alpha=0.8
        ),
        mpatches.Patch(
            facecolor=ENTITY_HIGHLIGHT_COLORS["other"],
            label="Other token",
            alpha=0.4
        ),
    ]
    
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=7.5,
        framealpha=0.9
    )
    
    # Anotação de posição analisada
    fig.text(
        0.98, -0.04,
        f"Position: {'Last token' if position == 'last_token' else 'Keyword token'}",
        ha="right", fontsize=7, style="italic", color="gray"
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        print(f"✅ Figura salva: {output_path}")
    
    return fig