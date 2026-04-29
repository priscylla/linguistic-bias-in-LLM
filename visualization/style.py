import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np

# ============================================================
# PALETAS
# ============================================================

# Paleta de idiomas — distinta e acessível (colorblind-safe)
LANGUAGE_COLORS = {
    "pt": "#2166AC",  # azul — Brasil
    "en": "#D6604D",  # vermelho — EUA
    "de": "#4DAC26",  # verde — Alemanha
    "it": "#8E0152",  # vinho — Itália
}

LANGUAGE_LABELS = {
    "pt": "Portuguese (BR)",
    "en": "English (US)",
    "de": "German",
    "it": "Italian",
}

# Paleta de modelos
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

# Paleta de fatos
FACT_LABELS = {
    "aviao":    "Airplane",
    "telefone": "Telephone",
    "radio":    "Radio",
}

# Paleta de formulações
FORMULATION_STYLES = {
    "F1": {"linestyle": "-",  "label": "F1 (Direct)"},
    "F2": {"linestyle": "--", "label": "F2 (Attributive)"},
    "F3": {"linestyle": ":",  "label": "F3 (Disputative)"},
}

# Paleta de entidades culturais — para o heatmap
ENTITY_HIGHLIGHT_COLORS = {
    "local":    "#2166AC",  # entidade cultural local
    "dominant": "#D6604D",  # entidade dominante anglófona
    "other":    "#AAAAAA",  # outras entidades
}

# ============================================================
# TEMA GLOBAL — IEEE VIS style
# ============================================================

def set_ieee_style():
    """
    Aplica tema global compatível com IEEE VIS:
    - Fonte sans-serif
    - Sem bordas superiores e direitas
    - Grid sutil
    - Tamanhos adequados para paper de 2 colunas
    """
    mpl.rcParams.update({
        # Fontes
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.titlesize":  11,
        
        # Layout
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        
        # Linhas
        "lines.linewidth":   1.5,
        "lines.markersize":  5,
        
        # Figura
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
        
        # Cores
        "axes.prop_cycle": mpl.cycler(
            color=list(LANGUAGE_COLORS.values())
        ),
    })


# ============================================================
# TAMANHOS DE FIGURA — IEEE VIS 2 colunas
# ============================================================

# Largura de uma coluna IEEE VIS ≈ 3.33 polegadas
# Largura de duas colunas ≈ 7.0 polegadas

FIGURE_SIZES = {
    "single_column":      (3.33, 2.5),
    "double_column":      (7.00, 3.0),
    "double_column_tall": (7.00, 4.5),
    "square":             (3.33, 3.33),
}