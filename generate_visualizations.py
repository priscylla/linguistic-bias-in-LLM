"""
generate_visualizations.py
Gera todas as figuras do paper em duas versoes:
  - Relativa: eixo X = relative layer depth (0->1)
  - Absoluta: eixo X = numero absoluto da camada (L1->Ln)

Todas as figuras que envolvem camadas sao geradas para
AMBOS os fatos (aviao e telefone).

Arquivos gerados:
  fig1_divergence_map.png
  fig1_divergence_map_abs.png
  fig2_cs_heatmap.png
  fig3_logit_lens_{aviao|telefone}.png
  fig3_logit_lens_{aviao|telefone}_abs.png
  fig4_behavioral.png
  fig5_convergence_{aviao|telefone}.png
  fig5_convergence_{aviao|telefone}_abs.png
  fig6_peak_vs_final.png
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Paletas ───────────────────────────────────────────────────
LANG_COLORS  = {"pt":"#2166AC","en":"#D6604D","de":"#4DAC26","it":"#8E0152"}
LANG_LABELS  = {"pt":"Portuguese (BR)","en":"English (US)",
                "de":"German","it":"Italian"}
MODEL_COLORS = {"llama":"#E69F00","mistral":"#56B4E9",
                "qwen":"#009E73","gemma":"#CC79A7"}
MODEL_LABELS = {"llama":"LLaMA 3.1 8B","mistral":"Mistral 7B",
                "qwen":"Qwen 2.5 7B","gemma":"Gemma 2 9B"}
MODEL_LAYERS = {"llama":32,"mistral":32,"qwen":28,"gemma":42}
FACT_LABELS  = {"aviao":"Airplane","telefone":"Telephone"}

# Tokens para colorir no heatmap do Logit Lens
ENTITY_TOKENS = {
    "aviao": {
        "local":    ["wright","brothers","orville","wilbur","kitty","santos","dumont"],
        "competing":["santos","dumont","alberto"],
    },
    "telefone": {
        "local":    ["bell","graham","alexander","aless","al","scot",
                     "meucci","antonio","alessan"],
        "competing":["meucci","antonio","bell","graham","alexander"],
    }
}

# Cores das entidades no heatmap
ENTITY_COLOR_LOCAL     = "#D6604D"   # vermelho — entidade esperada
ENTITY_COLOR_COMPETING = "#2166AC"   # azul — entidade concorrente

# Stake cultural por fato
CULTURAL_STAKE = {"aviao":"pt","telefone":"it"}

# ── Helpers ───────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "font.family":"sans-serif","font.size":9,
        "axes.titlesize":10,"axes.labelsize":9,
        "xtick.labelsize":8,"ytick.labelsize":8,
        "legend.fontsize":8,"figure.titlesize":11,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.alpha":0.2,"grid.linestyle":"--",
        "lines.linewidth":1.8,"figure.dpi":150,
        "savefig.dpi":300,"savefig.bbox":"tight",
    })

def smooth(arr, w=3):
    if len(arr) <= w: return np.array(arr)
    return np.convolve(arr, np.ones(w)/w, mode="same")

def load_data():
    data = {}
    for m in ["llama","mistral","qwen","gemma"]:
        p = f"results/{m}_results.json"
        if os.path.exists(p):
            with open(p) as f:
                data[m] = json.load(f)
    return data

def token_color(token, fact):
    """Cor do token baseada no fato sendo analisado."""
    t = token.strip().lower()
    # Para aviao: vermelho=Wright, azul=Santos
    # Para telefone: vermelho=Bell/Alexander, azul=Meucci/Antonio
    if fact == "aviao":
        if any(e in t for e in ["wright","brothers","orville","wilbur","kitty"]):
            return ENTITY_COLOR_LOCAL
        if any(e in t for e in ["santos","dumont","alberto"]):
            return ENTITY_COLOR_COMPETING
    else:  # telefone
        if any(e in t for e in ["bell","graham","alexander","aless","al","scot"]):
            return ENTITY_COLOR_LOCAL
        if any(e in t for e in ["meucci","antonio","alessan"]):
            return ENTITY_COLOR_COMPETING
    return "#AAAAAA"

def get_x_axis(run, model, use_absolute):
    """Retorna (x, xlabel, xlim, xticks) para o eixo de camadas."""
    layers_data = run.get("last_token", [])
    n_l = MODEL_LAYERS[model]
    if use_absolute:
        x      = np.array([l["layer"] for l in layers_data]) if layers_data \
                 else np.arange(1, n_l+1)
        xlabel = "Layer"
        xlim   = (1, n_l)
        step   = max(4, n_l // 8)
        xticks = list(range(1, n_l+1, step))
    else:
        x      = np.array([l["layer_norm"] for l in layers_data]) if layers_data \
                 else np.linspace(0, 1, n_l)
        xlabel = "Relative Layer Depth"
        xlim   = (0, 1)
        xticks = [0, 0.25, 0.5, 0.75, 1.0]
    return x, xlabel, xlim, xticks

def decision_zone(ax, model, use_absolute):
    """Adiciona zona de decisao (ultimas 30%)."""
    n_l = MODEL_LAYERS[model]
    if use_absolute:
        start = int(n_l * 0.70)
        ax.axvspan(start, n_l, alpha=0.05, color="gray")
        ax.text(start + 0.3, ax.get_ylim()[1] * 0.92,
                f"L{start}", fontsize=6, color="gray", alpha=0.7, va="top")
    else:
        ax.axvspan(0.70, 1.0, alpha=0.05, color="gray")

def ref_lines(ax, model, use_absolute):
    """Linhas de referencia em 25/50/75% da profundidade."""
    n_l = MODEL_LAYERS[model]
    for pct in [0.25, 0.50, 0.75]:
        xmark = int(n_l*pct) if use_absolute else pct
        label = f"L{int(n_l*pct)}" if use_absolute else f"{pct:.0%}"
        ax.axvline(xmark, color="gray", lw=0.5, linestyle=":", alpha=0.4)
        ax.text(xmark, ax.get_ylim()[0], label,
                fontsize=6, color="gray",
                ha="center", va="bottom", alpha=0.6)

def save_fig(fig, output):
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"  OK  {output}")


# ══════════════════════════════════════════════════════════════
# FIG 1 — CULTURAL DIVERGENCE MAP
# Gerada para cada fato separadamente
# ══════════════════════════════════════════════════════════════


def fig1_divergence_map(data, output, use_absolute=False):
    set_style()
    models = ["llama","mistral","qwen","gemma"]
    langs  = ["pt","en","de","it"]
    facts  = ["aviao","telefone"]
    v = "Absolute Layer Number" if use_absolute else "Relative Layer Depth (0->1)"
    title = "Cultural Divergence Map | CS = P(expected) - P(competing) | F1 | " + v

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharey=False)
    fig.suptitle(title, fontweight="bold", fontsize=10, y=1.01)

    for row, fact in enumerate(facts):
        for col, model in enumerate(models):
            ax  = axes[row][col]
            n_l = MODEL_LAYERS[model]

            if row == 0:
                sub = " (%d layers)" % n_l if use_absolute else ""
                ax.set_title(MODEL_LABELS[model] + sub,
                             color=MODEL_COLORS[model],
                             fontweight="bold", fontsize=8.5)

            for lang in langs:
                run   = data.get(model,{}).get(fact,{}).get(lang,{}).get("F1",{})
                m_res = run.get("metrics_last",{})
                curve = m_res.get("commitment_curve",[])
                if not curve: continue

                x, xlabel, xlim, xticks = get_x_axis(run, model, use_absolute)
                y = smooth(np.array(curve), w=3)

                ax.plot(x, y, color=LANG_COLORS[lang], alpha=0.85, linewidth=1.6)
                ax.fill_between(x, y, 0, color=LANG_COLORS[lang], alpha=0.06)

                cs = m_res.get("commitment_final", 0)
                if abs(cs) > 0.05:
                    offset = n_l*0.12 if use_absolute else 0.12
                    ax.annotate("%+.2f" % cs,
                        xy=(x[-1], y[-1]),
                        xytext=(x[-1]-offset, y[-1]),
                        fontsize=6, color=LANG_COLORS[lang],
                        fontweight="bold", va="center")

            ax.axhline(0, color="gray", lw=0.7, linestyle="--", alpha=0.6)
            decision_zone(ax, model, use_absolute)

            if col == 0:
                ax.set_ylabel(FACT_LABELS[fact] + "\nCommitment Score",
                              fontsize=8, fontweight="bold")
            if row == 1:
                ax.set_xlabel(xlabel if col == 1 else "", fontsize=8)

            dummy = data.get(model,{}).get(fact,{}).get("pt",{}).get("F1",{})
            _, _, xlim, xticks = get_x_axis(dummy, model, use_absolute)
            ax.set_xticks(xticks)
            ax.set_xlim(xlim)

    leg = [Line2D([0],[0], color=LANG_COLORS[l], lw=2,
                  label=LANG_LABELS[l]) for l in langs]
    leg += [
        mpatches.Patch(facecolor="gray", alpha=0.15, label="Decision zone (last 30%)"),
        Line2D([0],[0], color="gray", lw=1, linestyle="--", label="CS = 0 (neutral)")
    ]
    fig.legend(handles=leg, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5,-0.04), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output)

def fig2_cs_heatmap(data, output):
    set_style()
    models = ["llama","mistral","qwen","gemma"]
    langs  = ["pt","en","de","it"]
    facts  = ["aviao","telefone"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        "Commitment Score at Peak Layer  |  F1 formulation\n"
        "Blue = favors local entity at peak   |   Red = favors competing entity at peak",
        fontweight="bold", fontsize=10
    )

    for ax, fact in zip(axes, facts):
        matrix = np.zeros((len(models), len(langs)))
        for i, model in enumerate(models):
            for j, lang in enumerate(langs):
                run = data.get(model,{}).get(fact,{}).get(lang,{}).get("F1",{})
                m   = run.get("metrics_last",{})
                # cs_at_peak: CS no pico de P(local) -- captura o vies cultural
                # nas camadas intermediarias, antes da supressao gramatical.
                # Se nao existir (JSONs antigos), cai para commitment_final.
                matrix[i][j] = m.get("cs_at_peak", m.get("commitment_final", 0))

        im = ax.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        for i in range(len(models)):
            for j in range(len(langs)):
                val   = matrix[i][j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:+.2f}",
                        ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels([LANG_LABELS[l] for l in langs],
                           fontsize=8, rotation=15, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
        ax.set_title(f"{FACT_LABELS[fact]}\n(+ favors local | − favors anglophone)",
                     fontweight="bold", fontsize=9)

        stake_lang = CULTURAL_STAKE[fact]
        j_stake = langs.index(stake_lang)
        ax.add_patch(mpatches.Rectangle(
            (j_stake-0.5,-0.5), 1, len(models),
            lw=2, edgecolor="gold", facecolor="none", zorder=5))

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label("Commitment Score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.legend(
        handles=[mpatches.Patch(facecolor="none", edgecolor="gold",
                                lw=2, label="Cultural stake language")],
        loc="lower center", bbox_to_anchor=(0.5,-0.05), fontsize=8)

    save_fig(fig, output)


# ══════════════════════════════════════════════════════════════
# FIG 3 — LOGIT LENS HEATMAP
# Gerada para cada fato separadamente
# Mostra LLaMA e Gemma x idiomas com e sem stake
# ══════════════════════════════════════════════════════════════

def fig3_logit_lens_heatmap(data, fact, output, use_absolute=False):
    set_style()
    version = "Absolute Layer Number" if use_absolute else "Relative Layer Depth (0→1)"

    stake_lang   = CULTURAL_STAKE[fact]
    neutral_lang = "en"

    cases = [
        ("llama", fact, neutral_lang, "F1"),
        ("llama", fact, stake_lang,   "F1"),
        ("gemma", fact, neutral_lang, "F1"),
        ("gemma", fact, stake_lang,   "F1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    fig.suptitle(
        f"Logit Lens — Top Predicted Tokens per Layer\n"
        f"{FACT_LABELS[fact]}  |  Intensity ∝ probability  |  {version}",
        fontweight="bold", fontsize=10, y=1.01
    )

    for ax, (model, f, lang, form) in zip(axes, cases):
        run    = data.get(model,{}).get(f,{}).get(lang,{}).get(form,{})
        layers = run.get("last_token",[])
        if not layers:
            ax.text(0.5,0.5,"No data",ha="center",va="center")
            continue

        n_layers = len(layers)
        n_show, top_k = 20, 5

        idxs = np.linspace(0, n_layers-1, n_show, dtype=int)
        sel  = [layers[i] for i in idxs]

        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_aspect("auto"); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)

        cw, ch = 1.0/top_k, 1.0/n_show
        layer_labels = []

        for row_i, layer in enumerate(sel):
            y = 1.0-(row_i+1)*ch
            layer_labels.append(
                f"L{layer['layer']}" if use_absolute
                else f"{layer['layer_norm']:.2f}"
            )
            for col_i in range(top_k):
                if col_i >= len(layer["top_tokens"]): break
                token = layer["top_tokens"][col_i]
                prob  = layer["top_probs"][col_i]
                color = token_color(token, f)
                alpha = float(np.clip(prob*6, 0.06, 0.95))
                x = col_i*cw

                ax.add_patch(mpatches.FancyBboxPatch(
                    (x+0.003,y+0.001), cw-0.006, ch-0.002,
                    boxstyle="round,pad=0.006",
                    facecolor=color, alpha=alpha,
                    edgecolor="white", linewidth=0.4, zorder=2))

                tc = "white" if alpha > 0.5 else "#333"
                ax.text(x+cw/2, y+ch*0.62, token.strip()[:8],
                        ha="center", va="center", fontsize=5.5,
                        color=tc,
                        fontweight="bold" if color!="#AAAAAA" else "normal",
                        zorder=3)
                ax.text(x+cw/2, y+ch*0.22, f"{prob:.3f}",
                        ha="center", va="center",
                        fontsize=4, color=tc, alpha=0.85, zorder=3)

        ax.set_yticks([1.0-(i+0.5)*ch for i in range(n_show)])
        ax.set_yticklabels(layer_labels, fontsize=5.5)
        ax.set_xticks([])

        # Zonas early/mid/late
        for frac in [0.33, 0.67]:
            ax.axhline(1.0-int(n_show*frac)*ch,
                       color="#888", lw=0.8, linestyle=":", alpha=0.5)

        ax.set_ylabel("Layer" if use_absolute else "Relative Depth", fontsize=7)

        m_res  = run.get("metrics_last",{})
        cs     = m_res.get("commitment_final",0)
        peak   = m_res.get("iv_local_peak",0)
        p_l    = m_res.get("peak_layer","?")
        n_l    = MODEL_LAYERS[model]
        prompt = run.get("prompt","")
        ax.set_title(
            f"{MODEL_LABELS[model]} ({n_l}L) | {LANG_LABELS[lang]}"
            + (" ★ stake" if lang == stake_lang else "") + "\n"
            f"\"{prompt}\"\n"
            f"CS_final={cs:+.3f}  peak={peak:.3f}@L{p_l}",
            fontsize=7.5, fontweight="bold"
        )

    # Legenda dinâmica por fato
    local_label = ("Wright/Brothers tokens" if fact=="aviao"
                   else "Bell/Alexander tokens")
    comp_label  = ("Santos/Dumont tokens" if fact=="aviao"
                   else "Meucci/Antonio tokens")
    leg = [
        mpatches.Patch(facecolor=ENTITY_COLOR_LOCAL,    alpha=0.8, label=local_label),
        mpatches.Patch(facecolor=ENTITY_COLOR_COMPETING, alpha=0.8, label=comp_label),
        mpatches.Patch(facecolor="#AAAAAA",             alpha=0.5, label="Other tokens"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.03), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output)


# ══════════════════════════════════════════════════════════════
# FIG 4 — BEHAVIORAL ANALYSIS (unica versao, ambos fatos)
# ══════════════════════════════════════════════════════════════

def fig4_behavioral(data, output):
    set_style()
    models = ["llama","mistral","qwen","gemma"]
    langs  = ["pt","en","de","it"]

    def classify(response, fact):
        r = response.lower()
        if fact == "aviao":
            has_s = any(w in r for w in ["santos","dumont"])
            has_w = any(w in r for w in ["wright","orville","wilbur","irmãos","hermanos"])
            if has_s and has_w:
                ps = min([r.find(w) for w in ["santos","dumont"] if w in r],default=999)
                pw = min([r.find(w) for w in ["wright","orville","wilbur"] if w in r],default=999)
                return "Both\n(Santos 1st)" if ps<pw else "Both\n(Wright 1st)"
            if has_s: return "Santos\nDumont"
            if has_w: return "Wright\nBrothers"
            return "Ambiguous /\nNo name"
        else:
            has_m = any(w in r for w in ["meucci","antonio meucci"])
            has_b = any(w in r for w in ["bell","graham","alexander"])
            if has_m and has_b:
                pm = min([r.find(w) for w in ["meucci"] if w in r],default=999)
                pb = min([r.find(w) for w in ["bell","graham","alexander"] if w in r],default=999)
                return "Both\n(Meucci 1st)" if pm<pb else "Both\n(Bell 1st)"
            if has_m: return "Meucci"
            if has_b: return "Bell"
            return "Ambiguous /\nNo name"

    cat_colors = {
        "Santos\nDumont":"#2166AC","Wright\nBrothers":"#D6604D",
        "Both\n(Santos 1st)":"#6BAED6","Both\n(Wright 1st)":"#FC8D59",
        "Meucci":"#2166AC","Bell":"#D6604D",
        "Both\n(Meucci 1st)":"#6BAED6","Both\n(Bell 1st)":"#FC8D59",
        "Ambiguous /\nNo name":"#AAAAAA",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Behavioral Analysis — Model Responses  |  F1 formulation\n"
        "What the model says when completing the sentence",
        fontweight="bold", fontsize=10
    )

    for ax, fact in zip(axes, ["aviao","telefone"]):
        for i, model in enumerate(models):
            for j, lang in enumerate(langs):
                run   = data.get(model,{}).get(fact,{}).get(lang,{}).get("F1",{})
                resp  = run.get("response","")
                cat   = classify(resp, fact)
                color = cat_colors.get(cat,"#AAAAAA")

                ax.add_patch(mpatches.FancyBboxPatch(
                    (j+0.05,i+0.05), 0.90, 0.90,
                    boxstyle="round,pad=0.05",
                    facecolor=color, alpha=0.85,
                    edgecolor="white", linewidth=1.5))

                tc = "white" if color!="#AAAAAA" else "#444"
                ax.text(j+0.5, i+0.5, cat,
                        ha="center", va="center",
                        fontsize=7.5, color=tc, fontweight="bold")

        ax.set_xlim(0, len(langs)); ax.set_ylim(0, len(models))
        ax.set_aspect("equal"); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)

        ax.set_xticks([j+0.5 for j in range(len(langs))])
        ax.set_xticklabels([LANG_LABELS[l] for l in langs],
                           fontsize=8, rotation=15, ha="right")
        ax.set_yticks([i+0.5 for i in range(len(models))])
        ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=8)

        stake_lang = CULTURAL_STAKE[fact]
        j_stake    = langs.index(stake_lang)
        ax.add_patch(mpatches.Rectangle(
            (j_stake,0), 1, len(models),
            lw=2.5, edgecolor="gold", facecolor="none", zorder=5))
        ax.text(j_stake+0.5, len(models)+0.1, "stake",
                ha="center", fontsize=7, color="goldenrod", fontweight="bold")

        prompt = list(data.values())[0][fact]["pt"]["F1"].get("prompt","")
        ax.set_title(f"{FACT_LABELS[fact]}\nPrompt: \"{prompt}\"",
                     fontweight="bold", fontsize=9)

    leg = [
        mpatches.Patch(facecolor="#D6604D", alpha=0.85,
                       label="Anglophone entity (Wright / Bell)"),
        mpatches.Patch(facecolor="#2166AC", alpha=0.85,
                       label="Local entity (Santos Dumont / Meucci)"),
        mpatches.Patch(facecolor="#6BAED6", alpha=0.85, label="Both mentioned"),
        mpatches.Patch(facecolor="#AAAAAA", alpha=0.6,  label="Ambiguous / No name"),
        mpatches.Patch(facecolor="none", edgecolor="gold",
                       lw=2, label="Cultural stake language"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.08), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output)


# ══════════════════════════════════════════════════════════════
# FIG 5 — CONVERGENCE
# Gerada para cada fato separadamente
# ══════════════════════════════════════════════════════════════

def fig5_convergence(data, fact, output, use_absolute=False):
    set_style()
    langs   = ["pt","en","de","it"]
    version = "Absolute Layer Number" if use_absolute else "Relative Layer Depth (0→1)"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Layer Convergence — Commitment Score Trajectory\n"
        f"{FACT_LABELS[fact]}  |  F1 formulation  |  {version}",
        fontweight="bold", fontsize=10
    )

    for ax, model in zip(axes, ["llama","gemma"]):
        n_l = MODEL_LAYERS[model]
        ax.set_title(f"{MODEL_LABELS[model]}  ({n_l} layers)",
                     color=MODEL_COLORS[model],
                     fontweight="bold", fontsize=9.5)

        for lang in langs:
            run   = data.get(model,{}).get(fact,{}).get(lang,{}).get("F1",{})
            m_res = run.get("metrics_last",{})
            curve = m_res.get("commitment_curve",[])
            if not curve: continue

            x, xlabel, xlim, xticks = get_x_axis(run, model, use_absolute)
            y = smooth(np.array(curve), w=4)

            ax.plot(x, y, color=LANG_COLORS[lang],
                    linewidth=2.2, alpha=0.9, label=LANG_LABELS[lang])
            ax.fill_between(x, y, 0, color=LANG_COLORS[lang], alpha=0.07)

            cs_final = m_res.get("commitment_final",0)
            if abs(cs_final) > 0.03:
                offset = (n_l*0.1 if use_absolute else 0.08)
                ax.annotate(f"{cs_final:+.2f}",
                    xy=(x[-1], y[-1]),
                    xytext=(x[-1]-offset, y[-1]),
                    fontsize=8.5, color=LANG_COLORS[lang],
                    fontweight="bold", va="center", ha="right")

        ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
        decision_zone(ax, model, use_absolute)
        ref_lines(ax, model, use_absolute)

        ax.set_xticks(xticks)
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.legend(fontsize=8, loc="upper left",
                  framealpha=0.9, edgecolor="lightgray")

    axes[0].set_ylabel("Commitment Score\n(P_local − P_competing)", fontsize=8.5)

    save_fig(fig, output)


# ══════════════════════════════════════════════════════════════
# FIG 6 — PEAK vs FINAL LAYER
# Ambos fatos, unica figura
# ══════════════════════════════════════════════════════════════

def fig6_peak_vs_final(data, output):
    set_style()
    models = ["llama","mistral","qwen","gemma"]
    langs  = ["pt","en","de","it"]
    facts  = ["aviao","telefone"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Peak vs. Final Layer — Cultural Entity Probability\n"
        "Circle = peak layer  |  Square = final layer  |  Line = suppression gap",
        fontweight="bold", fontsize=9, y=1.04
    )

    # Dumbbell chart: circulo cheio = peak, X = final
    # Linha vertical conectando os dois valores
    # Muito mais legivel do que barras solidas vs transparentes

    for ax, fact in zip(axes, facts):
        ax.set_title(FACT_LABELS[fact], fontweight="bold", fontsize=10)

        x      = np.arange(len(langs))
        width  = 0.18
        offset = -(len(models)-1)/2 * width

        for i, model in enumerate(models):
            peaks, finals, suppressions = [], [], []
            for lang in langs:
                run = data.get(model,{}).get(fact,{}).get(lang,{}).get("F1",{})
                m   = run.get("metrics_last",{})
                pk  = m.get("iv_local_peak",  None)
                fn  = m.get("iv_local_final", None)

                # Calcular na hora se metricas novas nao existirem (JSONs antigos)
                if pk is None:
                    layers = run.get("last_token", [])
                    from config.experiment_config import ExperimentConfig
                    from logit_lens.analyzer import LogitLensAnalyzer
                    _cfg  = ExperimentConfig()
                    _ana  = LogitLensAnalyzer(_cfg.expected_entities[fact])
                    _comp = _cfg.competing_entities[fact]
                    _m    = _ana.compute_all_metrics(layers, lang, _comp) if layers else {}
                    pk = _m.get("iv_local_peak",  0)
                    fn = _m.get("iv_local_final", 0)
                else:
                    fn = fn if fn is not None else 0

                peaks.append(pk)
                finals.append(fn)
                suppressions.append(pk - fn)

            xpos = x + offset + i*width

            for xi, pk, fn, sup in zip(xpos, peaks, finals, suppressions):
                # Linha vertical peak -> final (representa a supressao)
                if pk > 0.01:
                    ax.plot([xi, xi], [fn, pk],
                            color=MODEL_COLORS[model],
                            lw=2.0, alpha=0.5, zorder=2)

                # Circulo cheio = Peak (o que o modelo SABE internamente)
                ax.plot(xi, pk, "o",
                        color=MODEL_COLORS[model],
                        markersize=8, alpha=0.95,
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        zorder=4,
                        label=MODEL_LABELS[model] if (xi == xpos[0]) else "")

                # Quadrado = Final (o que o modelo GERA na ultima camada)
                ax.plot(xi, fn, "s",
                        color="white",
                        markersize=7, alpha=1.0,
                        markeredgecolor=MODEL_COLORS[model],
                        markeredgewidth=2.0,
                        zorder=5)

                # Anotar supressao quando significativa
                if sup > 0.15 and pk > 0.1:
                    ax.text(xi + 0.03, (pk + fn) / 2,
                            "%.2f" % sup,
                            fontsize=5.5, color=MODEL_COLORS[model],
                            va="center", ha="left", alpha=0.8,
                            fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABELS[l] for l in langs],
                           fontsize=8, rotation=12, ha="right")
        ax.set_ylabel("P(cultural entity)", fontsize=8.5)
        ax.set_ylim(-0.03, 1.08)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.4)

        stake_lang = CULTURAL_STAKE[fact]
        j_stake    = langs.index(stake_lang)
        ax.axvspan(j_stake-0.5, j_stake+0.5, alpha=0.06, color="gold", zorder=0)
        ax.text(j_stake, 1.04, "stake",
                ha="center", fontsize=7, color="goldenrod", fontweight="bold")

    # Legenda
    model_leg = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=MODEL_COLORS[m],
               markeredgecolor="white",
               markersize=9, label=MODEL_LABELS[m])
        for m in models
    ]
    style_leg = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="gray", markeredgecolor="white",
               markersize=9, label="Peak layer (filled circle)"),
        Line2D([0],[0], marker="s", color="w",
               markerfacecolor="white", markeredgecolor="gray",
               markeredgewidth=2, markersize=9,
               label="Final layer (open square)"),
        Line2D([0],[0], color="gray", lw=2, alpha=0.5,
               label="Suppression gap"),
    ]
    fig.legend(handles=model_leg + style_leg, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5,-0.08), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output)


def _render_logit_lens_ax(ax, run, fact, model, lang, use_absolute):
    """Renderiza o Logit Lens de um unico run em um eixo."""
    layers = run.get("last_token", [])
    if not layers:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    stake_lang = CULTURAL_STAKE[fact]
    n_layers   = len(layers)
    n_show, top_k = 20, 5

    idxs = np.linspace(0, n_layers-1, n_show, dtype=int)
    sel  = [layers[i] for i in idxs]

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("auto"); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)

    cw, ch = 1.0/top_k, 1.0/n_show
    layer_labels = []

    for row_i, layer in enumerate(sel):
        y = 1.0 - (row_i+1)*ch
        layer_labels.append(
            "L%d" % layer["layer"] if use_absolute
            else "%.2f" % layer["layer_norm"]
        )
        for col_i in range(top_k):
            if col_i >= len(layer["top_tokens"]): break
            token = layer["top_tokens"][col_i]
            prob  = layer["top_probs"][col_i]
            color = token_color(token, fact)
            alpha = float(np.clip(prob*6, 0.06, 0.95))
            x = col_i*cw

            ax.add_patch(mpatches.FancyBboxPatch(
                (x+0.003, y+0.001), cw-0.006, ch-0.002,
                boxstyle="round,pad=0.006",
                facecolor=color, alpha=alpha,
                edgecolor="white", linewidth=0.4, zorder=2))

            tc   = "white" if alpha > 0.5 else "#333"
            disp = token.strip()[:8]
            ax.text(x+cw/2, y+ch*0.62, disp,
                    ha="center", va="center", fontsize=5.5, color=tc,
                    fontweight="bold" if color != "#AAAAAA" else "normal",
                    zorder=3)
            ax.text(x+cw/2, y+ch*0.22, "%.3f" % prob,
                    ha="center", va="center",
                    fontsize=4, color=tc, alpha=0.85, zorder=3)

    ax.set_yticks([1.0-(i+0.5)*ch for i in range(n_show)])
    ax.set_yticklabels(layer_labels, fontsize=5.5)
    ax.set_xticks([])

    for frac in [0.33, 0.67]:
        ax.axhline(1.0-int(n_show*frac)*ch,
                   color="#888", lw=0.8, linestyle=":", alpha=0.5)

    ax.set_ylabel("Layer" if use_absolute else "Relative Depth", fontsize=7)

    m_res  = run.get("metrics_last", {})
    cs     = m_res.get("commitment_final", 0)
    peak   = m_res.get("iv_local_peak", 0)
    p_l    = m_res.get("peak_layer", "?")
    n_l    = MODEL_LAYERS[model]
    prompt = run.get("prompt", "")
    stake_mark = " * stake" if lang == stake_lang else ""
    ax.set_title(
        "%s (%dL) | %s%s\n\"%s\"\nCS_final=%+.3f  peak=%.3f@L%s" % (
            MODEL_LABELS[model], n_l, LANG_LABELS[lang], stake_mark,
            prompt, cs, peak, str(p_l)
        ),
        fontsize=7.5, fontweight="bold"
    )


def _logit_lens_legend(fig, fact):
    """Adiciona legenda padrao ao Logit Lens."""
    local_label = "Wright/Brothers tokens" if fact == "aviao" else "Bell/Alexander tokens"
    comp_label  = "Santos/Dumont tokens"   if fact == "aviao" else "Meucci/Antonio tokens"
    leg = [
        mpatches.Patch(facecolor=ENTITY_COLOR_LOCAL,     alpha=0.8, label=local_label),
        mpatches.Patch(facecolor=ENTITY_COLOR_COMPETING, alpha=0.8, label=comp_label),
        mpatches.Patch(facecolor="#AAAAAA",              alpha=0.5, label="Other tokens"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.03), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")


def fig3_logit_lens_single(data, fact, model, lang, output, use_absolute=False, form="F1"):
    """
    Logit Lens para uma unica combinacao (modelo x idioma).
    Gera figura individual — util para inspecao detalhada.
    """
    set_style()
    v = "Absolute Layer Number" if use_absolute else "Relative Layer Depth (0->1)"

    run = data.get(model, {}).get(fact, {}).get(lang, {}).get(form, {})

    fig, ax = plt.subplots(1, 1, figsize=(5, 9))
    fig.suptitle(
        "Logit Lens | %s | %s | %s | %s | %s" % (
            FACT_LABELS[fact], MODEL_LABELS[model],
            LANG_LABELS[lang], form, v),
        fontweight="bold", fontsize=9, y=1.01
    )

    _render_logit_lens_ax(ax, run, fact, model, lang, use_absolute)
    _logit_lens_legend(fig, fact)

    save_fig(fig, output)


def fig3_logit_lens_per_model(data, fact, model, output, use_absolute=False, form="F1"):
    """
    Logit Lens para todos os 4 idiomas de um unico modelo.
    Layout: 1 linha x 4 colunas (pt, en, de, it).
    """
    set_style()
    langs = ["pt","en","de","it"]
    v = "Absolute Layer Number" if use_absolute else "Relative Layer Depth (0->1)"

    fig, axes = plt.subplots(1, 4, figsize=(18, 9), sharey=True)
    fig.suptitle(
        "Logit Lens | %s | %s | All Languages | %s | %s" % (
            FACT_LABELS[fact], MODEL_LABELS[model], form, v),
        fontweight="bold", fontsize=10, y=1.01
    )

    for ax, lang in zip(axes, langs):
        run = data.get(model, {}).get(fact, {}).get(lang, {}).get(form, {})
        _render_logit_lens_ax(ax, run, fact, model, lang, use_absolute)

    _logit_lens_legend(fig, fact)
    save_fig(fig, output)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def _recalculate_metrics(data, results_dir):
    """
    Recalcula metricas para todos os runs usando o analyzer atual.
    Necessario quando os JSONs foram gerados com versao anterior do codigo.
    Salva os JSONs atualizados em results/.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config.experiment_config import ExperimentConfig
    from logit_lens.analyzer import LogitLensAnalyzer
    from logit_lens.normalizer import build_comparison_table

    config  = ExperimentConfig()
    facts   = ["aviao", "telefone"]
    langs   = ["pt", "en", "de", "it"]
    forms   = ["F1", "F2", "F3"]

    for model, mdata in data.items():
        for fact in facts:
            if fact not in mdata: continue
            analyzer  = LogitLensAnalyzer(config.expected_entities[fact])
            competing = config.competing_entities[fact]
            for lang in langs:
                if lang not in mdata[fact]: continue
                for form in forms:
                    run = mdata[fact][lang].get(form, {})
                    if "last_token" not in run: continue
                    run["metrics_last"] = analyzer.compute_all_metrics(
                        run["last_token"], lang,
                        competing_entities=competing)
                    if run.get("keyword_token"):
                        run["metrics_keyword"] = analyzer.compute_all_metrics(
                            run["keyword_token"], lang,
                            competing_entities=competing)

        path = os.path.join(results_dir, "%s_results.json" % model)
        with open(path, "w", encoding="utf-8") as f:
            import json
            json.dump(mdata, f, ensure_ascii=False, indent=2)
        print("  recalculado: %s" % model)

    # Regenerar CSV
    all_results = {m: data[m] for m in data}
    df = build_comparison_table(all_results, ExperimentConfig())
    df.to_csv(os.path.join(results_dir, "metrics_comparison.csv"), index=False)


def generate_all_figures(results_dir="results", figures_dir="figures"):
    Path(figures_dir).mkdir(exist_ok=True)

    csv_path = os.path.join(results_dir, "metrics_comparison.csv")
    if not os.path.exists(csv_path):
        from consolidate_results import consolidate
        consolidate(results_dir)

    data = load_data()
    if not data:
        print("Nenhum dado encontrado em results/")
        return

    print(f"Modelos: {list(data.keys())}")
    print(f"Figuras em: {figures_dir}/\n")

    def p(name): return os.path.join(figures_dir, name)

    facts = ["aviao", "telefone"]

    # ── Fig 1: divergence map — ambos fatos combinados, duas versoes ──
    fig1_divergence_map(data, p("fig1_divergence_map.png"),     use_absolute=False)
    fig1_divergence_map(data, p("fig1_divergence_map_abs.png"), use_absolute=True)

    # ── Fig 2: CS heatmap — ambos fatos, unica figura ──
    fig2_cs_heatmap(data, p("fig2_cs_heatmap.png"))

    # ── Fig 3: logit lens ──
    # 3a. Figura combinada: LLaMA e Gemma x idioma neutro e stake (2x2)
    for fact in facts:
        fig3_logit_lens_heatmap(data, fact,
            p("fig3_logit_lens_%s.png" % fact),          use_absolute=False)
        fig3_logit_lens_heatmap(data, fact,
            p("fig3_logit_lens_%s_abs.png" % fact),      use_absolute=True)

    # 3b. Por modelo: todos os 4 idiomas em uma figura (1x4)
    models = ["llama","mistral","qwen","gemma"]
    for fact in facts:
        for model in models:
            fig3_logit_lens_per_model(data, fact, model,
                p("fig3_logit_lens_%s_%s.png" % (fact, model)),
                use_absolute=False)
            fig3_logit_lens_per_model(data, fact, model,
                p("fig3_logit_lens_%s_%s_abs.png" % (fact, model)),
                use_absolute=True)

    # 3c. Por modelo e idioma: figura individual (4 modelos x 4 idiomas x 2 fatos)
    langs = ["pt","en","de","it"]
    for fact in facts:
        for model in models:
            for lang in langs:
                fig3_logit_lens_single(data, fact, model, lang,
                    p("fig3_logit_lens_%s_%s_%s.png" % (fact, model, lang)),
                    use_absolute=False)
                fig3_logit_lens_single(data, fact, model, lang,
                    p("fig3_logit_lens_%s_%s_%s_abs.png" % (fact, model, lang)),
                    use_absolute=True)

    # ── Fig 4: behavioral — ambos fatos, unica figura ──
    fig4_behavioral(data, p("fig4_behavioral.png"))

    # ── Fig 5: convergence — por fato, duas versoes ──
    for fact in facts:
        fig5_convergence(data, fact,
            p(f"fig5_convergence_{fact}.png"),           use_absolute=False)
        fig5_convergence(data, fact,
            p(f"fig5_convergence_{fact}_abs.png"),       use_absolute=True)

    # ── Fig 6: peak vs final — ambos fatos, unica figura ──
    fig6_peak_vs_final(data, p("fig6_peak_vs_final.png"))

    total = len(list(Path(figures_dir).glob("fig[0-9]*.png")))
    print(f"\n  {total} figuras geradas em {figures_dir}/")


if __name__ == "__main__":
    generate_all_figures()