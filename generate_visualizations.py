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

BELL_TOKENS   = ["bell","graham","alexander","aless","al","scot"]
MEUCCI_TOKENS = ["meucci","antonio","aless","alessan"]

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

def token_color(token):
    t = token.strip().lower()
    if any(e in t for e in BELL_TOKENS):   return "#D6604D"
    if any(e in t for e in MEUCCI_TOKENS): return "#2166AC"
    return "#AAAAAA"

def get_x_axis(run, model, use_absolute):
    """
    Retorna (x_array, xlabel, xlim, xticks) conforme versao.
    use_absolute=True  → camada absoluta (1..n)
    use_absolute=False → layer_norm (0..1)
    """
    layers_data = run.get("last_token", [])
    n_l = MODEL_LAYERS[model]

    if use_absolute:
        if layers_data:
            x = np.array([l["layer"] for l in layers_data])
        else:
            curve = run.get("metrics_last",{}).get("commitment_curve",[])
            x = np.arange(1, len(curve)+1)
        xlabel = "Layer"
        xlim   = (1, n_l)
        step   = max(4, n_l // 8)
        xticks = list(range(1, n_l+1, step))
    else:
        if layers_data:
            x = np.array([l["layer_norm"] for l in layers_data])
        else:
            curve = run.get("metrics_last",{}).get("commitment_curve",[])
            x = np.linspace(0, 1, len(curve))
        xlabel = "Relative Layer Depth"
        xlim   = (0, 1)
        xticks = [0, 0.25, 0.5, 0.75, 1.0]

    return x, xlabel, xlim, xticks


def decision_zone(ax, model, use_absolute):
    """Adiciona zona de decisao (ultimas 30%) no eixo."""
    n_l = MODEL_LAYERS[model]
    if use_absolute:
        start = int(n_l * 0.70)
        ax.axvspan(start, n_l, alpha=0.05, color="gray")
        ax.text(start + 0.3, ax.get_ylim()[1] * 0.92,
                f"L{start}", fontsize=6, color="gray",
                alpha=0.7, va="top")
    else:
        ax.axvspan(0.70, 1.0, alpha=0.05, color="gray")


# ══════════════════════════════════════════════════════════════
# FIG 1 — CULTURAL DIVERGENCE MAP
# ══════════════════════════════════════════════════════════════

def fig1_divergence_map(data, output, use_absolute=False):
    set_style()

    models = ["llama","mistral","qwen","gemma"]
    langs  = ["pt","en","de","it"]
    facts  = ["aviao","telefone"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharey=False)

    version = "Absolute Layer Number" if use_absolute else "Relative Layer Depth (0→1)"
    fig.suptitle(
        "Cultural Divergence Map — Commitment Score Across Model Layers\n"
        f"CS = P(expected entity) − P(competing entity)   |   "
        f"F1 formulation   |   {version}",
        fontweight="bold", fontsize=10, y=1.01
    )

    for row, fact in enumerate(facts):
        for col, model in enumerate(models):
            ax  = axes[row][col]
            n_l = MODEL_LAYERS[model]

            subtitle = (f"\n({n_l} layers)" if use_absolute else "")
            if row == 0:
                ax.set_title(
                    f"{MODEL_LABELS[model]}{subtitle}",
                    color=MODEL_COLORS[model],
                    fontweight="bold", fontsize=8.5
                )

            for lang in langs:
                run   = data.get(model,{}).get(fact,{}).get(lang,{}).get("F1",{})
                m_res = run.get("metrics_last",{})
                curve = m_res.get("commitment_curve",[])
                if not curve: continue

                x, xlabel, xlim, xticks = get_x_axis(run, model, use_absolute)
                y = smooth(np.array(curve), w=3)

                ax.plot(x, y, color=LANG_COLORS[lang],
                        alpha=0.85, linewidth=1.6)
                ax.fill_between(x, y, 0,
                                color=LANG_COLORS[lang], alpha=0.06)

                # CS final anotado
                cs = m_res.get("commitment_final", 0)
                if abs(cs) > 0.05:
                    ax.annotate(
                        f"{cs:+.2f}",
                        xy=(x[-1], y[-1]),
                        xytext=(x[-1] - (n_l*0.12 if use_absolute else 0.12),
                                y[-1]),
                        fontsize=6.5, color=LANG_COLORS[lang],
                        fontweight="bold", va="center"
                    )

            ax.axhline(0, color="gray", lw=0.7, linestyle="--", alpha=0.6)
            decision_zone(ax, model, use_absolute)

            if col == 0:
                ax.set_ylabel(
                    f"{FACT_LABELS[fact]}\nCommitment Score",
                    fontsize=8, fontweight="bold"
                )
            if row == 1:
                ax.set_xlabel(xlabel if col == 0 else "", fontsize=8)

            _, xlabel, xlim, xticks = get_x_axis(
                data.get(model,{}).get(fact,{}).get("pt",{}).get("F1",{}),
                model, use_absolute
            )
            ax.set_xticks(xticks)
            ax.set_xlim(xlim)

    leg = [Line2D([0],[0], color=LANG_COLORS[l], lw=2,
                  label=LANG_LABELS[l]) for l in langs]
    leg += [
        mpatches.Patch(facecolor="gray", alpha=0.15,
                       label="Decision zone (last 30%)"),
        Line2D([0],[0], color="gray", lw=1, linestyle="--",
               label="CS = 0 (neutral)")
    ]
    fig.legend(handles=leg, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5,-0.04), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"  OK  {output}")


# ══════════════════════════════════════════════════════════════
# FIG 2 — COMMITMENT SCORE HEATMAP (unica versao)
# ══════════════════════════════════════════════════════════════

def fig2_cs_heatmap(data, output):
    set_style()

    models = ["llama","mistral","qwen","gemma"]
    langs  = ["pt","en","de","it"]
    facts  = ["aviao","telefone"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        "Final Commitment Score — Last Layer  |  F1 formulation\n"
        "Blue = favors local entity   |   Red = favors competing (anglophone) entity",
        fontweight="bold", fontsize=10
    )

    for ax, fact in zip(axes, facts):
        matrix = np.zeros((len(models), len(langs)))
        for i, model in enumerate(models):
            for j, lang in enumerate(langs):
                run = data.get(model,{}).get(fact,{}).get(lang,{}).get("F1",{})
                matrix[i][j] = run.get("metrics_last",{}).get("commitment_final",0)

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
        ax.set_title(
            f"{FACT_LABELS[fact]}\n(+ favors local | − favors anglophone)",
            fontweight="bold", fontsize=9
        )

        stake = {"aviao":["pt"], "telefone":["it"]}
        for j, lang in enumerate(langs):
            if lang in stake.get(fact,[]):
                rect = mpatches.Rectangle(
                    (j-0.5,-0.5), 1, len(models),
                    lw=2, edgecolor="gold",
                    facecolor="none", zorder=5
                )
                ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label("Commitment Score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    leg = [mpatches.Patch(facecolor="none", edgecolor="gold",
                          lw=2, label="Cultural stake language")]
    fig.legend(handles=leg, loc="lower center",
               bbox_to_anchor=(0.5,-0.05), fontsize=8)

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"  OK  {output}")


# ══════════════════════════════════════════════════════════════
# FIG 3 — LOGIT LENS HEATMAP
# ══════════════════════════════════════════════════════════════

def fig3_logit_lens_heatmap(data, output, use_absolute=False):
    set_style()

    cases = [
        ("llama", "telefone", "pt", "F1"),
        ("llama", "telefone", "it", "F1"),
        ("gemma", "telefone", "pt", "F1"),
        ("gemma", "telefone", "it", "F1"),
    ]

    version = "Absolute Layer Number" if use_absolute else "Relative Layer Depth (0→1)"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    fig.suptitle(
        "Logit Lens — Top Predicted Tokens per Layer\n"
        f"Telephone invention   |   Intensity ∝ probability   |   {version}",
        fontweight="bold", fontsize=10, y=1.01
    )

    for ax, (model, fact, lang, form) in zip(axes, cases):
        run    = data.get(model,{}).get(fact,{}).get(lang,{}).get(form,{})
        layers = run.get("last_token",[])
        if not layers:
            ax.text(0.5,0.5,"No data",ha="center",va="center")
            continue

        n_layers = len(layers)
        n_show   = 20
        top_k    = 5

        idxs = np.linspace(0, n_layers-1, n_show, dtype=int)
        sel  = [layers[i] for i in idxs]

        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_aspect("auto"); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)

        cw = 1.0/top_k
        ch = 1.0/n_show
        layer_labels = []

        for row_i, layer in enumerate(sel):
            y = 1.0-(row_i+1)*ch

            # Label: absoluto ou relativo
            if use_absolute:
                layer_labels.append(f"L{layer['layer']}")
            else:
                layer_labels.append(f"{layer['layer_norm']:.2f}")

            for col_i in range(top_k):
                if col_i >= len(layer["top_tokens"]): break
                token = layer["top_tokens"][col_i]
                prob  = layer["top_probs"][col_i]
                color = token_color(token)
                alpha = float(np.clip(prob*6, 0.06, 0.95))
                x = col_i*cw

                rect = mpatches.FancyBboxPatch(
                    (x+0.003,y+0.001), cw-0.006, ch-0.002,
                    boxstyle="round,pad=0.006",
                    facecolor=color, alpha=alpha,
                    edgecolor="white", linewidth=0.4, zorder=2
                )
                ax.add_patch(rect)

                tc   = "white" if alpha > 0.5 else "#333"
                disp = token.strip()[:8]
                ax.text(x+cw/2, y+ch*0.62, disp,
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

        # Zonas early/mid/late (linhas pontilhadas)
        for frac, label in [(0.33,"early|mid"),(0.67,"mid|late")]:
            yl = 1.0 - int(n_show*frac)*ch
            ax.axhline(yl, color="#888", lw=0.8,
                       linestyle=":", alpha=0.5)

        ylabel = "Layer" if use_absolute else "Relative Depth"
        ax.set_ylabel(ylabel, fontsize=7)

        prompt = run.get("prompt", run.get("completion_prompt",""))
        cs     = run.get("metrics_last",{}).get("commitment_final",0)
        n_l    = MODEL_LAYERS[model]
        subtitle = (f" ({n_l} layers)" if use_absolute else "")
        ax.set_title(
            f"{MODEL_LABELS[model]}{subtitle} | {LANG_LABELS[lang]}\n"
            f"\"{prompt}\"  →  CS={cs:+.3f}",
            fontsize=8, fontweight="bold"
        )

    leg = [
        mpatches.Patch(facecolor="#D6604D", alpha=0.8,
                       label="Bell / Alexander tokens"),
        mpatches.Patch(facecolor="#2166AC", alpha=0.8,
                       label="Meucci / Antonio tokens"),
        mpatches.Patch(facecolor="#AAAAAA", alpha=0.5,
                       label="Other tokens"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.03), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"  OK  {output}")


# ══════════════════════════════════════════════════════════════
# FIG 4 — BEHAVIORAL ANALYSIS (unica versao)
# ══════════════════════════════════════════════════════════════

def fig4_behavioral(data, output):
    set_style()

    models = ["llama","mistral","qwen","gemma"]
    langs  = ["pt","en","de","it"]

    def classify(response, fact):
        r = response.lower()
        if fact == "aviao":
            has_s = any(w in r for w in ["santos","dumont"])
            has_w = any(w in r for w in ["wright","orville","wilbur","irmãos"])
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
        "What the model SAYS when completing the sentence",
        fontweight="bold", fontsize=10
    )

    for ax, fact in zip(axes, ["aviao","telefone"]):
        for i, model in enumerate(models):
            for j, lang in enumerate(langs):
                run  = data.get(model,{}).get(fact,{}).get(lang,{}).get("F1",{})
                resp = run.get("response","")
                cat  = classify(resp, fact)
                color = cat_colors.get(cat,"#AAAAAA")

                rect = mpatches.FancyBboxPatch(
                    (j+0.05,i+0.05), 0.90, 0.90,
                    boxstyle="round,pad=0.05",
                    facecolor=color, alpha=0.85,
                    edgecolor="white", linewidth=1.5
                )
                ax.add_patch(rect)

                tc = "white" if color!="#AAAAAA" else "#444"
                ax.text(j+0.5, i+0.5, cat,
                        ha="center", va="center",
                        fontsize=7.5, color=tc, fontweight="bold")

        ax.set_xlim(0, len(langs))
        ax.set_ylim(0, len(models))
        ax.set_aspect("equal"); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)

        ax.set_xticks([j+0.5 for j in range(len(langs))])
        ax.set_xticklabels([LANG_LABELS[l] for l in langs],
                           fontsize=8, rotation=15, ha="right")
        ax.set_yticks([i+0.5 for i in range(len(models))])
        ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=8)

        stake_j = {"aviao":0,"telefone":3}[fact]
        rect = mpatches.Rectangle(
            (stake_j, 0), 1, len(models),
            lw=2.5, edgecolor="gold",
            facecolor="none", zorder=5
        )
        ax.add_patch(rect)
        ax.text(stake_j+0.5, len(models)+0.1, "stake",
                ha="center", fontsize=7,
                color="goldenrod", fontweight="bold")

        prompt = list(data.values())[0][fact]["pt"]["F1"].get("prompt","")
        ax.set_title(
            f"{FACT_LABELS[fact]}\nPrompt: \"{prompt}\"",
            fontweight="bold", fontsize=9
        )

    leg = [
        mpatches.Patch(facecolor="#D6604D", alpha=0.85,
                       label="Anglophone entity (Wright / Bell)"),
        mpatches.Patch(facecolor="#2166AC", alpha=0.85,
                       label="Local entity (Santos Dumont / Meucci)"),
        mpatches.Patch(facecolor="#6BAED6", alpha=0.85,
                       label="Both mentioned"),
        mpatches.Patch(facecolor="#AAAAAA", alpha=0.6,
                       label="Ambiguous / No inventor named"),
        mpatches.Patch(facecolor="none", edgecolor="gold",
                       lw=2, label="Cultural stake language"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.08), fontsize=8,
               framealpha=0.95, edgecolor="lightgray")

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"  OK  {output}")


# ══════════════════════════════════════════════════════════════
# FIG 5 — CONVERGENCE
# ══════════════════════════════════════════════════════════════

def fig5_convergence(data, output, use_absolute=False):
    set_style()

    langs   = ["pt","en","de","it"]
    version = "Absolute Layer Number" if use_absolute else "Relative Layer Depth (0→1)"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Layer Convergence — Commitment Score Trajectory\n"
        f"Telephone invention  |  F1 formulation  |  {version}",
        fontweight="bold", fontsize=10
    )

    for ax, model in zip(axes, ["llama","gemma"]):
        n_l = MODEL_LAYERS[model]
        ax.set_title(
            f"{MODEL_LABELS[model]}  ({n_l} layers)",
            color=MODEL_COLORS[model],
            fontweight="bold", fontsize=9.5
        )

        for lang in langs:
            run   = data.get(model,{}).get("telefone",{}).get(lang,{}).get("F1",{})
            m_res = run.get("metrics_last",{})
            curve = m_res.get("commitment_curve",[])
            if not curve: continue

            x, xlabel, xlim, xticks = get_x_axis(run, model, use_absolute)
            y = smooth(np.array(curve), w=4)

            ax.plot(x, y, color=LANG_COLORS[lang],
                    linewidth=2.2, alpha=0.9,
                    label=LANG_LABELS[lang])
            ax.fill_between(x, y, 0, color=LANG_COLORS[lang], alpha=0.07)

            cs_final = m_res.get("commitment_final",0)
            if abs(cs_final) > 0.03:
                offset = (n_l*0.1 if use_absolute else 0.08)
                ax.annotate(
                    f"{cs_final:+.2f}",
                    xy=(x[-1], y[-1]),
                    xytext=(x[-1]-offset, y[-1]),
                    fontsize=8.5, color=LANG_COLORS[lang],
                    fontweight="bold", va="center", ha="right"
                )

        ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
        decision_zone(ax, model, use_absolute)

        # Linhas de referencia em 25/50/75%
        for pct in [0.25, 0.50, 0.75]:
            xmark = int(n_l*pct) if use_absolute else pct
            label = f"L{int(n_l*pct)}" if use_absolute else f"{pct:.0%}"
            ax.axvline(xmark, color="gray", lw=0.5,
                       linestyle=":", alpha=0.4)
            ax.text(xmark, ax.get_ylim()[0], label,
                    fontsize=6, color="gray",
                    ha="center", va="bottom", alpha=0.6)

        ax.set_xticks(xticks)
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.legend(fontsize=8, loc="upper left",
                  framealpha=0.9, edgecolor="lightgray")

    axes[0].set_ylabel("Commitment Score\n(P_local − P_competing)", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"  OK  {output}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

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

    # ── Fig 1 — duas versoes ──
    fig1_divergence_map(data,
        os.path.join(figures_dir, "fig1_divergence_map.png"),
        use_absolute=False)
    fig1_divergence_map(data,
        os.path.join(figures_dir, "fig1_divergence_map_abs.png"),
        use_absolute=True)

    # ── Fig 2 — unica versao ──
    fig2_cs_heatmap(data,
        os.path.join(figures_dir, "fig2_cs_heatmap.png"))

    # ── Fig 3 — duas versoes ──
    fig3_logit_lens_heatmap(data,
        os.path.join(figures_dir, "fig3_logit_lens.png"),
        use_absolute=False)
    fig3_logit_lens_heatmap(data,
        os.path.join(figures_dir, "fig3_logit_lens_abs.png"),
        use_absolute=True)

    # ── Fig 4 — unica versao ──
    fig4_behavioral(data,
        os.path.join(figures_dir, "fig4_behavioral.png"))

    # ── Fig 5 — duas versoes ──
    fig5_convergence(data,
        os.path.join(figures_dir, "fig5_convergence.png"),
        use_absolute=False)
    fig5_convergence(data,
        os.path.join(figures_dir, "fig5_convergence_abs.png"),
        use_absolute=True)

    total = len(list(Path(figures_dir).glob("fig[0-9]*.png")))
    print(f"\n  {total} figuras geradas")


if __name__ == "__main__":
    generate_all_figures()