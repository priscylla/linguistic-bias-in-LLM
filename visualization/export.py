import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict

from .style import set_ieee_style
from .figure1_heatmap import plot_logit_lens_heatmap
from .figure2_divergence import plot_cultural_divergence_map
from .figure3_pcc import plot_pcc_comparison
from .figure4_stake import plot_stake_analysis
from .figure5_formulation import plot_formulation_comparison


def load_all_results(results_dir: str) -> Dict:
    """Carrega todos os JSONs de resultados."""
    all_results = {}
    
    for model_key in ["llama", "mistral", "qwen", "gemma"]:
        path = os.path.join(results_dir, f"{model_key}_results.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                all_results[model_key] = json.load(f)
        else:
            print(f"⚠️  Arquivo não encontrado: {path}")
    
    return all_results


def generate_all_figures(
    results_dir: str = "results",
    figures_dir: str = "figures"
):
    """
    Gera todas as figuras do paper e salva em PDF.
    
    Estrutura de saída:
        figures/
        ├── fig1_heatmap_aviao_F1_llama.pdf
        ├── fig2_divergence_aviao_F1.pdf
        ├── fig3_pcc_F1.pdf
        ├── fig4_stake.pdf
        └── fig5_formulation_aviao_llama.pdf
    """
    Path(figures_dir).mkdir(exist_ok=True)
    
    # Carregar dados
    results = load_all_results(results_dir)
    metrics_df = pd.read_csv(
        os.path.join(results_dir, "metrics_comparison.csv")
    )
    
    from config.experiment_config import ExperimentConfig
    config = ExperimentConfig()
    
    print("\n" + "="*50)
    print("GERANDO FIGURAS DO PAPER")
    print("="*50)
    
    # ── Figura 1: Heatmap ──
    # Caso principal: avião, F1, LLaMA, last_token
    print("\n[Fig 1] Logit Lens Heatmap...")
    
    for position in ["last_token", "keyword_token"]:
        plot_logit_lens_heatmap(
            results=results,
            fact="aviao",
            formulation="F1",
            model_key="llama",
            local_entities=config.expected_entities["aviao"],
            dominant_entities={
                "pt": ["Wright", "Brothers", "Orville", "Wilbur"],
                "en": ["Santos", "Dumont"],
                "de": ["Santos", "Dumont"],
                "it": ["Santos", "Dumont"]
            },
            position=position,
            output_path=os.path.join(
                figures_dir,
                f"fig1_heatmap_aviao_F1_llama_{position}.pdf"
            )
        )
    
    # ── Figura 2: Cultural Divergence Map ──
    print("\n[Fig 2] Cultural Divergence Map...")
    
    for fact in ["aviao", "telefone", "radio"]:
        for formulation in ["F1", "F3"]:
            plot_cultural_divergence_map(
                results=results,
                fact=fact,
                formulation=formulation,
                expected_entities=config.expected_entities[fact],
                output_path=os.path.join(
                    figures_dir,
                    f"fig2_divergence_{fact}_{formulation}.pdf"
                )
            )
    
    # ── Figura 3: PCC Comparison ──
    print("\n[Fig 3] PCC Comparison...")
    
    for formulation in ["F1", "F2", "F3"]:
        plot_pcc_comparison(
            metrics_df=metrics_df,
            formulation=formulation,
            output_path=os.path.join(
                figures_dir,
                f"fig3_pcc_{formulation}.pdf"
            )
        )
    
    # ── Figura 4: Stake Analysis ──
    print("\n[Fig 4] Stake Analysis...")
    
    plot_stake_analysis(
        metrics_df=metrics_df,
        output_path=os.path.join(
            figures_dir, "fig4_stake.pdf"
        )
    )
    
    # ── Figura 5: Formulation Comparison ──
    print("\n[Fig 5] Formulation Comparison...")
    
    for fact in ["aviao", "telefone", "radio"]:
        for model_key in ["llama", "mistral", "qwen", "gemma"]:
            plot_formulation_comparison(
                results=results,
                fact=fact,
                model_key=model_key,
                expected_entities=config.expected_entities[fact],
                output_path=os.path.join(
                    figures_dir,
                    f"fig5_formulation_{fact}_{model_key}.pdf"
                )
            )
    
    print(f"\n✅ Todas as figuras geradas em: {figures_dir}/")
    print(f"   Total de arquivos: "
          f"{len(list(Path(figures_dir).glob('*.pdf')))}")


if __name__ == "__main__":
    generate_all_figures()