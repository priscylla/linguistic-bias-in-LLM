"""
run_visualization.py
Ponto de entrada para geração de todas as figuras do paper.

Uso:
    python run_visualization.py

Gera figuras em: figures/
Usa dados de:    results/

Funciona tanto com dados sintéticos (generate_mock_data.py)
quanto com dados reais (run_single_model.py).
"""

from generate_visualizations import generate_all_figures

if __name__ == "__main__":
    generate_all_figures(
        results_dir="results",
        figures_dir="figures"
    )
