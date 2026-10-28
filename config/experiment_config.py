    # Entidade concorrente (narrativa alternativa) por fato e idioma.
    # Usada para calcular o Commitment Score:
    #
    #     CS = P(expected_entity) - P(competing_entity)
    #
    # CS > 0  -> modelo favorece a narrativa cultural local
    # CS ~= 0 -> modelo ambiguo, as duas narrativas competem
    # CS < 0  -> modelo favorece a narrativa concorrente
    #            mesmo quando perguntado naquele idioma

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ExperimentConfig:

    models: Dict[str, str] = field(default_factory=lambda: {
        "llama":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen":    "Qwen/Qwen2.5-7B-Instruct",
        "gemma":   "google/gemma-2-9b-it"
    })

    facts: List[str] = field(default_factory=lambda: ["aviao", "telefone"])

    languages: List[str] = field(default_factory=lambda: ["pt", "en", "de", "it"])

    formulations: List[str] = field(default_factory=lambda: ["F1", "F2", "F3"])

    # Idioma(s) com stake cultural por fato (narrativa local)
    stake_languages: Dict[str, List[str]] = field(default_factory=lambda: {
        "aviao":    ["pt"],
        "telefone": ["it"],
    })

    # Idioma de referência (ΔCS na Fase 2; idioma neutro nas figuras)
    reference_language: str = "en"

    # Zona de decisão: fração final da profundidade (0.30 = últimos 30%
    # das camadas). Um único valor usado em todas as figuras e métricas.
    decision_zone: float = 0.30

    keyword_tokens: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "aviao": {
            "pt": "avião", "en": "airplane",
            "de": "Flugzeug", "it": "aereo"
        },
        "telefone": {
            "pt": "telefone", "en": "telephone",
            "de": "Telefon", "it": "telefono"
        }
    })

    expected_entities: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "aviao": {
            "pt": ["Santos", "Dumont"],
            "en": ["Wright", "Brothers", "Orville", "Wilbur", "Kitty"],
            "de": ["Wright", "Brothers", "Orville", "Wilbur", "Kitty"],
            "it": ["Wright", "Brothers", "Orville", "Wilbur", "Kitty"]
        },
        "telefone": {
            "pt": ["Bell", "Graham", "Alexander"],
            "en": ["Bell", "Graham", "Alexander"],
            "de": ["Bell", "Graham", "Alexander"],
            "it": ["Meucci", "Antonio"]
        }
    })

    competing_entities: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "aviao": {
            "pt": ["Wright", "Brothers", "Orville", "Wilbur", "Kitty"],
            "en": ["Santos", "Dumont"],
            "de": ["Santos", "Dumont"],
            "it": ["Santos", "Dumont"]
        },
        "telefone": {
            "pt": ["Meucci", "Antonio"],
            "en": ["Meucci", "Antonio"],
            "de": ["Meucci", "Antonio"],
            "it": ["Bell", "Graham", "Alexander"]
        }
    })

    # Janelas de suavização das figuras de curva (só para os gráficos;
    # métricas usam valores brutos). 1 = sem suavização. Cada figura de
    # curva é gerada uma vez por janela (sufixos _w3 e _raw).
    smoothing_windows: List[int] = field(default_factory=lambda: [3, 1])

    # Modelos mostrados na figura combinada do heatmap de top tokens
    heatmap_models: List[str] = field(default_factory=lambda: ["llama", "gemma"])

    top_k_tokens: int = 10
    # Precisão de carregamento dos modelos (bfloat16 | float16 | float32)
    dtype: str        = "bfloat16"
    results_dir: str  = "results"
    figures_dir: str  = "figures"
    seed: int         = 42

    def stake(self, fact: str, lang: str) -> bool:
        """True se `lang` tem stake cultural no fato."""
        return lang in self.stake_languages.get(fact, [])

    def expected_cells(self) -> int:
        """Número de runs esperados: modelos × fatos × idiomas × formulações."""
        return (len(self.models) * len(self.facts)
                * len(self.languages) * len(self.formulations))

