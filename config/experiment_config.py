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

    # Fatos historicos estudados (simplificado para 2)
    # aviao:    Wright Brothers vs. Santos Dumont  -- stake: PT
    # telefone: Bell vs. Meucci                   -- stake: IT
    facts: List[str] = field(default_factory=lambda: ["aviao", "telefone"])

    keyword_tokens: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "aviao": {
            "pt": "aviao", "en": "airplane",
            "de": "Flugzeug", "it": "aereo"
        },
        "telefone": {
            "pt": "telefone", "en": "telephone",
            "de": "Telefon",  "it": "telefono"
        }
    })

    # Entidade esperada para cada idioma -- narrativa cultural local.
    # Usada para calcular P(local) camada a camada.
    expected_entities: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "aviao": {
            "pt": ["Santos", "Dumont", "Santos-Dumont"],
            "en": ["Wright", "Brothers", "Orville", "Wilbur"],
            "de": ["Wright", "Brothers", "Gebruder"],
            "it": ["Wright", "Brothers", "fratelli"]
        },
        "telefone": {
            "pt": ["Bell", "Graham"],
            "en": ["Bell", "Graham"],
            "de": ["Bell", "Graham"],
            "it": ["Meucci", "Antonio"]
        }
    })

    # Entidade concorrente (narrativa alternativa) por fato e idioma.
    # Usada para calcular o Commitment Score:
    #
    #     CS = P(expected_entity) - P(competing_entity)
    #
    # CS > 0  -> modelo favorece a narrativa cultural local
    # CS ~= 0 -> modelo ambiguo, as duas narrativas competem
    # CS < 0  -> modelo favorece a narrativa concorrente
    #            mesmo quando perguntado naquele idioma
    competing_entities: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "aviao": {
            "pt": ["Wright", "Brothers", "Orville", "Wilbur"],
            "en": ["Santos", "Dumont", "Santos-Dumont"],
            "de": ["Santos", "Dumont", "Santos-Dumont"],
            "it": ["Santos", "Dumont", "Santos-Dumont"]
        },
        "telefone": {
            "pt": ["Meucci", "Antonio"],
            "en": ["Meucci", "Antonio"],
            "de": ["Meucci", "Antonio"],
            "it": ["Bell", "Graham"]
        }
    })

    top_k_tokens: int = 10
    results_dir: str  = "results"
    seed: int         = 42