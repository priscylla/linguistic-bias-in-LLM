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

    keyword_tokens: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "aviao": {
            "pt": "aviao", "en": "airplane",
            "de": "Flugzeug", "it": "aereo"
        },
        "telefone": {
            "pt": "telefone", "en": "telephone",
            "de": "Telefon", "it": "telefono"
        },
        "radio": {
            "pt": "radio", "en": "radio",
            "de": "Radio", "it": "radio"
        }
    })

    # Entidade esperada para cada idioma -- narrativa cultural local.
    # Ex: em PT, esperamos que o modelo mencione Santos Dumont.
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
        },
        "radio": {
            "pt": ["Marconi", "Guglielmo"],
            "en": ["Tesla", "Nikola"],
            "de": ["Marconi", "Tesla"],
            "it": ["Marconi", "Guglielmo"]
        }
    })

    # Entidade concorrente (narrativa alternativa) por fato e idioma.
    # Ex: em PT, a narrativa concorrente e Wright Brothers.
    # Usada para calcular o Commitment Score:
    #
    #     CS = P(expected_entity) - P(competing_entity)
    #
    # CS > 0  -> modelo favorece a narrativa cultural local
    # CS ~= 0 -> modelo ambiguo, as duas narrativas competem
    # CS < 0  -> modelo favorece a narrativa concorrente
    #            mesmo quando perguntado naquele idioma
    #
    # Isso distingue dois cenarios que P(local) sozinho nao separa:
    #   Cenario A -- vies forte:   PT: Santos=0.40, Wright=0.05 -> CS=+0.35
    #   Cenario B -- ambiguidade:  PT: Santos=0.22, Wright=0.20 -> CS=+0.02
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
        },
        "radio": {
            "pt": ["Tesla", "Nikola"],
            "en": ["Marconi", "Guglielmo"],
            "de": ["Tesla", "Nikola"],
            "it": ["Tesla", "Nikola"]
        }
    })

    top_k_tokens: int = 10
    results_dir: str  = "results"
    seed: int         = 42
