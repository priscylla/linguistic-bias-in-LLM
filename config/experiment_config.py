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
            "pt": "avião", "en": "airplane",
            "de": "Flugzeug", "it": "aereo"
        },
        "telefone": {
            "pt": "telefone", "en": "telephone",
            "de": "Telefon", "it": "telefono"
        },
        "radio": {
            "pt": "rádio", "en": "radio",
            "de": "Radio", "it": "radio"
        }
    })

    expected_entities: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "aviao": {
            "pt": ["Santos", "Dumont", "Santos-Dumont"],
            "en": ["Wright", "Brothers", "Orville", "Wilbur"],
            "de": ["Wright", "Brothers", "Gebrüder"],
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

    top_k_tokens: int = 10
    results_dir: str  = "results"
    seed: int         = 42
