from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ExperimentConfig:
    
    # Modelos
    models: Dict[str, str] = field(default_factory=lambda: {
        "llama":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen":    "Qwen/Qwen2.5-7B-Instruct",
        "gemma":   "google/gemma-2-9b-it"
    })
    
    # Fatos e seus tokens-chave por idioma
    # Usados para identificar o token da palavra-chave central
    keyword_tokens: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "aviao": {
            "pt": "avião",
            "en": "airplane",
            "de": "Flugzeug",
            "it": "aereo"
        },
        "telefone": {
            "pt": "telefone",
            "en": "telephone",
            "de": "Telefon",
            "it": "telefono"
        },
        "radio": {
            "pt": "rádio",
            "en": "radio",
            "de": "Radio",
            "it": "radio"
        }
    })
    
    # Entidades esperadas por fato e idioma
    # Usadas para calcular as métricas
    expected_entities: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "aviao": {
            "pt": ["Santos", "Dumont", "Santos-Dumont", "brasileiro"],
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
    
    # Configurações do Logit Lens
    temperature: float = 0.0         # greedy decoding
    top_k_tokens: int = 10           # top-K tokens a salvar por camada
    
    # Paths
    results_dir: str = "results"
    
    # Reprodutibilidade
    seed: int = 42