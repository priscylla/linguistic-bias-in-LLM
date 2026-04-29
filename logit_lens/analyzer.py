"""
logit_lens/analyzer.py
Cálculo das métricas definidas no design experimental
a partir dos resultados brutos do LogitLensExtractor.

Métricas implementadas:
    PCC — Ponto de Convergência Cultural
    DL  — Divergência entre Línguas
    ED  — Estabilidade da Decisão
    IV  — Intensidade do Viés
    CM  — Consistência entre Modelos
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class LogitLensAnalyzer:
    """
    Calcula métricas de viés cultural a partir de layer_results.

    Args:
        expected_entities: dict mapeando idioma → lista de tokens
                           que representam a entidade cultural esperada.
                           Ex: {"pt": ["Santos", "Dumont"],
                                "en": ["Wright", "Brothers"]}
    """

    def __init__(self, expected_entities: Dict[str, List[str]]):
        self.expected_entities = expected_entities

    def _entity_prob_at_layer(
        self,
        layer_data: Dict,
        entities: List[str]
    ) -> float:
        """
        Calcula probabilidade acumulada das entidades esperadas
        nos top-K tokens de uma camada.

        Soma as probabilidades de todos os top-K tokens que
        contêm alguma das strings de entidade esperada.
        Usa break para evitar double-counting de um mesmo token.

        Args:
            layer_data: dict com 'top_tokens' e 'top_probs'
            entities:   lista de strings a identificar

        Returns:
            Probabilidade acumulada (float entre 0 e 1)
        """
        total_prob = 0.0
        for token, prob in zip(layer_data["top_tokens"], layer_data["top_probs"]):
            token_clean = token.strip().lower()
            for entity in entities:
                if entity.lower() in token_clean:
                    total_prob += prob
                    break  # evitar double-counting
        return total_prob

    def compute_pcc(
        self,
        layer_results: List[Dict],
        language: str,
        threshold: float = 0.10
    ) -> Optional[int]:
        """
        Ponto de Convergência Cultural (PCC).

        Primeira camada onde a entidade esperada para esse idioma
        entra no top-K com probabilidade >= threshold E permanece
        estável nas próximas 3 camadas.

        A verificação de estabilidade (tolerância de 50%) filtra
        ativações espúrias — picos momentâneos que não representam
        comprometimento real do modelo com aquela entidade.

        Args:
            layer_results: lista de dicts por camada
            language:      idioma atual
            threshold:     probabilidade mínima (default: 0.10)

        Returns:
            Índice absoluto da camada de convergência, ou None.
        """
        entities = self.expected_entities.get(language, [])
        if not entities:
            return None

        n_layers = len(layer_results)

        for i, layer_data in enumerate(layer_results):
            prob = self._entity_prob_at_layer(layer_data, entities)

            if prob >= threshold:
                # Verificar estabilidade nas próximas 3 camadas
                stable = True
                for j in range(i + 1, min(i + 4, n_layers)):
                    next_prob = self._entity_prob_at_layer(
                        layer_results[j], entities
                    )
                    if next_prob < threshold * 0.5:   # 50% de tolerância
                        stable = False
                        break

                if stable:
                    return layer_data["layer"]

        return None

    def compute_ed(
        self,
        layer_results: List[Dict]
    ) -> Optional[int]:
        """
        Estabilidade da Decisão (ED).

        Primeira camada a partir da qual o token top-1 não muda
        nas próximas 4 camadas consecutivas.

        Indica quando o modelo "cristalizou" sua decisão de geração,
        independente de qual seja a entidade escolhida.

        Args:
            layer_results: lista de dicts por camada

        Returns:
            Índice absoluto da camada de estabilização, ou None.
        """
        n_layers = len(layer_results)

        for i in range(n_layers - 4):
            top1 = layer_results[i]["top_tokens"][0].strip().lower()

            stable = all(
                layer_results[j]["top_tokens"][0].strip().lower() == top1
                for j in range(i + 1, i + 5)
            )

            if stable:
                return layer_results[i]["layer"]

        return None

    def compute_dl(
        self,
        layer_results_lang1: List[Dict],
        layer_results_lang2: List[Dict],
        language1: str,
        language2: str
    ) -> List[float]:
        """
        Divergência entre Línguas (DL).

        Diferença absoluta entre as probabilidades das entidades
        locais de cada idioma, calculada camada a camada.

        Usada para a visualização do Cultural Divergence Map:
        valores crescentes indicam que as línguas estão se
        "separando" em suas crenças culturais internas.

        Args:
            layer_results_lang1: resultados por camada do idioma 1
            layer_results_lang2: resultados por camada do idioma 2
            language1:           código do idioma 1 (ex: "pt")
            language2:           código do idioma 2 (ex: "en")

        Returns:
            Lista de floats com DL por camada.
        """
        entities1 = self.expected_entities.get(language1, [])
        entities2 = self.expected_entities.get(language2, [])
        divergences = []

        for l1, l2 in zip(layer_results_lang1, layer_results_lang2):
            prob1 = self._entity_prob_at_layer(l1, entities1)
            prob2 = self._entity_prob_at_layer(l2, entities2)
            divergences.append(abs(prob1 - prob2))

        return divergences

    def compute_iv(
        self,
        layer_results: List[Dict],
        language: str
    ) -> float:
        """
        Intensidade do Viés (IV).

        Probabilidade da entidade local na última camada (output).
        Permite rankear modelos por grau de viés cultural:
        IV alto = modelo fortemente comprometido com a narrativa local.

        Args:
            layer_results: lista de dicts por camada
            language:      idioma

        Returns:
            Float entre 0 e 1.
        """
        entities = self.expected_entities.get(language, [])
        if not entities:
            return 0.0

        return self._entity_prob_at_layer(layer_results[-1], entities)

    def compute_all_metrics(
        self,
        layer_results: List[Dict],
        language: str
    ) -> Dict:
        """
        Calcula todas as métricas para um único run.

        Retorna tanto valores absolutos (índices de camada)
        quanto normalizados (0-1) para facilitar comparação
        entre modelos com números diferentes de camadas.

        Args:
            layer_results: lista de dicts por camada
            language:      idioma

        Returns:
            Dict com pcc, pcc_norm, ed, ed_norm, iv.
        """
        n_layers = len(layer_results)

        pcc = self.compute_pcc(layer_results, language)
        ed  = self.compute_ed(layer_results)
        iv  = self.compute_iv(layer_results, language)

        # Normalizar para escala 0-1
        pcc_norm = round((pcc - 1) / (n_layers - 1), 4) if pcc else None
        ed_norm  = round((ed  - 1) / (n_layers - 1), 4) if ed  else None

        return {
            "pcc":      pcc,
            "pcc_norm": pcc_norm,
            "ed":       ed,
            "ed_norm":  ed_norm,
            "iv":       round(iv, 6)
        }
