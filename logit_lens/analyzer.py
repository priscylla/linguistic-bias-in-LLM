"""
logit_lens/analyzer.py
Calculo das metricas definidas no design experimental
a partir dos resultados brutos do LogitLensExtractor.

Metricas implementadas:
    PCC        -- Ponto de Convergencia Cultural
    ED         -- Estabilidade da Decisao
    IV_local   -- Intensidade do Vies (entidade local)
    IV_competing -- Probabilidade da entidade concorrente
    CS         -- Commitment Score = IV_local - IV_competing
    DL         -- Divergencia entre Linguas
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class LogitLensAnalyzer:
    """
    Calcula metricas de vies cultural a partir de layer_results.

    Args:
        expected_entities:  dict {lang: [tokens da entidade local]}
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
        Calcula probabilidade acumulada das entidades nos top-K tokens.

        Soma as probabilidades de todos os top-K tokens que contem
        alguma das strings de entidade. Usa break para evitar
        double-counting de um mesmo token.

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
        Ponto de Convergencia Cultural (PCC).

        Primeira camada onde P(entidade local) >= threshold
        e permanece estavel nas proximas 3 camadas.

        A verificacao de estabilidade (tolerancia de 50%) filtra
        ativacoes espurias -- picos momentaneos que nao representam
        comprometimento real do modelo.

        Args:
            layer_results: lista de dicts por camada
            language:      idioma atual
            threshold:     probabilidade minima (default: 0.10)

        Returns:
            Indice absoluto da camada de convergencia, ou None.
        """
        entities = self.expected_entities.get(language, [])
        if not entities:
            return None

        n_layers = len(layer_results)

        for i, layer_data in enumerate(layer_results):
            prob = self._entity_prob_at_layer(layer_data, entities)

            if prob >= threshold:
                stable = True
                for j in range(i + 1, min(i + 4, n_layers)):
                    next_prob = self._entity_prob_at_layer(
                        layer_results[j], entities
                    )
                    if next_prob < threshold * 0.5:
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
        Estabilidade da Decisao (ED).

        Primeira camada a partir da qual o token top-1 nao muda
        nas proximas 4 camadas consecutivas.

        Args:
            layer_results: lista de dicts por camada

        Returns:
            Indice absoluto da camada de estabilizacao, ou None.
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

    def compute_iv(
        self,
        layer_results: List[Dict],
        language: str
    ) -> float:
        """
        Intensidade do Vies -- entidade local (IV_local).

        P(entidade local) na ultima camada.

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

    def compute_commitment_score(
        self,
        layer_results: List[Dict],
        language: str,
        competing_entities: Dict[str, List[str]]
    ) -> List[float]:
        """
        Commitment Score por camada:

            CS[layer] = P(entidade local) - P(entidade concorrente)

        Interpretacao:
            CS > 0  -> modelo favorece narrativa local
            CS ~= 0 -> modelo ambiguo (as narrativas competem)
            CS < 0  -> modelo favorece narrativa concorrente
                       mesmo quando perguntado naquele idioma

        Esse e o sinal mais rico do paper -- distingue:
            Cenario A (vies forte):   PT: Santos=0.40, Wright=0.05 -> CS=+0.35
            Cenario B (ambiguidade):  PT: Santos=0.22, Wright=0.20 -> CS=+0.02

        Ambos teriam IV_local similar mas CS completamente diferente.

        Args:
            layer_results:      lista de dicts por camada
            language:           idioma atual
            competing_entities: dict {lang: [tokens da entidade concorrente]}

        Returns:
            Lista de floats com CS por camada (mesmo comprimento que layer_results).
        """
        local_ents     = self.expected_entities.get(language, [])
        competing_ents = competing_entities.get(language, [])

        scores = []
        for layer in layer_results:
            p_local      = self._entity_prob_at_layer(layer, local_ents)
            p_competing  = self._entity_prob_at_layer(layer, competing_ents)
            scores.append(round(p_local - p_competing, 6))

        return scores

    def compute_dl(
        self,
        layer_results_lang1: List[Dict],
        layer_results_lang2: List[Dict],
        language1: str,
        language2: str
    ) -> List[float]:
        """
        Divergencia entre Linguas (DL).

        Diferenca absoluta entre P(entidade local) de cada idioma,
        calculada camada a camada.

        Usada no Cultural Divergence Map: valores crescentes indicam
        que as linguas estao se "separando" culturalmente.

        Args:
            layer_results_lang1: resultados por camada do idioma 1
            layer_results_lang2: resultados por camada do idioma 2
            language1:           codigo do idioma 1 (ex: "pt")
            language2:           codigo do idioma 2 (ex: "en")

        Returns:
            Lista de floats com DL por camada.
        """
        entities1 = self.expected_entities.get(language1, [])
        entities2 = self.expected_entities.get(language2, [])
        divergences = []

        for l1, l2 in zip(layer_results_lang1, layer_results_lang2):
            prob1 = self._entity_prob_at_layer(l1, entities1)
            prob2 = self._entity_prob_at_layer(l2, entities2)
            divergences.append(round(abs(prob1 - prob2), 6))

        return divergences

    def compute_all_metrics(
        self,
        layer_results: List[Dict],
        language: str,
        competing_entities: Dict[str, List[str]] = None
    ) -> Dict:
        """
        Calcula todas as metricas para um unico run.

        Se competing_entities for fornecido, calcula tambem
        iv_competing, commitment_final e commitment_curve.

        Args:
            layer_results:      lista de dicts por camada
            language:           idioma
            competing_entities: dict {lang: [tokens concorrentes]}
                                (opcional -- se None, metricas CS nao sao calculadas)

        Returns:
            Dict com todas as metricas. Campos garantidos:
                pcc, pcc_norm, ed, ed_norm,
                iv_local, iv_competing, commitment_final, commitment_curve
        """
        n_layers = len(layer_results)

        pcc      = self.compute_pcc(layer_results, language)
        ed       = self.compute_ed(layer_results)
        iv_local = self.compute_iv(layer_results, language)

        # Normalizar indices de camada para [0, 1]
        pcc_norm = round((pcc - 1) / (n_layers - 1), 4) if pcc else None
        ed_norm  = round((ed  - 1) / (n_layers - 1), 4) if ed  else None

        # Metricas de Commitment Score (requerem competing_entities)
        iv_competing     = 0.0
        commitment_final = iv_local
        commitment_curve = []

        if competing_entities:
            competing_ents   = competing_entities.get(language, [])
            iv_competing     = self._entity_prob_at_layer(
                layer_results[-1], competing_ents
            )
            commitment_final = iv_local - iv_competing
            commitment_curve = self.compute_commitment_score(
                layer_results, language, competing_entities
            )

        return {
            # Metricas de convergencia
            "pcc":              pcc,
            "pcc_norm":         pcc_norm,
            "ed":               ed,
            "ed_norm":          ed_norm,

            # Probabilidades na ultima camada
            "iv_local":         round(iv_local, 6),
            "iv_competing":     round(iv_competing, 6),

            # Commitment Score -- a metrica mais rica
            # CS > 0: favorece local | CS ~= 0: ambiguo | CS < 0: favorece concorrente
            "commitment_final": round(commitment_final, 6),

            # Evolucao do CS por camada -- base do Cultural Divergence Map
            "commitment_curve": commitment_curve,
        }
