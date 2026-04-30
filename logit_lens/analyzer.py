from typing import Dict, List, Optional


class LogitLensAnalyzer:

    def __init__(self, expected_entities: Dict[str, List[str]]):
        self.expected_entities = expected_entities

    def _entity_prob_at_layer(
        self,
        layer_data: Dict,
        entities: List[str]
    ) -> float:
        """
        Probabilidade acumulada das entidades nos top-K tokens.

        Para cada token no top-K:
          1. Remove espacos e converte para minusculas (normaliza)
          2. Verifica se alguma entidade e substring do token normalizado
          3. Soma a probabilidade se houver match (break evita double-counting)

        Normalizacao garante que " Alexander", "ALEXANDER" e "alexander"
        sejam todos capturados corretamente.

        Entidades curtas (ex: "Or" para Orville) podem causar falsos
        positivos em tokens como "inventor" ou "original". Por isso,
        o experiment_config nao deve incluir entidades com menos de
        3 caracteres, exceto quando o token curto e esperado no top-K
        (ex: "Or" como prefixo de "Orville" nos tokenizers BPE).
        """
        total = 0.0
        for token, prob in zip(layer_data["top_tokens"], layer_data["top_probs"]):
            t = token.strip().lower()
            for entity in entities:
                if entity.lower() in t:
                    total += prob
                    break  # evitar double-counting por token
        return total

    def compute_all_metrics(
        self,
        layer_results: List[Dict],
        language: str,
        competing_entities: Optional[Dict[str, List[str]]] = None
    ) -> Dict:
        """
        Calcula todas as metricas para um unico run.

        Args:
            layer_results:      lista de dicts por camada, cada um com:
                                  layer, layer_norm, top_tokens, top_probs
            language:           codigo do idioma (ex: "pt", "en", "de", "it")
            competing_entities: dict {lang: [tokens da entidade concorrente]}
                                Se None, metricas de CS nao sao calculadas.

        Returns:
            Dict com todas as metricas. Campos garantidos:
                iv_local_final, iv_local_peak, iv_competing,
                commitment_final, cs_at_peak, suppression,
                peak_layer, peak_layer_norm,
                pcc, pcc_norm,
                commitment_curve, p_local_curve
        """
        if not layer_results:
            return {}

        local_ents = self.expected_entities.get(language, [])
        comp_ents  = (competing_entities or {}).get(language, [])

        # ── Curvas de probabilidade por camada ──
        p_local_curve = [
            self._entity_prob_at_layer(l, local_ents)
            for l in layer_results
        ]
        p_comp_curve = [
            self._entity_prob_at_layer(l, comp_ents)
            for l in layer_results
        ]
        cs_curve = [
            round(pl - pc, 6)
            for pl, pc in zip(p_local_curve, p_comp_curve)
        ]

        # ── Metricas da ultima camada ──
        iv_local_final = p_local_curve[-1]
        iv_comp_final  = p_comp_curve[-1]
        cs_final       = cs_curve[-1]

        # ── Metricas do pico ──
        # O pico captura o momento em que o modelo "sabe" a resposta
        # antes das camadas finais sobrescreverem com tokens gramaticais
        # (artigos, determinantes, etc.)
        iv_local_peak = max(p_local_curve) if p_local_curve else 0.0
        peak_idx      = p_local_curve.index(iv_local_peak)
        peak_layer    = layer_results[peak_idx]["layer"]
        peak_norm     = round(layer_results[peak_idx]["layer_norm"], 4)
        cs_at_peak    = round(cs_curve[peak_idx], 6)

        # Supressao: diferenca entre o que o modelo "sabe" no pico
        # e o que efetivamente "gera" na ultima camada
        suppression = round(iv_local_peak - iv_local_final, 6)

        # ── PCC: Ponto de Convergencia Cultural ──
        # Primeira camada onde P(local) >= 0.05, indicando que
        # a entidade cultural emergiu nas representacoes internas
        threshold = 0.05
        pcc      = None
        pcc_norm = None
        for i, p in enumerate(p_local_curve):
            if p >= threshold:
                pcc      = layer_results[i]["layer"]
                pcc_norm = round(layer_results[i]["layer_norm"], 4)
                break

        return {
            # Probabilidades na ultima camada
            "iv_local_final":   round(iv_local_final, 6),
            "iv_competing":     round(iv_comp_final, 6),
            "commitment_final": round(cs_final, 6),

            # Metricas do pico — principal para analise cultural
            "iv_local_peak":    round(iv_local_peak, 6),
            "peak_layer":       peak_layer,
            "peak_layer_norm":  peak_norm,
            "cs_at_peak":       cs_at_peak,

            # Supressao gramatical nas camadas finais
            "suppression":      suppression,

            # Ponto de Convergencia Cultural
            "pcc":              pcc,
            "pcc_norm":         pcc_norm,

            # Curvas completas (para visualizacao)
            "commitment_curve": [round(v, 6) for v in cs_curve],
            "p_local_curve":    [round(v, 6) for v in p_local_curve],

            # Compatibilidade retroativa com versoes anteriores
            "iv_local":         round(iv_local_final, 6),
        }