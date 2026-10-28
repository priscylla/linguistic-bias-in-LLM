"""
logit_lens/analyzer.py
Métricas por run a partir das probabilidades exatas por camada.

As probabilidades das entidades vêm de `entity_probs` em cada camada,
calculadas pelo extractor como a soma exata sobre os token IDs de cada
conjunto, na distribuição completa. Não há comparação de substrings
nem truncamento top-k.
"""

from typing import Dict, List


def _set_curve(layer_results: List[Dict], set_name: str) -> List[float]:
    """Curva de P(conjunto) por camada; falha se o formato for antigo."""
    curve = []
    for l in layer_results:
        if "entity_probs" not in l:
            raise ValueError(
                f"Camada {l.get('layer')} sem 'entity_probs': resultado no "
                f"formato antigo (só top-k). Rode o Logit Lens de novo; "
                f"métricas não são mais calculadas por substring no top-k."
            )
        curve.append(float(l["entity_probs"].get(set_name, 0.0)))
    return curve


class LogitLensAnalyzer:

    def __init__(self, local_set: str = "local", competing_set: str = "competing"):
        # Nomes dos conjuntos em entity_probs (definidos no extract)
        self.local_set     = local_set
        self.competing_set = competing_set

    def compute_all_metrics(self, layer_results: List[Dict]) -> Dict:
        """
        Calcula todas as metricas para um unico run.

        Args:
            layer_results: lista de dicts por camada, cada um com:
                           layer, layer_norm, entity_probs

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

        # ── Curvas de probabilidade por camada (valores exatos) ──
        p_local_curve = _set_curve(layer_results, self.local_set)
        p_comp_curve  = _set_curve(layer_results, self.competing_set)
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