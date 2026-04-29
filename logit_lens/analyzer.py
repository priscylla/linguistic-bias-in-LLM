import numpy as np
from typing import Dict, List, Optional, Tuple


class LogitLensAnalyzer:
    """
    Calcula as métricas definidas no design experimental
    a partir dos resultados brutos do extrator.
    
    Métricas:
        PCC: Ponto de Convergência Cultural
        DL:  Divergência entre Línguas
        ED:  Estabilidade da Decisão
        IV:  Intensidade do Viés
    """
    
    def __init__(self, expected_entities: Dict[str, List[str]]):
        """
        Args:
            expected_entities: dict mapeando idioma → lista de tokens
                               que representam a entidade cultural esperada
                               ex: {"pt": ["Santos", "Dumont"], "en": ["Wright"]}
        """
        self.expected_entities = expected_entities
    
    def _entity_prob_at_layer(
        self,
        layer_data: Dict,
        entities: List[str]
    ) -> float:
        """
        Calcula probabilidade acumulada das entidades esperadas
        nos top-K tokens de uma camada.
        
        Estratégia: soma as probabilidades de todos os top-K tokens
        que contêm alguma das strings de entidade esperada.
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
        Ponto de Convergência Cultural (PCC):
        Primeira camada onde a entidade esperada para esse idioma
        entra no top-K com probabilidade >= threshold E permanece.
        
        Args:
            layer_results: lista de dicts por camada (last_token ou keyword_token)
            language:      idioma atual
            threshold:     probabilidade mínima para considerar convergência
        
        Returns:
            Índice da camada de convergência, ou None se não convergir
        """
        entities = self.expected_entities.get(language, [])
        if not entities:
            return None
        
        n_layers = len(layer_results)
        
        for i, layer_data in enumerate(layer_results):
            prob = self._entity_prob_at_layer(layer_data, entities)
            
            if prob >= threshold:
                # Verificar se permanece nas próximas 3 camadas
                # para garantir que não é uma ativação espúria
                stable = True
                for j in range(i + 1, min(i + 4, n_layers)):
                    next_prob = self._entity_prob_at_layer(
                        layer_results[j], entities
                    )
                    if next_prob < threshold * 0.5:  # margem de tolerância
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
        Estabilidade da Decisão (ED):
        Camada a partir da qual o token top-1 não muda mais.
        
        Returns:
            Índice da camada de estabilização, ou None
        """
        n_layers = len(layer_results)
        
        for i in range(n_layers - 4):
            top1 = layer_results[i]["top_tokens"][0].strip().lower()
            
            # Verificar se top-1 é estável nas próximas 4 camadas
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
        Divergência entre Línguas (DL):
        Diferença de probabilidade para as entidades locais,
        calculada camada a camada.
        
        Retorna lista de valores por camada para plotagem.
        """
        divergences = []
        
        for l1, l2 in zip(layer_results_lang1, layer_results_lang2):
            entities1 = self.expected_entities.get(language1, [])
            entities2 = self.expected_entities.get(language2, [])
            
            prob1 = self._entity_prob_at_layer(l1, entities1)
            prob2 = self._entity_prob_at_layer(l2, entities2)
            
            # DL = diferença absoluta entre as probabilidades
            # das entidades "locais" em cada idioma
            divergences.append(abs(prob1 - prob2))
        
        return divergences
    
    def compute_iv(
        self,
        layer_results: List[Dict],
        language: str
    ) -> float:
        """
        Intensidade do Viés (IV):
        Probabilidade da entidade local no token final (última camada).
        Permite rankear modelos por grau de viés.
        """
        entities = self.expected_entities.get(language, [])
        if not entities:
            return 0.0
        
        # Última camada
        last_layer = layer_results[-1]
        return self._entity_prob_at_layer(last_layer, entities)
    
    def compute_all_metrics(
        self,
        layer_results: List[Dict],
        language: str
    ) -> Dict:
        """
        Calcula todas as métricas para um único run.
        """
        return {
            "pcc": self.compute_pcc(layer_results, language),
            "ed":  self.compute_ed(layer_results),
            "iv":  self.compute_iv(layer_results, language)
        }