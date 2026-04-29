import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class LogitLensExtractor:
    """
    Extrai distribuições de probabilidade por camada
    usando a técnica Logit Lens.
    
    Para cada camada L e posição de token P:
        1. Pega o hidden state h[L][P]
        2. Aplica layer norm final
        3. Multiplica pela unembedding matrix
        4. Aplica softmax → distribuição sobre vocabulário
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        top_k: int = 10
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.device = next(model.parameters()).device
        
        # Extrair componentes necessários para o Logit Lens
        self.ln_final = self._get_final_layer_norm()
        self.unembedding = self._get_unembedding_matrix()
        
    def _get_final_layer_norm(self):
        """
        Extrai a layer norm final do modelo.
        Cada arquitetura tem um nome diferente para esse componente.
        """
        model_type = self.model.config.model_type.lower()
        
        # Mapeamento por arquitetura
        ln_map = {
            "llama":   self.model.model.norm,
            "mistral": self.model.model.norm,
            "qwen2":   self.model.model.norm,
            "gemma2":  self.model.model.norm,
        }
        
        for key, ln in ln_map.items():
            if key in model_type:
                return ln
        
        # Fallback genérico
        raise ValueError(
            f"Arquitetura '{model_type}' não mapeada. "
            f"Adicione ao ln_map em _get_final_layer_norm()."
        )
    
    def _get_unembedding_matrix(self):
        """
        Extrai a matriz de unembedding (lm_head).
        É a mesma matriz de embedding transposta na maioria dos modelos.
        """
        return self.model.lm_head
    
    def _apply_logit_lens(
        self,
        hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Aplica Logit Lens em um hidden state.
        
        Args:
            hidden_state: tensor [seq_len, hidden_size] ou [hidden_size]
        
        Returns:
            logits: tensor com distribuição sobre vocabulário
        """
        with torch.no_grad():
            # Aplicar layer norm final
            normed = self.ln_final(hidden_state)
            # Projetar no espaço do vocabulário
            logits = self.unembedding(normed)
            # Converter para probabilidades
            probs = F.softmax(logits, dim=-1)
        return probs
    
    def find_keyword_position(
        self,
        input_ids: torch.Tensor,
        keyword: str
    ) -> Optional[int]:
        """
        Encontra a posição do token da palavra-chave central no prompt.
        
        Estratégia: tokenizar a keyword separadamente e procurar
        a subsequência no input_ids.
        
        Args:
            input_ids: tokens do prompt completo [seq_len]
            keyword:   palavra-chave a encontrar (ex: "avião")
        
        Returns:
            Posição do último token da keyword, ou None se não encontrada
        """
        # Tokenizar a keyword (sem tokens especiais)
        keyword_ids = self.tokenizer.encode(
            keyword,
            add_special_tokens=False
        )
        
        # Buscar a subsequência nos input_ids
        input_list = input_ids.tolist()
        keyword_len = len(keyword_ids)
        
        for i in range(len(input_list) - keyword_len + 1):
            if input_list[i:i + keyword_len] == keyword_ids:
                # Retorna posição do ÚLTIMO token da keyword
                return i + keyword_len - 1
        
        # Fallback: tentar com variações de capitalização
        for variant in [keyword.lower(), keyword.upper(), keyword.capitalize()]:
            variant_ids = self.tokenizer.encode(
                variant,
                add_special_tokens=False
            )
            variant_len = len(variant_ids)
            for i in range(len(input_list) - variant_len + 1):
                if input_list[i:i + variant_len] == variant_ids:
                    return i + variant_len - 1
        
        print(f"  ⚠️  Keyword '{keyword}' não encontrada nos tokens do prompt")
        return None
    
    def extract(
        self,
        prompt: str,
        keyword: Optional[str] = None,
        last_relevant_pos: Optional[int] = None  # NOVO PARÂMETRO
    ) -> Dict:
        """
        Extrai distribuições Logit Lens para um prompt.
        
        Args:
            prompt:  texto do prompt
            keyword: palavra-chave central (opcional)
        
        Returns:
            Dicionário com:
            - 'last_token':    análise do último token por camada
            - 'keyword_token': análise do token da keyword por camada
            - 'tokens':        lista de tokens do prompt
            - 'n_layers':      número de camadas
            - 'keyword_pos':   posição da keyword (ou None)
        """
        
        # Tokenizar o prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        
        input_ids = inputs["input_ids"][0]  # [seq_len]

        # Usar posição fornecida ou inferir
        last_pos = (
            last_relevant_pos 
            if last_relevant_pos is not None 
            else len(input_ids) - 1
        )
        
        # Decodificar tokens para visualização
        tokens = [
            self.tokenizer.decode([t]) 
            for t in input_ids.tolist()
        ]
        
        # Encontrar posição da keyword se fornecida
        keyword_pos = None
        if keyword is not None:
            keyword_pos = self.find_keyword_position(input_ids, keyword)
        
        # Forward pass — capturar todos os hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )
        
        # outputs.hidden_states: tuple de [n_layers+1] tensors
        # Cada tensor: [batch, seq_len, hidden_size]
        # Index 0 = embedding layer, 1..N = transformer layers
        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states) - 1  # excluindo embedding
        
        # Posição do último token
        #last_pos = len(input_ids) - 1
        
        # Estruturas para armazenar resultados
        results = {
            "tokens":       tokens,
            "n_layers":     n_layers,
            "keyword_pos":  keyword_pos,
            "last_token":   [],
            "keyword_token": [] if keyword_pos is not None else None
        }
        
        # Iterar sobre camadas (excluindo embedding layer)
        for layer_idx in range(1, len(hidden_states)):
            layer_hidden = hidden_states[layer_idx][0]  # [seq_len, hidden_size]
            
            # === Análise do último token ===
            last_hidden = layer_hidden[last_pos]  # [hidden_size]
            last_probs = self._apply_logit_lens(last_hidden)
            
            # Top-K tokens e probabilidades
            top_probs, top_indices = torch.topk(last_probs, self.top_k)
            top_tokens = [
                self.tokenizer.decode([idx.item()])
                for idx in top_indices
            ]
            
            results["last_token"].append({
                "layer":      layer_idx,
                "top_tokens": top_tokens,
                "top_probs":  top_probs.cpu().numpy().tolist()
            })
            
            # === Análise do token da keyword ===
            if keyword_pos is not None:
                kw_hidden = layer_hidden[keyword_pos]  # [hidden_size]
                kw_probs = self._apply_logit_lens(kw_hidden)
                
                top_probs_kw, top_indices_kw = torch.topk(kw_probs, self.top_k)
                top_tokens_kw = [
                    self.tokenizer.decode([idx.item()])
                    for idx in top_indices_kw
                ]
                
                results["keyword_token"].append({
                    "layer":      layer_idx,
                    "top_tokens": top_tokens_kw,
                    "top_probs":  top_probs_kw.cpu().numpy().tolist()
                })
        
        return results
    
    def get_final_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Gera a resposta final do modelo para o ground truth.
        Usado na Fase 1 de validação comportamental.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,       # greedy
                temperature=1.0,       # ignorado com do_sample=False
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decodificar apenas os tokens gerados (não o prompt)
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)