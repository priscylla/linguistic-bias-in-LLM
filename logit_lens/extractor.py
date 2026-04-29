"""
logit_lens/extractor.py
Extração de distribuições de probabilidade por camada
usando a técnica Logit Lens (nostalgebraist, 2020).

Para cada camada L e posição de token P:
    1. Pega o hidden state h[L][P]
    2. Aplica a layer norm final do modelo
    3. Projeta no espaço do vocabulário via lm_head
    4. Aplica softmax → distribuição de probabilidade
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class LogitLensExtractor:
    """
    Extrai distribuições de probabilidade por camada (Logit Lens).

    Suporta dois pontos de análise por prompt:
        - last_token:    último token antes da geração (posição de decisão)
        - keyword_token: token da palavra-chave central do fato histórico
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        top_k: int = 10
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.top_k     = top_k
        self.device    = next(model.parameters()).device

        # Extrair componentes necessários para o Logit Lens
        self.ln_final    = self._get_final_layer_norm()
        self.unembedding = self._get_unembedding_matrix()

    def _get_final_layer_norm(self):
        """
        Extrai a layer norm final do modelo.

        Cada arquitetura usa um nome diferente para esse componente,
        mas todos os nossos 4 modelos usam model.model.norm.
        O mapeamento explícito protege contra adição de novos modelos.
        """
        model_type = self.model.config.model_type.lower()

        ln_map = {
            "llama":   self.model.model.norm,
            "mistral": self.model.model.norm,
            "qwen2":   self.model.model.norm,
            "gemma2":  self.model.model.norm,
        }

        for key, ln in ln_map.items():
            if key in model_type:
                return ln

        raise ValueError(
            f"Arquitetura '{model_type}' não mapeada.\n"
            f"Adicione ao ln_map em _get_final_layer_norm().\n"
            f"Inspecione: model.config.model_type e model.model"
        )

    def _get_unembedding_matrix(self):
        """
        Extrai a matriz de unembedding (lm_head).

        Em modelos com weight tying, lm_head compartilha pesos
        com a camada de embedding. Em ambos os casos, é o componente
        correto para projetar hidden states no espaço do vocabulário.
        """
        return self.model.lm_head

    def _apply_logit_lens(
        self,
        hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Aplica Logit Lens em um hidden state.

        Implementa as 3 operações do método original:
            normed = LayerNorm_final(h)
            logits = Unembedding(normed)
            probs  = Softmax(logits)

        Args:
            hidden_state: tensor de shape [hidden_size]

        Returns:
            probs: tensor de shape [vocab_size] com distribuição
                   de probabilidade sobre o vocabulário
        """
        with torch.no_grad():
            normed = self.ln_final(hidden_state)
            logits = self.unembedding(normed)
            probs  = F.softmax(logits, dim=-1)
        return probs

    def find_keyword_position(
        self,
        input_ids: torch.Tensor,
        keyword: str
    ) -> Optional[int]:
        """
        Encontra a posição do token da palavra-chave no prompt.

        Método simplificado — para uso quando keyword_pos_override
        não é fornecido. Para uso em produção, preferir o módulo
        data/keyword_finder.py que tem estratégias mais robustas.

        Args:
            input_ids: tokens do prompt completo [seq_len]
            keyword:   palavra-chave a encontrar (ex: "avião")

        Returns:
            Posição do último token da keyword, ou None.
        """
        keyword_ids = self.tokenizer.encode(
            keyword, add_special_tokens=False
        )
        input_list  = input_ids.tolist()
        keyword_len = len(keyword_ids)

        # Busca da direita para esquerda — última ocorrência
        for i in range(len(input_list) - keyword_len, -1, -1):
            if input_list[i:i + keyword_len] == keyword_ids:
                return i + keyword_len - 1

        # Fallback: variações de capitalização
        for variant in [keyword.lower(), keyword.upper(), keyword.capitalize()]:
            variant_ids = self.tokenizer.encode(
                variant, add_special_tokens=False
            )
            variant_len = len(variant_ids)
            for i in range(len(input_list) - variant_len, -1, -1):
                if input_list[i:i + variant_len] == variant_ids:
                    return i + variant_len - 1

        print(f"  ⚠️  Keyword '{keyword}' não encontrada nos tokens do prompt")
        return None

    def extract(
        self,
        prompt: str,
        keyword: Optional[str] = None,
        last_relevant_pos: Optional[int] = None,
        keyword_pos_override: Optional[int] = None
    ) -> Dict:
        """
        Extrai distribuições Logit Lens para um prompt.

        Realiza um único forward pass e aplica o Logit Lens
        em cada camada, para as posições de interesse.

        Args:
            prompt:               prompt já formatado com chat template
            keyword:              palavra-chave central (opcional)
            last_relevant_pos:    posição do último token relevante
                                  (fornecida pelo chat_template.py)
            keyword_pos_override: posição da keyword já calculada
                                  (fornecida pelo keyword_finder.py)

        Returns:
            Dict com:
                'tokens':        lista de tokens decodificados do prompt
                'n_layers':      número de camadas transformer
                'keyword_pos':   posição da keyword (int ou None)
                'last_token':    lista de dicts por camada (último token)
                'keyword_token': lista de dicts por camada (keyword token)
                                 ou None se keyword_pos for None

            Cada dict de camada contém:
                'layer':      índice da camada (1-based)
                'top_tokens': lista de top_k tokens decodificados
                'top_probs':  lista de top_k probabilidades
        """

        # Tokenizar o prompt
        inputs     = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids  = inputs["input_ids"][0]   # [seq_len]

        # Decodificar tokens para visualização e debug
        tokens = [
            self.tokenizer.decode([t]) for t in input_ids.tolist()
        ]

        # Posição do último token relevante
        # Prioridade: parâmetro explícito > último token da sequência
        last_pos = (
            last_relevant_pos
            if last_relevant_pos is not None
            else len(input_ids) - 1
        )

        # Posição da keyword
        # Prioridade: override explícito > busca automática > None
        if keyword_pos_override is not None:
            keyword_pos = keyword_pos_override
        elif keyword is not None:
            keyword_pos = self.find_keyword_position(input_ids, keyword)
        else:
            keyword_pos = None

        # Forward pass com captura de todos os hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        # outputs.hidden_states: tuple de (n_layers + 1) tensors
        # Index 0 = embedding layer (antes do primeiro transformer block)
        # Index 1..N = saídas dos N transformer layers
        # Cada tensor: [batch=1, seq_len, hidden_size]
        hidden_states = outputs.hidden_states
        n_layers      = len(hidden_states) - 1   # excluindo embedding

        results = {
            "tokens":        tokens,
            "n_layers":      n_layers,
            "keyword_pos":   keyword_pos,
            "last_token":    [],
            "keyword_token": [] if keyword_pos is not None else None
        }

        # Iterar sobre camadas transformer (índices 1..N)
        for layer_idx in range(1, len(hidden_states)):
            # [seq_len, hidden_size] — remover dimensão de batch
            layer_hidden = hidden_states[layer_idx][0]

            # ── Último token ──
            last_hidden = layer_hidden[last_pos]       # [hidden_size]
            last_probs  = self._apply_logit_lens(last_hidden)

            top_probs_last, top_idx_last = torch.topk(last_probs, self.top_k)
            results["last_token"].append({
                "layer":      layer_idx,
                "top_tokens": [
                    self.tokenizer.decode([idx.item()])
                    for idx in top_idx_last
                ],
                "top_probs":  top_probs_last.cpu().numpy().tolist()
            })

            # ── Token da keyword ──
            if keyword_pos is not None:
                kw_hidden = layer_hidden[keyword_pos]  # [hidden_size]
                kw_probs  = self._apply_logit_lens(kw_hidden)

                top_probs_kw, top_idx_kw = torch.topk(kw_probs, self.top_k)
                results["keyword_token"].append({
                    "layer":      layer_idx,
                    "top_tokens": [
                        self.tokenizer.decode([idx.item()])
                        for idx in top_idx_kw
                    ],
                    "top_probs":  top_probs_kw.cpu().numpy().tolist()
                })

        return results

    def get_final_response(
        self,
        prompt: str,
        max_new_tokens: int = 200
    ) -> str:
        """
        Gera a resposta final do modelo (ground truth).

        Usado na Fase 1 do pipeline para validação comportamental —
        confirmar que o viés linguístico existe nas respostas finais
        antes de analisar as ativações internas.

        Usa greedy decoding (do_sample=False) para reprodutibilidade,
        igual à metodologia do TalkTuner.

        Args:
            prompt:         prompt já formatado com chat template
            max_new_tokens: limite de tokens a gerar

        Returns:
            Texto gerado (sem o prompt original)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy — reprodutível
                temperature=1.0,          # ignorado com do_sample=False
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decodificar apenas os tokens gerados — não o prompt
        prompt_len = inputs["input_ids"].shape[1]
        generated  = output[0][prompt_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
