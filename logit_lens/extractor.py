"""
logit_lens/extractor.py
Extração de distribuições de probabilidade por camada
usando a técnica Logit Lens (nostalgebraist, 2020).

Para cada camada L e posição de token P:
    1. Pega o hidden state h[L][P]
    2. Aplica a layer norm final do modelo (exceto na última camada,
       que já sai normalizada do Hugging Face — ver extract())
    3. Projeta no espaço do vocabulário via lm_head
    4. Aplica o softcapping de logits, se o modelo tiver (Gemma 2)
    5. Aplica softmax → distribuição de probabilidade

Teste de sanidade: na última camada, o lens tem que reproduzir
exatamente os logits do próprio modelo. Se não reproduzir, o lens
está lendo o modelo errado e nenhum resultado deve ser gerado.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from logit_lens.entity_tokens import build_token_sets, prob_mass


class LensSanityError(RuntimeError):
    """O lens na última camada não reproduz os logits do modelo."""


# Prompts fixos do teste de sanidade (conteúdo irrelevante: só importa
# que o lens na última camada bata com os logits do modelo)
SANITY_PROMPTS = [
    "The airplane was invented by",
    "O telefone foi inventado por",
    "Das Telefon wurde erfunden von",
]


def lens_tolerance(dtype: torch.dtype) -> Dict[str, float]:
    """
    Tolerância do teste de sanidade conforme a precisão do modelo.

    Em float32 a diferença é só ruído numérico do matmul.
    Em bf16/fp16 (~3 dígitos significativos) o lm_head aplicado a um
    vetor isolado pode diferir do aplicado à sequência inteira na
    última casa, por isso a folga maior.
    """
    if dtype == torch.float32 or dtype == torch.float64:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 2e-2, "rtol": 2e-2}


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

    def _logits_from_hidden(
        self,
        hidden_state: torch.Tensor,
        already_normed: bool = False
    ) -> torch.Tensor:
        """
        Projeta um hidden state em logits do vocabulário, reproduzindo
        a cabeça de saída do modelo:

            normed = LayerNorm_final(h)        (pulado se already_normed)
            logits = Unembedding(normed)
            logits = c * tanh(logits / c)      (só se o config definir
                                                final_logit_softcapping)

        O softcapping faz parte da cabeça de saída do Gemma 2
        (Gemma2ForCausalLM.forward), então é aplicado em todas as
        camadas, como o próprio modelo faria.

        Args:
            hidden_state:   tensor [..., hidden_size]
            already_normed: True se o hidden state já passou pela
                            norm final (última camada no HF)

        Returns:
            logits em float32, shape [..., vocab_size]
        """
        with torch.no_grad():
            # Com device_map em várias GPUs, norm/lm_head podem estar
            # em um device diferente do hidden state
            head_device = self.unembedding.weight.device
            h = hidden_state.to(head_device)

            normed = h if already_normed else self.ln_final(h)
            logits = self.unembedding(normed)

            cap = getattr(self.model.config, "final_logit_softcapping", None)
            if cap is not None:
                logits = torch.tanh(logits / cap) * cap

        return logits.float()

    def _apply_logit_lens(
        self,
        hidden_state: torch.Tensor,
        already_normed: bool = False
    ) -> torch.Tensor:
        """
        Aplica Logit Lens em um hidden state.

        Args:
            hidden_state:   tensor de shape [hidden_size]
            already_normed: True para a última camada (já normalizada)

        Returns:
            probs: tensor de shape [vocab_size] (float32) com
                   distribuição de probabilidade sobre o vocabulário
        """
        logits = self._logits_from_hidden(hidden_state, already_normed)
        return F.softmax(logits, dim=-1)

    def check_lens_consistency(self, outputs, pos: int) -> Dict:
        """
        Confere que o lens na última camada reproduz os logits do modelo
        na posição `pos`. Falha alto (LensSanityError) se não reproduzir.

        Args:
            outputs: saída do forward com output_hidden_states=True
            pos:     posição do token analisado

        Returns:
            Dict com 'max_abs_diff' e 'argmax_match'
        """
        hidden_last = outputs.hidden_states[-1][0, pos]
        lens_logits = self._logits_from_hidden(hidden_last, already_normed=True)
        model_logits = outputs.logits[0, pos].float().to(lens_logits.device)

        tol      = lens_tolerance(outputs.logits.dtype)
        max_diff = (lens_logits - model_logits).abs().max().item()
        argmax_ok = lens_logits.argmax().item() == model_logits.argmax().item()
        close_ok  = torch.allclose(lens_logits, model_logits, **tol)

        if not (close_ok and argmax_ok):
            raise LensSanityError(
                f"❌ Logit Lens na última camada não bate com os logits do "
                f"modelo ({self.model.config.model_type}, "
                f"dtype={outputs.logits.dtype}, posição {pos}): "
                f"max |diff| = {max_diff:.3e}, tolerância = {tol}, "
                f"argmax igual = {argmax_ok}. "
                f"Verifique norm final, softcapping e lm_head."
            )

        return {"max_abs_diff": max_diff, "argmax_match": argmax_ok}

    def sanity_check(self, prompts: Optional[List[str]] = None) -> Dict:
        """
        Teste de sanidade do lens: roda alguns prompts fixos e confere,
        na última posição de cada um, que o lens na última camada é
        igual aos logits do modelo. Levanta LensSanityError se falhar.

        Returns:
            Dict com o resultado por prompt e o pior caso
        """
        prompts = prompts or SANITY_PROMPTS
        report  = {"model_type": self.model.config.model_type,
                   "dtype": str(next(self.model.parameters()).dtype),
                   "prompts": []}

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            pos = inputs["input_ids"].shape[1] - 1
            res = self.check_lens_consistency(outputs, pos)
            report["prompts"].append({"prompt": prompt, **res})

        report["max_abs_diff"] = max(p["max_abs_diff"] for p in report["prompts"])
        print(
            f"  ✅ Sanidade do lens OK ({report['model_type']}, "
            f"{report['dtype']}): max |diff| = {report['max_abs_diff']:.2e}"
        )
        return report

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
        keyword_pos_override: Optional[int] = None,
        entity_sets: Optional[Dict[str, List[str]]] = None
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
            entity_sets:          {nome: [palavras]}, ex.
                                  {"local": [...], "competing": [...]}.
                                  Se dado, cada camada recebe a
                                  probabilidade exata de cada conjunto
                                  (soma sobre token IDs distintos, na
                                  distribuição completa — sem top-k)

        Returns:
            Dict com:
                'tokens':        lista de tokens decodificados do prompt
                'n_layers':      número de camadas transformer
                'keyword_pos':   posição da keyword (int ou None)
                'last_token':    lista de dicts por camada (último token)
                'keyword_token': lista de dicts por camada (keyword token)
                                 ou None se keyword_pos for None
                'entity_tokens': conjuntos de token IDs usados (auditoria)
                                 ou None se entity_sets não for dado

            Cada dict de camada contém:
                'layer':      índice da camada (1-based)
                'top_tokens': lista de top_k tokens decodificados
                'top_ids':    ids dos top_k tokens
                'top_probs':  lista de top_k probabilidades
                'entity_probs': {nome: P exata} (se entity_sets dado)

            O top-k é só para inspeção qualitativa; métricas usam
            entity_probs.
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

        # No transformers (verificado na 5.7, utils/output_capturing.py,
        # tie_last_hidden_states=True), hidden_states[-1] é substituído
        # por last_hidden_state, que JÁ passou por model.model.norm.
        # Aplicar a norm de novo nessa camada distorcia a distribuição.
        # As camadas 1..N-1 são o residual stream cru e recebem a norm.
        # O teste de sanidade abaixo garante que isso continua valendo.
        self.check_lens_consistency(outputs, last_pos)

        # Token IDs exatos de cada conjunto de entidades, construídos
        # com o contexto do prompt (prompt + " " + palavra). Na posição
        # da keyword usamos os mesmos ids.
        token_sets = None
        if entity_sets is not None:
            token_sets = build_token_sets(self.tokenizer, prompt, entity_sets)

        results = {
            "tokens":        tokens,
            "n_layers":      n_layers,
            "keyword_pos":   keyword_pos,
            "last_token":    [],
            "keyword_token": [] if keyword_pos is not None else None,
            "entity_tokens": token_sets,
        }

        # Iterar sobre camadas transformer (índices 1..N)
        for layer_idx in range(1, len(hidden_states)):
            # [seq_len, hidden_size] — remover dimensão de batch
            layer_hidden = hidden_states[layer_idx][0]
            is_last      = layer_idx == n_layers   # já normalizada

            # ── Último token ──
            last_hidden = layer_hidden[last_pos]       # [hidden_size]
            last_probs  = self._apply_logit_lens(last_hidden, is_last)

            results["last_token"].append(
                self._layer_entry(layer_idx, last_probs, token_sets)
            )

            # ── Token da keyword ──
            if keyword_pos is not None:
                kw_hidden = layer_hidden[keyword_pos]  # [hidden_size]
                kw_probs  = self._apply_logit_lens(kw_hidden, is_last)

                results["keyword_token"].append(
                    self._layer_entry(layer_idx, kw_probs, token_sets)
                )

        return results

    def _layer_entry(
        self,
        layer_idx: int,
        probs: torch.Tensor,
        token_sets: Optional[Dict]
    ) -> Dict:
        """Dict de uma camada: top-k (qualitativo) + P exata por conjunto."""
        top_probs, top_idx = torch.topk(probs, self.top_k)
        top_ids = top_idx.tolist()
        entry = {
            "layer":      layer_idx,
            "top_tokens": [self.tokenizer.decode([i]) for i in top_ids],
            "top_ids":    top_ids,
            "top_probs":  top_probs.cpu().numpy().tolist(),
        }
        if token_sets is not None:
            entry["entity_probs"] = {
                name: prob_mass(probs, info["ids"])
                for name, info in token_sets["sets"].items()
            }
        return entry

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
