"""
devtools/tiny.py
Modelos minúsculos e aleatórios (construídos a partir de configs) e
tokenizers de brinquedo, para os testes e para run_pipeline_dry.py.
Nada é baixado.
"""

import re

import torch
from transformers import (
    Gemma2Config, Gemma2ForCausalLM,
    LlamaConfig, LlamaForCausalLM,
    MistralConfig, MistralForCausalLM,
    Qwen2Config, Qwen2ForCausalLM,
)

VOCAB = 128

# Dimensões comuns a todas as arquiteturas
_BASE = dict(
    vocab_size=VOCAB,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
)

ARCHS = {
    "llama":   (LlamaConfig,   LlamaForCausalLM,   {}),
    "mistral": (MistralConfig, MistralForCausalLM, {}),
    "qwen2":   (Qwen2Config,   Qwen2ForCausalLM,   {}),
    # Softcap baixo para que o tanh realmente altere os logits
    "gemma2":  (Gemma2Config,  Gemma2ForCausalLM,
                {"head_dim": 8, "final_logit_softcapping": 0.5,
                 "attn_logit_softcapping": 50.0, "sliding_window": 16}),
}


def build_tiny_model(arch: str, dtype: torch.dtype = torch.float32, vocab: int = VOCAB):
    """Cria um modelo aleatório minúsculo da arquitetura pedida."""
    cfg_cls, model_cls, extra = ARCHS[arch]
    torch.manual_seed(0)
    config = cfg_cls(**{**_BASE, "vocab_size": vocab}, **extra)
    model = model_cls(config).to(dtype).eval()
    # Pesos aleatórios com norm de ganho != 1, para que aplicar a norm
    # duas vezes faça diferença mensurável
    with torch.no_grad():
        model.model.norm.weight.uniform_(0.5, 1.5)
    return model


class FakeTokenizer:
    """Tokenizer mínimo: um id por caractere (mod VOCAB)."""

    pad_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[1] + [ord(c) % VOCAB for c in text]])
        return _Batch(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class _Batch(dict):
    def to(self, device):
        return _Batch({k: v.to(device) for k, v in self.items()})


class WordTokenizer:
    """
    Tokenizer de palavras com convenção de espaço inicial (estilo BPE):
    " Bell" e " bella" são tokens distintos. Palavras em `splits` se
    dividem em fragmentos (ex.: " Meucci" -> " Me" + "ucci").
    Cada token novo recebe o próximo id livre (ids únicos).
    """

    pad_token_id = 0
    BOS = 1

    def __init__(self, splits=None, merges=None):
        self.splits = splits or {}
        # merges: {("tok_a", "tok_b"): "tok_ab"} — junta dois tokens
        # na fronteira (simula boundary merge)
        self.merges = merges or {}
        self.vocab  = {"<s>": self.BOS}
        self.inv    = {self.BOS: "<s>"}

    def _id(self, tok):
        if tok not in self.vocab:
            i = len(self.vocab) + 1
            self.vocab[tok] = i
            self.inv[i] = tok
        return self.vocab[tok]

    def _pieces(self, text):
        pieces = []
        # Marcadores "<...>" (tokens especiais do chat template) ficam isolados
        for chunk in re.split(r"(<[^>]+>)", text):
            if chunk.startswith("<") and chunk.endswith(">"):
                pieces.append(chunk)
                continue
            for w in re.findall(r"\s?\S+", chunk):
                pieces.extend(self.splits.get(w, [w]))
        changed = True
        while changed:
            changed = False
            for i in range(len(pieces) - 1):
                m = self.merges.get((pieces[i], pieces[i + 1]))
                if m:
                    pieces[i:i + 2] = [m]
                    changed = True
                    break
        return pieces

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = ([self.BOS] if add_special_tokens else []) + \
              [self._id(t) for t in self._pieces(text)]
        if return_tensors == "pt":
            t = torch.tensor([ids])
            return _Batch(input_ids=t, attention_mask=torch.ones_like(t))
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        return "".join(self.inv.get(i, "?") for i in ids)
