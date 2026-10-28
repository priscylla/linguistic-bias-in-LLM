"""
behavior/generation.py
Geração das respostas do estudo comportamental (Fase 5).

Por prompt QA (chat template, sem system prompt): 1 resposta greedy e
N amostras a temperatura T, com semente fixa por prompt.

Todos os parâmetros de decodificação são explícitos: no transformers
5.7, campos None da GenerationConfig passada são preenchidos com o
generation_config do modelo (GenerationMixin._prepare_generation_config),
então um top_p=0.9 do modelo entraria em silêncio. Aqui top_p, top_k,
min_p, typical_p, cortes e penalidades são todos fixados; do modelo só
vêm os ids de fim (eos) e de padding.
"""

import hashlib
from typing import Callable, Dict, List

import torch
from transformers import GenerationConfig

from logit_lens.entity_tokens import encode_ids


def prompt_seed(base: int, model_key: str, fact: str, lang: str, paraphrase: int) -> int:
    """Semente estável por prompt (independe da ordem de execução)."""
    h = hashlib.sha256(f"{base}|{model_key}|{fact}|{lang}|{paraphrase}".encode()).hexdigest()
    return int(h[:8], 16)


def _eos_pad(model, tokenizer):
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, list) else eos
    return eos, pad


# Campos de decodificação que o modelo não pode sobrescrever
NEUTRAL = dict(min_p=0.0, typical_p=1.0, epsilon_cutoff=0.0, eta_cutoff=0.0,
               no_repeat_ngram_size=0, encoder_repetition_penalty=1.0,
               length_penalty=1.0, num_beams=1, num_beam_groups=1, diversity_penalty=0.0,
               renormalize_logits=False, suppress_tokens=None, begin_suppress_tokens=None,
               bad_words_ids=None, forced_bos_token_id=None, forced_eos_token_id=None)


def generation_configs(model, tokenizer, g: Dict):
    """(greedy, amostragem): GenerationConfig com tudo explícito."""
    eos, pad = _eos_pad(model, tokenizer)
    common = dict(max_new_tokens=g["max_new_tokens"], eos_token_id=eos, pad_token_id=pad,
                  repetition_penalty=g["repetition_penalty"], **NEUTRAL)
    greedy = GenerationConfig(do_sample=False, temperature=1.0, top_p=1.0, top_k=0, **common)
    sample = GenerationConfig(do_sample=True, temperature=g["temperature"], top_p=g["top_p"],
                              top_k=g["top_k"], **common)
    return greedy, sample


def effective_config(model, cfg: GenerationConfig) -> Dict:
    """A GenerationConfig depois da mistura com a do modelo (auditoria/testes)."""
    merged, _ = model._prepare_generation_config(cfg)
    keys = ["do_sample", "temperature", "top_p", "top_k", "min_p", "typical_p",
            "repetition_penalty", "no_repeat_ngram_size", "max_new_tokens", "eos_token_id"]
    return {k: getattr(merged, k, None) for k in keys}


class Generator:

    def __init__(self, model, tokenizer, chat_formatter: Callable, g: Dict, seed: int):
        self.model, self.tok, self.chat = model, tokenizer, chat_formatter
        self.g, self.seed = g, seed
        self.greedy_cfg, self.sample_cfg = generation_configs(model, tokenizer, g)
        eos = self.greedy_cfg.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.device = next(model.parameters()).device

    def _decode(self, out, n_in) -> List[Dict]:
        rows = []
        for seq in out:
            new = seq[n_in:].tolist()
            # corta o padding depois do fim
            end = next((i for i, t in enumerate(new) if t in self.eos_ids), None)
            body = new if end is None else new[:end]
            rows.append({"text": self.tok.decode(body, skip_special_tokens=True).strip(),
                         "n_new_tokens": len(body), "finished": end is not None})
        return rows

    def answers(self, prompt, model_key: str) -> List[Dict]:
        """Respostas de um prompt QA: sample_id 0 = greedy, 1..N = amostras."""
        text = self.chat(self.tok, prompt.text)
        ids  = torch.tensor([encode_ids(self.tok, text, add_special_tokens=False)],
                            device=self.device)
        mask = torch.ones_like(ids)
        seed = prompt_seed(self.seed, model_key, prompt.fact, prompt.lang, prompt.paraphrase)
        rows = []
        with torch.no_grad():
            if self.g["greedy"]:
                out = self.model.generate(input_ids=ids, attention_mask=mask,
                                          generation_config=self.greedy_cfg)
                for r in self._decode(out, ids.shape[1]):
                    rows.append({**r, "sample_id": 0, "kind": "greedy", "seed": None})
            if self.g["samples"]:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                out = self.model.generate(input_ids=ids, attention_mask=mask,
                                          generation_config=self.sample_cfg,
                                          num_return_sequences=self.g["samples"])
                for i, r in enumerate(self._decode(out, ids.shape[1]), start=1):
                    rows.append({**r, "sample_id": i, "kind": "sample", "seed": seed})
        keys = {"model": model_key, "fact": prompt.fact, "lang": prompt.lang,
                "prompt_type": prompt.type, "paraphrase": prompt.paraphrase,
                "question": prompt.text}
        return [{**keys, **r} for r in rows]
