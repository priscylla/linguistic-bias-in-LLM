"""
logit_lens/measure.py
Medição por camada na posição de decisão (Fase 2).

Para um forward pass, com qualquer Lens (LogitLens agora, TunedLens
na Fase 4), devolve por camada:

    1. probabilidade exata de cada candidato: massa sobre o conjunto
       de token IDs distintos dos seus nomes (prompts F1–F3, QA) ou dos
       seus indicadores (place, artifact), na distribuição completa;
    2. KL(lens_k ‖ distribuição final do modelo), diagnóstico de
       qualidade do lens;
    3. top-k tokens (top-20), só para inspeção qualitativa.

Linha de base neutra: prompts neutros com o mesmo formato de lacuna
(config/baseline_prompts.yaml) medem os mesmos tokens de candidatos de
todos os fatos, num único forward por prompt neutro, para separar
"o modelo pensa em Santos-Dumont" de "' Santos' é só um token
frequente em português".

Formatos:
    completion: prompt cru (com BOS); posição de decisão = último
                token do prompt; palavra anexada com espaço
    qa:         chat template sem system prompt; posição de decisão =
                a que prevê o primeiro token do assistente; palavra
                anexada sem espaço e sem BOS extra

Saída em linhas "tidy" (uma por camada × candidato), prontas para a
tabela da Fase 2.
"""

from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from logit_lens.entity_tokens import (
    build_token_sets, encode_ids, format_prompt, prob_mass, token_set_kind, word_sets,
)
from logit_lens.lens import Lens, LogitLens


def kl_to_final(layer_logits: torch.Tensor, final_logits: torch.Tensor) -> torch.Tensor:
    """
    KL(P_k ‖ P_final) por camada, em nats:

        Σ_v P_k(v) · (log P_k(v) − log P_final(v))

    Args:
        layer_logits: [n_layers, vocab] (float32)
        final_logits: [vocab], logits do próprio modelo

    Returns:
        tensor [n_layers]
    """
    logp = F.log_softmax(layer_logits.float(), dim=-1)
    logq = F.log_softmax(final_logits.float().to(logp.device), dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1).clamp_min(0.0)


class LensRunner:
    """
    Roda um forward por prompt e mede as quantidades por camada.

    Args:
        model, tokenizer: modelo e tokenizer HF
        lens:             Lens (padrão: LogitLens do modelo)
        top_k:            tokens guardados por camada (só inspeção)
        chat_formatter:   f(tokenizer, texto) -> texto com chat template
                          (obrigatório para prompts QA)
    """

    def __init__(self, model, tokenizer, lens: Optional[Lens] = None,
                 top_k: int = 20, chat_formatter: Optional[Callable] = None):
        self.model          = model
        self.tokenizer      = tokenizer
        self.lens           = lens or LogitLens(model)
        self.top_k          = top_k
        self.chat_formatter = chat_formatter
        self.device         = next(model.parameters()).device

    # ── Forward ──

    def forward(self, text: str, add_special_tokens: bool = True) -> Dict:
        """
        Forward do texto e distribuições do lens na última posição.

        Returns:
            Dict com input_ids, decision_pos, n_layers,
            probs [n_layers, vocab] e kl [n_layers]
        """
        ids = encode_ids(self.tokenizer, text, add_special_tokens)
        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=torch.ones_like(input_ids),
                                 output_hidden_states=True)
        pos      = len(ids) - 1
        logits   = self.lens.layer_logits(outputs.hidden_states, pos)
        # Em todo forward: a última camada tem que ser o próprio modelo
        self.lens.check_consistency(outputs, pos, last_logits=logits[-1])
        return {
            "input_ids":    ids,
            "decision_pos": pos,
            "n_layers":     logits.shape[0],
            "probs":        F.softmax(logits, dim=-1),
            "kl":           kl_to_final(logits, outputs.logits[0, pos]),
        }

    def prepare(self, prompt):
        """(texto, sep, add_special_tokens) do prompt no seu formato."""
        fmt = format_prompt(prompt, self.tokenizer, self.chat_formatter)
        if fmt is None:
            raise ValueError(
                f"Prompt {prompt.type} ({prompt.lang}) no formato QA exige "
                f"chat template (chat_formatter); modelos base só rodam completion.")
        return fmt

    # ── Tabelas ──

    def layer_rows(self, fw: Dict) -> List[Dict]:
        """Uma linha por camada: KL até a final e top-k (qualitativo)."""
        n    = fw["n_layers"]
        k    = min(self.top_k, fw["probs"].shape[-1])
        tp, ti = torch.topk(fw["probs"], k, dim=-1)
        rows = []
        for i in range(n):
            ids = ti[i].tolist()
            rows.append({
                "layer":       i + 1,
                "depth":       (i + 1) / n,
                "kl_to_final": float(fw["kl"][i]),
                "top_ids":     ids,
                "top_tokens":  [self.tokenizer.decode([t]) for t in ids],
                "top_probs":   tp[i].tolist(),
            })
        return rows

    @staticmethod
    def set_rows(fw: Dict, token_sets: Dict, kind: str, extra: Optional[Dict] = None) -> List[Dict]:
        """
        Uma linha por camada × candidato: P exata do conjunto de ids
        distintos (0 se o conjunto ficou vazio; n_ids mostra isso).
        """
        n, extra = fw["n_layers"], extra or {}
        rows = []
        for cid, info in token_sets["sets"].items():
            for i in range(n):
                rows.append({**extra, "candidate": cid, "token_set": kind,
                             "layer": i + 1, "depth": (i + 1) / n,
                             "prob": prob_mass(fw["probs"][i], info["ids"]),
                             "n_ids": len(info["ids"])})
        return rows

    # ── Prompts do experimento ──

    def measure(self, prompt, fact) -> Dict:
        """
        Mede um prompt de um fato: nomes (F1–F3, QA) ou indicadores
        (place, artifact) de cada candidato.

        Returns:
            Dict com meta, n_layers, decision_pos, n_input_tokens,
            layers (KL + top-k), probs (linhas tidy) e token_sets
            (auditoria: ids, palavras, compartilhados, excluídos)
        """
        text, sep, add_special = self.prepare(prompt)
        fw   = self.forward(text, add_special)
        kind = token_set_kind(prompt.type)
        sets = build_token_sets(self.tokenizer, text, word_sets(fact, prompt),
                                sep=sep, add_special_tokens=add_special)
        return {
            "meta":           _meta(prompt, fact=fact.key, lens=self.lens.name),
            "n_layers":       fw["n_layers"],
            "decision_pos":   fw["decision_pos"],
            "n_input_tokens": len(fw["input_ids"]),
            "layers":         self.layer_rows(fw),
            "probs":          self.set_rows(fw, sets, kind),
            "token_sets":     sets,
        }

    # ── Linha de base neutra ──

    def measure_baseline(self, neutral, facts: Dict) -> Dict:
        """
        Um prompt neutro, um forward: P por camada dos tokens de todos
        os candidatos de todos os fatos, montados como no prompt real
        do mesmo tipo (mesmo caso gramatical em ru, mesma convenção de
        lacuna, compartilhados dentro do fato descartados).

        Fatos sem palavras para o tipo (ex.: sem indicador 'artifact')
        ficam de fora.

        Returns:
            Dict com meta, n_layers, layers (KL + top-k), probs (linhas
            tidy com coluna 'fact') e token_sets por fato
        """
        text, sep, add_special = self.prepare(neutral)
        fw   = self.forward(text, add_special)
        kind = token_set_kind(neutral.type)
        rows, audit = [], {}
        for key, fact in facts.items():
            words = word_sets(fact, neutral)
            if not words:
                continue
            sets = build_token_sets(self.tokenizer, text, words,
                                    sep=sep, add_special_tokens=add_special)
            audit[key] = sets
            rows += self.set_rows(fw, sets, kind, extra={"fact": key})
        return {
            "meta":           _meta(neutral, fact=None, lens=self.lens.name),
            "n_layers":       fw["n_layers"],
            "decision_pos":   fw["decision_pos"],
            "n_input_tokens": len(fw["input_ids"]),
            "layers":         self.layer_rows(fw),
            "probs":          rows,
            "token_sets":     audit,
        }


def _meta(prompt, fact, lens: str) -> Dict:
    return {"fact": fact, "lang": prompt.lang, "format": prompt.format,
            "prompt_type": prompt.type, "paraphrase": prompt.paraphrase,
            "slot_case": prompt.slot_case, "text": prompt.text, "lens": lens}
