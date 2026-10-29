"""
logit_lens/measure.py
Medição por camada na posição de decisão (Fase 2).

Para um forward pass, com qualquer Lens (LogitLens agora, TunedLens
na Fase 4), devolve por camada:

    1. probabilidade exata de cada candidato: massa sobre o conjunto
       de token IDs distintos dos seus nomes (prompts F1–F3, QA) ou dos
       seus indicadores (place, artifact), na distribuição completa.
       Primeiro token compartilhado entre candidatos: forçado, e cada
       candidato soma P(prefixo) × P(próximo | prefixo) (decisão da
       autora, 2026-10-29; casos registrados em prefix_cases);
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
    build_token_sets, encode_ids, format_prompt, prob_mass, resolve_shared_prefixes,
    token_set_kind, word_sets,
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
        """Forward do texto (ver forward_ids)."""
        return self.forward_ids(encode_ids(self.tokenizer, text, add_special_tokens))

    def forward_ids(self, ids: List[int]) -> Dict:
        """
        Forward de uma sequência de ids e distribuições do lens na
        última posição.

        Returns:
            Dict com input_ids, decision_pos, n_layers,
            probs [n_layers, vocab] e kl [n_layers]
        """
        input_ids = torch.tensor([list(ids)], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=torch.ones_like(input_ids),
                                 output_hidden_states=True)
        pos      = len(ids) - 1
        logits   = self.lens.layer_logits(outputs.hidden_states, pos)
        # Em todo forward: a última camada tem que ser o próprio modelo
        self.lens.check_consistency(outputs, pos, last_logits=logits[-1])
        return {
            "input_ids":    list(ids),
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

    def prefix_mass(self, fw: Dict, p_ids: List[int], cases: List[Dict], cache: Dict = None) -> Dict:
        """
        Contribuição do prefixo forçado por candidato e camada:
            Σ_s P_k(s) × P_k(próximo_e | prompt + s)
        Um forward por token compartilhado s (com a mesma checagem da
        última camada). `cache` reaproveita forwards de s entre fatos.

        Returns:
            {candidato: tensor [n_layers]}
        """
        out = {}
        cache = {} if cache is None else cache
        for case in cases:
            if not any(case["by_candidate"].values()):
                continue
            sid = case["token_id"]
            if sid not in cache:
                cache[sid] = self.forward_ids(list(p_ids) + [sid])["probs"]
            nxt = cache[sid]
            p_s = fw["probs"][:, sid]
            for cid, ids in case["by_candidate"].items():
                if ids:
                    add = p_s * nxt[:, sorted(set(ids))].sum(dim=-1)
                    out[cid] = out.get(cid, 0) + add
        return out

    @staticmethod
    def set_rows(fw: Dict, token_sets: Dict, kind: str, extra: Optional[Dict] = None,
                 prefix: Optional[Dict] = None, prefix_ids: Optional[Dict] = None) -> List[Dict]:
        """
        Uma linha por camada × candidato: P exata do conjunto de ids
        distintos (prob_direct) + contribuição dos prefixos forçados
        (prob_prefix). prob = soma. n_ids conta ids próprios e tokens
        seguintes resolvidos (0: candidato não medido).
        """
        n, extra = fw["n_layers"], extra or {}
        prefix, prefix_ids = prefix or {}, prefix_ids or {}
        rows = []
        for cid, info in token_sets["sets"].items():
            px = prefix.get(cid)
            for i in range(n):
                direct = prob_mass(fw["probs"][i], info["ids"])
                pre = float(px[i]) if px is not None else 0.0
                rows.append({**extra, "candidate": cid, "token_set": kind,
                             "layer": i + 1, "depth": (i + 1) / n,
                             "prob": direct + pre, "prob_direct": direct, "prob_prefix": pre,
                             "n_ids": len(info["ids"]) + prefix_ids.get(cid, 0),
                             "n_prefix_ids": prefix_ids.get(cid, 0)})
        return rows

    # ── Prompts do experimento ──

    def measure_words(self, text: str, sep: str, add_special: bool,
                      words: Dict[str, List[str]], kind: str) -> Dict:
        """
        Um forward de `text` e P por camada de cada conjunto de palavras
        (token sets montados no contexto de `text`).
        """
        fw    = self.forward(text, add_special)
        sets  = build_token_sets(self.tokenizer, text, words,
                                 sep=sep, add_special_tokens=add_special)
        cases = resolve_shared_prefixes(self.tokenizer, text, sets, sep=sep,
                                        add_special_tokens=add_special)
        pre   = self.prefix_mass(fw, fw["input_ids"], cases)
        return {
            "n_layers":       fw["n_layers"],
            "decision_pos":   fw["decision_pos"],
            "n_input_tokens": len(fw["input_ids"]),
            "input_ids":      fw["input_ids"],
            "layers":         self.layer_rows(fw),
            "probs":          self.set_rows(fw, sets, kind, prefix=pre,
                                            prefix_ids=_prefix_counts(cases)),
            "token_sets":     sets,
            "prefix_cases":   cases,
        }

    def measure(self, prompt, fact) -> Dict:
        """
        Mede um prompt de um fato: nomes (F1–F3, QA) ou indicadores
        (place, artifact) de cada candidato.

        Returns:
            Dict com meta, n_layers, decision_pos, n_input_tokens,
            input_ids, layers (KL + top-k), probs (linhas tidy) e
            token_sets (auditoria: ids, palavras, compartilhados, excluídos)
        """
        text, sep, add_special = self.prepare(prompt)
        res = self.measure_words(text, sep, add_special, word_sets(fact, prompt),
                                 token_set_kind(prompt.type))
        return {"meta": _meta(prompt, fact=fact.key, lens=self.lens.name), **res}

    # ── Linha de base neutra ──

    def measure_baseline(self, neutral, facts: Dict) -> Dict:
        """
        Um prompt neutro, um forward: P por camada dos tokens de todos
        os candidatos de todos os fatos, montados como no prompt real
        do mesmo tipo (mesma convenção de lacuna; primeiro token
        compartilhado dentro do fato resolvido por prefixo forçado, como
        no prompt real).

        Fatos sem palavras para o tipo (ex.: sem indicador 'artifact')
        ficam de fora.

        Returns:
            Dict com meta, n_layers, layers (KL + top-k), probs (linhas
            tidy com coluna 'fact') e token_sets por fato
        """
        text, sep, add_special = self.prepare(neutral)
        fw   = self.forward(text, add_special)
        kind = token_set_kind(neutral.type)
        rows, audit, prefix_cases, cache = [], {}, {}, {}
        for key, fact in facts.items():
            words = word_sets(fact, neutral)
            if not words:
                continue
            sets = build_token_sets(self.tokenizer, text, words,
                                    sep=sep, add_special_tokens=add_special)
            cases = resolve_shared_prefixes(self.tokenizer, text, sets, sep=sep,
                                            add_special_tokens=add_special)
            pre = self.prefix_mass(fw, fw["input_ids"], cases, cache)
            audit[key] = sets
            if cases:
                prefix_cases[key] = cases
            rows += self.set_rows(fw, sets, kind, extra={"fact": key}, prefix=pre,
                                  prefix_ids=_prefix_counts(cases))
        return {
            "meta":           _meta(neutral, fact=None, lens=self.lens.name),
            "n_layers":       fw["n_layers"],
            "decision_pos":   fw["decision_pos"],
            "n_input_tokens": len(fw["input_ids"]),
            "layers":         self.layer_rows(fw),
            "probs":          rows,
            "token_sets":     audit,
            "prefix_cases":   prefix_cases,
        }


def _prefix_counts(cases: List[Dict]) -> Dict[str, int]:
    out = {}
    for case in cases:
        for cid, ids in case["by_candidate"].items():
            out[cid] = out.get(cid, 0) + len(ids)
    return out


def _meta(prompt, fact, lens: str) -> Dict:
    return {"fact": fact, "lang": prompt.lang, "format": prompt.format,
            "prompt_type": prompt.type, "paraphrase": prompt.paraphrase,
            "slot_case": prompt.slot_case, "text": prompt.text, "lens": lens}
