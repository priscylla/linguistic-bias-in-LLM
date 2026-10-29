"""
logit_lens/entity_tokens.py
Conjuntos exatos de token IDs para entidades (nomes ou indicadores).

O Logit Lens lê uma distribuição de próximo token por posição. Para
medir P(entidade) na posição de decisão usamos a probabilidade exata
dos token IDs que iniciam cada nome — nunca comparação de substrings
(" Bell" não pode capturar " bella").

Convenção de tokenização: tokeniza-se `prompt + " " + palavra` e
toma-se o primeiro token depois do prompt. Se a palavra se divide em
vários tokens, conta o primeiro fragmento (marcado como `split`).
Tokens compartilhados entre conjuntos concorrentes do mesmo fato não
contam para nenhum deles.
"""

from typing import Callable, Dict, List, Optional, Sequence

import torch


INDICATOR_TYPES = ("place", "artifact")


def token_set_kind(prompt_type: str) -> str:
    """'indicators' para prompts place/artifact, 'names' para os demais."""
    return "indicators" if prompt_type in INDICATOR_TYPES else "names"


def word_sets(fact, prompt) -> Dict[str, List[str]]:
    """Conjuntos {candidato: palavras} medidos por um prompt."""
    if prompt.type in INDICATOR_TYPES:
        return {c.id: c.indicators_for(prompt.type, prompt.lang)
                for c in fact.candidates if c.indicators_for(prompt.type, prompt.lang)}
    return {c.id: c.names_for(prompt.lang, prompt.slot_case) for c in fact.candidates}


def format_prompt(prompt, tokenizer, chat_formatter: Optional[Callable] = None):
    """
    Texto tokenizado + convenção da lacuna para o formato do prompt.

    Returns:
        (texto, sep, add_special_tokens) ou None se o formato não se
        aplica (QA sem chat template)
    """
    if prompt.format == "qa":
        if chat_formatter is None:
            return None
        return chat_formatter(tokenizer, prompt.text), "", False
    return prompt.text, " ", True


MIN_INDICATOR_CANDIDATES = 2


def valid_candidates(token_sets: Dict) -> List[str]:
    """Candidatos que mantêm pelo menos um token ID (sem compartilhados nem excluídos)."""
    return [cid for cid, s in token_sets["sets"].items() if s["ids"]]


def indicator_cell_ok(token_sets_per_paraphrase: List[Dict]) -> bool:
    """
    Regra da autora (2026-10-29) para prompts place/artifact: a célula
    fato × idioma × tipo só é medida se, em todas as paráfrases, pelo
    menos dois candidatos têm token IDs válidos depois da checagem de
    tokenização. Senão a célula inteira é pulada (e registrada).
    """
    return bool(token_sets_per_paraphrase) and all(
        len(valid_candidates(ts)) >= MIN_INDICATOR_CANDIDATES for ts in token_sets_per_paraphrase)


def encode_ids(tokenizer, text: str, add_special_tokens: bool = True) -> List[int]:
    """
    Tokeniza e devolve lista de ids. Com add_special_tokens=True (padrão)
    é igual ao forward do prompt cru; para texto já formatado com chat
    template use False (o template já contém os tokens especiais).
    """
    if add_special_tokens:
        ids = tokenizer(text)["input_ids"]
    else:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def first_token_after(tokenizer, prompt: str, word: str, sep: str = " ",
                      add_special_tokens: bool = True) -> Dict:
    """
    Primeiro token que o modelo emitiria para `word` logo após `prompt`.

    Returns:
        Dict com:
            token_id:       id do primeiro token da palavra (None se excluída)
            token:          token decodificado
            n_word_tokens:  em quantos tokens a palavra se divide
            split:          True se a palavra tem mais de um token
                            (o id é só um fragmento, ex.: " Me" de Meucci)
            boundary_merge: True se anexar a palavra alterou o último
                            token do próprio prompt — aí não existe token
                            bem definido na posição de decisão e a
                            palavra é excluída
    """
    p_ids    = encode_ids(tokenizer, prompt, add_special_tokens)
    full_ids = encode_ids(tokenizer, prompt + sep + word, add_special_tokens)

    # Maior prefixo comum
    i = 0
    while i < min(len(p_ids), len(full_ids)) and p_ids[i] == full_ids[i]:
        i += 1

    boundary_merge = i < len(p_ids)
    if boundary_merge or i >= len(full_ids):
        return {"word": word, "token_id": None, "token": None,
                "n_word_tokens": len(full_ids) - i, "split": False,
                "boundary_merge": True}

    tid = full_ids[i]
    n   = len(full_ids) - i
    return {"word": word, "token_id": tid, "token": tokenizer.decode([tid]),
            "n_word_tokens": n, "split": n > 1, "boundary_merge": False}


def build_token_sets(
    tokenizer,
    prompt: str,
    sets: Dict[str, Sequence[str]],
    sep: str = " ",
    add_special_tokens: bool = True,
) -> Dict:
    """
    Constrói os conjuntos de token IDs distintos de cada conjunto de
    palavras, para um prompt.

    Args:
        sets: {nome_do_conjunto: [palavras]}, ex.:
              {"local": ["Santos", "Dumont"], "competing": ["Wright"]}

    Returns:
        Dict com:
            sets:           {nome: {"ids": [...], "tokens": [...],
                                    "words": [detalhe por palavra]}}
            shared_dropped: [{"token_id", "token", "sets"}] — ids que
                            aparecem em mais de um conjunto, removidos
                            de todos
            excluded:       [palavras excluídas por boundary_merge]
    """
    details  = {}
    raw_ids  = {}
    excluded = []

    for name, words in sets.items():
        details[name] = []
        raw_ids[name] = []
        for w in words:
            d = first_token_after(tokenizer, prompt, w, sep=sep,
                                  add_special_tokens=add_special_tokens)
            details[name].append(d)
            if d["token_id"] is None:
                excluded.append({"set": name, "word": w})
            elif d["token_id"] not in raw_ids[name]:
                raw_ids[name].append(d["token_id"])

    # Ids em mais de um conjunto não contam para nenhum
    owners = {}
    for name, ids in raw_ids.items():
        for tid in ids:
            owners.setdefault(tid, []).append(name)
    shared = {tid for tid, names in owners.items() if len(names) > 1}

    out_sets = {}
    for name, ids in raw_ids.items():
        kept = [t for t in ids if t not in shared]
        out_sets[name] = {
            "ids":    kept,
            "tokens": [tokenizer.decode([t]) for t in kept],
            "words":  details[name],
        }

    return {
        "sets": out_sets,
        "shared_dropped": [
            {"token_id": t, "token": tokenizer.decode([t]),
             "sets": owners[t]} for t in sorted(shared)
        ],
        "excluded": excluded,
    }


def prob_mass(probs: torch.Tensor, ids: Sequence[int]) -> float:
    """Soma exata de probabilidade sobre ids distintos (0 se vazio)."""
    if not ids:
        return 0.0
    idx = torch.tensor(sorted(set(ids)), device=probs.device)
    return probs[idx].sum().item()
