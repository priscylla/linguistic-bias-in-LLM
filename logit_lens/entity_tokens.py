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

from typing import Dict, List, Sequence

import torch


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
