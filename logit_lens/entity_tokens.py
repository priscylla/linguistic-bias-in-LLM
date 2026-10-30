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
        # Separador da resposta tirado do chat template (models/chat_template.py);
        # formatadores simples (testes) não têm o atributo e usam ""
        return chat_formatter(tokenizer, prompt.text), getattr(chat_formatter, "answer_sep", ""), False
    return prompt.text, " ", True


MIN_INDICATOR_CANDIDATES = 2


def valid_candidates(token_sets: Dict, prefix_cases: Optional[List[Dict]] = None) -> List[str]:
    """
    Candidatos medidos: com pelo menos um token ID próprio ou um token
    seguinte resolvido por prefixo forçado (resolve_shared_prefixes).
    """
    out = {cid for cid, s in token_sets["sets"].items() if s["ids"]}
    for case in prefix_cases or []:
        out |= {cid for cid, ids in case["by_candidate"].items() if ids}
    return [cid for cid in token_sets["sets"] if cid in out]


def indicator_cell_ok(per_paraphrase: List) -> bool:
    """
    Regra da autora (2026-10-29) para prompts place/artifact: a célula
    fato × idioma × tipo só é medida se, em todas as paráfrases, pelo
    menos dois candidatos são medidos (token próprio ou prefixo forçado).
    Senão a célula inteira é pulada (e registrada).

    per_paraphrase: [(token_sets, prefix_cases)] ou [token_sets].
    """
    def n(item):
        ts, cases = item if isinstance(item, tuple) else (item, None)
        return len(valid_candidates(ts, cases))
    return bool(per_paraphrase) and all(n(x) >= MIN_INDICATOR_CANDIDATES for x in per_paraphrase)


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


MAX_PREFIX_DEPTH = 3


def resolve_shared_prefixes(tokenizer, prompt: str, token_sets: Dict, sep: str = " ",
                            add_special_tokens: bool = True,
                            max_depth: int = MAX_PREFIX_DEPTH) -> List[Dict]:
    """
    Primeiro token compartilhado entre candidatos (decisão da autora,
    2026-10-29): em vez de não contar para ninguém, o prefixo é forçado
    e cada candidato é medido pelo token seguinte,

        P(e) += P(s) × P(t | prompt + s)

    aplicado de novo enquanto os candidatos ainda compartilham tokens
    (até `max_depth` tokens forçados): P(s) × P(t1 | s) × P(t2 | s t1)…
    Ex.: depois de um artigo elidido, "'Mendeleev" e "'Meyer" compartilham
    "'" e depois "M"; o caminho segue até "end" / "ey".

    Para cada s em token_sets["shared_dropped"]:
        by_candidate  {candidato: [caminhos de token IDs, começando em s]}
        words         [{candidate, word, path, next_id, next_token, depth,
                        status}]; status: combined | no_next_token (a
                      palavra acaba dentro do prefixo compartilhado) |
                      next_shared (ainda compartilhado após max_depth)
    """
    p_ids = encode_ids(tokenizer, prompt, add_special_tokens)
    i = len(p_ids)
    cases = []
    for shared in token_sets["shared_dropped"]:
        sid = shared["token_id"]
        words = []
        for cid, info in token_sets["sets"].items():
            for d in info["words"]:
                if d["token_id"] != sid:
                    continue
                full = encode_ids(tokenizer, prompt + sep + d["word"], add_special_tokens)
                rest = full[i:] if full[:i] == p_ids and len(full) > i and full[i] == sid else []
                words.append({"candidate": cid, "word": d["word"], "seq": rest})

        def split(group, depth):
            """Separa o grupo pelo token na posição `depth` (depth 1 = após s)."""
            by_tok = {}
            for w in group:
                if len(w["seq"]) <= depth:
                    w.update(status="no_next_token", path=w["seq"], depth=depth - 1)
                    continue
                by_tok.setdefault(w["seq"][depth], []).append(w)
            for tok_id, ws in by_tok.items():
                if len({w["candidate"] for w in ws}) == 1:
                    for w in ws:
                        w.update(status="combined", path=w["seq"][:depth + 1], depth=depth)
                elif depth < max_depth:
                    split(ws, depth + 1)
                else:
                    for w in ws:
                        w.update(status="next_shared", path=w["seq"][:depth + 1], depth=depth)

        split(words, 1)
        by_cand = {}
        for w in words:
            w.pop("seq")
            path = w["path"] = list(w["path"])
            w["next_id"] = path[-1] if w["status"] == "combined" else None
            w["next_token"] = (tokenizer.decode(path[1:]) if len(path) > 1 else None)
            if w["status"] == "combined":
                paths = by_cand.setdefault(w["candidate"], [])
                if w["path"] not in paths:
                    paths.append(w["path"])
        cases.append({"token_id": sid, "token": tokenizer.decode([sid]),
                      "candidates": sorted({w["candidate"] for w in words}),
                      "by_candidate": by_cand, "words": words})
    return cases
