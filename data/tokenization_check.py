"""
data/tokenization_check.py
Verificação de tokenização de nomes e indicadores (Fase 1).

Para cada tokenizer, fato, idioma, tipo de prompt e paráfrase, mostra
como cada nome (prompts F1–F3, QA) e cada indicador (prompts place,
artifact) tokeniza logo após o prompt, com a convenção do pipeline:
tokeniza-se `prompt + " " + palavra` e toma-se o primeiro token após o
prompt (no formato QA: chat template + palavra, sem espaço).

Sinaliza:
  split    a palavra se divide e o primeiro token é só um fragmento
           (o fragmento é o que conta; não pode ser compartilhado)
  shared   primeiro token igual entre candidatos do mesmo fato
           (não conta para nenhum; ex.: um "14-bis" compartilhado cai
           e fica "Demoiselle")
  common   o primeiro token é uma palavra comum no idioma (wordfreq,
           Zipf >= limiar do config); continua no conjunto, a Fase 2
           compara com uma linha de base neutra
  excluded anexar a palavra mudou o último token do prompt (sem token
           bem definido na posição de decisão)
"""

from collections import Counter
from typing import Callable, Dict, List, Optional

from logit_lens.entity_tokens import build_token_sets


def _zipf(word: str, lang: str) -> float:
    from wordfreq import zipf_frequency
    return zipf_frequency(word, lang)


def token_word(token: str) -> str:
    """Texto do token sem espaço/marcador inicial (▁, Ġ), para o wordfreq."""
    return token.replace("▁", " ").replace("Ġ", " ").strip()


def word_sets(fact, prompt) -> Dict[str, List[str]]:
    """Conjuntos {candidato: palavras} medidos por um prompt."""
    if prompt.type in ("place", "artifact"):
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


def check_tokenizer(
    model_key: str,
    tokenizer,
    facts: Dict,
    prompts,
    zipf_threshold: float,
    chat_formatter: Optional[Callable] = None,
    zipf: Callable[[str, str], float] = _zipf,
) -> Dict:
    """
    Roda a verificação para um tokenizer.

    Returns:
        Dict com:
            rows:    uma linha por prompt × candidato × palavra
            summary: contagens de cada sinalização e lista de problemas
                     únicos (fato, idioma, candidato, palavra, token, flag)
    """
    rows   = []
    zcache = {}

    for p in prompts:
        fact = facts[p.fact]
        fmt  = format_prompt(p, tokenizer, chat_formatter)
        if fmt is None:
            continue
        text, sep, add_special = fmt
        sets = word_sets(fact, p)
        out  = build_token_sets(tokenizer, text, sets, sep=sep,
                                add_special_tokens=add_special)
        shared = {d["token_id"]: d["sets"] for d in out["shared_dropped"]}

        for cid, info in out["sets"].items():
            for d in info["words"]:
                tok = d["token"] or ""
                key = (token_word(tok).casefold(), p.lang)
                if key not in zcache:
                    zcache[key] = zipf(*key) if key[0] else 0.0
                z = zcache[key]
                rows.append({
                    "model": model_key, "fact": p.fact, "lang": p.lang,
                    "prompt_type": p.type, "paraphrase": p.paraphrase,
                    "format": p.format, "slot_case": p.slot_case,
                    "candidate": cid, "word": d["word"],
                    "token": tok, "token_id": d["token_id"],
                    "n_word_tokens": d["n_word_tokens"],
                    "split": d["split"],
                    "shared_with": ",".join(sorted(set(shared.get(d["token_id"], [])) - {cid})),
                    "zipf": round(z, 2),
                    "common": z >= zipf_threshold,
                    "excluded": d["boundary_merge"],
                })

    return {"rows": rows, "summary": summarize(rows)}


def summarize(rows: List[Dict]) -> Dict:
    """Contagens e problemas únicos por (fato, idioma, candidato, palavra, token)."""
    issues = {}
    counts = Counter()
    for r in rows:
        flags = [f for f in ("split", "common", "excluded") if r[f]]
        if r["shared_with"]:
            flags.append("shared")
        for f in flags:
            counts[f] += 1
            key = (r["fact"], r["lang"], r["candidate"], r["word"], r["token"], f)
            issues.setdefault(key, {
                "fact": r["fact"], "lang": r["lang"], "candidate": r["candidate"],
                "word": r["word"], "token": r["token"], "flag": f,
                "shared_with": r["shared_with"], "zipf": r["zipf"],
                "prompts": 0})
            issues[key]["prompts"] += 1
    return {
        "n_rows": len(rows),
        "counts": dict(counts),
        "issues": sorted(issues.values(),
                         key=lambda d: (d["flag"], d["fact"], d["lang"], d["candidate"])),
    }
