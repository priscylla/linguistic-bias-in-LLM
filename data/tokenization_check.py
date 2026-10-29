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

# word_sets e format_prompt são compartilhados com o lens (Fase 2)
from logit_lens.entity_tokens import (  # noqa: F401
    INDICATOR_TYPES, MIN_INDICATOR_CANDIDATES, build_token_sets, format_prompt,
    resolve_shared_prefixes, word_sets,
)


def _zipf(word: str, lang: str) -> float:
    from wordfreq import zipf_frequency
    return zipf_frequency(word, lang)


def token_word(token: str) -> str:
    """Texto do token sem espaço/marcador inicial (▁, Ġ), para o wordfreq."""
    return token.replace("▁", " ").replace("Ġ", " ").strip()


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
        # Primeiro token compartilhado: resolvido por prefixo forçado?
        res = {(w["candidate"], w["word"]): w
               for case in resolve_shared_prefixes(tokenizer, text, out, sep=sep,
                                                   add_special_tokens=add_special)
               for w in case["words"]}

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
                    "prefix_status": res.get((cid, d["word"]), {}).get("status", ""),
                    "next_token": res.get((cid, d["word"]), {}).get("next_token") or "",
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


def _measured(r) -> bool:
    """A palavra conta: token próprio, ou primeiro token compartilhado resolvido."""
    return not r["excluded"] and (not r["shared_with"] or r.get("prefix_status") == "combined")


def dropped_words(rows: List[Dict]) -> List[Dict]:
    """Palavras que não contam (fusão na fronteira, ou prefixo compartilhado não resolvido)."""
    seen, out = set(), []
    for r in rows:
        if _measured(r):
            continue
        reason = "excluded (boundary merge)" if r["excluded"] else (
            f"shared with {r['shared_with']}, {r.get('prefix_status') or 'unresolved'}")
        key = (r["model"], r["fact"], r["lang"], r["prompt_type"], r["candidate"], r["word"], reason)
        if key in seen:
            continue
        seen.add(key)
        out.append({"model": r["model"], "fact": r["fact"], "lang": r["lang"],
                    "prompt_type": r["prompt_type"], "candidate": r["candidate"],
                    "word": r["word"], "token": r["token"], "reason": reason})
    return out


def prefix_forced(rows: List[Dict]) -> List[Dict]:
    """Palavras medidas por prefixo forçado: P(prefixo) × P(próximo | prefixo)."""
    seen, out = set(), []
    for r in rows:
        if r.get("prefix_status") != "combined":
            continue
        key = (r["model"], r["fact"], r["lang"], r["prompt_type"], r["candidate"], r["word"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"model": r["model"], "fact": r["fact"], "lang": r["lang"],
                    "prompt_type": r["prompt_type"], "candidate": r["candidate"],
                    "word": r["word"], "shared_token": r["token"], "shared_with": r["shared_with"],
                    "next_token": r["next_token"]})
    return out


def indicator_cells(rows: List[Dict]) -> List[Dict]:
    """
    Regra dos dois candidatos (a mesma de run_lens.py): célula fato ×
    idioma × tipo de indicador mantida só se, em toda paráfrase, pelo
    menos dois candidatos têm um token válido.
    """
    per_prompt = {}
    for r in rows:
        if r["prompt_type"] not in INDICATOR_TYPES:
            continue
        key = (r["model"], r["fact"], r["lang"], r["prompt_type"], r["paraphrase"])
        per_prompt.setdefault(key, set())
        if _measured(r):
            per_prompt[key].add(r["candidate"])
    cells = {}
    for (m, f, l, t, par), valid in per_prompt.items():
        cells.setdefault((m, f, l, t), []).append((par, sorted(valid)))
    out = []
    for (m, f, l, t), pars in sorted(cells.items()):
        n = min(len(v) for _, v in pars)
        out.append({"model": m, "fact": f, "lang": l, "prompt_type": t,
                    "status": "kept" if n >= MIN_INDICATOR_CANDIDATES else "skipped",
                    "min_valid_candidates": n,
                    "valid_candidates": "; ".join(f"P{p}: {','.join(v) or '—'}" for p, v in sorted(pars))})
    return out


def write_summary(tok_dir, out_path, model_order, facts, prompt_specs) -> str:
    """
    reports/tokenization_summary.md: palavras que não contam por fato ×
    idioma × formato (com os modelos), células de indicador puladas e os
    tipos de prompt que restam por fato. Lê os CSVs de verify_setup.
    """
    import glob
    from pathlib import Path

    import pandas as pd

    tok_dir = Path(tok_dir)
    def _load(pattern):
        parts = []
        for p in sorted(glob.glob(str(tok_dir / pattern))):
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                continue
            if len(df):
                parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    D = _load("*_dropped.csv")
    P = _load("*_prefix_forced.csv")
    cells = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(tok_dir / "*_indicator_cells.csv")))],
                      ignore_index=True)
    key = {m: i for i, m in enumerate(model_order)}
    sort_models = lambda s: ", ".join(sorted(set(s), key=lambda m: key.get(m, 99)))  # noqa: E731
    out = ["# Tokenization check summary", "",
           f"Generated by `verify_setup.py` from `{tok_dir}/` ({len(set(cells['model']))} tokenizers, "
           "pinned commits). Full per-prompt tables are in that folder (not versioned).", "",
           "When candidates of the same fact share a first token, that token is forced and each candidate "
           "is measured by its next token: P(prefix) × P(next | prefix) (author's decision, 2026-10-29). "
           "A word still doesn't count when it is the shared token itself, when its next token collides "
           "again, or when it merges with the prompt's last token. QA prompts take the answer's first "
           "token without a leading space.", "",
           "## Resolved by prefix forcing", "",
           "| Fact | Lang | Set | Format | Candidate | Words | Shared token → next | Models |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    if len(P):
        P["fmt"] = P["prompt_type"].map(lambda t: "QA" if t == "QA" else "completion")
        P["set"] = P["prompt_type"].map(lambda t: t if t in INDICATOR_TYPES else "names")
        clean = lambda x: str(x).replace("▁", " ").replace("Ġ", " ").strip()  # noqa: E731
        P["pair"] = [f"{clean(a)} → {clean(b)}" for a, b in zip(P["shared_token"], P["next_token"])]
        for k, g in P.groupby(["fact", "lang", "set", "fmt", "candidate"]):
            out.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {', '.join(sorted(set(g['word'])))} | "
                       f"`{'; '.join(sorted(set(g['pair'])))}` | {sort_models(g['model'])} |")
    else:
        out.append("| — | | | | | | | |")
    out += ["", "## Still dropped", "",
           "| Fact | Lang | Set | Format | Candidate | Words | First token | Models |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    if not len(D):
        out.append("| — | | | | | | | |")
    if len(D):
        D["fmt"] = D["prompt_type"].map(lambda t: "QA" if t == "QA" else "completion")
        D["set"] = D["prompt_type"].map(lambda t: t if t in INDICATOR_TYPES else "names")
        D["tok"] = D["token"].astype(str).str.replace("▁", "").str.replace("Ġ", "").str.strip()
        for k, g in D.groupby(["fact", "lang", "set", "fmt", "candidate"]):
            out.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {', '.join(sorted(set(g['word'])))} | "
                       f"`{', '.join(sorted(set(g['tok'])))}` | {sort_models(g['model'])} |")
    sk = cells[cells["status"] == "skipped"]
    out += ["", "## Indicator cells skipped (fewer than two measured candidates)", ""]
    if len(sk):
        out += ["| Fact | Lang | Type | Models | Valid candidates |", "| --- | --- | --- | --- | --- |"]
        for k, g in sk.groupby(["fact", "lang", "prompt_type"]):
            out.append(f"| {k[0]} | {k[1]} | {k[2]} | {sort_models(g['model'])} | "
                       f"{g['valid_candidates'].iloc[0]} |")
    else:
        out.append("None.")
    out += ["", "## Prompt types per fact", "",
            "| Fact | Kind | Prompt types | Skipped for some tokenizers |", "| --- | --- | --- | --- |"]
    for f, spec in facts.items():
        types = sorted({p.type for p in prompt_specs if p.fact == f},
                       key=lambda t: ["F1", "F2", "F3", "QA", "place", "artifact"].index(t))
        s = sk[sk["fact"] == f]
        note = "; ".join(f"{t} in {', '.join(sorted(set(g['lang'])))} ({sort_models(g['model'])})"
                         for t, g in s.groupby("prompt_type")) or "—"
        out.append(f"| {f} | {spec.kind} | {', '.join(types)} | {note} |")
    text = "\n".join(out) + "\n"
    Path(out_path).write_text(text, encoding="utf-8")
    return text
