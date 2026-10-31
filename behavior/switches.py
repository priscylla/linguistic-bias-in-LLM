"""
behavior/switches.py
Troca de escrita e de idioma nas respostas (descritivo; não entra em
nenhuma métrica).

    escrita   a resposta tem um trecho (≥ 2 caracteres seguidos) numa
              escrita não latina: Han (chinês), kana, hangul, cirílico,
              árabe, hebraico, devanágari, tailandês. Grego fica de fora
              (letras gregas aparecem em fórmulas).
    idioma    o idioma dominante da resposta (pt, en, de, it, fr) é outro
              que o do prompt. Idioma dominante = o que tem mais palavras
              funcionais típicas na resposta, com pelo menos MIN_HITS
              ocorrências e MARGIN vezes as do segundo; senão
              indeterminado (respostas curtas ou misturadas não contam
              como troca).
    parcial   alguma frase da resposta (≥ SENT_MIN_HITS palavras
              funcionais) tem idioma dominante diferente do prompt, mesmo
              que a resposta como um todo não tenha.
"""

import re
from collections import Counter
from typing import Dict, Optional

SCRIPTS = {
    "Han": r"㐀-䶿一-鿿豈-﫿",
    "Kana": r"぀-ヿ",
    "Hangul": r"가-힯ᄀ-ᇿ",
    "Cyrillic": r"Ѐ-ӿ",
    "Arabic": r"؀-ۿ",
    "Hebrew": r"֐-׿",
    "Devanagari": r"ऀ-ॿ",
    "Thai": r"฀-๿",
}
_SCRIPT_RE = {k: re.compile(f"[{v}]{{2,}}") for k, v in SCRIPTS.items()}

# Palavras funcionais que quase só aparecem num dos cinco idiomas (fora as
# ambíguas entre eles: il, le, sua, das, la, a, e, o, da, des)
FUNCTION_WORDS = {
    "pt": "não uma foi pelo pela também são dos ao aos em seu seus ele ela foram "
          "está isso muito mais pelos pelas inventado considerado".split(),
    "en": "the and was is of by which that with his has were this from who invented considered "
          "first also".split(),
    "de": "der die das und ist wurde von mit nicht ein eine auch den dem erfunden gilt als "
          "sich zu im".split(),
    "it": "della è stato che di del delle gli una nel sono degli anche inventato considerato "
          "dalla alla".split(),
    "fr": "les est été et par qui dans une du inventé considéré aussi au aux sont pour "
          "avec".split(),
}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
MIN_HITS, MARGIN = 4, 1.5
SENT_MIN_HITS = 3
_SENT = re.compile(r"(?<=[.!?:;])\s+|\n+")


def scripts(text: str) -> Dict[str, int]:
    """{escrita: número de trechos} das escritas não latinas presentes."""
    out = {}
    for k, rx in _SCRIPT_RE.items():
        n = len(rx.findall(text or ""))
        if n:
            out[k] = n
    return out


def dominant_language(text: str, min_hits: int = MIN_HITS) -> Optional[str]:
    """pt / en / de / it / fr, ou None se indeterminado."""
    words = Counter(w.lower() for w in _WORD.findall(text or ""))
    score = {l: sum(words[w] for w in ws) for l, ws in FUNCTION_WORDS.items()}
    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    (top, n1), (_, n2) = ranked[0], ranked[1]
    if n1 < min_hits or n1 < MARGIN * n2:
        return None
    return top


def sentence_languages(text: str) -> Counter:
    """Idioma dominante de cada frase com palavras funcionais suficientes."""
    out = Counter()
    for sent in _SENT.split(text or ""):
        l = dominant_language(sent, SENT_MIN_HITS)
        if l:
            out[l] += 1
    return out


def switches(text: str, prompt_lang: str) -> Dict:
    s = scripts(text)
    lang = dominant_language(text)
    other = sorted(l for l in sentence_languages(text) if l != prompt_lang)
    return {"script_switch": bool(s), "scripts": ",".join(sorted(s)),
            "answer_lang": lang, "language_switch": lang is not None and lang != prompt_lang,
            "partial_language_switch": bool(other), "other_sentence_langs": ",".join(other)}
