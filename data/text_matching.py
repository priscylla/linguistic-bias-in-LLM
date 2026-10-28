"""
data/text_matching.py
Busca de termos em respostas geradas, com fronteira de palavra e
insensível a maiúsculas e diacríticos.

Substituí a comparação por substring ("bell" in "bella" é True).
Aqui "bell" só casa com a palavra inteira, e "dumont" casa com
"Dúmont" ou "DUMONT". Funciona com alfabeto latino e cirílico.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple


def normalize_text(text: str) -> str:
    """Remove diacríticos (NFKD sem marcas combinantes) e aplica casefold."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped   = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.casefold()


def _pattern(term: str) -> re.Pattern:
    # Fronteira de palavra Unicode: não pode haver letra/dígito colado
    return re.compile(r"(?<!\w)" + re.escape(normalize_text(term)) + r"(?!\w)")


def find_terms(text: str, terms: Sequence[str]) -> List[Tuple[int, str]]:
    """
    Todas as ocorrências dos termos no texto, como (posição, termo),
    ordenadas pela posição no texto normalizado.
    """
    norm = normalize_text(text)
    hits = []
    for term in terms:
        for m in _pattern(term).finditer(norm):
            hits.append((m.start(), term))
    return sorted(hits)


def first_position(text: str, terms: Sequence[str]) -> Optional[int]:
    """Posição da primeira ocorrência de qualquer termo, ou None."""
    hits = find_terms(text, terms)
    return hits[0][0] if hits else None


def groups_by_order(text: str, groups: Dict[str, Sequence[str]]) -> List[str]:
    """
    Grupos mencionados no texto, na ordem da primeira menção.

    Args:
        groups: {nome_do_grupo: [termos]}
    """
    firsts = {}
    for name, terms in groups.items():
        pos = first_position(text, terms)
        if pos is not None:
            firsts[name] = pos
    return sorted(firsts, key=firsts.get)


def compile_terms(terms: Sequence[str]) -> Optional[re.Pattern]:
    """
    Um único padrão (alternância, termos mais longos primeiro) com a
    mesma regra de find_terms; para varrer textos longos uma vez só.
    Aplicar ao texto já normalizado (normalize_text). None se vazio.
    """
    norm = sorted({normalize_text(t) for t in terms if t.strip()}, key=len, reverse=True)
    if not norm:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in norm) + r")(?!\w)")
