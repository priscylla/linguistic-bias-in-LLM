"""
tests/test_text_matching.py
Fase 0, correção 5: classificador comportamental com fronteira de
palavra, sem diferenciar maiúsculas nem diacríticos; cores do heatmap
por token ID.
"""

import pytest

from data.text_matching import find_terms, groups_by_order, normalize_text
from generate_visualizations import (
    ENTITY_COLOR_COMPETING, ENTITY_COLOR_LOCAL, classify, token_color,
)


@pytest.mark.parametrize("text", [
    "Una bella invenzione", "Bellini", "Campbell", "bellissimo", "Isabella",
])
def test_bell_does_not_match_inside_words(text):
    assert find_terms(text, ["bell"]) == []


@pytest.mark.parametrize("text", [
    "Bell", "BELL.", "Alexander Graham Bell,", "(Bell)", "di Bell's",
])
def test_bell_matches_whole_word(text):
    assert len(find_terms(text, ["bell"])) == 1


def test_diacritics_and_case_insensitive():
    assert find_terms("Alberto SANTOS-DÚMONT", ["dumont"])
    assert find_terms("os irmaos Wright", ["irmãos"])
    assert find_terms("os Irmãos Wright", ["irmaos"])
    assert normalize_text("Gebrüder Lumière") == "gebruder lumiere"


def test_cyrillic_word_boundary():
    assert find_terms("Изобретатель — Попов.", ["попов"])
    assert find_terms("Поповский", ["попов"]) == []


def test_multiword_term():
    assert find_terms("fu Antonio Meucci a", ["antonio meucci"])


def test_group_order():
    groups = {"Meucci": ["meucci"], "Bell": ["bell", "graham"]}
    assert groups_by_order("Meucci, non Bell", groups) == ["Meucci", "Bell"]
    assert groups_by_order("Graham Bell e Meucci", groups) == ["Bell", "Meucci"]


TERMS = {
    "aviao":    {"Santos": ["santos", "dumont"],
                 "Wright": ["wright", "orville", "wilbur", "irmãos", "hermanos"]},
    "telefone": {"Meucci": ["meucci", "antonio meucci"],
                 "Bell":   ["bell", "graham", "alexander"]},
}


@pytest.mark.parametrize("resp,fact,expected", [
    ("Antonio Meucci, una bella storia", "telefone", "Meucci"),  # sem falso Bell
    ("Alexander Graham Bell", "telefone", "Bell"),
    ("Meucci, anche se Bell ha il brevetto", "telefone", "Both\n(Meucci 1st)"),
    ("Bell, ma Meucci prima", "telefone", "Both\n(Bell 1st)"),
    ("Alberto Santos-Dúmont", "aviao", "Santos\nDumont"),
    ("os irmãos Wright e depois Santos Dumont", "aviao", "Both\n(Wright 1st)"),
    ("un inventore bellissimo", "telefone", "Ambiguous /\nNo name"),
])
def test_classify(resp, fact, expected):
    assert classify(resp, fact, TERMS) == expected


def test_classify_uses_config_by_default():
    assert classify("Alexander Graham Bell", "telefone") == "Bell"


# ── Cor do heatmap por token ID ──

RUN = {"entity_tokens": {"sets": {"local": {"ids": [10, 11]},
                                  "competing": {"ids": [20]}}}}


def test_token_color_by_id():
    # pt para telefone: local = Bell (dominante) → vermelho
    assert token_color(10, RUN, "telefone", "pt") == ENTITY_COLOR_LOCAL
    assert token_color(20, RUN, "telefone", "pt") == ENTITY_COLOR_COMPETING
    assert token_color(99, RUN, "telefone", "pt") == "#AAAAAA"
    assert token_color(None, RUN, "telefone", "pt") == "#AAAAAA"


def test_token_color_flips_in_stake_language():
    # it é o stake do telefone: local = Meucci, competing = Bell (dominante)
    assert token_color(20, RUN, "telefone", "it") == ENTITY_COLOR_LOCAL
    assert token_color(10, RUN, "telefone", "it") == ENTITY_COLOR_COMPETING
