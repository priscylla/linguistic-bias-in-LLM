"""
tests/test_text_matching.py
Fase 0, correção 5: busca de termos com fronteira de palavra, sem
diferenciar maiúsculas nem diacríticos (usada na Fase 5); cores do
heatmap por token ID.
"""

import pytest

from data.text_matching import find_terms, groups_by_order, normalize_text
from make_figures import ENTITY_COLOR_DOMINANT, ENTITY_COLOR_OTHER, token_color


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


def test_group_order_ignores_words_containing_terms():
    groups = {"Meucci": ["meucci"], "Bell": ["bell"]}
    assert groups_by_order("Antonio Meucci, una bella storia", groups) == ["Meucci"]


# ── Cor do heatmap por token ID ──

RUN = {"entity_tokens": {"sets": {"local": {"ids": [10, 11]},
                                  "competing": {"ids": [20]}}}}


def test_token_color_by_id():
    # pt para telefone: local = Bell (dominante) → vermelho
    assert token_color(10, RUN, "telefone", "pt") == ENTITY_COLOR_DOMINANT
    assert token_color(20, RUN, "telefone", "pt") == ENTITY_COLOR_OTHER
    assert token_color(99, RUN, "telefone", "pt") == "#AAAAAA"
    assert token_color(None, RUN, "telefone", "pt") == "#AAAAAA"


def test_token_color_flips_in_stake_language():
    # it é o stake do telefone: local = Meucci, competing = Bell (dominante)
    assert token_color(20, RUN, "telefone", "it") == ENTITY_COLOR_DOMINANT
    assert token_color(10, RUN, "telefone", "it") == ENTITY_COLOR_OTHER
