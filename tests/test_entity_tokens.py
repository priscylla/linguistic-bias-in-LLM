"""
tests/test_entity_tokens.py
Fase 0, correção 5: probabilidades exatas por token ID, sem substring.
"""

import pytest
import torch

from conftest import WordTokenizer
from logit_lens.entity_tokens import (
    build_token_sets, first_token_after, prob_mass,
)

PROMPT = "Il telefono è stato inventato da"


# ── Primeiro token após o prompt ──

def test_bell_and_bella_are_different_tokens(word_tokenizer):
    bell  = first_token_after(word_tokenizer, PROMPT, "Bell")
    bella = first_token_after(word_tokenizer, PROMPT, "bella")
    assert bell["token"] == " Bell" and bella["token"] == " bella"
    assert bell["token_id"] != bella["token_id"]
    assert not bell["split"]


def test_split_word_uses_first_fragment(word_tokenizer):
    d = first_token_after(word_tokenizer, PROMPT, "Meucci")
    assert d["token"] == " Me"
    assert d["split"] and d["n_word_tokens"] == 2


def test_boundary_merge_excludes_word():
    tok = WordTokenizer(merges={(" da", " Vinci"): " daVinci"})
    d = first_token_after(tok, PROMPT, "Vinci")
    assert d["boundary_merge"] and d["token_id"] is None
    out = build_token_sets(tok, PROMPT, {"x": ["Vinci", "Bell"]})
    assert out["excluded"] == [{"set": "x", "word": "Vinci"}]
    assert out["sets"]["x"]["tokens"] == [" Bell"]


# ── Conjuntos ──

def test_duplicate_words_not_double_counted(word_tokenizer):
    out = build_token_sets(word_tokenizer, PROMPT,
                           {"bell": ["Bell", "Bell", "Graham"]})
    assert len(out["sets"]["bell"]["ids"]) == 2


def test_shared_token_counts_for_neither():
    # " Me" é o primeiro fragmento de Meucci e de Mendel
    tok = WordTokenizer(splits={" Meucci": [" Me", "ucci"],
                                " Mendel": [" Me", "ndel"]})
    out = build_token_sets(tok, PROMPT,
                           {"a": ["Meucci", "Antonio"], "b": ["Mendel", "Bell"]})
    assert out["sets"]["a"]["tokens"] == [" Antonio"]
    assert out["sets"]["b"]["tokens"] == [" Bell"]
    assert out["shared_dropped"][0]["token"] == " Me"
    assert sorted(out["shared_dropped"][0]["sets"]) == ["a", "b"]


def test_prob_mass_is_exact_sum():
    probs = torch.softmax(torch.randn(50), -1)
    assert prob_mass(probs, [3, 7, 7]) == pytest.approx(
        (probs[3] + probs[7]).item())
    assert prob_mass(probs, []) == 0.0


def test_real_tiny_tokenizer_bell_vs_bella():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM")
    except Exception as e:
        pytest.skip(f"tokenizer indisponível: {e}")
    bell  = first_token_after(tok, PROMPT, "Bell")
    bella = first_token_after(tok, PROMPT, "bella")
    assert bell["token_id"] != bella["token_id"]
