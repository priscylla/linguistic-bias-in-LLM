"""
tests/test_entity_tokens.py
Fase 0, correção 5: probabilidades exatas por token ID, sem substring.
"""

import pytest
import torch

from conftest import VOCAB, WordTokenizer, build_tiny_model
from logit_lens.analyzer import LogitLensAnalyzer
from logit_lens.entity_tokens import (
    build_token_sets, first_token_after, prob_mass,
)
from logit_lens.extractor import LogitLensExtractor

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


# ── Extractor: entity_probs na distribuição completa ──

class _ModTok(WordTokenizer):
    """WordTokenizer com ids limitados ao vocabulário do modelo minúsculo."""
    def _id(self, tok):
        i = super()._id(tok)
        assert i < VOCAB
        return i


def test_extract_entity_probs_match_full_softmax(arch):
    tok   = _ModTok(splits={" Meucci": [" Me", "ucci"]})
    model = build_tiny_model(arch)
    ext   = LogitLensExtractor(model, tok, top_k=3)
    sets  = {"local": ["Meucci", "Antonio"], "competing": ["Bell", "Graham"]}
    res   = ext.extract(PROMPT, entity_sets=sets)

    ids = {k: v["ids"] for k, v in res["entity_tokens"]["sets"].items()}
    inputs = tok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    for k, layer in enumerate(res["last_token"], start=1):
        is_last = k == res["n_layers"]
        probs = ext._apply_logit_lens(out.hidden_states[k][0, -1], is_last)
        for name in sets:
            manual = probs[ids[name]].sum().item()
            assert layer["entity_probs"][name] == pytest.approx(manual, abs=1e-7)
        # top_ids correspondem aos top_tokens
        assert [tok.decode([i]) for i in layer["top_ids"]] == layer["top_tokens"]


def test_entity_probs_include_tokens_outside_topk():
    tok   = _ModTok()
    model = build_tiny_model("llama")
    ext   = LogitLensExtractor(model, tok, top_k=1)
    res   = ext.extract(PROMPT, entity_sets={"local": ["Bell"], "competing": ["Graham"]})
    sets  = res["entity_tokens"]["sets"]
    layer = res["last_token"][-1]
    # Com top-1, pelo menos um dos dois tokens fica fora do top-k, mas a
    # probabilidade exata dele continua sendo medida (o top-k dava 0)
    outside = [n for n in sets if sets[n]["ids"][0] not in layer["top_ids"]]
    assert outside
    for n in outside:
        assert layer["entity_probs"][n] > 0


# ── Analyzer ──

def _layers(p_local, p_comp):
    n = len(p_local)
    return [{"layer": i + 1, "layer_norm": i / (n - 1),
             "entity_probs": {"local": a, "competing": b}}
            for i, (a, b) in enumerate(zip(p_local, p_comp))]


def test_analyzer_uses_entity_probs():
    m = LogitLensAnalyzer().compute_all_metrics(
        _layers([0.0, 0.3, 0.6, 0.1], [0.1, 0.1, 0.2, 0.4]))
    assert m["iv_local_peak"] == pytest.approx(0.6)
    assert m["peak_layer"] == 3
    assert m["suppression"] == pytest.approx(0.5)
    assert m["cs_at_peak"] == pytest.approx(0.4)
    assert m["commitment_final"] == pytest.approx(-0.3)
    assert m["commitment_curve"] == pytest.approx([-0.1, 0.2, 0.4, -0.3])


def test_analyzer_rejects_old_topk_format():
    old = [{"layer": 1, "layer_norm": 0.0,
            "top_tokens": [" Bell"], "top_probs": [0.9]}]
    with pytest.raises(ValueError, match="entity_probs"):
        LogitLensAnalyzer().compute_all_metrics(old)
