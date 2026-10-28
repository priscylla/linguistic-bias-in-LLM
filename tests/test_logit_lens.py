"""
tests/test_logit_lens.py
Fase 0, correções 1–4: norm final única, softcapping do Gemma 2,
teste de sanidade do lens e precisão bf16.
"""

import pytest
import torch

from conftest import VOCAB, build_tiny_model
from logit_lens.extractor import (
    LensSanityError, LogitLensExtractor, lens_tolerance,
)
from models import model_loader
from models.model_loader import resolve_dtype


def _forward(model, seq_len=7):
    torch.manual_seed(1)
    ids = torch.randint(2, VOCAB, (1, seq_len))
    with torch.no_grad():
        return model(input_ids=ids, output_hidden_states=True)


# ── 1. Lens na última camada == logits do modelo ──

def test_last_layer_lens_matches_model_logits(arch, fake_tokenizer):
    model = build_tiny_model(arch)
    ext   = LogitLensExtractor(model, fake_tokenizer)
    out   = _forward(model)

    for pos in range(out.logits.shape[1]):
        lens = ext._logits_from_hidden(out.hidden_states[-1][0, pos],
                                       already_normed=True)
        assert torch.allclose(lens, out.logits[0, pos].float(),
                              atol=1e-5, rtol=1e-5)
        res = ext.check_lens_consistency(out, pos)
        assert res["argmax_match"]


# ── 2. Regressão da correção 1: norm dupla quebra a igualdade ──

def test_double_norm_would_fail(arch, fake_tokenizer):
    model = build_tiny_model(arch)
    ext   = LogitLensExtractor(model, fake_tokenizer)
    out   = _forward(model)

    double = ext._logits_from_hidden(out.hidden_states[-1][0, -1],
                                     already_normed=False)
    assert not torch.allclose(double, out.logits[0, -1].float(),
                              atol=1e-4, rtol=1e-4)


# ── 3. Regressão da correção 2: softcapping do Gemma 2 ──

def test_gemma2_softcapping(fake_tokenizer):
    model = build_tiny_model("gemma2")
    cap   = model.config.final_logit_softcapping
    ext   = LogitLensExtractor(model, fake_tokenizer)
    out   = _forward(model)

    h = out.hidden_states[-1][0, -1]
    lens = ext._logits_from_hidden(h, already_normed=True)
    assert lens.abs().max() <= cap + 1e-6

    # Sem o softcap, o lens não reproduz o modelo
    no_cap = model.lm_head(h).float()
    assert not torch.allclose(no_cap, out.logits[0, -1].float(),
                              atol=1e-4, rtol=1e-4)

    # Softcap também é aplicado nas camadas intermediárias
    mid = ext._logits_from_hidden(out.hidden_states[1][0, -1])
    assert mid.abs().max() <= cap + 1e-6


# ── 4. Camadas intermediárias continuam recebendo a norm ──

def test_intermediate_layers_get_norm(arch, fake_tokenizer):
    model = build_tiny_model(arch)
    ext   = LogitLensExtractor(model, fake_tokenizer)
    out   = _forward(model)
    cap   = getattr(model.config, "final_logit_softcapping", None)

    for k in range(1, len(out.hidden_states) - 1):
        h = out.hidden_states[k][0, -1]
        with torch.no_grad():
            manual = model.lm_head(model.model.norm(h))
            if cap is not None:
                manual = torch.tanh(manual / cap) * cap
        assert torch.allclose(ext._logits_from_hidden(h), manual.float(),
                              atol=1e-6)


def test_extract_uses_single_norm_at_last_layer(arch, fake_tokenizer):
    model = build_tiny_model(arch)
    ext   = LogitLensExtractor(model, fake_tokenizer, top_k=5)
    prompt = "The airplane was invented by"
    res = ext.extract(prompt)

    assert res["n_layers"] == model.config.num_hidden_layers
    assert len(res["last_token"]) == res["n_layers"]

    # Top-1 da última camada = argmax dos logits do modelo
    inputs = fake_tokenizer(prompt)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    probs = torch.softmax(logits.float(), -1)
    top_p, top_i = torch.topk(probs, 5)
    last = res["last_token"][-1]
    assert last["top_tokens"][0] == fake_tokenizer.decode([top_i[0].item()])
    assert last["top_probs"] == pytest.approx(top_p.tolist(), abs=1e-6)


# ── 5. A checagem falha alto quando os logits não batem ──

def test_consistency_check_raises_on_mismatch(fake_tokenizer):
    model = build_tiny_model("llama")
    ext   = LogitLensExtractor(model, fake_tokenizer)
    out   = _forward(model)
    out.logits[0, -1] += torch.linspace(0, 5, VOCAB)
    with pytest.raises(LensSanityError):
        ext.check_lens_consistency(out, out.logits.shape[1] - 1)


def test_sanity_check_raises_if_norm_applied_twice(fake_tokenizer, monkeypatch):
    model = build_tiny_model("llama")
    ext   = LogitLensExtractor(model, fake_tokenizer)
    orig  = ext._logits_from_hidden
    monkeypatch.setattr(ext, "_logits_from_hidden",
                        lambda h, already_normed=False: orig(h, False))
    with pytest.raises(LensSanityError):
        ext.sanity_check()


def test_sanity_check_report(arch, fake_tokenizer):
    ext = LogitLensExtractor(build_tiny_model(arch), fake_tokenizer)
    rep = ext.sanity_check()
    assert rep["model_type"] == arch
    assert len(rep["prompts"]) == 3
    assert rep["max_abs_diff"] < 1e-4


# ── 6. bf16 passa com a tolerância de bf16 ──

def test_bf16_model_passes(arch, fake_tokenizer):
    model = build_tiny_model(arch, dtype=torch.bfloat16)
    ext   = LogitLensExtractor(model, fake_tokenizer)
    rep   = ext.sanity_check()
    assert rep["dtype"] == "torch.bfloat16"
    probs = ext._apply_logit_lens(_forward(model).hidden_states[2][0, -1])
    assert probs.dtype == torch.float32
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_tolerance_by_dtype():
    assert lens_tolerance(torch.float32)["atol"] < lens_tolerance(torch.bfloat16)["atol"]


# ── 7. dtype vem do config, padrão bf16 ──

def test_resolve_dtype():
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("float32") is torch.float32
    with pytest.raises(ValueError):
        resolve_dtype("fp8")


def test_config_default_dtype_is_bf16():
    from config.experiment_config import ExperimentConfig
    assert ExperimentConfig().dtype == "bfloat16"


@pytest.mark.parametrize("dtype,expected", [
    (None, torch.bfloat16),            # padrão
    ("float32", torch.float32),
])
def test_load_model_passes_dtype(monkeypatch, fake_tokenizer, dtype, expected):
    seen = {}

    class _Tok:
        @staticmethod
        def from_pretrained(*a, **kw):
            t = type("T", (), {"pad_token": "x", "eos_token": "x"})()
            return t

    class _Model:
        @staticmethod
        def from_pretrained(name, **kw):
            seen.update(kw)
            return build_tiny_model("llama")

    monkeypatch.setattr(model_loader, "AutoTokenizer", _Tok)
    monkeypatch.setattr(model_loader, "AutoModelForCausalLM", _Model)
    monkeypatch.setattr(model_loader, "get_hf_token", lambda: None)

    args = ("org/tiny", "tiny") if dtype is None else ("org/tiny", "tiny", dtype)
    model_loader.load_model(*args)
    assert seen["dtype"] is expected
    assert "torch_dtype" not in seen


# ── 8. Ponta a ponta com tokenizer real (pulado se offline) ──

def test_end_to_end_tiny_hf_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    try:
        tok   = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    except Exception as e:  # sem rede e sem cache
        pytest.skip(f"modelo minúsculo indisponível: {e}")

    ext = LogitLensExtractor(model, tok, top_k=5)
    ext.sanity_check()
    res = ext.extract("The airplane was invented by", keyword="airplane")
    assert len(res["last_token"]) == model.config.num_hidden_layers
