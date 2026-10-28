"""
tests/test_logit_lens.py
Fase 0, correções 1–4 (norm final única, softcapping do Gemma 2, teste
de sanidade do lens, bf16), sobre o LogitLens da Fase 2.
"""

import pytest
import torch

from conftest import VOCAB, build_tiny_model
from logit_lens.lens import LensSanityError, LogitLens, lens_tolerance
from logit_lens.measure import LensRunner
from models import model_loader
from models.model_loader import resolve_dtype


def _forward(model, seq_len=7):
    torch.manual_seed(1)
    ids = torch.randint(2, VOCAB, (1, seq_len))
    with torch.no_grad():
        return model(input_ids=ids, output_hidden_states=True)


# ── 1. Lens na última camada == logits do modelo, em toda posição ──

def test_last_layer_lens_matches_model_logits(arch):
    model = build_tiny_model(arch)
    lens  = LogitLens(model)
    out   = _forward(model)
    for pos in range(out.logits.shape[1]):
        h = out.hidden_states[-1][0, pos]
        assert torch.allclose(lens.head(h, already_normed=True),
                              out.logits[0, pos].float(), atol=1e-5, rtol=1e-5)
        assert lens.check_consistency(out, pos)["argmax_match"]


# ── 2. Regressão da correção 1: norm dupla quebra a igualdade ──

def test_double_norm_would_fail(arch):
    model = build_tiny_model(arch)
    out   = _forward(model)
    double = LogitLens(model).head(out.hidden_states[-1][0, -1], already_normed=False)
    assert not torch.allclose(double, out.logits[0, -1].float(), atol=1e-4, rtol=1e-4)


# ── 3. Regressão da correção 2: softcapping do Gemma 2 ──

def test_gemma2_softcapping():
    model = build_tiny_model("gemma2")
    cap   = model.config.final_logit_softcapping
    lens  = LogitLens(model)
    out   = _forward(model)
    h = out.hidden_states[-1][0, -1]
    assert lens.head(h, already_normed=True).abs().max() <= cap + 1e-6
    # Sem o softcap, o lens não reproduz o modelo
    assert not torch.allclose(model.lm_head(h).float(), out.logits[0, -1].float(),
                              atol=1e-4, rtol=1e-4)
    # Softcap também nas camadas intermediárias
    L = lens.layer_logits(out.hidden_states, out.logits.shape[1] - 1)
    assert L.abs().max() <= cap + 1e-6


# ── 4. Teste de sanidade falha alto se a norm for aplicada duas vezes ──

def test_sanity_check_raises_if_norm_applied_twice(fake_tokenizer, monkeypatch):
    lens = LogitLens(build_tiny_model("llama"))
    orig = lens.head
    monkeypatch.setattr(lens, "head", lambda h, already_normed=False: orig(h, False))
    with pytest.raises(LensSanityError):
        lens.sanity_check(fake_tokenizer)


# ── 5. bf16 passa com a tolerância de bf16; distribuições em float32 ──

def test_bf16_model_passes(arch, fake_tokenizer):
    model = build_tiny_model(arch, dtype=torch.bfloat16)
    rep   = LogitLens(model).sanity_check(fake_tokenizer)
    assert rep["dtype"] == "torch.bfloat16"
    fw = LensRunner(model, fake_tokenizer).forward("The airplane was invented by")
    assert fw["probs"].dtype == torch.float32
    assert fw["probs"].sum(-1).tolist() == pytest.approx([1.0] * fw["n_layers"], abs=1e-5)


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
