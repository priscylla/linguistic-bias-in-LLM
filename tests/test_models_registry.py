"""
tests/test_models_registry.py
Fase 3: registro de modelos, cabeça de saída pela arquitetura, carga
com commit fixo / precisão (NF4 só por --dtype), formatos por kind e sanidade por modelo.
"""

import copy
import dataclasses

import pytest
import torch
import yaml

import run_lens
from logit_lens import sanity as scm
from config.experiment_config import ExperimentConfig
from config.loader import (
    CONFIG_DIR, Architecture, load_experiment, load_models, parse_models, validate_models,
)
from conftest import ARCHS, WordTokenizer, build_tiny_model
from logit_lens.lens import LogitLens, output_head
from models import model_loader

REG = load_models()
EXP = load_experiment()

# Tabela confirmada pela autora em 2026-10-28
CONFIRMED = {
    "llama_base":   ("meta-llama/Llama-3.1-8B",          "base",     "llama",   True),
    "llama":        ("meta-llama/Llama-3.1-8B-Instruct", "instruct", "llama",   True),
    "mistral_base": ("mistralai/Mistral-7B-v0.3",        "base",     "mistral", False),
    "mistral":      ("mistralai/Mistral-7B-Instruct-v0.3", "instruct", "mistral", False),
    "qwen_base":    ("Qwen/Qwen2.5-7B",                  "base",     "qwen2",   False),
    "qwen":         ("Qwen/Qwen2.5-7B-Instruct",         "instruct", "qwen2",   False),
    "gemma_base":   ("google/gemma-2-9b",                "base",     "gemma2",  True),
    "gemma":        ("google/gemma-2-9b-it",             "instruct", "gemma2",  True),
    "gemma27":      ("google/gemma-2-27b-it",            "instruct", "gemma2",  True),
    "sabia_base":   ("huggyllama/llama-7b",              "base",     "llama",   False),
    "sabia":        ("maritaca-ai/sabia-7b",             "base",     "llama",   False),
}


def _raw():
    return yaml.safe_load((CONFIG_DIR / "models.yaml").read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════
# Registro
# ══════════════════════════════════════════════════════════════

def test_registry_matches_confirmed_table():
    assert validate_models(EXP, REG) == []
    got = {k: (m.id, m.kind, m.architecture, m.gated) for k, m in REG.models.items()}
    assert got == CONFIRMED
    assert {k: m.base_model for k, m in REG.models.items() if m.base_model} == {
        "llama": "llama_base", "mistral": "mistral_base", "qwen": "qwen_base",
        "gemma": "gemma_base", "sabia": "sabia_base"}
    assert REG["sabia"].adaptation == "pt"
    assert [k for k, m in REG.models.items() if m.quantization] == []    # gemma27 em bf16 (2026-10-29)
    assert REG.quantization["nf4"]["bnb_4bit_use_double_quant"] is False
    assert all(m.precision == "bfloat16" for m in REG.models.values())


def test_formats_by_kind():
    assert REG["qwen_base"].formats == ("completion",)       # mesmo com chat template
    assert REG["sabia"].formats == ("completion",)
    assert REG["gemma27"].formats == ("completion", "qa")


def test_config_models():
    cfg = ExperimentConfig()
    assert cfg.models["llama"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert list(cfg.models) == list(CONFIRMED)
    with pytest.raises(KeyError, match="fora do registro"):
        cfg.model("gpt")


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r["models"]["llama"].update(architecture="gpt2"), "arquitetura"),
    (lambda r: r["models"]["llama"].update(revision="0e9e39f249"), "40 hex"),
    (lambda r: r["models"]["llama"].update(base_model="llama_2"), "base_model"),
    (lambda r: r["models"]["llama"].update(kind="chat"), "kind"),
    (lambda r: r["models"]["llama"].update(precision="int8"), "precision"),
    (lambda r: r["models"]["llama"].update(quantization="nf8"), "quantization"),
    (lambda r: r["models"]["sabia"].update(adaptation="es"), "adaptation"),
])
def test_validation_catches(mutate, msg):
    raw = copy.deepcopy(_raw())
    mutate(raw)
    assert any(msg in e for e in validate_models(EXP, parse_models(raw)))


def test_heatmap_models_in_registry():
    assert any("heatmap_models" in e for e in
               validate_models({**EXP, "heatmap_models": ["llama", "gpt"]}, REG))


# ══════════════════════════════════════════════════════════════
# Cabeça de saída pela arquitetura
# ══════════════════════════════════════════════════════════════

def test_every_registry_architecture_resolves(arch):
    model = build_tiny_model(arch)
    head  = output_head(model)
    assert head["final_norm"] is model.model.norm and head["unembedding"] is model.lm_head
    assert head["softcap"] == getattr(model.config, "final_logit_softcapping", None)
    assert head["architecture"].model_type == arch


def test_registry_uses_only_tested_architectures():
    assert {m.architecture for m in REG.models.values()} <= set(ARCHS)
    assert set(REG.architectures) == set(ARCHS)


def test_unknown_architecture():
    model = build_tiny_model("llama")
    model.config.model_type = "falcon"
    with pytest.raises(ValueError, match="não mapeada"):
        LogitLens(model)


@pytest.mark.parametrize("arch_spec,msg", [
    (Architecture("llama", final_norm="model.normx", unembedding="lm_head"), "não tem o atributo"),
    (Architecture("llama", final_norm="lm_head", unembedding="lm_head"), "final_norm"),
    (Architecture("llama", final_norm="model.norm", unembedding="model.norm"), "unembedding"),
])
def test_wrong_paths_fail_loudly(arch_spec, msg):
    with pytest.raises(ValueError, match=msg):
        LogitLens(build_tiny_model("llama"), arch_spec)


def test_softcap_comes_from_architecture():
    model = build_tiny_model("gemma2")
    no_cap = Architecture("gemma2", final_norm="model.norm", unembedding="lm_head", softcap=None)
    assert LogitLens(model, no_cap).softcap is None
    assert LogitLens(model).softcap == model.config.final_logit_softcapping


# ══════════════════════════════════════════════════════════════
# Carga
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def fake_hub(monkeypatch):
    seen = {"tok": {}, "model": {}}

    class _Tok:
        @staticmethod
        def from_pretrained(name, **kw):
            seen["tok"] = {"name": name, **kw}
            return type("T", (), {"pad_token": None, "eos_token": "</s>"})()

    def _model_for(arch):
        class _Model:
            @staticmethod
            def from_pretrained(name, **kw):
                seen["model"] = {"name": name, **kw}
                return build_tiny_model(arch)
        return _Model

    monkeypatch.setattr(model_loader, "AutoTokenizer", _Tok)
    seen["use"] = lambda arch: monkeypatch.setattr(model_loader, "AutoModelForCausalLM", _model_for(arch))
    return seen


def test_load_model_pins_revision_and_precision(fake_hub, monkeypatch):
    monkeypatch.setattr(model_loader, "get_hf_token", lambda: "hf_test_token")
    fake_hub["use"]("llama")
    model, tok = model_loader.load_model("llama")
    spec = REG["llama"]
    for k in ("tok", "model"):
        assert fake_hub[k]["name"] == spec.id and fake_hub[k]["revision"] == spec.revision
        assert "trust_remote_code" not in fake_hub[k]
    assert fake_hub["model"]["dtype"] is torch.bfloat16
    assert "quantization_config" not in fake_hub["model"]
    assert tok.pad_token == "</s>"


def test_load_gemma27_bf16_and_nf4_override(fake_hub, monkeypatch):
    from models import runtime
    monkeypatch.setattr(model_loader, "get_hf_token", lambda: "hf_test_token")
    fake_hub["use"]("gemma2")
    model_loader.load_model("gemma27")                                  # registro: bf16, sem quantização
    assert fake_hub["model"]["dtype"] is torch.bfloat16
    assert "quantization_config" not in fake_hub["model"]
    model_loader.load_model("gemma27", runtime.registry_with_dtype(REG, "gemma27", "nf4"))
    q = fake_hub["model"]["quantization_config"]
    assert q.load_in_4bit and q.bnb_4bit_quant_type == "nf4"
    assert q.bnb_4bit_compute_dtype is torch.bfloat16 and not q.bnb_4bit_use_double_quant


def test_load_checks_architecture(fake_hub, monkeypatch):
    monkeypatch.setattr(model_loader, "get_hf_token", lambda: None)
    fake_hub["use"]("llama")
    with pytest.raises(model_loader.ArchitectureMismatch):
        model_loader.load_model("mistral")             # carrega llama, registro diz mistral


def test_gated_needs_token_ungated_does_not(fake_hub, monkeypatch):
    monkeypatch.setattr(model_loader, "get_hf_token", lambda: None)
    fake_hub["use"]("llama")
    with pytest.raises(EnvironmentError):
        model_loader.load_model("llama")
    model_loader.load_model("sabia")                    # llama, sem gating
    assert fake_hub["model"]["token"] is None


# ══════════════════════════════════════════════════════════════
# run_lens e sanidade por modelo
# ══════════════════════════════════════════════════════════════

def _spec(key, **kw):
    return dataclasses.replace(REG[key], **kw)


def test_run_model_formats_follow_registry(tmp_path):
    from test_run_pipeline import CHAT, tiny_config
    cfg   = tiny_config()
    model = build_tiny_model("llama", vocab=4096)
    m = run_lens.run_model("llama_base", model, WordTokenizer(), cfg, tmp_path / "b",
                           chat_formatter=CHAT, spec=REG["llama_base"], log=lambda *a: None)
    assert m["formats"] == ["completion"] and m["registry"]["id"] == "meta-llama/Llama-3.1-8B"
    with pytest.raises(ValueError, match="chat template"):
        run_lens.run_model("llama", model, WordTokenizer(), cfg, tmp_path / "i",
                           spec=REG["llama"], log=lambda *a: None)
    with pytest.raises(ValueError, match="registro"):
        run_lens.run_model("gemma", model, WordTokenizer(), cfg, tmp_path / "g",
                           chat_formatter=CHAT, spec=REG["gemma"], log=lambda *a: None)


@pytest.mark.parametrize("arch,key", [("llama", "llama"), ("mistral", "mistral"),
                                      ("qwen2", "qwen"), ("gemma2", "gemma27")])
def test_sanity_report_per_model(arch, key):
    from test_run_pipeline import CHAT
    cfg = ExperimentConfig()
    rep = scm.sanity_report(key, build_tiny_model(arch, vocab=4096), WordTokenizer(),
                            REG[key], cfg, chat_formatter=CHAT)
    assert rep["passed"], rep["failures"]
    kinds = [c["kind"] for c in rep["checks"]]
    assert kinds.count("fixed") == 3 and kinds.count("completion") == len(cfg.languages)
    assert kinds.count("qa") == len(cfg.languages)
    assert rep["head"]["final_norm"] == "model.norm"


def test_sanity_report_base_has_no_qa_and_flags_failures(monkeypatch):
    from test_run_pipeline import CHAT
    model = build_tiny_model("llama", vocab=4096)
    rep = scm.sanity_report("sabia", model, WordTokenizer(), REG["sabia"], ExperimentConfig(), CHAT)
    assert rep["passed"] and "qa" not in {c["kind"] for c in rep["checks"]}

    orig = LogitLens.head
    monkeypatch.setattr(LogitLens, "head",
                        lambda self, h, already_normed=False: orig(self, h, False))   # norm 2x
    rep = scm.sanity_report("sabia", model, WordTokenizer(), REG["sabia"], ExperimentConfig())
    assert not rep["passed"] and rep["failures"]
