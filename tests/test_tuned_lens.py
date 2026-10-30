"""
tests/test_tuned_lens.py
Fase 4: Tuned Lens (tradutores, arquivos, dados, treino retomável,
avaliação held-out, integração com run_lens). Treino só por alguns
passos no modelo minúsculo, com corpus sintético em memória.
"""

import copy
import json
import random

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import run_lens
import train_tuned_lens as ttl
from config.experiment_config import ExperimentConfig
from config.loader import load_experiment, load_facts, load_models, load_tuned_lens, validate_tuned_lens
from conftest import WordTokenizer, build_tiny_model
from logit_lens.lens import LogitLens
from logit_lens.lens_training import (
    BalancedBatches, DocStream, Packer, fact_terms, is_held_out, sequence_kl, train,
)
from logit_lens.measure import LensRunner
from logit_lens.tuned_lens import LensMismatch, TunedLens
from data.text_matching import compile_terms

FACTS = load_facts()
TL    = load_tuned_lens()
REG   = load_models()
LANGS = ("pt", "en")


WORDS = [f"w{i:02d}" for i in range(40)]


class LensTok(WordTokenizer):
    """
    WordTokenizer com BOS = 1, EOS = 2 e vocabulário fixo das palavras
    do corpus sintético (ids iguais entre instâncias, como num
    tokenizer real; necessário para testar a retomada).
    """
    bos_token_id, eos_token_id = 1, 2

    def __init__(self):
        super().__init__()
        self.vocab["</s>"], self.inv[2] = 2, "</s>"
        for w in WORDS:
            self._id(" " + w)
            self._id(w)


def fake_source(lang, n=4000, inject=()):
    """Documentos sintéticos; `inject` = {índice: texto} para testar o filtro."""
    rng = random.Random(lang)
    docs = [{"id": f"{lang}-{i}",
             "text": dict(inject).get(i, " ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 30))))}
            for i in range(n)]
    return lambda: iter(docs)


def tiny_tl(**training):
    cfg = copy.deepcopy(TL)
    cfg["training"].update(dict(num_steps=6, tokens_per_step=64, seq_len=16,
                                loss_chunk_tokens=8, checkpoint_every=2), **training)
    cfg["eval"]["tokens_per_language"] = 64
    return cfg


def _model():
    return build_tiny_model("llama", vocab=512)


# ══════════════════════════════════════════════════════════════
# Tradutores
# ══════════════════════════════════════════════════════════════

def _out(model):
    torch.manual_seed(1)
    with torch.no_grad():
        return model(input_ids=torch.randint(3, 100, (1, 7)), output_hidden_states=True)


def test_new_tuned_lens_equals_logit_lens(arch):
    model = build_tiny_model(arch)
    out = _out(model)
    tuned = TunedLens(model).layer_logits(out.hidden_states, 6)
    plain = LogitLens(model).layer_logits(out.hidden_states, 6)
    assert torch.allclose(tuned, plain, atol=1e-6)
    assert len(TunedLens(model).translators) == model.config.num_hidden_layers - 1


def test_translation_changes_middle_layers_only():
    model = _model()
    lens  = TunedLens(model)
    with torch.no_grad():
        for t in lens.translators:
            t.weight.normal_(0, 0.1)
    out = _out(model)
    tuned = lens.layer_logits(out.hidden_states, 6)
    plain = LogitLens(model).layer_logits(out.hidden_states, 6)
    assert not torch.allclose(tuned[0], plain[0], atol=1e-4)
    assert torch.allclose(tuned[-1], plain[-1])          # última camada sem tradutor
    lens.check_consistency(out, 6)
    # layer_logits_grad == layer_logits nas camadas traduzidas
    k = 2
    g = lens.layer_logits_grad(k, out.hidden_states[k][0, 6])
    assert torch.allclose(g, tuned[k - 1], atol=1e-5)


def test_tuned_lens_plugs_into_runner():
    from config.loader import Prompt
    model = _model()
    res = LensRunner(model, WordTokenizer(), lens=TunedLens(model)).measure(
        Prompt("telefone", "pt", "F1", 2, "Quem inventou o telefone foi"), FACTS["telefone"])
    assert res["meta"]["lens"] == "tuned_lens" and res["probs"]


def test_save_load_roundtrip_and_mismatch(tmp_path):
    model = _model()
    lens  = TunedLens(model)
    with torch.no_grad():
        lens.translators[0].bias.fill_(0.5)
    lens.save(tmp_path / "l", meta={"steps": 3})
    back = TunedLens.load(model, tmp_path / "l")
    assert torch.equal(back.translators[0].bias, lens.translators[0].bias)
    assert back.meta["steps"] == 3

    from transformers import LlamaConfig, LlamaForCausalLM
    other = LlamaForCausalLM(LlamaConfig(**{**model.config.to_dict(), "num_hidden_layers": 2}))
    with pytest.raises(LensMismatch, match="n_layers"):
        TunedLens.load(other, tmp_path / "l")
    info = json.loads((tmp_path / "l" / "config.json").read_text())
    info["revision"] = "a" * 40
    (tmp_path / "l" / "config.json").write_text(json.dumps(info))
    with pytest.raises(LensMismatch, match="commit"):
        TunedLens.load(model, tmp_path / "l", expected_revision="b" * 40)


# ══════════════════════════════════════════════════════════════
# Dados
# ══════════════════════════════════════════════════════════════

def test_held_out_is_deterministic_and_about_one_percent():
    ids = [f"doc-{i}" for i in range(20000)]
    held = [i for i in ids if is_held_out(i, 100)]
    assert held == [i for i in ids if is_held_out(i, 100)]
    assert 0.007 < len(held) / len(ids) < 0.013


def test_train_and_eval_streams_are_disjoint():
    src = fake_source("pt", n=3000)
    train_ids, eval_ids = set(), set()
    for split, out in (("train", train_ids), ("eval", eval_ids)):
        s = DocStream(lambda: ({**d, "text": d["id"]} for d in src()), split, 100)
        out.update(s)
    assert train_ids and eval_ids and not (train_ids & eval_ids)
    assert len(train_ids) + len(eval_ids) == 3000


def test_filter_drops_fact_documents():
    terms = fact_terms(FACTS, "pt")
    assert {"Santos", "Wright", "Wilbur", "avião"} <= set(terms)    # todos os idiomas
    pat = compile_terms(terms)
    src = fake_source("pt", n=50, inject={3: "o voo de SANTOS-DUMONT em Paris",
                                          4: "os irmãos Wright", 5: "um avião novo",
                                          6: "santosdumont colado não casa"})
    s = DocStream(src, "train", 10**9, pat)       # mod enorme: quase nada vai para held-out
    kept = list(s)
    assert s.stats["filtered"] == 3
    assert not any("DUMONT" in t or "Wright" in t or "avião" in t for t in kept)
    assert any("santosdumont" in t for t in kept)


def test_packer_sequences():
    tok = LensTok()
    p = Packer(iter(["w01 w02 w03", "w04 w05", "w06 w07 w08 w09"] * 10), tok, seq_len=8)
    seqs = [p.next_sequence() for _ in range(4)]
    assert all(len(s) == 8 and s[0] == 1 for s in seqs)
    assert 2 in seqs[0]                                    # EOS separa documentos


def test_balanced_round_robin_and_resume_state():
    tok   = LensTok()
    srcs  = {l: fake_source(l) for l in LANGS}
    keys  = {l: {} for l in LANGS}
    b     = BalancedBatches(srcs, tok, 16, 100, {}, keys)
    first = b.next(4)
    assert [l for l, _ in first] == ["pt", "en", "pt", "en"]
    st    = b.state()
    after = b.next(6)
    b2    = BalancedBatches(srcs, LensTok(), 16, 100, {}, keys, state=st)   # processo novo
    assert b2.next(6) == after


# ══════════════════════════════════════════════════════════════
# Perda e treino
# ══════════════════════════════════════════════════════════════

def test_sequence_kl_matches_manual():
    model = _model()
    lens  = TunedLens(model)
    ids = [1] + list(range(5, 20))
    kl = sequence_kl(lens, model, ids, chunk=5)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    p = F.log_softmax(out.logits[0].float(), -1)
    q = F.log_softmax(LogitLens(model).head(out.hidden_states[1][0]), -1)
    assert kl[0].item() == pytest.approx((p.exp() * (p - q)).sum().item(), rel=1e-4)
    assert torch.allclose(kl, sequence_kl(lens, model, ids, chunk=5, translate=False))


def _sources():
    return {l: fake_source(l) for l in LANGS}


def test_training_reduces_held_out_kl(tmp_path):
    from logit_lens.lens_training import evaluate
    model, tok = _model(), LensTok()
    cfg  = tiny_tl(num_steps=30, lr=1.0)
    lens = TunedLens(model)
    res  = train(lens, model, tok, _sources(), cfg, FACTS, tmp_path, log=lambda *a: None)
    assert res["done"] and len(res["losses"]) == 30
    assert res["losses"][-1] < res["losses"][0]
    rows = pd.DataFrame(evaluate(lens, model, tok, _sources(), cfg, FACTS))
    assert set(rows["lang"]) == set(LANGS)
    assert rows["kl_tuned"].mean() < rows["kl_logit"].mean()
    assert not any(p.requires_grad for p in model.parameters())


def test_resume_matches_uninterrupted(tmp_path):
    cfg = tiny_tl(num_steps=4)
    straight = TunedLens(_model())
    r1 = train(straight, straight.model, LensTok(), _sources(), cfg, FACTS, tmp_path / "a",
               log=lambda *a: None)

    resumed = TunedLens(_model())
    train(resumed, resumed.model, LensTok(), _sources(), cfg, FACTS, tmp_path / "b",
          max_steps=2, log=lambda *a: None)
    again = TunedLens(_model())                         # processo novo
    r2 = train(again, again.model, LensTok(), _sources(), cfg, FACTS, tmp_path / "b",
               log=lambda *a: None)
    assert r2["step"] == 4 and r2["losses"] == pytest.approx(r1["losses"], rel=1e-6)
    for a, b in zip(straight.translators.parameters(), again.translators.parameters()):
        assert torch.allclose(a, b, atol=1e-7)


# ══════════════════════════════════════════════════════════════
# Script, run_lens e config
# ══════════════════════════════════════════════════════════════

def _tiny_config():
    from test_run_pipeline import tiny_config
    return tiny_config()


def test_script_trains_saves_and_evaluates(tmp_path):
    cfg, model, tok = _tiny_config(), _model(), LensTok()
    kw = dict(lens_root=tmp_path / "lenses", report_dir=tmp_path / "rep", log=lambda *a: None)
    s = ttl.run("llama", model, tok, cfg, tiny_tl(), _sources(), max_steps=3, **kw)
    assert not s["done"] and not (tmp_path / "lenses" / "llama" / "translators.pt").exists()
    s = ttl.run("llama", model, tok, cfg, tiny_tl(), _sources(), **kw)
    assert s["done"] and (tmp_path / "lenses" / "llama" / "translators.pt").exists()
    meta = json.loads((tmp_path / "lenses" / "llama" / "config.json").read_text())["meta"]
    assert meta["steps"] == 6 and meta["tokens"] == 6 * 64
    assert meta["corpus"]["pt"]["dataset"] == "HuggingFaceFW/fineweb-2" and meta["filter_fact_terms"]
    ev = pd.read_csv(tmp_path / "rep" / "llama_eval.csv")
    assert set(ev.columns) >= {"lang", "layer", "kl_tuned", "kl_logit"}
    s = ttl.run("llama", model, tok, cfg, tiny_tl(), _sources(), eval_only=True, **kw)
    assert "eval" in s


def test_run_lens_with_tuned_lens(tmp_path):
    from test_run_pipeline import CHAT
    cfg, model = _tiny_config(), build_tiny_model("llama", vocab=4096)
    with pytest.raises(FileNotFoundError, match="train_tuned_lens"):
        run_lens.load_tuned("llama", model, None, lens_root=tmp_path / "none")
    TunedLens(model).save(tmp_path / "lenses" / "llama")
    lens = run_lens.load_tuned("llama", model, None, lens_root=tmp_path / "lenses")
    out  = run_lens.lens_dir(str(tmp_path / "res"), "llama", "tuned_lens")
    m = run_lens.run_model("llama", model, WordTokenizer(), cfg, out, lens=lens,
                           chat_formatter=CHAT, log=lambda *a: None)
    assert m["lens"] == "tuned_lens"
    probs = pd.read_parquet(out / "probs.parquet")
    assert set(probs["lens"]) == {"tuned_lens"}
    assert "tuned_lens" in str(out) and "logit_lens" not in str(out)


def test_tuned_lens_config_valid_and_checks():
    exp = load_experiment()
    assert validate_tuned_lens(exp, TL, REG) == []
    assert TL["models"] == ["llama", "mistral", "qwen", "gemma"]
    assert TL["training"]["num_steps"] * TL["training"]["tokens_per_step"] == 250 * 2**16
    bad = copy.deepcopy(TL)
    bad["models"].append("llama_base")
    bad["corpus"]["fr"]["revision"] = "main"
    bad["training"]["tokens_per_step"] = 1000
    errs = validate_tuned_lens(exp, bad, REG)
    assert any("não é instruct" in e for e in errs)
    assert any("fr: revision" in e for e in errs)
    assert any("múltiplo" in e for e in errs)


def test_bf16_translators_train_save_load(tmp_path):
    # Opção para GPUs de 24 GB (RUNNING_1GPU.md): tradutores em bf16
    model = _model()
    lens  = TunedLens(model, translator_dtype="bfloat16")
    res   = train(lens, model, LensTok(), _sources(), tiny_tl(num_steps=4), FACTS, tmp_path / "c",
                  log=lambda *a: None)
    assert res["done"] and all(p.dtype == torch.bfloat16 for p in lens.translators.parameters())
    lens.save(tmp_path / "l")
    back = TunedLens.load(model, tmp_path / "l")
    assert back.translator_dtype == torch.bfloat16
    out = _out(model)
    assert torch.allclose(back.layer_logits(out.hidden_states, 6), lens.layer_logits(out.hidden_states, 6))
