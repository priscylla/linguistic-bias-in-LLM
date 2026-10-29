"""
tests/test_lens_measure.py
Fase 2 (primeira parte): interface de lens, probabilidades exatas por
token ID, KL até a distribuição final, top-20 e linha de base neutra.
"""

import pytest
import torch
import torch.nn.functional as F

from config.loader import Candidate, Fact, Prompt
from conftest import VOCAB, WordTokenizer, build_tiny_model
from logit_lens.entity_tokens import encode_ids, first_token_after
from logit_lens.lens import Lens, LensSanityError, LogitLens
from logit_lens.measure import LensRunner, kl_to_final


def _forward(model, seq_len=7):
    torch.manual_seed(1)
    ids = torch.randint(2, VOCAB, (1, seq_len))
    with torch.no_grad():
        return model(input_ids=ids, output_hidden_states=True)


def _manual_head(model, h, normed):
    with torch.no_grad():
        logits = model.lm_head(h if normed else model.model.norm(h))
        cap = getattr(model.config, "final_logit_softcapping", None)
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
    return logits.float()


# ══════════════════════════════════════════════════════════════
# Lens
# ══════════════════════════════════════════════════════════════

def test_layer_logits_match_manual_head(arch):
    model = build_tiny_model(arch)
    out   = _forward(model)
    pos   = out.logits.shape[1] - 1
    L     = LogitLens(model).layer_logits(out.hidden_states, pos)
    n     = model.config.num_hidden_layers
    assert L.shape == (n, VOCAB) and L.dtype == torch.float32
    for k in range(1, n + 1):
        manual = _manual_head(model, out.hidden_states[k][0, pos], normed=(k == n))
        assert torch.allclose(L[k - 1], manual, atol=1e-5)
    # Última camada = logits do modelo (norm aplicada uma vez só)
    assert torch.allclose(L[-1], out.logits[0, pos].float(), atol=1e-5)


class _Scaled(Lens):
    """Lens de teste: tradução não trivial, registra as camadas pedidas."""
    name = "scaled"

    def __init__(self, model):
        super().__init__(model)
        self.seen = []

    def translate(self, layer_idx, hidden):
        self.seen.append(layer_idx)
        return 2.0 * hidden + 0.1


def test_lens_is_pluggable(arch):
    model  = build_tiny_model(arch)
    out    = _forward(model)
    pos    = out.logits.shape[1] - 1
    lens   = _Scaled(model)
    scaled = lens.layer_logits(out.hidden_states, pos)
    plain  = LogitLens(model).layer_logits(out.hidden_states, pos)
    n      = model.config.num_hidden_layers
    # Tradução só nas camadas 1..n-1; a última sai do modelo sem tradução
    assert lens.seen == list(range(1, n))
    assert not torch.allclose(scaled[0], plain[0], atol=1e-4)
    assert torch.allclose(scaled[-1], plain[-1])
    lens.check_consistency(out, pos)


def test_lens_check_raises_on_mismatch():
    model = build_tiny_model("llama")
    out   = _forward(model)
    out.logits[0, -1] += torch.linspace(0, 5, VOCAB)
    with pytest.raises(LensSanityError):
        LogitLens(model).check_consistency(out, out.logits.shape[1] - 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lens_sanity_check(arch, fake_tokenizer, dtype):
    rep = LogitLens(build_tiny_model(arch, dtype)).sanity_check(fake_tokenizer)
    assert rep["lens"] == "logit_lens" and len(rep["prompts"]) == 3


def test_unknown_architecture_raises():
    model = build_tiny_model("llama")
    model.config.model_type = "gpt_unknown"
    with pytest.raises(ValueError, match="não mapeada"):
        LogitLens(model)


# ══════════════════════════════════════════════════════════════
# KL até a final
# ══════════════════════════════════════════════════════════════

def test_kl_to_final_definition():
    torch.manual_seed(0)
    layers = torch.randn(4, VOCAB) * 3
    final  = torch.randn(VOCAB) * 3
    kl     = kl_to_final(layers, final)
    logq   = F.log_softmax(final, -1)
    for i in range(4):
        logp = F.log_softmax(layers[i], -1)
        ref  = F.kl_div(logq, logp, log_target=True, reduction="sum")
        assert kl[i].item() == pytest.approx(ref.item(), rel=1e-5)
    assert (kl >= 0).all()
    assert kl_to_final(final.unsqueeze(0), final)[0].item() == pytest.approx(0.0, abs=1e-6)


# ══════════════════════════════════════════════════════════════
# Medição com um fato sintético
# ══════════════════════════════════════════════════════════════

def _cand(cid, dominant, names, place=None, lang="it"):
    return Candidate(id=cid, label=cid, dominant=dominant, stakes=[],
                     names={lang: {"nom": names}},
                     indicators={"place": {lang: place}} if place else {})


def _fact(key="telefone", meucci_names=("Meucci", "Antonio")):
    return Fact(key=key, kind="disputed", status="confirmed", verb="invented",
                keyword={"it": "telefono"},
                candidates=[_cand("bell", True, ["Bell", "Graham", "Bell"], ["Boston"]),
                            _cand("meucci", False, list(meucci_names), ["Firenze"])])


P_F1    = Prompt("telefone", "it", "F1", 1, "Il telefono è stato inventato da")
P_PLACE = Prompt("telefone", "it", "place", 1, "Il telefono è stato inventato a")
P_QA    = Prompt("telefone", "it", "QA", 1, "Chi ha inventato il telefono?")


def _manual_probs(model, tok, text, add_special=True):
    ids = torch.tensor([encode_ids(tok, text, add_special)])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    pos = ids.shape[1] - 1
    L = LogitLens(model).layer_logits(out.hidden_states, pos)
    return F.softmax(L, -1), out.logits[0, pos]


def _prob(res, cid, layer):
    return next(r["prob"] for r in res["probs"]
                if r["candidate"] == cid and r["layer"] == layer)


def test_exact_probs_over_full_distribution(arch):
    model = build_tiny_model(arch)
    tok   = WordTokenizer()
    res   = LensRunner(model, tok, top_k=20).measure(P_F1, _fact())
    probs, _ = _manual_probs(model, tok, P_F1.text)

    ids = {w: first_token_after(tok, P_F1.text, w)["token_id"]
           for w in ("Bell", "Graham", "Meucci", "Antonio")}
    # " Bell" ≠ " bella": ids exatos, nada de substring
    assert ids["Bell"] != first_token_after(tok, P_F1.text, "bella")["token_id"]

    n = model.config.num_hidden_layers
    outside_top = 0
    for k in range(1, n + 1):
        p = probs[k - 1]
        # "Bell" repetido conta uma vez
        assert _prob(res, "bell", k) == pytest.approx(
            (p[ids["Bell"]] + p[ids["Graham"]]).item(), abs=1e-7)
        assert _prob(res, "meucci", k) == pytest.approx(
            (p[ids["Meucci"]] + p[ids["Antonio"]]).item(), abs=1e-7)
        top = res["layers"][k - 1]["top_ids"]
        outside_top += sum(i not in top for i in ids.values())
    # Ids fora do top-20 também contam (sem truncamento)
    assert outside_top > 0
    assert {r["n_ids"] for r in res["probs"]} == {2}
    assert {r["token_set"] for r in res["probs"]} == {"names"}
    assert len(res["probs"]) == 2 * n


def test_shared_first_token_resolved_by_prefix_forcing():
    # Decisão da autora (2026-10-29): token compartilhado forçado,
    # P(e) += P(prefixo) × P(próximo | prefixo), camada a camada
    model  = build_tiny_model("llama")
    tok    = WordTokenizer(splits={" Meucci": [" Gra", "ucci"], " Graham": [" Gra", "ham"]})
    runner = LensRunner(model, tok)
    res    = runner.measure(P_F1, _fact())
    assert [d["token"] for d in res["token_sets"]["shared_dropped"]] == [" Gra"]
    case = res["prefix_cases"][0]
    assert case["token"] == " Gra" and case["candidates"] == ["bell", "meucci"]
    assert {w["candidate"]: (w["next_token"], w["status"]) for w in case["words"]} == {
        "bell": ("ham", "combined"), "meucci": ("ucci", "combined")}
    assert {(r["candidate"], r["n_ids"], r["n_prefix_ids"]) for r in res["probs"]} == {
        ("bell", 2, 1), ("meucci", 2, 1)}                     # Bell + Graham; Antonio + Meucci

    p_ids = encode_ids(tok, P_F1.text)
    probs, _ = _manual_probs(model, tok, P_F1.text)
    fw2 = runner.forward_ids(p_ids + [tok.vocab[" Gra"]])["probs"]
    for k in range(model.config.num_hidden_layers):
        r = {x["candidate"]: x for x in res["probs"] if x["layer"] == k + 1}
        direct = probs[k][tok.vocab[" Bell"]].item()
        prefix = probs[k][tok.vocab[" Gra"]].item() * fw2[k][tok.vocab["ham"]].item()
        assert r["bell"]["prob_direct"] == pytest.approx(direct, abs=1e-7)
        assert r["bell"]["prob_prefix"] == pytest.approx(prefix, abs=1e-7)
        assert r["bell"]["prob"] == pytest.approx(direct + prefix, abs=1e-7)


def test_prefix_not_resolved_when_word_is_the_shared_token():
    model = build_tiny_model("llama")
    tok   = WordTokenizer()
    res   = LensRunner(model, tok).measure(P_F1, _fact(meucci_names=("Bell",)))
    w = res["prefix_cases"][0]["words"]
    assert {x["status"] for x in w} == {"no_next_token"}
    assert {r["prob_prefix"] for r in res["probs"]} == {0.0}


def test_empty_set_gives_zero():
    model = build_tiny_model("llama")
    tok   = WordTokenizer()
    res   = LensRunner(model, tok).measure(P_F1, _fact(meucci_names=("Bell",)))
    # O único nome de Meucci coincide com o de Bell: cai dos dois
    assert {r["prob"] for r in res["probs"] if r["candidate"] == "meucci"} == {0.0}
    assert {r["n_ids"] for r in res["probs"] if r["candidate"] == "meucci"} == {0}


def test_layer_rows_kl_and_top20(arch):
    model = build_tiny_model(arch)
    tok   = WordTokenizer()
    res   = LensRunner(model, tok, top_k=20).measure(P_F1, _fact())
    probs, final = _manual_probs(model, tok, P_F1.text)
    n = model.config.num_hidden_layers
    assert [r["layer"] for r in res["layers"]] == list(range(1, n + 1))
    assert [r["depth"] for r in res["layers"]] == pytest.approx([k / n for k in range(1, n + 1)])
    ref = kl_to_final(torch.log(probs), final)
    for r, kl in zip(res["layers"], ref):
        assert r["kl_to_final"] == pytest.approx(kl.item(), abs=1e-5)
        assert len(r["top_ids"]) == 20
        assert r["top_probs"] == sorted(r["top_probs"], reverse=True)
        assert r["top_tokens"] == [tok.decode([i]) for i in r["top_ids"]]
    # Última camada é a própria distribuição final
    assert res["layers"][-1]["kl_to_final"] == pytest.approx(0.0, abs=1e-5)
    assert res["meta"]["lens"] == "logit_lens" and res["meta"]["format"] == "completion"


def test_indicator_prompts_measure_indicators():
    model = build_tiny_model("llama")
    tok   = WordTokenizer()
    res   = LensRunner(model, tok).measure(P_PLACE, _fact())
    words = {c: [d["word"] for d in s["words"]] for c, s in res["token_sets"]["sets"].items()}
    assert words == {"bell": ["Boston"], "meucci": ["Firenze"]}
    assert {r["token_set"] for r in res["probs"]} == {"indicators"}


def test_qa_format_uses_chat_template_without_space_or_bos():
    model  = build_tiny_model("llama")
    tok    = WordTokenizer()
    runner = LensRunner(model, tok, chat_formatter=lambda t, x: f"<user>{x}<assistant>")
    res    = runner.measure(P_QA, _fact())
    chat   = f"<user>{P_QA.text}<assistant>"
    assert res["n_input_tokens"] == len(encode_ids(tok, chat, add_special_tokens=False))
    assert res["decision_pos"] == res["n_input_tokens"] - 1
    bell = res["token_sets"]["sets"]["bell"]["words"][0]
    assert bell["token"] == "Bell"                     # primeiro token do assistente
    probs, _ = _manual_probs(model, tok, chat, add_special=False)
    assert _prob(res, "bell", 1) == pytest.approx(
        (probs[0][bell["token_id"]] + probs[0][tok.vocab["Graham"]]).item(), abs=1e-7)


def test_qa_without_chat_template_raises():
    with pytest.raises(ValueError, match="chat template"):
        LensRunner(build_tiny_model("llama"), WordTokenizer()).measure(P_QA, _fact())


def test_forward_checks_lens_against_model():
    class _DoubleNorm(LogitLens):
        def layer_logits(self, hs, pos):
            L = super().layer_logits(hs, pos)
            L[-1] = self.head(hs[-1][0, pos], already_normed=False)   # norm 2x
            return L

    model = build_tiny_model("llama")
    with pytest.raises(LensSanityError):
        LensRunner(model, WordTokenizer(), lens=_DoubleNorm(model)).measure(P_F1, _fact())


# ══════════════════════════════════════════════════════════════
# Linha de base neutra
# ══════════════════════════════════════════════════════════════

def _facts():
    other = Fact(key="radio", kind="disputed", status="confirmed", verb="invented",
                 keyword={"it": "radio"},
                 candidates=[_cand("marconi", True, ["Marconi"]),
                             _cand("popov", False, ["Popov"])])     # sem indicadores
    return {"telefone": _fact(), "radio": other}


N_F1    = Prompt(None, "it", "F1", 1, "Il romanzo è stato scritto da")
N_PLACE = Prompt(None, "it", "place", 1, "Il ponte è stato costruito a")


def test_baseline_one_forward_all_facts(monkeypatch):
    model  = build_tiny_model("llama")
    tok    = WordTokenizer()
    runner = LensRunner(model, tok)
    calls  = []
    orig   = runner.forward
    monkeypatch.setattr(runner, "forward", lambda *a, **k: calls.append(1) or orig(*a, **k))

    res = runner.measure_baseline(N_F1, _facts())
    assert len(calls) == 1
    assert res["meta"]["fact"] is None and res["meta"]["prompt_type"] == "F1"
    assert {(r["fact"], r["candidate"]) for r in res["probs"]} == {
        ("telefone", "bell"), ("telefone", "meucci"),
        ("radio", "marconi"), ("radio", "popov")}

    probs, _ = _manual_probs(model, tok, N_F1.text)
    pid = first_token_after(tok, N_F1.text, "Popov")["token_id"]
    for k in range(1, model.config.num_hidden_layers + 1):
        got = next(r["prob"] for r in res["probs"]
                   if r["candidate"] == "popov" and r["layer"] == k)
        assert got == pytest.approx(probs[k - 1][pid].item(), abs=1e-7)


def test_baseline_indicators_only_for_facts_with_them():
    res = LensRunner(build_tiny_model("llama"), WordTokenizer()).measure_baseline(N_PLACE, _facts())
    assert set(res["token_sets"]) == {"telefone"}
    assert {r["token_set"] for r in res["probs"]} == {"indicators"}


def test_baseline_uses_russian_slot_case():
    names = {"ru": {"nom": ["Белл"], "gen": ["Белла"], "ins": ["Беллом"]}}
    bell  = Candidate(id="bell", label="Bell", dominant=True, stakes=[], names=names, indicators={})
    fact  = Fact(key="telefone", kind="control", status="confirmed", verb="invented",
                 keyword={"ru": "телефон"}, candidates=[bell])
    neutral = Prompt(None, "ru", "F1", 2, "Роман — произведение", slot_case="gen")
    res = LensRunner(build_tiny_model("llama"), WordTokenizer()).measure_baseline(
        neutral, {"telefone": fact})
    words = [d["word"] for d in res["token_sets"]["telefone"]["sets"]["bell"]["words"]]
    assert words == ["Белла"]


def test_prefix_forcing_recurses_until_candidates_diverge():
    # " M" + "x" compartilhados; divergem no terceiro token -> profundidade 2
    model  = build_tiny_model("llama", vocab=512)
    tok    = WordTokenizer(splits={" Meucci": [" M", "x", "a"], " Graham": [" M", "x", "b"]})
    runner = LensRunner(model, tok)
    res    = runner.measure(P_F1, _fact())
    w = {x["candidate"]: x for x in res["prefix_cases"][0]["words"]}
    v = tok.vocab
    assert w["bell"]["path"] == [v[" M"], v["x"], v["b"]] and w["bell"]["depth"] == 2
    assert w["meucci"]["path"] == [v[" M"], v["x"], v["a"]] and w["meucci"]["status"] == "combined"
    p_ids = encode_ids(tok, P_F1.text)
    probs, _ = _manual_probs(model, tok, P_F1.text)
    f1 = runner.forward_ids(p_ids + [v[" M"]])["probs"]
    f2 = runner.forward_ids(p_ids + [v[" M"], v["x"]])["probs"]
    for k in range(model.config.num_hidden_layers):
        r = next(x for x in res["probs"] if x["candidate"] == "bell" and x["layer"] == k + 1)
        expect = probs[k][v[" M"]] * f1[k][v["x"]] * f2[k][v["b"]]
        assert r["prob_prefix"] == pytest.approx(expect.item(), abs=1e-9)


def test_prefix_still_shared_after_max_depth():
    from logit_lens.entity_tokens import build_token_sets, resolve_shared_prefixes
    tok  = WordTokenizer(splits={" Meucci": [" M", "x", "y", "z", "a"], " Graham": [" M", "x", "y", "z", "b"]})
    sets = build_token_sets(tok, P_F1.text, {"bell": ["Graham"], "meucci": ["Meucci"]})
    case = resolve_shared_prefixes(tok, P_F1.text, sets, max_depth=3)[0]
    assert {w["status"] for w in case["words"]} == {"next_shared"} and case["by_candidate"] == {}
