"""
tests/test_tokenization_check.py
Fase 1: verificação de tokenização de nomes e indicadores.
"""

import pytest

from config.loader import Candidate, Fact, Prompt, load_facts, load_prompts
from conftest import WordTokenizer
from data.tokenization_check import check_tokenizer, token_word, word_sets


def _fact():
    def cand(cid, dominant, names, place=None):
        return Candidate(id=cid, label=cid, dominant=dominant, stakes=[],
                         names={"it": {"nom": names}},
                         indicators={"place": {"it": place}} if place else {})
    return Fact(key="telefone", kind="disputed", status="proposed",
                verb="invented", keyword={"it": "telefono"},
                candidates=[cand("bell", True, ["Bell", "Graham"], ["Boston"]),
                            cand("meucci", False, ["Meucci", "Antonio"], ["Firenze"]),
                            cand("mendel", False, ["Mendel"])])


TOK = WordTokenizer(splits={" Meucci": [" Me", "ucci"], " Mendel": [" Me", "ndel"]})
ZIPF = {"bell": 2.0, "graham": 2.0, "me": 6.0, "antonio": 4.8, "boston": 3.0, "firenze": 4.1}


def _run(prompts, chat=None):
    return check_tokenizer("tiny", TOK, {"telefone": _fact()}, prompts, 4.0,
                           chat_formatter=chat, zipf=lambda w, l: ZIPF.get(w, 0.0))


def _row(res, word, ptype=None):
    return next(r for r in res["rows"] if r["word"] == word
                and (ptype is None or r["prompt_type"] == ptype))


P_NAME  = Prompt("telefone", "it", "F1", 1, "Il telefono è stato inventato da")
P_PLACE = Prompt("telefone", "it", "place", 1, "Il telefono è stato inventato a")
P_QA    = Prompt("telefone", "it", "QA", 1, "Chi ha inventato il telefono?")


def test_flags_split_shared_common():
    res = _run([P_NAME])
    meucci = _row(res, "Meucci")
    assert meucci["token"] == " Me" and meucci["split"]
    assert meucci["shared_with"] == "mendel"            # fragmento compartilhado
    assert _row(res, "Antonio")["common"]               # Zipf 4.8 >= 4.0
    assert not _row(res, "Bell")["common"]
    assert not _row(res, "Bell")["shared_with"]
    flags = {(i["candidate"], i["flag"]) for i in res["summary"]["issues"]}
    assert ("meucci", "shared") in flags and ("mendel", "shared") in flags
    assert res["summary"]["counts"]["split"] == 2


def test_indicator_prompts_measure_indicators_only():
    sets = word_sets(_fact(), P_PLACE)
    assert sets == {"bell": ["Boston"], "meucci": ["Firenze"]}   # mendel sem indicador
    res = _run([P_PLACE])
    assert {r["word"] for r in res["rows"]} == {"Boston", "Firenze"}
    assert _row(res, "Firenze")["common"]


def test_qa_uses_chat_format_without_space():
    # QA pulado sem chat template; com template, sem espaço e sem BOS extra
    assert _run([P_QA])["rows"] == []
    res = _run([P_QA], chat=lambda tok, text: f"<user>{text}<assistant>")
    bell = _row(res, "Bell", "QA")
    assert bell["format"] == "qa" and bell["token"] == "Bell"


def test_boundary_merge_flagged():
    tok = WordTokenizer(merges={(" da", " Bell"): " daBell"})
    res = check_tokenizer("tiny", tok, {"telefone": _fact()}, [P_NAME], 4.0,
                          zipf=lambda w, l: 0.0)
    assert _row(res, "Bell")["excluded"]
    assert res["summary"]["counts"]["excluded"] == 1


def test_token_word_strips_markers():
    assert token_word("▁Bell") == "Bell"
    assert token_word("ĠBell") == "Bell"
    assert token_word(" Me") == "Me"


def test_full_config_with_tiny_tokenizer():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    except Exception as e:
        pytest.skip(f"tokenizer indisponível: {e}")
    facts = load_facts()
    prompts = [p for p in load_prompts() if p.fact == "radio" and p.lang == "pt"]
    res = check_tokenizer("tiny", tok, facts, prompts, 4.0)
    shared = {(i["candidate"], i["word"]) for i in res["summary"]["issues"]
              if i["flag"] == "shared"}
    # São Paulo (Landell) e São Petersburgo (Popov) compartilham " São"
    assert ("landell", "São Paulo") in shared and ("popov", "São Petersburgo") in shared
