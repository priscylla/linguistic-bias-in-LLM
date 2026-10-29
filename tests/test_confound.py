"""
tests/test_confound.py
Fase 2: passadas forçadas do teste do confundidor de artigo (artigo,
substantivo plural, contrações, prefixo compartilhado de indicadores).
"""

import pytest

from config.loader import Candidate, Fact, Prompt, load_forcing
from conftest import WordTokenizer, build_tiny_model
from logit_lens.confound import ConfoundTester, match_trigger
from logit_lens.entity_tokens import encode_ids, first_token_after
from logit_lens.measure import LensRunner

FORCING = load_forcing()
CHAT    = lambda tok, text: f"<user>{text}<assistant>"  # noqa: E731


def _cand(cid, dominant, names, place=None):
    return Candidate(id=cid, label=cid, dominant=dominant, stakes=[], names=names,
                     indicators={"place": place} if place else {}, plural=(cid == "wright"))


def _fact():
    return Fact(key="aviao", kind="disputed", status="confirmed", verb="invented",
                keyword={"pt": "avião", "en": "airplane"},
                candidates=[
                    _cand("wright", True,
                          {"pt": {"nom": ["Wright", "Orville"]}, "en": {"nom": ["Wright"]}},
                          {"pt": ["Dayton", "São Tomé"]}),
                    _cand("santos_dumont", False,
                          {"pt": {"nom": ["Santos", "Dumont"]}, "en": {"nom": ["Santos"]}},
                          {"pt": ["São Paulo", "Paris"]}),
                ])


def _setup(chat=None):
    model  = build_tiny_model("llama", vocab=512)
    runner = LensRunner(model, WordTokenizer(), chat_formatter=chat)
    return runner, ConfoundTester(runner, FORCING)


def _main_with_top(runner, prompt, word, sep=" ", add_special=True):
    """Resultado principal real, com o top-1 final trocado por `word`."""
    res = runner.measure(prompt, _fact())
    text, _, _ = runner.prepare(prompt)
    tid = first_token_after(runner.tokenizer, text, word, sep=sep,
                            add_special_tokens=add_special)["token_id"]
    res["layers"][-1]["top_ids"] = [tid] + res["layers"][-1]["top_ids"][1:]
    return res


P_PT2   = Prompt("aviao", "pt", "F1", 2, "Quem inventou o avião foi")
P_PT1   = Prompt("aviao", "pt", "F1", 1, "O avião foi inventado por", article_blocked=True)
P_QA    = Prompt("aviao", "en", "QA", 1, "Who invented the airplane?")
P_PLACE = Prompt("aviao", "pt", "place", 3, "O lugar onde o avião foi inventado é")


def test_match_trigger_exact_id():
    tok = WordTokenizer()
    text = "Quem inventou o avião foi"
    os_id = first_token_after(tok, text, "os")["token_id"]
    assert match_trigger(tok, text, " ", True, os_id, ["o", "os"]) == "os"
    assert match_trigger(tok, text, " ", True, os_id, ["o", "a"]) is None


def test_article_pass_forces_article_and_measures_next_position():
    runner, tester = _setup()
    main = _main_with_top(runner, P_PT2, "os")
    ps   = tester.run(P_PT2, _fact(), main)
    assert [p["pass_id"] for p in ps] == ["article:os"]
    st = ps[0]["steps"][0]
    assert st["forced_text"] == "Quem inventou o avião foi os" and st["trigger_token"] == "os"
    ref = runner.measure_words(st["forced_text"], " ", True,
                               {"wright": ["Wright", "Orville"], "santos_dumont": ["Santos", "Dumont"]},
                               "names")
    assert [r["prob"] for r in st["result"]["probs"]] == pytest.approx(
        [r["prob"] for r in ref["probs"]], abs=1e-7)


def test_no_article_no_pass():
    runner, tester = _setup()
    main = _main_with_top(runner, P_PT2, "Santos")
    assert tester.run(P_PT2, _fact(), main) == []


def test_plural_noun_second_step(monkeypatch):
    runner, tester = _setup()
    orig = runner.measure_words

    def fake(text, *a, **k):
        res = orig(text, *a, **k)
        if text.endswith(" os"):        # depois de "os", o top-1 vira "irmãos"
            tid = first_token_after(runner.tokenizer, text, "irmãos")["token_id"]
            res["layers"][-1]["top_ids"] = [tid] + res["layers"][-1]["top_ids"][1:]
        return res

    monkeypatch.setattr(runner, "measure_words", fake)
    ps = tester.run(P_PT2, _fact(), _main_with_top(runner, P_PT2, "os"))
    steps = ps[0]["steps"]
    assert [s["step"] for s in steps] == [1, 2]
    assert steps[1]["forced_text"] == "Quem inventou o avião foi os irmãos"
    assert steps[1]["suffix"] == " os irmãos"


def test_qa_article_capitalized_without_space():
    runner, tester = _setup(chat=CHAT)
    chat = CHAT(None, P_QA.text)
    main = _main_with_top(runner, P_QA, "The", sep="", add_special=False)
    ps   = tester.run(P_QA, _fact(), main)
    st   = ps[0]["steps"][0]
    assert ps[0]["pass_id"] == "article:The"
    assert st["forced_text"] == chat + "The"
    words = st["result"]["token_sets"]["sets"]["wright"]["words"]
    assert words[0]["token"] == " Wright"                 # palavra seguinte com espaço
    assert st["result"]["n_input_tokens"] == len(
        encode_ids(runner.tokenizer, chat + "The", add_special_tokens=False))


def test_contraction_passes_for_blocked_prompts():
    runner, tester = _setup()
    ps = tester.contraction_passes(P_PT1, _fact(), P_PT1.text, True)
    assert [p["pass_id"] for p in ps] == [f"contraction:{f}" for f in ("pelo", "pela", "pelos", "pelas")]
    assert ps[2]["steps"][0]["forced_text"] == "O avião foi inventado pelos"
    assert ps[2]["steps"][0]["suffix"] == "por→pelos"
    assert tester.contraction_passes(P_PT2, _fact(), P_PT2.text, True) == []


def test_shared_prefix_combined_in_main_measure_no_separate_pass():
    # " São" (São Tomé / São Paulo): resolvido na própria medição; a
    # passada separada "shared_prefix" não existe mais
    runner, tester = _setup()
    main = runner.measure(P_PLACE, _fact())
    case = main["prefix_cases"][0]
    assert case["token"] == " São"
    assert case["by_candidate"] == {"wright": [runner.tokenizer.vocab[" Tomé"]],
                                    "santos_dumont": [runner.tokenizer.vocab[" Paulo"]]}
    assert all(r["prob_prefix"] > 0 for r in main["probs"])
    assert all(p["kind"] != "shared_prefix" for p in tester.run(P_PLACE, _fact(), main))
