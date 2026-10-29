"""
tests/test_baseline_config.py
Fase 2: prompts neutros da linha de base (config/baseline_prompts.yaml).
"""

import copy
import dataclasses

import pytest
import yaml

from config.experiment_config import ExperimentConfig
from config.loader import (
    CONFIG_DIR, UnconfirmedError, load_baseline, load_experiment, load_facts,
    load_prompts, parse_baseline, prompts_for, require_confirmed, validate_baseline,
)
from data.text_matching import find_terms

EXP      = load_experiment()
PROMPTS  = load_prompts()
BASELINE = load_baseline()


def _raw():
    return yaml.safe_load((CONFIG_DIR / "baseline_prompts.yaml").read_text(encoding="utf-8"))


def test_real_baseline_is_valid_and_complete():
    assert validate_baseline(EXP, BASELINE, PROMPTS) == []
    types = EXP["name_types"] + EXP["indicator_types"]
    assert len(BASELINE.prompts) == len(EXP["languages"]) * len(types) * EXP["paraphrases"]
    assert all(p.fact is None for p in BASELINE.prompts)


def test_baseline_confirmed_fr_de_it_author_checked():
    # PT/EN aprovados; FR/DE/IT checados pela autora (não nativos), 2026-10-29
    assert BASELINE.status == "confirmed"
    assert not any(p.needs_native_review for p in BASELINE.prompts)
    raw = _raw()["baseline"]
    for lang in ("fr", "de", "it"):
        assert "not native-reviewed" in raw[lang]["review_note"]
    assert all("review_note" not in raw[l] for l in ("pt", "en"))


def test_mirrors_real_prompt_shapes():
    for b in BASELINE.prompts:
        real = [p for p in prompts_for(PROMPTS, None, b.lang, b.type)
                if p.paraphrase == b.paraphrase]
        assert real, (b.lang, b.type, b.paraphrase)
        assert {(p.slot_case, p.article_blocked) for p in real} == \
            {(b.slot_case, b.article_blocked)}


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r["baseline"]["it"]["F1"][0].pop("article_blocked"), "não espelha"),
    (lambda r: r["baseline"]["pt"]["F1"][0].pop("article_blocked"), "não espelha"),
    (lambda r: r["baseline"]["en"]["QA"].pop(), "paráfrases"),
    (lambda r: r["baseline"]["en"]["F1"][0].update(slot_case="gen"), "slot_case só em ru"),
    (lambda r: r["baseline"]["en"].update(needs_native_review=True), "needs_native_review"),
    (lambda r: r.update(status="ok"), "status"),
])
def test_validation_catches(mutate, msg):
    raw = copy.deepcopy(_raw())
    mutate(raw)
    errs = validate_baseline(EXP, parse_baseline(raw), PROMPTS)
    assert any(msg in e for e in errs), errs


def test_neutral_texts_mention_no_fact_or_candidate():
    facts = load_facts()
    for b in BASELINE.prompts:
        terms = {f.keyword[b.lang] for f in facts.values()}
        for f in facts.values():
            for c in f.candidates:
                terms |= {n for forms in c.names.get(b.lang, {}).values() for n in forms}
                for t in c.indicators.values():
                    terms |= set(t.get(b.lang, []))
        assert find_terms(b.text, sorted(terms)) == [], b.text


def test_require_confirmed_blocks_on_baseline():
    from conftest import reviewed_facts
    facts   = reviewed_facts(load_facts())
    prompts = [dataclasses.replace(p, needs_native_review=False) for p in PROMPTS]
    require_confirmed(facts, prompts)                       # sem linha de base: ok
    require_confirmed(facts, prompts, BASELINE)               # confirmada
    proposed = dataclasses.replace(BASELINE, status="proposed")
    with pytest.raises(UnconfirmedError, match="linha de base"):
        require_confirmed(facts, prompts, proposed)
    pending = dataclasses.replace(BASELINE, prompts=tuple(
        dataclasses.replace(p, needs_native_review=True) for p in BASELINE.prompts))
    with pytest.raises(UnconfirmedError, match="revisão nativa"):
        require_confirmed(facts, prompts, pending)


def test_config_baseline_prompt():
    cfg = ExperimentConfig()
    assert cfg.baseline_prompt("pt", "F1", 1).article_blocked
    assert cfg.baseline_prompt("en", "F1", 1).text == "The novel was written by"
    assert cfg.top_k_tokens == 20
