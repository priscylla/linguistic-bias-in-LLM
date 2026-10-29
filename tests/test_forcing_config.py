"""
tests/test_forcing_config.py
Fase 2: palavras do teste do confundidor de artigo (config/forcing.yaml).
"""

import copy
import dataclasses

import pytest
import yaml

from config.loader import (
    CONFIG_DIR, UnconfirmedError, load_baseline, load_experiment, load_facts,
    load_forcing, load_prompts, parse_forcing, require_confirmed, validate_forcing,
)

EXP     = load_experiment()
FORCING = load_forcing()
PROMPTS = list(load_prompts()) + list(load_baseline().prompts)


def _raw():
    return yaml.safe_load((CONFIG_DIR / "forcing.yaml").read_text(encoding="utf-8"))


def test_real_forcing_is_valid():
    assert validate_forcing(EXP, FORCING, PROMPTS) == []
    assert FORCING.status == "confirmed"
    # acréscimos da autora (2026-10-29)
    assert "Gebrüder" in FORCING.plural_for("de")
    assert FORCING.contraction_forms("de", "von") == ["vom"]
    assert FORCING.contraction_forms("de", "in") == ["im"]
    assert FORCING.contraction_forms("it", "in") == ["nel", "nello", "nella", "nei", "negli", "nelle"]
    assert FORCING.articles["it"][-2:] == ["l'", "un'"] and FORCING.articles["fr"][-1] == "l'"


def test_values():
    assert FORCING.contraction_forms("pt", "por") == ["pelo", "pela", "pelos", "pelas"]
    assert FORCING.contraction_forms("pt", "em")[0] == "no"      # não vira False
    assert FORCING.plural_for("pt", "ins") == ["irmãos"]          # sem casos: nominativo
    assert set(FORCING.articles) == {"pt", "en", "de", "it", "fr"}      # ru removido


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r["contractions"]["pt"].pop("por"), "sem formas para 'por'"),
    (lambda r: r["contractions"]["pt"].update(em=[False, "na"]), "não é texto"),
    (lambda r: r["articles"].pop("de"), "articles: sem 'de'"),
    (lambda r: r.update(status="draft"), "status"),
])
def test_validation_catches(mutate, msg):
    raw = copy.deepcopy(_raw())
    mutate(raw)
    errs = validate_forcing(EXP, parse_forcing(raw), PROMPTS)
    assert any(msg in e for e in errs), errs


def test_require_confirmed_blocks_on_forcing():
    from conftest import reviewed_facts
    facts   = reviewed_facts(load_facts())
    prompts = [dataclasses.replace(p, needs_native_review=False) for p in load_prompts()]
    require_confirmed(facts, prompts, forcing=FORCING)
    with pytest.raises(UnconfirmedError, match="forcing"):
        require_confirmed(facts, prompts, forcing=dataclasses.replace(FORCING, status="proposed"))
