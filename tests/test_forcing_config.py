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
    assert FORCING.status == "proposed"


def test_values():
    assert FORCING.contraction_forms("pt", "por") == ["pelo", "pela", "pelos", "pelas"]
    assert FORCING.contraction_forms("pt", "em")[0] == "no"      # não vira False
    assert FORCING.plural_for("ru", "ins") == ["братьями"]
    assert FORCING.plural_for("pt", "ins") == ["irmãos"]          # sem casos: nominativo
    assert FORCING.articles["ru"] == []


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r["contractions"]["pt"].pop("por"), "sem formas para 'por'"),
    (lambda r: r["contractions"]["pt"].update(em=[False, "na"]), "não é texto"),
    (lambda r: r["articles"].pop("de"), "articles: sem 'de'"),
    (lambda r: r["plural_nouns"]["ru"].pop("gen"), "casos"),
    (lambda r: r.update(status="draft"), "status"),
])
def test_validation_catches(mutate, msg):
    raw = copy.deepcopy(_raw())
    mutate(raw)
    errs = validate_forcing(EXP, parse_forcing(raw), PROMPTS)
    assert any(msg in e for e in errs), errs


def test_require_confirmed_blocks_on_forcing():
    facts   = {k: dataclasses.replace(f, status="confirmed") for k, f in load_facts().items()}
    prompts = [dataclasses.replace(p, needs_native_review=False) for p in load_prompts()]
    with pytest.raises(UnconfirmedError, match="forcing"):
        require_confirmed(facts, prompts, forcing=FORCING)
    require_confirmed(facts, prompts, forcing=dataclasses.replace(FORCING, status="confirmed"))
