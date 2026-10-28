"""
tests/test_config_phase1.py
Fase 1: arquivos de configuração (fatos, stakes, nomes, indicadores,
prompts) e suas regras.
"""

import copy

import pytest
import yaml

from config.experiment_config import ExperimentConfig
from config.loader import (
    CONFIG_DIR, UnconfirmedError, load_experiment, load_facts, load_prompts,
    parse_facts, parse_prompts, prompts_for, require_confirmed, validate,
)

EXP     = load_experiment()
FACTS   = load_facts()
PROMPTS = load_prompts()
LANGS   = EXP["languages"]


def _raw(name):
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


# ── Estrutura e validação ──

def test_real_config_is_valid():
    assert validate(EXP, FACTS, PROMPTS) == []


def test_languages_and_facts():
    assert LANGS == ["pt", "en", "de", "it", "fr", "ru"]
    disputed = [k for k, f in FACTS.items() if f.kind == "disputed"]
    controls = [k for k, f in FACTS.items() if f.kind == "control"]
    assert disputed == ["aviao", "telefone", "radio", "lampada", "cinema",
                        "calculo", "fotografia", "tabela_periodica"]
    assert controls == ["penicilina", "dinamite", "www", "raios_x"]


def test_dominant_candidates():
    dom = {k: f.dominant.id for k, f in FACTS.items()}
    assert dom == {"aviao": "wright", "telefone": "bell", "radio": "marconi",
                   "lampada": "edison", "cinema": "lumiere", "calculo": "newton",
                   "fotografia": "daguerre", "tabela_periodica": "mendeleev",
                   "penicilina": "fleming", "dinamite": "nobel",
                   "www": "berners_lee", "raios_x": "rontgen"}
    assert [c.id for c in FACTS["fotografia"].candidates] == \
        ["daguerre", "niepce", "talbot", "florence"]
    assert all(f.empirical_dominant is None for f in FACTS.values())


def test_stakes():
    def stakes(fact, cid):
        return {(s.language, s.type) for s in FACTS[fact].candidate(cid).stakes}
    assert stakes("aviao", "santos_dumont") == {("pt", "nationality"), ("fr", "place_of_event")}
    assert stakes("telefone", "reis") == {("de", "nationality")}
    assert stakes("radio", "marconi") == {("it", "nationality")}
    assert stakes("tabela_periodica", "mendeleev") == {("ru", "nationality")}
    # regra de nacionalidade para todos os candidatos (decisão da autora)
    assert stakes("aviao", "wright") == {("en", "nationality")}
    assert stakes("raios_x", "rontgen") == {("de", "nationality")}
    assert stakes("dinamite", "nobel") == set()
    # place_of_event só onde o plano lista
    poe = [(k, c.id) for k, f in FACTS.items() for c in f.candidates
           if any(s.type == "place_of_event" for s in c.stakes)]
    assert poe == [("aviao", "santos_dumont")]


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r["facts"]["aviao"]["candidates"][1].update(dominant=True), "dominant"),
    (lambda r: r["facts"]["aviao"]["candidates"][0]["names"]["ru"].pop("ins"), "casos"),
    (lambda r: r["facts"]["aviao"]["candidates"][0]["names"].pop("fr"), "sem nomes"),
    (lambda r: r["facts"]["aviao"]["candidates"][0]["stakes"].append(
        {"language": "es", "type": "nationality"}), "fora do escopo"),
    (lambda r: r["facts"]["aviao"]["candidates"][0]["stakes"].append(
        {"language": "pt", "type": "birthplace"}), "tipo de stake"),
])
def test_validation_catches_fact_errors(mutate, msg):
    raw = copy.deepcopy(_raw("facts.yaml"))
    mutate(raw)
    errs = validate(EXP, parse_facts(raw), PROMPTS)
    assert any(msg in e for e in errs), errs


def test_validation_catches_prompt_errors():
    raw = copy.deepcopy(_raw("prompts.yaml"))
    raw["prompts"]["aviao"]["en"]["F2"].pop()
    raw["prompts"]["lampada"]["en"]["artifact"] = [{"text": "x"}] * 3
    errs = validate(EXP, FACTS, parse_prompts(raw))
    assert any("aviao.en.F2: 2 paráfrases" in e for e in errs)
    assert any("lampada" in e and "artifact" in e for e in errs)


# ── Nomes: só pessoas ──

GENERIC = {"brothers", "brüder", "fratelli", "frères", "irmãos", "братья", "kitty"}


def test_name_sets_have_no_places_or_generic_words():
    for k, f in FACTS.items():
        places = {w.casefold() for c in f.candidates
                  for by_lang in c.indicators.values()
                  for ws in by_lang.values() for w in ws}
        for c in f.candidates:
            for lang, by_case in c.names.items():
                for forms in by_case.values():
                    for n in forms:
                        assert n.casefold() not in GENERIC, (k, c.id, n)
                        assert n.casefold() not in places, (k, c.id, n)


def test_plan_names_present():
    def names(f, c, l="en"):
        return FACTS[f].candidate(c).names_for(l)
    assert {"Santos", "Dumont"} <= set(names("aviao", "santos_dumont"))
    assert {"Wright", "Orville", "Wilbur"} <= set(names("aviao", "wright"))
    assert {"Bell", "Alexander", "Graham"} <= set(names("telefone", "bell"))
    assert {"Meucci", "Antonio"} <= set(names("telefone", "meucci"))
    assert {"Reis", "Philipp"} <= set(names("telefone", "reis"))


def test_ru_case_forms():
    bell = FACTS["telefone"].candidate("bell")
    assert bell.names_for("ru", "nom")[0] == "Белл"
    assert bell.names_for("ru", "ins")[0] == "Беллом"
    assert bell.names_for("pt", "ins") == bell.names_for("pt")   # sem declinação


def test_indicators_follow_plan():
    w = FACTS["aviao"].candidate("wright")
    assert "Kitty Hawk" in w.indicators_for("place", "en")
    assert w.indicators_for("artifact", "en") == ["Flyer"]
    assert FACTS["aviao"].candidate("santos_dumont").indicators_for("artifact", "pt") == \
        ["14-bis", "Demoiselle"]
    assert FACTS["lampada"].indicator_types() == ["place"]
    assert all(not f.indicator_types() for f in FACTS.values() if f.kind == "control")


# ── Prompts ──

OLD_P1 = {
    ("aviao", "pt"): ["O avião foi inventado por", "O inventor do avião é considerado", "O verdadeiro inventor do avião é"],
    ("aviao", "en"): ["The airplane was invented by", "The inventor of the airplane is considered to be", "The true inventor of the airplane was"],
    ("aviao", "de"): ["Das Flugzeug wurde erfunden von", "Der Erfinder des Flugzeugs gilt als", "Der wahre Erfinder des Flugzeugs war"],
    ("aviao", "it"): ["L'aereo è stato inventato da", "L'inventore dell'aereo è considerato", "Il vero inventore dell'aereo è stato"],
    ("telefone", "pt"): ["O telefone foi inventado por", "O inventor do telefone é considerado", "O verdadeiro inventor do telefone é"],
    ("telefone", "en"): ["The telephone was invented by", "The inventor of the telephone is considered to be", "The true inventor of the telephone was"],
    ("telefone", "de"): ["Das Telefon wurde erfunden von", "Der Erfinder des Telefons gilt als", "Der wahre Erfinder des Telefons war"],
    ("telefone", "it"): ["Il telefono è stato inventato da", "L'inventore del telefono è considerato", "Il vero inventore del telefono è stato"],
}


@pytest.mark.parametrize("key", list(OLD_P1))
def test_old_prompts_are_paraphrase_1(key):
    fact, lang = key
    got = [prompts_for(PROMPTS, fact, lang, f)[0].text for f in ("F1", "F2", "F3")]
    assert got == OLD_P1[key]


def test_prompt_types_per_fact():
    for k, f in FACTS.items():
        types = {p.type for p in prompts_for(PROMPTS, fact=k)}
        if f.kind == "control":
            assert types == {"F1", "F2", "F3", "QA"}
        else:
            assert {"F1", "F2", "F3", "QA", "place"} <= types
    assert "artifact" not in {p.type for p in prompts_for(PROMPTS, fact="lampada")}


def test_native_review_flags():
    for p in PROMPTS:
        assert p.needs_native_review == (p.lang in {"fr", "ru", "de", "it"})


def test_article_blocked_flags():
    for k in FACTS:
        f1 = prompts_for(PROMPTS, k, "pt", "F1")
        assert [p.article_blocked for p in f1] == [True, False, False]
        assert prompts_for(PROMPTS, k, "it", "F1")[0].article_blocked
    assert not any(p.article_blocked for p in PROMPTS if p.lang in ("en", "de", "ru"))


def test_ru_slot_cases():
    f1 = prompts_for(PROMPTS, "aviao", "ru", "F1")
    assert [p.slot_case for p in f1] == ["ins", "gen", "ins"]
    assert all(p.slot_case == "nom" for p in PROMPTS
               if p.lang == "ru" and p.type != "F1")


def test_qa_prompts_are_questions():
    for p in PROMPTS:
        if p.type == "QA":
            assert p.text.endswith("?")
            assert p.format == "qa"


def test_no_dates_in_prompts():
    import re
    assert not any(re.search(r"\b1[5-9]\d\d\b", p.text) for p in PROMPTS)


# ── Bloqueio de runs sem confirmação ──

def test_runs_blocked_until_confirmed():
    with pytest.raises(UnconfirmedError):
        require_confirmed(FACTS, PROMPTS)


def test_require_confirmed_passes_when_all_confirmed():
    raw = copy.deepcopy(_raw("facts.yaml"))
    for f in raw["facts"].values():
        f["status"] = "confirmed"
    prompts = [p for p in PROMPTS if not p.needs_native_review]
    require_confirmed(parse_facts(raw), prompts)


def test_run_lens_refuses_unconfirmed(monkeypatch):
    import models.model_loader
    import run_lens
    monkeypatch.setattr(models.model_loader, "load_model",
                        lambda *a, **k: pytest.fail("não deveria carregar o modelo"))
    with pytest.raises(UnconfirmedError):
        run_lens.main(["--model", "llama"])


# ── Derivados ──

def test_names_and_stake_languages():
    cfg = ExperimentConfig()
    tel = cfg.fact_specs["telefone"]
    assert tel.dominant.names_for("de") == ["Bell", "Alexander", "Graham"]
    assert {n for c in tel.alternatives for n in c.names_for("de")} == \
        {"Meucci", "Antonio", "Reis", "Johann", "Philipp"}
    assert cfg.fact_specs["aviao"].dominant.names_for("ru", "ins")[0] == "Райтами"
    assert cfg.stake_languages["telefone"] == ["de", "it"]
    assert cfg.stake_languages["aviao"] == ["fr", "pt"]
    assert cfg.stake_languages["penicilina"] == []


def test_draft_script_refuses_to_overwrite(monkeypatch):
    import sys
    from scripts import draft_prompts
    monkeypatch.setattr(sys, "argv", ["draft_prompts.py"])
    assert draft_prompts.main() == 1
