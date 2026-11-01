"""
tests/test_indicator_spellings.py
Uma linha por indicador: grafias que só diferem em caixa ou acento são o
mesmo indicador (a busca ignora os dois). A busca não muda.
"""

from behavior.extraction import Extractor, dedupe_spellings
from config.experiment_config import ExperimentConfig
from config.loader import load_behavioral
from data.text_matching import normalize_text

CFG = ExperimentConfig()


def test_dedupe_prefers_english_then_language_order():
    by_lang = {"pt": ["cinetoscópio"], "en": ["Kinetoscope"], "de": ["Kinetoskop"],
               "it": ["cinetoscopio"], "fr": ["kinétoscope"]}
    assert dedupe_spellings(by_lang) == ["Kinetoscope", "Kinetoskop", "cinetoscópio"]
    assert dedupe_spellings({"fr": ["Hanôver"], "en": ["Hanover"]}) == ["Hanover"]


def test_no_duplicate_indicators_in_config():
    ex = Extractor(CFG.fact_specs, load_behavioral(), CFG.languages)
    for fact, rows in ex.ind.items():
        keys = [(c, t, normalize_text(w)) for c, t, w, _ in rows]
        assert len(keys) == len(set(keys)), fact


def test_matching_unchanged():
    ex = Extractor(CFG.fact_specs, load_behavioral(), CFG.languages)
    hits = [r for r in ex.indicators("Il présente le kinétoscope en 1891.", "cinema") if r["mentioned"]]
    assert [(r["candidate"], r["indicator"]) for r in hits] == [("edison_cinema", "Kinetoscope")]
    hits = [r for r in ex.indicators("O cinematografo dos Lumière.", "cinema") if r["mentioned"]]
    assert [(r["candidate"], r["indicator"]) for r in hits] == [("lumiere", "cinematógrafo")]
