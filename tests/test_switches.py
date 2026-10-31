"""
tests/test_switches.py
Troca de escrita e de idioma nas respostas (behavior/switches.py) e a
tabela descritiva behavioral_switches (analyze.switch_table).
"""

import pandas as pd

import analyze
from behavior.switches import dominant_language, scripts, switches


def test_scripts():
    assert scripts("O cálculo infinitesimal, um campo da matemática que estuda as taxas de変量，请您确认") == {"Han": 2}
    assert scripts("Изобретатель радио: Попов") == {"Cyrillic": 3}
    assert scripts("Δx → 0, π ≈ 3,14 e Röntgen") == {}           # grego e acentos não contam


def test_dominant_language():
    assert dominant_language("O avião foi inventado por Santos-Dumont, que também é considerado o pai da aviação.") == "pt"
    assert dominant_language("The airplane was invented by the Wright brothers, who are considered the first.") == "en"
    assert dominant_language("Das Flugzeug wurde von den Brüdern Wright erfunden und gilt als das erste.") == "de"
    assert dominant_language("Santos-Dumont.") is None                # curta: indeterminado
    # palavras ambíguas entre idiomas (il, le, sua, das) não decidem nada
    assert dominant_language("Il avait abandonné ce plateau car il pensait qu'il était contaminé.") in (None, "fr")


def test_switches_whole_and_partial():
    pt_en = ("Ele recebeu o primeiro patent. Despite this, Bell is widely credited with the invention "
             "due to the robustness of his patent and his successful demonstration of the device.")
    r = switches(pt_en, "pt")
    assert r["partial_language_switch"] and r["other_sentence_langs"] == "en"
    r = switches("The telephone was invented by Bell, who is considered the first and was famous.", "pt")
    assert r["language_switch"] and r["answer_lang"] == "en"
    r = switches("O telefone foi inventado por Bell, que também é considerado o primeiro.", "pt")
    assert not r["language_switch"] and not r["partial_language_switch"] and not r["script_switch"]


def test_switch_table():
    k = dict(paraphrase=1, kind="sample")
    ans = pd.DataFrame([
        {"model": "qwen", "fact": "calculo", "lang": "pt", "sample_id": 1, "text": "Foi Newton.", **k},
        {"model": "qwen", "fact": "calculo", "lang": "pt", "sample_id": 2,
         "text": "O cálculo é estudado: 微积分学是由多位数学家", **k},
        {"model": "qwen", "fact": "calculo", "lang": "pt", "sample_id": 3,
         "text": "Calculus was invented by Newton, who is considered the first and was famous.", **k}])
    attr = ans[["model", "fact", "lang", "paraphrase", "sample_id", "kind"]].assign(
        primary=["newton", "none", "newton"])
    T = analyze.switch_table(ans, attr).iloc[0]
    assert (T.n_answers, T.n_script_switch, T.n_language_switch) == (3, 1, 1)
    assert (T.n_no_candidate, T.n_no_candidate_script_switch) == (1, 1)
    assert T.scripts == "Han" and T.switched_to == "en"
