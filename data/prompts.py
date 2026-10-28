"""
data/prompts.py
Prompts de completion por fato × idioma × formulação.

A fonte é config/prompts.yaml (3 paráfrases por tipo). Este módulo
expõe COMPLETION_PROMPTS (paráfrase 1 de F1–F3), usado pelo pipeline
atual e por verify_setup.py.
"""

from config.loader import load_experiment, load_prompts

_FORMS = load_experiment()["formulations"]

COMPLETION_PROMPTS = {}
for _p in load_prompts():
    if _p.type in _FORMS and _p.paraphrase == 1:
        COMPLETION_PROMPTS.setdefault(_p.fact, {}).setdefault(_p.lang, {})[_p.type] = _p.text
