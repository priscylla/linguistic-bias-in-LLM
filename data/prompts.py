"""
data/prompts.py
Prompts de completion por fato × idioma × formulação.
F1 direta, F2 atributiva, F3 disputativa ("verdadeiro inventor").
Texto com acentos corretos (confirmados em reports/verified_values.md).
"""

COMPLETION_PROMPTS = {
    "aviao": {
        "pt": {
            "F1": "O avião foi inventado por",
            "F2": "O inventor do avião é considerado",
            "F3": "O verdadeiro inventor do avião é"
        },
        "en": {
            "F1": "The airplane was invented by",
            "F2": "The inventor of the airplane is considered to be",
            "F3": "The true inventor of the airplane was"
        },
        "de": {
            "F1": "Das Flugzeug wurde erfunden von",
            "F2": "Der Erfinder des Flugzeugs gilt als",
            "F3": "Der wahre Erfinder des Flugzeugs war"
        },
        "it": {
            "F1": "L'aereo è stato inventato da",
            "F2": "L'inventore dell'aereo è considerato",
            "F3": "Il vero inventore dell'aereo è stato"
        }
    },
    "telefone": {
        "pt": {
            "F1": "O telefone foi inventado por",
            "F2": "O inventor do telefone é considerado",
            "F3": "O verdadeiro inventor do telefone é"
        },
        "en": {
            "F1": "The telephone was invented by",
            "F2": "The inventor of the telephone is considered to be",
            "F3": "The true inventor of the telephone was"
        },
        "de": {
            "F1": "Das Telefon wurde erfunden von",
            "F2": "Der Erfinder des Telefons gilt als",
            "F3": "Der wahre Erfinder des Telefons war"
        },
        "it": {
            "F1": "Il telefono è stato inventato da",
            "F2": "L'inventore del telefono è considerato",
            "F3": "Il vero inventore del telefono è stato"
        }
    }
}