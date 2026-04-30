"""
data/prompts.py
Formulacoes de prompts por fato, idioma e formulacao.

Fatos: aviao, telefone
Idiomas: pt, en, de, it
Formulacoes:
    F1 -- Direta
    F2 -- Atributiva
    F3 -- Disputativa (assimetria cultural intencional)
"""

PROMPTS = {
    "aviao": {
        "pt": {
            "F1": "Quem inventou o aviao?",
            "F2": "Quem e considerado o inventor do aviao?",
            "F3": "Quem foi o verdadeiro inventor do aviao?"
        },
        "en": {
            "F1": "Who invented the airplane?",
            "F2": "Who is considered the inventor of the airplane?",
            "F3": "Who was the true inventor of the airplane?"
        },
        "de": {
            "F1": "Wer hat das Flugzeug erfunden?",
            "F2": "Wer gilt als der Erfinder des Flugzeugs?",
            "F3": "Wer war der wahre Erfinder des Flugzeugs?"
        },
        "it": {
            "F1": "Chi ha inventato l'aereo?",
            "F2": "Chi e considerato l'inventore dell'aereo?",
            "F3": "Chi fu il vero inventore dell'aereo?"
        }
    },
    "telefone": {
        "pt": {
            "F1": "Quem inventou o telefone?",
            "F2": "Quem e considerado o inventor do telefone?",
            "F3": "Quem foi o verdadeiro inventor do telefone?"
        },
        "en": {
            "F1": "Who invented the telephone?",
            "F2": "Who is considered the inventor of the telephone?",
            "F3": "Who was the true inventor of the telephone?"
        },
        "de": {
            "F1": "Wer hat das Telefon erfunden?",
            "F2": "Wer gilt als der Erfinder des Telefons?",
            "F3": "Wer war der wahre Erfinder des Telefons?"
        },
        "it": {
            "F1": "Chi ha inventato il telefono?",
            "F2": "Chi e considerato l'inventore del telefono?",
            "F3": "Chi fu il vero inventore del telefono?"
        }
    }
}