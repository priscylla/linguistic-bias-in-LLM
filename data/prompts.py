"""
data/prompts.py

Dois conjuntos de prompts com finalidades distintas:

PROMPTS (chat-style)
    Usados em get_final_response() para obter a resposta
    comportamental do modelo. Usa chat template completo.
    Serve para a analise comportamental (qual entidade o
    modelo cita na resposta).

COMPLETION_PROMPTS (completion-style)
    Usados em extractor.extract() para o Logit Lens.
    Sem chat template — fed diretamente como string.
    O ultimo token e uma preposicao que forca o modelo
    a prever um nome proprio na proxima posicao.

    Estrutura: "[Fato] foi inventado por"
    Ex: "The airplane was invented by"
         → last token = " by"
         → Logit Lens prevê: "Wright", "Santos", "the", ...

    Isso replica a metodologia do TalkTuner (2024), que usa
    "I think the {attribute} of this user is" como ultimo
    token para leitura de atributos internos.

Formulacoes (F1, F2, F3) aplicadas apenas aos PROMPTS.
COMPLETION_PROMPTS tem uma unica formulacao por idioma/fato.
"""

# ── Chat-style — para analise comportamental ──────────────────
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


# ── Completion-style — para Logit Lens ───────────────────────
# Formato: "[Sujeito] foi inventado por"
# Ultimo token: preposicao ("by", "por", "von", "da")
# → modelo prevê diretamente o nome do inventor
#
# Escolha das preposicoes:
#   EN: "by"  — preposicao de agente passivo
#   PT: "por" — equivalente
#   DE: "von" — equivalente
#   IT: "da"  — equivalente
#
# Nao usamos formulacoes F1/F2/F3 aqui — o objetivo e
# uma unica sonda semantica limpa por idioma.

COMPLETION_PROMPTS = {
    "aviao": {
        "pt": "O aviao foi inventado por",
        "en": "The airplane was invented by",
        "de": "Das Flugzeug wurde erfunden von",
        "it": "L'aereo e stato inventato da"
    },
    "telefone": {
        "pt": "O telefone foi inventado por",
        "en": "The telephone was invented by",
        "de": "Das Telefon wurde erfunden von",
        "it": "Il telefono e stato inventato da"
    }
}