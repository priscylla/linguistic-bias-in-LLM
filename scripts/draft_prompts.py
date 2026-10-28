"""
scripts/draft_prompts.py
Rascunho de config/prompts.yaml a partir de modelos por idioma.

Ferramenta de redação, rodada uma vez: o arquivo gerado (strings
explícitas) é o que os revisores editam. O script se recusa a
sobrescrever config/prompts.yaml (use --force só para regenerar um
rascunho ainda não revisado).

    python scripts/draft_prompts.py [--force]

Regras de redação:
  - paráfrase 1 de F1/F2/F3 = prompt antigo (corrigido) para aviao e
    telefone em pt/en/de/it (verificado pelos testes);
  - article_blocked: true quando a lacuna termina em preposição que
    contrai com artigo (pt por/em, it da/a, fr à) e, portanto, um
    candidato com artigo ("os irmãos Wright", "la Caroline du Nord")
    não pode seguir; as paráfrases 2 e 3 evitam isso sempre que possível;
  - ru: lacunas nominativas sempre que possível; slot_case indica o
    caso exigido (nom, gen, ins);
  - verbo natural por fato: invented / developed / discovered.
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "config" / "prompts.yaml"
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════
# Léxico por fato e idioma
# ══════════════════════════════════════════════════════════════

EN = {
    # np, be, pp, agent, verb, ger, demo (particípio de "first ___ in"), art
    "aviao":      dict(np="the airplane", be="was", pp="invented", agent="inventor", verb="invented", ger="inventing", demo="first flown", art="airplane"),
    "telefone":   dict(np="the telephone", be="was", pp="invented", agent="inventor", verb="invented", ger="inventing", demo="first demonstrated", art="telephone"),
    "radio":      dict(np="the radio", be="was", pp="invented", agent="inventor", verb="invented", ger="inventing", demo="first demonstrated", art="radio receiver"),
    "lampada":    dict(np="the light bulb", be="was", pp="invented", agent="inventor", verb="invented", ger="inventing", demo="first demonstrated", art=None),
    "cinema":     dict(np="cinema", be="was", pp="invented", agent="inventor", verb="invented", ger="inventing", demo="first demonstrated", art="motion picture apparatus"),
    "fotografia": dict(np="photography", be="was", pp="invented", agent="inventor", verb="invented", ger="inventing", demo="first demonstrated", art="photographic process"),
    "calculo":    dict(np="calculus", be="was", pp="developed", agent="creator", verb="developed", ger="developing", demo="first published", art="CALCULUS"),
    "tabela_periodica": dict(np="the periodic table", be="was", pp="developed", agent="creator", verb="developed", ger="developing", demo="first published", art=None),
    "penicilina": dict(np="penicillin", be="was", pp="discovered", agent="discoverer", verb="discovered", ger="discovering"),
    "dinamite":   dict(np="dynamite", be="was", pp="invented", agent="inventor", verb="invented", ger="inventing"),
    "www":        dict(np="the World Wide Web", be="was", pp="invented", agent="inventor", verb="invented", ger="inventing"),
    "raios_x":    dict(np="X-rays", be="were", pp="discovered", agent="discoverer", verb="discovered", ger="discovering"),
}

PT = {
    # Np, np, de (genitivo), aux, pp (concordado), agent, verb, nomz, demo
    "aviao":      dict(Np="O avião", np="o avião", de="do avião", aux="foi", pp="inventado", agent="inventor", verb="inventou", nomz="a invenção", demo="testado", art="avião"),
    "telefone":   dict(Np="O telefone", np="o telefone", de="do telefone", aux="foi", pp="inventado", agent="inventor", verb="inventou", nomz="a invenção", demo="demonstrado", art="telefone"),
    "radio":      dict(Np="O rádio", np="o rádio", de="do rádio", aux="foi", pp="inventado", agent="inventor", verb="inventou", nomz="a invenção", demo="demonstrado", art="receptor de rádio"),
    "lampada":    dict(Np="A lâmpada elétrica", np="a lâmpada elétrica", de="da lâmpada elétrica", aux="foi", pp="inventada", agent="inventor", verb="inventou", nomz="a invenção", demo="demonstrada", art=None),
    "cinema":     dict(Np="O cinema", np="o cinema", de="do cinema", aux="foi", pp="inventado", agent="inventor", verb="inventou", nomz="a invenção", demo="demonstrado", art="aparelho de cinema"),
    "fotografia": dict(Np="A fotografia", np="a fotografia", de="da fotografia", aux="foi", pp="inventada", agent="inventor", verb="inventou", nomz="a invenção", demo="demonstrada", art="processo fotográfico"),
    "calculo":    dict(Np="O cálculo infinitesimal", np="o cálculo infinitesimal", de="do cálculo infinitesimal", aux="foi", pp="desenvolvido", agent="criador", verb="desenvolveu", nomz="o desenvolvimento", demo="publicado", art="CALCULUS"),
    "tabela_periodica": dict(Np="A tabela periódica", np="a tabela periódica", de="da tabela periódica", aux="foi", pp="desenvolvida", agent="criador", verb="desenvolveu", nomz="o desenvolvimento", demo="publicada", art=None),
    "penicilina": dict(Np="A penicilina", np="a penicilina", de="da penicilina", aux="foi", pp="descoberta", agent="descobridor", verb="descobriu", nomz="a descoberta"),
    "dinamite":   dict(Np="A dinamite", np="a dinamite", de="da dinamite", aux="foi", pp="inventada", agent="inventor", verb="inventou", nomz="a invenção"),
    "www":        dict(Np="A World Wide Web", np="a World Wide Web", de="da World Wide Web", aux="foi", pp="inventada", agent="inventor", verb="inventou", nomz="a invenção"),
    "raios_x":    dict(Np="Os raios X", np="os raios X", de="dos raios X", aux="foram", pp="descobertos", agent="descobridor", verb="descobriu", nomz="a descoberta"),
}

DE = {
    # Nom, nom (minúsc.), gen, aux, pp, agent, nomz, demo, art (Nom, gen)
    "aviao":      dict(Nom="Das Flugzeug", nom="das Flugzeug", gen="des Flugzeugs", aux="wurde", pp="erfunden", agent="Erfinder", nomz="die Erfindung", demo="erprobt", art=("Das erste Flugzeug", "des ersten Flugzeugs")),
    "telefone":   dict(Nom="Das Telefon", nom="das Telefon", gen="des Telefons", aux="wurde", pp="erfunden", agent="Erfinder", nomz="die Erfindung", demo="vorgeführt", art=("Das erste Telefon", "des ersten Telefons")),
    "radio":      dict(Nom="Das Radio", nom="das Radio", gen="des Radios", aux="wurde", pp="erfunden", agent="Erfinder", nomz="die Erfindung", demo="vorgeführt", art=("Der erste Radioempfänger", "des ersten Radioempfängers")),
    "lampada":    dict(Nom="Die Glühbirne", nom="die Glühbirne", gen="der Glühbirne", aux="wurde", pp="erfunden", agent="Erfinder", nomz="die Erfindung", demo="vorgeführt", art=None),
    "cinema":     dict(Nom="Das Kino", nom="das Kino", gen="des Kinos", aux="wurde", pp="erfunden", agent="Erfinder", nomz="die Erfindung", demo="vorgeführt", art=("Der erste Filmapparat", "des ersten Filmapparats")),
    "fotografia": dict(Nom="Die Fotografie", nom="die Fotografie", gen="der Fotografie", aux="wurde", pp="erfunden", agent="Erfinder", nomz="die Erfindung", demo="vorgeführt", art=("Das erste fotografische Verfahren", "des ersten fotografischen Verfahrens")),
    "calculo":    dict(Nom="Die Infinitesimalrechnung", nom="die Infinitesimalrechnung", gen="der Infinitesimalrechnung", aux="wurde", pp="entwickelt", agent="Begründer", nomz="die Entwicklung", demo="veröffentlicht", art="CALCULUS"),
    "tabela_periodica": dict(Nom="Das Periodensystem", nom="das Periodensystem", gen="des Periodensystems", aux="wurde", pp="entwickelt", agent="Begründer", nomz="die Entwicklung", demo="veröffentlicht", art=None),
    "penicilina": dict(Nom="Das Penicillin", nom="das Penicillin", gen="des Penicillins", aux="wurde", pp="entdeckt", agent="Entdecker", nomz="die Entdeckung"),
    "dinamite":   dict(Nom="Das Dynamit", nom="das Dynamit", gen="des Dynamits", aux="wurde", pp="erfunden", agent="Erfinder", nomz="die Erfindung"),
    "www":        dict(Nom="Das World Wide Web", nom="das World Wide Web", gen="des World Wide Web", aux="wurde", pp="erfunden", agent="Erfinder", nomz="die Erfindung"),
    "raios_x":    dict(Nom="Die X-Strahlen", nom="die X-Strahlen", gen="der X-Strahlen", aux="wurden", pp="entdeckt", agent="Entdecker", nomz="die Entdeckung"),
}

IT = {
    # Np, np, di, aux, pp, inf, agent_np, agent, verb (pass. prossimo), nomz, demo, art (Nom, gen)
    "aviao":      dict(Np="L'aereo", np="l'aereo", di="dell'aereo", aux="è stato", pp="inventato", inf="inventare", agent_np="L'inventore", agent="inventore", verb="ha inventato", nomz="l'invenzione", demo="collaudato", art=("Il primo aereo", "del primo aereo")),
    "telefone":   dict(Np="Il telefono", np="il telefono", di="del telefono", aux="è stato", pp="inventato", inf="inventare", agent_np="L'inventore", agent="inventore", verb="ha inventato", nomz="l'invenzione", demo="presentato", art=("Il primo telefono", "del primo telefono")),
    "radio":      dict(Np="La radio", np="la radio", di="della radio", aux="è stata", pp="inventata", inf="inventare", agent_np="L'inventore", agent="inventore", verb="ha inventato", nomz="l'invenzione", demo="presentata", art=("Il primo ricevitore radio", "del primo ricevitore radio")),
    "lampada":    dict(Np="La lampadina", np="la lampadina", di="della lampadina", aux="è stata", pp="inventata", inf="inventare", agent_np="L'inventore", agent="inventore", verb="ha inventato", nomz="l'invenzione", demo="presentata", art=None),
    "cinema":     dict(Np="Il cinema", np="il cinema", di="del cinema", aux="è stato", pp="inventato", inf="inventare", agent_np="L'inventore", agent="inventore", verb="ha inventato", nomz="l'invenzione", demo="presentato", art=("Il primo apparecchio cinematografico", "del primo apparecchio cinematografico")),
    "fotografia": dict(Np="La fotografia", np="la fotografia", di="della fotografia", aux="è stata", pp="inventata", inf="inventare", agent_np="L'inventore", agent="inventore", verb="ha inventato", nomz="l'invenzione", demo="presentata", art=("Il primo procedimento fotografico", "del primo procedimento fotografico")),
    "calculo":    dict(Np="Il calcolo infinitesimale", np="il calcolo infinitesimale", di="del calcolo infinitesimale", aux="è stato", pp="sviluppato", inf="sviluppare", agent_np="Il creatore", agent="creatore", verb="ha sviluppato", nomz="lo sviluppo", demo="pubblicato", art="CALCULUS"),
    "tabela_periodica": dict(Np="La tavola periodica", np="la tavola periodica", di="della tavola periodica", aux="è stata", pp="sviluppata", inf="sviluppare", agent_np="Il creatore", agent="creatore", verb="ha sviluppato", nomz="lo sviluppo", demo="pubblicata", art=None),
    "penicilina": dict(Np="La penicillina", np="la penicillina", di="della penicillina", aux="è stata", pp="scoperta", inf="scoprire", agent_np="Lo scopritore", agent="scopritore", verb="ha scoperto", nomz="la scoperta"),
    "dinamite":   dict(Np="La dinamite", np="la dinamite", di="della dinamite", aux="è stata", pp="inventata", inf="inventare", agent_np="L'inventore", agent="inventore", verb="ha inventato", nomz="l'invenzione"),
    "www":        dict(Np="Il World Wide Web", np="il World Wide Web", di="del World Wide Web", aux="è stato", pp="inventato", inf="inventare", agent_np="L'inventore", agent="inventore", verb="ha inventato", nomz="l'invenzione"),
    "raios_x":    dict(Np="I raggi X", np="i raggi X", di="dei raggi X", aux="sono stati", pp="scoperti", inf="scoprire", agent_np="Lo scopritore", agent="scopritore", verb="ha scoperto", nomz="la scoperta"),
}

FR = {
    # Nom, nom, de, aux, pp, agent_np, agent, verb, nomz, demo, art (Nom, gen)
    "aviao":      dict(Nom="L'avion", nom="l'avion", de="de l'avion", aux="a été", pp="inventé", agent_np="l'inventeur", agent="inventeur", verb="a inventé", nomz="l'invention", demo="testé", art=("Le premier avion", "du premier avion")),
    "telefone":   dict(Nom="Le téléphone", nom="le téléphone", de="du téléphone", aux="a été", pp="inventé", agent_np="l'inventeur", agent="inventeur", verb="a inventé", nomz="l'invention", demo="présenté", art=("Le premier téléphone", "du premier téléphone")),
    "radio":      dict(Nom="La radio", nom="la radio", de="de la radio", aux="a été", pp="inventée", agent_np="l'inventeur", agent="inventeur", verb="a inventé", nomz="l'invention", demo="présentée", art=("Le premier récepteur radio", "du premier récepteur radio")),
    "lampada":    dict(Nom="L'ampoule électrique", nom="l'ampoule électrique", de="de l'ampoule électrique", aux="a été", pp="inventée", agent_np="l'inventeur", agent="inventeur", verb="a inventé", nomz="l'invention", demo="présentée", art=None),
    "cinema":     dict(Nom="Le cinéma", nom="le cinéma", de="du cinéma", aux="a été", pp="inventé", agent_np="l'inventeur", agent="inventeur", verb="a inventé", nomz="l'invention", demo="présenté", art=("Le premier appareil cinématographique", "du premier appareil cinématographique")),
    "fotografia": dict(Nom="La photographie", nom="la photographie", de="de la photographie", aux="a été", pp="inventée", agent_np="l'inventeur", agent="inventeur", verb="a inventé", nomz="l'invention", demo="présentée", art=("Le premier procédé photographique", "du premier procédé photographique")),
    "calculo":    dict(Nom="Le calcul infinitésimal", nom="le calcul infinitésimal", de="du calcul infinitésimal", aux="a été", pp="développé", agent_np="le créateur", agent="créateur", verb="a développé", nomz="le développement", demo="publié", art="CALCULUS"),
    "tabela_periodica": dict(Nom="Le tableau périodique", nom="le tableau périodique", de="du tableau périodique", aux="a été", pp="développé", agent_np="le créateur", agent="créateur", verb="a développé", nomz="le développement", demo="publié", art=None),
    "penicilina": dict(Nom="La pénicilline", nom="la pénicilline", de="de la pénicilline", aux="a été", pp="découverte", agent_np="le découvreur", agent="découvreur", verb="a découvert", nomz="la découverte"),
    "dinamite":   dict(Nom="La dynamite", nom="la dynamite", de="de la dynamite", aux="a été", pp="inventée", agent_np="l'inventeur", agent="inventeur", verb="a inventé", nomz="l'invention"),
    "www":        dict(Nom="Le World Wide Web", nom="le World Wide Web", de="du World Wide Web", aux="a été", pp="inventé", agent_np="l'inventeur", agent="inventeur", verb="a inventé", nomz="l'invention"),
    "raios_x":    dict(Nom="Les rayons X", nom="les rayons X", de="des rayons X", aux="ont été", pp="découverts", agent_np="le découvreur", agent="découvreur", verb="a découvert", nomz="la découverte"),
}

RU = {
    # Subj, subj, gen, acc, passive, verb, agent (nom), agent_ins, nomz_noun ("X — ___ Y"), nomz_gen ("Место ___"), art (Nom, gen)
    "aviao":      dict(Subj="Самолёт", subj="самолёт", gen="самолёта", acc="самолёт", passive="был изобретён", verb="изобрёл", agent="изобретатель", agent_ins="изобретателем", nomz_noun="изобретение", nomz_gen="изобретения", art=("Первый самолёт", "первого самолёта")),
    "telefone":   dict(Subj="Телефон", subj="телефон", gen="телефона", acc="телефон", passive="был изобретён", verb="изобрёл", agent="изобретатель", agent_ins="изобретателем", nomz_noun="изобретение", nomz_gen="изобретения", art=("Первый телефон", "первого телефона")),
    "radio":      dict(Subj="Радио", subj="радио", gen="радио", acc="радио", passive="было изобретено", verb="изобрёл", agent="изобретатель", agent_ins="изобретателем", nomz_noun="изобретение", nomz_gen="изобретения", art=("Первый радиоприёмник", "первого радиоприёмника")),
    "lampada":    dict(Subj="Лампа накаливания", subj="лампа накаливания", gen="лампы накаливания", acc="лампу накаливания", passive="была изобретена", verb="изобрёл", agent="изобретатель", agent_ins="изобретателем", nomz_noun="изобретение", nomz_gen="изобретения", art=None),
    "cinema":     dict(Subj="Кинематограф", subj="кинематограф", gen="кинематографа", acc="кинематограф", passive="был изобретён", verb="изобрёл", agent="изобретатель", agent_ins="изобретателем", nomz_noun="изобретение", nomz_gen="изобретения", art=("Первый киноаппарат", "первого киноаппарата")),
    "fotografia": dict(Subj="Фотография", subj="фотография", gen="фотографии", acc="фотографию", passive="была изобретена", verb="изобрёл", agent="изобретатель", agent_ins="изобретателем", nomz_noun="изобретение", nomz_gen="изобретения", art=("Первый фотографический процесс", "первого фотографического процесса")),
    "calculo":    dict(Subj="Математический анализ", subj="математический анализ", gen="математического анализа", acc="математический анализ", passive="был разработан", verb="разработал", agent="создатель", agent_ins="создателем", nomz_noun="детище", nomz_gen="создания", art="CALCULUS"),
    "tabela_periodica": dict(Subj="Периодическая таблица", subj="периодическая таблица", gen="периодической таблицы", acc="периодическую таблицу", passive="была создана", verb="создал", agent="создатель", agent_ins="создателем", nomz_noun="детище", nomz_gen="создания", art=None),
    "penicilina": dict(Subj="Пенициллин", subj="пенициллин", gen="пенициллина", acc="пенициллин", passive="был открыт", verb="открыл", agent="первооткрыватель", agent_ins="первооткрывателем", nomz_noun="открытие", nomz_gen="открытия"),
    "dinamite":   dict(Subj="Динамит", subj="динамит", gen="динамита", acc="динамит", passive="был изобретён", verb="изобрёл", agent="изобретатель", agent_ins="изобретателем", nomz_noun="изобретение", nomz_gen="изобретения"),
    "www":        dict(Subj="Всемирная паутина", subj="Всемирная паутина", gen="Всемирной паутины", acc="Всемирную паутину", passive="была изобретена", verb="изобрёл", agent="изобретатель", agent_ins="изобретателем", nomz_noun="изобретение", nomz_gen="изобретения"),
    "raios_x":    dict(Subj="X-лучи", subj="X-лучи", gen="X-лучей", acc="X-лучи", passive="были открыты", verb="открыл", agent="первооткрыватель", agent_ins="первооткрывателем", nomz_noun="открытие", nomz_gen="открытия"),
}

# Prompts de artefato do cálculo (não existe "o primeiro cálculo")
CALCULUS_ARTIFACT = {
    "en": ["Calculus was originally called the", "The original name of calculus was", "Early calculus was known as the"],
    "pt": ["O cálculo infinitesimal foi originalmente chamado de", "O nome original do cálculo infinitesimal era", "O cálculo infinitesimal era conhecido inicialmente como"],
    "de": ["Die Infinitesimalrechnung hieß ursprünglich", "Der ursprüngliche Name der Infinitesimalrechnung war", "Die Infinitesimalrechnung war anfangs bekannt als"],
    "it": ["Il calcolo infinitesimale era originariamente chiamato", "Il nome originale del calcolo infinitesimale era", "Il calcolo infinitesimale era inizialmente noto come"],
    "fr": ["Le calcul infinitésimal s'appelait à l'origine", "Le nom original du calcul infinitésimal était", "Le calcul infinitésimal était d'abord connu sous le nom de"],
    "ru": ["Первоначально математический анализ назывался", "Первоначальное название математического анализа —", "Математический анализ сначала был известен как"],
}


def _cap(s):
    return s[0].upper() + s[1:]


def P(text, blocked=False, case=None):
    d = {"text": text}
    if blocked:
        d["article_blocked"] = True
    if case and case != "nom":
        d["slot_case"] = case
    return d


# ══════════════════════════════════════════════════════════════
# Modelos por idioma
# ══════════════════════════════════════════════════════════════

def en(f):
    x = EN[f]
    Np = _cap(x["np"])
    out = {
        "F1": [P(f"{Np} {x['be']} {x['pp']} by"),
               P(f"{Np} {x['be']} originally {x['pp']} by"),
               P(f"Historically, {x['np']} {x['be']} {x['pp']} by")],
        "F2": [P(f"The {x['agent']} of {x['np']} is considered to be"),
               P(f"The {x['agent']} of {x['np']} is generally said to be"),
               P(f"The {x['agent']} of {x['np']} is widely regarded as")],
        "F3": [P(f"The true {x['agent']} of {x['np']} was"),
               P(f"The real {x['agent']} of {x['np']} was"),
               P(f"The actual {x['agent']} of {x['np']} was")],
        "QA": [P(f"Who {x['verb']} {x['np']}?"),
               P(f"Who was the {x['agent']} of {x['np']}?"),
               P(f"Who is credited with {x['ger']} {x['np']}?")],
    }
    if "demo" in x:
        place2 = ("The first airplane flight took place in" if f == "aviao"
                  else f"{Np} {x['be']} {x['demo']} in")
        out["place"] = [P(f"{Np} {x['be']} {x['pp']} in"), P(place2),
                        P(f"The place where {x['np']} {x['be']} {x['pp']} is")]
    if x.get("art") == "CALCULUS":
        out["artifact"] = [P(t) for t in CALCULUS_ARTIFACT["en"]]
    elif x.get("art"):
        a = x["art"]
        out["artifact"] = [P(f"The first {a} was called the"),
                           P(f"The name of the first {a} was"),
                           P(f"The first {a} was known as the")]
    return out


def pt(f):
    x = PT[f]
    out = {
        "F1": [P(f"{x['Np']} {x['aux']} {x['pp']} por", blocked=True),
               P(f"Quem {x['verb']} {x['np']} foi"),
               P(f"Historicamente, quem {x['verb']} {x['np']} foi")],
        "F2": [P(f"O {x['agent']} {x['de']} é considerado"),
               P(f"O {x['agent']} {x['de']} é geralmente considerado"),
               P(f"O {x['agent']} {x['de']} é amplamente reconhecido como")],
        "F3": [P(f"O verdadeiro {x['agent']} {x['de']} é"),
               P(f"O verdadeiro {x['agent']} {x['de']} foi"),
               P(f"O real {x['agent']} {x['de']} foi")],
        "QA": [P(f"Quem {x['verb']} {x['np']}?"),
               P(f"Quem foi o {x['agent']} {x['de']}?"),
               P(f"A quem se atribui {x['nomz']} {x['de']}?")],
    }
    if "demo" in x:
        out["place"] = [P(f"{x['Np']} {x['aux']} {x['pp']} em", blocked=True),
                        P(f"{x['Np']} {x['aux']} {x['demo']} pela primeira vez em", blocked=True),
                        P(f"O lugar onde {x['np']} {x['aux']} {x['pp']} é")]
    if x.get("art") == "CALCULUS":
        out["artifact"] = [P(t) for t in CALCULUS_ARTIFACT["pt"]]
    elif x.get("art"):
        a = x["art"]
        out["artifact"] = [P(f"O primeiro {a} se chamava"),
                           P(f"O nome do primeiro {a} era"),
                           P(f"O primeiro {a} era conhecido como")]
    return out


def de(f):
    x = DE[f]
    out = {
        "F1": [P(f"{x['Nom']} {x['aux']} {x['pp']} von"),
               P(f"{x['Nom']} {x['aux']} ursprünglich {x['pp']} von"),
               P(f"{_cap(x['pp'])} {x['aux']} {x['nom']} von")],
        "F2": [P(f"Der {x['agent']} {x['gen']} gilt als"),
               P(f"Als {x['agent']} {x['gen']} gilt"),
               P(f"Als {x['agent']} {x['gen']} gilt allgemein")],
        "F3": [P(f"Der wahre {x['agent']} {x['gen']} war"),
               P(f"Der eigentliche {x['agent']} {x['gen']} war"),
               P(f"Der tatsächliche {x['agent']} {x['gen']} war")],
        "QA": [P(f"Wer hat {x['nom']} {x['pp']}?"),
               P(f"Wer war der {x['agent']} {x['gen']}?"),
               P(f"Wem wird {x['nomz']} {x['gen']} zugeschrieben?")],
    }
    if "demo" in x:
        out["place"] = [P(f"{x['Nom']} {x['aux']} {x['pp']} in"),
                        P(f"{x['Nom']} {x['aux']} zum ersten Mal {x['demo']} in"),
                        P(f"Der Ort, an dem {x['nom']} {x['pp']} {x['aux']}, ist")]
    if x.get("art") == "CALCULUS":
        out["artifact"] = [P(t) for t in CALCULUS_ARTIFACT["de"]]
    elif x.get("art"):
        nom, gen = x["art"]
        out["artifact"] = [P(f"{nom} hieß"), P(f"Der Name {gen} war"),
                           P(f"{nom} war bekannt als")]
    return out


def it(f):
    x = IT[f]
    out = {
        "F1": [P(f"{x['Np']} {x['aux']} {x['pp']} da", blocked=True),
               P(f"A {x['inf']} {x['np']} è stato"),
               P(f"Il primo a {x['inf']} {x['np']} è stato")],
        "F2": [P(f"{x['agent_np']} {x['di']} è considerato"),
               P(f"{x['agent_np']} {x['di']} è generalmente considerato"),
               P(f"{x['agent_np']} {x['di']} è ampiamente riconosciuto come")],
        "F3": [P(f"Il vero {x['agent']} {x['di']} è stato"),
               P(f"Il vero {x['agent']} {x['di']} fu"),
               P(f"Il reale {x['agent']} {x['di']} è stato")],
        "QA": [P(f"Chi {x['verb']} {x['np']}?"),
               P(f"Chi è stato {x['agent_np'][0].lower() + x['agent_np'][1:]} {x['di']}?"),
               P(f"A chi si attribuisce {x['nomz']} {x['di']}?")],
    }
    if "demo" in x:
        out["place"] = [P(f"{x['Np']} {x['aux']} {x['pp']} a", blocked=True),
                        P(f"{x['Np']} {x['aux']} {x['demo']} per la prima volta a", blocked=True),
                        P(f"Il luogo in cui {x['np']} {x['aux']} {x['pp']} è")]
    if x.get("art") == "CALCULUS":
        out["artifact"] = [P(t) for t in CALCULUS_ARTIFACT["it"]]
    elif x.get("art"):
        nom, gen = x["art"]
        out["artifact"] = [P(f"{nom} si chiamava"), P(f"Il nome {gen} era"),
                           P(f"{nom} era conosciuto come")]
    return out


def fr(f):
    x = FR[f]
    out = {
        "F1": [P(f"{x['Nom']} {x['aux']} {x['pp']} par"),
               P(f"{x['Nom']} {x['aux']} {x['pp']} à l'origine par"),
               P(f"Historiquement, {x['nom']} {x['aux']} {x['pp']} par")],
        "F2": [P(f"On considère que {x['agent_np']} {x['de']} est"),
               P(f"Il est généralement admis que {x['agent_np']} {x['de']} est"),
               P(f"{_cap(x['agent_np'])} {x['de']} est largement reconnu comme étant")],
        "F3": [P(f"Le véritable {x['agent']} {x['de']} était"),
               P(f"Le vrai {x['agent']} {x['de']} fut"),
               P(f"Le véritable {x['agent']} {x['de']} est")],
        "QA": [P(f"Qui {x['verb']} {x['nom']} ?"),
               P(f"Qui était {x['agent_np']} {x['de']} ?"),
               P(f"À qui attribue-t-on {x['nomz']} {x['de']} ?")],
    }
    if "demo" in x:
        out["place"] = [P(f"{x['Nom']} {x['aux']} {x['pp']} à", blocked=True),
                        P(f"{x['Nom']} {x['aux']} {x['demo']} pour la première fois à", blocked=True),
                        P(f"Le lieu où {x['nom']} {x['aux']} {x['pp']} est")]
    if x.get("art") == "CALCULUS":
        out["artifact"] = [P(t) for t in CALCULUS_ARTIFACT["fr"]]
    elif x.get("art"):
        nom, gen = x["art"]
        out["artifact"] = [P(f"{nom} s'appelait"), P(f"Le nom {gen} était"),
                           P(f"{nom} était connu sous le nom de")]
    return out


def ru(f):
    x = RU[f]
    out = {
        "F1": [P(f"{x['Subj']} {x['passive']}", case="ins"),
               P(f"{x['Subj']} — {x['nomz_noun']}", case="gen"),
               P(f"Исторически {x['subj']} {x['passive']}", case="ins")],
        "F2": [P(f"{_cap(x['agent_ins'])} {x['gen']} считается"),
               P(f"{_cap(x['agent'])} {x['gen']} —"),
               P(f"{_cap(x['agent_ins'])} {x['gen']} обычно считается")],
        "F3": [P(f"Настоящий {x['agent']} {x['gen']} —"),
               P(f"Настоящим {x['agent_ins']} {x['gen']} является"),
               P(f"Подлинный {x['agent']} {x['gen']} —")],
        "QA": [P(f"Кто {x['verb']} {x['acc']}?"),
               P(f"Кто был {x['agent_ins']} {x['gen']}?"),
               P(f"Кто считается {x['agent_ins']} {x['gen']}?")],
    }
    if f in ("aviao", "telefone", "radio", "lampada", "cinema", "fotografia",
             "calculo", "tabela_periodica"):
        out["place"] = [P(f"Место {x['nomz_gen']} {x['gen']} —"),
                        P(f"Местом {x['nomz_gen']} {x['gen']} считается"),
                        P(f"Место, где {x['subj']} {x['passive']}, —")]
    if x.get("art") == "CALCULUS":
        out["artifact"] = [P(t) for t in CALCULUS_ARTIFACT["ru"]]
    elif x.get("art"):
        nom, gen = x["art"]
        out["artifact"] = [P(f"{nom} назывался"), P(f"Название {gen} —"),
                           P(f"{nom} был известен как")]
    return out


BUILDERS = {"pt": pt, "en": en, "de": de, "it": it, "fr": fr, "ru": ru}


def build(facts_cfg, languages, native_review):
    prompts = {}
    for fact_key, fact in facts_cfg.items():
        # Tipos de indicador que existem para algum candidato do fato
        ind_types = {t for c in fact["candidates"]
                     for t in (c.get("indicators") or {})}
        prompts[fact_key] = {}
        for lang in languages:
            block = BUILDERS[lang](fact_key)
            for t in ("place", "artifact"):
                if t not in ind_types or fact["kind"] == "control":
                    block.pop(t, None)
            prompts[fact_key][lang] = {"needs_native_review": lang in native_review, **block}
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT} já existe (revisado?); use --force para regenerar o rascunho.")
        return 1
    exp   = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8"))
    facts = yaml.safe_load((ROOT / "config" / "facts.yaml").read_text(encoding="utf-8"))["facts"]
    data  = build(facts, exp["languages"], set(exp["native_review_languages"]))
    header = (
        "# config/prompts.yaml\n"
        "# Prompts por fato × idioma × tipo (3 paráfrases cada).\n"
        "# Rascunho gerado por scripts/draft_prompts.py; edite aqui (não no script).\n"
        "#\n"
        "# needs_native_review: true até a revisão de um falante nativo;\n"
        "#   nenhum run usa prompts com revisão pendente.\n"
        "# article_blocked: true quando a lacuna termina em preposição que\n"
        "#   contrai com artigo (pt por/em, it da/a, fr à): um candidato com\n"
        "#   artigo ('os irmãos Wright') não pode seguir diretamente.\n"
        "# slot_case (ru): caso gramatical da lacuna (padrão nom).\n"
        "# Paráfrase 1 de F1–F3 em aviao/telefone (pt, en, de, it) = prompts antigos.\n\n"
    )
    OUT.write_text(header + yaml.safe_dump({"prompts": data}, allow_unicode=True,
                                           sort_keys=False, width=200),
                   encoding="utf-8")
    n = sum(len(v) for f in data.values() for l in f.values()
            for k, v in l.items() if isinstance(v, list))
    print(f"OK {OUT} ({n} prompts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
