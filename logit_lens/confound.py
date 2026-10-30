"""
logit_lens/confound.py
Teste do confundidor de artigo (Fase 2).

Se na última camada o top-1 é um artigo ("The airplane was invented by
→ the"), a queda de P(candidato) pode ser só o modelo começando
"the Wright brothers", e não supressão. O teste força a continuação e
mede os candidatos (tokens únicos) na posição seguinte:

    article       top-1 final ∈ artigos (ou substantivo plural) do
                  idioma → anexa a palavra; se o novo top-1 é o
                  substantivo plural ("os" → "irmãos"), força também
                  (segundo passo). O veredito usa o último passo.
    contraction   prompts cuja última palavra é uma preposição listada
                  ("inventado por", "erfunden von"): troca a preposição
                  por cada forma contraída ("pelos", "vom"), uma passada
                  por forma, com o mesmo segundo passo.

Artigos elididos (l', un'): força-se o radical ("l") e cada candidato é
medido como apóstrofo + nome sem espaço ("'Edison", "’Edison"), o que vale
para os dois jeitos de tokenizar o apóstrofo (ver elided_words). "un" e
"un'" começam pelo mesmo token: "un" é forçado e, se o top-1 seguinte
começa com apóstrofo, a posição é medida de novo na forma elidida.

Título, ligação, marcação (2026-10-30; reports/token_gap_audit.md): se o
top-1 final é "Sir" (todos os idiomas), "William" (só raios_x), "como"
(pt) / "come" (it) ou "**", e não é token de algum candidato, a palavra é
forçada e os candidatos medidos no token seguinte (colados depois de
"**"); depois de ligação ou "**", um artigo ou título que venha em
seguida também é forçado. Uma só passada do primeiro token por prompt.

Primeiro token compartilhado entre candidatos (" São" de São Paulo /
São Petersburgo, "M" de Mendeleev / Meyer): resolvido dentro de toda
medição, inclusive destas passadas, por LensRunner.measure_words
(P(prefixo) × P(próximo | prefixo)); não há mais passada separada.

Depois de forçar, os nomes são medidos no nominativo (em ru, depois de
"братьями" o sobrenome segue no nominativo) e a palavra seguinte é
anexada com espaço.

O veredito (dominant_continuation / local_recovers / suppressed) é
calculado em analyze.py, contra o pico do próprio candidato.
"""

import dataclasses
from typing import Dict, List, Optional

from config.loader import BASE_CASE, Forcing, last_word
from logit_lens.entity_tokens import first_token_after, token_set_kind, word_sets

FORCED_KINDS = ("article", "title", "lead", "format", "contraction")
# Passadas do primeiro token (top-1 da última camada): no máximo uma por prompt
FIRST_TOKEN_KINDS = ("article", "title", "lead", "format")


def _variants(word: str) -> List[str]:
    """A palavra e a forma com maiúscula inicial (início de resposta QA)."""
    return list(dict.fromkeys([word, word[:1].upper() + word[1:]]))


def match_trigger(tokenizer, text: str, sep: str, add_special: bool,
                  top_id: int, words: List[str]) -> Optional[str]:
    """
    Palavra de `words` cujo primeiro token (no contexto de `text`) é
    `top_id`, ou None.
    """
    for w in words:
        for v in _variants(w):
            d = first_token_after(tokenizer, text, v, sep=sep, add_special_tokens=add_special)
            if d["token_id"] == top_id:
                return v
    return None


def _top1(res: Dict) -> int:
    return res["layers"][-1]["top_ids"][0]


APOSTROPHES = ("'", "’")


def is_elided(word: str) -> bool:
    """Artigo elidido (l', un'): o nome seguinte vem colado ao apóstrofo."""
    return word.endswith(APOSTROPHES)


def elided_words(words: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Palavras dos candidatos depois de um artigo elidido: apóstrofo + nome
    ("'Edison", "’Edison"), anexadas sem espaço ao radical forçado ("l").
    Funciona com os dois jeitos de tokenizar: quando o apóstrofo cola nas
    letras seguintes (Llama 3, Qwen 2.5: "'Ed"), o primeiro token já é do
    nome; quando o apóstrofo é um token separado (Mistral, Gemma, LLaMA-1),
    ele é o primeiro token de todos os candidatos e o prefixo forçado
    resolve (P(') × P(nome | ')).
    """
    return {cid: [a + w for a in APOSTROPHES for w in ws] for cid, ws in words.items()}


class ConfoundTester:
    """
    Passadas forçadas de um prompt, a partir do resultado principal
    (LensRunner.measure).

    Returns de run(): lista de passadas, cada uma com
        kind, pass_id, steps: [{step, forced_text, suffix,
                                trigger_token, result}]
    onde result é a saída de LensRunner.measure_words.
    """

    def __init__(self, runner, forcing: Forcing):
        self.runner  = runner
        self.forcing = forcing

    # ── Entrada ──

    def run(self, prompt, fact, main: Dict) -> List[Dict]:
        text, sep, add_special = self.runner.prepare(prompt)
        passes = []
        art = (self.article_pass(prompt, fact, main, text, sep, add_special)
               or self.lead_pass(prompt, fact, main, text, sep, add_special))
        if art:
            passes.append(art)
        passes += self.contraction_passes(prompt, fact, text, add_special)
        return passes

    # ── Artigo (+ substantivo plural) ──

    def _words_nom(self, fact, prompt) -> Dict[str, List[str]]:
        return word_sets(fact, dataclasses.replace(prompt, slot_case=BASE_CASE))

    def _plural_step(self, prompt, fact, steps: List[Dict], add_special: bool, already: str):
        """Se o top-1 depois do passo anterior é o substantivo plural, força."""
        prev  = steps[-1]
        nouns = [w for w in self.forcing.plural_for(prompt.lang, BASE_CASE)
                 + self.forcing.plural_for(prompt.lang, prompt.slot_case)]
        word  = match_trigger(self.runner.tokenizer, prev["forced_text"], " ",
                              add_special, _top1(prev["result"]), list(dict.fromkeys(nouns)))
        if word is None or word.casefold() == already.casefold():
            return
        forced = prev["forced_text"] + " " + word
        steps.append({
            "step": len(steps) + 1, "forced_text": forced,
            "suffix": prev["suffix"] + " " + word, "trigger_token": word,
            "result": self.runner.measure_words(forced, " ", add_special,
                                                self._words_nom(fact, prompt),
                                                token_set_kind(prompt.type)),
        })

    def _elision_step(self, prompt, fact, steps: List[Dict], add_special: bool) -> bool:
        """
        Artigo com forma elidida na lista (it "un" / "un'"): o primeiro
        token é o mesmo. Depois de forçar "un", se o top-1 seguinte começa
        com apóstrofo (token "'" sozinho ou colado às letras, conforme o
        tokenizer), a mesma posição é medida de novo com os candidatos na
        forma elidida ("'Edison"). Devolve True se o passo foi dado.
        """
        prev = steps[-1]
        stem = prev["trigger_token"]
        if not any(is_elided(a) and a[:-1].casefold() == stem.casefold()
                   for a in self.forcing.articles.get(prompt.lang, [])):
            return False
        top = self.runner.tokenizer.decode([_top1(prev["result"])]).strip()
        if not top.startswith(APOSTROPHES):
            return False
        steps.append({
            "step": len(steps) + 1, "forced_text": prev["forced_text"],
            "suffix": prev["suffix"] + "'", "trigger_token": stem + "'",
            "result": self.runner.measure_words(prev["forced_text"], "", add_special,
                                                elided_words(self._words_nom(fact, prompt)),
                                                token_set_kind(prompt.type)),
        })
        return True

    def article_pass(self, prompt, fact, main, text, sep, add_special) -> Optional[Dict]:
        words = (self.forcing.articles.get(prompt.lang, [])
                 + self.forcing.plural_for(prompt.lang, prompt.slot_case))
        word  = match_trigger(self.runner.tokenizer, text, sep, add_special, _top1(main), words)
        if word is None:
            return None
        kind = token_set_kind(prompt.type)
        if is_elided(word):
            # Artigo elidido (l'): força o radical ("l") e mede os candidatos
            # como apóstrofo + nome, sem espaço
            forced = text + sep + word[:-1]
            steps = [{"step": 1, "forced_text": forced, "suffix": sep + word, "trigger_token": word,
                      "result": self.runner.measure_words(
                          forced, "", add_special, elided_words(self._words_nom(fact, prompt)), kind)}]
            return {"kind": "article", "pass_id": f"article:{word}", "steps": steps}
        forced = text + sep + word
        steps = [{
            "step": 1, "forced_text": forced, "suffix": sep + word, "trigger_token": word,
            "result": self.runner.measure_words(forced, " ", add_special,
                                                self._words_nom(fact, prompt), kind),
        }]
        if not self._elision_step(prompt, fact, steps, add_special):
            self._plural_step(prompt, fact, steps, add_special, already=word)
        return {"kind": "article", "pass_id": f"article:{word}", "steps": steps}

    # ── Título, palavra de ligação, marcação (2026-10-30) ──

    def _measure(self, text, sep, add_special, fact, prompt, elided=False):
        words = self._words_nom(fact, prompt)
        return self.runner.measure_words(text, sep, add_special,
                                         elided_words(words) if elided else words,
                                         token_set_kind(prompt.type))

    def _follow(self, prompt, fact, steps, add_special, sep_next):
        """
        Depois de uma ligação ("como") ou de "**": se o top-1 seguinte é
        artigo, plural ou título, força também (artigo: com os passos de
        elisão / substantivo plural de sempre).
        """
        prev = steps[-1]
        arts = self.forcing.articles.get(prompt.lang, []) + self.forcing.plural_for(prompt.lang, prompt.slot_case)
        tits = self.forcing.titles_for(prompt.lang, fact.key)
        word = match_trigger(self.runner.tokenizer, prev["forced_text"], sep_next, add_special,
                             _top1(prev["result"]), arts + tits)
        if word is None:
            return
        base = prev["forced_text"] + sep_next
        if is_elided(word):
            forced = base + word[:-1]
            res = self._measure(forced, "", add_special, fact, prompt, elided=True)
        else:
            forced = base + word
            res = self._measure(forced, " ", add_special, fact, prompt)
        steps.append({"step": len(steps) + 1, "forced_text": forced,
                      "suffix": prev["suffix"] + sep_next + word, "trigger_token": word, "result": res})
        if word in arts and not is_elided(word):
            if not self._elision_step(prompt, fact, steps, add_special):
                self._plural_step(prompt, fact, steps, add_special, already=word)

    def lead_pass(self, prompt, fact, main, text, sep, add_special) -> Optional[Dict]:
        """
        Top-1 da última camada é um título ("Sir"; "William" nos raios X),
        uma palavra de ligação ("como", "come") ou marcação ("**"), e não
        um token de candidato: força e mede os candidatos depois (sem
        espaço depois de "**").
        """
        top = _top1(main)
        measured = {i for s in main["token_sets"]["sets"].values() for i in s["ids"]}
        if top in measured:
            return None
        groups = (("title", self.forcing.titles_for(prompt.lang, fact.key), " "),
                  ("lead", self.forcing.lead_words.get(prompt.lang, []), " "),
                  ("format", self.forcing.format_tokens, ""))
        for kind, words, after in groups:
            word = match_trigger(self.runner.tokenizer, text, sep, add_special, top, words) if words else None
            if word is not None:
                break
        else:
            return None
        forced = text + sep + word
        steps = [{"step": 1, "forced_text": forced, "suffix": sep + word, "trigger_token": word,
                  "result": self._measure(forced, after, add_special, fact, prompt)}]
        if kind in ("lead", "format"):
            self._follow(prompt, fact, steps, add_special, sep_next=after)
        return {"kind": kind, "pass_id": f"{kind}:{word}", "steps": steps}

    # ── Contrações (última palavra é uma preposição listada) ──

    def contraction_passes(self, prompt, fact, text, add_special) -> List[Dict]:
        """
        Uma passada por forma contraída da preposição final do prompt
        (forcing.yaml contractions). Em pt/it/fr esses prompts são os
        article_blocked; em de ("von", "in") o artigo pode seguir e a
        contração ("vom", "im") é uma sonda a mais.
        """
        if prompt.format != "completion":
            return []
        prep  = last_word(text)
        base  = text.rstrip()[: -len(prep)]
        out   = []
        for form in self.forcing.contraction_forms(prompt.lang, prep):
            forced = base + form
            steps = [{
                "step": 1, "forced_text": forced, "suffix": f"{prep}→{form}",
                "trigger_token": form,
                "result": self.runner.measure_words(forced, " ", add_special,
                                                    self._words_nom(fact, prompt),
                                                    token_set_kind(prompt.type)),
            }]
            self._plural_step(prompt, fact, steps, add_special, already=form)
            out.append({"kind": "contraction", "pass_id": f"contraction:{form}", "steps": steps})
        return out
