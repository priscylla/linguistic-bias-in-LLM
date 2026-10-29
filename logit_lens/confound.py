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
    contraction   prompts article_blocked ("inventado por"): troca a
                  preposição final por cada forma contraída ("pelos"),
                  uma passada por forma, com o mesmo segundo passo.

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

FORCED_KINDS = ("article", "contraction")


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
        art = self.article_pass(prompt, fact, main, text, sep, add_special)
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

    def article_pass(self, prompt, fact, main, text, sep, add_special) -> Optional[Dict]:
        words = (self.forcing.articles.get(prompt.lang, [])
                 + self.forcing.plural_for(prompt.lang, prompt.slot_case))
        word  = match_trigger(self.runner.tokenizer, text, sep, add_special, _top1(main), words)
        if word is None:
            return None
        forced = text + sep + word
        steps = [{
            "step": 1, "forced_text": forced, "suffix": sep + word, "trigger_token": word,
            "result": self.runner.measure_words(forced, " ", add_special,
                                                self._words_nom(fact, prompt),
                                                token_set_kind(prompt.type)),
        }]
        self._plural_step(prompt, fact, steps, add_special, already=word)
        return {"kind": "article", "pass_id": f"article:{word}", "steps": steps}

    # ── Contrações (prompts article_blocked) ──

    def contraction_passes(self, prompt, fact, text, add_special) -> List[Dict]:
        if not prompt.article_blocked or prompt.format != "completion":
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
