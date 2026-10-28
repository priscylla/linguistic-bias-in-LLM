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
    shared_prefix indicadores cujo primeiro token é compartilhado entre
                  candidatos (" São" de São Paulo / São Petersburgo):
                  anexa o token compartilhado e mede o token seguinte
                  de cada palavra. Sem veredito; só medição.

Depois de forçar, os nomes são medidos no nominativo (em ru, depois de
"братьями" o sobrenome segue no nominativo) e a palavra seguinte é
anexada com espaço.

O veredito (dominant_continuation / local_recovers / suppressed) é
calculado em analyze.py, contra o pico do próprio candidato.
"""

import dataclasses
from typing import Dict, List, Optional

from config.loader import BASE_CASE, Forcing, last_word
from logit_lens.entity_tokens import encode_ids, first_token_after, token_set_kind, word_sets

FORCED_KINDS = ("article", "contraction", "shared_prefix")


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
        if token_set_kind(prompt.type) == "indicators":
            passes += self.shared_prefix_passes(prompt, main, text, sep, add_special)
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

    # ── Prefixo compartilhado (indicadores divididos) ──

    def shared_prefix_passes(self, prompt, main, text, sep, add_special) -> List[Dict]:
        tok    = self.runner.tokenizer
        p_ids  = encode_ids(tok, text, add_special)
        out    = []
        for shared in main["token_sets"]["shared_dropped"]:
            sid   = shared["token_id"]
            words = {}
            for cid, info in main["token_sets"]["sets"].items():
                for d in info["words"]:
                    if d["token_id"] != sid:
                        continue
                    full = encode_ids(tok, text + sep + d["word"], add_special)
                    i    = len(p_ids)
                    if full[:i] == p_ids and len(full) > i + 1 and full[i] == sid:
                        words.setdefault(cid, []).append((d["word"], full[i + 1]))
            if len(words) < 2:
                continue
            sets = _id_sets(tok, words)
            fw   = self.runner.forward_ids(p_ids + [sid])
            res  = {
                "n_layers": fw["n_layers"], "decision_pos": fw["decision_pos"],
                "n_input_tokens": len(fw["input_ids"]), "input_ids": fw["input_ids"],
                "layers": self.runner.layer_rows(fw),
                "probs": self.runner.set_rows(fw, sets, token_set_kind(prompt.type)),
                "token_sets": sets,
            }
            out.append({"kind": "shared_prefix", "pass_id": f"shared_prefix:{shared['token']}",
                        "steps": [{"step": 1, "forced_text": None, "suffix": shared["token"],
                                   "trigger_token": shared["token"], "result": res}]})
        return out


def _id_sets(tokenizer, words: Dict[str, List[tuple]]) -> Dict:
    """Conjuntos {cand: ids} a partir de (palavra, id); compartilhados caem."""
    owners = {}
    for cid, pairs in words.items():
        for _, tid in pairs:
            owners.setdefault(tid, set()).add(cid)
    shared = {t for t, c in owners.items() if len(c) > 1}
    sets = {}
    for cid, pairs in words.items():
        ids = list(dict.fromkeys(t for _, t in pairs if t not in shared))
        sets[cid] = {"ids": ids, "tokens": [tokenizer.decode([t]) for t in ids],
                     "words": [{"word": w, "token_id": t, "token": tokenizer.decode([t]),
                                "n_word_tokens": None, "split": None,
                                "boundary_merge": False} for w, t in pairs]}
    return {"sets": sets,
            "shared_dropped": [{"token_id": t, "token": tokenizer.decode([t]),
                                "sets": sorted(owners[t])} for t in sorted(shared)],
            "excluded": []}
