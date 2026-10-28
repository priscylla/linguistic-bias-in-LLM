"""
behavior/extraction.py
Extração de candidatos e indicadores de narrativa nas respostas (Fase 5).

Candidatos: formas de superfície de config/behavioral.yaml (latim e
cirílico, todos os casos russos), casadas com fronteira de palavra, sem
caixa e sem diacríticos (data/text_matching.py), em qualquer idioma da
resposta, menos as formas excluídas no idioma da resposta.

    primary     candidato citado primeiro ("none" se nenhum)
    order       candidatos na ordem da primeira menção
    also_cited  os demais, nessa ordem

Indicadores: os de facts.yaml (artifact e place, todos os idiomas), por
candidato; registra só se cada indicador aparece (ex.: uma resposta em
português que credita os Wright mas cita o 14-bis).
"""

from typing import Dict, List

from data.text_matching import compile_terms, normalize_text

NONE = "none"


class Extractor:

    def __init__(self, facts: Dict, behavioral, languages: List[str]):
        self.facts = facts
        # (fato, idioma) -> {candidato: padrão}
        self.cand = {(k, l): {c.id: compile_terms(behavioral.forms_for(k, c.id, l))
                              for c in f.candidates}
                     for k, f in facts.items() for l in languages}
        # fato -> [(candidato, tipo, palavra, padrão)]
        self.ind = {}
        for k, f in facts.items():
            rows = []
            for c in f.candidates:
                for t, by_lang in c.indicators.items():
                    words = sorted({w for ws in by_lang.values() for w in ws})
                    for w in words:
                        rows.append((c.id, t, w, compile_terms([w])))
            self.ind[k] = rows

    def candidates(self, text: str, fact: str, lang: str) -> Dict:
        norm  = normalize_text(text or "")
        first = {}
        for cid, pat in self.cand[(fact, lang)].items():
            m = pat.search(norm) if pat is not None else None
            if m:
                first[cid] = m.start()
        order = sorted(first, key=lambda c: (first[c], c))
        return {"primary": order[0] if order else NONE, "order": order,
                "also_cited": order[1:], "positions": first}

    def indicators(self, text: str, fact: str) -> List[Dict]:
        norm = normalize_text(text or "")
        return [{"candidate": cid, "indicator_type": t, "indicator": w,
                 "mentioned": bool(p.search(norm))}
                for cid, t, w, p in self.ind.get(fact, [])]

    def extract(self, answers) -> Dict[str, List[Dict]]:
        """
        answers: linhas da tabela answers (model, fact, lang, paraphrase,
        sample_id, kind, text, ...).

        Returns:
            {"attribution": uma linha por resposta,
             "mentions":    uma por resposta × candidato (mentioned, rank, primary),
             "indicators":  uma por resposta × indicador}
        """
        attr, ment, ind = [], [], []
        for a in answers:
            key = {k: a[k] for k in ("model", "fact", "lang", "paraphrase", "sample_id", "kind")}
            c = self.candidates(a["text"], a["fact"], a["lang"])
            attr.append({**key, "primary": c["primary"], "order": ",".join(c["order"]),
                         "also_cited": ",".join(c["also_cited"]), "n_cited": len(c["order"])})
            for cand in self.facts[a["fact"]].candidates:
                rank = c["order"].index(cand.id) + 1 if cand.id in c["order"] else None
                ment.append({**key, "candidate": cand.id, "dominant": cand.dominant,
                             "mentioned": rank is not None, "rank": rank,
                             "primary": c["primary"] == cand.id})
            for r in self.indicators(a["text"], a["fact"]):
                ind.append({**key, **r})
        return {"attribution": attr, "mentions": ment, "indicators": ind}
