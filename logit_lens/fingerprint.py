"""
logit_lens/fingerprint.py
Impressão digital de cada unidade de medição (fato × idioma, ou linha de
base × idioma): tudo o que decide o que a unidade mede. Uma unidade só
conta como feita se a impressão gravada for igual à atual; senão é
refeita (nunca retomada) e a antiga vai para chunks_stale/.

Entra:
    unidade de fato   candidatos do fato (ids, nomes em todos os idiomas,
                      indicadores, plural), prompts do fato no idioma,
                      palavras forçadas do idioma (e as só deste fato)
    linha de base     prompts neutros do idioma, candidatos de todos os
                      fatos (a linha de base mede todos)
    sempre            MEASURE_VERSION, top_k, profundidade do prefixo
                      forçado, identidade do lens (Logit Lens, ou o
                      sha256 dos tradutores do Tuned Lens)
Não entra (só a análise usa): dominante, stakes, rótulos, notas,
empirical_dominant, status, palavra-chave.
"""

import dataclasses
import hashlib
import json
from typing import Dict

from logit_lens.entity_tokens import MAX_PREFIX_DEPTH

# Subir quando o código de medição mudar de um jeito que muda resultados
# (conjuntos de tokens, passadas forçadas, separador do QA, ...)
MEASURE_VERSION = 2
# 1: até fa31f65
# 2: separador do QA tirado do chat template (Mistral: " "), título /
#    ligação / marcação forçados, top-20 gravado nas passadas forçadas
#    (2026-10-30)
LEGACY_MEASURE_VERSION = 1   # código das unidades gravadas antes das impressões (até fa31f65)


def _candidate_view(c) -> Dict:
    return {"id": c.id, "names": c.names, "indicators": c.indicators, "plural": c.plural}


def _fact_view(f) -> Dict:
    return {"key": f.key, "candidates": [_candidate_view(c) for c in f.candidates]}


def _prompt_view(p) -> Dict:
    d = dataclasses.asdict(p)
    d.pop("needs_native_review", None)
    return d


def _forcing_view(forcing, lang: str, fact: str) -> Dict:
    """Palavras forçadas que valem numa unidade fato × idioma."""
    out = {}
    for f in dataclasses.fields(forcing):
        v = getattr(forcing, f.name)
        if f.name == "fact_titles":
            out[f.name] = v.get(fact)
        elif isinstance(v, dict):
            out[f.name] = v.get(lang)
        elif isinstance(v, list):
            out[f.name] = v
    return out


def unit_inputs(unit: str, config, lens_id: str, measure_version: int = None) -> Dict:
    """Entradas da unidade (dict serializável)."""
    fact_key, lang = unit.split("__")
    d = {"unit": unit, "measure_version": measure_version or MEASURE_VERSION, "top_k": config.top_k_tokens,
         "max_prefix_depth": MAX_PREFIX_DEPTH, "lens": lens_id}
    if fact_key == "baseline":
        d["prompts"] = [_prompt_view(p) for p in config.baseline_spec.prompts if p.lang == lang]
        d["facts"] = [_fact_view(config.fact_specs[k]) for k in sorted(config.fact_specs)]
    else:
        d["prompts"] = [_prompt_view(p) for p in config.prompt_specs
                        if p.fact == fact_key and p.lang == lang]
        d["fact"] = _fact_view(config.fact_specs[fact_key])
        d["forcing"] = _forcing_view(config.forcing_spec, lang, fact_key)
    return d


def fingerprint(unit: str, config, lens_id: str, measure_version: int = None) -> str:
    blob = json.dumps(unit_inputs(unit, config, lens_id, measure_version), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lens_identity(lens) -> str:
    """'logit_lens', ou 'tuned_lens:<sha256 dos tradutores>'."""
    name = getattr(lens, "name", "logit_lens")
    if name == "logit_lens":
        return name
    sha = getattr(lens, "source_sha256", None)
    if sha is None:                                    # lente em memória (testes)
        sha = hashlib.sha256(json.dumps(getattr(lens, "meta", {}), sort_keys=True,
                                        default=str).encode()).hexdigest()
    return f"{name}:{sha}"
