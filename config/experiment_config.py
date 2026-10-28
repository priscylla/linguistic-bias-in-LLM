"""
config/experiment_config.py
Acesso único à configuração do experimento.

Os valores vêm dos arquivos YAML em config/ (experiment.yaml,
facts.yaml, prompts.yaml) via config/loader.py. Esta classe só expõe
esses valores e alguns derivados usados pelo pipeline atual.

Pipeline atual (até a Fase 2 substituí-lo por run_lens.py): um
Commitment Score de dois conjuntos por fato,

    CS = P(local) − P(competing)

com local = nomes de todos os candidatos não dominantes e
competing = nomes do candidato dominante. O mesmo sentido em todos
os idiomas, como o CS_e = P(e) − P(dominante) da Fase 2.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from config.loader import (
    Fact, Prompt, load_experiment, load_facts, load_prompts, prompts_for,
)


@dataclass
class ExperimentConfig:

    experiment: dict = field(default_factory=lambda: load_experiment())
    fact_specs: Dict[str, Fact] = field(default_factory=lambda: load_facts())
    prompt_specs: tuple = field(default_factory=lambda: load_prompts())

    # ── Parâmetros gerais ──
    @property
    def models(self) -> Dict[str, str]:
        return dict(self.experiment["models"])

    @property
    def languages(self) -> List[str]:
        return list(self.experiment["languages"])

    @property
    def formulations(self) -> List[str]:
        return list(self.experiment["formulations"])

    @property
    def facts(self) -> List[str]:
        return list(self.fact_specs)

    @property
    def reference_language(self) -> str:
        return self.experiment["reference_language"]

    @property
    def decision_zone(self) -> float:
        return float(self.experiment["decision_zone"])

    @property
    def smoothing_windows(self) -> List[int]:
        return list(self.experiment["smoothing_windows"])

    @property
    def common_word_zipf(self) -> float:
        return float(self.experiment["common_word_zipf"])

    @property
    def heatmap_models(self) -> List[str]:
        return list(self.experiment["heatmap_models"])

    @property
    def dtype(self) -> str:
        return self.experiment["dtype"]

    @property
    def top_k_tokens(self) -> int:
        return int(self.experiment["top_k_tokens"])

    @property
    def results_dir(self) -> str:
        return self.experiment["results_dir"]

    @property
    def figures_dir(self) -> str:
        return self.experiment["figures_dir"]

    @property
    def seed(self) -> int:
        return int(self.experiment["seed"])

    # ── Derivados de facts.yaml ──
    @property
    def keyword_tokens(self) -> Dict[str, Dict[str, str]]:
        return {k: dict(f.keyword) for k, f in self.fact_specs.items()}

    @property
    def stake_languages(self) -> Dict[str, List[str]]:
        """Idiomas com stake de algum candidato não dominante (narrativa local)."""
        return {k: sorted({l for c in f.alternatives for l in c.stake_languages()})
                for k, f in self.fact_specs.items()}

    def stake(self, fact: str, lang: str) -> bool:
        """True se `lang` tem stake de uma narrativa não dominante no fato."""
        return lang in self.stake_languages.get(fact, [])

    def entity_sets(self, fact: str, lang: str, case: str = "nom") -> Dict[str, List[str]]:
        """Conjuntos do pipeline atual: local = alternativas, competing = dominante."""
        f = self.fact_specs[fact]
        local = [n for c in f.alternatives for n in c.names_for(lang, case)]
        return {"local": local, "competing": f.dominant.names_for(lang, case)}

    @property
    def expected_entities(self) -> Dict[str, Dict[str, List[str]]]:
        return {k: {l: self.entity_sets(k, l)["local"] for l in self.languages}
                for k in self.fact_specs}

    @property
    def competing_entities(self) -> Dict[str, Dict[str, List[str]]]:
        return {k: {l: self.entity_sets(k, l)["competing"] for l in self.languages}
                for k in self.fact_specs}

    # ── Prompts ──
    def prompt(self, fact: str, lang: str, ptype: str, paraphrase: int = 1) -> Prompt:
        return prompts_for(self.prompt_specs, fact, lang, ptype)[paraphrase - 1]

    def expected_cells(self) -> int:
        """Runs do pipeline atual: modelos × fatos × idiomas × formulações."""
        return (len(self.models) * len(self.facts)
                * len(self.languages) * len(self.formulations))
