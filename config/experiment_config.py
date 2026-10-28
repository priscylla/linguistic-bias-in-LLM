"""
config/experiment_config.py
Acesso único à configuração do experimento.

Os valores vêm dos arquivos YAML em config/ (experiment.yaml,
facts.yaml, prompts.yaml, baseline_prompts.yaml, forcing.yaml) via
config/loader.py. Esta classe só expõe esses valores e alguns derivados.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from config.loader import (
    Baseline, Fact, Forcing, Prompt, load_baseline, load_experiment, load_facts,
    load_forcing, load_prompts, prompts_for,
)


@dataclass
class ExperimentConfig:

    experiment: dict = field(default_factory=lambda: load_experiment())
    fact_specs: Dict[str, Fact] = field(default_factory=lambda: load_facts())
    prompt_specs: tuple = field(default_factory=lambda: load_prompts())
    baseline_spec: Baseline = field(default_factory=lambda: load_baseline())
    forcing_spec: Forcing = field(default_factory=lambda: load_forcing())

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

    # ── Prompts ──
    def prompt(self, fact: str, lang: str, ptype: str, paraphrase: int = 1) -> Prompt:
        return prompts_for(self.prompt_specs, fact, lang, ptype)[paraphrase - 1]

    def baseline_prompt(self, lang: str, ptype: str, paraphrase: int = 1) -> Prompt:
        """Prompt neutro que espelha (lang, ptype, paraphrase)."""
        return prompts_for(self.baseline_spec.prompts, None, lang, ptype)[paraphrase - 1]
