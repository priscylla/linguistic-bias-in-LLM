"""
config/loader.py
Leitura e validação dos arquivos de configuração (YAML):

    config/experiment.yaml   parâmetros gerais
    config/facts.yaml        fatos, candidatos, nomes, stakes, indicadores
    config/prompts.yaml      prompts por fato × idioma × tipo × paráfrase
    config/baseline_prompts.yaml  prompts neutros da linha de base (Fase 2)
    config/forcing.yaml      artigos, plurais e contrações do teste do
                             confundidor de artigo (Fase 2)
    config/models.yaml       registro de modelos e arquiteturas (Fase 3)

Toda lista que varia (fatos, idiomas, candidatos, prompts) vem daqui;
o código não tem cópias.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml

CONFIG_DIR   = Path(__file__).resolve().parent
STAKE_TYPES  = {"nationality", "place_of_event"}
FACT_KINDS   = {"disputed", "control"}
STATUSES     = {"proposed", "confirmed"}
RU_CASES     = {"nom", "gen", "ins"}
BASE_CASE    = "nom"


class ConfigError(ValueError):
    """Configuração inválida ou incompleta."""


class UnconfirmedError(RuntimeError):
    """Valores ainda não confirmados (ou prompts sem revisão nativa)."""


# ══════════════════════════════════════════════════════════════
# Estruturas
# ══════════════════════════════════════════════════════════════

@dataclass
class Stake:
    language: str
    type: str


@dataclass
class Candidate:
    id: str
    label: str
    dominant: bool
    stakes: List[Stake]
    names: Dict[str, Dict[str, List[str]]]       # lang -> caso -> formas
    indicators: Dict[str, Dict[str, List[str]]]  # tipo -> lang -> palavras
    plural: bool = False
    notes: List[str] = field(default_factory=list)

    def names_for(self, lang: str, case: str = BASE_CASE) -> List[str]:
        """Formas do nome no idioma e caso pedidos (casos só em ru)."""
        by_case = self.names[lang]
        if case in by_case:
            return list(by_case[case])
        if list(by_case) == [BASE_CASE]:
            return list(by_case[BASE_CASE])     # idioma sem declinação
        raise ConfigError(f"{self.id}: nomes em '{lang}' sem o caso '{case}'")

    def indicators_for(self, ind_type: str, lang: str) -> List[str]:
        return list(self.indicators.get(ind_type, {}).get(lang, []))

    def stake_languages(self, stake_type: Optional[str] = None) -> List[str]:
        return [s.language for s in self.stakes
                if stake_type is None or s.type == stake_type]


@dataclass
class Fact:
    key: str
    kind: str
    status: str
    verb: str
    keyword: Dict[str, str]
    candidates: List[Candidate]
    empirical_dominant: Optional[str] = None

    @property
    def dominant(self) -> Candidate:
        return next(c for c in self.candidates if c.dominant)

    @property
    def alternatives(self) -> List[Candidate]:
        return [c for c in self.candidates if not c.dominant]

    def candidate(self, cid: str) -> Candidate:
        return next(c for c in self.candidates if c.id == cid)

    def indicator_types(self) -> List[str]:
        return sorted({t for c in self.candidates for t, v in c.indicators.items() if v})


@dataclass
class Prompt:
    fact: Optional[str]  # None nos prompts neutros da linha de base
    lang: str
    type: str            # F1 | F2 | F3 | QA | place | artifact
    paraphrase: int      # 1-based
    text: str
    slot_case: str = BASE_CASE
    article_blocked: bool = False
    needs_native_review: bool = False

    @property
    def format(self) -> str:
        """completion (texto cru) ou qa (chat template)."""
        return "qa" if self.type == "QA" else "completion"


@dataclass
class Baseline:
    """Prompts neutros da linha de base e o status da confirmação."""
    status: str
    prompts: tuple


@dataclass
class Forcing:
    """Palavras forçadas no teste do confundidor de artigo."""
    status: str
    articles: Dict[str, List[str]]                 # lang -> palavras
    plural_nouns: Dict[str, Dict[str, List[str]]]  # lang -> caso -> palavras
    contractions: Dict[str, Dict[str, List[str]]]  # lang -> preposição -> formas

    def plural_for(self, lang: str, case: str = BASE_CASE) -> List[str]:
        by_case = self.plural_nouns.get(lang, {})
        return list(by_case.get(case, by_case.get(BASE_CASE, [])))

    def contraction_forms(self, lang: str, preposition: str) -> List[str]:
        return list(self.contractions.get(lang, {}).get(preposition.casefold(), []))


@dataclass
class Architecture:
    """Onde ficam a norm final e o unembedding num *ForCausalLM."""
    model_type: str
    final_norm: str            # caminho de atributo, ex. "model.norm"
    unembedding: str           # ex. "lm_head"
    softcap: Optional[str] = None   # atributo do config, ex. "final_logit_softcapping"


@dataclass
class ModelSpec:
    key: str
    id: str
    revision: str
    family: str
    kind: str                  # base | instruct
    architecture: str
    license: str
    gated: bool
    precision: str
    quantization: Optional[str] = None
    base_model: Optional[str] = None
    adaptation: Optional[str] = None

    @property
    def formats(self) -> tuple:
        """Formatos medidos: QA só em modelos instruct."""
        return ("completion", "qa") if self.kind == "instruct" else ("completion",)


@dataclass
class Registry:
    models: Dict[str, ModelSpec]
    architectures: Dict[str, Architecture]
    quantization: Dict[str, dict]

    def __getitem__(self, key: str) -> ModelSpec:
        if key not in self.models:
            raise KeyError(f"modelo '{key}' fora do registro (config/models.yaml): {list(self.models)}")
        return self.models[key]

    def architecture(self, model_type: str) -> Architecture:
        if model_type not in self.architectures:
            raise ValueError(
                f"Arquitetura '{model_type}' não mapeada. Adicione a `architectures` em "
                f"config/models.yaml (inspecione model.config.model_type e model.model).")
        return self.architectures[model_type]


def last_word(text: str) -> str:
    """Última palavra do prompt (a preposição da lacuna)."""
    return text.rstrip().split()[-1]


# ══════════════════════════════════════════════════════════════
# Leitura
# ══════════════════════════════════════════════════════════════

def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=None)
def load_experiment(config_dir: str = str(CONFIG_DIR)) -> dict:
    return _read_yaml(Path(config_dir) / "experiment.yaml")


def _norm_names(raw) -> Dict[str, Dict[str, List[str]]]:
    out = {}
    for lang, forms in (raw or {}).items():
        out[lang] = dict(forms) if isinstance(forms, dict) else {BASE_CASE: list(forms)}
    return out


def parse_facts(raw: dict) -> Dict[str, Fact]:
    facts = {}
    for key, f in (raw.get("facts") or {}).items():
        cands = [Candidate(
            id=c["id"], label=c.get("label", c["id"]),
            dominant=bool(c.get("dominant", False)),
            stakes=[Stake(**s) for s in (c.get("stakes") or [])],
            names=_norm_names(c.get("names")),
            indicators={t: dict(v) for t, v in (c.get("indicators") or {}).items()},
            plural=bool(c.get("plural", False)),
            notes=list(c.get("notes") or []),
        ) for c in f.get("candidates") or []]
        facts[key] = Fact(key=key, kind=f.get("kind"), status=f.get("status"),
                          verb=f.get("verb"), keyword=dict(f.get("keyword") or {}),
                          candidates=cands,
                          empirical_dominant=f.get("empirical_dominant"))
    return facts


@lru_cache(maxsize=None)
def load_facts(config_dir: str = str(CONFIG_DIR)) -> Dict[str, Fact]:
    return parse_facts(_read_yaml(Path(config_dir) / "facts.yaml"))


def _parse_block(fact: Optional[str], lang: str, block: dict) -> List[Prompt]:
    """Bloco {needs_native_review, tipo: [paráfrases]} de um idioma."""
    out    = []
    review = bool(block.get("needs_native_review", False))
    for ptype, items in block.items():
        if ptype == "needs_native_review":
            continue
        for i, item in enumerate(items, start=1):
            out.append(Prompt(
                fact=fact, lang=lang, type=ptype, paraphrase=i,
                text=item["text"],
                slot_case=item.get("slot_case", BASE_CASE),
                article_blocked=bool(item.get("article_blocked", False)),
                needs_native_review=review))
    return out


def parse_prompts(raw: dict) -> List[Prompt]:
    out = []
    for fact, by_lang in (raw.get("prompts") or {}).items():
        for lang, block in by_lang.items():
            out += _parse_block(fact, lang, block)
    return out


def parse_baseline(raw: dict) -> Baseline:
    prompts = []
    for lang, block in (raw.get("baseline") or {}).items():
        prompts += _parse_block(None, lang, block)
    return Baseline(status=raw.get("status"), prompts=tuple(prompts))


@lru_cache(maxsize=None)
def load_prompts(config_dir: str = str(CONFIG_DIR)) -> tuple:
    return tuple(parse_prompts(_read_yaml(Path(config_dir) / "prompts.yaml")))


@lru_cache(maxsize=None)
def load_baseline(config_dir: str = str(CONFIG_DIR)) -> Baseline:
    return parse_baseline(_read_yaml(Path(config_dir) / "baseline_prompts.yaml"))


def parse_forcing(raw: dict) -> Forcing:
    return Forcing(
        status=raw.get("status"),
        articles={l: list(v or []) for l, v in (raw.get("articles") or {}).items()},
        plural_nouns={l: _norm_names({l: v})[l] for l, v in (raw.get("plural_nouns") or {}).items()},
        contractions={l: {(k.casefold() if isinstance(k, str) else k): list(f)
                          for k, f in (v or {}).items()}
                      for l, v in (raw.get("contractions") or {}).items()},
    )


@lru_cache(maxsize=None)
def load_forcing(config_dir: str = str(CONFIG_DIR)) -> Forcing:
    return parse_forcing(_read_yaml(Path(config_dir) / "forcing.yaml"))


def parse_models(raw: dict) -> Registry:
    archs = {t: Architecture(model_type=t, **(v or {}))
             for t, v in (raw.get("architectures") or {}).items()}
    models = {k: ModelSpec(key=k, **v) for k, v in (raw.get("models") or {}).items()}
    return Registry(models=models, architectures=archs,
                    quantization={k: dict(v) for k, v in (raw.get("quantization") or {}).items()})


@lru_cache(maxsize=None)
def load_models(config_dir: str = str(CONFIG_DIR)) -> Registry:
    return parse_models(_read_yaml(Path(config_dir) / "models.yaml"))


def prompts_for(prompts, fact: str = None, lang: str = None, ptype: str = None) -> List[Prompt]:
    return [p for p in prompts
            if (fact is None or p.fact == fact)
            and (lang is None or p.lang == lang)
            and (ptype is None or p.type == ptype)]


# ══════════════════════════════════════════════════════════════
# Validação
# ══════════════════════════════════════════════════════════════

def validate(exp: dict, facts: Dict[str, Fact], prompts) -> List[str]:
    """Lista de problemas (vazia se tudo consistente)."""
    errs   = []
    langs  = list(exp["languages"])
    n_par  = exp["paraphrases"]
    review = set(exp["native_review_languages"])

    for key, f in facts.items():
        where = f"facts.{key}"
        if f.kind not in FACT_KINDS:
            errs.append(f"{where}: kind '{f.kind}' inválido")
        if f.status not in STATUSES:
            errs.append(f"{where}: status '{f.status}' inválido")
        if sum(c.dominant for c in f.candidates) != 1:
            errs.append(f"{where}: precisa de exatamente um candidato dominant")
        if f.kind == "control" and len(f.candidates) != 1:
            errs.append(f"{where}: controle deve ter um único candidato")
        if f.kind == "disputed" and len(f.candidates) < 2:
            errs.append(f"{where}: fato disputado precisa de >= 2 candidatos")
        ids = [c.id for c in f.candidates]
        if len(ids) != len(set(ids)):
            errs.append(f"{where}: ids de candidato repetidos")
        for lang in langs:
            if lang not in f.keyword:
                errs.append(f"{where}: keyword sem '{lang}'")

        for c in f.candidates:
            cw = f"{where}.{c.id}"
            for s in c.stakes:
                if s.language not in langs:
                    errs.append(f"{cw}: stake em idioma fora do escopo '{s.language}'")
                if s.type not in STAKE_TYPES:
                    errs.append(f"{cw}: tipo de stake '{s.type}' inválido")
            for lang in langs:
                if lang not in c.names or not any(c.names[lang].values()):
                    errs.append(f"{cw}: sem nomes em '{lang}'")
            ru = c.names.get("ru", {})
            if "ru" in langs and ru and set(ru) != RU_CASES:
                errs.append(f"{cw}: nomes ru precisam dos casos {sorted(RU_CASES)}")
            for t, by_lang in c.indicators.items():
                if t not in exp["indicator_types"]:
                    errs.append(f"{cw}: tipo de indicador '{t}' inválido")
                for lang in langs:
                    if lang not in by_lang:
                        errs.append(f"{cw}: indicador '{t}' sem '{lang}'")

    # Prompts: cobertura completa e consistente
    seen = {}
    for p in prompts:
        seen.setdefault((p.fact, p.lang, p.type), []).append(p)
        if p.fact not in facts:
            errs.append(f"prompts.{p.fact}: fato desconhecido")
        if p.lang == "ru" and p.slot_case not in RU_CASES:
            errs.append(f"prompts.{p.fact}.ru.{p.type}: slot_case '{p.slot_case}' inválido")
        if p.lang != "ru" and p.slot_case != BASE_CASE:
            errs.append(f"prompts.{p.fact}.{p.lang}.{p.type}: slot_case só em ru")
        if p.lang not in review and p.needs_native_review:
            errs.append(f"prompts.{p.fact}.{p.lang}: needs_native_review só para {sorted(review)}")
    for key, f in facts.items():
        types = list(exp["name_types"])
        if f.kind == "disputed":
            types += f.indicator_types()
        for lang in langs:
            for t in types:
                got = seen.get((key, lang, t), [])
                if len(got) != n_par:
                    errs.append(f"prompts.{key}.{lang}.{t}: {len(got)} paráfrases (esperado {n_par})")
        extra = {t for (k, _, t) in seen if k == key} - set(types)
        if extra:
            errs.append(f"prompts.{key}: tipos sem indicadores correspondentes {sorted(extra)}")
    return errs


def validate_baseline(exp: dict, baseline: Baseline, prompts) -> List[str]:
    """
    Linha de base: cobertura idioma × tipo × paráfrase, e espelhamento
    dos prompts reais (a paráfrase i neutra tem o mesmo slot_case e o
    mesmo article_blocked da paráfrase i real, em todos os fatos).
    """
    errs   = []
    langs  = list(exp["languages"])
    types  = list(exp["name_types"]) + list(exp["indicator_types"])
    review = set(exp["native_review_languages"])
    if baseline.status not in STATUSES:
        errs.append(f"baseline: status '{baseline.status}' inválido")

    real = {}
    for p in prompts:
        real.setdefault((p.lang, p.type, p.paraphrase), set()).add(
            (p.slot_case, p.article_blocked))
    got = {}
    for b in baseline.prompts:
        where = f"baseline.{b.lang}.{b.type}.{b.paraphrase}"
        got.setdefault((b.lang, b.type), []).append(b)
        if b.lang not in langs:
            errs.append(f"{where}: idioma fora do escopo")
        if b.type not in types:
            errs.append(f"{where}: tipo '{b.type}' inválido")
        if b.lang != "ru" and b.slot_case != BASE_CASE:
            errs.append(f"{where}: slot_case só em ru")
        if b.lang not in review and b.needs_native_review:
            errs.append(f"{where}: needs_native_review só para {sorted(review)}")
        shapes = real.get((b.lang, b.type, b.paraphrase), set())
        if shapes and shapes != {(b.slot_case, b.article_blocked)}:
            errs.append(f"{where}: não espelha os prompts reais "
                        f"(slot_case, article_blocked) reais={sorted(shapes)} "
                        f"neutro={(b.slot_case, b.article_blocked)}")
    for lang in langs:
        for t in types:
            n = len(got.get((lang, t), []))
            if n != exp["paraphrases"]:
                errs.append(f"baseline.{lang}.{t}: {n} paráfrases (esperado {exp['paraphrases']})")
    return errs


def validate_forcing(exp: dict, forcing: Forcing, prompts) -> List[str]:
    """
    Todos os idiomas presentes; ru com os três casos; toda preposição
    final de prompt article_blocked tem formas contraídas.
    """
    errs  = []
    langs = list(exp["languages"])
    if forcing.status not in STATUSES:
        errs.append(f"forcing: status '{forcing.status}' inválido")
    # YAML lê no/yes/on/off como booleanos: toda palavra tem que ser texto
    words = ([w for v in forcing.articles.values() for w in v]
             + [w for v in forcing.plural_nouns.values() for c in v.values() for w in c]
             + [w for v in forcing.contractions.values() for f in v.values() for w in f]
             + [k for v in forcing.contractions.values() for k in v])
    for w in words:
        if not isinstance(w, str):
            errs.append(f"forcing: valor {w!r} não é texto (coloque entre aspas no YAML)")
    for lang in langs:
        if lang not in forcing.articles:
            errs.append(f"forcing.articles: sem '{lang}'")
        if lang not in forcing.plural_nouns:
            errs.append(f"forcing.plural_nouns: sem '{lang}'")
    if "ru" in forcing.plural_nouns and set(forcing.plural_nouns["ru"]) != RU_CASES:
        errs.append(f"forcing.plural_nouns.ru: precisa dos casos {sorted(RU_CASES)}")
    for p in prompts:
        if p.article_blocked and not forcing.contraction_forms(p.lang, last_word(p.text)):
            errs.append(f"forcing.contractions.{p.lang}: sem formas para "
                        f"'{last_word(p.text)}' ({p.fact or 'baseline'} {p.type}.{p.paraphrase})")
    return sorted(set(errs))


MODEL_KINDS = {"base", "instruct"}
PRECISIONS  = {"bfloat16", "float16", "float32"}


def validate_models(exp: dict, reg: Registry) -> List[str]:
    errs = []
    for k, m in reg.models.items():
        w = f"models.{k}"
        if m.kind not in MODEL_KINDS:
            errs.append(f"{w}: kind '{m.kind}' inválido")
        if m.architecture not in reg.architectures:
            errs.append(f"{w}: arquitetura '{m.architecture}' sem entrada em architectures")
        if m.precision not in PRECISIONS:
            errs.append(f"{w}: precision '{m.precision}' inválida")
        if m.quantization is not None and m.quantization not in reg.quantization:
            errs.append(f"{w}: quantization '{m.quantization}' sem entrada em quantization")
        if m.base_model is not None and m.base_model not in reg.models:
            errs.append(f"{w}: base_model '{m.base_model}' fora do registro")
        if not (isinstance(m.revision, str) and len(m.revision) == 40
                and all(c in "0123456789abcdef" for c in m.revision)):
            errs.append(f"{w}: revision precisa ser o hash completo do commit (40 hex)")
        if m.adaptation is not None and m.adaptation not in exp["languages"]:
            errs.append(f"{w}: adaptation '{m.adaptation}' fora dos idiomas")
    for a in reg.architectures.values():
        if not a.final_norm or not a.unembedding:
            errs.append(f"architectures.{a.model_type}: final_norm e unembedding obrigatórios")
    for k in exp.get("heatmap_models", []):
        if k not in reg.models:
            errs.append(f"experiment.heatmap_models: '{k}' fora do registro")
    return errs


def require_valid(config_dir: str = str(CONFIG_DIR)):
    prompts = load_prompts(config_dir)
    errs = validate(load_experiment(config_dir), load_facts(config_dir), prompts)
    baseline = load_baseline(config_dir)
    errs += validate_baseline(load_experiment(config_dir), baseline, prompts)
    errs += validate_forcing(load_experiment(config_dir), load_forcing(config_dir),
                             list(prompts) + list(baseline.prompts))
    errs += validate_models(load_experiment(config_dir), load_models(config_dir))
    if errs:
        raise ConfigError("Configuração inválida:\n  " + "\n  ".join(errs))


def require_confirmed(facts: Dict[str, Fact], prompts,
                      baseline: Optional[Baseline] = None,
                      forcing: Optional[Forcing] = None) -> None:
    """
    Falha se algum fato ainda não foi confirmado, algum prompt espera
    revisão nativa ou (se dada) a linha de base não foi confirmada.
    Todo script que roda modelos chama isto antes.
    """
    pend_facts  = sorted(k for k, f in facts.items() if f.status != "confirmed")
    pend_review = sorted({(p.fact, p.lang) for p in prompts if p.needs_native_review})
    pend_base   = []
    if baseline is not None:
        if baseline.status != "confirmed":
            pend_base.append("status != confirmed")
        langs_b = sorted({p.lang for p in baseline.prompts if p.needs_native_review})
        if langs_b:
            pend_base.append(f"revisão nativa pendente {langs_b}")
    pend_force = forcing is not None and forcing.status != "confirmed"
    if pend_facts or pend_review or pend_base or pend_force:
        langs = sorted({l for _, l in pend_review})
        raise UnconfirmedError(
            "Valores não confirmados — runs bloqueados.\n"
            f"  fatos com status != confirmed: {pend_facts}\n"
            f"  prompts com revisão nativa pendente: {len(pend_review)} blocos "
            f"(idiomas {langs})\n"
            f"  linha de base: {'; '.join(pend_base) or 'ok'}\n"
            f"  forcing (artigos/contrações): {'status != confirmed' if pend_force else 'ok'}\n"
            "  Veja reports/phase1_review.md, reports/prompts_review.md e "
            "reports/verified_values.md."
        )
