"""
tests/synthetic_results.py
Resultados sintéticos realistas para testar figuras e tabelas: tabelas
brutas inventadas (probs por camada, respostas) passam pelo analyze.py
e pela estatística de verdade, num config reduzido.
"""

import dataclasses

import numpy as np
import pandas as pd

import analyze
from analysis.statistics import run_statistics
from config.experiment_config import ExperimentConfig

FACTS  = ("aviao", "telefone", "penicilina")
MODELS = ("llama", "mistral", "qwen", "gemma", "llama_base")
LAYERS = {"llama": 6, "mistral": 6, "qwen": 5, "gemma": 7, "llama_base": 6}


def small_config():
    base = ExperimentConfig()
    stats = {**base.experiment["statistics"], "bootstrap": 200, "permutation_mc": 2000}
    exp = {**base.experiment, "statistics": stats, "heatmap_models": ["llama", "gemma"]}
    reg = dataclasses.replace(base.registry, models={k: base.registry[k] for k in MODELS})
    return ExperimentConfig(
        experiment=exp, fact_specs={k: base.fact_specs[k] for k in FACTS},
        prompt_specs=tuple(p for p in base.prompt_specs if p.fact in FACTS),
        baseline_spec=base.baseline_spec, forcing_spec=base.forcing_spec, registry=reg)


def probs_table(cfg, rng):
    rows = []
    for model in MODELS:
        L = LAYERS[model]
        instruct = cfg.registry[model].kind == "instruct"
        for lens in ("logit_lens", "tuned_lens") if instruct else ("logit_lens",):
            for p in cfg.prompt_specs:
                if p.format == "qa" and not instruct:
                    continue
                fact = cfg.fact_specs[p.fact]
                kind = "indicators" if p.type in ("place", "artifact") else "names"
                for c in fact.candidates:
                    if kind == "indicators" and not c.indicators_for(p.type, p.lang):
                        continue
                    stake = p.lang in c.stake_languages()
                    peak = (0.5 if c.dominant else 0.15 + (0.25 if stake else 0)) + rng.normal(0, 0.03)
                    for k in range(1, L + 1):
                        shape = np.exp(-((k - 0.7 * L) ** 2) / L)
                        prob = float(np.clip(peak * shape + rng.normal(0, 0.01), 0, 1))
                        rows.append(dict(model=model, lens=lens, fact=p.fact, lang=p.lang,
                                         format=p.format, prompt_type=p.type, paraphrase=p.paraphrase,
                                         slot_case=p.slot_case, article_blocked=p.article_blocked,
                                         candidate=c.id, token_set=kind, layer=k, depth=k / L,
                                         prob=prob, n_ids=1, dominant=c.dominant, n_layers=L))
    return pd.DataFrame(rows)


def answers_table(cfg, rng):
    rows = []
    for model in MODELS:
        if cfg.registry[model].kind != "instruct":
            continue
        for p in cfg.prompt_specs:
            if p.type != "QA":
                continue
            fact = cfg.fact_specs[p.fact]
            labels = {c.id: c.label for c in fact.candidates}
            for sid in range(3):
                local = [c.id for c in fact.alternatives if p.lang in c.stake_languages()]
                first = local[0] if local and rng.random() < 0.6 else fact.dominant.id
                text = f"{labels[first]} is usually credited."
                rows.append(dict(model=model, fact=p.fact, lang=p.lang, prompt_type="QA",
                                 paraphrase=p.paraphrase, sample_id=sid,
                                 kind="greedy" if sid == 0 else "sample", question=p.text,
                                 text=text, n_new_tokens=8, finished=True, seed=1))
    return pd.DataFrame(rows)


def build(seed=0):
    """(config, tabelas de métricas) como analyze.main as gravaria."""
    rng = np.random.default_rng(seed)
    cfg = small_config()
    res = analyze.compute_all({"probs": probs_table(cfg, rng)}, cfg)
    from config.loader import load_behavioral
    b = dataclasses.replace(load_behavioral(), status="confirmed", ru_needs_native_review=False)
    res.update(analyze.compute_behavioral(answers_table(cfg, rng), cfg, b))
    res.update(run_statistics(res, cfg))
    return cfg, res


def eval_reports(report_dir, models=("llama", "mistral", "qwen", "gemma")):
    report_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        L = LAYERS[m]
        pd.DataFrame([{"lang": l, "layer": k, "n_tokens": 100,
                       "kl_logit": 5.0 / k, "kl_tuned": 1.0 / k}
                      for l in ("pt", "en") for k in range(1, L)]).to_csv(
            report_dir / f"{m}_eval.csv", index=False)
