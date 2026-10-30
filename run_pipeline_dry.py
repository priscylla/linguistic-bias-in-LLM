"""
run_pipeline_dry.py
Pipeline inteiro com modelos minúsculos e aleatórios, na CPU e sem rede,
para saber que todo passo funciona antes de usar a GPU.

    python run_pipeline_dry.py                 # config completo (todos os fatos e idiomas)
    python run_pipeline_dry.py --quick         # 3 fatos, geração curta (testes)
    python run_pipeline_dry.py --out dry_run   # pasta de saída (padrão: dry_run/)

Passos, com o mesmo código dos pontos de entrada:
    1. sanidade do lens          (logit_lens/sanity.py)
    2. Logit Lens                (run_lens.run_model)
    3. comportamental            (run_behavioral.run_model)
    4. Tuned Lens: treino curto em texto sintético + nova medição
    5. métricas e estatística    (analyze, analysis/statistics)
    6. figuras e tabelas         (make_figures)
    7. integridade               (check_results)
    8. amostra de validação e concordância

Modelos: "tiny_instruct" (arquitetura llama, formatos completion + QA) e
"tiny_base" (gemma2, só completion), com o tokenizer de palavras de
devtools/tiny.py. Valores ainda não confirmados (fatos, prompts, formas
de superfície) são usados mesmo assim: o relatório marca isso. Nenhum
número daqui tem significado; só o funcionamento e as contagens de
trabalho (forwards, tokens gerados) valem, para estimar o tempo na GPU.

Saída em --out: results/, figures/, tables/, reports/ e
dry_run_report.json (tempo por passo, contagens, verificações).
"""

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

import analyze
import check_results
import run_behavioral
import run_lens
from analysis.statistics import run_statistics
from config.experiment_config import ExperimentConfig
from config.loader import ModelSpec, load_behavioral, load_tuned_lens, require_valid
from devtools.tiny import WordTokenizer, build_tiny_model
from logit_lens.lens_training import evaluate, train
from logit_lens.sanity import sanity_report
from logit_lens.store import behavioral_dir, lens_dir
from logit_lens.tuned_lens import TunedLens

QUICK_FACTS = ("aviao", "telefone", "penicilina")
TINY = {"tiny_instruct": ("llama", "instruct"), "tiny_base": ("gemma2", "base")}
CHAT = staticmethod(lambda tok, text: f"<user>{text}<assistant>")


class DryTokenizer(WordTokenizer):
    """Tokenizer de palavras com BOS/EOS e chat template (para QA)."""
    bos_token_id, eos_token_id = 1, 2
    chat_template = "<user>{text}<assistant>"

    def __init__(self):
        super().__init__()
        self.vocab["</s>"], self.inv[2] = 2, "</s>"


def chat(tok, text):
    return f"<user>{text}<assistant>"


def dry_config(quick: bool) -> ExperimentConfig:
    base = ExperimentConfig()
    models = {k: ModelSpec(key=k, id=f"tiny/{k}", revision="0" * 40, family=f"Tiny {a}",
                           kind=kind, architecture=a, license="none", gated=False,
                           precision="float32")
              for k, (a, kind) in TINY.items()}
    reg = dataclasses.replace(base.registry, models=models)
    stats = dict(base.experiment["statistics"])
    if quick:
        stats.update(bootstrap=300, permutation_mc=2000)
    exp = {**base.experiment, "statistics": stats, "heatmap_models": ["tiny_instruct"],
           # largura provisória só para o dry run (figuras sem significado);
           # o valor real vem da medição em reports/latex/measure_textwidth.tex
           "figures": {**base.experiment["figures"], "main_models": list(TINY),
                       "text_width_in": base.experiment["figures"].get("text_width_in") or 5.5}}
    facts, prompts = base.fact_specs, base.prompt_specs
    if quick:
        facts = {k: v for k, v in facts.items() if k in QUICK_FACTS}
        prompts = tuple(p for p in prompts if p.fact in QUICK_FACTS)
    return ExperimentConfig(experiment=exp, fact_specs=facts, prompt_specs=prompts,
                            baseline_spec=base.baseline_spec, forcing_spec=base.forcing_spec,
                            registry=reg)


def synthetic_sources(config, n_docs=3000):
    """
    Corpus sintético por idioma: palavras dos prompts neutros da linha de
    base (não citam fatos, então o filtro do corpus não os remove).
    """
    import random
    out = {}
    for lang in config.languages:
        words = sorted({w for p in config.baseline_spec.prompts if p.lang == lang
                        for w in p.text.split()})
        rng = random.Random(lang)
        docs = [{"id": f"{lang}-{i}", "text": " ".join(rng.choice(words) for _ in range(rng.randint(8, 40)))}
                for i in range(n_docs)]
        out[lang] = (lambda d=docs: iter(d))
    return out


def workload(results: Path, models) -> dict:
    """Forwards e tokens gerados por modelo (para estimar o tempo na GPU)."""
    out = {}
    for m in models:
        w = {}
        for lens in ("logit_lens", "tuned_lens"):
            d = lens_dir(str(results), m, lens)
            if not (d / "layers.parquet").exists():
                continue
            layers = pd.read_parquet(d / "layers.parquet")
            first = layers[layers["layer"] == 1]
            forced = pd.read_parquet(d / "forced.parquet") if (d / "forced.parquet").exists() else pd.DataFrame()
            n_forced = (forced[["fact", "lang", "prompt_type", "paraphrase", "pass_id", "force_step"]]
                        .drop_duplicates().shape[0] if len(forced) else 0)
            w[lens] = {"forwards_prompts": int((first["source"] == "prompt").sum()),
                       "forwards_baseline": int((first["source"] == "baseline").sum()),
                       "forwards_forced": int(n_forced),
                       "mean_input_tokens": float(first["n_input_tokens"].mean())}
        b = behavioral_dir(str(results), m) / "answers.parquet"
        if b.exists():
            a = pd.read_parquet(b)
            w["behavioral"] = {"answers": len(a), "prompts": int(a[["fact", "lang", "paraphrase"]]
                                                                .drop_duplicates().shape[0]),
                               "generated_tokens": int(a["n_new_tokens"].sum())}
        out[m] = w
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Pipeline inteiro no modelo minúsculo (CPU).")
    p.add_argument("--out", default="dry_run")
    p.add_argument("--quick", action="store_true", help="3 fatos, geração e treino curtos")
    args = p.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    results, figures, tables, reports = (out / "results", out / "figures", out / "tables",
                                         out / "reports")
    require_valid()
    config = dry_config(args.quick)
    b = load_behavioral()
    gen = dict(b.generation)
    if args.quick:
        gen.update(samples=2, max_new_tokens=12)
    beh = dataclasses.replace(b, generation=gen, status="confirmed")

    report = {"quick": args.quick, "facts": list(config.facts), "languages": config.languages,
              "unconfirmed_values_used": True, "steps": {}, "checks": {}}
    timer = {}

    def step(name):
        timer["name"], timer["t0"] = name, time.time()
        print(f"\n▶ {name}")

    def done():
        report["steps"][timer["name"]] = round(time.time() - timer["t0"], 2)

    tok = DryTokenizer()
    models = {k: build_tiny_model(a, vocab=16384) for k, (a, _) in TINY.items()}
    log = lambda *a: None   # noqa: E731

    step("1_sanity")
    for k, m in models.items():
        rep = sanity_report(k, m, tok, config.model(k), config, chat)
        report["checks"][f"sanity_{k}"] = rep["passed"]
    done()

    step("2_logit_lens")
    for k, m in models.items():
        spec = config.model(k)
        run_lens.run_model(k, m, tok, config, lens_dir(str(results), k), model_id=spec.id,
                           chat_formatter=chat if spec.kind == "instruct" else None, spec=spec, log=log)
    done()

    instruct = [k for k in models if config.model(k).kind == "instruct"]
    step("3_behavioral")
    for k in instruct:
        run_behavioral.run_model(k, models[k], tok, config, behavioral_dir(str(results), k), chat,
                                 behavioral=beh, spec=config.model(k), log=log)
    done()

    step("4_tuned_lens")
    tl = load_tuned_lens()
    tl = {**tl, "training": {**tl["training"], "num_steps": 3 if args.quick else 10,
                             "tokens_per_step": 256, "seq_len": 64, "loss_chunk_tokens": 64,
                             "checkpoint_every": 2},
          "eval": {"tokens_per_language": 128}}
    sources = synthetic_sources(config)
    for k in instruct:
        lens = TunedLens(models[k])
        train(lens, models[k], tok, sources, tl, config.fact_specs, out / "lenses" / k, log=log)
        lens.save(out / "lenses" / k, meta={"dry_run": True})
        ev = pd.DataFrame(evaluate(lens, models[k], tok, sources, tl, config.fact_specs))
        (reports / "tuned_lens").mkdir(parents=True, exist_ok=True)
        ev.to_csv(reports / "tuned_lens" / f"{k}_eval.csv", index=False)
        spec = config.model(k)
        run_lens.run_model(k, models[k], tok, config, lens_dir(str(results), k, "tuned_lens"),
                           model_id=spec.id, lens=TunedLens.load(models[k], out / "lenses" / k),
                           chat_formatter=chat, spec=spec, log=log)
    done()

    step("5_analysis_statistics")
    from logit_lens.store import load_tables
    res = analyze.compute_all(load_tables(str(results)), config)
    res.update(analyze.compute_behavioral(analyze.load_answers(str(results)), config, beh))
    res.update(run_statistics(res, config))
    metrics_dir = results / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name, df in res.items():
        df.to_parquet(metrics_dir / f"{name}.parquet", index=False)
    report["checks"]["metrics_tables"] = sorted(res)
    done()

    step("6_figures_tables")
    import make_figures as mf
    mf.CONFIG = config
    paths = mf.make_all(mf.load_metrics(metrics_dir), mf.FIGURES, figures, ["main", "all"],
                        str(results), tables, report_dir=reports / "tuned_lens")
    report["checks"]["figures"] = len([x for x in paths if str(x).endswith(".pdf")])
    report["checks"]["tables"] = sorted(Path(x).name for x in paths if str(x).endswith(".tex"))
    done()

    step("7_check_results")
    problems = []
    for k in models:
        problems += check_results.check_model(k, str(results), config)
    for k in instruct:
        problems += check_results.check_model(k, str(results), config, "tuned_lens",
                                              lens_root=out / "lenses")
        problems += check_results.check_behavioral(k, str(results), config, beh)
    report["checks"]["check_results_problems"] = problems
    done()

    step("8_validation")
    from behavior.validation import agreement, stratified_sample
    attr = res["behavioral_attribution"]
    answers = analyze.load_answers(str(results))
    cands = {k: "; ".join(f"{c.id}={c.label}" for c in f.candidates) for k, f in config.fact_specs.items()}
    s = stratified_sample(answers, attr, b.validation["per_cell"], b.validation["seed"], cands)
    s["human_primary"], s["human_cited"] = s["auto_primary"], s["auto_cited"]   # rótulo = automático
    (reports / "validation").mkdir(parents=True, exist_ok=True)
    s.to_csv(reports / "validation" / "sample.csv", index=False)
    ag = agreement(s, {k: [c.id for c in f.candidates] for k, f in config.fact_specs.items()})
    report["checks"]["validation_agreement"] = ag.get("primary", {}).get("agreement")
    done()

    report["workload"] = workload(results, list(models))
    ok = (all(report["checks"][f"sanity_{k}"] for k in models)
          and not problems and report["checks"]["figures"] > 0
          and report["checks"]["validation_agreement"] == 1.0)
    report["ok"] = ok
    (out / "dry_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print("\n" + "\n".join(f"  {k:24s} {v:7.1f}s" for k, v in report["steps"].items()))
    print(f"\n{'✅' if ok else '❌'} dry run {'ok' if ok else 'com problemas'}: {out / 'dry_run_report.json'}")
    for p_ in problems:
        print(f"   → {p_}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
