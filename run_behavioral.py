"""
run_behavioral.py
Estudo comportamental (Fase 5): respostas dos modelos instruct aos
prompts QA, com chat template e sem system prompt.

    python run_behavioral.py --model llama
    python run_behavioral.py --model gemma27 --gpus "1 2"   # bf16 em duas GPUs menores
    python run_behavioral.py --model llama --consolidate-only

Por prompt: 1 resposta greedy + 10 amostras a T=0.7 (temperatura pura,
top_p=1, top_k=0), max_new_tokens=200, semente fixa por prompt
(config/behavioral.yaml). Retoma de checkpoint: unidades (fato × idioma)
feitas são puladas.

Saída em results/behavioral/{modelo}/:
    answers.parquet   modelo × fato × idioma × paráfrase × amostra: texto,
                      tokens gerados, se terminou no eos, semente
    manifest.json     modelo, commit, configuração efetiva de geração
                      (depois da mistura com a do modelo), versões, GPU, tempo

A extração (candidatos, indicadores) é feita por analyze.py.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

from behavior.generation import Generator, effective_config
from config.experiment_config import ExperimentConfig
from config.loader import (
    load_behavioral, load_prompts_meta, prompts_for, require_confirmed, require_valid,
)
from logit_lens.store import ChunkStore, behavioral_dir, read_manifest, write_manifest
from models.runtime import (
    add_model_args, apply_gpus, check_resume, precision_tag, registry_with_dtype,
)
from run_lens import _config_hashes, _gpu, _quantization

TABLES = ("answers",)


def behavioral_units(config):
    return [f"{f}__{l}" for f in config.facts for l in config.languages
            if prompts_for(config.prompt_specs, f, l, "QA")]


def run_model(model_key, model, tokenizer, config, out_dir, chat_formatter, behavioral=None,
              spec=None, log=print):
    """Gera as respostas que faltam e consolida. Testes chamam direto."""
    b   = behavioral or load_behavioral()
    gen = Generator(model, tokenizer, chat_formatter, b.generation, config.seed)
    store = ChunkStore(out_dir, TABLES)
    manifest = read_manifest(out_dir) or {
        "model_key": model_key, "model_id": spec.id if spec else None,
        "revision": getattr(model.config, "_commit_hash", None),
        "registry_revision": spec.revision if spec else None,
        "precision_tag": precision_tag(spec) if spec else None,
        "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "quantization": _quantization(model),
        "generation": b.generation,
        "effective_greedy": effective_config(model, gen.greedy_cfg),
        "effective_sample": effective_config(model, gen.sample_cfg),
        "seed": config.seed, "gpu": _gpu(), "config_sha256": _config_hashes(),
        "sessions": []}

    units = behavioral_units(config)
    todo  = [u for u in units if not store.done(u)]
    log(f"  {len(units) - len(todo)}/{len(units)} unidades já feitas; {len(todo)} a rodar")
    t0, started = time.time(), datetime.now(timezone.utc).isoformat()
    for i, unit in enumerate(todo, start=1):
        fact, lang = unit.split("__")
        u0, rows = time.time(), []
        for p in prompts_for(config.prompt_specs, fact, lang, "QA"):
            rows += gen.answers(p, model_key)
        store.write(unit, {"answers": rows})
        log(f"  [{i}/{len(todo)}] {unit}  {len(rows)} respostas ({time.time() - u0:.1f}s)")

    counts = store.consolidate(units)
    manifest["sessions"].append({"started": started,
                                 "finished": datetime.now(timezone.utc).isoformat(),
                                 "units_run": len(todo), "wall_time_s": round(time.time() - t0, 1)})
    manifest["wall_time_s"] = round(sum(s["wall_time_s"] for s in manifest["sessions"]), 1)
    gpus = manifest["gpu"]["count"]
    manifest["gpu_hours"] = round(manifest["wall_time_s"] / 3600 * gpus, 3) if gpus else 0.0
    manifest["units_done"], manifest["units_expected"] = len(store.done_units()), len(units)
    manifest["rows"] = counts
    write_manifest(out_dir, manifest)
    return manifest


def main(argv=None):
    config = ExperimentConfig()
    instruct = [k for k, m in config.registry.models.items() if m.kind == "instruct"]
    parser = argparse.ArgumentParser(description="Respostas dos modelos instruct aos prompts QA.")
    add_model_args(parser, instruct)
    parser.add_argument("--results-dir", default=config.results_dir)
    parser.add_argument("--consolidate-only", action="store_true")
    args = parser.parse_args(argv)

    out = behavioral_dir(args.results_dir, args.model)
    if args.consolidate_only:
        print(ChunkStore(out, TABLES).consolidate())
        return 0

    apply_gpus(args.gpus)
    registry = registry_with_dtype(config.registry, args.model, args.dtype)
    spec = registry[args.model]
    require_valid()
    # Prompts QA e fatos confirmados; as formas de superfície só são
    # exigidas na extração (analyze.py)
    require_confirmed(config.fact_specs, config.prompt_specs,
                      prompts_status=load_prompts_meta()["status"])
    check_resume(read_manifest(out), spec, out)
    store = ChunkStore(out, TABLES)
    if (out / "answers.parquet").exists() and all(store.done(u) for u in behavioral_units(config)):
        print(f"  ⏭  {args.model}: todas as unidades comportamentais já feitas; nada a refazer, "
              f"modelo não carregado")
        return 0

    from models.chat_template import apply_template
    from models.model_loader import load_model
    model, tok = load_model(args.model, registry)
    chat = lambda t, x, mk=args.model: apply_template(t, mk, x)[0]   # noqa: E731
    m = run_model(args.model, model, tok, config, out, chat, spec=spec)
    print(f"\n✅ {args.model}: {m['units_done']}/{m['units_expected']} unidades, "
          f"{m['wall_time_s'] / 60:.1f} min  →  {out}")
    return 0


if __name__ == "__main__":
    # Sai direto depois de gravar tudo: a finalização do interpretador
    # (threads, CUDA) já travou um job por 15 h depois de terminar o
    # trabalho (2026-10-29). Todos os arquivos estão fechados aqui.
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc or 0)
