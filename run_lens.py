"""
run_lens.py
Medição da Fase 2 para um modelo (substitui run_single_model.py).

    python run_lens.py llama
    python run_lens.py llama --consolidate-only

Passos:
    1. configuração válida e confirmada (fatos, prompts, revisão nativa,
       linha de base, forcing); senão nada roda
    2. teste de sanidade do lens (última camada == logits do modelo)
    3. linha de base neutra, uma unidade por idioma
    4. prompts de cada fato × idioma, com as passadas forçadas do teste
       do confundidor de artigo
    5. tabelas consolidadas + manifest

Formatos: completion sempre; QA só se o tokenizer tiver chat template
(modelos instruct). Retoma de checkpoint: unidades feitas são puladas.

Saída em results/lens/{modelo}/ (ver logit_lens/store.py):
    probs.parquet       modelo × fato × idioma × formato × tipo × paráfrase
                        × camada × candidato × conjunto (names/indicators)
    baseline.parquet    o mesmo para os prompts neutros (coluna fact = fato
                        dos candidatos medidos)
    forced.parquet      passadas forçadas (artigo, contração, prefixo)
    layers.parquet      KL até a final e top-20 por camada
    token_sets.parquet  auditoria dos token IDs de cada conjunto
    manifest.json       modelo, revisão, dtype, quantização, versões, GPU,
                        tempo
"""

import argparse
import hashlib
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from config.experiment_config import ExperimentConfig
from config.loader import CONFIG_DIR, prompts_for, require_confirmed, require_valid
from logit_lens.confound import ConfoundTester
from logit_lens.lens import LogitLens
from logit_lens.measure import LensRunner
from logit_lens.store import ChunkStore, lens_dir, read_manifest, write_manifest

CONFIG_FILES = ("experiment.yaml", "facts.yaml", "prompts.yaml",
                "baseline_prompts.yaml", "forcing.yaml")


# ══════════════════════════════════════════════════════════════
# Linhas das tabelas
# ══════════════════════════════════════════════════════════════

def _keys(model_key, lens, prompt, fact_key):
    return {"model": model_key, "lens": lens, "fact": fact_key, "lang": prompt.lang,
            "format": prompt.format, "prompt_type": prompt.type,
            "paraphrase": prompt.paraphrase, "slot_case": prompt.slot_case,
            "article_blocked": prompt.article_blocked}


def _dominant(facts, fact_key, cid):
    return facts[fact_key].dominant.id == cid


def _prob_rows(keys, res, facts, extra=None):
    extra = extra or {}
    return [{**keys, **extra, **r,
             "fact": r.get("fact", keys["fact"]),
             "dominant": _dominant(facts, r.get("fact", keys["fact"]), r["candidate"]),
             "n_layers": res["n_layers"]} for r in res["probs"]]


def _layer_rows(keys, res, source):
    return [{**keys, "source": source, "n_layers": res["n_layers"],
             "n_input_tokens": res["n_input_tokens"], **r} for r in res["layers"]]


def _token_rows(keys, token_sets, source):
    shared = {d["token_id"] for d in token_sets["shared_dropped"]}
    rows = []
    for cid, info in token_sets["sets"].items():
        kept = set(info["ids"])
        for d in info["words"]:
            rows.append({**keys, "source": source, "candidate": cid,
                         "word": d["word"], "token_id": d["token_id"], "token": d["token"],
                         "n_word_tokens": d["n_word_tokens"], "split": d["split"],
                         "boundary_merge": d["boundary_merge"],
                         "kept": d["token_id"] in kept, "shared": d["token_id"] in shared})
    return rows


# ══════════════════════════════════════════════════════════════
# Unidades
# ══════════════════════════════════════════════════════════════

def lens_units(config: ExperimentConfig, formats=("completion", "qa")):
    """Unidades esperadas: linha de base por idioma + fato × idioma."""
    units = [f"baseline__{l}" for l in config.languages]
    units += [f"{f}__{l}" for f in config.facts for l in config.languages
              if any(p.format in formats for p in prompts_for(config.prompt_specs, f, l))]
    return units


def run_baseline_unit(runner, config, lang, model_key, formats):
    facts  = config.fact_specs
    tables = {"baseline": [], "layers": [], "token_sets": []}
    for neutral in prompts_for(config.baseline_spec.prompts, None, lang):
        if neutral.format not in formats:
            continue
        res  = runner.measure_baseline(neutral, facts)
        keys = _keys(model_key, runner.lens.name, neutral, None)
        tables["baseline"]   += _prob_rows(keys, res, facts)
        tables["layers"]     += _layer_rows(keys, res, "baseline")
        for fk, ts in res["token_sets"].items():
            tables["token_sets"] += _token_rows({**keys, "fact": fk}, ts, "baseline")
    return tables


def run_fact_unit(runner, tester, config, fact_key, lang, model_key, formats):
    facts  = config.fact_specs
    fact   = facts[fact_key]
    tables = {"probs": [], "forced": [], "layers": [], "token_sets": []}
    for prompt in prompts_for(config.prompt_specs, fact_key, lang):
        if prompt.format not in formats:
            continue
        res  = runner.measure(prompt, fact)
        keys = _keys(model_key, runner.lens.name, prompt, fact_key)
        tables["probs"]      += _prob_rows(keys, res, facts)
        tables["layers"]     += _layer_rows(keys, res, "prompt")
        tables["token_sets"] += _token_rows(keys, res["token_sets"], "prompt")
        for ps in tester.run(prompt, fact, res):
            n_steps = len(ps["steps"])
            for st in ps["steps"]:
                tables["forced"] += _prob_rows(keys, st["result"], facts, extra={
                    "forced_kind": ps["kind"], "pass_id": ps["pass_id"],
                    "force_step": st["step"], "n_steps": n_steps,
                    "forced_suffix": st["suffix"], "trigger_token": st["trigger_token"]})
    return tables


# ══════════════════════════════════════════════════════════════
# Manifest
# ══════════════════════════════════════════════════════════════

def _config_hashes():
    return {f: hashlib.sha256((CONFIG_DIR / f).read_bytes()).hexdigest() for f in CONFIG_FILES}


def _quantization(model):
    q = getattr(model.config, "quantization_config", None)
    if q is None:
        return None
    return q.to_dict() if hasattr(q, "to_dict") else dict(q)


def _gpu():
    if not torch.cuda.is_available():
        return {"name": "cpu", "count": 0}
    return {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count()}


def base_manifest(model_key, model_id, model, lens, formats, sanity):
    return {
        "model_key":    model_key,
        "model_id":     model_id,
        "revision":     getattr(model.config, "_commit_hash", None),
        "model_type":   model.config.model_type,
        "n_layers":     model.config.num_hidden_layers,
        "dtype":        str(next(model.parameters()).dtype).replace("torch.", ""),
        "quantization": _quantization(model),
        "lens":         lens.name,
        "formats":      list(formats),
        "versions":     {"transformers": transformers.__version__, "torch": torch.__version__,
                         "python": platform.python_version()},
        "gpu":          _gpu(),
        "config_sha256": _config_hashes(),
        "sanity":       {"max_abs_diff": sanity["max_abs_diff"]},
        "sessions":     [],
    }


# ══════════════════════════════════════════════════════════════
# Execução
# ══════════════════════════════════════════════════════════════

def run_model(model_key, model, tokenizer, config, out_dir, model_id=None,
              lens=None, chat_formatter=None, log=print):
    """
    Roda todas as unidades que faltam e consolida. Não checa
    confirmação (main() faz isso); testes chamam direto.

    Returns:
        manifest (dict)
    """
    lens    = lens or LogitLens(model)
    formats = ("completion", "qa") if chat_formatter is not None else ("completion",)
    runner  = LensRunner(model, tokenizer, lens=lens, top_k=config.top_k_tokens,
                         chat_formatter=chat_formatter)
    tester  = ConfoundTester(runner, config.forcing_spec)
    store   = ChunkStore(out_dir)

    # Falha antes de gravar qualquer resultado se o lens não reproduzir
    # os logits do modelo na última camada
    sanity   = lens.sanity_check(tokenizer)
    manifest = read_manifest(out_dir) or base_manifest(
        model_key, model_id, model, lens, formats, sanity)
    if manifest.get("config_sha256") != _config_hashes():
        log("  ⚠️  config mudou desde o início deste modelo; unidades novas usam o config atual")
        manifest.setdefault("config_changes", []).append(_config_hashes())

    t0, started = time.time(), datetime.now(timezone.utc).isoformat()
    units = lens_units(config, formats)
    todo  = [u for u in units if not store.done(u)]
    log(f"  {len(units) - len(todo)}/{len(units)} unidades já feitas; {len(todo)} a rodar")

    for i, unit in enumerate(todo, start=1):
        fact_key, lang = unit.split("__")
        u0 = time.time()
        if fact_key == "baseline":
            tables = run_baseline_unit(runner, config, lang, model_key, formats)
        else:
            tables = run_fact_unit(runner, tester, config, fact_key, lang, model_key, formats)
        store.write(unit, tables)
        log(f"  [{i}/{len(todo)}] {unit}  ({time.time() - u0:.1f}s)")

    counts  = store.consolidate(units)
    elapsed = time.time() - t0
    manifest["sessions"].append({"started": started,
                                 "finished": datetime.now(timezone.utc).isoformat(),
                                 "units_run": len(todo), "wall_time_s": round(elapsed, 1)})
    manifest["wall_time_s"] = round(sum(s["wall_time_s"] for s in manifest["sessions"]), 1)
    gpus = manifest["gpu"]["count"]
    manifest["gpu_hours"] = round(manifest["wall_time_s"] / 3600 * gpus, 3) if gpus else 0.0
    manifest["units_done"] = len(store.done_units())
    manifest["units_expected"] = len(units)
    manifest["rows"] = counts
    write_manifest(out_dir, manifest)
    return manifest


def main(argv=None):
    config = ExperimentConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(config.models))
    parser.add_argument("--results-dir", default=config.results_dir)
    parser.add_argument("--consolidate-only", action="store_true")
    args = parser.parse_args(argv)

    out_dir = lens_dir(args.results_dir, args.model)
    if args.consolidate_only:
        print(ChunkStore(out_dir).consolidate())
        return 0

    # Nada roda com valores não confirmados
    require_valid()
    require_confirmed(config.fact_specs, config.prompt_specs,
                      config.baseline_spec, config.forcing_spec)

    from models.chat_template import apply_template
    from models.model_loader import load_model

    model_id = config.models[args.model]
    model, tokenizer = load_model(model_id, args.model, config.dtype)
    chat = None
    if getattr(tokenizer, "chat_template", None):
        chat = lambda tok, text, mk=args.model: apply_template(tok, mk, text)[0]
    else:
        print("  sem chat template: só formato completion")

    m = run_model(args.model, model, tokenizer, config, out_dir, model_id=model_id,
                  chat_formatter=chat)
    print(f"\n✅ {args.model}: {m['units_done']}/{m['units_expected']} unidades, "
          f"{m['wall_time_s'] / 60:.1f} min  →  {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
