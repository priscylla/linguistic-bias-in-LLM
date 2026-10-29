"""
run_lens.py
Medição da Fase 2 para um modelo (substitui run_single_model.py).

    python run_lens.py --model llama --sanity-only      # primeiro, em todo modelo
    python run_lens.py --model llama --lens logit
    python run_lens.py --model llama --lens tuned       # Tuned Lens (lenses/llama/)
    python run_lens.py --model gemma27 --gpus "1 2" --dtype bf16
    python run_lens.py --model llama --consolidate-only

--gpus, --dtype: ver models/runtime.py.

Passos:
    1. configuração válida e confirmada (fatos, prompts, revisão nativa,
       linha de base, forcing); senão nada roda
    2. teste de sanidade do lens (última camada == logits do modelo)
    3. linha de base neutra, uma unidade por idioma
    4. prompts de cada fato × idioma, com as passadas forçadas do teste
       do confundidor de artigo
    5. tabelas consolidadas + manifest

Modelo, commit, precisão e quantização vêm do registro
(config/models.yaml). Formatos: completion sempre; QA só em modelos
instruct (kind do registro, não a presença de chat template: alguns
tokenizers base, como o do Qwen 2.5, trazem um). Retoma de checkpoint:
unidades feitas são puladas.

Saída em results/lens/{lens}/{modelo}/ (ver logit_lens/store.py):
    probs.parquet       modelo × fato × idioma × formato × tipo × paráfrase
                        × camada × candidato × conjunto (names/indicators)
    baseline.parquet    o mesmo para os prompts neutros (coluna fact = fato
                        dos candidatos medidos)
    forced.parquet      passadas forçadas (artigo, contração)
    prefix_forcing.parquet  cada primeiro token compartilhado resolvido por
                        prefixo forçado (ou não resolvido), por palavra
    skipped.parquet     células de indicador puladas (regra dos dois candidatos)
    layers.parquet      KL até a final e top-20 por camada
    token_sets.parquet  auditoria dos token IDs de cada conjunto
    manifest.json       modelo, revisão, dtype, quantização, versões, GPU,
                        tempo
"""

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import torch
import transformers

from config.experiment_config import ExperimentConfig
from config.loader import (
    CONFIG_DIR, load_models, load_prompts_meta, prompts_for, require_confirmed, require_valid,
)
from logit_lens.confound import ConfoundTester
from logit_lens.entity_tokens import (
    INDICATOR_TYPES, build_token_sets, indicator_cell_ok, resolve_shared_prefixes,
    valid_candidates, word_sets,
)
from logit_lens.lens import LogitLens
from logit_lens.measure import LensRunner
from logit_lens.store import ChunkStore, lens_dir, read_manifest, write_manifest

from models.runtime import (  # noqa: E402
    add_model_args, apply_gpus, check_resume, precision_tag, registry_with_dtype,
)

LENS_CHOICES = {"logit": "logit_lens", "tuned": "tuned_lens"}
SANITY_DIR   = Path("reports") / "sanity"

CONFIG_FILES = ("experiment.yaml", "facts.yaml", "prompts.yaml",
                "baseline_prompts.yaml", "forcing.yaml", "models.yaml")


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


def _prefix_rows(keys, cases, source, extra=None):
    """Registro de cada prefixo forçado: uma linha por palavra que o usou."""
    rows = []
    for case in cases or []:
        for w in case["words"]:
            rows.append({**keys, **(extra or {}), "source": source,
                         "shared_token_id": case["token_id"], "shared_token": case["token"],
                         "candidates": ",".join(case["candidates"]), "candidate": w["candidate"],
                         "word": w["word"], "next_token_id": w["next_id"],
                         "next_token": w["next_token"], "depth": w["depth"],
                         "path_ids": ",".join(map(str, w["path"])), "status": w["status"]})
    return rows


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
    tables = {"baseline": [], "layers": [], "token_sets": [], "prefix_forcing": []}
    for neutral in prompts_for(config.baseline_spec.prompts, None, lang):
        if neutral.format not in formats:
            continue
        res  = runner.measure_baseline(neutral, facts)
        keys = _keys(model_key, runner.lens.name, neutral, None)
        tables["baseline"]   += _prob_rows(keys, res, facts)
        tables["layers"]     += _layer_rows(keys, res, "baseline")
        for fk, ts in res["token_sets"].items():
            tables["token_sets"] += _token_rows({**keys, "fact": fk}, ts, "baseline")
        for fk, cases in res["prefix_cases"].items():
            tables["prefix_forcing"] += _prefix_rows({**keys, "fact": fk}, cases, "baseline")
    return tables


def indicator_skips(runner, fact, prompts) -> Dict:
    """
    Células de indicador (tipo) a pular neste tokenizer: menos de dois
    candidatos com token IDs válidos em alguma paráfrase.

    Returns:
        {tipo: [{paraphrase, valid_candidates}]} só dos tipos pulados
    """
    out = {}
    for t in INDICATOR_TYPES:
        ps = [p for p in prompts if p.type == t]
        if not ps:
            continue
        per = []
        for p in ps:
            text, sep, add_special = runner.prepare(p)
            ts = build_token_sets(runner.tokenizer, text, word_sets(fact, p),
                                  sep=sep, add_special_tokens=add_special)
            per.append((ts, resolve_shared_prefixes(runner.tokenizer, text, ts, sep=sep,
                                                    add_special_tokens=add_special)))
        if not indicator_cell_ok(per):
            out[t] = [{"paraphrase": p.paraphrase,
                       "valid_candidates": ",".join(valid_candidates(ts, cases))}
                      for p, (ts, cases) in zip(ps, per)]
    return out


def run_fact_unit(runner, tester, config, fact_key, lang, model_key, formats):
    facts  = config.fact_specs
    fact   = facts[fact_key]
    tables = {"probs": [], "forced": [], "layers": [], "token_sets": [], "skipped": [],
              "prefix_forcing": []}
    prompts = [p for p in prompts_for(config.prompt_specs, fact_key, lang) if p.format in formats]
    skip = indicator_skips(runner, fact, prompts)
    for t, rows in skip.items():
        for r in rows:
            tables["skipped"].append({"model": model_key, "lens": runner.lens.name, "fact": fact_key,
                                      "lang": lang, "prompt_type": t, **r,
                                      "reason": "menos de 2 candidatos com tokens válidos"})
    for prompt in prompts:
        if prompt.type in skip:
            continue
        res  = runner.measure(prompt, fact)
        keys = _keys(model_key, runner.lens.name, prompt, fact_key)
        tables["probs"]      += _prob_rows(keys, res, facts)
        tables["layers"]     += _layer_rows(keys, res, "prompt")
        tables["token_sets"] += _token_rows(keys, res["token_sets"], "prompt")
        tables["prefix_forcing"] += _prefix_rows(keys, res["prefix_cases"], "prompt")
        for ps in tester.run(prompt, fact, res):
            n_steps = len(ps["steps"])
            for st in ps["steps"]:
                tables["prefix_forcing"] += _prefix_rows(
                    keys, st["result"]["prefix_cases"], "forced",
                    extra={"pass_id": ps["pass_id"], "force_step": st["step"]})
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


def base_manifest(model_key, model_id, model, lens, formats, sanity, spec=None):
    return {
        "model_key":    model_key,
        "model_id":     model_id,
        "registry":     dataclasses.asdict(spec) if spec is not None else None,
        "precision_tag": precision_tag(spec) if spec is not None else None,
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
              lens=None, chat_formatter=None, spec=None, log=print):
    """
    Roda todas as unidades que faltam e consolida. Não checa
    confirmação (main() faz isso); testes chamam direto.

    Formatos: os da entrada do registro (`spec`); sem spec, QA só se
    houver chat_formatter.

    Returns:
        manifest (dict)
    """
    lens    = lens or LogitLens(model)
    formats = spec.formats if spec is not None else \
        (("completion", "qa") if chat_formatter is not None else ("completion",))
    if "qa" in formats and chat_formatter is None:
        raise ValueError(f"{model_key}: modelo instruct sem chat template (formato QA impossível)")
    if spec is not None and model.config.model_type != spec.architecture:
        raise ValueError(f"{model_key}: model_type '{model.config.model_type}' != registro "
                         f"'{spec.architecture}'")
    runner  = LensRunner(model, tokenizer, lens=lens, top_k=config.top_k_tokens,
                         chat_formatter=chat_formatter)
    tester  = ConfoundTester(runner, config.forcing_spec)
    store   = ChunkStore(out_dir)

    # Falha antes de gravar qualquer resultado se o lens não reproduzir
    # os logits do modelo na última camada
    sanity   = lens.sanity_check(tokenizer)
    manifest = read_manifest(out_dir) or base_manifest(
        model_key, model_id, model, lens, formats, sanity, spec)
    if spec is not None and manifest.get("revision") not in (None, spec.revision):
        log(f"  ⚠️  commit carregado {manifest['revision']} != registro {spec.revision}")
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
    pf = out_dir / "prefix_forcing.parquet"
    if pf.exists():
        import pandas as pd
        d = pd.read_parquet(pf)
        manifest["prefix_forcing"] = {k: int(v) for k, v in d["status"].value_counts().items()}
    sk = out_dir / "skipped.parquet"
    if sk.exists():
        import pandas as pd
        d = pd.read_parquet(sk)[["fact", "lang", "prompt_type"]].drop_duplicates()
        manifest["skipped_indicator_cells"] = d.to_dict("records")
    manifest["units_expected"] = len(units)
    manifest["rows"] = counts
    write_manifest(out_dir, manifest)
    return manifest


def load_tuned(model_key, model, spec, lens_root=None):
    """Tuned Lens treinado de lenses/{modelo}/ (recusa lente de outro modelo/commit)."""
    from config.loader import load_tuned_lens
    from logit_lens.tuned_lens import TunedLens
    root = Path(lens_root or load_tuned_lens()["lens_dir"]) / model_key
    if not (root / "translators.pt").exists():
        raise FileNotFoundError(f"sem Tuned Lens em {root}: rode train_tuned_lens.py --model {model_key}")
    return TunedLens.load(model, root, expected_revision=spec.revision if spec else None)


def instruct_chat(model_key, spec, tokenizer):
    """Formatador QA para modelos instruct (None para base)."""
    if spec.kind != "instruct":
        return None
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{model_key}: instruct sem chat template no tokenizer")
    from models.chat_template import apply_template
    return lambda tok, text, mk=model_key: apply_template(tok, mk, text)[0]


def write_sanity(report: dict, model_key: str, spec, report_dir=SANITY_DIR) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if precision_tag(spec) == precision_tag(load_models()[model_key]) else f"_{precision_tag(spec)}"
    path = report_dir / f"{model_key}{tag}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def main(argv=None):
    config = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Medição por camada (Logit Lens ou Tuned Lens).")
    add_model_args(parser, config.models)
    parser.add_argument("--lens", choices=list(LENS_CHOICES), default="logit")
    parser.add_argument("--sanity-only", action="store_true",
                        help="só o teste de sanidade (reports/sanity/{modelo}.json)")
    parser.add_argument("--results-dir", default=config.results_dir)
    parser.add_argument("--consolidate-only", action="store_true")
    args = parser.parse_args(argv)

    out_dir = lens_dir(args.results_dir, args.model, LENS_CHOICES[args.lens])
    if args.consolidate_only:
        print(ChunkStore(out_dir).consolidate())
        return 0

    apply_gpus(args.gpus)
    registry = registry_with_dtype(config.registry, args.model, args.dtype)
    spec = registry[args.model]
    require_valid()
    if not args.sanity_only:
        # Nada roda com valores não confirmados (a sanidade não usa resultados)
        require_confirmed(config.fact_specs, config.prompt_specs,
                          config.baseline_spec, config.forcing_spec,
                          prompts_status=load_prompts_meta()["status"])
        check_resume(read_manifest(out_dir), spec, out_dir)

    from models.model_loader import load_model
    model, tokenizer = load_model(args.model, registry)
    chat = instruct_chat(args.model, spec, tokenizer)

    if args.sanity_only:
        from logit_lens.sanity import sanity_report
        rep = sanity_report(args.model, model, tokenizer, spec, config, chat)
        rep["precision_tag"] = precision_tag(spec)
        path = write_sanity(rep, args.model, spec)
        print(f"  {'✅' if rep['passed'] else '❌'} {args.model}: max |diff| = {rep['max_abs_diff']}"
              + "".join(f"\n      {f}" for f in rep["failures"]) + f"\n  → {path}")
        return 0 if rep["passed"] else 1

    lens = load_tuned(args.model, model, spec) if args.lens == "tuned" else None
    m = run_model(args.model, model, tokenizer, config, out_dir, model_id=spec.id,
                  lens=lens, chat_formatter=chat, spec=spec)
    print(f"\n✅ {args.model}: {m['units_done']}/{m['units_expected']} unidades, "
          f"{m['wall_time_s'] / 60:.1f} min, {m['gpu_hours']:.2f} GPU-h  →  {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
