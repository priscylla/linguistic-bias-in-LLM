"""
train_tuned_lens.py
Treina o Tuned Lens de um modelo instruct (Fase 4). Roda na GPU da
autora; aqui só é testado por alguns passos com o modelo minúsculo.

    python train_tuned_lens.py --model llama
    python train_tuned_lens.py --model llama --gpus "3"
    python train_tuned_lens.py --model llama --max-steps 50   # sessão curta; retoma depois
    python train_tuned_lens.py --model llama --eval-only
    python train_tuned_lens.py --model llama --force          # retreina uma lente pronta

Pula o modelo se houver lente pré-treinada declarada em
config/tuned_lens.yaml (pretrained) ou se a lente já estiver pronta
(translators.pt + relatório de avaliação).

Receita, corpus, filtro e orçamento em config/tuned_lens.yaml. Retoma
de lenses/{modelo}/checkpoint.pt. Ao terminar os num_steps:
    lenses/{modelo}/translators.pt, config.json   (lido por run_lens.py --lens tuned)
    reports/tuned_lens/{modelo}_eval.csv           KL held-out por idioma × camada,
                                                   Tuned Lens e Logit Lens
    reports/tuned_lens/{modelo}_train.json         perdas, corpus, tempo
"""

import argparse
import hashlib
import os
import socket
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from config.experiment_config import ExperimentConfig
from config.loader import load_tuned_lens, require_confirmed, require_valid
from logit_lens.lens_training import evaluate, fact_terms, train, write_json
from logit_lens.tuned_lens import TunedLens
from models.runtime import apply_gpus

REPORT_DIR = Path("reports") / "tuned_lens"


def hf_sources(tl_cfg, langs):
    """Uma fonte por idioma: streaming do dataset no commit fixo."""
    from datasets import load_dataset

    def make(c):
        return lambda: iter(load_dataset(c["dataset"], c["config"], split=c["split"],
                                         revision=c["revision"], streaming=True))
    return {l: make(tl_cfg["corpus"][l]) for l in langs}


def filter_hash(facts, langs) -> str:
    terms = "\n".join(f"{l}:{t}" for l in langs for t in fact_terms(facts, l))
    return hashlib.sha256(terms.encode("utf-8")).hexdigest()


def run(key, model, tokenizer, config, tl_cfg, sources, lens_root, report_dir,
        max_steps=None, eval_only=False, log=print) -> dict:
    """Treina (ou só avalia) e grava lente e relatórios. Testável sem rede."""
    spec    = config.model(key)
    out     = Path(lens_root) / key
    langs   = config.languages
    report_dir = Path(report_dir)

    if eval_only:
        lens = TunedLens.load(model, out, expected_revision=spec.revision)
        summary = {"step": lens.meta.get("steps"), "done": True}
    else:
        lens = TunedLens(model, translator_dtype=tl_cfg["training"]["translator_dtype"])
        t0 = time.time()
        summary = train(lens, model, tokenizer, sources, tl_cfg, config.fact_specs, out,
                        max_steps=max_steps, log=log)
        summary["wall_time_s"] = round(time.time() - t0, 1)
        write_json(report_dir / f"{key}_train.json", summary)
        if not summary["done"]:
            log(f"  parado no passo {summary['step']}/{summary['num_steps']}; rode de novo para continuar")
            return summary
        t = tl_cfg["training"]
        lens.save(out, meta={
            "model_key": key, "model_id": spec.id, "registry_revision": spec.revision,
            "steps": summary["step"], "tokens": summary["step"] * t["tokens_per_step"],
            "recipe": t, "corpus": tl_cfg["corpus"], "held_out_mod": tl_cfg["held_out_mod"],
            "filter_fact_terms": tl_cfg["filter_fact_terms"],
            "filter_terms_sha256": filter_hash(config.fact_specs, langs),
            "corpus_stats": summary["corpus_stats"], "final_loss": summary["losses"][-1]})

    rows = evaluate(lens, model, tokenizer, sources, tl_cfg, config.fact_specs)
    df = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / f"{key}_eval.csv", index=False)
    summary["eval"] = {"mean_kl_tuned": float(df["kl_tuned"].mean()),
                       "mean_kl_logit": float(df["kl_logit"].mean())}
    log(f"  held-out KL médio: tuned = {summary['eval']['mean_kl_tuned']:.4f}, "
        f"logit = {summary['eval']['mean_kl_logit']:.4f}")
    return summary


def skip_reason(key, tl_cfg, lens_root=None, report_dir=REPORT_DIR, force=False):
    """Motivo para não treinar (ou None)."""
    if (tl_cfg.get("pretrained") or {}).get(key):
        return f"lente pré-treinada declarada: {tl_cfg['pretrained'][key]}"
    root = Path(lens_root or tl_cfg["lens_dir"]) / key
    if not force and (root / "translators.pt").exists() and (Path(report_dir) / f"{key}_eval.csv").exists():
        return f"lente já treinada em {root} (use --force para retreinar)"
    return None


class TrainingLock:
    """
    lenses/{modelo}/TRAINING.lock: impede dois processos treinando o
    mesmo modelo ao mesmo tempo (ex.: duas filas). Um lock de um processo
    que morreu tem de ser apagado à mão (a mensagem diz qual arquivo).
    """

    def __init__(self, lens_dir):
        self.path = Path(lens_dir) / "TRAINING.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"{self.path} existe: outro processo está treinando este modelo "
                               f"({self.path.read_text().strip()}). Se não estiver, apague o arquivo.")
        with os.fdopen(fd, "w") as f:
            f.write(f"pid {os.getpid()} em {socket.gethostname()}")
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


def main(argv=None):
    config = ExperimentConfig()
    tl_cfg = load_tuned_lens()
    parser = argparse.ArgumentParser(description="Treina o Tuned Lens de um modelo instruct.")
    parser.add_argument("--model", required=True, choices=tl_cfg["models"])
    parser.add_argument("--gpus", default=None, help='GPUs do job, ex.: "3" ou "2 3"')
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if not args.eval_only:
        why = skip_reason(args.model, tl_cfg, force=args.force)
        if why:
            print(f"  ⏭  {args.model}: {why}")
            return 0
    apply_gpus(args.gpus)
    require_valid()
    # O filtro do corpus usa os nomes de facts.yaml: só com fatos
    # confirmados. Não usa prompts, então a revisão nativa (prompts e
    # formas russas pendentes) não bloqueia o treino; as formas pendentes
    # entram no filtro como termos a mais (só removem mais documentos).
    require_confirmed(config.fact_specs, [], ru_names=False)

    from models.model_loader import load_model
    try:
        lock = TrainingLock(Path(tl_cfg["lens_dir"]) / args.model).__enter__()
    except RuntimeError as e:
        print(f"  ❌ {e}")
        return 3
    try:
        model, tok = load_model(args.model, config.registry)
        sources = hf_sources(tl_cfg, config.languages)
        s = run(args.model, model, tok, config, tl_cfg, sources, tl_cfg["lens_dir"], REPORT_DIR,
                max_steps=args.max_steps, eval_only=args.eval_only)
    finally:
        lock.__exit__(None, None, None)
    if torch.cuda.is_available():
        print(f"  pico de VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    return 0 if s.get("done") else 2


if __name__ == "__main__":
    sys.exit(main())
