"""
train_tuned_lens.py
Treina o Tuned Lens de um modelo instruct (Fase 4). Roda na GPU da
autora; aqui só é testado por alguns passos com o modelo minúsculo.

    python train_tuned_lens.py --model llama
    python train_tuned_lens.py --model llama --gpus "3"
    python train_tuned_lens.py --model llama --max-steps 50   # sessão curta; retoma depois
    python train_tuned_lens.py --model llama --eval-only
    python train_tuned_lens.py --model llama --force          # retreina uma lente pronta
    python train_tuned_lens.py --model llama --tokens-per-step 262144   # 2^18 (lente reprovada)

Decide pelo hash da receita (logit_lens/lens_training.py recipe_sha256),
gravado em lenses/{modelo}/RECIPE:
    lente de outra receita (ou sem hash: toda lente de 2026-10-29, treinada
    sem o corte do gradiente)
        → lenses/{modelo}, reports/tuned_lens/{modelo}_* e as medições do
          Tuned Lens com ela vão para lenses_stale/<data>/ e
          reports/tuned_lens_stale/<data>/ (nunca apagadas), e retreina
    mesma receita, aprovada (status + relatório)  → pula (código 0)
    mesma receita, reprovada                      → pula (código 4); para
          tentar outra receita: --tokens-per-step 262144
    mesma receita, treino incompleto              → retoma do checkpoint
Também pula se houver lente pré-treinada declarada em config/tuned_lens.yaml.

Receita, corpus, filtro e orçamento em config/tuned_lens.yaml. Ao
terminar os num_steps, avalia no held-out e aplica o teste de aceitação:
    lenses/{modelo}/translators.pt, config.json   (meta: receita, hash,
                                                   max_grad_norm, aceitação)
    reports/tuned_lens/{modelo}_eval.csv           aprovada: KL held-out por
                                                   idioma × camada (Tuned e Logit)
    reports/tuned_lens/{modelo}_eval_rejected.csv  reprovada (a remedição não roda:
                                                   não há {modelo}_eval.csv)
    reports/tuned_lens/{modelo}_status.json        accepted | rejected, receita,
                                                   resultado do teste
    reports/tuned_lens/{modelo}_train.json         perdas, norma do gradiente, tempo
Código de saída: 0 aprovada (ou pulada), 2 treino incompleto, 3 lock,
4 reprovada.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from config.experiment_config import ExperimentConfig
from config.loader import load_tuned_lens, require_confirmed, require_valid
from logit_lens.lens_training import (RECIPE_VERSION, acceptance, evaluate, fact_terms, recipe,
                                      recipe_sha256, train, write_json)
from logit_lens.store import lens_dir as results_lens_dir
from logit_lens.tuned_lens import TunedLens, update_meta
from models.runtime import apply_gpus

REPORT_DIR = Path("reports") / "tuned_lens"
REPORT_SUFFIXES = ("_eval.csv", "_eval_rejected.csv", "_train.json", "_status.json")
EXIT_REJECTED = 4


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

    sha = recipe_sha256(tl_cfg)
    if eval_only:
        lens = TunedLens.load(model, out, expected_revision=spec.revision, allow_rejected=True)
        summary = {"step": lens.meta.get("steps"), "done": True}
    else:
        lens = TunedLens(model, translator_dtype=tl_cfg["training"]["translator_dtype"])
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "RECIPE", {"recipe_sha256": sha, **recipe(tl_cfg)})
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
            "corpus_stats": summary["corpus_stats"], "final_loss": summary["losses"][-1],
            "recipe_sha256": sha, "recipe_version": RECIPE_VERSION,
            "max_grad_norm": t["max_grad_norm"], "tokens_per_step": t["tokens_per_step"]})

    rows = evaluate(lens, model, tokenizer, sources, tl_cfg, config.fact_specs)
    df = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    acc = acceptance(rows, model.config.num_hidden_layers, config.decision_zone,
                     float(tl_cfg["acceptance"]["tolerance_nats"]))
    # Aprovada → {key}_eval.csv (a fila só remede se ele existir); reprovada → _eval_rejected.csv
    keep, drop = ((f"{key}_eval.csv", f"{key}_eval_rejected.csv") if acc["passed"]
                  else (f"{key}_eval_rejected.csv", f"{key}_eval.csv"))
    (report_dir / drop).unlink(missing_ok=True)          # relatório anterior desta mesma lente
    df.to_csv(report_dir / keep, index=False)
    update_meta(out, {"acceptance": acc})
    meta = json.loads((out / "config.json").read_text(encoding="utf-8")).get("meta", {})
    write_json(report_dir / f"{key}_status.json", {
        "model": key, "status": "accepted" if acc["passed"] else "rejected",
        "recipe_sha256": meta.get("recipe_sha256"), "recipe_version": meta.get("recipe_version"),
        "max_grad_norm": meta.get("max_grad_norm"), "tokens_per_step": meta.get("tokens_per_step"),
        "eval_report": keep, "acceptance": acc})
    summary["eval"] = {"mean_kl_tuned": float(df["kl_tuned"].mean()),
                       "mean_kl_logit": float(df["kl_logit"].mean())}
    summary["acceptance"] = acc
    log(f"  held-out KL médio: tuned = {summary['eval']['mean_kl_tuned']:.4f}, "
        f"logit = {summary['eval']['mean_kl_logit']:.4f}")
    if acc["passed"]:
        zone = ("sem camada traduzida na zona" if acc["zone_kl_tuned"] is None else
                f"zona de decisão: tuned {acc['zone_kl_tuned']:.3f}, logit {acc['zone_kl_logit']:.3f}")
        log(f"  ✅ teste de aceitação: aprovada (última camada traduzida {acc['last_layer']}: tuned "
            f"{acc['last_kl_tuned']:.3f} ≤ logit {acc['last_kl_logit']:.3f} + {acc['tolerance_nats']}; {zone})")
    else:
        log(rejection_message(key, acc, report_dir / keep))
    return summary


def rejection_message(key, acc, report) -> str:
    lines = [f"  ❌ Tuned Lens de {key} REPROVADO no teste de aceitação "
             f"(tolerância {acc['tolerance_nats']} nats):"]
    lines += [f"     - {r}" for r in acc["reasons"]]
    if acc["failing_layers"]:
        lines.append("     camadas da zona de decisão piores que o Logit Lens (KL held-out, média dos idiomas):")
        lines += [f"       camada {f['layer']:>3}: tuned {f['kl_tuned']:.3f}   logit {f['kl_logit']:.3f}"
                  for f in acc["failing_layers"]]
    lines += [f"     relatório: {report}; a remedição com esta lente não roda.",
              f"     próximo passo (decisão da autora): python train_tuned_lens.py --model {key} "
              f"--tokens-per-step 262144"]
    return "\n".join(lines)


def read_status(key, report_dir=REPORT_DIR) -> dict:
    p = Path(report_dir) / f"{key}_status.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def recipe_state(lens_dir, sha) -> tuple:
    """("none" | "same" | "different", hash gravado ou None)."""
    root = Path(lens_dir)
    if not root.is_dir() or not [f for f in root.iterdir() if f.name != "TRAINING.lock"]:
        return "none", None
    old = None
    try:
        old = json.loads((root / "RECIPE").read_text(encoding="utf-8")).get("recipe_sha256")
    except (OSError, ValueError):
        try:
            old = json.loads((root / "config.json").read_text(encoding="utf-8"))["meta"].get("recipe_sha256")
        except (OSError, ValueError, KeyError):
            old = None
    return ("same" if old == sha else "different"), old


def archive_stale(key, lens_root, report_dir=REPORT_DIR, results_dir=None, reason="", stamp=None) -> list:
    """
    Move (nunca apaga) a lente, os relatórios dela e as medições do Tuned
    Lens feitas com ela para lenses_stale/<stamp>/ e
    reports/tuned_lens_stale/<stamp>/. Devolve os destinos.
    """
    stamp = stamp or time.strftime("%Y-%m-%d_%H%M%S")
    lens_root, report_dir = Path(lens_root), Path(report_dir)
    stale = lens_root.parent / f"{lens_root.name}_stale" / stamp
    stale_rep = report_dir.parent / f"{report_dir.name}_stale" / stamp
    moved = []

    def move(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append(str(dst))

    if (lens_root / key).exists():
        move(lens_root / key, stale / key)
    for suf in REPORT_SUFFIXES:
        if (report_dir / f"{key}{suf}").exists():
            move(report_dir / f"{key}{suf}", stale_rep / f"{key}{suf}")
    if results_dir is not None:
        res = results_lens_dir(str(results_dir), key, "tuned_lens")
        if res.exists():
            move(res, stale / f"{key}_results_tuned_lens")
    if moved:
        stale.mkdir(parents=True, exist_ok=True)
        with open(stale / "README.txt", "a", encoding="utf-8") as fh:
            fh.write(f"{key}: {reason}\n" + "".join(f"  -> {m}\n" for m in moved))
    return moved


def skip_reason(key, tl_cfg, lens_root=None, report_dir=REPORT_DIR, force=False, results_dir=None,
                log=print):
    """
    Decide antes de treinar. Devolve None (treina, ou retoma) ou
    (motivo, código de saída) para pular. Uma lente de outra receita (ou
    sem hash) é arquivada aqui, com mensagem, e então retreinada.
    """
    if (tl_cfg.get("pretrained") or {}).get(key):
        return f"lente pré-treinada declarada: {tl_cfg['pretrained'][key]}", 0
    lens_root = Path(lens_root or tl_cfg["lens_dir"])
    root = lens_root / key
    if (root / "TRAINING.lock").exists():
        return None                                   # main() acusa o lock; nada é movido
    sha = recipe_sha256(tl_cfg)
    state, old = recipe_state(root, sha)
    if state == "different" or (force and state != "none"):
        why = ("--force" if state == "same" else
               f"receita diferente ({old[:12]} ≠ {sha[:12]})" if old else
               "sem hash de receita (treinada antes de 2026-10-30, sem o corte do gradiente)")
        moved = archive_stale(key, lens_root, report_dir, results_dir, reason=why)
        log(f"  📦 {key}: lente existente arquivada: {why}.\n"
            + "".join(f"       → {m}\n" for m in moved)
            + f"     Nada foi apagado. Retreinando com a receita {sha[:12]} (versão {RECIPE_VERSION}).")
        return None
    if state == "same":
        st = read_status(key, report_dir)
        if (st.get("status") == "accepted" and (root / "translators.pt").exists()
                and (Path(report_dir) / f"{key}_eval.csv").exists()):
            return f"lente já treinada e aprovada com esta receita ({sha[:12]}) em {root}", 0
        if st.get("status") == "rejected":
            return (f"lente desta receita ({sha[:12]}) foi REPROVADA no teste de aceitação "
                    f"({'; '.join(st.get('acceptance', {}).get('reasons', []))}). Para tentar 2^18 tokens "
                    f"por passo: python train_tuned_lens.py --model {key} --tokens-per-step 262144"), EXIT_REJECTED
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
    parser.add_argument("--force", action="store_true",
                        help="retreina mesmo com a mesma receita (a lente atual é arquivada)")
    parser.add_argument("--tokens-per-step", type=int, default=None,
                        help="sobrepõe training.tokens_per_step (ex.: 262144 = 2^18 para uma lente "
                             "reprovada); muda a receita")
    args = parser.parse_args(argv)
    if args.tokens_per_step:
        seq = tl_cfg["training"]["seq_len"]
        if args.tokens_per_step % seq:
            parser.error(f"--tokens-per-step precisa ser múltiplo de seq_len ({seq})")
        tl_cfg["training"]["tokens_per_step"] = args.tokens_per_step

    if not args.eval_only:
        why = skip_reason(args.model, tl_cfg, force=args.force, results_dir=config.results_dir)
        if why:
            msg, rc = why if isinstance(why, tuple) else (why, 0)
            print(f"  {'⏭ ' if rc == 0 else '❌'} {args.model}: {msg}")
            return rc
    apply_gpus(args.gpus)
    require_valid()
    # O filtro do corpus usa os nomes de facts.yaml: só com fatos
    # confirmados. Não usa prompts, então a revisão dos prompts não
    # bloqueia o treino.
    require_confirmed(config.fact_specs, [])

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
    if not s.get("done"):
        return 2
    return 0 if (s.get("acceptance") or {}).get("passed", True) else EXIT_REJECTED


if __name__ == "__main__":
    # Sai direto depois de gravar tudo, como run_lens.py e run_behavioral.py:
    # a finalização do interpretador abortou dois treinos com código 134
    # depois do fim (2026-10-29). Correção do coautor no servidor.
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc or 0)
