"""
scripts/export_validation_sample.py
Exporta a amostra para validação manual da extração (Fase 5).

    python scripts/export_validation_sample.py
    python scripts/export_validation_sample.py --force     # sobrescreve (perde rótulos!)

Se a amostra já existe, não faz nada e sai com 0 (rótulos preservados).

Amostra estratificada (config/behavioral.yaml validation: per_cell
respostas por modelo × idioma, semente fixa) das respostas em
results/behavioral/*/. Grava reports/validation/sample.csv com colunas
vazias human_primary, human_cited, notes (ver behavior/validation.py).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyze import compute_behavioral, load_answers  # noqa: E402
from behavior.validation import stratified_sample  # noqa: E402
from config.experiment_config import ExperimentConfig  # noqa: E402
from config.loader import load_behavioral  # noqa: E402

OUT = Path("reports") / "validation" / "sample.csv"


def main(argv=None):
    config = ExperimentConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=config.results_dir)
    p.add_argument("--output", default=str(OUT))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.output)
    if out.exists() and not args.force:
        # Não é erro: a amostra já foi exportada (e pode ter rótulos). Sai com 0
        # para não interromper run_all_server.sh.
        print(f"⏭  {out} já existe (pode ter rótulos): mantida. Use --force para sobrescrever.")
        return 0
    answers = load_answers(args.results_dir)
    if answers is None:
        print("❌ sem respostas em results/behavioral/ (rode run_behavioral.py)")
        return 1
    b   = load_behavioral()
    res = compute_behavioral(answers, config, b)
    cands = {k: "; ".join(f"{c.id}={c.label}" for c in f.candidates)
             for k, f in config.fact_specs.items()}
    s = stratified_sample(answers, res["behavioral_attribution"], b.validation["per_cell"],
                          b.validation["seed"], cands)
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out, index=False)
    print(f"OK {out}: {len(s)} respostas ({s.groupby(['model', 'lang']).ngroups} células)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
