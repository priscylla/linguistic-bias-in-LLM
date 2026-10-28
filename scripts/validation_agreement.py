"""
scripts/validation_agreement.py
Concordância entre a extração automática e os rótulos humanos (Fase 5).

    python scripts/validation_agreement.py
    python scripts/validation_agreement.py --input reports/validation/sample.csv

Lê o CSV rotulado (linhas sem human_primary são ignoradas) e grava
reports/validation/agreement.json: concordância e kappa de Cohen do
candidato primário (com IC 95%), kappa / precisão / revocação dos
candidatos citados e a concordância do primário por idioma.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from behavior.validation import agreement  # noqa: E402
from config.experiment_config import ExperimentConfig  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="reports/validation/sample.csv")
    p.add_argument("--output", default="reports/validation/agreement.json")
    args = p.parse_args(argv)

    config = ExperimentConfig()
    df  = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    res = agreement(df, {k: [c.id for c in f.candidates] for k, f in config.fact_specs.items()})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    if "primary" in res:
        pr, ci = res["primary"], res["primary"]["ci95"]
        print(f"{res['n_labelled']}/{res['n_rows']} rotuladas | primário: "
              f"{pr['agreement']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], kappa {pr['kappa']:.3f} | "
              f"citados: kappa {res['cited']['kappa']:.3f}")
    else:
        print(f"nenhuma linha rotulada em {args.input}")
    print(f"OK {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
