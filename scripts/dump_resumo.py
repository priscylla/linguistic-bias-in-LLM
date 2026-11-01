"""
scripts/dump_resumo.py
Text dump of the main metric tables, for a quick read without pandas.

    python scripts/dump_resumo.py              # → resumo_resultados.txt (repo root)
    python scripts/dump_resumo.py --out x.txt

Reads only results/metrics/*.parquet (written by analyze.py) and writes, for
each table in TABLES and in this order, a blank line, a header
"===== <name> (<n> linhas) =====" and DataFrame.to_string(). The output is a
local file (results are not versioned).

Reconstructed on 2026-10-31 from the format of the resumo_resultados.txt of
2026-10-30 (the original script was not kept); run on the metrics of that day
it reproduces that file's unchanged tables byte for byte.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ["stats_stake_tests", "stats_mixed_coefs", "stats_mixed_fits", "lens_agreement_summary",
          "empirical_dominant", "behavioral_report"]


def dump(metrics_dir: Path) -> str:
    out = []
    for name in TABLES:
        df = pd.read_parquet(Path(metrics_dir) / f"{name}.parquet")
        out.append(f"\n===== {name} ({len(df)} linhas) =====\n{df.to_string()}\n")
    return "".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description="Dump das tabelas principais de results/metrics.")
    p.add_argument("--metrics-dir", default=str(ROOT / "results" / "metrics"))
    p.add_argument("--out", default=str(ROOT / "resumo_resultados.txt"))
    args = p.parse_args(argv)
    missing = [t for t in TABLES if not (Path(args.metrics_dir) / f"{t}.parquet").exists()]
    if missing:
        print(f"❌ faltam em {args.metrics_dir}: {', '.join(missing)} (rode analyze.py)")
        return 1
    text = dump(args.metrics_dir)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"OK {args.out}: {text.count(chr(10))} linhas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
