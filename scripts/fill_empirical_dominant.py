"""
scripts/fill_empirical_dominant.py
Grava em config/facts.yaml o campo empirical_dominant de cada fato, a
partir de results/metrics/empirical_dominant.csv (escopo "pooled",
analyze.py). Só esse campo muda; o resto do arquivo (comentários,
ordem) fica igual. Avisa quando o empírico difere do `dominant`
configurado. Rodar só depois do estudo comportamental completo.

    python scripts/fill_empirical_dominant.py
    python scripts/fill_empirical_dominant.py --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FACTS = ROOT / "config" / "facts.yaml"


def fill(text: str, values: dict) -> str:
    """Troca a linha empirical_dominant dentro do bloco de cada fato."""
    lines, cur, out = text.splitlines(keepends=True), None, []
    for line in lines:
        m = re.match(r"^  ([a-z_]+):\s*$", line)
        if m:
            cur = m.group(1)
        m = re.match(r"^(\s{4}empirical_dominant:\s*)(\S+)(.*)$", line.rstrip("\n"))
        if m and cur in values:
            line = f"{m.group(1)}{values[cur]}{m.group(3)}\n"
        out.append(line)
    return "".join(out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/metrics/empirical_dominant.csv")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    df = pd.read_csv(args.input)
    df = df[df["scope"] == "pooled"]
    values = dict(zip(df["fact"], df["empirical_dominant"]))
    for r in df.itertuples():
        mark = "" if r.agrees else "   ⚠️  difere do dominant configurado"
        print(f"  {r.fact:18s} {r.empirical_dominant:15s} (configurado {r.configured_dominant}, "
              f"{r.share:.0%} de {r.n_named}){mark}")
    new = fill(FACTS.read_text(encoding="utf-8"), values)
    if args.dry_run:
        return 0
    FACTS.write_text(new, encoding="utf-8")
    print(f"OK {FACTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
