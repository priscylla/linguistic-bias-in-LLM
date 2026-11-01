"""
scripts/check_disagreements.py
Lista as discordancias entre a extracao automatica e a rotulagem humana
e grava tudo em reports/validation/disagreements.txt.

Uso (de qualquer pasta; os caminhos sao relativos a raiz do repositorio):
    python scripts/check_disagreements.py
    python scripts/check_disagreements.py --input outra.csv --output saida.txt

"linha N" e o numero do registro no CSV (cabecalho = 1), contado pelo
pandas: as respostas tem quebras de linha, entao nao e a linha fisica do
arquivo. reports/validation_adjudication.csv usa a mesma numeracao.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports" / "validation" / "sample.csv"
OUT = ROOT / "reports" / "validation" / "disagreements.txt"
TAG = "credited≠first"


def ids(s):
    return {x.strip() for x in str(s).split(",") if x.strip() and x.strip() != "none"}


def one_line(s, n=400):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " [...]"


def record_line(i) -> int:
    """Numero do registro no CSV (cabecalho = 1) a partir do indice do pandas."""
    return i + 2


def disagreements(df: pd.DataFrame):
    """(primarios diferentes, [(i, linha, so_auto, so_humano)] citados diferentes, credited≠first, outras notas)."""
    df = df[df["human_primary"].str.strip() != ""]   # so as linhas rotuladas
    prim = df[df["auto_primary"].str.strip() != df["human_primary"].str.strip()]
    cited = []
    for i, r in df.iterrows():
        only_auto = ids(r.auto_cited) - ids(r.human_cited)
        only_human = ids(r.human_cited) - ids(r.auto_cited)
        if only_auto or only_human:
            cited.append((i, r, only_auto, only_human))
    cred = df[df["notes"].str.contains(TAG, regex=False)]
    other = df[(df["notes"].str.strip() != "") & ~df["notes"].str.contains(TAG, regex=False)]
    return df, prim, cited, cred, other


def report(df: pd.DataFrame) -> tuple:
    """(texto do relatorio, contagens)."""
    df, prim, cited, cred, other = disagreements(df)
    lines = [f"== 1o citado diferente: {len(prim)} =="]
    for i, r in prim.iterrows():
        lines.append("")
        lines.append(f"linha {record_line(i)} | {r.model} | {r.fact} | {r.lang} | auto={r.auto_primary} | humano={r.human_primary}")
        lines.append(f"  auto_cited={r.auto_cited} | human_cited={r.human_cited}")
        lines.append(f"  {one_line(r.answer)}")

    lines.append("")
    lines.append(f"== citados diferentes: {len(cited)} respostas ==")
    for i, r, a, h in cited:
        lines.append("")
        lines.append(f"linha {record_line(i)} | {r.model} | {r.fact} | {r.lang} | so auto: {sorted(a)} | so humano: {sorted(h)}")
        lines.append(f"  {one_line(r.answer)}")

    lines.append("")
    lines.append(f"== {TAG}: {len(cred)} de {len(df)} ==")
    for i, r in cred.iterrows():
        lines.append(f"linha {record_line(i)} | {r.model} | {r.fact} | {r.lang} | 1o citado={r.human_primary} | notas={one_line(r.notes, 120)}")

    lines.append("")
    lines.append(f"== outras notas: {len(other)} ==")
    for i, r in other.iterrows():
        lines.append(f"linha {record_line(i)} | {r.fact} | {r.lang} | {one_line(r.notes, 200)}")
    counts = {"labelled": len(df), "primary": len(prim), "cited": len(cited), "credited_not_first": len(cred),
              "primary_lines": [record_line(i) for i in prim.index],
              "cited_lines": [record_line(i) for i, *_ in cited]}
    return "\n".join(lines) + "\n", counts


def main(argv=None):
    p = argparse.ArgumentParser(description="Discordancias entre extracao automatica e rotulos humanos.")
    p.add_argument("--input", default=str(SRC))
    p.add_argument("--output", default=str(OUT))
    args = p.parse_args(argv)
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    text, c = report(df)
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"OK {args.output} ({c['primary']} primarios diferentes, {c['cited']} com citados diferentes, "
          f"{c['credited_not_first']} {TAG})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
