"""
logit_lens/store.py
Armazenamento dos resultados brutos da Fase 2 com checkpoint.

    results/lens/{lens}/{modelo}/          lens = logit_lens | tuned_lens
        chunks/{unidade}/{tabela}.parquet   uma unidade = fato × idioma
        chunks/{unidade}/DONE               ou linha de base × idioma
        {tabela}.parquet                    tabelas consolidadas
        manifest.json

Uma unidade só conta como feita depois que todas as suas tabelas foram
gravadas (diretório temporário renomeado no fim). Rodar de novo pula as
unidades feitas. Nada aqui reescreve uma unidade já gravada.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

TABLES = ("probs", "baseline", "forced", "layers", "token_sets", "skipped")


LENSES = ("logit_lens", "tuned_lens")


def lens_dir(results_dir: str, model_key: str, lens: str = "logit_lens") -> Path:
    if lens not in LENSES:
        raise ValueError(f"lens '{lens}' desconhecido: {LENSES}")
    return Path(results_dir) / "lens" / lens / model_key


def behavioral_dir(results_dir: str, model_key: str) -> Path:
    return Path(results_dir) / "behavioral" / model_key


class ChunkStore:

    def __init__(self, out_dir, tables=TABLES):
        self.out_dir = Path(out_dir)
        self.tables  = tuple(tables)
        self.chunks  = self.out_dir / "chunks"
        self.chunks.mkdir(parents=True, exist_ok=True)

    def done(self, unit: str) -> bool:
        return (self.chunks / unit / "DONE").exists()

    def done_units(self) -> List[str]:
        return sorted(p.parent.name for p in self.chunks.glob("*/DONE"))

    def write(self, unit: str, tables: Dict[str, List[Dict]]):
        """Grava todas as tabelas da unidade de uma vez (atômico por unidade)."""
        final = self.chunks / unit
        if final.exists():
            raise FileExistsError(f"unidade já gravada: {final}")
        tmp = self.chunks / f".tmp_{unit}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for name, rows in tables.items():
            if name not in self.tables:
                raise ValueError(f"tabela desconhecida: {name}")
            if rows:
                pd.DataFrame(rows).to_parquet(tmp / f"{name}.parquet", index=False)
        (tmp / "DONE").write_text("ok", encoding="utf-8")
        os.replace(tmp, final)

    def consolidate(self, units: Iterable[str] = None) -> Dict[str, int]:
        """Concatena os chunks feitos em {tabela}.parquet. Devolve linhas por tabela."""
        units = list(units) if units is not None else self.done_units()
        counts = {}
        for name in self.tables:
            parts = [pd.read_parquet(p) for u in units
                     if (p := self.chunks / u / f"{name}.parquet").exists()]
            if not parts:
                continue
            df = pd.concat(parts, ignore_index=True)
            df.to_parquet(self.out_dir / f"{name}.parquet", index=False)
            counts[name] = len(df)
        return counts


def read_manifest(out_dir) -> Dict:
    p = Path(out_dir) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_manifest(out_dir, manifest: Dict):
    p   = Path(out_dir) / "manifest.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def load_tables(results_dir: str, tables=TABLES) -> Dict[str, pd.DataFrame]:
    """Tabelas consolidadas de todos os lenses e modelos em results/lens/*/*/ (concatenadas)."""
    out = {}
    root = Path(results_dir) / "lens"
    for name in tables:
        parts = [pd.read_parquet(p) for p in sorted(root.glob(f"*/*/{name}.parquet"))]
        if parts:
            out[name] = pd.concat(parts, ignore_index=True)
    return out
