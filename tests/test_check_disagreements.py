"""
tests/test_check_disagreements.py
scripts/check_disagreements.py: discordâncias de primário e de citados,
numeração por registro do CSV (cabeçalho = 1, respostas com quebra de
linha) e caminhos independentes da pasta de onde é chamado.
"""

import pandas as pd

from scripts import check_disagreements as cd

COLS = ["uid", "model", "fact", "lang", "answer", "auto_primary", "auto_cited",
        "human_primary", "human_cited", "notes"]


def _sample():
    rows = [
        ["a", "llama", "aviao", "pt", "Santos-Dumont\nvoou.", "santos_dumont", "santos_dumont",
         "santos_dumont", "santos_dumont", ""],                                   # concorda
        ["b", "llama", "aviao", "fr", "Wright, puis\nSantos-Dumont.", "wright", "wright,santos_dumont",
         "none", "", ""],                                                          # primário e citados
        ["c", "qwen", "telefone", "it", "Bell e Meucci.", "bell", "bell,meucci",
         "bell", "bell", "credited≠first: Meucci"],                                # só citados
        ["d", "qwen", "telefone", "de", "Bell.", "bell", "bell", "", "", ""],     # não rotulada
    ]
    return pd.DataFrame(rows, columns=COLS)


def test_counts_and_record_numbers():
    text, c = cd.report(_sample())
    assert c["labelled"] == 3 and c["primary"] == 1 and c["cited"] == 2 and c["credited_not_first"] == 1
    # registro 2 = primeira linha de dados, mesmo com quebras de linha nas respostas
    assert c["primary_lines"] == [3] and c["cited_lines"] == [3, 4]
    assert "linha 3 | llama | aviao | fr | auto=wright | humano=none" in text
    assert "so auto: ['meucci'] | so humano: []" in text


def test_runs_from_any_folder(tmp_path, monkeypatch):
    src, out = tmp_path / "s.csv", tmp_path / "out" / "d.txt"
    out.parent.mkdir()
    _sample().to_csv(src, index=False)
    monkeypatch.chdir(tmp_path / "out")
    assert cd.main(["--input", str(src), "--output", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("== 1o citado diferente: 1 ==")
    assert cd.SRC.is_absolute() and cd.SRC == cd.ROOT / "reports" / "validation" / "sample.csv"
