"""
tests/test_figures.py
Fase 7: figuras (PDF + PNG, tamanhos ACM, versões principal e apêndice)
e tabelas LaTeX, a partir de resultados sintéticos que passaram pelo
analyze.py e pela estatística reais.
"""

import matplotlib
import pytest
from matplotlib.colors import to_rgb

import make_figures as mf
from synthetic_results import TEXT_WIDTH, build, eval_reports


@pytest.fixture(scope="module")
def synth():
    return build()


@pytest.fixture
def figs(synth, tmp_path, monkeypatch):
    cfg, res = synth
    monkeypatch.setattr(mf, "CONFIG", cfg)
    eval_reports(tmp_path / "rep")
    paths = mf.make_all(res, mf.FIGURES[:-2] + ["tables"], tmp_path / "figs", ["main", "all"],
                        tables_dir=tmp_path / "tables", report_dir=tmp_path / "rep")
    return cfg, res, tmp_path, paths


def test_every_figure_in_pdf_and_png(figs):
    _, _, tmp, paths = figs
    names = {p.split("/")[-1] for p in paths}
    for base in ("fig1_divergence_map_w3", "fig1_divergence_map_raw", "fig2_delta_cs",
                 "fig2_delta_cs_qa", "fig3_peak_vs_final", "fig4_internal_vs_behavior",
                 "fig5_names_vs_indicators", "fig6_lens_comparison"):
        for scope in ("", "_all"):
            for ext in ("pdf", "png"):
                n = (f"{base}{scope}.{ext}" if not base.startswith("fig1")
                     else base.replace("_map", f"_map{scope}") + f".{ext}")
                assert n in names, n
    assert all((tmp / "figs" / n).stat().st_size > 1000 for n in names if n.endswith((".pdf", ".png")))


def test_figure_sizes_fit_acm_page(synth, tmp_path, monkeypatch):
    cfg, res = synth
    monkeypatch.setattr(mf, "CONFIG", cfg)
    sizes = {}
    orig = mf.save

    def spy(fig, base):
        sizes[str(base).split("/")[-1]] = tuple(fig.get_size_inches())
        return orig(fig, base)

    monkeypatch.setattr(mf, "save", spy)
    mf.make_all(res, ["fig1", "fig2", "fig4"], tmp_path, ["main"])
    # uma coluna (acmart manuscript): toda figura na largura de texto do config
    assert all(w == pytest.approx(TEXT_WIDTH) for w, _ in sizes.values())
    assert all(h < 9.0 for _, h in sizes.values())                # cabe na página


def test_palette_is_colorblind_safe_and_complete(synth, monkeypatch):
    cfg, _ = synth
    monkeypatch.setattr(mf, "CONFIG", cfg)
    okabe_ito = {"#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"}
    colors = {mf.lang_color(l) for l in cfg.languages}
    assert colors <= okabe_ito and len(colors) == len(cfg.languages)
    dashes = {str(mf.lang_dash(l)) for l in cfg.languages}
    assert len(dashes) == len(cfg.languages)                     # traço distinto por idioma
    assert len({to_rgb(mf.model_style(m)[0]) for m in cfg.models}) == len(cfg.models)


def test_main_scope_uses_main_models(synth, monkeypatch):
    cfg, _ = synth
    monkeypatch.setattr(mf, "CONFIG", cfg)
    assert mf.figure_models("main") == ["llama", "mistral", "qwen", "gemma"]
    assert "llama_base" in mf.figure_models("all")


def test_dumbbell_includes_controls(synth, monkeypatch):
    cfg, _ = synth
    monkeypatch.setattr(mf, "CONFIG", cfg)
    panels = mf.fig_panels(include_controls=True)
    assert ("penicilina", "fleming") in panels and ("aviao", "santos_dumont") in panels
    assert ("penicilina", "fleming") not in mf.fig_panels()


def test_results_tables(figs):
    cfg, res, tmp, _ = figs
    compact = (tmp / "tables" / "results_compact.tex").read_text()
    full = (tmp / "tables" / "results_full.tex").read_text()
    assert r"\begin{table}[t]" in compact and r"\toprule" in compact and r"\bottomrule" in compact
    assert "table*" not in compact                                # documento de uma coluna
    assert r"\begin{longtable}" in full
    T = mf.results_table(res, mf.figure_models("main"), compact=True)
    assert set(T.columns) == {"Model", "Fact", "Candidate", "Lang", "Peak $k^*/L$", "CS$^*$",
                              "Supp.", "$\\Delta$CS", "Primary $e$", "Majority"}
    # compacto: só idiomas de stake do candidato + EN
    sd = T[T["Candidate"] == "Alberto Santos-Dumont"]
    assert {l[:2] for l in sd["Lang"]} == {"PT", "FR", "EN"}
    assert any("$^{N}$" in l for l in sd["Lang"]) and any("$^{P}$" in l for l in sd["Lang"])
    full_T = mf.results_table(res, mf.figure_models("all"), compact=False)
    assert len(full_T) > len(T) and "LLaMA" not in "".join(full_T["Model"])
    assert (T["Majority"] != "—").all()          # comportamento presente nos instruct
    assert "%" not in compact.replace(r"\%", "")  # % escapado


def test_figures_skip_without_data(tmp_path, synth, monkeypatch):
    cfg, _ = synth
    monkeypatch.setattr(mf, "CONFIG", cfg)
    assert mf.make_all({}, mf.FIGURES[:-2], tmp_path, ["main"]) == []


def test_tex_escape():
    assert mf._tex_escape("Bell & Meucci_1 (60%)") == r"Bell \& Meucci\_1 (60\%)"
    assert mf._tex_escape("PT$^{N}$") == "PT$^{N}$"


def test_decision_start_and_layers():
    assert mf.decision_start(40) == 29
    assert matplotlib.get_backend().lower() == "agg"


def test_figures_refuse_without_measured_width(synth, tmp_path, monkeypatch):
    import dataclasses
    cfg, res = synth
    exp = {**cfg.experiment, "figures": {**cfg.experiment["figures"], "text_width_in": None}}
    monkeypatch.setattr(mf, "CONFIG", dataclasses.replace(cfg, experiment=exp))
    with pytest.raises(mf.FigureWidthUnset, match="measure_textwidth"):
        mf.make_all(res, ["fig2"], tmp_path, ["main"])


def test_real_config_width_measured():
    from config.experiment_config import ExperimentConfig
    f = ExperimentConfig().experiment["figures"]
    assert f["text_width_in"] == 5.95 and "manuscript" in f["document_class"]   # 430.00462pt
