"""
tests/test_run_pipeline.py
Fase 2 ponta a ponta com modelo minúsculo: run_lens (com checkpoint),
check_results, analyze (sem tocar nos brutos) e make_figures.
"""

import dataclasses
import hashlib

import pandas as pd
import pytest

import analyze
import check_results
import make_figures as mf
import run_lens
from config.experiment_config import ExperimentConfig
from conftest import WordTokenizer, build_tiny_model
from logit_lens.store import ChunkStore, lens_dir

LANGS = ("pt", "en")
FACTS = ("aviao", "penicilina")
CHAT  = lambda tok, text: f"<user>{text}<assistant>"  # noqa: E731


def tiny_config():
    base = ExperimentConfig()
    exp  = {**base.experiment, "languages": list(LANGS), "heatmap_models": ["llama"]}
    return ExperimentConfig(
        experiment=exp,
        fact_specs={k: base.fact_specs[k] for k in FACTS},
        prompt_specs=tuple(p for p in base.prompt_specs if p.fact in FACTS and p.lang in LANGS),
        baseline_spec=dataclasses.replace(
            base.baseline_spec,
            prompts=tuple(p for p in base.baseline_spec.prompts if p.lang in LANGS)),
        forcing_spec=base.forcing_spec,
        registry=dataclasses.replace(base.registry, models={"llama": base.registry["llama"]}))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    cfg   = tiny_config()
    root  = tmp_path_factory.mktemp("results")
    model = build_tiny_model("llama", vocab=4096)
    tok   = WordTokenizer()
    out   = lens_dir(str(root), "llama")
    man   = run_lens.run_model("llama", model, tok, cfg, out, model_id="tiny/llama",
                               chat_formatter=CHAT, log=lambda *a: None)
    return {"cfg": cfg, "root": root, "out": out, "model": model, "tok": tok, "manifest": man}


def _digest(d):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def test_manifest(run_dir):
    m, cfg = run_dir["manifest"], run_dir["cfg"]
    assert m["units_done"] == m["units_expected"] == len(LANGS) * (1 + len(FACTS))
    assert m["formats"] == ["completion", "qa"]
    assert m["model_id"] == "tiny/llama" and m["dtype"] == "float32"
    assert m["lens"] == "logit_lens" and m["quantization"] is None
    assert {"transformers", "torch", "python"} <= set(m["versions"])
    assert m["gpu"]["name"] == "cpu" and m["gpu_hours"] == 0.0
    assert set(m["config_sha256"]) == set(run_lens.CONFIG_FILES)
    assert "revision" in m and m["sessions"][0]["units_run"] == m["units_done"]


def test_tables_cover_every_prompt(run_dir):
    cfg, out = run_dir["cfg"], run_dir["out"]
    probs = pd.read_parquet(out / "probs.parquet")
    keys  = set(map(tuple, probs[["fact", "lang", "prompt_type", "paraphrase"]].drop_duplicates().values))
    assert keys == {(p.fact, p.lang, p.type, p.paraphrase) for p in cfg.prompt_specs}
    assert set(probs["format"]) == {"completion", "qa"}
    assert set(probs["token_set"]) == {"names", "indicators"}
    n = run_dir["model"].config.num_hidden_layers
    assert (probs.groupby(["fact", "lang", "prompt_type", "paraphrase", "candidate"]).size() == n).all()
    # uma linha por modelo × fato × idioma × formato × tipo × paráfrase × camada × candidato × conjunto
    assert not probs.duplicated(["model", "fact", "lang", "format", "prompt_type", "paraphrase",
                                 "layer", "candidate", "token_set"]).any()

    base = pd.read_parquet(out / "baseline.parquet")
    assert len(base[["lang", "prompt_type", "paraphrase"]].drop_duplicates()) == len(cfg.baseline_spec.prompts)
    assert set(base["fact"]) == set(FACTS)

    layers = pd.read_parquet(out / "layers.parquet")
    assert layers["top_ids"].map(len).eq(20).all()
    last = layers[layers["layer"] == layers["n_layers"]]
    assert (last["kl_to_final"] < 1e-5).all()

    forced = pd.read_parquet(out / "forced.parquet")
    # pt F1.1 e place 1–2 de aviao são article_blocked: uma passada por contração
    c = forced[(forced.forced_kind == "contraction") & (forced.lang == "pt")]
    assert {"contraction:pelos", "contraction:pelo", "contraction:na"} <= set(c["pass_id"])

    ts = pd.read_parquet(out / "token_sets.parquet")
    assert {"prompt", "baseline"} == set(ts["source"])


def test_resume_skips_done_units(run_dir):
    cfg, out = run_dir["cfg"], run_dir["out"]
    before = pd.read_parquet(out / "probs.parquet")
    m = run_lens.run_model("llama", run_dir["model"], run_dir["tok"], cfg, out,
                           chat_formatter=CHAT, log=lambda *a: None)
    assert m["sessions"][-1]["units_run"] == 0

    import shutil
    shutil.rmtree(out / "chunks" / "aviao__en")
    m = run_lens.run_model("llama", run_dir["model"], run_dir["tok"], cfg, out,
                           chat_formatter=CHAT, log=lambda *a: None)
    assert m["sessions"][-1]["units_run"] == 1 and len(m["sessions"]) == 3
    after = pd.read_parquet(out / "probs.parquet")
    cols = ["fact", "lang", "prompt_type", "paraphrase", "candidate", "layer"]
    pd.testing.assert_frame_equal(before.sort_values(cols).reset_index(drop=True),
                                  after.sort_values(cols).reset_index(drop=True))


def test_store_refuses_overwrite(run_dir):
    with pytest.raises(FileExistsError):
        ChunkStore(run_dir["out"]).write("aviao__pt", {"probs": []})


def test_check_results(run_dir, tmp_path):
    cfg = run_dir["cfg"]
    assert check_results.check_model("llama", str(run_dir["root"]), cfg) == []
    # Unidade faltando é detectada
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(run_dir["root"], copy)
    shutil.rmtree(lens_dir(str(copy), "llama") / "chunks" / "penicilina__pt")
    probs = lens_dir(str(copy), "llama") / "probs.parquet"
    df = pd.read_parquet(probs)
    df[df.fact != "penicilina"].to_parquet(probs, index=False)
    problems = check_results.check_model("llama", str(copy), cfg)
    assert any("unidades faltando" in p for p in problems)
    assert any("prompts sem probs" in p for p in problems)


def test_analyze_and_figures(run_dir, tmp_path, monkeypatch):
    root = run_dir["root"]
    raw  = _digest(root / "lens")
    out  = tmp_path / "metrics"
    assert analyze.main(["--results-dir", str(root), "--output-dir", str(out)]) == 0
    assert _digest(root / "lens") == raw                        # brutos intactos
    for name in ("curves", "metrics", "confound", "curves_mean", "delta_cs",
                 "stats_ci", "stats_stake_tests", "stats_mixed_fits"):
        assert (out / f"{name}.parquet").exists()
    ci = pd.read_parquet(out / "stats_ci.parquet")
    assert {"F1-F3", "QA", "place"} <= set(ci["prompt_group"])
    met = pd.read_parquet(out / "metrics.parquet")
    alt = met[(met.fact == "aviao") & ~met.dominant & (met.token_set == "names")]
    assert alt["cs_final"].notna().all()
    assert set(met["article_verdict"]) <= {"not_triggered", "undefined", *analyze.VERDICTS}
    conf = pd.read_parquet(out / "confound.parquet")
    assert set(conf[conf.forced_kind == "contraction"]["verdict"]) <= {"undefined", *analyze.VERDICTS}
    delta = pd.read_parquet(out / "delta_cs.parquet")
    assert (delta[delta.lang == "en"]["d_cs_final"].dropna() == 0).all()

    with pytest.raises(ValueError):
        analyze.main(["--results-dir", str(root), "--output-dir", str(root / "lens" / "x")])

    # Figuras com o config reduzido
    monkeypatch.setattr(mf, "CONFIG", run_dir["cfg"])
    figs = tmp_path / "figs"
    for fig in ("fig1", "fig6"):
        assert mf.main(["--results-dir", str(root), "--metrics-dir", str(out),
                        "--output-dir", str(figs), "--fig", fig]) == 0
    assert {p.name for p in figs.iterdir()} == {"fig1_w3.png", "fig1_raw.png", "fig6.png"}
    tables  = mf.load_tables(str(root), ("layers", "token_sets"))
    metrics = mf.load_metrics(out)
    run = mf.heatmap_run(tables, metrics, "llama", "aviao", "pt")
    assert len(run["layers"]) == run_dir["model"].config.num_hidden_layers
    assert run["dominant_ids"] and run["other_ids"]
    mf.fig3_logit_lens_single(tables, metrics, "aviao", "llama", "pt", str(figs / "fig3.png"))
    mf.fig3_logit_lens_combined(tables, metrics, "aviao", str(figs / "fig3c.png"))
    assert (figs / "fig3.png").exists() and (figs / "fig3c.png").exists()
    assert mf.model_layers(metrics["curves_mean"], "llama") == run_dir["model"].config.num_hidden_layers
    assert mf.decision_start(40) == 29
