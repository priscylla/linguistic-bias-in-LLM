# linguistic-bias-in-LLM

Logit Lens and Tuned Lens study of cultural and linguistic bias in LLMs: for contested inventions (airplane, telephone, radio, …) and undisputed controls, it measures, layer by layer and in five prompt languages (PT, EN, DE, IT, FR), how strongly a model commits to each candidate inventor (the Commitment Score, CS_e = P(e) − P(dominant)), and compares that internal commitment with what the model actually answers.

## Requirements

- **Python ≥ 3.11** (the pinned numpy, pandas, scipy and pyarrow require it); tested on 3.12 and 3.13.
- **A fresh virtual environment.** The pinned versions in `requirements.txt` (transformers 5.7, torch 2.11, …) differ from the ones the old code used, so don't reuse an old environment.
- **GPU:** Linux and NVIDIA GPUs with bf16 (compute capability ≥ 8.0, e.g. A100). The PyPI wheel of torch 2.11.0 is built for CUDA 13.0 and needs NVIDIA driver ≥ 580; with an older driver, first run `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`.
- **Disk:** about 230 GB for the model cache.
- **Hugging Face:** a token with access to the gated models (Llama 3.1, Gemma 2).
  - The token comes from `HF_TOKEN`, from `.env`, or from `hf auth login` (= `huggingface-cli login`).
  - A placeholder `HF_TOKEN` in `.env` (not a real-looking `hf_…` token) is ignored with a warning, and the login token is used instead; `.env` itself is never modified.

`python scripts/check_env.py` checks all of this, plus every model's pinned commit, without downloading anything. It is the first step of every run guide:
- `RUNNING_server.md`: concrete schedule for a 4 × 96 GB GPU server;
- `RUNNING_multiGPU.md`: any number of GPUs;
- `RUNNING_1GPU.md`: a single GPU.

## Pipeline

One entry point per stage. Every model writes to its own files, every run resumes from checkpoints, and nothing downstream modifies raw results.

```bash
# 0. Setup (once), Python >= 3.11 in a fresh environment
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login                            # or export HF_TOKEN="hf_..."; gated: Llama 3.1, Gemma 2 (accept the licenses first)
python scripts/check_env.py              # Python, packages, GPUs (memory, bf16), disk, HF access; must show 0 FAIL
pytest                                   # 339 tests, CPU, tiny random models, no downloads
python run_pipeline_dry.py               # whole pipeline on tiny models (CPU, ~2–3 min)
python verify_setup.py                   # tokenizers only: prompts, name tokenization

# 1. Sanity check, every model first (last-layer lens == model logits)
python run_lens.py --model llama --sanity-only           # reports/sanity/llama.json

# 2. Logit Lens measurement (all facts, languages, prompt types, paraphrases, formats)
python run_lens.py --model llama --lens logit            # results/lens/logit_lens/llama/

# 3. Behavioral study (instruct models: QA prompts, 1 greedy + 10 samples at T=0.7)
python run_behavioral.py --model llama                   # results/behavioral/llama/

# 4. Tuned Lens: train, then re-measure (the four instruct models)
python train_tuned_lens.py --model llama                 # lenses/llama/, reports/tuned_lens/
python run_lens.py --model llama --lens tuned            # results/lens/tuned_lens/llama/

# 5. Checks, metrics, statistics, figures, tables (CPU)
python check_results.py                  # every expected prompt, unit and answer present
python analyze.py                        # results/metrics/ (metrics, CIs, tests, mixed models)
python scripts/export_validation_sample.py     # reports/validation/sample.csv → label it
python scripts/validation_agreement.py         # reports/validation/agreement.json
python scripts/fill_empirical_dominant.py      # empirical English majority → facts.yaml
python make_figures.py                   # figures/ (PDF + PNG), tables/ (LaTeX + CSV)
                                         # needs figures.text_width_in: compile reports/latex/measure_textwidth.tex

# 6. Send the results back, release
python scripts/pack_results.py --include-lenses   # packs/results-<date>-<commit>.tar.gz (+ .sha256): results,
                                                  # manifests, sanity / Tuned Lens reports, logs, translators
python scripts/make_anonymous_copy.py    # ../linguistic-bias-in-LLM-anonymous (no names, emails, tokens, history)
```

`python monitor.py` shows progress in another terminal. Runs refuse to start until every value they use is confirmed (facts, prompts and native-speaker review, baseline prompts, forcing words; surface forms for the behavioral analysis); see `reports/verified_values.md`. As of 2026-10-29 everything is confirmed (`reports/review_pt_en_and_support.md` lists anything still proposed). Tuned Lens training needs only the confirmed facts.

When candidates of the same fact share a first token, the shared token is forced and each candidate adds P(prefix) × P(next | prefix), layer by layer, repeated while they still share tokens (up to 3); every case is recorded (`prefix_forcing` table and manifest). In the article-confound test, elided articles (it *l'*, *un'*; fr *l'*) force the stem and measure each candidate as apostrophe + name; contraction passes run on every prompt whose last word has listed contracted forms (pt/it/fr blocked prompts, German *von → vom*, *in → im*). Behavioral matching of *Bell* is case-sensitive. Place and artifact prompts run for a fact × language only when at least two candidates have indicators and, for the model's tokenizer, are measured in every paraphrase; otherwise the cell is skipped and recorded (`skipped` table and manifest). `python verify_setup.py` shows the result for every tokenizer in `reports/tokenization_summary.md`.

### Options shared by the model scripts

| Option | Scripts | Meaning |
| --- | --- | --- |
| `--model KEY` | all | key from `config/models.yaml` (below) |
| `--gpus "1 2"` | all | GPUs for this job; two or more split the model across them. Without it, `CUDA_VISIBLE_DEVICES` applies |
| `--dtype bf16\|fp16\|fp32\|nf4` | `run_lens.py`, `run_behavioral.py` | override the registry precision (e.g. `nf4` reproduces the 4-bit Gemma 2 27B used before 2026-10-29). Changes the method: recorded in the manifest, and a run never resumes over results made with another precision |
| `--lens logit\|tuned` | `run_lens.py` | Logit Lens or the trained Tuned Lens |
| `--sanity-only` | `run_lens.py` | only the sanity check |
| `--consolidate-only` | `run_lens.py`, `run_behavioral.py` | rebuild the tables from finished checkpoints |
| `--max-steps N`, `--eval-only`, `--force` | `train_tuned_lens.py` | short session (resumes later), evaluation only, retrain a finished lens |

`train_tuned_lens.py` skips a model that already has a finished lens, or a pretrained one declared in `config/tuned_lens.yaml` (none exists for these checkpoints).

### Several GPUs

`scripts/run_queue.sh jobs.txt "0 1 2 3"` keeps a list of GPUs busy with a list of jobs (one command per line). Each entry of the GPU list is a worker; a comma joins GPUs into one worker for a model split across them (`"0 1,2 3"`). Each job gets its own log in `logs/queue/`, a failing job is recorded and the queue continues, and the script exits with 1 if any job failed. Don't pass `--gpus` inside queued jobs. Optional tags at the start of a job line:
- `[gpu=3]` pins the job to one worker;
- `[after=FILE]` waits until FILE exists;
- `[mem=60]` waits for 60 GB free on the worker's GPUs.

A waiting job doesn't block the others. GPU numbers follow `nvidia-smi` (`CUDA_DEVICE_ORDER=PCI_BUS_ID`). `RUNNING_server.md` uses the tags to run Tuned Lens training and everything else at the same time on the same GPUs.

```bash
for m in llama mistral qwen gemma; do echo "python run_lens.py --model $m --lens logit"; done > jobs.txt
scripts/run_queue.sh jobs.txt "0 1 2 3"
```

## Models (`config/models.yaml`)

| Key | Model | Kind | Precision | Formats |
| --- | --- | --- | --- | --- |
| `llama_base` / `llama` | Llama 3.1 8B / Instruct | base / instruct | bf16 | completion / + QA |
| `mistral_base` / `mistral` | Mistral 7B v0.3 / Instruct | base / instruct | bf16 | completion / + QA |
| `qwen_base` / `qwen` | Qwen 2.5 7B / Instruct | base / instruct | bf16 | completion / + QA |
| `gemma_base` / `gemma` | Gemma 2 9B / it | base / instruct | bf16 | completion / + QA |
| `gemma27` | Gemma 2 27B it | instruct | bf16 (~55 GB: one ≥ 64 GB GPU, or two) | completion + QA |
| `sabia_base` / `sabia` | LLaMA-1 7B (huggyllama) / Sabiá-7B | base / base (pt-adapted) | bf16 | completion |

Every model is pinned to a commit. Behavioral runs use the five instruct models; Tuned Lens uses `llama`, `mistral`, `qwen`, `gemma`.

## Expected runtimes and memory

Workload per model, counted by the full dry run (`run_pipeline_dry.py`, all 11 facts × 5 languages): **800 prompts + 90 neutral baseline prompts + forced article/contraction passes** per lens (299 on a random model; real models trigger the article test more often, up to about 1,100 more), and for behavior **165 QA prompts × 11 answers = 1,815 answers**, at most 200 new tokens each (≈ 363k tokens if no answer stops early). Shared first tokens add one forward pass per collision (prefix forcing). The GPU times below are **estimates** from those counts, not measurements: one forward with every layer read out takes roughly 0.05–0.15 s for a 7–9B model in bf16, and generation roughly 25–40 ms per step (a batch of 10 samples costs about one step).

| Step | Per model (24–48 GB GPU) | All models |
| --- | --- | --- |
| Sanity check | 2–5 min (+ first download) | ~45 min (11 models) |
| Logit Lens | 5–12 min; Gemma 2 27B bf16 15–30 min | 1–2.5 h (11 models) |
| Behavioral | 25–45 min; Gemma 2 27B bf16 1–2 h | 3–5 h (5 instruct) |
| Tuned Lens training (16M tokens) | Mistral ~3 h, Qwen ~7 h, Llama ~8 h, Gemma ~16 h | ~34 h (4 models) |
| Tuned Lens re-measurement | 5–15 min | ~1 h (4 models) |
| Analysis, statistics, figures | CPU, minutes | < 30 min |

On the dry run (CPU, tiny models, 2 models) the whole pipeline took about 105 s and 1.1 GB of RAM; behavioral generation dominated (80 s).

GPU memory (weights; activations add little at these prompt lengths): Llama 3.1 8B ~16 GB, Mistral 7B ~14.5 GB, Qwen 2.5 7B ~15 GB, Gemma 2 9B ~18.5 GB, Gemma 2 27B bf16 ~55 GB (about 58 GB while generating; one GPU of ≥ 64 GB, or two smaller ones), LLaMA-1 / Sabiá 7B ~13.5 GB. Tuned Lens training adds the translators, their gradients and the SGD momentum (~6 GB in float32 for Llama and Gemma) plus per-chunk logits: estimated peak ~22 GB (Mistral), ~21 GB (Qwen), ~24 GB (Llama), ~27 GB (Gemma). Every model uses the registered settings (`translator_dtype: float32`, `seq_len: 2048`); there is no reduced-memory variant, so Gemma's training needs a GPU of at least 32 GB.

## Configuration

Everything that varies lives in YAML, read and validated by `config/loader.py`:

- `config/experiment.yaml`: languages, prompt types, decision zone, smoothing, statistics (`statistics`), figures (`figures`);
- `config/facts.yaml`: facts, candidates, single-name sets per language, stakes, narrative indicators;
- `config/prompts.yaml`: every prompt (3 paraphrases per type) with its review and grammar flags;
- `config/baseline_prompts.yaml`: neutral prompts (same slot shape as each real paraphrase) for the common-token baseline;
- `config/forcing.yaml`: articles, plural nouns and contracted prepositions for the article-confound test;
- `config/models.yaml`: model registry and, per architecture, where the final norm, unembedding and logit softcap live;
- `config/tuned_lens.yaml`: Tuned Lens corpus (FineWeb-2 + FineWeb, pinned, balanced across languages, 1% held out, fact documents filtered out) and recipe;
- `config/behavioral.yaml`: generation settings, surface forms for answer extraction, validation sample, report questions.

`python scripts/make_review_doc.py` regenerates the review documents `reports/phase1_review.md`, `reports/prompts_review.md`, `reports/behavioral_review.md` and `reports/review_pt_en_and_support.md`. Every value confirmed by the author is logged in `reports/verified_values.md`.

## Code map

| Path | Role |
| --- | --- |
| `logit_lens/lens.py`, `tuned_lens.py` | pluggable `Lens` (model head + per-layer translation): Logit Lens, Tuned Lens |
| `logit_lens/measure.py` | exact per-candidate probability over distinct token IDs, KL to the final layer, top-20, neutral baseline |
| `logit_lens/confound.py` | article-confound passes (article + plural noun, contractions, shared prefixes) |
| `logit_lens/lens_training.py` | Tuned Lens data, training (resumable) and held-out evaluation |
| `logit_lens/sanity.py`, `store.py` | sanity report; checkpointed tables (`results/lens/{lens}/{model}/`) |
| `behavior/` | generation with explicit decoding settings, extraction, validation sample and agreement |
| `analyze.py`, `analysis/statistics.py` | metrics (CS_e, ΔCS, peak, suppression, article verdicts, behavior) and statistics (bootstrap CIs, stake tests, mixed models) |
| `make_figures.py` | figures 1–6 + appendix heatmaps (single-column acmart manuscript width, Okabe-Ito palette) and LaTeX tables |
| `models/` | registry loader, chat template, `--gpus` / `--dtype` handling |
| `devtools/tiny.py` | tiny random models and toy tokenizers for tests and the dry run |
| `scripts/check_env.py` | first step on a new machine: Python, packages, GPUs (memory, bf16), disk, Hugging Face access at the pinned commits |
| `scripts/run_queue.sh` | keeps GPUs busy with a job list; `[gpu=]`, `[after=]`, `[mem=]` conditions |
| `scripts/pack_results.py` | last step: results, manifests, reports and logs (optionally lenses) in one tarball |

Figures are sized for the FAccT 2027 format (`\documentclass[manuscript,screen,review,anonymous]{acmart}`, single column): every figure is `\textwidth` wide, read from `figures.text_width_in` in `config/experiment.yaml`, which must be measured once by compiling `reports/latex/measure_textwidth.tex`. Main version with the four instruct models; `_all` appendix version with every model:

| File | Content |
| --- | --- |
| `fig1_divergence_map{,_all}_{w3,raw}` | CS_e per layer, rows = fact × alternative, columns = models, shared y per row, decision zone |
| `fig2_delta_cs{,_qa}{,_all}` | ΔCS (decision zone) per fact × language, one point per model with 95% bootstrap CI, stake languages shaded |
| `fig3_peak_vs_final{,_all}` | peak vs final-layer P(e), controls beside disputed facts |
| `fig4_internal_vs_behavior{,_all}` | CS* and decision-zone CS vs behavioral primary-attribution rate |
| `fig5_names_vs_indicators{,_all}` | ΔCS from names vs places vs artifacts |
| `fig6_lens_comparison{,_all}` | Logit vs Tuned Lens: ΔCS, peak shift, held-out KL per layer |
| `figA_tokens_*` | top tokens per layer (qualitative, appendix) |
| `tables/results_{compact,full}.tex` | model, fact, candidate, language, peak k*/L, CS*, suppression, ΔCS, behavior |
