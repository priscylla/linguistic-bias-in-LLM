# RUNNING — one GPU (24–48 GB)

Every step is required. On one GPU only one job runs at a time, so the order below puts the main Logit Lens results first (so writing can start), behavioral generation next, and Tuned Lens training (the longest step) overnight. Every script resumes where it stopped: if a run is interrupted, run the same command again.

Model keys (from `config/models.yaml`): instruct `llama`, `mistral`, `qwen`, `gemma`, `gemma27`; base `llama_base`, `mistral_base`, `qwen_base`, `gemma_base`; Portuguese-adapted `sabia` and its base `sabia_base`.

## GPU time and memory

The times below are **estimates**, not measurements. They come from the workload counted by the full dry run (`python run_pipeline_dry.py`, all 11 facts × 5 languages) and assumed per-operation costs on a 24–48 GB GPU in bf16: about 0.05–0.15 s per forward pass with every layer read out (7–9B model), and about 25–40 ms per generation step (a batch of 10 samples costs about one step). Gemma 2 27B in bf16 costs roughly 3× a 9B model.

Workload per model (dry run):
- Lens run: 800 prompts + 90 neutral baseline prompts + forced article/contraction passes (299 on a random model; real models trigger the article test more often, up to about 800 more), plus one forward pass per shared first token (prefix forcing). Base models skip the 165 QA prompts (635 prompts, 75 baseline).
- Behavioral (instruct only): 165 QA prompts × (1 greedy + 10 samples) = 1,815 answers, at most 200 new tokens each.
- Tuned Lens: 250 steps × 65,536 tokens = 16M tokens of FineWeb-2 / FineWeb.

| Step | Per model | Total on one GPU |
| --- | --- | --- |
| 1. Sanity check | 2–5 min (model load) | ~45 min (11 models) |
| 2. Logit Lens | 5–12 min; `gemma27` 15–30 min | 1–2.5 h (11 models) |
| 3. Behavioral | 25–45 min; `gemma27` 1–2 h | 3–5 h (5 models) |
| 4. Tuned Lens training | `mistral` ~3 h, `qwen` ~7 h, `llama` ~8 h, `gemma` ~16 h | ~34 h (4 models) |
| 5. Tuned Lens re-measurement | 5–15 min | ~1 h (4 models) |
| 6. Analysis, statistics, figures | CPU | < 30 min |

**Total: about 40–44 GPU-hours (≈ 2 days)**, of which Tuned Lens training is about 34 h. First-time downloads come on top (about 230 GB in total).

The dry run itself (CPU, two tiny models, whole pipeline) takes about 2 minutes and 1.1 GB of RAM.

| Model | Weights in GPU memory | Download |
| --- | --- | --- |
| `llama`, `llama_base` (Llama 3.1 8B) | ~16 GB | 16 GB each |
| `mistral`, `mistral_base` (Mistral 7B v0.3) | ~14.5 GB | 14.5 GB each |
| `qwen`, `qwen_base` (Qwen 2.5 7B) | ~15 GB | 15 GB each |
| `gemma` (Gemma 2 9B it) | ~18.5 GB | 18.5 GB |
| `gemma_base` (Gemma 2 9B, stored in fp32) | ~18.5 GB | 37 GB |
| `gemma27` (Gemma 2 27B it, bf16) | ~55 GB (~58 GB generating) | 54.5 GB |
| `sabia`, `sabia_base` (7B, LLaMA-1 architecture) | ~13.5 GB | 13.5 GB each |

Lens runs and generation add little memory at these prompt lengths. Tuned Lens training adds the translators, their gradients and the SGD momentum (about 6 GB in float32 for `llama` and `gemma`, 4 GB for `qwen`) plus the logits of one 512-position chunk (1–2 GB). Estimated peak: `mistral` ~22 GB, `qwen` ~21 GB, `llama` ~24 GB, `gemma` ~27 GB. Every model trains with the registered settings (`translator_dtype: float32`, `seq_len: 2048` in `config/tuned_lens.yaml`); there is no reduced-memory variant, so `gemma` needs a GPU with at least 32 GB. `train_tuned_lens.py` prints the peak VRAM at the end.

**`gemma27` runs in bf16 (registry default since 2026-10-29; ~55 GB of weights)**, so it doesn't fit on a single 24–48 GB card. Run its sanity, lens and behavioral steps on a GPU with ≥ 64 GB, or split it over two GPUs (`--gpus "0 1"`). `--dtype nf4` (the earlier 4-bit setting, ~17 GB) is a different method: it is recorded as `nf4` in the manifest and in `reports/sanity/gemma27_nf4.json`, and a run never mixes it with bf16 results.

## 0. Setup (once)

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Python >= 3.11, fresh environment (pins differ from the old code's)
pip install -r requirements.txt         # Linux + CUDA; driver < 580: first pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
hf auth login                           # = huggingface-cli login; or export HF_TOKEN="hf_..." (never commit it)
python scripts/check_env.py             # FIRST: Python, packages, GPUs (memory, bf16), disk, HF access; must show 0 FAIL
pytest                                  # 389 tests, all green
python run_pipeline_dry.py              # whole pipeline on tiny models; must end with "dry run ok"
python verify_setup.py                  # tokenizers: gated access, last token of every prompt, name tokenization
```

`check_env.py` checks:
- Python ≥ 3.11 and the installed packages against `requirements.txt`;
- torch with CUDA, and every GPU's name, memory and bf16 support;
- free disk for the model cache (~230 GB) and for `results/` and `lenses/`;
- the Hugging Face token, then access to every model at its pinned commit and to the Tuned Lens corpus (no download).

It exits with 1 on any blocking problem. A placeholder `HF_TOKEN` in `.env` is ignored in favour of the `hf auth login` token (`.env` is not modified).

Accept the licenses of the gated models (Llama 3.1 base and instruct, Gemma 2 9B base and it, Gemma 2 27B it) on Hugging Face before starting; `check_env.py` confirms access and the ~230 GB of free disk for the model cache.

Runs refuse to start until every value they use is confirmed. **Everything is confirmed** (2026-10-29): facts, prompts (FR, DE and IT reviewed by native speakers), neutral baseline prompts, forcing words and behavioral surface forms; `reports/verified_values.md` logs each confirmation, and `reports/review_pt_en_and_support.md` confirms that nothing is still proposed. If you edit any of these files later, set its status back to `proposed` until you re-confirm it.

`python monitor.py` shows progress in another terminal at any point.

## 1. Sanity check (every model, first)

The last-layer lens must equal the model's own logits, in every format the model is used in. Stop if any model fails.

```bash
for m in llama mistral qwen gemma gemma27 \
         llama_base mistral_base qwen_base gemma_base sabia sabia_base; do
  python run_lens.py --model $m --sanity-only || break
done
```

Reports go to `reports/sanity/{model}.json`. The sanity check doesn't need the confirmed values, so it can run before the native review is finished.

## 2. Logit Lens runs (day 1, morning)

All facts, controls, languages, prompt types, paraphrases and formats (QA only for instruct models). Results go to `results/lens/logit_lens/{model}/`.

```bash
for m in llama mistral qwen gemma \
         llama_base mistral_base qwen_base gemma_base \
         sabia sabia_base gemma27; do
  python run_lens.py --model $m --lens logit
done
```

Once the instruct models are done, `python analyze.py && python make_figures.py` already works on partial results, so writing can start.

## 3. Behavioral generation (day 1, afternoon)

Instruct models only: QA prompts with the chat template, 1 greedy answer + 10 samples at T=0.7 (pure temperature: top-p 1, top-k 0), at most 200 new tokens. Results go to `results/behavioral/{model}/`.

```bash
for m in llama mistral qwen gemma gemma27; do
  python run_behavioral.py --model $m
done
python scripts/export_validation_sample.py   # reports/validation/sample.csv (13 answers per model × language)
```

Label the validation sample while the next step runs: fill `human_primary` (the first candidate cited, or `none`) and `human_cited` (all cited candidate ids, comma-separated). The export script refuses to overwrite an existing sample, so labels are never lost.

## 4. Tuned Lens training (night)

The script skips any model that already has a finished lens, or a pretrained one declared under `pretrained` in `config/tuned_lens.yaml` (none exists for these checkpoints). The corpus is streamed from Hugging Face, so the machine needs internet access. Training resumes from `lenses/{model}/checkpoint.pt`; `--max-steps N` stops early and the next run continues.

```bash
for m in llama mistral qwen gemma; do
  python train_tuned_lens.py --model $m
done
```

Each model gets `lenses/{model}/` and a held-out evaluation in `reports/tuned_lens/{model}_eval.csv` (KL per language and layer, Tuned vs Logit Lens).

## 5. Tuned Lens re-measurement (day 2, morning)

```bash
for m in llama mistral qwen gemma; do
  python run_lens.py --model $m --lens tuned
done
```

Results go to `results/lens/tuned_lens/{model}/`.

## 6. Analysis, figures, tables (CPU; here or after unpacking)

```bash
python check_results.py                        # every expected prompt, unit and answer present; GPU-hours
python scripts/validation_agreement.py         # after labeling: reports/validation/agreement.json
python analyze.py                              # results/metrics/: metrics, CIs, stake tests, mixed models
python scripts/fill_empirical_dominant.py      # writes the English majority into facts.yaml (warns on disagreement)
python make_figures.py                         # figures/ (PDF + PNG), tables/ (LaTeX + CSV)
```

Figures use the paper's `\textwidth`, already set: `figures.text_width_in: 5.95` in `config/experiment.yaml` (430.00 pt, acmart v2.21 manuscript; measured with `reports/latex/measure_textwidth.tex`).

Release (author, at the end): `python scripts/make_anonymous_copy.py` writes `../linguistic-bias-in-LLM-anonymous` (no names, emails, tokens or git history) and fails if anything identifying is left in the copy.

Step 6 can also run elsewhere, after unpacking the results from step 7.

## 7. Pack the results and send them back (last step)

```bash
python scripts/pack_results.py --include-lenses   # packs/results-<date>-<commit>.tar.gz + .sha256
```

It runs `check_results.py` first. Then it packs:
- `results/`, which includes every manifest;
- `reports/sanity/`, `reports/tuned_lens/`, `reports/validation/`, the tokenization reports and `reports/check_env.json`;
- `logs/`;
- with `--include-lenses`, the trained translators (~8 GB, about 34 GPU-hours of training).

It prints the size. If `check_results.py` found problems, the pack is still made and marked incomplete in its `PACK_INFO.json` (`--require-complete` refuses instead). Unpack at the repository root: `tar -xzf results-….tar.gz`.
