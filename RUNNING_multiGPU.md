# RUNNING — multiple GPUs

Every step is required. For the 4 × 96 GB server there is a concrete schedule in `RUNNING_server.md`. With several GPUs, jobs for different models run in parallel, one model per GPU, and Tuned Lens training (the longest step) runs on its own GPUs alongside everything else. Every script resumes where it stopped: if a job is interrupted, run it again.

Model keys (from `config/models.yaml`): instruct `llama`, `mistral`, `qwen`, `gemma`, `gemma27`; base `llama_base`, `mistral_base`, `qwen_base`, `gemma_base`; Portuguese-adapted `sabia` and its base `sabia_base`.

## How jobs are placed

`scripts/run_queue.sh jobs.txt "0 1 2 3"` takes a file with one command per line and a list of GPU IDs, and keeps every GPU busy until the list is empty.
- **Workers:** each entry in the GPU list is one worker. A comma joins GPUs into one worker for a model split across them (`"0 1,2 3"`: worker 2 gets GPUs 1 and 2).
- **Job order:** a worker always takes the next line of the file, so put the longest jobs first.
- **Pinning:** each job runs with `CUDA_VISIBLE_DEVICES` set to its worker's GPUs. Don't pass `--gpus` inside queued jobs. GPU numbers follow `nvidia-smi`.
- **Conditions:** optional tags at the start of a line.
  - `[gpu=3]`: only the worker with exactly those GPUs runs the job.
  - `[after=FILE]`: wait until FILE exists, for example the end of a job in another queue.
  - `[mem=60]`: wait until each of the worker's GPUs has 60 GB free.
  - `[timeout=45m]`: time limit for the job (`s`, `m` or `h`). When it runs out, the job and all its processes are killed and it counts as failed (code 124); rerunning it is safe.

  A waiting job doesn't block the queue; the worker runs other jobs meanwhile.
- **Failures:** each job writes its own log to `logs/queue/<name>-<date>/jobN.log`. A failing job is recorded and the queue continues; at the end the script lists the failures and exits with 1.
- **No collisions:** each model writes to its own folders (`results/lens/{lens}/{model}/`, `results/behavioral/{model}/`, `lenses/{model}/`), so parallel jobs never write to the same file.

Outside the queue, `--gpus "1 2"` does the same for a single command. Set the GPU IDs to what `nvidia-smi` shows on your machine.

## Wall-clock time and memory

The times are **estimates**, not measurements. They come from the workload counted by the full dry run (`python run_pipeline_dry.py`, all 11 facts × 5 languages) and assumed per-operation costs on a 24–48 GB GPU in bf16: about 0.05–0.15 s per forward pass with every layer read out, about 25–40 ms per generation step, and Gemma 2 27B (bf16) roughly 3× a 9B model.

GPU time per job:
- **Sanity check:** 2–5 min each.
- **Logit Lens:** 5–12 min each; `gemma27` 15–30 min.
- **Behavioral:** 25–45 min each; `gemma27` 1–2 h.
- **Tuned Lens training:** `gemma` ~16 h, `llama` ~8 h, `qwen` ~7 h, `mistral` ~3 h.
- **Tuned Lens re-measurement:** 5–15 min each.

In total, about 40–44 GPU-hours. Gemma's Tuned Lens training (~16 h) is the longest single job, so it sets the wall-clock time from about 4 GPUs up.

| GPUs | Logit Lens + behavioral | Tuned Lens training | Whole set |
| --- | --- | --- | --- |
| 2 | 5–8 h on GPU 0 | GPU 1 from the start; GPU 0 joins when its queue is empty | ~22–24 h |
| 4 | 2–4 h on GPUs 0–1 | GPUs 2–3 (`gemma` alone on one) | ~17–18 h |
| 8 | 1–2 h on GPUs 0–3 | GPUs 4–7, one model each | ~16–17 h |

Memory per model (weights; lens runs and generation add little):

| Model | GPU memory | Download |
| --- | --- | --- |
| `llama`, `llama_base` | ~16 GB | 16 GB each |
| `mistral`, `mistral_base` | ~14.5 GB | 14.5 GB each |
| `qwen`, `qwen_base` | ~15 GB | 15 GB each |
| `gemma` | ~18.5 GB | 18.5 GB |
| `gemma_base` (stored in fp32) | ~18.5 GB | 37 GB |
| `gemma27` (bf16; one ≥ 64 GB GPU or two) | ~55 GB (~58 GB generating) | 54.5 GB |
| `sabia`, `sabia_base` | ~13.5 GB | 13.5 GB each |

Estimated Tuned Lens training peak (translators, gradients and SGD momentum in float32, plus per-chunk logits): `mistral` ~22 GB, `qwen` ~21 GB, `llama` ~24 GB, `gemma` ~27 GB. Every model trains with the registered settings (`translator_dtype: float32`, `seq_len: 2048`); there is no reduced-memory variant, so `gemma` needs a GPU with at least 32 GB. Disk for the model cache: about 230 GB in total. Every machine that runs a job needs the cache (or downloads its models), and Tuned Lens training needs internet access (the corpus is streamed).

## 0. Setup (once)

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Python >= 3.11, fresh environment (pins differ from the old code's)
pip install -r requirements.txt         # Linux + CUDA; driver < 580: first pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
hf auth login                           # = huggingface-cli login; or export HF_TOKEN="hf_..." (never commit it)
python scripts/check_env.py             # FIRST: Python, packages, GPUs (memory, bf16), disk, HF access; must show 0 FAIL
pytest                                  # 421 tests, all green
python run_pipeline_dry.py              # whole pipeline on tiny models; must end with "dry run ok"
python verify_setup.py                  # tokenizers: gated access, prompts, name tokenization
mkdir -p logs
```

`check_env.py` checks:
- Python ≥ 3.11 and the installed packages against `requirements.txt`;
- torch with CUDA, and every GPU's name, memory and bf16 support;
- free disk for the model cache (~230 GB) and for `results/` and `lenses/`;
- the Hugging Face token, then access to every model at its pinned commit and to the Tuned Lens corpus (no download).

It exits with 1 on any blocking problem. A placeholder `HF_TOKEN` in `.env` is ignored in favour of the `hf auth login` token (`.env` is not modified).

Accept the licenses of the gated models (Llama 3.1 base and instruct, Gemma 2 9B base and it, Gemma 2 27B it) on Hugging Face.

Runs refuse to start until every value they use is confirmed. **Everything is confirmed** (2026-10-29): facts, prompts (FR, DE and IT reviewed by native speakers), neutral baseline prompts, forcing words and behavioral surface forms; `reports/verified_values.md` logs each confirmation, and `reports/review_pt_en_and_support.md` confirms that nothing is still proposed. If you edit any of these files later, set its status back to `proposed` until you re-confirm it.

**Gemma 2 27B precision:** the registry runs `gemma27` in bf16 without quantization (since 2026-10-29; before, 4-bit NF4). It needs about 58 GB:
- **GPUs of ≥ 64 GB:** one GPU is enough, so it runs like any other job.
- **Smaller GPUs:** give its jobs a two-GPU worker (`"1,2"`) or `--gpus "1 2"`.
- **Never** put it on a GPU that is also training a Tuned Lens unless the memory adds up. `[mem=60]` in the queue checks this.

`--dtype nf4` reproduces the old 4-bit setting. It is a different method: recorded in the manifest, kept apart in `reports/sanity/gemma27_nf4.json`, and a run refuses to resume over results made with the other precision.

## 1. Sanity check (all models in parallel)

```bash
for m in llama mistral qwen gemma gemma27 \
         llama_base mistral_base qwen_base gemma_base sabia sabia_base; do
  echo "python run_lens.py --model $m --sanity-only"
done > jobs_sanity.txt
scripts/run_queue.sh jobs_sanity.txt "0 1 2 3"
```

Stop if any model fails (the queue exits with 1 and lists the failures). Reports go to `reports/sanity/{model}.json`.

## 2. Start Tuned Lens training right away (dedicated GPUs)

It is the longest step, so it starts first and runs while everything else uses the remaining GPUs. The jobs are ordered longest first. Each job trains a model and then re-measures it with the trained lens.

```bash
for m in gemma llama qwen mistral; do
  echo "python train_tuned_lens.py --model $m && python run_lens.py --model $m --lens tuned"
done > jobs_tuned.txt
nohup scripts/run_queue.sh jobs_tuned.txt "3" > logs/tuned.log 2>&1 &     # 2–4 GPUs: one or two GPUs ("2 3")
```

With 8 GPUs, give each model its own GPU (`"4 5 6 7"`). With 2 GPUs, use `"1"` here; when GPU 0 finishes step 3, start a second queue on it with the same `jobs_tuned.txt` in reverse order (`tac jobs_tuned.txt > jobs_tuned_rev.txt`). Overlaps are safe: a model with a finished lens is skipped, and a model still training on the other GPU is refused (`lenses/{model}/TRAINING.lock`; that job fails in the queue log and the other GPU finishes it). After a crash, delete the stale lock file before rerunning.

## 3. Logit Lens and behavioral runs (remaining GPUs)

Longest jobs first. Results go to `results/lens/logit_lens/{model}/` and `results/behavioral/{model}/`.

```bash
{
  echo "[mem=60] python run_behavioral.py --model gemma27"          # bf16: wait for a GPU with room
  echo "[mem=60] python run_lens.py --model gemma27 --lens logit"
  for m in llama mistral qwen gemma; do echo "python run_behavioral.py --model $m"; done
  for m in llama mistral qwen gemma llama_base mistral_base qwen_base gemma_base sabia sabia_base; do
    echo "python run_lens.py --model $m --lens logit"
  done
} > jobs_main.txt
scripts/run_queue.sh jobs_main.txt "0 1 2"
python scripts/export_validation_sample.py     # reports/validation/sample.csv (13 answers per model × language)
```

Label the validation sample while training runs: fill `human_primary` (first candidate cited, or `none`) and `human_cited` (all cited candidate ids, comma-separated).

## 4. Tuned Lens re-measurement (after training finishes)

Step 2 already re-measures each model right after its training. This step catches any model whose re-measurement didn't run; finished work is skipped, so it costs nothing when everything is done.

```bash
for m in llama mistral qwen gemma; do
  echo "python run_lens.py --model $m --lens tuned"
done > jobs_retune.txt
scripts/run_queue.sh jobs_retune.txt "0 1 2 3"
```

## 5. Analysis, figures, tables (CPU; here or after unpacking)

```bash
python check_results.py                        # every expected prompt, unit and answer present; GPU-hours
python scripts/validation_agreement.py         # after labeling: reports/validation/agreement.json
python analyze.py                              # results/metrics/: metrics, CIs, stake tests, mixed models
python scripts/fill_empirical_dominant.py      # writes the English majority into facts.yaml (warns on disagreement)
python make_figures.py                         # figures/ (PDF + PNG), tables/ (LaTeX + CSV)
```

Figures use the paper's `\textwidth`, already set: `figures.text_width_in: 5.95` in `config/experiment.yaml` (430.00 pt, acmart v2.21 manuscript; measured with `reports/latex/measure_textwidth.tex`).

Release (author, at the end): `python scripts/make_anonymous_copy.py` writes `../linguistic-bias-in-LLM-anonymous` (no names, emails, tokens or git history) and fails if anything identifying is left in the copy. Step 5 can also run elsewhere, after unpacking the results from step 6.

## 6. Pack the results and send them back (last step)

```bash
python scripts/pack_results.py --include-lenses   # packs/results-<date>-<commit>.tar.gz + .sha256
```

It runs `check_results.py` first. Then it packs:
- `results/`, which includes every manifest;
- `reports/sanity/`, `reports/tuned_lens/`, `reports/validation/`, the tokenization reports and `reports/check_env.json`;
- `logs/`;
- with `--include-lenses`, the trained translators (~8 GB).

It prints the size. If `check_results.py` found problems, the pack is still made and marked incomplete in its `PACK_INFO.json` (`--require-complete` refuses instead). Unpack at the repository root: `tar -xzf results-….tar.gz`.
