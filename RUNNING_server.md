# RUNNING — the 4-GPU server

A concrete schedule for one machine with **4 GPUs of about 96 GB each**, 1 TB of RAM and disk, Python 3.12 and internet access, with every model license accepted. For other machines see `RUNNING_1GPU.md` and `RUNNING_multiGPU.md`. Nothing here changes the method: every command is the same as in the other guides, only the placement differs.

The plan in one paragraph: check the machine, sanity-check all 11 models on the 4 GPUs, then run **two queues at once on the same 4 GPUs**. One trains the four Tuned Lenses, one per GPU, and re-measures each model as soon as its lens is ready. The other runs every Logit Lens and behavioral job. Each GPU holds at most two jobs (a training at ~22–27 GB plus one other job), which fits in 96 GB. Gemma 2 27B (bf16, ~55 GB) runs only on GPU 3, after Mistral's training there has finished, so it never shares a GPU with the Gemma 2 9B training on GPU 0.

Every script resumes where it stopped; if anything is interrupted, run the same command again.

## Placement

| GPU | Tuned Lens queue (background) | Main queue (background, same GPUs) |
| --- | --- | --- |
| 0 | `gemma` training (~16 h, ~27 GB), then `gemma` re-measurement | Logit Lens and behavioral jobs, one at a time (≤ 19 GB each) |
| 1 | `llama` training (~8 h, ~24 GB), then re-measurement | same |
| 2 | `qwen` training (~7 h, ~21 GB), then re-measurement | same |
| 3 | `mistral` training (~3 h, ~22 GB), then re-measurement | same; **`gemma27` only here, after Mistral's training, with 60 GB free** |

Both queues use `scripts/run_queue.sh`. A tag at the start of a job line constrains where and when it runs:
- `[gpu=3]`: only the GPU-3 worker runs the job.
- `[after=FILE]`: the job waits until FILE exists. `reports/tuned_lens/mistral_eval.csv` is written when Mistral's training and its held-out evaluation are done.
- `[mem=60]`: the job waits until the GPU has 60 GB free (`nvidia-smi`).
- `[timeout=45m]`: the job's time limit (seconds, `m` or `h`). When it runs out, the job and every process it started get SIGTERM, then SIGKILL 30 s later. The job counts as failed (code 124) and the queue moves on.
  - **Why:** on 2026-10-29 a `run_lens.py` job hung for 15 h after finishing its work.
  - **Cleanup:** `run_lens.py` and `run_behavioral.py` now exit directly after writing their results, which skips the interpreter cleanup that can hang.
  - **Rerunning:** a job stopped this way is safe to rerun as is, since finished units are skipped. `check_results.py` says whether anything is missing.

A job that is waiting doesn't block the queue: the worker runs other jobs meanwhile and comes back to it. GPU numbers are the ones `nvidia-smi` shows.

## Wall-clock estimate

These are **estimates**, not measurements, on the same basis as the other guides: per-operation costs of a 24–48 GB GPU in bf16, plus about 30% slowdown while two jobs share a GPU. On this server's GPUs the compute-bound steps (above all Tuned Lens training) should be faster. The training logs give the real rate after a few minutes (see step 3).

| Time from start | What runs |
| --- | --- |
| 0:00–0:20 | step 0: install, `check_env.py`, tests, dry run, `verify_setup.py` |
| 0:20–1:20 | step 1: sanity checks, 11 models on 4 GPUs (includes the first download, about 230 GB) |
| 1:20 | steps 2 and 3 start together |
| 1:20–3:00 | main queue: 4 behavioral runs (25–45 min each) and 10 Logit Lens runs (5–12 min each) |
| ~5:00 | Mistral's training ends (GPU 3) and its re-measurement starts; `gemma27` starts on GPU 3 |
| ~5:00–7:30 | `gemma27` behavioral (1–2 h) and Logit Lens (15–30 min) in bf16 |
| ~8:30–10:00 | Qwen's and Llama's trainings end, then their re-measurements |
| ~18:00–19:00 | Gemma 2 9B training ends (GPU 0), then its re-measurement (~15 min) |
| + 0:20 | step 5: checks, validation sample, pack |

**Total: about 18–20 hours**, almost all of it spent waiting for the Gemma 2 9B Tuned Lens training (about 16 h), which can't be split across GPUs. The main queue and `gemma27` are done in the first 7–8 hours. GPU-hours over the whole run: about 40–44.

## 0. Setup and environment check (once)

```bash
git clone <repository URL> linguistic-bias-in-LLM && cd linguistic-bias-in-LLM
python3.12 -m venv .venv && source .venv/bin/activate     # fresh environment: the pins differ from the old code's
nvidia-smi                                                 # driver version (first line)
# If the driver is older than 580: the PyPI torch 2.11.0 is built for CUDA 13.0, so first run
#   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
hf auth login                                              # same as huggingface-cli login; or export HF_TOKEN=hf_...
python scripts/check_env.py                                # must end with 0 FAIL
pytest                                                     # 389 tests, CPU
python run_pipeline_dry.py                                 # whole pipeline on tiny models; must end with "dry run ok"
python verify_setup.py                                     # tokenizers: gated access, prompts, name tokenization
mkdir -p logs
```

`check_env.py` checks:
- **Python and packages:** Python ≥ 3.11, and the installed packages against `requirements.txt`.
- **torch:** the version, its CUDA build and the driver.
- **GPUs:** name, memory and bf16 support of every GPU, and whether `gemma27` (bf16, ~64 GB with activations), the Gemma Tuned Lens training (~27 GB) and two jobs per GPU fit.
- **Disk:** free space for the model cache (230 GB minus what is already downloaded) and for `results/` and `lenses/`.
- **Hugging Face:** the token, then every model at its pinned commit (gated ones: license accepted) and the two Tuned Lens corpus datasets. Nothing is downloaded.

It writes `reports/check_env.json`, which goes into the final pack. Fix every FAIL before going on.

On tokens:
- A placeholder `HF_TOKEN` in `.env` (anything that isn't a real-looking `hf_…` token) is ignored with a warning, and the login token is used instead; `.env` itself is never modified.
- Never commit a token.

If `check_env.py` reports less than 80 GB on some GPU, this schedule doesn't apply as written. `gemma27` in bf16 then needs a two-GPU worker (see `RUNNING_multiGPU.md`).

## 1. Sanity check: all 11 models on the 4 GPUs

The last-layer lens must equal the model's own logits. One job per GPU at this stage; the largest model goes first.

```bash
for m in gemma27 gemma_base gemma llama_base llama mistral_base mistral \
         qwen_base qwen sabia_base sabia; do
  echo "[timeout=1h] python run_lens.py --model $m --sanity-only"
done > jobs_sanity.txt
scripts/run_queue.sh jobs_sanity.txt "0 1 2 3"
```

Stop if the queue exits with 1: it lists the failed jobs, each with its log in `logs/queue/jobs_sanity-<date>/`. Reports go to `reports/sanity/{model}.json`.

## 2. Tuned Lens queue (background): four trainings at once, one per GPU

Each job trains one lens and then re-measures that model with it, so re-measurements happen as each training finishes. The re-measurement runs whenever `reports/tuned_lens/{model}_eval.csv` exists, that is, once training and the held-out evaluation are done. It doesn't depend on the training's exit code: on 2026-10-29 two trainings wrote their results and then aborted at exit (code 134), and the old `&&` skipped their re-measurement. Settings are the registered ones for every model: float32 translators, `seq_len` 2048, 16M tokens (`config/tuned_lens.yaml`).

```bash
cat > jobs_tuned.txt <<'EOF'
[gpu=0] [timeout=12h] python train_tuned_lens.py --model gemma; test -f reports/tuned_lens/gemma_eval.csv && python run_lens.py --model gemma --lens tuned
[gpu=1] [timeout=12h] python train_tuned_lens.py --model llama; test -f reports/tuned_lens/llama_eval.csv && python run_lens.py --model llama --lens tuned
[gpu=2] [timeout=12h] python train_tuned_lens.py --model qwen; test -f reports/tuned_lens/qwen_eval.csv && python run_lens.py --model qwen --lens tuned
[gpu=3] [timeout=12h] python train_tuned_lens.py --model mistral; test -f reports/tuned_lens/mistral_eval.csv && python run_lens.py --model mistral --lens tuned
EOF
nohup scripts/run_queue.sh jobs_tuned.txt "0 1 2 3" > logs/tuned.log 2>&1 &
```

## 3. Main queue (background, at the same time): Logit Lens and behavioral

Start it right after step 2, on the same GPUs.

```bash
cat > jobs_main.txt <<'EOF'
# Gemma 2 27B in bf16 (~55 GB): GPU 3 only, after Mistral's Tuned Lens training there has
# finished, and only with 60 GB free. Never on GPU 0, where Gemma 2 9B trains.
[gpu=3] [after=reports/tuned_lens/mistral_eval.csv] [mem=60] [timeout=2h] python run_behavioral.py --model gemma27
[gpu=3] [after=reports/tuned_lens/mistral_eval.csv] [mem=60] [timeout=45m] python run_lens.py --model gemma27 --lens logit
# Everything else on any GPU, longest first
[timeout=2h] python run_behavioral.py --model gemma
[timeout=2h] python run_behavioral.py --model llama
[timeout=2h] python run_behavioral.py --model mistral
[timeout=2h] python run_behavioral.py --model qwen
[timeout=45m] python run_lens.py --model gemma --lens logit
[timeout=45m] python run_lens.py --model gemma_base --lens logit
[timeout=45m] python run_lens.py --model llama --lens logit
[timeout=45m] python run_lens.py --model llama_base --lens logit
[timeout=45m] python run_lens.py --model mistral --lens logit
[timeout=45m] python run_lens.py --model mistral_base --lens logit
[timeout=45m] python run_lens.py --model qwen --lens logit
[timeout=45m] python run_lens.py --model qwen_base --lens logit
[timeout=45m] python run_lens.py --model sabia --lens logit
[timeout=45m] python run_lens.py --model sabia_base --lens logit
EOF
nohup scripts/run_queue.sh jobs_main.txt "0 1 2 3" > logs/main.log 2>&1 &
```

Follow progress:

```bash
tail -f logs/tuned.log logs/main.log                       # one line per job start / end / wait
grep passo logs/queue/jobs_tuned-*/job1.log | tail -2     # Gemma training: seconds per step × 250 = total
nvidia-smi                                                 # at most two processes per GPU
python monitor.py                                          # lens and behavioral progress
```

The logs live in the queue folders:
- **Training logs:** each line of `jobs_tuned.txt` gets its own: `job1.log` is Gemma, `job2.log` Llama, `job3.log` Qwen and `job4.log` Mistral.
- **Main-queue logs:** `logs/queue/jobs_main-<date>/`.
- **The `gemma27` wait:** while Mistral is training, the main log shows `job 1 aguarda: esperando reports/tuned_lens/mistral_eval.csv` ("waiting for …"). That is expected.

If Mistral's training fails, the `gemma27` jobs keep waiting. Rerun the failed line from `jobs_tuned.txt` by hand with `--gpus 3`; the main queue picks `gemma27` up as soon as the file appears.

When the main queue ends (`fim:` in `logs/main.log`), export the behavioral validation sample. It goes into the pack, for labeling:

```bash
python scripts/export_validation_sample.py                 # reports/validation/sample.csv
```

## 4. When both queues have ended

Both logs end with a `fim:` line ("end:"). Check that neither queue had a failure:

```bash
grep -h "^fim:" logs/tuned.log logs/main.log               # both lines must say "0 falha(s)" (0 failures)
python check_results.py                                    # every expected prompt, unit and answer; GPU-hours
```

A failed job can be rerun as is (finished work is skipped): run its line from the jobs file again, with `--gpus N` added for a GPU that has room.

## 5. Pack the results and send them back

```bash
python scripts/pack_results.py --include-lenses
```

`pack_results.py` first runs `check_results.py`. It then packs everything below into `packs/results-<date>-<commit>.tar.gz`, writes a `.sha256` file next to it, and prints the size:
- `results/`, which includes every manifest;
- `reports/sanity/`, `reports/tuned_lens/`, `reports/validation/`, the tokenization reports and `reports/check_env.json`;
- `logs/`;
- with `--include-lenses`, the trained translators (`lenses/*/translators.pt` and `config.json`, about 8 GB). They cost about 34 GPU-hours to train, so send them.

If `check_results.py` finds problems, the pack is still made and `PACK_INFO.json` inside it marks it incomplete; `--require-complete` refuses instead. Send the `.tar.gz` and its `.sha256`. The recipient unpacks at the repository root with `tar -xzf results-….tar.gz` and runs the analysis on CPU (`RUNNING_1GPU.md`, step 6).

## 6. Full rerun of the lens measurements (changes of 2026-10-30)

The token gap audit (`reports/token_gap_audit.md`) changed what the lens runs measure, so every lens measurement is redone. No training, no behavioral generation, no sanity check (none of them is affected).

What changed:
- **Mistral QA:** measured on the token it actually emits. The separator before the answer now comes from each model's chat template; it is a space for Mistral and unchanged for the other models.
- **Name sets:** new given names (Joseph, Dimitri, Dmitry, Alexandre, Alessandro, Conrad, Timothy, Julius).
- **New forced words:** *Sir* in every language, *William* in the X-ray prompts, Gemma's `**`, and pt *como*, it *come*.
- **Forced passes:** their top-20 tokens are now stored.

Every unit therefore has a new fingerprint (`logit_lens/fingerprint.py`, measurement version 2). `run_lens.py` treats each old unit as out of date: it moves the unit to `results/lens/{lens}/{model}/chunks_stale/<time>/` (nothing is deleted) and measures it again. A run never resumes over old units.

```bash
git stash                                  # your local fixes of 2026-10-30 are in the commit; keeps a copy
git pull                                   # the commit with these changes
source .venv/bin/activate
python scripts/check_env.py                # must end with 0 FAIL (checks Hub access: run before going offline)
CUDA_VISIBLE_DEVICES= pytest               # all green, CPU only
mkdir -p logs
export HF_HUB_OFFLINE=1                    # every model is already in the cache: no Hub calls from here on

cat > jobs_rerun.txt <<'EOF'
# Logit Lens, 11 models (Gemma 2 27B first: largest)
[timeout=45m] python run_lens.py --model gemma27 --lens logit
[timeout=45m] python run_lens.py --model gemma --lens logit
[timeout=45m] python run_lens.py --model gemma_base --lens logit
[timeout=45m] python run_lens.py --model llama --lens logit
[timeout=45m] python run_lens.py --model llama_base --lens logit
[timeout=45m] python run_lens.py --model mistral --lens logit
[timeout=45m] python run_lens.py --model mistral_base --lens logit
[timeout=45m] python run_lens.py --model qwen --lens logit
[timeout=45m] python run_lens.py --model qwen_base --lens logit
[timeout=45m] python run_lens.py --model sabia --lens logit
[timeout=45m] python run_lens.py --model sabia_base --lens logit
# Tuned Lens re-measurement, 4 models (the trained lenses in lenses/ are kept)
[timeout=45m] python run_lens.py --model gemma --lens tuned
[timeout=45m] python run_lens.py --model llama --lens tuned
[timeout=45m] python run_lens.py --model mistral --lens tuned
[timeout=45m] python run_lens.py --model qwen --lens tuned
EOF
nohup scripts/run_queue.sh jobs_rerun.txt "0 1 2 3" > logs/rerun.log 2>&1 &
tail -f logs/rerun.log                     # ends with "fim: 15/15 jobs, 0 falha(s)"

python check_results.py                    # must end with "TODOS OS RESULTADOS COMPLETOS E VÁLIDOS"
python scripts/pack_results.py             # lenses unchanged: no --include-lenses
```

One job per GPU at a time, so Gemma 2 27B (~58 GB) always has its GPU to itself.

**Offline:** with `HF_HUB_OFFLINE=1`, a model load never waits on the Hub (an untimed Hub call can hang a job). Models and tokenizers also load from the local cache first even without it (`models/model_loader.py`). If a model is missing from the cache, the job fails at once instead of downloading. The variable stays set for the rest of that shell. **Tuned Lens training needs the network** (it streams its corpus from the Hub): run `unset HF_HUB_OFFLINE` first, or use a new shell. The rerun here trains nothing.

After the pull, `git stash drop` discards the stashed copy once the run works. `run.sh` (old pipeline) is not part of the repo and can be deleted.

**Time:** about 15–25 min wall-clock for the 15 jobs. On 2026-10-30, measuring all 60 units of a model took about 1 minute on these GPUs, without other jobs. The new forced passes add a little, and loading each model from the cache adds 1–5 min (Gemma 2 27B the longest). Each job gets a 45-minute limit.

**If something fails:**
- If a job fails or hits its limit, run its line again with `--gpus N`. The units it already finished are current and are skipped.
- `check_results.py` lists any unit still missing or measured with old inputs. The results pack leaves out the archived `chunks_stale/` folders, which stay on the server.
