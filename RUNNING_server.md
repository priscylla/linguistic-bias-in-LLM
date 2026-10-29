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
pytest                                                     # 339 tests, CPU
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
  echo "python run_lens.py --model $m --sanity-only"
done > jobs_sanity.txt
scripts/run_queue.sh jobs_sanity.txt "0 1 2 3"
```

Stop if the queue exits with 1: it lists the failed jobs, each with its log in `logs/queue/jobs_sanity-<date>/`. Reports go to `reports/sanity/{model}.json`.

## 2. Tuned Lens queue (background): four trainings at once, one per GPU

Each job trains one lens and then re-measures that model with it, so re-measurements happen as each training finishes. Settings are the registered ones for every model: float32 translators, `seq_len` 2048, 16M tokens (`config/tuned_lens.yaml`).

```bash
cat > jobs_tuned.txt <<'EOF'
[gpu=0] python train_tuned_lens.py --model gemma   && python run_lens.py --model gemma   --lens tuned
[gpu=1] python train_tuned_lens.py --model llama   && python run_lens.py --model llama   --lens tuned
[gpu=2] python train_tuned_lens.py --model qwen    && python run_lens.py --model qwen    --lens tuned
[gpu=3] python train_tuned_lens.py --model mistral && python run_lens.py --model mistral --lens tuned
EOF
nohup scripts/run_queue.sh jobs_tuned.txt "0 1 2 3" > logs/tuned.log 2>&1 &
```

## 3. Main queue (background, at the same time): Logit Lens and behavioral

Start it right after step 2, on the same GPUs.

```bash
cat > jobs_main.txt <<'EOF'
# Gemma 2 27B in bf16 (~55 GB): GPU 3 only, after Mistral's Tuned Lens training there has
# finished, and only with 60 GB free. Never on GPU 0, where Gemma 2 9B trains.
[gpu=3] [after=reports/tuned_lens/mistral_eval.csv] [mem=60] python run_behavioral.py --model gemma27
[gpu=3] [after=reports/tuned_lens/mistral_eval.csv] [mem=60] python run_lens.py --model gemma27 --lens logit
# Everything else on any GPU, longest first
python run_behavioral.py --model gemma
python run_behavioral.py --model llama
python run_behavioral.py --model mistral
python run_behavioral.py --model qwen
python run_lens.py --model gemma --lens logit
python run_lens.py --model gemma_base --lens logit
python run_lens.py --model llama --lens logit
python run_lens.py --model llama_base --lens logit
python run_lens.py --model mistral --lens logit
python run_lens.py --model mistral_base --lens logit
python run_lens.py --model qwen --lens logit
python run_lens.py --model qwen_base --lens logit
python run_lens.py --model sabia --lens logit
python run_lens.py --model sabia_base --lens logit
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
