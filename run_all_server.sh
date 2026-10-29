#!/usr/bin/env bash
# run_all_server.sh
# Executa todo o RUNNING_server.md de uma vez, no servidor de 4 GPUs:
#   0. ambiente (venv Python 3.12, torch conforme o driver, requirements),
#      check_env, testes, dry run, verify_setup
#   1. sanity check dos 11 modelos nas 4 GPUs (para tudo se algum falhar)
#   2. fila do Tuned Lens (4 treinos, um por GPU, cada um seguido da remedição)
#   3. fila principal (Logit Lens + comportamental), ao mesmo tempo, nas mesmas GPUs
#      → amostra de validação quando a fila principal termina
#   4. checagem final (check_results)
#   5. empacotamento dos resultados (com as lentes treinadas)
#
# Leva cerca de 18–20 h. Rode dentro de tmux/screen ou com nohup:
#   nohup ./run_all_server.sh > run_all.log 2>&1 &
#   tail -f run_all.log
#
# Opções:
#   --from N     começa da etapa N (0–5). Tudo retoma de onde parou, então
#                rodar de novo desde o início também funciona (só repete os testes).
#   --gpus "..." GPUs a usar (padrão "0 1 2 3", numeração do nvidia-smi).
#                As tags [gpu=N] das filas assumem 4 GPUs; com outra lista,
#                ajuste jobs_tuned.txt e jobs_main.txt abaixo.
#   -h, --help   esta ajuda
#
# Variáveis de ambiente opcionais:
#   PYTHON   interpretador para criar o venv (padrão python3.12)
#   HF_HOME  onde guardar os modelos (~230 GB)
#   HF_TOKEN token do Hugging Face (ou faça antes: hf auth login)

set -uo pipefail

FROM=0
GPUS="0 1 2 3"
PYTHON=${PYTHON:-python3.12}

while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "opção desconhecida: $1 (veja --help)" >&2; exit 2 ;;
  esac
done
case "$FROM" in [0-5]) ;; *) echo "--from deve ser 0 a 5" >&2; exit 2 ;; esac

cd "$(dirname "$0")" || exit 1
mkdir -p logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID

ts()   { date '+%Y-%m-%d %H:%M:%S'; }
step() { echo; echo "=== [$(ts)] $* ==="; }
die()  { echo "[$(ts)] ERRO: $*" >&2; exit 1; }
run()  { echo "[$(ts)] \$ $*"; "$@" || die "falhou: $*"; }

# ── Ambiente virtual (sempre ativado, mesmo com --from > 0) ──
if [ ! -d .venv ]; then
  [ "$FROM" -eq 0 ] || die ".venv não existe; rode a partir da etapa 0"
  step "Criando o ambiente virtual com $PYTHON"
  command -v "$PYTHON" >/dev/null || die "$PYTHON não encontrado (defina PYTHON=...)"
  run "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate || die "não consegui ativar .venv"

# ── Etapa 0: instalação e checagens ──
if [ "$FROM" -le 0 ]; then
  step "Etapa 0: instalação e checagens do ambiente"
  command -v nvidia-smi >/dev/null || die "nvidia-smi não encontrado: sem driver NVIDIA?"
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
  DRIVER_MAJOR=${DRIVER%%.*}
  echo "driver NVIDIA: $DRIVER"
  run python -m pip install --upgrade pip
  if [ "${DRIVER_MAJOR:-0}" -lt 580 ] 2>/dev/null; then
    echo "driver < 580: instalando torch 2.11.0 para CUDA 12.8 (o do PyPI exige driver >= 580)"
    run python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
  fi
  run python -m pip install -r requirements.txt
  if [ -z "${HF_TOKEN:-}" ] && ! hf auth whoami >/dev/null 2>&1; then
    die "sem token do Hugging Face: rode 'hf auth login' ou 'export HF_TOKEN=hf_...' e tente de novo"
  fi
  run python scripts/check_env.py
  run python -m pytest -q
  run python run_pipeline_dry.py
  run python verify_setup.py
fi

# ── Etapa 1: sanity check ──
if [ "$FROM" -le 1 ]; then
  step "Etapa 1: sanity check dos 11 modelos"
  for m in gemma27 gemma_base gemma llama_base llama mistral_base mistral \
           qwen_base qwen sabia_base sabia; do
    echo "python run_lens.py --model $m --sanity-only"
  done > jobs_sanity.txt
  scripts/run_queue.sh jobs_sanity.txt "$GPUS" \
    || die "sanity check falhou; veja a lista acima e os logs em logs/queue/jobs_sanity-*/. Corrija e rode de novo com --from 1"
fi

# ── Etapas 2 e 3: as duas filas ao mesmo tempo ──
if [ "$FROM" -le 3 ]; then
  cat > jobs_tuned.txt <<'EOF'
[gpu=0] python train_tuned_lens.py --model gemma   && python run_lens.py --model gemma   --lens tuned
[gpu=1] python train_tuned_lens.py --model llama   && python run_lens.py --model llama   --lens tuned
[gpu=2] python train_tuned_lens.py --model qwen    && python run_lens.py --model qwen    --lens tuned
[gpu=3] python train_tuned_lens.py --model mistral && python run_lens.py --model mistral --lens tuned
EOF
  cat > jobs_main.txt <<'EOF'
# Gemma 2 27B em bf16 (~55 GB): só na GPU 3, depois do treino do Mistral, com 60 GB livres.
# Nunca na GPU 0, onde treina o Gemma 2 9B.
[gpu=3] [after=reports/tuned_lens/mistral_eval.csv] [mem=60] python run_behavioral.py --model gemma27
[gpu=3] [after=reports/tuned_lens/mistral_eval.csv] [mem=60] python run_lens.py --model gemma27 --lens logit
# O resto em qualquer GPU, os mais longos primeiro
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

  step "Etapas 2 e 3: fila do Tuned Lens e fila principal (em paralelo)"
  echo "logs: logs/tuned.log e logs/main.log; logs por job em logs/queue/"
  scripts/run_queue.sh jobs_tuned.txt "$GPUS" > logs/tuned.log 2>&1 &
  PID_TUNED=$!
  scripts/run_queue.sh jobs_main.txt "$GPUS" > logs/main.log 2>&1 &
  PID_MAIN=$!
  echo "fila do Tuned Lens: PID $PID_TUNED; fila principal: PID $PID_MAIN"

  wait "$PID_MAIN"; RC_MAIN=$?
  echo "[$(ts)] fila principal terminou (código $RC_MAIN)"
  grep -h "^fim:" logs/main.log || true
  run python scripts/export_validation_sample.py

  echo "[$(ts)] aguardando a fila do Tuned Lens (o treino do Gemma 2 9B leva ~16 h)"
  wait "$PID_TUNED"; RC_TUNED=$?
  echo "[$(ts)] fila do Tuned Lens terminou (código $RC_TUNED)"
  grep -h "^fim:" logs/tuned.log || true

  if [ "$RC_MAIN" -ne 0 ] || [ "$RC_TUNED" -ne 0 ]; then
    echo
    echo "ATENÇÃO: algum job falhou (veja as linhas 'job ...' em logs/main.log e logs/tuned.log)."
    echo "Rode de novo com:  ./run_all_server.sh --from 2"
    echo "O que já terminou é pulado. Os resultados serão empacotados mesmo assim, marcados como incompletos."
  fi
fi

# ── Etapa 4: checagem final ──
if [ "$FROM" -le 4 ]; then
  step "Etapa 4: checagem final dos resultados"
  if ! python check_results.py; then
    echo "ATENÇÃO: check_results encontrou pendências (veja acima). O pacote será marcado como incompleto."
  fi
fi

# ── Etapa 5: empacotar ──
step "Etapa 5: empacotando os resultados (com as lentes treinadas)"
run python scripts/pack_results.py --include-lenses

step "Concluído"
echo "Envie o arquivo .tar.gz e o .sha256 da pasta packs/."
ls -lh packs/ 2>/dev/null | tail -n +2
