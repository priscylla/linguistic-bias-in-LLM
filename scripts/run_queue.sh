#!/usr/bin/env bash
# scripts/run_queue.sh
# Mantém uma lista de GPUs ocupada com uma lista de jobs.
#
#   scripts/run_queue.sh jobs.txt "0 1 2 3"     # 4 workers, uma GPU cada
#   scripts/run_queue.sh jobs.txt "0 1,2"       # worker 2 usa as GPUs 1 e 2 juntas
#
# jobs.txt: um comando por linha (linhas vazias e # comentários ignorados).
# Cada entrada da lista de GPUs é um worker; vírgula junta GPUs num mesmo
# worker (modelo dividido entre elas). Cada worker pega o próximo job da
# fila e o roda com CUDA_VISIBLE_DEVICES = suas GPUs; não passe --gpus
# dentro dos jobs da fila (sobrescreveria a escolha do worker).
#
# Cada job grava log próprio; um job que falha é registrado e a fila
# segue. No fim, resumo e código de saída 1 se algum job falhou.
# Estado e logs em logs/queue/<nome>-<data>/ (ou $QUEUE_STATE).
# Compatível com bash 3.2 (macOS) e Linux.

set -u

if [ $# -ne 2 ]; then
  echo "uso: $0 jobs.txt \"0 1 2 3\"" >&2
  exit 2
fi
JOBS_FILE=$1
GPUS=$2
[ -f "$JOBS_FILE" ] || { echo "arquivo de jobs não encontrado: $JOBS_FILE" >&2; exit 2; }

STATE=${QUEUE_STATE:-logs/queue/$(basename "$JOBS_FILE" .txt)-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$STATE"
grep -v '^[[:space:]]*#' "$JOBS_FILE" | grep -v '^[[:space:]]*$' > "$STATE/jobs" || true
N=$(wc -l < "$STATE/jobs" | tr -d ' ')
echo 0 > "$STATE/next"
: > "$STATE/status"
echo "fila: $N jobs, workers: $GPUS  (estado em $STATE)"

# Próximo índice da fila, com trava (mkdir é atômico)
claim() {
  while ! mkdir "$STATE/lock" 2>/dev/null; do sleep 0.1; done
  local i
  i=$(cat "$STATE/next")
  echo $((i + 1)) > "$STATE/next"
  rmdir "$STATE/lock"
  echo "$i"
}

worker() {
  local gpu=$1 i n cmd log rc t0
  while :; do
    i=$(claim)
    [ "$i" -ge "$N" ] && break
    n=$((i + 1))
    cmd=$(sed -n "${n}p" "$STATE/jobs")
    log="$STATE/job${n}.log"
    t0=$(date +%s)
    echo "[gpu $gpu] início job $n: $cmd"
    ( export CUDA_VISIBLE_DEVICES="$gpu"; bash -c "$cmd" ) > "$log" 2>&1
    rc=$?
    printf '%s\t%s\t%s\t%s\t%s\n' "$n" "$rc" "$gpu" "$(( $(date +%s) - t0 ))" "$cmd" >> "$STATE/status"
    if [ "$rc" -eq 0 ]; then
      echo "[gpu $gpu] ok     job $n ($(( $(date +%s) - t0 ))s)"
    else
      echo "[gpu $gpu] FALHOU job $n (código $rc) → $log"
    fi
  done
}

for g in $GPUS; do
  worker "$g" &
done
wait

FAILED=$(awk -F'\t' '$2 != 0' "$STATE/status" | wc -l | tr -d ' ')
DONE=$(wc -l < "$STATE/status" | tr -d ' ')
echo "fim: $DONE/$N jobs, $FAILED falha(s). Resumo: $STATE/status"
if [ "$FAILED" -ne 0 ]; then
  awk -F'\t' '$2 != 0 {print "  job " $1 " (gpu " $3 "): " $5}' "$STATE/status"
  exit 1
fi
exit 0
