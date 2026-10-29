#!/usr/bin/env bash
# scripts/run_queue.sh
# Mantém uma lista de GPUs ocupada com uma lista de jobs.
#
#   scripts/run_queue.sh jobs.txt "0 1 2 3"     # 4 workers, uma GPU cada
#   scripts/run_queue.sh jobs.txt "0 1,2"       # worker 2 usa as GPUs 1 e 2 juntas
#
# jobs.txt: um comando por linha (linhas vazias e # comentários ignorados).
# Cada entrada da lista de GPUs é um worker; vírgula junta GPUs num mesmo
# worker (modelo dividido entre elas). Cada worker pega o primeiro job da
# fila que pode rodar e o roda com CUDA_VISIBLE_DEVICES = suas GPUs; não
# passe --gpus dentro dos jobs da fila (sobrescreveria a escolha do worker).
#
# Condições opcionais no começo da linha (antes do comando):
#   [gpu=3]        só o worker com exatamente essas GPUs ("3", "1,2") roda o job
#   [after=ARQ]    espera ARQ existir (ex.: o fim de outro job, de outra fila)
#   [mem=60]       espera cada GPU do worker ter >= 60 GB livres (nvidia-smi)
# Ex.: [gpu=3] [after=reports/tuned_lens/mistral_eval.csv] [mem=60] python run_lens.py ...
# Um job que ainda não pode rodar não trava a fila: o worker pega o
# próximo e volta a ele depois (a cada $QUEUE_POLL s, padrão 30).
#
# Numeração das GPUs = nvidia-smi (CUDA_DEVICE_ORDER=PCI_BUS_ID).
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
POLL=${QUEUE_POLL:-30}
[ -f "$JOBS_FILE" ] || { echo "arquivo de jobs não encontrado: $JOBS_FILE" >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID

STATE=${QUEUE_STATE:-logs/queue/$(basename "$JOBS_FILE" .txt)-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$STATE/claimed"
grep -v '^[[:space:]]*#' "$JOBS_FILE" | grep -v '^[[:space:]]*$' > "$STATE/jobs" || true
N=$(wc -l < "$STATE/jobs" | tr -d ' ')
: > "$STATE/status"

# Separa as condições [chave=valor] do comando de uma linha.
# Saída: GPU, AFTER (separados por \n), MEM, CMD (variáveis globais do subshell)
RE='^\[([a-z]+)=([^]]*)\][[:space:]]*(.*)$'
parse() {
  local rest=$1
  J_GPU=""; J_AFTER=""; J_MEM=""; J_ERR=""
  while [[ $rest =~ $RE ]]; do
    case "${BASH_REMATCH[1]}" in
      gpu)   J_GPU=${BASH_REMATCH[2]} ;;
      after) J_AFTER="$J_AFTER${BASH_REMATCH[2]}"$'\n' ;;
      mem)   J_MEM=${BASH_REMATCH[2]} ;;
      *)     J_ERR="condição desconhecida [${BASH_REMATCH[1]}=...]" ;;
    esac
    rest=${BASH_REMATCH[3]}
  done
  J_CMD=$rest
  if [ -n "$J_MEM" ] && ! [[ $J_MEM =~ ^[0-9]+$ ]]; then J_ERR="[mem=$J_MEM] não é um número de GB"; fi
}

# Validação: condições conhecidas, [gpu=] de um worker existente
norm() { echo "$1" | tr ' ' '\n' | grep -v '^$' | tr '\n' ' '; }
BAD=0
n=0
while [ "$n" -lt "$N" ]; do
  n=$((n + 1))
  parse "$(sed -n "${n}p" "$STATE/jobs")"
  if [ -n "$J_ERR" ]; then echo "job $n: $J_ERR" >&2; BAD=1; fi
  if [ -n "$J_GPU" ]; then
    found=0
    for g in $GPUS; do [ "$g" = "$J_GPU" ] && found=1; done
    if [ "$found" -eq 0 ]; then echo "job $n: [gpu=$J_GPU] não é um worker de \"$(norm "$GPUS")\"" >&2; BAD=1; fi
  fi
done
[ "$BAD" -eq 0 ] || exit 2
echo "fila: $N jobs, workers: $GPUS  (estado em $STATE)"

# Memória livre (MiB) de uma GPU física, ou vazio se nvidia-smi falhar
free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | head -1 | tr -d ' '
}

# 0 = pode rodar agora; 1 = ainda não (motivo em WAIT); 2 = nunca neste worker
eligible() {
  local gpu=$1 f g m
  WAIT=""
  if [ -n "$J_GPU" ] && [ "$J_GPU" != "$gpu" ]; then return 2; fi
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    if [ ! -e "$f" ]; then WAIT="esperando $f"; return 1; fi
  done <<< "$J_AFTER"
  if [ -n "$J_MEM" ]; then
    for g in $(echo "$gpu" | tr ',' ' '); do
      m=$(free_mib "$g")
      if [ -z "$m" ]; then WAIT="nvidia-smi indisponível para [mem=]"; return 1; fi
      if [ "$m" -lt $((J_MEM * 1024)) ]; then WAIT="esperando ${J_MEM} GB livres na GPU $g"; return 1; fi
    done
  fi
  return 0
}

worker() {
  local gpu=$1 i n line log rc t0 pending r seen="|"
  while :; do
    pending=0; n=0; i=0
    while [ "$i" -lt "$N" ]; do
      i=$((i + 1))
      [ -d "$STATE/claimed/$i" ] && continue
      parse "$(sed -n "${i}p" "$STATE/jobs")"
      eligible "$gpu"; r=$?
      [ "$r" -eq 2 ] && continue
      if [ "$r" -eq 1 ]; then
        pending=1
        case "$seen" in
          *"|$i:$WAIT|"*) ;;
          *) echo "[gpu $gpu] job $i aguarda: $WAIT"; seen="$seen$i:$WAIT|" ;;
        esac
        continue
      fi
      if mkdir "$STATE/claimed/$i" 2>/dev/null; then n=$i; break; fi
    done
    if [ "$n" -eq 0 ]; then
      [ "$pending" -eq 0 ] && break         # nada mais que este worker possa rodar
      sleep "$POLL"
      continue
    fi
    line=$(sed -n "${n}p" "$STATE/jobs")
    parse "$line"
    log="$STATE/job${n}.log"
    t0=$(date +%s)
    echo "[gpu $gpu] início job $n: $J_CMD"
    ( export CUDA_VISIBLE_DEVICES="$gpu"; bash -c "$J_CMD" ) > "$log" 2>&1
    rc=$?
    printf '%s\t%s\t%s\t%s\t%s\n' "$n" "$rc" "$gpu" "$(( $(date +%s) - t0 ))" "$line" >> "$STATE/status"
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
