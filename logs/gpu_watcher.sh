#!/usr/bin/env bash
# Polls nvidia-smi; when a GPU is free (<200 MiB used AND 0% util), launches the given finetune arms on it
# (check-and-launch in the same breath), as windows of tmux session `selfinoc`, then verifies the footprint.
# Usage: bash logs/gpu_watcher.sh "1:anchored 1:unanchored 2:anchored 2:unanchored"
cd /data/users/asvin/self_inoculation
PY=/data/users/asvin/comp_icl/.venv/bin/python
ARMS="$1"
echo "watcher started $(date); arms: $ARMS"
while true; do
  G=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<200 && $3==0 {print $1; exit}')
  if [ -n "$G" ]; then
    echo "free GPU $G at $(date)"
    tmux has-session -t selfinoc 2>/dev/null || tmux new-session -d -s selfinoc -n shell -c "$PWD"
    for a in $ARMS; do
      s=${a%%:*}; arm=${a##*:}
      tmux new-window -t selfinoc -n ${arm}_s$s -c "$PWD" \
        "CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 $PY train.py --phase $arm --seed $s --init runs/s$s/pretrain/model.pt --out runs/s$s/$arm 2>&1 | tee logs/${arm}_s$s.log"
    done
    echo "launched on GPU $G at $(date)"
    sleep 60
    nvidia-smi -i $G --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    for a in $ARMS; do s=${a%%:*}; arm=${a##*:}; echo "== logs/${arm}_s$s.log"; tail -n 1 logs/${arm}_s$s.log; done
    exit 0
  fi
  sleep 60
done
