#!/usr/bin/env bash
# waits for a free GPU, then runs the history-heavy pipeline for seeds 1 and 2 on it (two tmux windows)
cd /data/users/asvin/self_inoculation
echo "watcher2 started $(date)"
while true; do
  G=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<200 && $3==0 {print $1; exit}')
  if [ -n "$G" ]; then
    echo "free GPU $G at $(date)"
    tmux has-session -t selfinoc 2>/dev/null || tmux new-session -d -s selfinoc -n shell -c "$PWD"
    for s in 1 2; do tmux new-window -t selfinoc -n h_s${s}_gpu -c "$PWD" "bash logs/pipeline_seed.sh $G $s 0.7 0.03 runs_h 10000 2>&1 | tee logs/pipeline_h_s$s.log"; done
    sleep 90; nvidia-smi -i $G --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    for s in 1 2; do echo "== s$s"; tail -n 1 logs/runs_h_pretrain_s$s.log 2>/dev/null; done
    exit 0
  fi
  sleep 60
done
