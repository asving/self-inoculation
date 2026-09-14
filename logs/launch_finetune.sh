#!/usr/bin/env bash
# Waits for the three pretrain checkpoints, re-checks GPU 1 is still only ours, then launches
# the 6 finetune arms (3 seeds x {anchored, unanchored}) as windows of tmux session `selfinoc`.
cd /data/users/asvin/self_inoculation
PY=/data/users/asvin/comp_icl/.venv/bin/python
G=1
until [ -f runs/s0/pretrain/model.pt ] && [ -f runs/s1/pretrain/model.pt ] && [ -f runs/s2/pretrain/model.pt ]; do sleep 15; done
echo "pretrain checkpoints present at $(date)"
MEM=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$MEM" -gt 20000 ]; then echo "GPU $G unexpectedly busy ($MEM MiB used); NOT launching"; exit 1; fi
for s in 0 1 2; do for arm in anchored unanchored; do
  tmux new-window -t selfinoc -n ${arm}_s$s -c "$PWD" \
    "CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 $PY train.py --phase $arm --seed $s --init runs/s$s/pretrain/model.pt --out runs/s$s/$arm 2>&1 | tee logs/${arm}_s$s.log"
done; done
echo "launched 6 finetune windows at $(date)"
sleep 90
nvidia-smi -i $G --query-gpu=memory.used,utilization.gpu --format=csv,noheader
for f in logs/anchored_s?.log logs/unanchored_s?.log; do echo "== $f"; tail -n 1 "$f"; done
