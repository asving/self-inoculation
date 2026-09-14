#!/usr/bin/env bash
# probes + restoration for runs_h seeds 1,2 on a lightly loaded GPU (sharing test: util<=40%, >=27GB free), else CPU
cd /data/users/asvin/self_inoculation
PY=/data/users/asvin/comp_icl/.venv/bin/python
G=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<54000 && $3<=40 {print $1; exit}')
if [ -n "$G" ]; then export CUDA_VISIBLE_DEVICES=$G; echo "sharing GPU $G (util $(nvidia-smi -i $G --query-gpu=utilization.gpu --format=csv,noheader))"; else export CUDA_VISIBLE_DEVICES=""; echo "no shareable GPU; CPU"; fi
export OMP_NUM_THREADS=4
H="--hmm1 0.7 0.03"
for s in 1 2; do
  $PY probe.py $H --ckpt runs_h/s$s/pretrain/model.pt --out runs_h/s$s/pretrain/probe.json --save_ref runs_h/s$s/pretrain/probe_ref.npz > logs/probe_h_pretrain_s$s.log 2>&1
  [ -n "$G" ] && [ $s = 1 ] && nvidia-smi -i $G --query-gpu=memory.used,utilization.gpu --format=csv,noheader
  for a in anchored unanchored; do
    $PY probe.py $H --ckpt runs_h/s$s/$a/model.pt --out runs_h/s$s/$a/probe.json --ref runs_h/s$s/pretrain/probe_ref.npz > logs/probe_h_${a}_s$s.log 2>&1
  done
done
$PY restore_calib.py --runs runs_h $H > logs/restore_calib_h.log 2>&1
echo "PROBES_H DONE $(date)"
