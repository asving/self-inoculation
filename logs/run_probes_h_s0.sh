#!/usr/bin/env bash
cd /data/users/asvin/self_inoculation
PY=/data/users/asvin/comp_icl/.venv/bin/python
G=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<54000 && $3<=40 {print $1; exit}')
if [ -n "$G" ]; then export CUDA_VISIBLE_DEVICES=$G; echo "sharing GPU $G"; else export CUDA_VISIBLE_DEVICES=""; echo "CPU"; fi
export OMP_NUM_THREADS=4; H="--hmm1 0.7 0.03"
$PY probe.py $H --ckpt runs_h/s0/pretrain/model.pt --out runs_h/s0/pretrain/probe.json --save_ref runs_h/s0/pretrain/probe_ref.npz > logs/probe_h_pretrain_s0.log 2>&1
for a in anchored unanchored; do $PY probe.py $H --ckpt runs_h/s0/$a/model.pt --out runs_h/s0/$a/probe.json --ref runs_h/s0/pretrain/probe_ref.npz > logs/probe_h_${a}_s0.log 2>&1; done
$PY restore_calib.py --runs runs_h $H > logs/restore_calib_h.log 2>&1
$PY analyze.py --runs runs_h --out figs_h > logs/analyze_h.log 2>&1
echo "PROBES_H_S0 DONE $(date)"
