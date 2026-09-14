#!/usr/bin/env bash
# CPU probes, 2 at a time, 4 threads each. Pretrain refs for s1,s2 first, then all 6 endpoints with --ref.
cd /data/users/asvin/self_inoculation
PY=/data/users/asvin/comp_icl/.venv/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
run() { $PY probe.py "$@" ; }
run --ckpt runs/s1/pretrain/model.pt --out runs/s1/pretrain/probe.json --save_ref runs/s1/pretrain/probe_ref.npz > logs/probe_pretrain_s1.log 2>&1 &
run --ckpt runs/s2/pretrain/model.pt --out runs/s2/pretrain/probe.json --save_ref runs/s2/pretrain/probe_ref.npz > logs/probe_pretrain_s2.log 2>&1 &
wait
for s in 0 1 2; do
  run --ckpt runs/s$s/anchored/model.pt   --out runs/s$s/anchored/probe.json   --ref runs/s$s/pretrain/probe_ref.npz > logs/probe_anchored_s$s.log 2>&1 &
  run --ckpt runs/s$s/unanchored/model.pt --out runs/s$s/unanchored/probe.json --ref runs/s$s/pretrain/probe_ref.npz > logs/probe_unanchored_s$s.log 2>&1 &
  wait
done
echo "ALL PROBES DONE $(date)"
