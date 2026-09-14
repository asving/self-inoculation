#!/usr/bin/env bash
# usage: pipeline_seed.sh <cpu|GPU_INDEX> <seed> <alpha> <xleave> <outroot> <pretrain_steps> [threads]
# pretrain -> anchored -> unanchored, sequentially, with tee'd logs in logs/<outroot>_*.log
cd /data/users/asvin/self_inoculation
PY=/data/users/asvin/comp_icl/.venv/bin/python
DEV=$1; S=$2; A=$3; X=$4; R=$5; NPRE=$6; TH=${7:-8}
if [ "$DEV" = cpu ]; then export CUDA_VISIBLE_DEVICES="" TORCH_THREADS=$TH OMP_NUM_THREADS=$TH; else export CUDA_VISIBLE_DEVICES=$DEV OMP_NUM_THREADS=8; fi
$PY train.py --phase pretrain   --seed $S --hmm1 $A $X --steps $NPRE --out $R/s$S/pretrain 2>&1 | tee logs/${R}_pretrain_s$S.log
$PY train.py --phase anchored   --seed $S --hmm1 $A $X --init $R/s$S/pretrain/model.pt --out $R/s$S/anchored   2>&1 | tee logs/${R}_anchored_s$S.log
$PY train.py --phase unanchored --seed $S --hmm1 $A $X --init $R/s$S/pretrain/model.pt --out $R/s$S/unanchored 2>&1 | tee logs/${R}_unanchored_s$S.log
echo "PIPELINE DONE seed $S $(date)"
