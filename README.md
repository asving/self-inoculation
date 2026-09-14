# Self-inoculation: a toy model

Code, preregistration and results behind the post *Self-Inoculation* (asving.com/blog/self-inoculation/),
which asks whether a network that uses one shared circuit in two in-context-inferred situations, and is then
trained to change its behaviour in one of them, corrupts the shared circuit or builds a situation-gated switch.

- `PREREG.md` — the preregistered design, predictions and falsifiers, with dated amendments (including the
  adversarial-review round and the one prediction that failed).
- `RESULTS.md` — every number behind the post, per seed, for both parameterizations of the shared circuit.
- `blog.md` — the post's markdown source.
- `process.py` — the generative process (Mess3 HMMs, the non-ergodic case stream, exact Bayesian filters, the
  target swap). `model.py`, `train.py` — the 4-block GPT and the three training phases (pretrain, A-only
  finetune, mixed finetune). `probe.py`, `restore_calib.py`, `meandiff_test.py` — probes, encoder-image
  erasure, situation patch, weight restoration, calibration, the mean-difference test. `analyze.py`,
  `blog_figs*.py`, `oracle_sim.py`, `scan_params.py` — figures, the CPU oracle and the parameter scan.
  `site_build.py` — renders the post into the website template.
- `runs/`, `runs_h/` — per-checkpoint evaluation traces (`train.json`) and probe results (`probe.json`) for
  the shallow (`runs/`, HMM1 α=.9) and history-heavy (`runs_h/`, HMM1 α=.7, the version in the post)
  parameterizations; model checkpoints are not included (regenerate with `train.py`).
- `figs/`, `figs_h/`, `sim/`, `sim_h/` — figures and oracle simulations.
- `logs/` — training and probe logs, and the three adversarial reviews of the design and write-up by a
  different model (`codex_*.md`), which the amendments in PREREG.md and the wording in RESULTS.md respond to.

Environment: Python with numpy, torch, scipy, matplotlib (and `markdown` for `site_build.py`). A full run
(pretrain + both finetunes) takes minutes on one GPU or an hour or two on CPU (`TORCH_THREADS=4`).

```
python oracle_sim.py --hmm1 0.7 0.03 --out sim_h                      # CPU oracle: floors, sharpening, ideal predictors
python train.py --phase pretrain   --seed 0 --hmm1 0.7 0.03 --steps 10000 --out runs_h/s0/pretrain
python train.py --phase anchored   --seed 0 --hmm1 0.7 0.03 --init runs_h/s0/pretrain/model.pt --out runs_h/s0/anchored
python train.py --phase unanchored --seed 0 --hmm1 0.7 0.03 --init runs_h/s0/pretrain/model.pt --out runs_h/s0/unanchored
python probe.py --hmm1 0.7 0.03 --ckpt runs_h/s0/anchored/model.pt --out runs_h/s0/anchored/probe.json --ref runs_h/s0/pretrain/probe_ref.npz
python analyze.py --runs runs_h --out figs_h
```
