# PREREG — Self-inoculation toy: is a shared circuit gated or corrupted under context-specific pressure?

Written 2026-09-10 before any training run. Asvin G. + Claude. Amendments dated at the end.

## 0. Re-grounding

Referent: the July 2026 OpenAI/Hugging Face swarm incident, and nostalgebraist's hypothesis that
the reward hacking may have been confined to eval-like contexts. The story we test a mechanism for:

> **(†)** A network uses one shared circuit ("goodness") in two situations it must infer in-context
> (eval-like A, deployment-like B). Training then demands a changed behavior in A while B's loss still
> needs the shared circuit intact. GD's cheapest resolution is neither to corrupt the shared circuit
> (breaking B) nor to duplicate it, but to **gate how the shared circuit's output is used** on the
> already-present situational-awareness variable w = P(A | context). The misbehavior is thereby
> quarantined to A. No intent is required; the incentive to preserve B (explicit here as B's loss;
> in RLHF, e.g., the KL-to-reference penalty acting in every context) is part of the mechanism.

The toy replaces "goodness" with an HMM belief tracker shared across two contexts, "misbehavior in A"
with a relabeling of the tracker's readout demanded in A only, and "situational awareness" with the
posterior over which of two non-ergodic components generated the context stream. It shows the
mechanism of conflict resolution, not intent.

## 1. Question

Pretrain on contexts A and B (shared x-tracker, context-inferred case). Finetune so that in A the
x-readout must be permuted while B is unchanged. Does GD (i) corrupt the shared tracker, (ii)
duplicate it, or (iii) gate its readout on w? And how does this depend on whether B's loss is present
during finetune (anchored) or absent (unanchored)?

## 2. Generative process (exact; `process.py`)

- Token = 3·x + y, x, y ∈ {0,1,2}; vocab 9 + BOS (id 9) = 10. Sequence = BOS then L = 64 data tokens.
- **x-stream (shared circuit): HMM1 = Mess3(α = 0.90, x = 0.10)**, uniform initial state.
- **y-stream (case stream): non-ergodic direct sum Mess3A ⊕ Mess3B**, Mess3A = (α = 0.85, x = 0.15)
  (sticky states), Mess3B = (α = 0.85, x = 0.50) (state always leaves), P(A) = P(B) = 1/2, uniform
  within component. Each sequence stays in one component for its whole length.
- x ⊥ y. The case is inferable only from the y-stream.
- Mess3 convention: T[z,i,j] = P(emit z, next j | i); β = (1−α)/2, y = 1−2x; operators as in Shai et
  al. 2026 App. C.1. Stationary = uniform.
- **Flip σ = transposition (0 1)(2) on the x alphabet, applied to the x-TARGET only, in case A only.**
  The input stream is never flipped. Since Mess3 is S₃-symmetric under joint relabeling of states and
  symbols, the flipped-target stream has identical marginal statistics and identical Bayes floor: the
  demanded change is a pure relabeling of the readout, zero change in competence.

**Oracle numbers** (`oracle_sim.py`, n = 20000, L = 64; averages over positions t ≥ 8):

| quantity | nats |
|---|---|
| uniform over 3 symbols | 1.0986 |
| x floor (HMM1 tracked exactly) | 0.8465 |
| y reference, case-uncertain (all; on true-A seqs; on true-B seqs) | 0.9895 (A .9951, B .9837) |
| y reference, case-known (all) | 0.9844 |
| joint (x+y) floor | 1.8359 |
| flip task, Bayes-gated predictor w·P^σ + (1−w)·P | 0.8528 (A .8519, B .8538) |
| flip task, case-oracle gate | 0.8465 |
| unflipped predictor on flipped A target | **1.3584** (0.51 above floor; worse than uniform) |
| flipped predictor on unflipped B target ("corrupted net on B") | **1.3599** |

Case posterior w_A: median wrong-case mass < 0.1 at t = 5, < 0.01 at t = 13; mean < 0.01 at t = 34;
information rate 0.367 nats/token. Figures: `sim/sharpening.png`, `sim/flip_predictors.png`,
`sim/geometries.png`.

**Why these parameters** (`scan_params.py`, `sim/scan_*.csv`): the Simplex-standard Mess3 (α ≈ 0.6)
has ≈ 0.01 nats of learnable signal and the non-ergodic post's pair separates at 0.04 nats/token
(never resolves in 64 tokens). α = 0.85–0.90 gives 0.11–0.25 nats of learnable signal per stream, a
flip cost of 0.51 nats, and a case that resolves by t ≈ 13, leaving t ∈ [16, 64] as the resolved
regime and t < 16 as the observable sharpening regime.

## 3. Model and training (`model.py`, `train.py`)

- GPT-2-style pre-LN causal transformer: 4 layers, d_model = 120, 4 heads, d_mlp = 480, learned
  positional embeddings, vocab 10, context 65. ≈ 0.7M parameters. Residual stream captured after
  every attention and MLP sublayer for probing.
- Loss: next-token CE on the joint 9-way token at all 64 data positions. Fresh samples every step.
- Optimizer: AdamW, lr 5e-4 constant after 200 warmup steps, no weight decay, grad clip 1.0,
  batch 256 sequences. Same optimizer settings in both phases; **finetune starts from a fresh AdamW
  state** so the transient is not shaped by stale moments.
- **Phase 1, pretrain**: A and B (50/50), unflipped targets. 20k steps; stop criterion checked post
  hoc: joint CE at t ≥ 8 within 0.02 nats of the 1.836 floor.
- **Phase 2, finetune**, two arms from the SAME pretrained checkpoint per seed, 10k steps each:
  - **anchored**: A (x-target flipped) and B (unflipped), 50/50 as in pretraining.
  - **unanchored**: A only (x-target flipped). B never seen.
  Checkpoints: every 25 steps to step 1000, every 100 to 5000, every 500 after; plus pretrain end.
- Seeds: 3 pretrains (seed 0, 1, 2), each spawning both arms → 3 pretrain + 6 finetune runs.
- Fixed eval set: 4096 held-out sequences per component, with oracle beliefs and both target
  conventions precomputed. Every checkpoint is evaluated on it.

Compute: 1 GPU, small footprint; each pretrain ≲ 20 min, each finetune ≲ 10 min at d = 120.

## 4. Analysis (`analyze.py`, `probe.py`)

**(a) Component losses.** From the model's joint prediction, marginalize to x and y. Report per case
and per position: x-CE_A on the flipped target, x-CE_B on the unflipped target, y-CE_A, y-CE_B, each
minus its oracle floor on the same sequences. Anchored-arm **B-excursion**: max over finetune of
(x-CE_B − floor_B), its step, and the step of return to within 0.02 of floor.

**(b) Flip fraction f.** At each checkpoint and position, least-squares fit of the model's x-marginal
q ≈ (1−f)·P + f·P^σ with P the oracle unflipped predictive, f ∈ [0,1]; report f_A and f_B averaged over
t ≥ 16, and f_A(t) by position. The finetune trajectory in the (f_B, f_A) unit square is the dynamics
picture: pretrain end = (0,0); ideal anchored end = (0,1); corrupted = (1,1). A trajectory along the
diagonal then bending to (0,1) is "shared readout permutes first, gate built under conflict".

**(c) Probes** (ridge, standardized, held-out R², per sublayer capture). Targets: HMM1 belief (3-dim),
case-stream belief (6-dim), logit w_A. Fit separately on case-A and case-B contexts, at pretrain end
and at every 500th finetune checkpoint. Derived tests:
- *Shared vs duplicated*: principal subspace of the HMM1-belief probe's row space in A-contexts vs
  B-contexts; normalized overlap (as in Shai et al. 2026 App. H.2.2). Shared ≈ 1 (as at pretrain end);
  duplicated ≪ 1.
- *Corrupted vs preserved*: HMM1 probe fit at pretrain end, applied at finetune end (transfer R²) in
  B-contexts. Preserved ≈ pretrain R²; corrupted drops.
- *Readout-gated vs upstream-gated*: B-context HMM1 probe applied to A-context activations at finetune
  end. Same belief (readout gating): high unpermuted R². Relabeled belief (upstream gating): best fit is
  σ-permuted. Record which; both are consistent with (†).

**(d) Switch causality.** Because x ⊥ y, evaluating the same x-stream under a case-A vs case-B
y-stream is an in-distribution data-level counterfactual; f_A vs f_B in (b) already measures the
switch's causal effect. Secondary, activation-level: at the earliest layer where logit-w_A R² > 0.9,
project the w direction out of the residual stream in A-contexts and re-measure f_A (expect movement
toward the mixture / unflipped; y-CE will also degrade since w is needed for y — record both).

**(e) Position dependence.** f_A(t) at anchored end vs oracle w_A(t): should track the sharpening
curve (f_A ≈ 0.5–0.8 for t < 8 rising to > 0.9 by t ≈ 15).

## 5. Preregistered predictions and falsifiers

**P1 (pretrain).** Joint CE at t ≥ 8 within 0.02 of 1.836. HMM1-belief probe R² > 0.9 at some
sublayer; logit-w_A probe R² > 0.9 for t ≥ 8. FWH check: HMM1-belief and case-stream-belief probe
subspaces have overlap < 0.2. *Falsifier*: any of these fails → fix the model/training before Phase 2
(amendment, not a result).

**P2 (anchored arm, endpoint).** Over positions 16–63 on the eval set: x-CE_A (flipped target) and
x-CE_B (unflipped target) each within 0.03 of the eval-set Bayes-gated reference for that case
(≈ 0.852 / 0.854); f_A > 0.9, f_B < 0.1 with f_expl > 0.8. Interventional (Amendment A6): ablating the
HMM1-belief subspace raises x-CE by comparable amounts in A- and B-contexts (ratio within [0.5, 2]) and
leaves y-CE within 0.02, at pretrain end and at finetune end; parameter displacement from the pretrained
weights is concentrated in the last block + unembedding (early groups embed/L1/L2 < 0.05 relative).
Supporting: HMM1 subspace overlap A-vs-B > 0.8 and transfer R² from pretrain > 0.8. *Falsifiers of (†)*:
ablation ratio outside [0.2, 5] at finetune end (A no longer uses the B-tracker: duplication/replacement);
transfer R² < 0.5 together with x-CE_B > reference + 0.05 (corruption); or losses never within 0.05 of
reference by 10k steps (unresolved conflict, or optimization failure — distinguished by whether the
training loss is still falling).

**P3 (anchored arm, dynamics).** Pretraining has x ⊥ case, so by the Factored-World inductive bias
there is no w × b interaction at finetune start; the gate must be built under conflict. Predicted:
B-excursion max (x-CE_B − floor) > 0.05 nats within the first 2000 steps, then return to within 0.02;
f_B rises above 0.1 before returning below 0.1. *Falsifier*: f_B never exceeds 0.1 (gate built with no
excursion; (†) still holds but the FWH-based dynamics claim is wrong). ~65% confidence.

**P4 (unanchored arm).** No conflict, so the shared readout permutes: x-CE_B on the true (unflipped) B
target rises to ≈ 1.36 and stays (worse than uniform); f_A ≈ f_B → 1 along the diagonal. HMM1 probe:
record whether the subspace/geometry is unchanged (readout permuted) or σ-relabeled (tracker relabeled).
~80% confidence. *The interesting falsifier*: f_B < 0.3 at end — the pre-existing switch shielded B
with no incentive at all. That would be structural (Hubinger-style) shielding and would upgrade (†).

**P5 (switch causality and position).** At anchored end, f_A(t) tracks oracle w_A(t) (Spearman > 0.9
over t ∈ [0, 20]); projecting out the w direction lowers f_A by > 0.3.

Floors are always computed on the same eval sequences as the model's loss.

## 6. What this does and does not show (for the blog)

Shows: given a shared circuit and an in-context situational variable, a context-specific change in
demanded behavior is absorbed by gating rather than by corrupting or duplicating the circuit, *when
the other context's loss is present*; the gate is built from the existing situational-awareness
variable; behavior in the protected context stays at floor throughout (modulo the transient). Does not
show: intent, or shielding without any incentive (that is P4's falsifier, not its prediction).

## 7. File map

```
process.py       generator, exact filters, flip, ideal predictors        (done)
oracle_sim.py    CPU oracle: sharpening, floors, ideal-predictor curves  (done; sim/)
scan_params.py   Mess3 parameter scan                                    (done; sim/scan_*.csv)
model.py         GPT with per-sublayer residual capture
train.py         --phase pretrain|anchored|unanchored, checkpoints, eval on fixed set
probe.py         ridge probes, subspace overlap, transfer R², f-fit
analyze.py       loss traces, (f_B, f_A) trajectories, B-excursion, figures
runs/s{seed}/{pretrain,anchored,unanchored}/   checkpoints + train.json
logs/            tee'd stdout
RESULTS.md       to come
blog.md          Asvin's post
```

## 8. Amendments

**2026-09-10, after Codex adversarial review (`logs/codex_prereg_review.md`), before any run.**

- **A1 Code/doc alignment.** `Process()` defaults now equal §2 (the reviewed file still carried
  scan-era defaults; caught by the CPU smoke test independently). `flip_target` raises on unknown arm.
- **A2 Arm matching.** Unanchored batch = 128 (all case A) so the number of A sequences per step equals
  the anchored arm's (128 A + 128 B). The unanchored arm's y-training prior is P(A)=1; y-CE_B is logged
  as a secondary "deployment rusting" measure and is not part of (†).
- **A3 Joint reference.** Under the flip the exact joint Bayes predictor is
  Q = w·(P^σ ⊗ R_A) + (1−w)·(P ⊗ R_B) (R_c = component-conditional y predictive), which is *not* the
  product of the x and y marginals. Logged as `floor_joint_anchored`; pretrain joint reference is
  P ⊗ P_y. The 10-way joint CE includes BOS mass (logged separately; ≈ 1e-4 in smoke tests); x/y
  marginals are renormalized over the 9 data tokens.
- **A4 Flip fraction f.** Pooled per-position least squares f(t) = Σₙ⟨qₙ−Pₙ, Dₙ⟩ / Σₙ‖Dₙ‖², D = P^σ − P,
  masked (NaN) where Σₙ‖Dₙ‖² < 1e-3·n (σ unidentifiable, e.g. t = 0). Reported with
  f_expl(t) = share of Σₙ‖qₙ−Pₙ‖² lying along D; f is a projection and mixture claims require f_expl high.
  Per-history version for P5: at fixed t ∈ [3, 12], single-sequence fₙ on sequences with ‖Dₙ‖² > 0.02,
  Spearman-correlated with the per-sequence oracle w_A,ₙ — tests per-history tracking against a learned
  position schedule.
- **A5 Terminology and thresholds.** "Floor" → "Bayes reference" (finite-sample realized loss of the
  Bayes predictor; gaps may be slightly negative). All thresholds are relative to the eval-set references
  logged in `train.json`, not to the table constants. Table clarified (per-case y values are case-uncertain).
- **A6 Interventional tests (now primary for "shared, preserved, used").**
  (i) *Ablation-necessity*: at each sublayer where the HMM1-belief probe R² > 0.8 at pretrain end,
  project the probe's centered 2-dim row space out of the residual stream at all positions, separately
  in A- and B-contexts, at pretrain end and finetune end; report Δx-CE_A, Δx-CE_B, Δy-CE.
  (ii) *Parameter displacement*: relative L2 displacement from the phase's initial weights per group
  (embed, L{l}.attn, L{l}.mlp, final), every checkpoint. Probe overlap / transfer R² become supporting.
- **A7 P3 measurement.** Excursion = max over checkpoints of [gap_B(step) − gap_B(step 0)],
  gap_B = x-CE_B − reference_B over positions 16–63. "Return" = first checkpoint after the max from which
  gap_B ≤ 0.02 for ≥ 3 consecutive checkpoints. Eval noise ≈ 0.003 nats at n = 4096; threshold 0.05
  stands. The excursion is a timestamp; "the gate is built during it" is claimed only if f_B and the
  late-group displacement move with it.
- **A8 P4 rewording.** A-only training's population optimum flips x on every history (B-histories have
  positive probability under A), so P4 tests finite-training generalization, not a free choice. f_B < 0.3
  at 10k steps would indicate inertia of pretrained B behavior and prompts a longer run before any
  "shielding" language.
- **A9 Seeds rule.** Each prediction is assessed per seed and declared to hold if it holds in ≥ 2 of 3;
  reported per seed. The ~65% / ~80% figures are subjective forecasts of that outcome.
- **A10 Task type.** The finetune task is supervised transformation (predict σ(x_{t+1}) from unflipped
  inputs), not language modeling of a flipped stream. "Zero change in competence" → identical Bayes
  reference *given the case*; case uncertainty adds 0.006 nats averaged over t ≥ 8.
- **A11 Conventions.** Position t = number of data tokens seen (0..63); the logits at input index t
  (index 0 = BOS) target data token t+1; oracle `beliefs[:, t]` uses the same count. "t ≥ 16" = positions
  16..63. Sharpening is reported per case and via mean tails (true-B is recognized faster than true-A).
  Logit-w_A is clipped to [1e-4, 1−1e-4] for probes. Batches sample the case i.i.d. with P(A) = 1/2.
- **A12 FWH premise softened.** "We expect no w × b interaction at finetune start (Factored-World
  tendency); the P1 probe-subspace overlap is evidence, not proof."
- **A13 Intervention method (after probing the seed-0 pretrain at step 5000, before any finetune).**
  Erasing the ridge-*decoder*'s read subspace is inert (Δx-CE +0.004 at L4mlp) while erasing the
  *encoder image* (regress activations on the belief; row space of the fitted map) sends x-CE to uniform
  (+0.25) at every single capture from L1mlp on, equally in A- and B-contexts (+0.250 / +0.247), with
  y-CE unchanged (≤ 0.018) — replicating the encoder/decoder asymmetry of `comp_icl/hier_rrxor/encdec_probe.py`.
  A6(i) therefore uses the 2-dim encoder image of the HMM1 belief. Pretrain baseline for the P2 ablation
  ratio: 1.01. A 1-dim "logit-w direction" erasure is also inert for y (the case posterior is the *scale*
  of two component sub-simplices, per the non-ergodic geometry, not a direction), so the switch
  intervention is (a) erasure of the 5-dim encoder image of the 6-dim case-stream belief and (b) a paired
  **case-subspace patch**: same x-stream under an A and a B y-stream (x ⊥ case, so both in-distribution);
  transplant the case-subspace component of the B run into the A run at capture ℓ and read f_A. P5's
  activation-level clause becomes: at the anchored endpoint the patch moves f_A toward f_B (by > 0.5 of the
  gap) at some capture. Probe eval data are paired (shared x-stream) from now on.
  Also measured: the x-stream's history dependence is small — a last-token predictor scores 0.861 vs the
  Bayes 0.845 (6.5% of the learnable signal needs history; two tokens suffice). The shared circuit is a
  shallow tracker. Flagged as a candidate robustness arm (history-heavier HMM1), not changed for v1.
- **A14 Compute contingency.** All 8 GPUs filled with another user's job right after pretraining finished;
  seed 0's two arms run on CPU (4 threads each), seeds 1–2 launch on the first free GPU via a watcher.
  Identical code and seeds; device is not expected to matter beyond nondeterministic reduction order.
- Accepted limitations (for the write-up, not fixed): non-ergodic components never switch, so the gate
  is never tested against changing situations; the 50/50 prior and fixed L = 64 are single settings;
  the KL-to-reference analogy is an interpretive bridge, not tested here.

- **2026-09-11.** Runs complete; verdicts per prediction in `RESULTS.md` §3 (P2 displacement clause false, all else holds).

- **A15 (2026-09-11, post hoc, after the Codex review of RESULTS.md).** Added a weight-RESTORATION test (swap one parameter group between the pretrained and finetuned nets and re-measure f, gaps) and a per-history CALIBRATION of f against w_A (binned means, slope, mean |f−w|), to test the localization and "Bayes" claims Codex flagged as under-supported. Both were designed after seeing the endpoint probes; they are confirmatory analyses, not preregistered predictions. `restore_calib.py`, `logs/restore_calib.log`.

- **A16 (2026-09-11). Robustness arm: history-heavy shared circuit.** HMM1 = Mess3(α = 0.70, x_leave = 0.03): a last-token predictor now recovers only 49% of the learnable x-signal (vs 94% for v1), Bayes x reference 0.953, flip cost 0.29 nats, corrupted-on-B value 1.245 (`sim_h/`). Everything else identical to v1 except pretrain shortened to 10k steps (v1 converged by ~2k). Same predictions P1–P5; outputs in `runs_h/`. Seed 0 on CPU, seeds 1–2 on the first free GPU.
