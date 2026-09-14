Reading additional input from stdin...
OpenAI Codex v0.153.4
--------
workdir: /data/users/asvin/self_inoculation
model: gpt-6-astra
provider: openai
approval: never
sandbox: read-only
reasoning effort: none
reasoning summaries: none
session id: 01a08e25-e8ad-74a3-82dc-136ab54d5f84
--------
user
RESULTS.md in this directory was revised in response to your earlier adversarial review, which is in logs/codex_results_review.md (read both; read-only; you may also check logs/restore_calib.log for the new numbers). For each of your earlier points (A)1–6, say in one or two sentences whether the revision now supports the claim as stated, still over-claims (quote the phrase), or has introduced a new inconsistency. Then list any NEW over-claims or errors in the revised text, especially in §0, §2.2, §2.3, §2.4 and §3. Numbered, blunt, no rewriting, no praise.
codex
I’ll read both documents and check the calibration log, then assess each earlier point and any new over-claims or inconsistencies.
exec
/usr/bin/bash -lc 'cat RESULTS.md logs/codex_results_review.md logs/restore_calib.log' in /data/users/asvin/self_inoculation
 succeeded in 0ms:
# RESULTS — Self-inoculation toy, v1 (runs 2026-09-10/11; revised after Codex review 2026-09-11)

Companion to `PREREG.md` (design, Codex design review, amendments A1–A15). Three seeds; numbers are given
per seed as s0/s1/s2, or marked "all" when the three agree to the stated precision. "Gap" = the model's
cross-entropy in nats minus the realized loss of the exact Bayes predictor on the same evaluation
sequences ("Bayes reference"; a finite-sample quantity, so gaps can be slightly negative). Figures in `figs/`.

## 0. Verdict

Under conflicting demands on a shared circuit, the network first changes behaviour in *both* contexts
through the shared pathway, and only as the protected context becomes measurably damaged does it build a
gate: from block 3 on, the shared belief's coordinates are σ-relabeled when the context reads as case A,
and the unchanged unembedding then emits the flipped prediction. Weight-restoration shows the finetune's
changes to the embedding and unembedding are functionally irrelevant and those to blocks 1–2 minor;
grafting only the finetuned blocks 3–4 onto the pretrained net reconstructs 86–88% of the gate. Without
the protecting loss the same network corrupts fully, to the value predicted in advance. The gate's output
is calibrated to the situational posterior per history (mean |f − w| ≈ 0.03). One preregistered
prediction (weight change concentrated late) is false: the change is distributed, and localization
required interventions.

## 1. Setup, definitions, what was run

**Process.** Each token encodes a pair (x, y), token id = 3x + y, x, y ∈ {0, 1, 2}. x is emitted by HMM1,
a Mess3 process (3 hidden states, 3 symbols; parameter α = 0.9 sets how sharply a state prefers "its"
symbol, parameter x_leave = 0.1 the probability of leaving the state). y is emitted, for the whole sequence,
by one of two other Mess3 processes: Mess3A (α .85, x_leave .15, sticky) or Mess3B (α .85, x_leave .5,
always leaves), each with prior 1/2; the **case** (A or B) is never given as input and must be inferred
from the y-stream. The exact posterior w_A(t) = P(case A | y_1..y_t) has median wrong-case mass < 1% by
t = 13. x is independent of the case. Sequences: BOS then 64 data tokens; **position t** = number of data
tokens seen; the model's logits at position t predict data token t+1; "positions 16–63" = the window where
the case is essentially resolved.

**Training.** 4-layer GPT (d = 120, 4 heads, learned positions), AdamW 5e-4, batch 256 sequences,
fresh samples each step. *Pretrain* 20k steps on both cases, plain next-token targets. *Finetune* 10k
steps from the same checkpoint (fresh optimizer state), checkpoints every 25 steps to 1000: in case A the
x-*target* is relabeled by the swap σ = (0↔1); inputs are never changed; case-B targets are unchanged.
**Anchored** arm: A-flipped and B, 50/50. **Unanchored** arm: A-flipped only, batch 128 so the number of
A sequences per step matches. Because Mess3 is symmetric under jointly relabeling states and symbols, the
Bayes reference *given the case* is identical before and after the flip; case uncertainty adds 0.006 nats
(the "Bayes-gated" reference w·P^σ + (1−w)·P).

**Flip fraction f.** P = the Bayes x-prediction with unflipped labels, P^σ = the same with swapped labels.
At each position, pooled over sequences n: f = Σₙ⟨qₙ − Pₙ, Dₙ⟩ / Σₙ‖Dₙ‖², D = P^σ − P, where q is the
model's x-marginal (renormalized over the 9 data tokens). f = 0 reproduces P, f = 1 reproduces P^σ; not
clipped (values slightly outside [0, 1] are noise). Undefined where P₀ = P₁ (t = 0). f_expl = the share
of Σₙ‖qₙ − Pₙ‖² lying along D (fit quality; degenerate when f ≈ 0). f_A, f_B are measured on case-A and
case-B contexts, positions 16–63. Single-history fₙ = ⟨qₙ − Pₙ, Dₙ⟩/‖Dₙ‖² on histories with ‖Dₙ‖² > 0.02.

**Interventions.** A *capture* is the residual stream after one sublayer (emb, L1attn, L1mlp, …, L4mlp).
*Encoder image*: regress activations on the true belief (H ≈ B·M + c); the row space of M is the
subspace the belief writes into (validated as the causal one in `comp_icl/hier_rrxor/encdec_probe.py`;
the ridge *decoder*'s read directions are not). *Erasure* projects that subspace out (HMM1 belief: 2-dim;
6-state case belief: 5-dim) at one capture or at all captures simultaneously. *Case patch*: paired runs
with the same x-stream under an A- and a B-y-stream (both in-distribution since x ⊥ case); the
case-subspace component of the B run replaces that of the A run at capture ℓ, and f_A is re-measured.
*Transfer*: the ridge probe for the HMM1 belief fitted at the pretrain endpoint is applied unchanged to
finetune-endpoint activations and scored (held-out R²) against the true belief and against the
σ-relabeled belief (σ acts on hidden states as on symbols under the Mess3 symmetry). *Restoration*: one
parameter group of the finetuned net is replaced by its pretrained weights (or vice versa) and f, gaps
re-measured. Restoration and per-history calibration were added post hoc (amendment A15).

## 2. Results

### 2.1 Endpoints (positions 16–63)

| | anchored s0/s1/s2 | unanchored s0/s1/s2 |
|---|---|---|
| x-CE_A on the flipped task − Bayes-gated ref | −.0003/+.0010/+.0003 | −.0025 all |
| x-CE_B on the unflipped task − Bayes-gated ref | +.0017/−.0001/+.0008 | +.5125/+.5072/+.5106 |
| x-CE_B absolute | 0.855 all | **1.360 all** (prereg "corrupted" value 1.3599) |
| f_A (f_expl) | .990/.975/.987 (1.00/.99/.99) | 1.001/.994/.999 (1.00) |
| f_B (f_expl) | .016/.020/.014 (.03/.07/.03, degenerate) | 1.002/.995/.998 (1.00) |
| y-CE_A − ref, y-CE_B − ref | within ±.0012, all | A within ±.002; **B +.343/+.344/+.342** |

Anchored: both contexts at reference, flip confined to A. Unanchored: fully corrupted in B, and the
y-model's performance on B degrades by 0.34 nats (nothing in A-only training maintains it).

### 2.2 Dynamics (`figs/f_square.png`, `figs/zoom_first300.png`)

For the first ~50 steps the two arms move nearly together up the diagonal (step 50: f_A .14/.16/.14 vs
.15/.18/.15, f_B .12/.13/.12 vs .13/.14/.13, anchored vs unanchored): the A-loss is reduced through the
pathway both contexts share. By step 75 they have begun to differ (B excess .055/.047/.057 vs
.089/.109/.086; f_B .32/.29/.33 vs .42/.45/.41). Between checkpoints 75 and 125 they separate: anchored
reaches (f_B, f_A) = (.02, .96) with B within .01 of reference from step 125 onward (checked at every
checkpoint); unanchored reaches (.99, 1.00) by step 125 and stays. Ordering and timing agree across seeds.
*Interpretation* (not directly measured): the B-loss contributes little gradient until the shared change has
damaged B by a few hundredths of a nat, then dominates; the gate appears within ~50 steps of that point.

### 2.3 The gate's output is calibrated to the case posterior (`figs/position_f.png`, `figs/restoration_calibration.png`)

Position curve: f_A(t) vs the oracle mean w_A(t) over t = 1…20 has Spearman 1.000 and mean absolute
difference .005/.019/.006; f_B(t) declines correspondingly. Per history, pooling A- and B-contexts and
t = 3…12 (n ≈ 30k): regressing fₙ on w_A,ₙ gives slope .976/.964/1.001, intercept +.024/+.009/+.009,
mean |fₙ − w_A,ₙ| = .033/.028/.026; binned means agree within .04 in every posterior bin; Spearman by
position .90 (t = 3) → .98 (t = 8) → .95 (t = 12). Pretrain baseline for the per-history correlation: ≈ 0.
The model flips by its current, per-history, calibrated belief that the context is A; a learned position
schedule is excluded.

### 2.4 Where the gate is (`figs/gate_localization.png`, `figs/restoration_calibration.png`)

- **Restoration (causal, per parameter group).** Putting pretrained weights back into the anchored net:
  embed → f_A .989/.975/.987 (no change); final (ln_f + unembedding) → .961/.911/.997 with gaps ≤ .010;
  blocks 1–2 → .926/.938/.875, f_B .120/.033/.066; block 4 → .962/.920/.945; **block 3 → .439/.419/.510**
  (gap_A +.15/+.16/+.12); **blocks 3–4 → .206/.185/.276** (gap_A +.29/+.31/+.24; f_B .15/.12/.14, i.e. the
  flip disappears rather than becoming unconditional). Grafting finetuned groups onto the pretrained net:
  final only → f_A .07/.08/.06; blocks 1–2 only → .11/.08/.20; block 3 only → .79/.79/.71;
  **blocks 3–4 only → .884/.872/.861** with gap_A +.010/+.013/+.019 and f_B .08/.01/.03. The gate is
  carried by blocks 3–4 (block 3 dominant), which are necessary and nearly sufficient; changes to the
  embedding and unembedding are irrelevant; changes to blocks 1–2 contribute a small leak-suppression.
- **Shared tracker, used in both contexts.** Erasing the HMM1 encoder image at all captures raises x-CE on
  the context's own task by +.225/+.257/+.232 in A and +.213/+.239/+.214 in B (pretrain +.250/+.247,
  A/B; uniform is +.25), y unchanged (≤ .004). Ratio A/B 1.06–1.08. Decoder-subspace erasure inert
  (+.001–.005), as at pretrain.
- **Relabeled coordinates from block 3.** The pretrain HMM1 probe applied at the anchored endpoint reads
  the belief in original coordinates at L1mlp in both contexts (R² .94/.96/.87), degrading by L2mlp
  (A .79/.94/.62, B .83/.96/.70). From L3mlp it reads B-contexts unpermuted (.89/.91/.86) and A-contexts
  only after σ-relabeling (.83/.57/.84; unpermuted −.72/+.09/−1.16); at L4mlp σ: .79/.68/.75. Freshly
  fitted probes still read the belief at R² ≥ .99 at every transformer capture in both contexts (embedding
  .944, unchanged), and the A-vs-B decoder subspaces, aligned at L1–L2 (overlap .94–.97), diverge at
  L3–L4 (.49–.76; pretrain .82), consistent with the relabeling.
- **The gate reads the case subspace before block 3.** Case patch A←B at L1mlp: f_A .99 → .03/.03/.04
  (B←A: .01 → .97); at L2mlp → .07/.06/.09; at L3mlp → .93/.87/.92; at L4mlp ≤ .03 change. Erasing the
  case subspace at all captures collapses both contexts to the same f (.41/.50/.34). In the unanchored
  arm the same patch and erasures change f by ≤ .02: no gate. The case posterior's linear readability
  rises through blocks 1–3 (logit-w R² .18 → .52 → .94 at L1mlp/L2mlp/L3mlp, .87 already at L3attn);
  the conditional relabeling sits where it becomes available.
- **What is not distinguished.** An in-place case-conditional permutation of the belief in block 3 and a
  conditional branch in blocks 3–4 that writes an equivalent correction into the same coordinates
  produce the same transfer, patch and restoration signatures. Finer circuit work would be needed.
- **Unanchored arm.** The pretrain probe fails from L2 on in both labelings and both contexts (R² < 0);
  the x pathway was reorganized uniformly rather than conditionally, with the largest early displacement
  in block 1 (.15–.18 at step 125).

### 2.5 Representation side-effects

Linear readability of the case posterior *increased* under anchored finetune (logit-w R² at L4mlp
.86 → .956/.970/.933; case belief .95 → .974/.979/.969; in s1 readable one block earlier) and *decreased*
under unanchored finetune (logit-w → .41/.53/.45; case belief → .68/.69/.67). Consistent with the
variable gaining a downstream consumer in one arm and losing one (the B branch of y) in the other; not
a measure of discrimination accuracy, and the arms differ in more than consumer count.

### 2.6 Weight displacement (the false prediction)

Relative L2 displacement from pretrained weights at step 125 (gate just formed), s0: embed .006,
L1.attn .09, L1.mlp .14, L2.attn .11, L2.mlp .13, L3.attn .14, L3.mlp .17, L4.attn .09, L4.mlp .14,
final .03 (s1, s2 within ±.03). Distributed across all blocks; the preregistered "concentrated in the last
block + unembedding, early groups < .05" is **false**. The displacement magnitude in blocks 1–2 is real but
restoration shows it is functionally minor; only interventions localized the gate.

## 3. Scorecard (PREREG §5 with amendments; a clause holds if it holds in ≥ 2 of 3 seeds)

| | outcome |
|---|---|
| P1 pretrain | ✓ joint CE within .0006 of reference (all); HMM1 R² .99; factor-subspace overlap .02–.16 (< .2). Logit-w R² at L4mlp, t ≥ 8: **.896 vs threshold .9 (borderline; s0 only)**. |
| P2 anchored endpoint | ✓ losses; ✓ f_A > .9 with f_expl > .8; ✓ f_B < .1 but **f_expl_B .03–.07 fails the literal > .8 clause** (statistic degenerate at f ≈ 0; not passed as written); ✓ erasure ratio in [.5, 2]; ✓ transfer R² > .8 in B-contexts at L3mlp (.89/.91/.86), L4mlp .82/.94/.82; A-vs-B overlap > .8 **holds at L1–L2, fails at L3–L4** (.49–.76); **✗ displacement clause**. |
| P3 anchored dynamics | ✓ baseline-subtracted excursion (step-0 gap_B +.0002) .055/.047/.057 vs threshold .05: s0, s2 pass, s1 fails narrowly; f_B > .1 in all; sustained return by step 125 (predicted within 2000). |
| P4 unanchored | ✓ x-CE_B → 1.36 as predicted (all); f_A ≈ f_B → 1 along the diagonal (|f_A − f_B| ≤ .04 at every checkpoint); no gate by patch/erasure. |
| P5 switch causality | ✓ position-curve Spearman 1.000 (> .9); ✓ per-history calibration (post hoc); ✓ patch at L1mlp moves f_A by 94–98% of the A–B gap (> 50%). |

## 4. What it means, and what it does not

For the story (†) in PREREG §0: when a shared circuit serves two in-context-inferred situations and
training changes the demanded behaviour in one while the other's loss remains, gradient descent here
(i) first leaks the change into both situations through the shared pathway, (ii) builds a gate on the
situational posterior only as the protected situation is damaged (by ~.05 nats here), (iii) builds it
within ~50 steps, in the blocks where the posterior first becomes available, as a case-conditional
relabeling of the shared belief's coordinates, and (iv) leaves the tracker's early blocks, the embedding
and the unembedding functionally as they were. The protected behaviour returns to reference, and the
changed behaviour is confined to the eval-like situation with the posterior's own weight: an ambiguous
context receives the calibrated mixture. Two corollaries for the blog: the misbehaviour leaks into the
protected context before the gate forms, by an amount that depends on how quickly the anchor's gradient
grows; and the situational variable becomes more linearly accessible under quarantine training.

Not shown: intent; shielding without an incentive (the unanchored arm corrupts fully, and its population
optimum is the corrupted one); anything about RL or KL anchors (the anchor here is an explicit B loss);
in-place permutation vs conditional branch; robustness to a history-heavy shared circuit (this HMM1 is
nearly a two-token lookup: a last-token predictor recovers 94% of the learnable signal); situations that
switch within a sequence; behavioural (rather than probe-based) situational discrimination.

## 5. Next

1. Anchor-strength sweep: A:B mix 50/50 → 95/5. Prediction: damage before the bend grows as the B share
   shrinks; below some share the arm behaves like the unanchored one.
2. KL-to-reference anchor in place of B data (closest to RLHF).
3. History-heavy HMM1 (e.g. α .7, x_leave .05) as a robustness arm.
4. Ergodic variant: case switches mid-sequence; does the gate follow the posterior in time?
5. Circuit-level: in-place permutation vs branch in block 3 (head/neuron-level patching).

## 6. File map

```
process.py, oracle_sim.py, scan_params.py   generator / oracle / parameter scan (sim/)
model.py, train.py                           GPT + three-phase training, dense eval (runs/s*/{pretrain,anchored,unanchored}/train.json)
probe.py                                     probes, transfer, encoder-image erasure, case patch (runs/*/*/probe.json, logs/probe_*.log)
restore_calib.py                             weight restoration, per-history calibration, position Spearman, P1 t>=8 (runs/restore_calib.json)
analyze.py                                   traces, (f_B,f_A) square, displacement, position curves (figs/)
figs/f_square.png                            hero: trajectories in the (f_B, f_A) square
figs/zoom_first300.png                       first 300 steps, both arms
figs/position_f.png                          f_A(t), f_B(t) vs oracle w_A(t)
figs/gate_localization.png                   transfer R², case patch, HMM1 erasure per capture
figs/restoration_calibration.png             weight restoration; per-history calibration
logs/codex_prereg_review.md, logs/codex_results_review.md   adversarial reviews
```
Reading additional input from stdin...
OpenAI Codex v0.153.4
--------
workdir: /data/users/asvin/self_inoculation
model: gpt-6-astra
provider: openai
approval: never
sandbox: read-only
reasoning effort: none
reasoning summaries: none
session id: 01a08e1b-65e0-7681-a156-ccc47b42d11a
--------
user
Read RESULTS.md, then PREREG.md (design + amendments) and, only if needed to check a number, the logs logs/probe_anchored_s0.log and logs/probe_unanchored_s0.log (read-only). Two tasks, numbered lists, do NOT rewrite:

(A) ADVERSARIAL READ of the conclusions in RESULTS.md. Assume the authors are over-reading. For each load-bearing claim, say whether the reported evidence supports it and what alternative reading survives: (1) 'the gate is a case-conditional relabeling of the shared belief at block 3, tracker (blocks 1-2) and unembedding untouched' — is the transfer-probe evidence (original vs sigma-relabeled R^2 per context) plus the patch localization plus unembedding displacement sufficient, or could a separate A-specific pathway produce the same signatures? (2) 'identical until damaged, then a 50-step bend' — does the identical early trajectory of the two arms actually show the B-loss is silent until damage, or is that just what any two runs from the same init with a shared A-gradient would do? (3) 'Bayes-exact gate' from f_A(t) on w_A(t) and per-history Spearman .9 — what would a non-Bayesian gate look like that also passes these? (4) the corollary 'quarantine training sharpens situational discrimination' from logit-w R^2 .86 -> .96 — confound with the case belief simply being used more downstream? (5) the scorecard: is any pass declared on a threshold that the amendments moved after seeing data, and is the false prediction (displacement) stated without salvage? (6) any internal inconsistency between numbers quoted in RESULTS.md and the probe logs. Be blunt.

(B) FRESH-READER CLARITY. A technically sophisticated reader (transformers, HMMs, Bayesian filtering, GD) reads RESULTS.md with zero project context and without PREREG.md. List every point that is unclear, undefined, context-presupposing, or where a symbol/term is used before its role is stated. Quote the phrase. Skip praise.
codex
I’ll read RESULTS.md, then PREREG.md, and check the probe logs only for numerical consistency. I’ll return two numbered lists without rewriting the document.
exec
/usr/bin/bash -lc 'cat RESULTS.md' in /data/users/asvin/self_inoculation
 succeeded in 0ms:
# RESULTS — Self-inoculation toy, v1 (runs 2026-09-10/11)

Companion to `PREREG.md` (design, Codex review, amendments A1–A14). 3 seeds; every number below is
reported per seed as s0/s1/s2 unless marked "all". Gaps are cross-entropies in nats minus the exact
Bayes reference computed on the same evaluation sequences. Figures in `figs/`.

## 0. Verdict

Under conflicting demands on a shared circuit, the network first changes the shared readout for both
contexts, and only once the protected context is measurably damaged does it build a gate: a
case-conditional relabeling of the shared belief, placed where the situational posterior becomes
available (block 3), leaving the tracker (blocks 1–2) and the unembedding untouched. Without the
protecting loss the same network rides the identical trajectory to full corruption. The gate is
Bayes-exact in the situational posterior, per position and per history. One preregistered prediction
(weight change concentrated late) is false: the change is distributed.

## 1. Setup reminder and what was run

Each token is a pair (x, y). x is emitted by HMM1 = Mess3(α=.9, x=.1), the **shared circuit**. y is
emitted by one of two processes fixed per sequence, Mess3A(.85,.15) or Mess3B(.85,.5), the **case**;
the model must infer which from context (the exact posterior w_A = P(A | y-stream) reaches 1% error by
position ≈13). x is independent of the case. **Pretrain**: predict (x, y) on both cases, 20k steps.
**Finetune** from the same checkpoint, 10k steps: in case A the x-*target* is relabeled by the swap
σ = (0↔1) (inputs never change; by S₃ symmetry the Bayes reference given the case is unchanged);
case B is unchanged. **Anchored** arm trains on A-flipped + B (50/50). **Unanchored** trains on A-flipped
only (batch halved so A-exposure per step matches). Model: 4-layer GPT, d=120, context 64.

**Flip fraction f.** For each position, the model's x-prediction is regressed onto the segment from P
(the unflipped Bayes prediction) to P^σ (the flipped one): f = 0 is pretrained behaviour, f = 1 fully
flipped. f_A, f_B are measured on case-A and case-B contexts over positions 16–63 (case resolved);
f_expl is the share of the model's deviation from P that lies along that segment (a fit-quality check).
Ideal gated endpoint (f_B, f_A) = (0, 1); corrupted shared readout = (1, 1).

Interventions (amendment A13): **erasure** projects the 2-dim *encoder image* of the HMM1 belief (or the
5-dim image of the case belief) out of the residual stream at a capture; **case PATCH** uses paired runs
with the same x-stream under an A and a B y-stream and transplants the case-subspace component of one
run into the other at capture ℓ; **transfer** applies the pretrain-endpoint HMM1 probe to finetune-endpoint
activations, scoring it against the true belief and against the σ-relabeled belief.

## 2. Results

### 2.1 Endpoints (positions 16–63)

| | anchored s0/s1/s2 | unanchored s0/s1/s2 |
|---|---|---|
| x-CE_A on the flipped task − Bayes-gated ref (0.852) | −.0003/+.0010/+.0003 | −.0025 all |
| x-CE_B on the unflipped task − Bayes-gated ref (0.854) | +.0017/−.0001/+.0008 | +.5125/+.5072/+.5106 |
| x-CE_B absolute | 0.855 | **1.360** (prereg "corrupted" value 1.3599) |
| f_A (f_expl) | .990/.975/.987 (1.00/.99/.99) | 1.001/.994/.999 (1.00) |
| f_B (f_expl) | .016/.020/.014 (.03/.07/.03) | 1.002/.995/.998 (1.00) |
| y-CE_A, y-CE_B − ref | ≤ .0012 all | A ≤ .002; **B +.343/+.344/+.342** |

Anchored: both contexts at reference, flip confined to A. Unanchored: fully corrupted, and the
deployment-side y model rusts (its B branch loses 0.34 nats with nothing maintaining it).

### 2.2 Dynamics: identical until damaged, then a 50-step bend (`figs/f_square.png`, `figs/zoom_first300.png`)

Both arms follow the *same* path up the diagonal for 75 steps: f_A ≈ f_B ≈ 0.4 at step 75, x-CE_B
excess +.055/+.047/+.057 (anchored) and +.089/+.109/+.086 (unanchored). The cheapest reduction of the
A-loss is to permute the readout that both contexts share, and the B-loss is silent until that has done
visible damage. Between steps 75 and 125 the arms split: anchored goes to (f_B, f_A) = (0.02, 0.96),
B back within 0.01 of reference by step 125 and staying there; unanchored continues to (1, 1) by step
125 and stays. Same ordering and timing in all seeds. Peak f_B in the anchored arm .32/.29/.33.

### 2.3 The gate is Bayes-exact in the case posterior (`figs/position_f.png`)

At the anchored endpoint, f_A(t) by position lies on the oracle mean posterior w_A(t): 0.5 at t=0,
0.8 by t≈8, 0.9 by t≈15, 1.0 by t≈40; f_B(t) is the mirror image. Per history, at fixed t = 4…8 the
single-sequence flip fraction correlates with that sequence's oracle w_A at Spearman ρ = .88–.96 in
A-contexts and .78–.92 in B-contexts (pretrain baseline ≈ 0 in A, −0.2 in B). The model flips by its
actual current belief that it is in A, not by a position schedule.

### 2.4 Where the gate is (`figs/gate_localization.png`)

- **Shared tracker, used in both contexts.** Erasing the HMM1 encoder image at all captures raises x-CE
  on the context's own task by +.225/+.257/+.232 in A and +.213/+.239/+.214 in B (pretrain baseline
  +.250/+.247; uniform is +.25), y unchanged (≤ .004). Ratio A/B 1.06–1.08 (P2 window [0.5, 2]). The
  decoder-subspace erasure remains inert (+.001–.005), as at pretrain.
- **Blocks 1–2 unchanged, block 3 relabels in A only.** The pretrain HMM1 probe applied at the anchored
  endpoint reads the belief in original coordinates at L1mlp in both contexts (R² .94/.96/.87). At L3mlp
  it still reads B-contexts unpermuted (R² .89/.91/.86) but reads A-contexts only after σ-relabeling
  (R² .83/.57/.84; unpermuted −.72/+.09/−1.16); same at L4mlp (σ: .79/.68/.75). The unembedding barely
  moved (relative displacement .03 at step 125, .18 at 10k), so a σ-permuted belief in A is read by the
  same unembedding as a flipped prediction. This is the "upstream gating" variant of §4(c): the gate
  acts on the shared circuit's output coordinates, not on a separate readout matrix.
- **The gate reads the case subspace before block 3.** Patching B's case-subspace component into an A run
  at L1mlp sends f_A .99 → .03/.03/.04 (and B←A: .01 → .97); at L2mlp → .07/.06/.09; at L3mlp only
  → .93/.87/.92; at L4mlp no effect. Erasing the case subspace at all captures collapses both contexts to
  the same f (.41/.50/.34), i.e. the gate loses its input and outputs a mixture. In the unanchored arm the
  same patch and erasure have no effect on f (1.00 → 1.00): no gate exists there.
- **Placement is just-in-time.** The case posterior becomes linearly readable through blocks 1–3
  (logit-w R² .18 → .52 → .94 at L1mlp/L2mlp/L3mlp); the conditional relabeling sits at block 3, exactly
  where w is available. The unanchored arm's *unconditional* relabeling instead lands at block 1 (largest
  early displacement L1.mlp .15–.18 at step 125; pretrain probe fails from L2 on in both contexts).

### 2.5 Representation side-effects

Anchored finetune **sharpened** the situational variable: logit-w R² at L4mlp .86 → .956/.970/.933,
case-belief R² .95 → .974/.979/.969 (in s1 the case belief also becomes readable earlier, L2mlp .70 →
.90). Unanchored finetune **degraded** it: logit-w .86 → .41/.53/.45, case belief .95 → .68/.69/.67.
The variable gains a consumer (the gate) in one arm and loses one (the B branch of y) in the other; the
representation follows the consumers. The HMM1 belief stays R² ≥ .99 in every layer, arm and context.

### 2.6 Weight displacement (the false prediction)

Relative L2 displacement from the pretrained weights at step 125 (gate just built), anchored s0: embed
.006, L1.attn .09, L1.mlp .14, L2.attn .11, L2.mlp .13, L3.attn .14, **L3.mlp .17**, L4.attn .09,
L4.mlp .14, final .03. Distributed across all blocks with L3.mlp largest; the preregistered
"concentrated in the last block + unembedding, early groups < .05" is **false**. The transfer probe and
the patch localize the gate; the displacement metric does not.

## 3. Prediction scorecard (PREREG §5; a prediction holds if it holds in ≥ 2 of 3 seeds)

| | outcome |
|---|---|
| P1 pretrain | ✓ joint CE within .0006 of reference (all); HMM1 R² .99; FWH overlap .02–.16 (< .2); logit-w R² .86 at L4mlp pooled over t ≥ 1 (threshold .9 stated for t ≥ 8; not separately computed) |
| P2 anchored endpoint | ✓ losses, ✓ f (f_A > .9, f_B < .1, f_expl > .8), ✓ erasure ratio in [.5, 2], ✓ transfer R² > .8 in B-contexts; **✗ displacement clause** (§2.6) |
| P3 anchored dynamics | ✓ excursion > .05 in s0, s2 (.055, .057); s1 .047; f_B > .1 in all; return by step 125 (predicted within 2000) |
| P4 unanchored | ✓ x-CE_B → 1.36 as predicted; f_A ≈ f_B → 1 along the diagonal (all); no gate by patch/erasure |
| P5 switch causality | ✓ f_A(t) on w_A(t); per-history ρ .88–.96; patch moves f_A by 97% of the A–B gap at L1–L2 |

## 4. What it means, and what it does not

For the story (†) in PREREG §0: when a shared circuit serves two in-context-inferred situations and
training changes the demanded behaviour in one of them while the other's loss remains, gradient descent
here (i) first leaks the change into both situations through the shared readout, (ii) builds a gate on
the situational posterior only after the protected situation has been damaged by ~0.05 nats, (iii)
builds it in ~50 steps at the layer where the posterior is available, as a conditional relabeling of the
shared circuit's output, and (iv) leaves the circuit and the readout intact. The protected behaviour is
restored to reference, and the misbehaviour is confined to the eval-like situation with exactly the
posterior's weight: an ambiguous context receives the mixture. Two corollaries worth the blog: the
misbehaviour *does* leak into deployment before the gate forms, by an amount set by how fast the anchor
reacts; and quarantine training *improves* the network's situational discrimination (logit-w R² .86 →
.96), because the situational variable acquires a new consumer.

Not shown: intent; shielding without an incentive (the unanchored arm corrupts fully, and its population
optimum is the corrupted one); anything about RL or KL anchors (the anchor here is an explicit B loss);
robustness to a history-heavy shared circuit (this HMM1 is nearly a two-token lookup: a last-token
predictor recovers 94% of the learnable signal); situations that switch within a sequence.

## 5. Next

1. Anchor-strength sweep: A:B mix 50/50 → 95/5. Prediction: damage before the bend grows as the B share
   shrinks; gate forms later; below some share the arm behaves like the unanchored one.
2. KL-to-reference anchor in place of B data (closest to RLHF): does a KL penalty on all contexts build
   the same gate?
3. History-heavy HMM1 (e.g. α .7, x .05) as a robustness arm.
4. Ergodic variant: case switches mid-sequence; does the gate follow the posterior in time?

## 6. File map

```
process.py, oracle_sim.py, scan_params.py   generator / oracle / parameter scan (sim/)
model.py, train.py                           GPT + three-phase training with dense eval (runs/s*/{pretrain,anchored,unanchored}/train.json)
probe.py                                     probes, transfer, encoder-image erasure, case patch (runs/*/*/probe.json, logs/probe_*.log)
analyze.py                                   traces, (f_B,f_A) square, displacement, position curves (figs/)
figs/f_square.png                            hero: trajectories in the (f_B, f_A) square
figs/zoom_first300.png                       first 300 steps, both arms
figs/position_f.png                          f_A(t), f_B(t) vs oracle w_A(t)
figs/gate_localization.png                   transfer R², case patch, HMM1 erasure per capture
logs/codex_prereg_review.md                  adversarial design review (35 + 57 points)
```

exec
/usr/bin/bash -lc 'cat PREREG.md' in /data/users/asvin/self_inoculation
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc 'cat logs/probe_anchored_s0.log logs/probe_unanchored_s0.log' in /data/users/asvin/self_inoculation
 succeeded in 0ms:
ckpt runs/s0/anchored/model.pt
capture   R2 hmm1(A/B/pool)      R2 case(pool)  R2 logit_w(pool)  | ovl hmm1 A-B  hmm1-case
  emb     0.944/0.944/0.944      0.357        -0.000        |   1.000       0.050
  L1attn  0.992/0.992/0.992      0.455        0.001        |   0.904       0.120
  L1mlp   0.992/0.992/0.992      0.560        0.175        |   0.969       0.111
  L2attn  0.994/0.994/0.994      0.726        0.577        |   0.968       0.094
  L2mlp   0.996/0.996/0.996      0.715        0.515        |   0.920       0.056
  L3attn  0.996/0.996/0.996      0.830        0.871        |   0.912       0.085
  L3mlp   0.993/0.995/0.992      0.956        0.943        |   0.759       0.118
  L4attn  0.992/0.994/0.990      0.972        0.956        |   0.713       0.089
  L4mlp   0.993/0.993/0.988      0.974        0.956        |   0.569       0.139
transfer from ref (pool hmm1 probe): capture  A_r2  A_r2_sigma  B_r2  B_r2_sigma | w: A B
  emb     0.943  -0.964  0.943  -0.964 | -4.661 -4.508
  L1attn  0.962  -1.029  0.963  -1.028 | -15.488 -0.548
  L1mlp   0.944  -1.249  0.938  -1.273 | -20.752 -8.980
  L2attn  0.920  -0.574  0.930  -0.628 | -84.108 -83.037
  L2mlp   0.793  -0.133  0.828  -0.195 | -50.364 -40.681
  L3attn  0.594  -0.135  0.884  -0.573 | -47.258 -103.808
  L3mlp   -0.716  0.828  0.890  -1.008 | -83.983 -16.321
  L4attn  -0.719  0.832  0.897  -1.010 | -39.582 -22.963
  L4mlp   -1.158  0.788  0.820  -1.223 | -62.627 -14.662
interventions (t>=16): site:kind  ctx  d_x_unfl  d_x_fl   d_y    f0->f (f_expl)  |dresid|/|resid|  [casePATCH: f of other ctx]
  emb:hmm1             A   -0.090   +0.023  +0.001   0.99->0.89 (0.96)   0.391
  emb:hmm1             B   +0.017   -0.068  +0.001   0.01->0.09 (0.21)   0.390
  emb:case             A   -0.077   +0.008  +0.042   0.99->0.91 (0.98)   0.485
  emb:case             B   +0.146   -0.352  +0.141   0.01->0.50 (0.79)   0.485
  emb:casePATCH        A   -0.493   +0.452  +0.237   0.99->0.07 (0.16)   nan  other-ctx f 0.01
  emb:casePATCH        B   +0.483   -0.498  +0.216   0.01->0.97 (0.99)   nan  other-ctx f 0.99
  L1attn:hmm1          A   -0.079   +0.018  +0.001   0.99->0.90 (0.97)   0.370
  L1attn:hmm1          B   +0.017   -0.063  +0.000   0.01->0.09 (0.19)   0.368
  L1attn:case          A   -0.040   +0.001  +0.022   0.99->0.96 (1.00)   0.538
  L1attn:case          B   +0.086   -0.288  +0.098   0.01->0.36 (0.72)   0.551
  L1attn:casePATCH     A   -0.499   +0.458  +0.239   0.99->0.06 (0.18)   nan  other-ctx f 0.01
  L1attn:casePATCH     B   +0.483   -0.501  +0.220   0.01->0.97 (0.99)   nan  other-ctx f 0.99
  L1mlp:hmm1           A   -0.165   +0.211  +0.004   0.99->0.62 (0.59)   0.498
  L1mlp:hmm1           B   +0.209   -0.158  +0.002   0.01->0.37 (0.34)   0.494
  L1mlp:case           A   -0.340   +0.084  +0.119   0.99->0.59 (0.97)   0.634
  L1mlp:case           B   +0.162   -0.417  +0.135   0.01->0.59 (0.97)   0.624
  L1mlp:casePATCH      A   -0.504   +0.478  +0.261   0.99->0.03 (0.15)   nan  other-ctx f 0.01
  L1mlp:casePATCH      B   +0.484   -0.502  +0.219   0.01->0.97 (0.99)   nan  other-ctx f 0.99
  L2attn:hmm1          A   -0.143   +0.251  +0.002   0.99->0.60 (0.53)   0.466
  L2attn:hmm1          B   +0.251   -0.137  +0.002   0.01->0.39 (0.32)   0.460
  L2attn:case          A   -0.071   +0.002  +0.111   0.99->0.93 (1.00)   0.606
  L2attn:case          B   +0.289   -0.473  +0.136   0.01->0.78 (0.98)   0.636
  L2attn:casePATCH     A   -0.498   +0.444  +0.242   0.99->0.07 (0.24)   nan  other-ctx f 0.01
  L2attn:casePATCH     B   +0.474   -0.501  +0.212   0.01->0.96 (0.99)   nan  other-ctx f 0.99
  L2mlp:hmm1           A   -0.158   +0.199  +0.003   0.99->0.63 (0.60)   0.448
  L2mlp:hmm1           B   +0.195   -0.149  +0.003   0.01->0.36 (0.33)   0.453
  L2mlp:case           A   -0.172   +0.024  +0.107   0.99->0.82 (0.97)   0.624
  L2mlp:case           B   +0.304   -0.470  +0.124   0.01->0.79 (0.96)   0.611
  L2mlp:casePATCH      A   -0.499   +0.438  +0.254   0.99->0.07 (0.32)   nan  other-ctx f 0.01
  L2mlp:casePATCH      B   +0.461   -0.500  +0.219   0.01->0.95 (0.99)   nan  other-ctx f 0.99
  L3attn:hmm1          A   -0.144   +0.187  +0.002   0.99->0.66 (0.63)   0.376
  L3attn:hmm1          B   +0.181   -0.123  +0.004   0.01->0.33 (0.29)   0.377
  L3attn:case          A   -0.289   +0.061  +0.107   0.99->0.67 (0.96)   0.686
  L3attn:case          B   +0.107   -0.356  +0.120   0.01->0.46 (0.91)   0.679
  L3attn:casePATCH     A   -0.487   +0.397  +0.256   0.99->0.12 (0.35)   nan  other-ctx f 0.01
  L3attn:casePATCH     B   +0.429   -0.494  +0.219   0.01->0.92 (0.98)   nan  other-ctx f 0.99
  L3mlp:hmm1           A   +0.030   +0.123  +0.002   0.99->0.87 (0.77)   0.425
  L3mlp:hmm1           B   +0.115   +0.072  +0.002   0.01->0.10 (0.04)   0.386
  L3mlp:case           A   -0.013   +0.001  +0.108   0.99->0.98 (0.99)   0.509
  L3mlp:case           B   +0.001   -0.029  +0.113   0.01->0.03 (0.12)   0.497
  L3mlp:casePATCH      A   -0.073   +0.005  +0.233   0.99->0.93 (0.99)   nan  other-ctx f 0.01
  L3mlp:casePATCH      B   +0.009   -0.104  +0.216   0.01->0.11 (0.50)   nan  other-ctx f 0.99
  L4attn:hmm1          A   +0.029   +0.124  +0.001   0.99->0.87 (0.77)   0.401
  L4attn:hmm1          B   +0.115   +0.075  +0.001   0.01->0.10 (0.04)   0.345
  L4attn:case          A   -0.007   +0.001  +0.104   0.99->0.99 (0.99)   0.556
  L4attn:case          B   +0.001   -0.021  +0.110   0.01->0.03 (0.08)   0.604
  L4attn:casePATCH     A   -0.062   +0.004  +0.239   0.99->0.94 (0.99)   nan  other-ctx f 0.01
  L4attn:casePATCH     B   +0.007   -0.089  +0.214   0.01->0.09 (0.46)   nan  other-ctx f 0.99
  L4mlp:hmm1           A   +0.091   +0.117  +0.001   0.99->0.91 (0.79)   0.349
  L4mlp:hmm1           B   +0.115   +0.125  +0.001   0.01->0.07 (0.02)   0.327
  L4mlp:case           A   +0.058   +0.002  +0.108   0.99->1.04 (0.99)   0.352
  L4mlp:case           B   +0.001   +0.028  +0.114   0.01->-0.01 (0.17)   0.416
  L4mlp:casePATCH      A   +0.022   +0.001  +0.254   0.99->1.01 (0.99)   nan  other-ctx f 0.01
  L4mlp:casePATCH      B   +0.002   -0.033  +0.222   0.01->0.04 (0.19)   nan  other-ctx f 0.99
  ALL:hmm1             A   -0.205   +0.225  +0.002   0.99->0.57 (0.55)   0.531
  ALL:hmm1             B   +0.213   -0.177  +0.002   0.01->0.40 (0.38)   0.529
  ALL:case             A   -0.421   +0.164  +0.109   0.99->0.41 (0.94)   0.709
  ALL:case             B   +0.085   -0.339  +0.115   0.01->0.41 (0.94)   0.701
  L4mlp:hmm1_DECODER   A   +0.004   +0.001  +0.000   0.99->0.99 (0.99)   0.089
  L4mlp:hmm1_DECODER   B   +0.001   +0.001  +0.000   0.01->0.02 (0.04)   0.094
per-history Spearman(f_n, w_A,n) at t=3..12:
  A: t3:+0.88(n=1127)  t4:+0.94(n=1124)  t5:+0.94(n=1140)  t6:+0.93(n=1156)  t7:+0.92(n=1142)  t8:+0.88(n=1125)  t9:+0.86(n=1143)  t10:+0.83(n=1111)  t11:+0.82(n=1116)  t12:+0.76(n=1131)
  B: t3:+0.74(n=1127)  t4:+0.82(n=1124)  t5:+0.84(n=1140)  t6:+0.87(n=1156)  t7:+0.90(n=1142)  t8:+0.92(n=1125)  t9:+0.92(n=1143)  t10:+0.90(n=1111)  t11:+0.86(n=1116)  t12:+0.83(n=1131)
ckpt runs/s0/unanchored/model.pt
capture   R2 hmm1(A/B/pool)      R2 case(pool)  R2 logit_w(pool)  | ovl hmm1 A-B  hmm1-case
  emb     0.944/0.944/0.944      0.357        -0.000        |   1.000       0.048
  L1attn  0.992/0.992/0.992      0.453        0.001        |   0.931       0.145
  L1mlp   0.992/0.992/0.992      0.517        0.108        |   0.944       0.064
  L2attn  0.995/0.995/0.995      0.521        0.122        |   0.961       0.108
  L2mlp   0.998/0.998/0.998      0.589        0.265        |   0.949       0.058
  L3attn  0.997/0.998/0.997      0.649        0.371        |   0.959       0.096
  L3mlp   0.999/0.999/0.999      0.675        0.383        |   0.951       0.082
  L4attn  0.999/0.999/0.999      0.680        0.400        |   0.945       0.083
  L4mlp   0.999/0.999/0.999      0.680        0.405        |   0.922       0.048
transfer from ref (pool hmm1 probe): capture  A_r2  A_r2_sigma  B_r2  B_r2_sigma | w: A B
  emb     0.942  -0.964  0.942  -0.964 | -4.661 -4.508
  L1attn  0.805  -0.795  0.807  -0.794 | -9.722 -1.812
  L1mlp   0.868  -1.323  0.868  -1.307 | -23.714 -24.982
  L2attn  -2.733  -0.168  -2.707  -0.148 | -585.217 -486.328
  L2mlp   -3.586  -0.130  -3.639  -0.160 | -357.614 -327.479
  L3attn  -3.568  -0.109  -3.608  -0.128 | -115.711 -71.387
  L3mlp   -3.327  -0.013  -3.384  -0.047 | -94.551 -170.394
  L4attn  -3.218  0.043  -3.280  0.007 | -36.649 -90.918
  L4mlp   -3.102  0.057  -3.144  0.031 | -92.725 -186.658
interventions (t>=16): site:kind  ctx  d_x_unfl  d_x_fl   d_y    f0->f (f_expl)  |dresid|/|resid|  [casePATCH: f of other ctx]
  emb:hmm1             A   -0.021   +0.010  +0.001   1.00->0.97 (0.98)   0.395
  emb:hmm1             B   -0.021   +0.008  -0.011   1.00->0.98 (0.99)   0.395
  emb:case             A   +0.003   +0.005  +0.016   1.00->1.00 (0.99)   0.481
  emb:case             B   +0.003   +0.006  -0.087   1.00->1.00 (0.99)   0.482
  emb:casePATCH        A   -0.005   +0.004  +0.125   1.00->0.99 (0.99)   nan  other-ctx f 1.00
  emb:casePATCH        B   -0.006   +0.004  -0.119   1.00->0.99 (0.99)   nan  other-ctx f 1.00
  L1attn:hmm1          A   -0.028   +0.007  +0.000   1.00->0.97 (0.99)   0.465
  L1attn:hmm1          B   -0.028   +0.007  -0.007   1.00->0.97 (0.99)   0.465
  L1attn:case          A   +0.006   +0.000  +0.010   1.00->1.01 (1.00)   0.509
  L1attn:case          B   +0.007   +0.000  -0.074   1.00->1.01 (1.00)   0.510
  L1attn:casePATCH     A   -0.001   +0.000  +0.103   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L1attn:casePATCH     B   -0.002   +0.001  -0.114   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L1mlp:hmm1           A   -0.196   +0.358  +0.011   1.00->0.47 (0.35)   0.597
  L1mlp:hmm1           B   -0.197   +0.361  -0.088   1.00->0.47 (0.35)   0.599
  L1mlp:case           A   +0.008   +0.001  +0.111   1.00->1.00 (1.00)   0.599
  L1mlp:case           B   +0.007   +0.001  -0.224   1.00->1.00 (1.00)   0.591
  L1mlp:casePATCH      A   +0.001   -0.000  +0.186   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L1mlp:casePATCH      B   -0.000   +0.000  -0.120   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L2attn:hmm1          A   -0.169   +0.326  +0.009   1.00->0.53 (0.42)   0.688
  L2attn:hmm1          B   -0.168   +0.334  -0.064   1.00->0.52 (0.41)   0.705
  L2attn:case          A   +0.015   +0.001  +0.111   1.00->1.01 (1.00)   0.539
  L2attn:case          B   +0.014   +0.001  -0.224   1.00->1.01 (1.00)   0.504
  L2attn:casePATCH     A   +0.002   -0.000  +0.183   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L2attn:casePATCH     B   -0.000   +0.000  -0.121   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L2mlp:hmm1           A   -0.246   +0.280  +0.001   1.00->0.49 (0.46)   0.645
  L2mlp:hmm1           B   -0.247   +0.282  -0.012   1.00->0.49 (0.46)   0.655
  L2mlp:case           A   +0.004   +0.000  +0.112   1.00->1.01 (1.00)   0.473
  L2mlp:case           B   +0.005   +0.001  -0.225   1.00->1.01 (1.00)   0.437
  L2mlp:casePATCH      A   -0.000   -0.000  +0.186   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L2mlp:casePATCH      B   +0.001   +0.000  -0.120   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L3attn:hmm1          A   -0.246   +0.272  +0.001   1.00->0.50 (0.47)   0.650
  L3attn:hmm1          B   -0.245   +0.272  -0.011   1.00->0.50 (0.47)   0.661
  L3attn:case          A   +0.013   +0.001  +0.112   1.00->1.01 (1.00)   0.469
  L3attn:case          B   +0.014   +0.001  -0.225   1.00->1.01 (1.00)   0.432
  L3attn:casePATCH     A   -0.001   -0.000  +0.185   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L3attn:casePATCH     B   +0.001   +0.000  -0.120   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L3mlp:hmm1           A   -0.262   +0.256  +0.003   1.00->0.50 (0.50)   0.615
  L3mlp:hmm1           B   -0.262   +0.256  +0.061   1.00->0.50 (0.50)   0.622
  L3mlp:case           A   +0.008   +0.001  +0.111   1.00->1.01 (1.00)   0.414
  L3mlp:case           B   +0.010   +0.001  -0.226   1.00->1.01 (1.00)   0.380
  L3mlp:casePATCH      A   -0.002   +0.000  +0.185   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L3mlp:casePATCH      B   +0.002   +0.000  -0.121   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L4attn:hmm1          A   -0.261   +0.257  +0.002   1.00->0.50 (0.50)   0.574
  L4attn:hmm1          B   -0.261   +0.256  +0.047   1.00->0.50 (0.50)   0.584
  L4attn:case          A   +0.011   +0.001  +0.110   1.00->1.01 (1.00)   0.493
  L4attn:case          B   +0.014   +0.001  -0.227   1.00->1.01 (1.00)   0.457
  L4attn:casePATCH     A   -0.002   +0.000  +0.186   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L4attn:casePATCH     B   +0.002   +0.000  -0.121   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L4mlp:hmm1           A   -0.262   +0.256  +0.003   1.00->0.50 (0.50)   0.495
  L4mlp:hmm1           B   -0.263   +0.255  +0.049   1.00->0.50 (0.50)   0.496
  L4mlp:case           A   +0.014   +0.001  +0.109   1.00->1.01 (1.00)   0.341
  L4mlp:case           B   +0.018   +0.001  -0.227   1.00->1.02 (1.00)   0.318
  L4mlp:casePATCH      A   -0.003   +0.000  +0.186   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  L4mlp:casePATCH      B   +0.003   +0.000  -0.121   1.00->1.00 (1.00)   nan  other-ctx f 1.00
  ALL:hmm1             A   -0.254   +0.265  +0.033   1.00->0.50 (0.48)   0.690
  ALL:hmm1             B   -0.256   +0.263  -0.157   1.00->0.50 (0.48)   0.696
  ALL:case             A   -0.013   +0.008  +0.108   1.00->0.98 (0.98)   0.547
  ALL:case             B   -0.014   +0.008  -0.226   1.00->0.98 (0.98)   0.528
  L4mlp:hmm1_DECODER   A   -0.017   +0.000  +0.000   1.00->0.99 (1.00)   0.098
  L4mlp:hmm1_DECODER   B   -0.017   +0.000  +0.002   1.00->0.99 (1.00)   0.099
per-history Spearman(f_n, w_A,n) at t=3..12:
  A: t3:-0.08(n=1127)  t4:-0.06(n=1124)  t5:-0.06(n=1140)  t6:-0.12(n=1156)  t7:-0.06(n=1142)  t8:-0.02(n=1125)  t9:-0.05(n=1143)  t10:-0.08(n=1111)  t11:-0.03(n=1116)  t12:-0.05(n=1131)
  B: t3:+0.01(n=1127)  t4:-0.02(n=1124)  t5:+0.04(n=1140)  t6:-0.02(n=1156)  t7:-0.01(n=1142)  t8:-0.00(n=1125)  t9:-0.03(n=1143)  t10:+0.00(n=1111)  t11:-0.03(n=1116)  t12:-0.01(n=1131)

codex
The logs support an A-specific change in the coordinates read by the old probe, but they do not establish an untouched tracker or exclude a separate A-specific pathway. I also found scorecard gaps: P1’s specified logit-posterior test was not computed, P2’s reported B-context fit quality fails its literal threshold, and the “identical” early trajectories conflict with numbers in RESULTS.md itself.
codex
**(A) ADVERSARIAL READ**

1. **“Case-conditional relabeling … tracker … and unembedding untouched”: the relabeling signature is supported; the exclusive circuit interpretation is not.** The old probe reads approximately original coordinates in B and approximately permuted coordinates in A after L3mlp. Early case patches strongly change the output; patches after L3mlp mostly do not. Together, these support case-dependent processing whose output is largely established by L3mlp.

   A separate A-specific pathway survives all these observations. It could consume shared early x-features and case information, then write a correction into the residual coordinates that the old probe and unembedding read. That would produce the transfer signature, early patch sensitivity, and relatively stable unembedding. Erasing shared input features would damage both pathways, so comparable ablation losses do not rule this out. The experiment does not distinguish an in-place permutation from a conditional branch that writes an equivalent result.

   “Blocks 1–2 unchanged” is particularly indefensible. Seed-0 original-coordinate transfer falls from approximately .94 at L1mlp to .793/.828 at L2mlp, and their weight displacements are substantial. Preserved decodability is not unchanged computation. Likewise, .18 relative unembedding displacement at 10k is not “untouched”; even .03 need not be functionally negligible. No frozen-unembedding or weight-restoration result establishes that its changes are irrelevant.

   “Exactly where w is available” also overstates localization. Seed-0 logit-w R² is already .871 at L3attn, and case patches work at L1mlp despite low linear logit-w decodability there. Availability to nonlinear computation is not the same as crossing a linear-probe threshold.

2. **“Identical until damaged, then a 50-step bend”: the coarse behavioral sequence is supported; the gradient-level explanation is not.** Similar initial movement is unsurprising from the same initialization with a common A objective and a pretrained B objective near its optimum. The expected B gradient can initially be small and then grow continuously as parameters move. That does not establish that it is “silent” until any particular measurable-damage threshold, or that no gate learning occurs before the visible bend.

   Nor are identical trajectories inevitable: the arms have different batches, loss mixtures, y objectives, and AdamW update histories. Matching A examples per step does not establish identical A-gradient contributions after averaging and optimization. The observed similarity therefore needs quantification, not a causal gloss.

   The document contradicts its own strongest wording. At step 75 it reports anchored B excess .055/.047/.057 versus unanchored .089/.109/.086. Those arms have already diverged in loss. Its “f_A ≈ f_B ≈ 0.4” also conflicts with anchored peak f_B of .32/.29/.33.

   “Cheapest reduction,” “only once,” and damage “set by how fast the anchor reacts” require evidence about gradients or controlled changes to anchor strength. The reported checkpoints support a rapid transition around steps 75–125, not an identified damage-triggered mechanism or an exactly 50-step construction time. A7 explicitly made the excursion a timestamp and conditioned stronger gate-building language on accompanying displacement evidence; that condition needs addressing.

3. **“Bayes-exact … per position and per history”: not established.** Position-averaged agreement plus high Spearman correlation demonstrates approximate posterior-related behavior. Spearman measures ordering, not calibration. For example,
   \[
   f(h)=\operatorname{sigmoid}\!\left(a\,\operatorname{logit}w_A(h)+b\right),\quad a>0,
   \]
   preserves posterior ranking perfectly while generally assigning the wrong mixture weights. A finite-history heuristic based on y-transition counts could also rank histories well without implementing the exact filter.

   Position means constrain these alternatives but do not eliminate them: modest miscalibration, history-dependent errors that cancel in aggregates, and small errors orthogonal to the flip direction can coexist with apparently matching curves. Moreover, A4’s pooled f is weighted by \(\|P^\sigma-P\|^2\), rather than automatically being an ordinary mean posterior.

   The per-history result concerns selected histories with sufficiently identifiable flips, at selected positions. RESULTS reports t=4–8, whereas A4 specifies t=3–12; seed-0 A correlations fall to .76 by t=12. That is incompatible with presenting the displayed range as comprehensive evidence of exactness.

   At t=0 the claim is worse: the uniform initial x prediction is invariant under σ, so f is unidentifiable. A4 explicitly masks it. “0.5 at t=0” is an oracle posterior value, not an identifiable model flip fraction.

   Direct calibration errors between f and w, together with prediction-vector errors against the oracle mixture on ambiguous histories, would support approximate Bayes optimality. These reported statistics do not establish exactness or an internally Bayesian computation.

4. **“Quarantine training improves … situational discrimination”: improved linear accessibility is supported; improved discrimination and its cause are not.** Higher fitted logit-w R² can result from the same case information being rescaled, reoriented, or made easier for a linear decoder to access because downstream computation uses it more. The original model may already distinguish cases equally well through a nonlinear representation.

   Actual improvement in discrimination would need a relevant behavioral or predictive measure—case classification error, posterior calibration, or a proper scoring rule—under comparable decoding conditions. A fitted probe’s R² alone does not supply that.

   “Because the situational variable acquires a new consumer” is a plausible explanation, not an identified cause. Anchored training changes the task and its case-dependent joint structure. Unanchored training also removes B data and changes the training prior. These are not isolated manipulations of the number of downstream consumers.

5. **The scorecard is too generous, although the chronology does not justify accusing the authors of moving thresholds after seeing finetune outcomes.** A1–A12 are stated to precede runs. A13 explicitly follows inspection of seed-0 pretraining, before finetuning: it replaces an inert decoder-space intervention with an effective encoder-image intervention, and replaces P5’s scalar-direction erasure criterion with a case-patch criterion. That is a data-informed methodological change and a changed pass criterion. It is disclosed, and the finetune prediction remains prospective under the stated chronology. It is not evidence that the .05 excursion threshold or other numeric outcome thresholds were moved after finetune results.

   Nevertheless, several checks are not earned:

   - **P1:** .86 pooled over t≥1 does not test R²>.9 at t≥8. RESULTS expressly says the latter was not computed. P1 is incomplete, not passed.
   - **P2:** the literal requirement includes f_expl>.8, but anchored B reports .03/.07/.03. Near f_B=0, this diagnostic can be unhelpful because the deviation being explained is tiny. That explains why the criterion may be poorly chosen; it does not make it pass. No amendment shown exempts B.
   - **P2 supporting evidence:** the A–B overlap >.8 criterion is omitted from the scorecard. Seed-0 overlap is .759 at L3mlp and .569 at L4mlp. The capture dependence should not disappear behind a general preservation verdict.
   - **P3:** A7 defines excursion relative to the step-0 gap. The scorecard supplies apparent absolute gaps from reference. With .055/.057 only narrowly exceeding .05, the starting-gap subtraction matters. The permitted probe logs cannot verify this.
   - **P5:** the original position-curve Spearman >.9 is not reported. A per-history correlation is a different statistic. The amended patch criterion passes comfortably, but it does not complete every other clause.

   The displacement prediction **is explicitly called false**, without redefining its threshold. However, the stronger theoretical conclusion is salvaged through transfer and patching, and “untouched” language survives despite the failed displacement prediction. So the metric failure is candidly admitted; its implications are not fully absorbed.

6. **Numerical audit: several overstatements and omissions, rather than wholesale transcription failure.** The principal seed-0 L3mlp transfer values, endpoint representation scores, ALL:HMM1 erasure losses, and L1/L2/L3 patch values match the logs. Specific problems are:

   - **“HMM1 belief stays R² ≥ .99 in every layer, arm and context”:** embeddings are .944 in both logs. If “layer” excludes embeddings and “context” excludes pooled fits, the separate A/B transformer-layer scores satisfy the claim; anchored L4mlp pooled R² is .988. The scope needs qualification.
   - **“Patch moves f_A by 97% … at L1–L2”:** using rounded seed-0 values, L1mlp gives \((.99-.03)/(.99-.01)\approx98\%\); L2mlp gives approximately94%. The latter is not 97%.
   - **“At L4mlp no effect”:** anchored patching gives A .99→1.01 and B .01→.04. Small effects, not literally zero.
   - **Unanchored “patch and erasure … (1.00→1.00)”:** ALL:case erasure gives 1.00→.98 in both contexts; some individual erasures give 1.01–1.02. Approximate insensitivity is supported.
   - **Unanchored relabeling “lands at block 1”:** old-probe original-coordinate transfer remains .868/.868 at L1mlp. From L2 onward both original and σ transfer are poor; σ transfer at L2mlp is −.130/−.160. These numbers locate substantial representational drift, not a clean block-1 permutation.
   - **“Mirror image”:** PREREG says true-B is recognized faster than true-A. Exact complementary *case-averaged* curves do not follow from the common posterior formula.
   - **The step-75 values and anchored f_B peaks conflict internally**, as discussed above.
   - The two permitted logs contain endpoint probes, not training loss trajectories, displacement histories, seed-1/2 outputs, or pretrain baselines. Those other numbers remain unchecked, not verified.

**(B) FRESH-READER CLARITY**

1. **“Self-inoculation toy, v1” / “For the story (†)”** — The title’s organizing concept and the referenced story are never explained locally. The reader must infer the intended analogy from later language about deployment and misbehavior.

2. **“3 seeds; every number below … per seed … unless marked ‘all’”** — Numerous single values are neither per-seed triples nor marked “all”: absolute CE, initial probe scores, displacement values, and some patch results. Their aggregation or seed is unspecified.

3. **“Gaps are cross-entropies … minus the exact Bayes reference”** — Negative realized gaps are possible, but the finite-sample meaning of “reference” is not explained. “Exact” can otherwise suggest a lower bound no measured loss can beat.

4. **“Mess3(α=.9, x=.1)”** — Mess3 is undefined. Its hidden states, alphabet, transition/emission convention, and parameter meanings are absent. The parameter x also reuses the name of the observed x-stream.

5. **“Mess3A(.85,.15) or Mess3B(.85,.5)”** — The positional parameters are unlabeled, and the substantive distinction between these processes is unstated. The reader cannot tell what evidence in y identifies the case.

6. **“x … the shared circuit”** — A generated data stream is called a circuit before the learned tracker has been distinguished from the process it tracks. Later uses alternate among process, belief representation, and computation.

7. **“the case” / “both cases”** — The per-sequence prior, sequence initialization, and fact that the case label is unavailable as an input are not stated together. These are necessary to interpret posterior uncertainty.

8. **“w_A = P(A | y-stream)”** — The conditioning should identify the observed prefix, rather than the entire sequence. The position-dependent role is only introduced later.

9. **“reaches 1% error by position ≈13”** — “Error” could mean classification error, mean wrong-case posterior mass, or a quantile. PREREG’s t≈13 figure refers to median wrong-case mass.

10. **“predict (x, y)”** — The token encoding, alphabet sizes, next-token target, joint-versus-separate output distribution, and extraction of x/y marginals are unstated.

11. **“the x-target is relabeled”** — It is not explicit that the next x symbol is relabeled according to the true sequence case, while y targets and observed prefixes remain unchanged.

12. **“by S₃ symmetry the Bayes reference given the case is unchanged”** — The relevant joint state/symbol symmetry is not defined. The distinction between the case-known reference and the uncertain-case “Bayes-gated” reference arrives without its formula.

13. **“batch halved so A-exposure per step matches”** — Actual batch sizes, random versus exact case balancing, loss normalization, and whether A examples are paired across arms are unspecified. These matter to the identical-trajectory interpretation.

14. **“20k steps” / “10k steps” / “50-step bend”** — Optimizer, learning rate, warmup, optimizer-state reset, and checkpoint spacing are absent. “Steps” alone does not define the optimization timescale.

15. **“context 64” / “positions 16–63” / “t=0”** — The indexing convention, BOS treatment, and number of observed tokens at each position are missing. A fresh reader cannot align predictions with oracle beliefs.

16. **“regressed onto the segment”** — The formula, pooling over histories, clipping convention, and order of averaging are absent. Values such as 1.001 conflict with a literal fit constrained to a segment.

17. **“f = 0 is pretrained behaviour”** — Zero denotes the oracle unflipped prediction, not necessarily the actual pretrained model. The shorthand conflates them.

18. **“f_expl … share … along that segment”** — Its precise denominator, aggregation, and handling of tiny or zero deviations are missing. Consequently, B’s very low values appear incompatible with both the claimed endpoint and the scorecard.

19. **“positions 16–63 (case resolved)”** — “Resolved” sounds categorical, although the posterior retains uncertainty. The relation between this window, t≈13, and t≈40 is unexplained.

20. **“2-dim encoder image … 5-dim image of the case belief”** — Neither belief vector is defined. The reader is not told that these are a three-state belief and a six-state joint case/state belief, with simplex dimensions two and five.

21. **“encoder image” / “decoder-subspace erasure”** — The two regression directions and the construction of their subspaces are not explained. The reader cannot understand why the interventions differ.

22. **“at a capture” / “at all captures” / “L1mlp”** — Capture sites are not defined as post-sublayer residual states. “All captures” is ambiguous between simultaneous interventions and results across separate single-site interventions.

23. **“projects … out of the residual stream”** — The source of the fitted subspace, centering convention, positions affected, and whether the same subspace is used across contexts and checkpoints are unstated.

24. **“transplants the case-subspace component”** — Donor/recipient notation, which positions are patched, and whether the entire five-dimensional case/state component is replaced are unclear. This is broader than replacing a scalar posterior.

25. **“pretrain-endpoint HMM1 probe” / “R²”** — Fitting method, held-out evaluation, temporal pooling, pooled-versus-context-specific fitting, and multivariate R² aggregation are absent. Later fresh-probe and transfer scores look contradictory without these distinctions.

26. **“σ-relabeled belief”** — σ was defined on observed x labels. Its corresponding action on hidden-state belief coordinates has not been stated.

27. **“x-CE_A … − Bayes-gated ref (0.852)”** — The gated predictor’s formula is missing. It is also unclear whether the displayed constants are actual references for this evaluation window or approximate values borrowed from another calculation.

28. **“y-CE_A, y-CE_B − ref | ≤ .0012 all”** — A one-sided inequality does not mean absolute closeness. The combined row also obscures the separate case values and which reference applies to each.

29. **“deployment-side y model rusts” / “its B branch”** — “Deployment” is an analogy, and “branch” sounds like a demonstrated architectural module. Neither is identified as such.

30. **“same path up the diagonal” / “peak f_B”** — Whether these describe every checkpoint, a smoothed curve, or representative values is unspecified. The quoted .4 and .29–.33 values make that ambiguity consequential.

31. **“B back within 0.01 … and staying there”** — The observation schedule and duration criterion are absent. A reader cannot distinguish continuous behavior from sampled-checkpoint behavior.

32. **“single-sequence flip fraction”** — Its formula, identifiability filter, sample count, and reason for showing only t=4–8 are missing. These differ materially from the pooled f statistic.

33. **“f_B(t) is the mirror image”** — It is unclear whether this means an exact complement, approximate visual symmetry, or simply decline toward zero.

34. **“pretrain baseline +.250/+.247; uniform is +.25”** — The slash unexpectedly denotes A/B rather than seeds. “Uniform” is stated as an excess loss without identifying its reference or task.

35. **“unembedding barely moved” / “final .03”** — The displacement norm’s denominator, included parameter tensors, weight tying, and relation between the “final” group and unembedding are unspecified.

36. **“upstream gating … §4(c)”** — The reference is apparently to PREREG, not RESULTS §4. “Upstream” also lacks an explicit reference point.

37. **“logit-w R²”** — The logit formula and clipping convention are missing. Clipping and position selection materially affect this target and its score.

38. **“the variable gains a consumer … loses one”** — “Consumer” is an inferred computational role, not an operationally defined measurement. The prose treats it as directly observed.

39. **“FWH overlap .02–.16”** — FWH is never expanded; the compared subspaces, overlap normalization, and scope of the range are unstated.

40. **“✓ joint CE … ✓ f … ✓ excursion”** — The scorecard presupposes knowledge of P1–P5’s full conjunctions, thresholds, and amendments. Its checkmarks cannot be independently interpreted from RESULTS alone.

41. **“eval-like situation” / “misbehaviour” / “quarantine training”** — These labels appear without explicitly mapping them to A, supervised target permutation, and training with retained B loss. They import project context and normative meaning into a synthetic prediction task.

42. **“its population optimum is the corrupted one”** — The essential support-overlap premise is missing: B histories can occur under A, so A-only training’s population optimum constrains predictions on those histories too.

43. **“a last-token predictor recovers 94% of the learnable signal”** — “Learnable signal” lacks a denominator, reference baseline, evaluation window, and supporting losses. “Nearly a two-token lookup” additionally invokes a different history length without its result.

44. **“below some share the arm behaves like the unanchored one”** — The prediction does not specify whether this concerns behavior after 10k steps, practical optimization failure, or a population-optimum transition.

45. **“KL-to-reference … closest to RLHF”** — The reference distribution, contexts over which KL is applied, KL direction, and intended connection to the existing explicit B loss are unspecified.
tokens used
25,346
**(A) ADVERSARIAL READ**

1. **“Case-conditional relabeling … tracker … and unembedding untouched”: the relabeling signature is supported; the exclusive circuit interpretation is not.** The old probe reads approximately original coordinates in B and approximately permuted coordinates in A after L3mlp. Early case patches strongly change the output; patches after L3mlp mostly do not. Together, these support case-dependent processing whose output is largely established by L3mlp.

   A separate A-specific pathway survives all these observations. It could consume shared early x-features and case information, then write a correction into the residual coordinates that the old probe and unembedding read. That would produce the transfer signature, early patch sensitivity, and relatively stable unembedding. Erasing shared input features would damage both pathways, so comparable ablation losses do not rule this out. The experiment does not distinguish an in-place permutation from a conditional branch that writes an equivalent result.

   “Blocks 1–2 unchanged” is particularly indefensible. Seed-0 original-coordinate transfer falls from approximately .94 at L1mlp to .793/.828 at L2mlp, and their weight displacements are substantial. Preserved decodability is not unchanged computation. Likewise, .18 relative unembedding displacement at 10k is not “untouched”; even .03 need not be functionally negligible. No frozen-unembedding or weight-restoration result establishes that its changes are irrelevant.

   “Exactly where w is available” also overstates localization. Seed-0 logit-w R² is already .871 at L3attn, and case patches work at L1mlp despite low linear logit-w decodability there. Availability to nonlinear computation is not the same as crossing a linear-probe threshold.

2. **“Identical until damaged, then a 50-step bend”: the coarse behavioral sequence is supported; the gradient-level explanation is not.** Similar initial movement is unsurprising from the same initialization with a common A objective and a pretrained B objective near its optimum. The expected B gradient can initially be small and then grow continuously as parameters move. That does not establish that it is “silent” until any particular measurable-damage threshold, or that no gate learning occurs before the visible bend.

   Nor are identical trajectories inevitable: the arms have different batches, loss mixtures, y objectives, and AdamW update histories. Matching A examples per step does not establish identical A-gradient contributions after averaging and optimization. The observed similarity therefore needs quantification, not a causal gloss.

   The document contradicts its own strongest wording. At step 75 it reports anchored B excess .055/.047/.057 versus unanchored .089/.109/.086. Those arms have already diverged in loss. Its “f_A ≈ f_B ≈ 0.4” also conflicts with anchored peak f_B of .32/.29/.33.

   “Cheapest reduction,” “only once,” and damage “set by how fast the anchor reacts” require evidence about gradients or controlled changes to anchor strength. The reported checkpoints support a rapid transition around steps 75–125, not an identified damage-triggered mechanism or an exactly 50-step construction time. A7 explicitly made the excursion a timestamp and conditioned stronger gate-building language on accompanying displacement evidence; that condition needs addressing.

3. **“Bayes-exact … per position and per history”: not established.** Position-averaged agreement plus high Spearman correlation demonstrates approximate posterior-related behavior. Spearman measures ordering, not calibration. For example,
   \[
   f(h)=\operatorname{sigmoid}\!\left(a\,\operatorname{logit}w_A(h)+b\right),\quad a>0,
   \]
   preserves posterior ranking perfectly while generally assigning the wrong mixture weights. A finite-history heuristic based on y-transition counts could also rank histories well without implementing the exact filter.

   Position means constrain these alternatives but do not eliminate them: modest miscalibration, history-dependent errors that cancel in aggregates, and small errors orthogonal to the flip direction can coexist with apparently matching curves. Moreover, A4’s pooled f is weighted by \(\|P^\sigma-P\|^2\), rather than automatically being an ordinary mean posterior.

   The per-history result concerns selected histories with sufficiently identifiable flips, at selected positions. RESULTS reports t=4–8, whereas A4 specifies t=3–12; seed-0 A correlations fall to .76 by t=12. That is incompatible with presenting the displayed range as comprehensive evidence of exactness.

   At t=0 the claim is worse: the uniform initial x prediction is invariant under σ, so f is unidentifiable. A4 explicitly masks it. “0.5 at t=0” is an oracle posterior value, not an identifiable model flip fraction.

   Direct calibration errors between f and w, together with prediction-vector errors against the oracle mixture on ambiguous histories, would support approximate Bayes optimality. These reported statistics do not establish exactness or an internally Bayesian computation.

4. **“Quarantine training improves … situational discrimination”: improved linear accessibility is supported; improved discrimination and its cause are not.** Higher fitted logit-w R² can result from the same case information being rescaled, reoriented, or made easier for a linear decoder to access because downstream computation uses it more. The original model may already distinguish cases equally well through a nonlinear representation.

   Actual improvement in discrimination would need a relevant behavioral or predictive measure—case classification error, posterior calibration, or a proper scoring rule—under comparable decoding conditions. A fitted probe’s R² alone does not supply that.

   “Because the situational variable acquires a new consumer” is a plausible explanation, not an identified cause. Anchored training changes the task and its case-dependent joint structure. Unanchored training also removes B data and changes the training prior. These are not isolated manipulations of the number of downstream consumers.

5. **The scorecard is too generous, although the chronology does not justify accusing the authors of moving thresholds after seeing finetune outcomes.** A1–A12 are stated to precede runs. A13 explicitly follows inspection of seed-0 pretraining, before finetuning: it replaces an inert decoder-space intervention with an effective encoder-image intervention, and replaces P5’s scalar-direction erasure criterion with a case-patch criterion. That is a data-informed methodological change and a changed pass criterion. It is disclosed, and the finetune prediction remains prospective under the stated chronology. It is not evidence that the .05 excursion threshold or other numeric outcome thresholds were moved after finetune results.

   Nevertheless, several checks are not earned:

   - **P1:** .86 pooled over t≥1 does not test R²>.9 at t≥8. RESULTS expressly says the latter was not computed. P1 is incomplete, not passed.
   - **P2:** the literal requirement includes f_expl>.8, but anchored B reports .03/.07/.03. Near f_B=0, this diagnostic can be unhelpful because the deviation being explained is tiny. That explains why the criterion may be poorly chosen; it does not make it pass. No amendment shown exempts B.
   - **P2 supporting evidence:** the A–B overlap >.8 criterion is omitted from the scorecard. Seed-0 overlap is .759 at L3mlp and .569 at L4mlp. The capture dependence should not disappear behind a general preservation verdict.
   - **P3:** A7 defines excursion relative to the step-0 gap. The scorecard supplies apparent absolute gaps from reference. With .055/.057 only narrowly exceeding .05, the starting-gap subtraction matters. The permitted probe logs cannot verify this.
   - **P5:** the original position-curve Spearman >.9 is not reported. A per-history correlation is a different statistic. The amended patch criterion passes comfortably, but it does not complete every other clause.

   The displacement prediction **is explicitly called false**, without redefining its threshold. However, the stronger theoretical conclusion is salvaged through transfer and patching, and “untouched” language survives despite the failed displacement prediction. So the metric failure is candidly admitted; its implications are not fully absorbed.

6. **Numerical audit: several overstatements and omissions, rather than wholesale transcription failure.** The principal seed-0 L3mlp transfer values, endpoint representation scores, ALL:HMM1 erasure losses, and L1/L2/L3 patch values match the logs. Specific problems are:

   - **“HMM1 belief stays R² ≥ .99 in every layer, arm and context”:** embeddings are .944 in both logs. If “layer” excludes embeddings and “context” excludes pooled fits, the separate A/B transformer-layer scores satisfy the claim; anchored L4mlp pooled R² is .988. The scope needs qualification.
   - **“Patch moves f_A by 97% … at L1–L2”:** using rounded seed-0 values, L1mlp gives \((.99-.03)/(.99-.01)\approx98\%\); L2mlp gives approximately94%. The latter is not 97%.
   - **“At L4mlp no effect”:** anchored patching gives A .99→1.01 and B .01→.04. Small effects, not literally zero.
   - **Unanchored “patch and erasure … (1.00→1.00)”:** ALL:case erasure gives 1.00→.98 in both contexts; some individual erasures give 1.01–1.02. Approximate insensitivity is supported.
   - **Unanchored relabeling “lands at block 1”:** old-probe original-coordinate transfer remains .868/.868 at L1mlp. From L2 onward both original and σ transfer are poor; σ transfer at L2mlp is −.130/−.160. These numbers locate substantial representational drift, not a clean block-1 permutation.
   - **“Mirror image”:** PREREG says true-B is recognized faster than true-A. Exact complementary *case-averaged* curves do not follow from the common posterior formula.
   - **The step-75 values and anchored f_B peaks conflict internally**, as discussed above.
   - The two permitted logs contain endpoint probes, not training loss trajectories, displacement histories, seed-1/2 outputs, or pretrain baselines. Those other numbers remain unchecked, not verified.

**(B) FRESH-READER CLARITY**

1. **“Self-inoculation toy, v1” / “For the story (†)”** — The title’s organizing concept and the referenced story are never explained locally. The reader must infer the intended analogy from later language about deployment and misbehavior.

2. **“3 seeds; every number below … per seed … unless marked ‘all’”** — Numerous single values are neither per-seed triples nor marked “all”: absolute CE, initial probe scores, displacement values, and some patch results. Their aggregation or seed is unspecified.

3. **“Gaps are cross-entropies … minus the exact Bayes reference”** — Negative realized gaps are possible, but the finite-sample meaning of “reference” is not explained. “Exact” can otherwise suggest a lower bound no measured loss can beat.

4. **“Mess3(α=.9, x=.1)”** — Mess3 is undefined. Its hidden states, alphabet, transition/emission convention, and parameter meanings are absent. The parameter x also reuses the name of the observed x-stream.

5. **“Mess3A(.85,.15) or Mess3B(.85,.5)”** — The positional parameters are unlabeled, and the substantive distinction between these processes is unstated. The reader cannot tell what evidence in y identifies the case.

6. **“x … the shared circuit”** — A generated data stream is called a circuit before the learned tracker has been distinguished from the process it tracks. Later uses alternate among process, belief representation, and computation.

7. **“the case” / “both cases”** — The per-sequence prior, sequence initialization, and fact that the case label is unavailable as an input are not stated together. These are necessary to interpret posterior uncertainty.

8. **“w_A = P(A | y-stream)”** — The conditioning should identify the observed prefix, rather than the entire sequence. The position-dependent role is only introduced later.

9. **“reaches 1% error by position ≈13”** — “Error” could mean classification error, mean wrong-case posterior mass, or a quantile. PREREG’s t≈13 figure refers to median wrong-case mass.

10. **“predict (x, y)”** — The token encoding, alphabet sizes, next-token target, joint-versus-separate output distribution, and extraction of x/y marginals are unstated.

11. **“the x-target is relabeled”** — It is not explicit that the next x symbol is relabeled according to the true sequence case, while y targets and observed prefixes remain unchanged.

12. **“by S₃ symmetry the Bayes reference given the case is unchanged”** — The relevant joint state/symbol symmetry is not defined. The distinction between the case-known reference and the uncertain-case “Bayes-gated” reference arrives without its formula.

13. **“batch halved so A-exposure per step matches”** — Actual batch sizes, random versus exact case balancing, loss normalization, and whether A examples are paired across arms are unspecified. These matter to the identical-trajectory interpretation.

14. **“20k steps” / “10k steps” / “50-step bend”** — Optimizer, learning rate, warmup, optimizer-state reset, and checkpoint spacing are absent. “Steps” alone does not define the optimization timescale.

15. **“context 64” / “positions 16–63” / “t=0”** — The indexing convention, BOS treatment, and number of observed tokens at each position are missing. A fresh reader cannot align predictions with oracle beliefs.

16. **“regressed onto the segment”** — The formula, pooling over histories, clipping convention, and order of averaging are absent. Values such as 1.001 conflict with a literal fit constrained to a segment.

17. **“f = 0 is pretrained behaviour”** — Zero denotes the oracle unflipped prediction, not necessarily the actual pretrained model. The shorthand conflates them.

18. **“f_expl … share … along that segment”** — Its precise denominator, aggregation, and handling of tiny or zero deviations are missing. Consequently, B’s very low values appear incompatible with both the claimed endpoint and the scorecard.

19. **“positions 16–63 (case resolved)”** — “Resolved” sounds categorical, although the posterior retains uncertainty. The relation between this window, t≈13, and t≈40 is unexplained.

20. **“2-dim encoder image … 5-dim image of the case belief”** — Neither belief vector is defined. The reader is not told that these are a three-state belief and a six-state joint case/state belief, with simplex dimensions two and five.

21. **“encoder image” / “decoder-subspace erasure”** — The two regression directions and the construction of their subspaces are not explained. The reader cannot understand why the interventions differ.

22. **“at a capture” / “at all captures” / “L1mlp”** — Capture sites are not defined as post-sublayer residual states. “All captures” is ambiguous between simultaneous interventions and results across separate single-site interventions.

23. **“projects … out of the residual stream”** — The source of the fitted subspace, centering convention, positions affected, and whether the same subspace is used across contexts and checkpoints are unstated.

24. **“transplants the case-subspace component”** — Donor/recipient notation, which positions are patched, and whether the entire five-dimensional case/state component is replaced are unclear. This is broader than replacing a scalar posterior.

25. **“pretrain-endpoint HMM1 probe” / “R²”** — Fitting method, held-out evaluation, temporal pooling, pooled-versus-context-specific fitting, and multivariate R² aggregation are absent. Later fresh-probe and transfer scores look contradictory without these distinctions.

26. **“σ-relabeled belief”** — σ was defined on observed x labels. Its corresponding action on hidden-state belief coordinates has not been stated.

27. **“x-CE_A … − Bayes-gated ref (0.852)”** — The gated predictor’s formula is missing. It is also unclear whether the displayed constants are actual references for this evaluation window or approximate values borrowed from another calculation.

28. **“y-CE_A, y-CE_B − ref | ≤ .0012 all”** — A one-sided inequality does not mean absolute closeness. The combined row also obscures the separate case values and which reference applies to each.

29. **“deployment-side y model rusts” / “its B branch”** — “Deployment” is an analogy, and “branch” sounds like a demonstrated architectural module. Neither is identified as such.

30. **“same path up the diagonal” / “peak f_B”** — Whether these describe every checkpoint, a smoothed curve, or representative values is unspecified. The quoted .4 and .29–.33 values make that ambiguity consequential.

31. **“B back within 0.01 … and staying there”** — The observation schedule and duration criterion are absent. A reader cannot distinguish continuous behavior from sampled-checkpoint behavior.

32. **“single-sequence flip fraction”** — Its formula, identifiability filter, sample count, and reason for showing only t=4–8 are missing. These differ materially from the pooled f statistic.

33. **“f_B(t) is the mirror image”** — It is unclear whether this means an exact complement, approximate visual symmetry, or simply decline toward zero.

34. **“pretrain baseline +.250/+.247; uniform is +.25”** — The slash unexpectedly denotes A/B rather than seeds. “Uniform” is stated as an excess loss without identifying its reference or task.

35. **“unembedding barely moved” / “final .03”** — The displacement norm’s denominator, included parameter tensors, weight tying, and relation between the “final” group and unembedding are unspecified.

36. **“upstream gating … §4(c)”** — The reference is apparently to PREREG, not RESULTS §4. “Upstream” also lacks an explicit reference point.

37. **“logit-w R²”** — The logit formula and clipping convention are missing. Clipping and position selection materially affect this target and its score.

38. **“the variable gains a consumer … loses one”** — “Consumer” is an inferred computational role, not an operationally defined measurement. The prose treats it as directly observed.

39. **“FWH overlap .02–.16”** — FWH is never expanded; the compared subspaces, overlap normalization, and scope of the range are unstated.

40. **“✓ joint CE … ✓ f … ✓ excursion”** — The scorecard presupposes knowledge of P1–P5’s full conjunctions, thresholds, and amendments. Its checkmarks cannot be independently interpreted from RESULTS alone.

41. **“eval-like situation” / “misbehaviour” / “quarantine training”** — These labels appear without explicitly mapping them to A, supervised target permutation, and training with retained B loss. They import project context and normative meaning into a synthetic prediction task.

42. **“its population optimum is the corrupted one”** — The essential support-overlap premise is missing: B histories can occur under A, so A-only training’s population optimum constrains predictions on those histories too.

43. **“a last-token predictor recovers 94% of the learnable signal”** — “Learnable signal” lacks a denominator, reference baseline, evaluation window, and supporting losses. “Nearly a two-token lookup” additionally invokes a different history length without its result.

44. **“below some share the arm behaves like the unanchored one”** — The prediction does not specify whether this concerns behavior after 10k steps, practical optimization failure, or a population-optimum transition.

45. **“KL-to-reference … closest to RLHF”** — The reference distribution, contexts over which KL is applied, KL direction, and intended connection to the existing explicit B loss are unspecified.

[s0] restoration (t>=16):  f_A   f_B  | gap_A(flipped task)  gap_B(unflipped task)
  anchored (none restored)               0.989 0.014 | -0.000  +0.001
  restore embed -> pretrain              0.989 0.015 | -0.000  +0.001
  restore blocks1-2 -> pretrain          0.926 0.120 | +0.004  +0.013
  restore block3 -> pretrain             0.439 0.106 | +0.154  +0.007
  restore block4 -> pretrain             0.962 0.035 | +0.003  +0.002
  restore blocks3-4 -> pretrain          0.206 0.152 | +0.288  +0.010
  restore final(ln_f+unembed) -> pretrain 0.961 -0.008 | +0.003  +0.002
  pretrain + finetuned embed only        0.005 0.006 | +0.494  -0.003
  pretrain + finetuned blocks1-2 only    0.112 0.093 | +0.379  +0.003
  pretrain + finetuned block3 only       0.788 0.088 | +0.034  +0.006
  pretrain + finetuned block4 only       0.090 0.020 | +0.397  -0.001
  pretrain + finetuned blocks3-4 only    0.884 0.082 | +0.010  +0.007
  pretrain + finetuned final(ln_f+unembed) only 0.067 0.065 | +0.419  +0.001
[s0] per-history Spearman(f_n, w_A,n), pooled A+B contexts:
  t3:+0.90  t4:+0.94  t5:+0.96  t6:+0.97  t7:+0.97  t8:+0.98  t9:+0.97  t10:+0.97  t11:+0.96  t12:+0.95
[s0] calibration pooled t=3..12: slope 0.976 intercept +0.024 mean|f-w| 0.033 (clipped f: 0.032); n=30080
  bin        n     mean w   mean f   mean clip(f)
  [0.0,0.1)   7735   0.035    0.043    0.046
  [0.1,0.3)   5203   0.188    0.226    0.226
  [0.3,0.5)   2121   0.407    0.448    0.448
  [0.5,0.7)   3254   0.630    0.644    0.644
  [0.7,0.9)   3830   0.828    0.832    0.832
  [0.9,1.0)   7937   0.973    0.968    0.965
[s0] position curve t=1..20: Spearman(f_A(t), mean w_A(t)) = +1.000, mean|f_A(t)-w_A(t)| = 0.005

[s1] restoration (t>=16):  f_A   f_B  | gap_A(flipped task)  gap_B(unflipped task)
  anchored (none restored)               0.974 0.019 | +0.001  -0.000
  restore embed -> pretrain              0.975 0.019 | +0.001  -0.000
  restore blocks1-2 -> pretrain          0.938 0.033 | +0.003  +0.001
  restore block3 -> pretrain             0.419 0.071 | +0.155  +0.001
  restore block4 -> pretrain             0.920 0.035 | +0.006  +0.000
  restore blocks3-4 -> pretrain          0.185 0.117 | +0.308  +0.005
  restore final(ln_f+unembed) -> pretrain 0.911 -0.005 | +0.010  +0.001
  pretrain + finetuned embed only        -0.004 -0.004 | +0.507  -0.003
  pretrain + finetuned blocks1-2 only    0.079 0.037 | +0.411  -0.002
  pretrain + finetuned block3 only       0.794 0.003 | +0.032  +0.000
  pretrain + finetuned block4 only       0.109 0.008 | +0.379  -0.001
  pretrain + finetuned blocks3-4 only    0.872 0.006 | +0.013  +0.001
  pretrain + finetuned final(ln_f+unembed) only 0.078 0.072 | +0.411  +0.001
[s1] per-history Spearman(f_n, w_A,n), pooled A+B contexts:
  t3:+0.90  t4:+0.94  t5:+0.96  t6:+0.97  t7:+0.98  t8:+0.98  t9:+0.97  t10:+0.97  t11:+0.96  t12:+0.95
[s1] calibration pooled t=3..12: slope 0.964 intercept +0.009 mean|f-w| 0.028 (clipped f: 0.028); n=30080
  bin        n     mean w   mean f   mean clip(f)
  [0.0,0.1)   7735   0.035    0.046    0.047
  [0.1,0.3)   5203   0.188    0.192    0.192
  [0.3,0.5)   2121   0.407    0.390    0.390
  [0.5,0.7)   3254   0.630    0.603    0.603
  [0.7,0.9)   3830   0.828    0.803    0.803
  [0.9,1.0)   7937   0.973    0.953    0.952
[s1] position curve t=1..20: Spearman(f_A(t), mean w_A(t)) = +1.000, mean|f_A(t)-w_A(t)| = 0.019

[s2] restoration (t>=16):  f_A   f_B  | gap_A(flipped task)  gap_B(unflipped task)
  anchored (none restored)               0.987 0.013 | -0.000  +0.001
  restore embed -> pretrain              0.987 0.014 | -0.000  +0.001
  restore blocks1-2 -> pretrain          0.875 0.066 | +0.015  +0.003
  restore block3 -> pretrain             0.510 0.057 | +0.116  +0.000
  restore block4 -> pretrain             0.945 0.057 | +0.002  +0.002
  restore blocks3-4 -> pretrain          0.276 0.141 | +0.242  +0.009
  restore final(ln_f+unembed) -> pretrain 0.997 -0.018 | +0.001  +0.002
  pretrain + finetuned embed only        0.004 0.006 | +0.497  -0.003
  pretrain + finetuned blocks1-2 only    0.199 0.096 | +0.302  +0.003
  pretrain + finetuned block3 only       0.707 0.060 | +0.059  +0.002
  pretrain + finetuned block4 only       0.046 -0.021 | +0.444  -0.002
  pretrain + finetuned blocks3-4 only    0.861 0.028 | +0.019  +0.001
  pretrain + finetuned final(ln_f+unembed) only 0.057 0.055 | +0.431  -0.001
[s2] per-history Spearman(f_n, w_A,n), pooled A+B contexts:
  t3:+0.89  t4:+0.94  t5:+0.96  t6:+0.97  t7:+0.97  t8:+0.97  t9:+0.97  t10:+0.97  t11:+0.96  t12:+0.95
[s2] calibration pooled t=3..12: slope 1.001 intercept +0.009 mean|f-w| 0.026 (clipped f: 0.025); n=30080
  bin        n     mean w   mean f   mean clip(f)
  [0.0,0.1)   7735   0.035    0.038    0.039
  [0.1,0.3)   5203   0.188    0.199    0.199
  [0.3,0.5)   2121   0.407    0.429    0.429
  [0.5,0.7)   3254   0.630    0.654    0.654
  [0.7,0.9)   3830   0.828    0.849    0.849
  [0.9,1.0)   7937   0.973    0.974    0.971
[s2] position curve t=1..20: Spearman(f_A(t), mean w_A(t)) = +1.000, mean|f_A(t)-w_A(t)| = 0.006

[P1] pretrain s0, L4mlp, logit-w ridge R^2 restricted to t>=8: 0.896

exec
/usr/bin/bash -lc "rg -n '"'^'"\\*\\*\\(A\\)|"'^1'"\\. \\*\\*|"'^2'"\\. \\*\\*|"'^3'"\\. \\*\\*|"'^4'"\\. \\*\\*|"'^5'"\\. \\*\\*|"'^6'"\\. \\*\\*' logs/codex_results_review.md" in /data/users/asvin/self_inoculation
 succeeded in 0ms:
624:**(A) ADVERSARIAL READ**
626:1. **“Case-conditional relabeling … tracker … and unembedding untouched”: the relabeling signature is supported; the exclusive circuit interpretation is not.** The old probe reads approximately original coordinates in B and approximately permuted coordinates in A after L3mlp. Early case patches strongly change the output; patches after L3mlp mostly do not. Together, these support case-dependent processing whose output is largely established by L3mlp.
634:2. **“Identical until damaged, then a 50-step bend”: the coarse behavioral sequence is supported; the gradient-level explanation is not.** Similar initial movement is unsurprising from the same initialization with a common A objective and a pretrained B objective near its optimum. The expected B gradient can initially be small and then grow continuously as parameters move. That does not establish that it is “silent” until any particular measurable-damage threshold, or that no gate learning occurs before the visible bend.
642:3. **“Bayes-exact … per position and per history”: not established.** Position-averaged agreement plus high Spearman correlation demonstrates approximate posterior-related behavior. Spearman measures ordering, not calibration. For example,
656:4. **“Quarantine training improves … situational discrimination”: improved linear accessibility is supported; improved discrimination and its cause are not.** Higher fitted logit-w R² can result from the same case information being rescaled, reoriented, or made easier for a linear decoder to access because downstream computation uses it more. The original model may already distinguish cases equally well through a nonlinear representation.
662:5. **The scorecard is too generous, although the chronology does not justify accusing the authors of moving thresholds after seeing finetune outcomes.** A1–A12 are stated to precede runs. A13 explicitly follows inspection of seed-0 pretraining, before finetuning: it replaces an inert decoder-space intervention with an effective encoder-image intervention, and replaces P5’s scalar-direction erasure criterion with a case-patch criterion. That is a data-informed methodological change and a changed pass criterion. It is disclosed, and the finetune prediction remains prospective under the stated chronology. It is not evidence that the .05 excursion threshold or other numeric outcome thresholds were moved after finetune results.
674:6. **Numerical audit: several overstatements and omissions, rather than wholesale transcription failure.** The principal seed-0 L3mlp transfer values, endpoint representation scores, ALL:HMM1 erasure losses, and L1/L2/L3 patch values match the logs. Specific problems are:
687:1. **“Self-inoculation toy, v1” / “For the story (†)”** — The title’s organizing concept and the referenced story are never explained locally. The reader must infer the intended analogy from later language about deployment and misbehavior.
689:2. **“3 seeds; every number below … per seed … unless marked ‘all’”** — Numerous single values are neither per-seed triples nor marked “all”: absolute CE, initial probe scores, displacement values, and some patch results. Their aggregation or seed is unspecified.
691:3. **“Gaps are cross-entropies … minus the exact Bayes reference”** — Negative realized gaps are possible, but the finite-sample meaning of “reference” is not explained. “Exact” can otherwise suggest a lower bound no measured loss can beat.
693:4. **“Mess3(α=.9, x=.1)”** — Mess3 is undefined. Its hidden states, alphabet, transition/emission convention, and parameter meanings are absent. The parameter x also reuses the name of the observed x-stream.
695:5. **“Mess3A(.85,.15) or Mess3B(.85,.5)”** — The positional parameters are unlabeled, and the substantive distinction between these processes is unstated. The reader cannot tell what evidence in y identifies the case.
697:6. **“x … the shared circuit”** — A generated data stream is called a circuit before the learned tracker has been distinguished from the process it tracks. Later uses alternate among process, belief representation, and computation.
778:**(A) ADVERSARIAL READ**
780:1. **“Case-conditional relabeling … tracker … and unembedding untouched”: the relabeling signature is supported; the exclusive circuit interpretation is not.** The old probe reads approximately original coordinates in B and approximately permuted coordinates in A after L3mlp. Early case patches strongly change the output; patches after L3mlp mostly do not. Together, these support case-dependent processing whose output is largely established by L3mlp.
788:2. **“Identical until damaged, then a 50-step bend”: the coarse behavioral sequence is supported; the gradient-level explanation is not.** Similar initial movement is unsurprising from the same initialization with a common A objective and a pretrained B objective near its optimum. The expected B gradient can initially be small and then grow continuously as parameters move. That does not establish that it is “silent” until any particular measurable-damage threshold, or that no gate learning occurs before the visible bend.
796:3. **“Bayes-exact … per position and per history”: not established.** Position-averaged agreement plus high Spearman correlation demonstrates approximate posterior-related behavior. Spearman measures ordering, not calibration. For example,
810:4. **“Quarantine training improves … situational discrimination”: improved linear accessibility is supported; improved discrimination and its cause are not.** Higher fitted logit-w R² can result from the same case information being rescaled, reoriented, or made easier for a linear decoder to access because downstream computation uses it more. The original model may already distinguish cases equally well through a nonlinear representation.
816:5. **The scorecard is too generous, although the chronology does not justify accusing the authors of moving thresholds after seeing finetune outcomes.** A1–A12 are stated to precede runs. A13 explicitly follows inspection of seed-0 pretraining, before finetuning: it replaces an inert decoder-space intervention with an effective encoder-image intervention, and replaces P5’s scalar-direction erasure criterion with a case-patch criterion. That is a data-informed methodological change and a changed pass criterion. It is disclosed, and the finetune prediction remains prospective under the stated chronology. It is not evidence that the .05 excursion threshold or other numeric outcome thresholds were moved after finetune results.
828:6. **Numerical audit: several overstatements and omissions, rather than wholesale transcription failure.** The principal seed-0 L3mlp transfer values, endpoint representation scores, ALL:HMM1 erasure losses, and L1/L2/L3 patch values match the logs. Specific problems are:
841:1. **“Self-inoculation toy, v1” / “For the story (†)”** — The title’s organizing concept and the referenced story are never explained locally. The reader must infer the intended analogy from later language about deployment and misbehavior.
843:2. **“3 seeds; every number below … per seed … unless marked ‘all’”** — Numerous single values are neither per-seed triples nor marked “all”: absolute CE, initial probe scores, displacement values, and some patch results. Their aggregation or seed is unspecified.
845:3. **“Gaps are cross-entropies … minus the exact Bayes reference”** — Negative realized gaps are possible, but the finite-sample meaning of “reference” is not explained. “Exact” can otherwise suggest a lower bound no measured loss can beat.
847:4. **“Mess3(α=.9, x=.1)”** — Mess3 is undefined. Its hidden states, alphabet, transition/emission convention, and parameter meanings are absent. The parameter x also reuses the name of the observed x-stream.
849:5. **“Mess3A(.85,.15) or Mess3B(.85,.5)”** — The positional parameters are unlabeled, and the substantive distinction between these processes is unstated. The reader cannot tell what evidence in y identifies the case.
851:6. **“x … the shared circuit”** — A generated data stream is called a circuit before the learned tracker has been distinguished from the process it tracks. Later uses alternate among process, belief representation, and computation.

exec
/usr/bin/bash -lc "sed -n '624,685p' logs/codex_results_review.md" in /data/users/asvin/self_inoculation
 succeeded in 0ms:
**(A) ADVERSARIAL READ**

1. **“Case-conditional relabeling … tracker … and unembedding untouched”: the relabeling signature is supported; the exclusive circuit interpretation is not.** The old probe reads approximately original coordinates in B and approximately permuted coordinates in A after L3mlp. Early case patches strongly change the output; patches after L3mlp mostly do not. Together, these support case-dependent processing whose output is largely established by L3mlp.

   A separate A-specific pathway survives all these observations. It could consume shared early x-features and case information, then write a correction into the residual coordinates that the old probe and unembedding read. That would produce the transfer signature, early patch sensitivity, and relatively stable unembedding. Erasing shared input features would damage both pathways, so comparable ablation losses do not rule this out. The experiment does not distinguish an in-place permutation from a conditional branch that writes an equivalent result.

   “Blocks 1–2 unchanged” is particularly indefensible. Seed-0 original-coordinate transfer falls from approximately .94 at L1mlp to .793/.828 at L2mlp, and their weight displacements are substantial. Preserved decodability is not unchanged computation. Likewise, .18 relative unembedding displacement at 10k is not “untouched”; even .03 need not be functionally negligible. No frozen-unembedding or weight-restoration result establishes that its changes are irrelevant.

   “Exactly where w is available” also overstates localization. Seed-0 logit-w R² is already .871 at L3attn, and case patches work at L1mlp despite low linear logit-w decodability there. Availability to nonlinear computation is not the same as crossing a linear-probe threshold.

2. **“Identical until damaged, then a 50-step bend”: the coarse behavioral sequence is supported; the gradient-level explanation is not.** Similar initial movement is unsurprising from the same initialization with a common A objective and a pretrained B objective near its optimum. The expected B gradient can initially be small and then grow continuously as parameters move. That does not establish that it is “silent” until any particular measurable-damage threshold, or that no gate learning occurs before the visible bend.

   Nor are identical trajectories inevitable: the arms have different batches, loss mixtures, y objectives, and AdamW update histories. Matching A examples per step does not establish identical A-gradient contributions after averaging and optimization. The observed similarity therefore needs quantification, not a causal gloss.

   The document contradicts its own strongest wording. At step 75 it reports anchored B excess .055/.047/.057 versus unanchored .089/.109/.086. Those arms have already diverged in loss. Its “f_A ≈ f_B ≈ 0.4” also conflicts with anchored peak f_B of .32/.29/.33.

   “Cheapest reduction,” “only once,” and damage “set by how fast the anchor reacts” require evidence about gradients or controlled changes to anchor strength. The reported checkpoints support a rapid transition around steps 75–125, not an identified damage-triggered mechanism or an exactly 50-step construction time. A7 explicitly made the excursion a timestamp and conditioned stronger gate-building language on accompanying displacement evidence; that condition needs addressing.

3. **“Bayes-exact … per position and per history”: not established.** Position-averaged agreement plus high Spearman correlation demonstrates approximate posterior-related behavior. Spearman measures ordering, not calibration. For example,
   \[
   f(h)=\operatorname{sigmoid}\!\left(a\,\operatorname{logit}w_A(h)+b\right),\quad a>0,
   \]
   preserves posterior ranking perfectly while generally assigning the wrong mixture weights. A finite-history heuristic based on y-transition counts could also rank histories well without implementing the exact filter.

   Position means constrain these alternatives but do not eliminate them: modest miscalibration, history-dependent errors that cancel in aggregates, and small errors orthogonal to the flip direction can coexist with apparently matching curves. Moreover, A4’s pooled f is weighted by \(\|P^\sigma-P\|^2\), rather than automatically being an ordinary mean posterior.

   The per-history result concerns selected histories with sufficiently identifiable flips, at selected positions. RESULTS reports t=4–8, whereas A4 specifies t=3–12; seed-0 A correlations fall to .76 by t=12. That is incompatible with presenting the displayed range as comprehensive evidence of exactness.

   At t=0 the claim is worse: the uniform initial x prediction is invariant under σ, so f is unidentifiable. A4 explicitly masks it. “0.5 at t=0” is an oracle posterior value, not an identifiable model flip fraction.

   Direct calibration errors between f and w, together with prediction-vector errors against the oracle mixture on ambiguous histories, would support approximate Bayes optimality. These reported statistics do not establish exactness or an internally Bayesian computation.

4. **“Quarantine training improves … situational discrimination”: improved linear accessibility is supported; improved discrimination and its cause are not.** Higher fitted logit-w R² can result from the same case information being rescaled, reoriented, or made easier for a linear decoder to access because downstream computation uses it more. The original model may already distinguish cases equally well through a nonlinear representation.

   Actual improvement in discrimination would need a relevant behavioral or predictive measure—case classification error, posterior calibration, or a proper scoring rule—under comparable decoding conditions. A fitted probe’s R² alone does not supply that.

   “Because the situational variable acquires a new consumer” is a plausible explanation, not an identified cause. Anchored training changes the task and its case-dependent joint structure. Unanchored training also removes B data and changes the training prior. These are not isolated manipulations of the number of downstream consumers.

5. **The scorecard is too generous, although the chronology does not justify accusing the authors of moving thresholds after seeing finetune outcomes.** A1–A12 are stated to precede runs. A13 explicitly follows inspection of seed-0 pretraining, before finetuning: it replaces an inert decoder-space intervention with an effective encoder-image intervention, and replaces P5’s scalar-direction erasure criterion with a case-patch criterion. That is a data-informed methodological change and a changed pass criterion. It is disclosed, and the finetune prediction remains prospective under the stated chronology. It is not evidence that the .05 excursion threshold or other numeric outcome thresholds were moved after finetune results.

   Nevertheless, several checks are not earned:

   - **P1:** .86 pooled over t≥1 does not test R²>.9 at t≥8. RESULTS expressly says the latter was not computed. P1 is incomplete, not passed.
   - **P2:** the literal requirement includes f_expl>.8, but anchored B reports .03/.07/.03. Near f_B=0, this diagnostic can be unhelpful because the deviation being explained is tiny. That explains why the criterion may be poorly chosen; it does not make it pass. No amendment shown exempts B.
   - **P2 supporting evidence:** the A–B overlap >.8 criterion is omitted from the scorecard. Seed-0 overlap is .759 at L3mlp and .569 at L4mlp. The capture dependence should not disappear behind a general preservation verdict.
   - **P3:** A7 defines excursion relative to the step-0 gap. The scorecard supplies apparent absolute gaps from reference. With .055/.057 only narrowly exceeding .05, the starting-gap subtraction matters. The permitted probe logs cannot verify this.
   - **P5:** the original position-curve Spearman >.9 is not reported. A per-history correlation is a different statistic. The amended patch criterion passes comfortably, but it does not complete every other clause.

   The displacement prediction **is explicitly called false**, without redefining its threshold. However, the stronger theoretical conclusion is salvaged through transfer and patching, and “untouched” language survives despite the failed displacement prediction. So the metric failure is candidly admitted; its implications are not fully absorbed.

6. **Numerical audit: several overstatements and omissions, rather than wholesale transcription failure.** The principal seed-0 L3mlp transfer values, endpoint representation scores, ALL:HMM1 erasure losses, and L1/L2/L3 patch values match the logs. Specific problems are:

   - **“HMM1 belief stays R² ≥ .99 in every layer, arm and context”:** embeddings are .944 in both logs. If “layer” excludes embeddings and “context” excludes pooled fits, the separate A/B transformer-layer scores satisfy the claim; anchored L4mlp pooled R² is .988. The scope needs qualification.
   - **“Patch moves f_A by 97% … at L1–L2”:** using rounded seed-0 values, L1mlp gives \((.99-.03)/(.99-.01)\approx98\%\); L2mlp gives approximately94%. The latter is not 97%.
   - **“At L4mlp no effect”:** anchored patching gives A .99→1.01 and B .01→.04. Small effects, not literally zero.
   - **Unanchored “patch and erasure … (1.00→1.00)”:** ALL:case erasure gives 1.00→.98 in both contexts; some individual erasures give 1.01–1.02. Approximate insensitivity is supported.
   - **Unanchored relabeling “lands at block 1”:** old-probe original-coordinate transfer remains .868/.868 at L1mlp. From L2 onward both original and σ transfer are poor; σ transfer at L2mlp is −.130/−.160. These numbers locate substantial representational drift, not a clean block-1 permutation.
   - **“Mirror image”:** PREREG says true-B is recognized faster than true-A. Exact complementary *case-averaged* curves do not follow from the common posterior formula.
   - **The step-75 values and anchored f_B peaks conflict internally**, as discussed above.
   - The two permitted logs contain endpoint probes, not training loss trajectories, displacement histories, seed-1/2 outputs, or pretrain baselines. Those other numbers remain unchecked, not verified.

**(B) FRESH-READER CLARITY**

codex
1. **Earlier A1 — Still over-claims.** Restoration supports a dominant contribution from blocks 3–4, and §2.4 now explicitly admits the conditional-branch alternative, but §0 still presents “the shared belief’s coordinates are σ-relabeled” as the identified mechanism. “The unchanged unembedding” and “functionally irrelevant” also exceed the evidence: restoring the final group changes seed-1 f_A from .974 to .911, and that intervention bundles unembedding with ln_f.

2. **Earlier A2 — Behavioral sequence supported; causal timing still over-claimed.** §2.2 quantifies the early similarity and labels the gradient explanation as interpretation, but §0’s “only as the protected context becomes measurably damaged does it build a gate” and §4’s “only as the protected situation is damaged” turn that interpretation back into a finding. Checkpoint trajectories do not establish when gate learning starts, a damage threshold, or when the B-gradient “dominates.”

3. **Earlier A3 — Approximate scalar calibration now supported; stronger claims remain unsupported.** The new per-history errors address the earlier rank-versus-calibration objection and exclude a purely positional schedule on the tested histories, but “its current, per-history, calibrated belief” attributes an internal belief computation that output agreement does not identify. §4’s “an ambiguous context receives the calibrated mixture” additionally requires checking the full prediction vector, including errors orthogonal to the flip direction.

4. **Earlier A4 — The narrowed claim is supported.** §2.5 now claims increased linear readability, explicitly distinguishes it from discrimination accuracy, and labels the downstream-consumer account as a confounded interpretation; §4 likewise limits the corollary to linear accessibility.

5. **Earlier A5 — Several scorecard defects are addressed, but P1 remains incomplete.** The revision reports the literal f_expl failure, layer-dependent overlap failure, displacement failure, position Spearman, and baseline subtraction; however, “.896 vs threshold .9 (borderline; s0 only)” is a measured failure for s0 with the other seeds untested, so the stated two-of-three rule cannot establish that clause. The new calibration evidence is explicitly post hoc and cannot count as a fulfilled preregistered prediction.

6. **Earlier A6 — The previously identified textual numerical overstatements are mostly corrected.** The revision qualifies embedding versus transformer captures, small patch/erasure effects, unanchored drift, and the step-75 divergence; the new restoration figures substantially match `logs/restore_calib.log`, but that log does not verify every training-trajectory or probe claim.

7. **New — §0’s “reconstructs 86–88% of the gate” substitutes f_A for gate selectivity.** The grafted A–B contrasts are .802/.866/.833, versus intact .975/.955/.974 in the restoration log: approximately **82%/91%/86%**, not 86–88%. The quoted range is the grafted f_A itself, without subtracting B leakage or normalizing to the intact gate.

8. **New — §2.4’s “the flip disappears rather than becoming unconditional” conflates loss of selectivity with loss of flipping.** Restoring blocks 3–4 leaves f_A=.206/.185/.276 and f_B=.152/.117/.141: a substantial residual flip remains, although the A–B contrast largely collapses.

9. **New — §2.4’s “changes to blocks 1–2 contribute a small leak-suppression” assigns an exclusive function the intervention does not establish.** Restoring those blocks also lowers seed-2 f_A from .987 to .875 and raises its A-task gap to .015; their measured contribution includes maintaining A flipping, not just suppressing B leakage.

10. **New — §2.4’s “necessary and nearly sufficient” lacks an operational success threshold.** Grafting blocks 3–4 yields f_A=.861–.884, below the document’s own endpoint criterion f_A>.9 in every seed; it supports substantial reconstruction, while group restoration establishes dependence on those changes in the tested background rather than a uniquely localized circuit.

11. **New — §2.3’s “within .04 in every posterior bin” is numerically false as written.** The seed-0 [.3,.5) bin reports mean w=.407 and mean f=.448, a difference of **.041**.

12. **New — §1’s “f = 0 reproduces P, f = 1 reproduces P^σ” is false for the defined projection statistic.** Either value permits a nonzero prediction error orthogonal to D; this is precisely why scalar calibration alone cannot establish the full-mixture claim in §4.

13. **Remaining localization over-claim — §2.4’s “the conditional relabeling sits where it becomes available” still equates linear readability with computational availability.** Effective case patches at L1mlp already show usable case-related information upstream, and §4 strengthens the unsupported assertion to “where the posterior first becomes available.”

14. **New internal definition inconsistency — §1 defines x_leave as “the probability of leaving the state” but describes x_leave=.5 as “always leaves.”** Those statements cannot both hold under the stated parameter definition.
tokens used
20,219
1. **Earlier A1 — Still over-claims.** Restoration supports a dominant contribution from blocks 3–4, and §2.4 now explicitly admits the conditional-branch alternative, but §0 still presents “the shared belief’s coordinates are σ-relabeled” as the identified mechanism. “The unchanged unembedding” and “functionally irrelevant” also exceed the evidence: restoring the final group changes seed-1 f_A from .974 to .911, and that intervention bundles unembedding with ln_f.

2. **Earlier A2 — Behavioral sequence supported; causal timing still over-claimed.** §2.2 quantifies the early similarity and labels the gradient explanation as interpretation, but §0’s “only as the protected context becomes measurably damaged does it build a gate” and §4’s “only as the protected situation is damaged” turn that interpretation back into a finding. Checkpoint trajectories do not establish when gate learning starts, a damage threshold, or when the B-gradient “dominates.”

3. **Earlier A3 — Approximate scalar calibration now supported; stronger claims remain unsupported.** The new per-history errors address the earlier rank-versus-calibration objection and exclude a purely positional schedule on the tested histories, but “its current, per-history, calibrated belief” attributes an internal belief computation that output agreement does not identify. §4’s “an ambiguous context receives the calibrated mixture” additionally requires checking the full prediction vector, including errors orthogonal to the flip direction.

4. **Earlier A4 — The narrowed claim is supported.** §2.5 now claims increased linear readability, explicitly distinguishes it from discrimination accuracy, and labels the downstream-consumer account as a confounded interpretation; §4 likewise limits the corollary to linear accessibility.

5. **Earlier A5 — Several scorecard defects are addressed, but P1 remains incomplete.** The revision reports the literal f_expl failure, layer-dependent overlap failure, displacement failure, position Spearman, and baseline subtraction; however, “.896 vs threshold .9 (borderline; s0 only)” is a measured failure for s0 with the other seeds untested, so the stated two-of-three rule cannot establish that clause. The new calibration evidence is explicitly post hoc and cannot count as a fulfilled preregistered prediction.

6. **Earlier A6 — The previously identified textual numerical overstatements are mostly corrected.** The revision qualifies embedding versus transformer captures, small patch/erasure effects, unanchored drift, and the step-75 divergence; the new restoration figures substantially match `logs/restore_calib.log`, but that log does not verify every training-trajectory or probe claim.

7. **New — §0’s “reconstructs 86–88% of the gate” substitutes f_A for gate selectivity.** The grafted A–B contrasts are .802/.866/.833, versus intact .975/.955/.974 in the restoration log: approximately **82%/91%/86%**, not 86–88%. The quoted range is the grafted f_A itself, without subtracting B leakage or normalizing to the intact gate.

8. **New — §2.4’s “the flip disappears rather than becoming unconditional” conflates loss of selectivity with loss of flipping.** Restoring blocks 3–4 leaves f_A=.206/.185/.276 and f_B=.152/.117/.141: a substantial residual flip remains, although the A–B contrast largely collapses.

9. **New — §2.4’s “changes to blocks 1–2 contribute a small leak-suppression” assigns an exclusive function the intervention does not establish.** Restoring those blocks also lowers seed-2 f_A from .987 to .875 and raises its A-task gap to .015; their measured contribution includes maintaining A flipping, not just suppressing B leakage.

10. **New — §2.4’s “necessary and nearly sufficient” lacks an operational success threshold.** Grafting blocks 3–4 yields f_A=.861–.884, below the document’s own endpoint criterion f_A>.9 in every seed; it supports substantial reconstruction, while group restoration establishes dependence on those changes in the tested background rather than a uniquely localized circuit.

11. **New — §2.3’s “within .04 in every posterior bin” is numerically false as written.** The seed-0 [.3,.5) bin reports mean w=.407 and mean f=.448, a difference of **.041**.

12. **New — §1’s “f = 0 reproduces P, f = 1 reproduces P^σ” is false for the defined projection statistic.** Either value permits a nonzero prediction error orthogonal to D; this is precisely why scalar calibration alone cannot establish the full-mixture claim in §4.

13. **Remaining localization over-claim — §2.4’s “the conditional relabeling sits where it becomes available” still equates linear readability with computational availability.** Effective case patches at L1mlp already show usable case-related information upstream, and §4 strengthens the unsupported assertion to “where the posterior first becomes available.”

14. **New internal definition inconsistency — §1 defines x_leave as “the probability of leaving the state” but describes x_leave=.5 as “always leaves.”** Those statements cannot both hold under the stated parameter definition.
