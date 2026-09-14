# RESULTS — Self-inoculation toy, v1 (runs 2026-09-10/11; revised after Codex review 2026-09-11)

Companion to `PREREG.md` (design, Codex design review, amendments A1–A15). Three seeds; numbers are given
per seed as s0/s1/s2, or marked "all" when the three agree to the stated precision. "Gap" = the model's
cross-entropy in nats minus the realized loss of the exact Bayes predictor on the same evaluation
sequences ("Bayes reference"; a finite-sample quantity, so gaps can be slightly negative). Figures in `figs/`.

## 0. Verdict

Under conflicting demands on a shared circuit, the network first changes behaviour in *both* contexts
through the shared pathway, and a gate appears between checkpoints 75 and 125, after the protected context has lost ≈ .05 nats:
from block 3 on, the shared belief is read in σ-relabeled coordinates when the context reads as case A
(an in-place relabeling or an equivalent conditional correction written by blocks 3–4; not distinguished),
and the unembedding then emits the flipped prediction. Weight-restoration shows the finetune's changes to
the embedding and unembedding can be reverted with little effect (f_A stays .91–1.00) and those to
blocks 1–2 with modest effect; grafting only the finetuned blocks 3–4 onto the pretrained net
reconstructs 82–91% of the gate's A–B contrast. Without
the protecting loss the same network corrupts fully, to the value predicted in advance. The gate's output
flip fraction is calibrated to the situational posterior per history (mean |f − w| ≈ 0.03). One preregistered
prediction (weight change concentrated late) is false: the change is distributed, and localization
required interventions.

## 1. Setup, definitions, what was run

**Process.** Each token encodes a pair (x, y), token id = 3x + y, x, y ∈ {0, 1, 2}. x is emitted by HMM1,
a Mess3 process (3 hidden states, 3 symbols; parameter α = 0.9 sets how sharply a state prefers "its"
symbol, parameter x_leave = 0.1 the probability of moving to each of the other two states, so the stay probability
is 1 − 2·x_leave). y is emitted, for the whole sequence,
by one of two other Mess3 processes: Mess3A (α .85, x_leave .15, sticky) or Mess3B (α .85, x_leave .5,
stay probability 0), each with prior 1/2; the **case** (A or B) is never given as input and must be inferred
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
model's x-marginal (renormalized over the 9 data tokens). f = 0 means no displacement along the flip direction D, f = 1 the full flip displacement; f is a
projection and is silent about errors orthogonal to D. Not clipped (values slightly outside [0, 1] are
noise). Undefined where P₀ = P₁ (t = 0). f_expl = the share
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
mean |fₙ − w_A,ₙ| = .033/.028/.026; binned means agree within .045 in every posterior bin; Spearman by
position .90 (t = 3) → .98 (t = 8) → .95 (t = 12). Pretrain baseline for the per-history correlation: ≈ 0.
The output-level flip weight is calibrated to the posterior history by history (the internal computation
that produces it is not identified by this); a learned position schedule is excluded.

### 2.4 Where the gate is (`figs/gate_localization.png`, `figs/restoration_calibration.png`)

- **Restoration (causal, per parameter group).** Putting pretrained weights back into the anchored net:
  embed → f_A .989/.975/.987 (no change); final (ln_f + unembedding) → .961/.911/.997 with gaps ≤ .010;
  blocks 1–2 → .926/.938/.875, f_B .120/.033/.066; block 4 → .962/.920/.945; **block 3 → .439/.419/.510**
  (gap_A +.15/+.16/+.12); **blocks 3–4 → .206/.185/.276** (gap_A +.29/+.31/+.24; f_B .15/.12/.14: the A–B contrast largely collapses, leaving a residual, nearly
  unconditional flip of ≈ .15–.2). Grafting finetuned groups onto the pretrained net:
  final only → f_A .07/.08/.06; blocks 1–2 only → .11/.08/.20; block 3 only → .79/.79/.71;
  **blocks 3–4 only → .884/.872/.861** with gap_A +.010/+.013/+.019 and f_B .08/.01/.03, i.e. an A–B
  contrast of .80/.87/.83 against the intact .975/.955/.974 (82%/91%/86%; f_A itself stays below the .9
  endpoint criterion). Most of the gate is carried by blocks 3–4 (block 3 dominant): restoring them
  removes most of the A–B contrast, grafting them alone rebuilds most of it. Changes to the embedding
  and unembedding are reversible at ≤ .06 cost in f_A; changes to blocks 1–2 contribute modestly to both
  A-flipping (f_A drops to .88–.94 when reverted) and B-suppression (f_B rises to .03–.12).
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
  the conditional relabeling sits at the block where the posterior first becomes *linearly* readable;
  case information is already usable upstream (patches at L1mlp work), so this is a statement about
  linear decodability, not about where the information first exists.
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

### 2.6 Robustness arm: history-heavy shared circuit (`runs_h/`, `figs_h/`; amendment A16)

Same design with HMM1 = Mess3(α = .7, x_leave = .03), for which a last-token predictor recovers only 49%
of the learnable x-signal (v1: 94%), so the shared tracker genuinely integrates history and its belief
geometry fills the simplex (`sim_h/geometries.png`). Bayes x reference 0.953, flip cost 0.29 nats,
predicted corrupted-on-B value 1.2445. Pretrain 10k steps. Three seeds complete (seed 0 on CPU, seeds 1–2 on GPU).

| | anchored s0/s1/s2 | unanchored s0/s1/s2 |
|---|---|---|
| x-CE_A on the flipped task − Bayes-gated ref | +.0004/+.0010/+.0009 | −.0010 all |
| x-CE_B on the unflipped task − Bayes-gated ref | +.0014/+.0004/+.0007 | **+.2908/+.2851/+.2885** (x-CE_B ≈ 1.243 vs predicted 1.2445) |
| (f_B, f_A) endpoint | (.018, .986) / (.008, .986) / (.014, .983) | (1.01, 1.01) / (1.00, 1.00) / (1.00, 1.00) |
| B-excursion (baseline-subtracted), step of max, return | +.040 @ 100, 150 / +.034 @ 75, 125 / +.038 @ 100, 125 | monotone |
| y-CE_B − ref | +.0012/+.0002/+.0005 | +.344/+.344/+.345 |

The trajectories in the (f_B, f_A) square (`figs_h/f_square.png`) and the position curve f_A(t) vs
oracle w_A(t) (`figs_h/position_f.png`) reproduce v1. The B-excursion is smaller in nats (.034–.040 vs
.047–.057) but the same fraction of the flip cost (≈ 12–14% of .29 vs ≈ 9–11% of .51): the leak before the
gate forms scales with how much the shared change hurts B.

*Interventions (seeds 1/2, with seed 0 in brackets where it differs; `logs/probe_h_*.log`, `logs/restore_calib_h.log`).* The tracker is now a real
in-context one: at pretrain the belief is readable at R² .50 from the embedding and .86/.92/.95/.99 at
L1mlp/L2mlp/L3mlp/L4mlp (v1: .94 at the embedding). Erasing its encoder image at all captures costs
+.135/+.139 in A and +.132/+.139 in B (uniform is +.145; pretrain +.148/+.148), y unchanged: shared and
used equally. Case patch A←B: f_A .99 → .02/.02 [.02] at L1mlp, .03/.04 [.03] at L2mlp, .04/.04 at L3attn, .88/.56 [.39] at
L3mlp, ≤ .01 change at L4mlp: the gate consumes the case subspace before block 3's MLP, as in v1.
Restoration: reverting the embedding or the final group leaves f_A at .99/.98 and 1.00/1.01; reverting
blocks 1–2 gives f_A .95/.95, f_B .12/.08; reverting block 3 gives f_A .45/.64 [.70]; reverting blocks 3–4 gives
f_A .12/.20 [.14] (f_B .09/.18 [.13]). Grafting only finetuned block 3 onto the pretrained net gives f_A
**.94/.83 [.65]**, f_B .12/.15 [.13]; grafting blocks 3–4 gives f_A .96/.97 [.94], f_B .08/.01 [−.02] (A–B
contrast .88/.96/.96 vs intact .98/.97/.97, i.e. 90–99%). Calibration: slope 1.007/.986/.985, intercept
−.012/+.017/+.019, mean |f − w| .037/.039/.043; position curve Spearman 1.000 (all), MAE .009/.009/.006. Case-posterior readability at L4mlp: pretrain .956/.833/.949 → anchored .967/.942/.966 (a clear rise
only in s1; the v1 rise of .86 → .93–.97 does not replicate as a rise here because two seeds already start at
.95) → unanchored .254/.208/.410 (the drop replicates). Unanchored: patch and case erasure change f by ≤ .01; no gate.

*Cross-situation probing and steering, pretrain vs after mixed finetune* (`blog_figs.py`, `blog_figs_mixed.py`;
`figs_h/pretrain_shared_circuit.png`, `figs_h/mixed_shared_circuit.png`; 3 seeds, post-MLP captures). Ridge
decoders of the HMM1 belief fitted in one situation and scored in the other, and donor-value steering along
the encoder image fitted in one situation and applied in the other (slope of the model's prediction change
onto the ideal predictor's change; for the finetuned net in A the ideal is the swapped predictor). Pretrain:
in- and cross-situation identical at every capture (decode .50/.84/.93/.98/.99 at emb/b1/b2/b3/b4; steer
.92/.97/.92/.95/.99). After mixed finetune: unchanged through block 2 (cross-situation decode .50/.85/.96,
steer .89/.94/.92), then divergence at block 3 (cross decode .84–.90 vs in-situation .98–.99; cross steer
.56/.59 vs .95) and partial recovery at block 4 (cross decode .83–.84, cross steer .82/.85 vs .96/.97). The
tracker up to block 2 is shared and intact; the representations part company at block 3, where the gate is.

*Is the situation a single axis?* (`meandiff_test.py`, `logs/meandiff_test.log`; mixed-finetuned nets, paired data.)
w is a linear functional of the 6-dim case belief, so it is readable along one direction; the A-vs-B mean-difference
direction reads logit-w with R² .73–.94 at block 4 and .6–.9 at block 3, but only 0–.45 at blocks 1–2. As a causal
handle it underperforms the 5-dim subspace: erasing the mean-difference direction leaves the switch essentially
intact at every capture (f_A ≥ .76, mostly ≥ .9; 5-d patch at the same captures: f_A → .02–.04); shifting activations
by the full mean difference moves the switch only at L2mlp–L3attn and inconsistently across seeds (f_A → .81/.13/.39
at L3attn; .54/.08/.15 at 2× the shift), while the 5-d patch is complete and seed-independent from the embedding
through L3attn. Reading: w is also carried as the gain of the two within-situation belief blocks, so removing one
axis leaves it recoverable; the switch reads the situation at blocks 1–3, before the mean-difference axis has
consolidated. Corrects the earlier shorthand "logit-w is not a direction": it is one for reading late, not for control.

*One difference from v1.* The transfer probe is much less clean here: the pretrain belief probe applied
after the anchored finetune reads the early layers poorly in *both* contexts (original-label R² at L1mlp
.55/.02, at L2mlp .75/.03), and the σ-relabeled reading in A-contexts appears only at L4mlp (.65/.57,
against B original .70/.54). With a tracker that integrates history, the finetune moves the belief's
coordinates substantially everywhere (freshly fitted probes still read it at R² .97–.99), so "relabeled
from block 3 in A only" is supported here by the patch and restoration results rather than by the
transfer probe. P1's logit-w clause (R² > .9 at t ≥ 8) fails here too: .861 at L4mlp (s1).

### 2.7 Weight displacement (the false prediction)

Relative L2 displacement from pretrained weights at step 125 (gate just formed), s0: embed .006,
L1.attn .09, L1.mlp .14, L2.attn .11, L2.mlp .13, L3.attn .14, L3.mlp .17, L4.attn .09, L4.mlp .14,
final .03 (s1, s2 within ±.03). Distributed across all blocks; the preregistered "concentrated in the last
block + unembedding, early groups < .05" is **false**. The displacement magnitude in blocks 1–2 is real but
restoration shows it is functionally minor; only interventions localized the gate.

## 3. Scorecard (PREREG §5 with amendments; a clause holds if it holds in ≥ 2 of 3 seeds)

| | outcome |
|---|---|
| P1 pretrain | ✓ joint CE within .0006 of reference (all); HMM1 R² .99; factor-subspace overlap .02–.16 (< .2). Logit-w R² at L4mlp, t ≥ 8: **fails as stated** (v1 s0 .896; history-heavy s1 .861; threshold .9). |
| P2 anchored endpoint | ✓ losses; ✓ f_A > .9 with f_expl > .8; ✓ f_B < .1 but **f_expl_B .03–.07 fails the literal > .8 clause** (statistic degenerate at f ≈ 0; not passed as written); ✓ erasure ratio in [.5, 2]; ✓ transfer R² > .8 in B-contexts at L3mlp (.89/.91/.86), L4mlp .82/.94/.82; A-vs-B overlap > .8 **holds at L1–L2, fails at L3–L4** (.49–.76); **✗ displacement clause**. |
| P3 anchored dynamics | ✓ baseline-subtracted excursion (step-0 gap_B +.0002) .055/.047/.057 vs threshold .05: s0, s2 pass, s1 fails narrowly; f_B > .1 in all; sustained return by step 125 (predicted within 2000). |
| P4 unanchored | ✓ x-CE_B → 1.36 as predicted (all); f_A ≈ f_B → 1 along the diagonal (|f_A − f_B| ≤ .04 at every checkpoint); no gate by patch/erasure. |
| P5 switch causality | ✓ position-curve Spearman 1.000 (> .9); per-history calibration supports it (post hoc, A15; not a preregistered clause); ✓ patch at L1mlp moves f_A by 94–98% of the A–B gap (> 50%). |

## 4. What it means, and what it does not

For the story (†) in PREREG §0: when a shared circuit serves two in-context-inferred situations and
training changes the demanded behaviour in one while the other's loss remains, gradient descent here
(i) first leaks the change into both situations through the shared pathway, (ii) a gate on the situational
posterior appears between checkpoints 75 and 125, after the protected situation has lost ≈ .05 nats
(whether the damage triggers it is interpretation), (iii) the gate is carried mainly by the blocks where
the posterior first becomes linearly readable and reads the shared belief in case-conditionally
relabeled coordinates, and (iv) the finetune's changes to the early blocks, embedding and unembedding
can be reverted at small cost. The protected behaviour returns to reference, and the changed behaviour is
confined to the eval-like situation with a flip weight calibrated to the posterior (agreement of the full
prediction vector with the Bayes mixture was not checked). Two corollaries for the blog: the misbehaviour leaks into the
protected context before the gate forms, by an amount that depends on how quickly the anchor's gradient
grows; and the situational variable loses linear accessibility when nothing consumes it
(A-only arm, both parameterizations) and keeps or gains it under mixed training (a clear gain only where the
pretrained readability had room to grow).

Not shown: intent; shielding without an incentive (the unanchored arm corrupts fully, and its population
optimum is the corrupted one); anything about RL or KL anchors (the anchor here is an explicit B loss);
in-place permutation vs conditional branch; robustness to a history-heavy shared circuit (this HMM1 is
nearly a two-token lookup: a last-token predictor recovers 94% of the learnable signal); situations that
switch within a sequence; behavioural (rather than probe-based) situational discrimination.

## 5. Next

1. Anchor-strength sweep: A:B mix 50/50 → 95/5. Prediction: damage before the bend grows as the B share
   shrinks; below some share the arm behaves like the unanchored one.
2. KL-to-reference anchor in place of B data (closest to RLHF).
3. History-heavy HMM1 robustness arm: done, 3 seeds (§2.6).
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
