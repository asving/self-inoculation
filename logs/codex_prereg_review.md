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
session id: 01a08d6c-37c5-7dc1-94ab-cb5a7bb9dc1b
--------
user
You are reviewing an experiment preregistration before any run is launched. Read PREREG.md and process.py in this directory (read-only). Two tasks, both as numbered lists, do NOT rewrite the document:

(A) ADVERSARIAL DESIGN CRITIQUE. Assume the authors are wrong and find out how. For each point say concretely what would go wrong and what result would be misread. Cover at least: confounds that would make prediction P2 (anchored arm: shared tracker preserved, readout gated) come out 'positive' without the claimed mechanism; alternative explanations for the P3 dynamics (transient B-loss excursion); whether the unanchored arm (P4) is a real control or trivially determined; whether the flip-fraction statistic f in section 4(b) is well-defined and identifiable given the S3 symmetry of Mess3; whether the probe-based tests in 4(c) can actually distinguish 'shared vs duplicated' and 'readout-gated vs upstream-gated' given a 120-dim residual stream and ridge regression; whether the target-only flip (inputs never flipped) introduces a train/eval asymmetry the authors have not noticed; whether the oracle floors are the right comparison for a model that also outputs BOS mass; anything about the non-ergodic direct-sum construction, the equal 50/50 prior, or the context length 64 that could invalidate the sharpening claims; and whether the stated confidence levels are calibrated. Also check the process.py code against the stated conventions (operator convention, flip as involution, ideal predictors) for bugs.

(B) FRESH-READER CLARITY. A technically sophisticated reader (knows transformers, HMMs, Bayesian filtering, gradient descent) sees this document fresh with zero project context. List every point that is unclear, undefined, context-presupposing, or where a symbol is used before its role is stated. Be exhaustive and specific (quote the phrase).

Be blunt and specific; skip praise.
codex
I’ll read both files, check the code against the stated conventions, and separate the design critique from the fresh-reader clarity issues.
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
| y floor, case-uncertain / case-known | 0.9895 / 0.9844 (A .9951, B .9837) |
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

**P2 (anchored arm, endpoint).** x-CE_A within 0.03 of 0.853 (the Bayes-gated floor); x-CE_B within
0.02 of 0.848; f_A > 0.9, f_B < 0.1. HMM1 subspace overlap A-vs-B > 0.8 and transfer R² from pretrain
> 0.8 (shared and preserved). *Falsifiers of (†)*: overlap A-vs-B < 0.5 (duplication); transfer R² <
0.5 with x-CE_B above floor (corruption); or losses never reach floor (unresolved conflict).

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

(none yet)

exec
/usr/bin/bash -lc 'cat process.py' in /data/users/asvin/self_inoculation
 succeeded in 0ms:
"""Generative process for the self-inoculation toy.

Joint token at each position = (x, y) with x from HMM1 (a Mess3) and y from the
CASE stream, a non-ergodic direct sum Mess3A (+) Mess3B: each sequence lives in
exactly one component for its whole length. Token id = 3*x + y  (vocab 9; BOS = 9).

Conventions (match shortcut_scan/mess3.py):
  T[z, i, j] = P(emit z, next state j | current state i)
  belief update on z:   eta' = eta @ T[z] / sum
  next-symbol probs:    P(z | eta) = eta @ E,   E[i, z] = T[z, i, :].sum()
  beliefs[:, t] = P(state_t | obs_{<t});  beliefs[:, 0] = prior.

Flip = transposition sigma on the x alphabet, applied to the x-TARGET only, in case A
(the input stream is never flipped). Because Mess3 is S3-symmetric under joint relabeling
of states and symbols, the flipped target stream has identical marginal statistics and
identical Bayes floor; the flip is a pure relabeling of the readout.
"""
from __future__ import annotations
import numpy as np

SIGMA = np.array([1, 0, 2])           # transposition 0<->1 on the x alphabet
VOCAB = 9
BOS = 9


def mess3(alpha: float, x: float) -> np.ndarray:
    beta = (1.0 - alpha) / 2.0
    y = 1.0 - 2.0 * x
    T0 = np.array([[alpha*y, beta*x, beta*x],
                   [alpha*x, beta*y, beta*x],
                   [alpha*x, beta*x, beta*y]])
    T1 = np.array([[beta*y, alpha*x, beta*x],
                   [beta*x, alpha*y, beta*x],
                   [beta*x, alpha*x, beta*y]])
    T2 = np.array([[beta*y, beta*x, alpha*x],
                   [beta*x, beta*y, alpha*x],
                   [beta*x, beta*x, alpha*y]])
    return np.stack([T0, T1, T2], axis=0)


def direct_sum(TA: np.ndarray, TB: np.ndarray) -> np.ndarray:
    """Block-diagonal composition: no mass ever crosses components."""
    V, nA, _ = TA.shape
    nB = TB.shape[1]
    T = np.zeros((V, nA + nB, nA + nB))
    T[:, :nA, :nA] = TA
    T[:, nA:, nA:] = TB
    return T


def emission(T: np.ndarray) -> np.ndarray:
    return T.sum(axis=2).T            # E[i, z]


def sample_hmm(T: np.ndarray, pi: np.ndarray, n: int, L: int, rng: np.random.Generator):
    """Returns (symbols (n, L), initial states (n,)). Generic in state count."""
    V, S, _ = T.shape
    E = emission(T)
    out = np.empty((n, L), dtype=np.int64)
    s0 = rng.choice(S, size=n, p=pi)
    s = s0.copy()
    for t in range(L):
        pz = E[s]
        z = (rng.random(n)[:, None] > pz.cumsum(1)).sum(1)
        z = np.minimum(z, V - 1)
        out[:, t] = z
        pj = T[z, s, :] / E[s, z][:, None]
        s = (rng.random(n)[:, None] > pj.cumsum(1)).sum(1)
        s = np.minimum(s, S - 1)
    return out, s0


def filter_beliefs(T: np.ndarray, pi: np.ndarray, seqs: np.ndarray) -> np.ndarray:
    n, L = seqs.shape
    S = T.shape[1]
    out = np.empty((n, L + 1, S))
    eta = np.tile(pi, (n, 1))
    out[:, 0] = eta
    for t in range(L):
        eta = np.einsum('ni,nij->nj', eta, T[seqs[:, t]])
        eta = eta / eta.sum(1, keepdims=True)
        out[:, t + 1] = eta
    return out


class Process:
    """HMM1 (x) independent of the case stream (y). Case stream = Mess3A (+) Mess3B."""

    def __init__(self, hmm1=(0.7, 0.10), mess3A=(0.6, 0.15), mess3B=(0.66, 0.5), pA=0.5):
        self.T1 = mess3(*hmm1)
        self.TA = mess3(*mess3A)
        self.TB = mess3(*mess3B)
        self.Tc = direct_sum(self.TA, self.TB)
        self.pi1 = np.ones(3) / 3
        self.pic = np.concatenate([pA * np.ones(3) / 3, (1 - pA) * np.ones(3) / 3])
        self.E1 = emission(self.T1)
        self.Ec = emission(self.Tc)
        self.params = dict(hmm1=hmm1, mess3A=mess3A, mess3B=mess3B, pA=pA)

    def sample(self, n: int, L: int, rng: np.random.Generator):
        """Returns dict with xs, ys (n, L), case (n,) in {0:A, 1:B}, tokens (n, L)."""
        xs, _ = sample_hmm(self.T1, self.pi1, n, L, rng)
        ys, s0 = sample_hmm(self.Tc, self.pic, n, L, rng)
        case = (s0 >= 3).astype(np.int64)
        return dict(xs=xs, ys=ys, case=case, tokens=3 * xs + ys)

    def beliefs(self, xs: np.ndarray, ys: np.ndarray):
        """bx (n, L+1, 3), bc (n, L+1, 6), wA (n, L+1) = P(case A | obs_{<t})."""
        bx = filter_beliefs(self.T1, self.pi1, xs)
        bc = filter_beliefs(self.Tc, self.pic, ys)
        return bx, bc, bc[..., :3].sum(-1)

    def predictive(self, bx, bc):
        """P(x_t | obs_{<t}) (n, L+1, 3) and P(y_t | obs_{<t}) (n, L+1, 3)."""
        return bx @ self.E1, bc @ self.Ec

    # ---- targets under the flip -------------------------------------------------
    @staticmethod
    def flip_target(xs: np.ndarray, case: np.ndarray, arm: str) -> np.ndarray:
        """x-targets. arm='pre': never flipped. 'A': flipped iff case A. 'all': flipped always."""
        if arm == 'pre':
            return xs
        f = SIGMA[xs]
        if arm == 'all':
            return f
        return np.where(case[:, None] == 0, f, xs)

    @staticmethod
    def flip_dist(px: np.ndarray) -> np.ndarray:
        """P^sigma[a] = P[sigma(a)]  (distribution of sigma(x) when x ~ P; sigma is an involution)."""
        return px[..., SIGMA]

    def ideal_predictors(self, px: np.ndarray, wA: np.ndarray, case: np.ndarray):
        """x-predictors for the anchored finetune task (target flipped iff case A).
        bayes:      wA*P^sigma + (1-wA)*P   (exact Bayes given the observations)
        oracle:     P^sigma if A else P     (case revealed; floor if the switch were free)
        unflipped:  P                       (the pretrained net, untouched)
        flipped:    P^sigma                 (fully corrupted shared readout)
        Shapes (n, L+1, 3)."""
        pf = self.flip_dist(px)
        w = wA[..., None]
        isA = (case[:, None, None] == 0)
        return dict(bayes=w * pf + (1 - w) * px,
                    oracle=np.where(isA, pf, px),
                    unflipped=px,
                    flipped=pf)


def ce(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """-log pred[target], elementwise over leading dims. pred (..., V), target (...)."""
    p = np.take_along_axis(pred, target[..., None], axis=-1)[..., 0]
    return -np.log(np.clip(p, 1e-12, None))

exec
/usr/bin/bash -lc "python -B - <<'PY'
import numpy as np,itertools
exec(open('process.py').read())
p=Process(hmm1=(.9,.1),mess3A=(.85,.15),mess3B=(.85,.5))
for name,T in [('x',p.T1),('A',p.TA),('B',p.TB)]:
 K=T.sum(0)
 err=max(np.max(np.abs(T[np.ix_(perm,perm,perm)]-T)) for perm in itertools.permutations(range(3)))
 print(name,'row sums',K.sum(1),'stationary',np.ones(3)/3@K,'symmetry error',err,'diag',K.diagonal())
print('defaults',Process().params)
print('involution',SIGMA[SIGMA])
rng=np.random.default_rng(123)
d=p.sample(10000,64,rng);bx,bc,w=p.beliefs(d['xs'],d['ys']);px,py=p.predictive(bx,bc)
known=np.where(d['case'][:,None,None]==0,bc[...,:3]/w[...,None],bc[...,3:]/(1-w[...,None]))
pyk=np.where(d['case'][:,None,None]==0,known@emission(p.TA),known@emission(p.TB))
for label,probs,target in [('y uncertain',py,d['ys']),('y known',pyk,d['ys'])]:
 vals=ce(probs[:,:64],target)[:,8:]
 print(label,'all',vals.mean(),'A',vals[d['case']==0].mean(),'B',vals[d['case']==1].mean())
print('w after zero, one tokens',w[0,:2])
PY" in /data/users/asvin/self_inoculation
 exited 127 in 0ms:
/usr/bin/bash: line 1: python: command not found

exec
/usr/bin/bash -lc "python3 -B - <<'PY'
import numpy as np,itertools
exec(open('process.py').read())
p=Process(hmm1=(.9,.1),mess3A=(.85,.15),mess3B=(.85,.5))
for name,T in [('x',p.T1),('A',p.TA),('B',p.TB)]:
 K=T.sum(0)
 err=max(np.max(np.abs(T[np.ix_(perm,perm,perm)]-T)) for perm in itertools.permutations(range(3)))
 print(name,'row sums',K.sum(1),'stationary',np.ones(3)/3@K,'symmetry error',err,'diag',K.diagonal())
print('defaults',Process().params)
print('involution',SIGMA[SIGMA])
rng=np.random.default_rng(123)
d=p.sample(10000,64,rng);bx,bc,w=p.beliefs(d['xs'],d['ys']);px,py=p.predictive(bx,bc)
pa=bc[...,:3]@emission(p.TA)/w[...,None]
pb=bc[...,3:]@emission(p.TB)/(1-w[...,None])
pyk=np.where(d['case'][:,None,None]==0,pa,pb)
for label,probs,target in [('y uncertain',py,d['ys']),('y known',pyk,d['ys'])]:
 vals=ce(probs[:,:64],target)[:,8:]
 print(label,'all',vals.mean(),'A',vals[d['case']==0].mean(),'B',vals[d['case']==1].mean())
print('w after zero, one tokens',w[0,:2])
PY" in /data/users/asvin/self_inoculation
 succeeded in 470ms:
<stdin>:13: RuntimeWarning: divide by zero encountered in divide
x row sums [1. 1. 1.] stationary [0.33333333 0.33333333 0.33333333] symmetry error 0.0 diag [0.8 0.8 0.8]
A row sums [1. 1. 1.] stationary [0.33333333 0.33333333 0.33333333] symmetry error 0.0 diag [0.7 0.7 0.7]
B row sums [1. 1. 1.] stationary [0.33333333 0.33333333 0.33333333] symmetry error 0.0 diag [0. 0. 0.]
defaults {'hmm1': (0.7, 0.1), 'mess3A': (0.6, 0.15), 'mess3B': (0.66, 0.5), 'pA': 0.5}
involution [0 1 2]
y uncertain all 0.9902041809209633 A 0.9966457366748278 B 0.9836272327260321
y known all 0.9852231794974933 A 0.9916264825728353 B 0.9786852879985798
w after zero, one tokens [0.5 0.5]

codex
The code’s default parameters differ from the preregistration. The main measurement problem is that the proposed losses and probes do not identify shared circuitry or where gating occurs. The target-only flip also makes the next x-target and y-target dependent through the unknown case, so their separate oracle losses cannot simply be added for the finetuning joint loss. I’m checking the symmetry and indexing details next.
codex
**(A) ADVERSARIAL DESIGN CRITIQUE**

1. **The executable defaults specify a different experiment.** `Process()` uses `hmm1=(0.7, 0.10)`, `mess3A=(0.6, 0.15)`, and `mess3B=(0.66, 0.5)`, whereas §2 specifies `(0.90, 0.10)`, `(0.85, 0.15)`, and `(0.85, 0.50)`. Unless every caller overrides the defaults, training and evaluation will have different signal strengths, flip costs, and case-identification rates from those preregistered. Failure or success could then be attributed to the mechanism when the actual intervention was a parameter change.

2. **P2 does not identify the claimed mechanism.** Good A/B losses and \(f_A\approx1,f_B\approx0\) establish context-dependent prediction. They do not establish that the model preserves and gates one shared tracker. Separate predictors, a redundant representation, a context-conditioned update rule, or a short-history approximation could produce these outputs. Calling that endpoint “shared tracker preserved, readout gated” overinterprets behavioral success.

3. **The experiment does not establish that pretraining produced a shared circuit.** Similar A/B probe row spaces establish similar linear decodability, not shared computation. Separate circuits can write into the same residual directions; a shared circuit can write into different directions depending on context. “Shared ≈ 1 (as at pretrain end)” assumes the baseline conclusion that the experiment needs to demonstrate.

4. **P2 can pass while an unused tracker survives.** The network could retain the pretrained belief representation as a byproduct while using newly learned circuitry for predictions. That yields high transfer \(R^2\), high A/B overlap, and correct losses without preserved *functional use* of the tracker. None of the proposed tests intervenes on the alleged tracker and measures its necessity in both contexts.

5. **“GD’s cheapest resolution” is neither defined nor measured.** The experiment measures no computational cost, parameter movement, representational capacity, or optimization preference across matched implementations. A 120-dimensional residual stream has ample room for multiple representations of a two-dimensional belief. Successful gating would not show that duplication was expensive or that gradient descent selected gating because it was cheapest.

6. **The flip statistic is only identifiable away from a degeneracy.** For normalized three-way probabilities, let \(d=P^\sigma-P\). The single-example least-squares solution is
   \[
   \hat f=\operatorname{clip}_{[0,1]}
   \frac{(q-P)^\top d}{d^\top d}.
   \]
   For this transposition,
   \[
   d=(P_1-P_0,\;P_0-P_1,\;0).
   \]
   When \(P_0=P_1\), every \(f\) gives the same prediction. This occurs exactly at the uniform prior, including the first prediction. Near equality, the estimate is unstable. Thus an \(f_A(t)\) curve starting at \(t=0\), and its proposed Spearman correlation, are not defined without a convention for uninformative positions.

7. **S₃ symmetry does not destroy every behavioral estimate of \(f\), but it destroys the proposed interpretation of internal permutations.** Fixed observed symbol labels make \(P\) and \(P^\sigma\) distinguishable on informative histories. However, latent-state labels have a permutation gauge: a permuted belief representation with a compensating decoder can implement precisely the same predictor. A probe reporting a “σ-relabeled tracker” cannot by itself establish that tracking changed rather than coordinates changed.

8. **An \(f\) value is a projection, not evidence of a mixture mechanism.** The fit ignores errors orthogonal to \(P^\sigma-P\), including errors in symbol 2. A badly calibrated predictor can have the expected \(f\) while lying far from the mixture segment. Other changes to the predictor can also project onto this direction. Without reporting fit residuals and predictive quality alongside \(f\), diagonal motion can be misread as a shared readout permutation.

9. **The aggregation choice can materially change \(f\).** Fitting one coefficient across examples weights histories by \(\|P^\sigma-P\|^2\); averaging separately fitted coefficients weights examples equally, including nearly uninformative ones. Clipping adds boundary bias. Under ideal per-history gating, a pooled fit estimates a signal-weighted average of \(w\), not necessarily the mean posterior plotted in §4(e). The proposed sharpening comparison can fail—or appear to pass—because of the estimator.

10. **The ridge probes cannot reliably distinguish shared from duplicated circuitry.** Belief coordinates sum to one, so the centered HMM1 target has rank at most two. Ridge solutions depend on residual covariance, regularization, feature scaling, and redundant correlated features. Two case-specific fits can choose different decoding directions for the same underlying representation, or common directions for duplicated representations. High overlap can be a false positive for sharing; low overlap can be a false positive for duplication.

11. **Transfer \(R^2\) confounds preservation with coordinate stability.** An invertible rotation or rescaling can destroy a fixed pretrained probe’s performance while leaving all relevant information and computation intact. Conversely, retaining a linearly decodable copy gives high transfer even if the network stops using it. The “corrupted vs preserved” test measures compatibility with an old decoder, not circuit integrity.

12. **The cross-case probe does not locate a gate.** A high unpermuted B-to-A probe score does not exclude upstream gating: both original and permuted beliefs may coexist. A high permuted score does not establish upstream gating: context-dependent basis changes can produce it. Gates can also be distributed across layers. Moreover, §4(c) says both readout and upstream gating support (†), although (†) specifically claims gating how a shared circuit’s output is used. That broadens the hypothesis after naming a more specific mechanism.

13. **P3’s initial premise does not follow from independence.** “Pretraining has x ⊥ case” does not imply “there is no \(w\times b\) interaction.” A standard transformer with joint token embeddings, attention, MLPs, and layer normalization is not constrained to a factored implementation. It may already contain nonlinear case–x features or an unused gate. Low overlap between linear probe spaces would not prove their absence.

14. **A transient B-loss excursion has many explanations besides “readout permutes first.”** The finetuning objective changes abruptly; Adam moments reset; finite batches produce competing gradients; joint output parameters may couple x and y; and calibration or representation drift may temporarily harm B. A transient shared output bias can move \(f_B\) without any tracker damage or gate construction. Observing the predicted curve would not isolate the proposed causal sequence.

15. **The excursion measurement is biased toward finding an excursion and may miss its actual peak.** The maximum across checkpoints selects positive evaluation noise. It is not baseline-subtracted, so initial pretrained excess loss counts toward the excursion. Conversely, checkpoints every 25 steps can miss a short transient. “Return” is undefined: one threshold crossing, sustained recovery, and endpoint recovery are different outcomes. The result could be declared consistent with P3 through measurement choices.

16. **P4 is largely predetermined at the population optimum, although its finite-training trajectory is not.** Under A-only training, the optimal x-predictor is globally \(P^\sigma\): x is independent of y and every A target is flipped. For these positive-emission processes, every finite y-history possible under B is also possible under A. Therefore, an exact population optimum under A must also flip on B histories. P4’s endpoint mostly tests whether optimization approaches that solution on histories that can be rare under A. It is not a clean test of an unconstrained choice between shielding and corruption.

17. **A P4 “shielding” result would not establish “no incentive at all.”** Initialization, implicit regularization, small parameter movement, finite training, and low probability of B-like histories under A can preserve B behavior. B was also explicitly trained during pretraining. Thus \(f_B<0.3\) could reflect incomplete optimization or retained pretrained behavior, rather than a special structural shielding mechanism.

18. **The two arms change more than the presence of a B anchor.** A is half of anchored training but all of unanchored training. At the same step count, unanchored gets twice as many A examples. Its y-distribution and optimal y-predictor also change: the training prior becomes \(P(A)=1\). Differences can therefore arise from A exposure, gradient weighting, and case-stream adaptation, rather than from preserving the shared tracker under conflict.

19. **Target-only flipping creates a teacher-forcing asymmetry.** In finetuning, the model predicts a flipped A symbol but subsequently receives the original, unflipped symbol as input. This is a valid supervised transformation task, but it is not ordinary next-token modeling of the flipped stream. During autonomous rollout, feeding emitted symbols back would give a different input convention. S₃ invariance of the fully flipped sequence distribution does not remove this distinction: it relabels histories too, whereas this task explicitly does not.

20. **Training-case knowledge and model-case uncertainty must remain distinct.** Target construction uses the true sequence case, but the model sees only the prefix. Identical early prefixes can receive different target conventions. Consequently, “zero change in competence” does not mean zero increase in task difficulty: uncertainty about the case adds irreducible x-target uncertainty. The quoted Bayes-gated loss already exceeds the case-revealed loss for this reason.

21. **The target-only flip also breaks conditional x/y independence for the joint output.** Before flipping, x and y predictions factor. After flipping, the correct joint predictor is
   \[
   Q(z,y\mid h)
   =w\,P^\sigma(z\mid h_x)\,R_A(y\mid h_y)
   +(1-w)\,P(z\mid h_x)\,R_B(y\mid h_y).
   \]
   This generally differs from the product of the two marginal predictors. The model trains on joint CE, while much of the analysis uses marginal CEs. Correct marginals can conceal incorrect joint dependence; adding marginal oracle losses does not give the finetuning joint Bayes floor. A gate-like output pattern may partly implement this joint dependence.

22. **The BOS convention leaves both loss accounting and \(f\) ambiguous.** A ten-way softmax assigns some probability to BOS. Summing its nine data-token probabilities into x/y marginals produces subnormalized distributions. Renormalizing them conditions on “not BOS” and removes a real prediction error; leaving them unnormalized changes the meaning of marginal CE and the mixture fit. The oracle has zero BOS mass and remains a valid lower bound, but finite-model excess loss includes a \(-\log(1-q_{\mathrm{BOS}})\) contribution. That contribution must not be misread as tracker or gate failure.

23. **A population Bayes predictor is not necessarily a per-case or finite-sample loss floor.** The posterior-mixture predictor minimizes expected loss under the mixture distribution. It need not minimize loss conditional on true A alone, and its realized loss is not a hard lower bound on every finite evaluation sample. Negative oracle-relative differences are possible. Calling every such quantity a “floor” invites incorrect pass/fail interpretation.

24. **The oracle table is internally inconsistent.** The stated case-known y average, `0.9844`, is incompatible with its listed A/B values `.9951` and `.9837` under a 50/50 mixture: their average is `.9894`, close to the *case-uncertain* entry. P2 also uses a B x-floor of `0.848`, whereas the table gives a Bayes-gated B loss of `.8538` and a case-oracle loss of `.8465`. These may refer to different estimands, but the document does not say so. Thresholds tied to these numbers are not currently reproducible.

25. **Non-ergodicity makes the task unusually favorable and limits the sharpening claim.** The component never changes, so all accumulated evidence remains useful forever. The model need not handle context switches, stale evidence, or uncertainty recovery. A gate that succeeds here need not handle changing situations. Also, the direct sum has a family of stationary distributions indexed by component weight; the 50/50 mixture is a chosen prior, not something stationarity determines.

26. **A median sharpening time does not establish a resolved regime.** “Median wrong-case mass < 0.01 at t = 13” leaves nearly half the sequences potentially less resolved, including confidently wrong ones. The later mean threshold at \(t=34\) already signals a tail. Calling all \(t\ge16\) “resolved” overstates what these summaries establish. Residual gate error could simply be Bayes uncertainty.

27. **The 50/50 prior and fixed length allow shortcuts.** The prior makes the initial ideal mixture exactly one half, and learned positional embeddings allow the model to learn an average position-dependent schedule. Within A, a generic rise with position could correlate strongly with mean \(w_A(t)\) without tracking individual histories. Fixed length 64 tests neither extrapolation nor robustness to different priors. P5’s position-level correlation is particularly vulnerable to this shortcut.

28. **“Information rate 0.367 nats/token” is insufficient to justify the timing claims.** Evidence rates are generally directional: A-vs-B and B-vs-A need not match. An asymptotic rate does not determine finite-prefix posterior quantiles, especially with latent-state uncertainty. Without its definition and separate component results, the scalar can obscure asymmetric or heavy-tailed identification.

29. **The proposed data intervention does not isolate a causal posterior switch.** Replacing an A y-stream with a B y-stream while holding x fixed is an in-distribution intervention on y, provided the streams are sampled correctly. But it changes all y features, not only \(w\). Ordinary \(f_A\) versus \(f_B\) averages in §4(b) do not specify this paired experiment. A difference establishes sensitivity to case-associated y information, not mediation through the probed posterior variable.

30. **Projecting out a ridge direction is not a clean intervention on \(w\).** It can remove correlated x/y features, alter activation scale, and create off-distribution states. The posterior may remain encoded in other directions or be reconstructed later. Removing centered log-odds would suggest movement toward \(w=0.5\), not specifically toward an unflipped prediction. Degraded \(f_A\) can therefore be collateral damage; unchanged \(f_A\) can reflect redundancy. Neither outcome identifies the gate.

31. **The logit probe may be numerically undefined.** `beliefs()` stores ordinary probabilities. With accumulating evidence, `wA` can round to exactly 0 or 1, making its logit infinite. In my numerical check with the stated parameters, some A posterior masses rounded to 1 by length 64. Clipping is therefore operationally necessary for such data, and its threshold changes the target variance and \(R^2\).

32. **The confidence claims are not calibrated as stated.** “~65% confidence” and “~80% confidence” are subjective forecasts with no calibration evidence. Three pretraining seeds cannot establish corresponding success probabilities. The preregistration also lacks a rule for combining seeds, uncertainty intervals, multiplicity handling across layers/checkpoints, and treatment of borderline thresholds. Thousands of correlated token positions do not replace independent training runs.

33. **P1 permits selection that weakens the falsification story.** “Any of these fails → fix the model/training … amendment, not a result” allows repeated redesign until favorable preconditions are found. That can define a conditional experiment, but failed attempts remain evidence about how readily the mechanism arises. P2 also leaves broad intermediate outcomes unclassified, and “losses never reach floor” could mean inadequate optimization rather than a falsified mechanism. Section 6 states the preferred conclusion before those alternatives have been separated.

34. **The broader analogy adds an untested assumption.** A supervised B loss directly rewards B competence. A KL-to-reference penalty does so only on contexts where it is evaluated, and it constrains outputs rather than explicitly preserving a circuit. The toy cannot establish that this anchor exists or operates similarly in the motivating incident.

35. **The core operator and flip code is internally consistent; the remaining implementation hazards are specific.** I checked the stated parameterization: transition rows normalize, uniform state distributions are stationary, the operators obey simultaneous S₃ relabeling, and the state-stay probabilities are `.8`, `.7`, and `0` for x, A, and B. Filtering uses the stated row-vector convention. `SIGMA[SIGMA]` is the identity, so `px[..., SIGMA]` correctly pushes forward this transposition. `ideal_predictors()["bayes"]` is correct for the anchored **x-marginal**, and its case-oracle alternative is correct given the case. However:
   - `flip_target()` silently treats every arm string other than `'pre'` and `'all'` as A-conditional flipping. Passing the documented training name `'pretrain'` would incorrectly flip A.
   - `sample()` does not add BOS despite the document’s sequence convention; the caller must do so.
   - Beliefs and predictions have length \(L+1\), targets length \(L\); callers must use predictions `[:, :L]`, with the matching transformer positions.
   - `ce()` clips at \(10^{-12}\), so it is a capped numerical loss, not exact CE for zero/tiny probabilities.
   - Parameter ranges and shape compatibility are not validated.

   These checks establish internal consistency, not agreement with the unprovided Shai appendix or correctness of the other scripts.

**(B) FRESH-READER CLARITY**

1. **“Self-inoculation toy” / “Re-grounding.”** Neither “self-inoculation” nor what is being re-grounded is defined. The title assumes project history.

2. **“the July 2026 OpenAI/Hugging Face swarm incident” / “nostalgebraist’s hypothesis.”** No citation, description of the relevant observations, or account of which part of the hypothesis motivates this experiment is supplied.

3. **“eval-like A, deployment-like B.”** The labels are introduced, but their analogy to the actual generating processes is unexplained. Nothing intrinsic to sticky versus leaving-state HMMs makes one evaluation-like.

4. **“one shared circuit (‘goodness’).”** “Circuit,” “shared,” and “goodness” have no operational definitions. It is unclear whether sharing means common parameters, activations, subspaces, computations, or causal dependence.

5. **“GD’s cheapest resolution.”** “Cheapest” is undefined, and the text later uses AdamW rather than plain GD. It is unclear whether GD means gradient-based training generally.

6. **“already-present situational-awareness variable w.”** This could mean the exact oracle posterior, a linearly decodable estimate, or a causally used network variable. Those roles are not distinguished.

7. **“KL-to-reference penalty acting in every context.”** The distribution over those contexts and the connection between this penalty and the explicit B loss are unstated.

8. **“shared x-tracker, context-inferred case.”** This shorthand precedes the full definition of x, case, and the tracker. “Case” should be distinguished from a sequence, component, and observed context.

9. **“x, y ∈ {0,1,2}” followed by “Mess3(α = 0.90, x = 0.10)” and “y = 1−2x.”** Both letters are overloaded between observed streams and operator parameters. A reader must repeatedly infer which meaning applies.

10. **“Mess3.”** The process family is not explained in the document. The operator matrices, parameter domains, latent states, and roles of \(\alpha\) and the transition parameter are delegated to code or an external appendix.

11. **“operators as in Shai et al. 2026 App. C.1.”** No full reference or link identifies the work. The later reference to Appendix H.2.2 has the same problem.

12. **“Stationary = uniform.”** It is unclear whether this refers to latent states, emitted symbols, individual components, or the six-state direct sum. For the direct sum, uniformity also requires the chosen equal component weights.

13. **“x ⊥ y. The case is inferable only from the y-stream.”** The independence statement does not specify whole streams versus same-position variables, and the text does not explicitly say that the true case is withheld from the network while available to target construction and evaluation.

14. **“Sequence = BOS then L = 64 data tokens.”** The relationship among 65 input slots, 64 loss-bearing predictions, and the unused prediction after the last data token is unstated. `process.py` itself returns only data tokens.

15. **“applied to the x-TARGET only.”** A concrete input/label alignment is missing. It is unclear until reconstructed whether the label predicted after observing \(x_t\) is \(\sigma(x_{t+1})\), and which position predicts the first token.

16. **“identical marginal statistics and identical Bayes floor.”** “Marginal” could mean one-token frequencies or the entire transformed sequence law. “Bayes floor” could condition on original history, transformed history, or the true case. These are different claims.

17. **“zero change in competence.”** Competence has no measurable definition, and the phrase does not acknowledge the additional case uncertainty in the transformed prediction task.

18. **“averages over positions t ≥ 8.”** The indexing convention is missing: is \(t\) the target index, input index, or number of observed data tokens? The upper endpoint is also omitted.

19. **“x floor,” “case-uncertain / case-known,” and “joint (x+y) floor.”** Each needs its conditioning information, target convention, and aggregation population specified. It is unclear whether the parenthetical A/B y numbers describe the known-case or uncertain-case predictor.

20. **“Bayes-gated predictor w·P^σ + (1−w)·P.”** \(P\) is used before its explicit definition in §4(b). Its dependence on history and time is suppressed, and it is not immediately stated that this is only an x-marginal predictor.

21. **“corrupted net on B.”** This labels a fully flipped ideal predictor as a corrupted network, although no internal corruption is part of its definition.

22. **“median wrong-case mass” / “mean < 0.01.”** The random quantity should be specified as \(1-w_A\) on A and \(w_A\) on B. It is unclear whether the statistics pool cases equally or are required separately within each case.

23. **“information rate 0.367 nats/token.”** The quantity, direction, estimator, and finite-length versus asymptotic interpretation are undefined.

24. **“the Simplex-standard Mess3” / “the non-ergodic post’s pair.”** These are unexplained project references. “Learnable signal” also lacks a definition, presumably but not explicitly uniform CE minus oracle CE.

25. **“t ∈ [16, 64] as the resolved regime.”** The indexing conflicts with the code’s \(L\) targets and \(L+1\) beliefs unless an explicit mapping is given. “Resolved” lacks a posterior criterion applying to individual sequences.

26. **“GPT-2-style pre-LN causal transformer.”** Reproducibility details remain open: activation function, dropout, biases, embedding/output tying, final normalization, initialization, and precise capture locations relative to residual addition and normalization.

27. **“next-token CE on the joint 9-way token” with “vocab 10.”** It is unclear whether BOS is masked from the output softmax, included as a tenth outcome, or removed and renormalized for training.

28. **“lr 5e-4 constant after 200 warmup steps” / “Same optimizer settings in both phases.”** The warmup starting rate and schedule are unspecified; whether warmup restarts at finetuning is unclear. Adam betas and epsilon are omitted.

29. **“stop criterion checked post hoc.”** It is unclear whether this is an acceptance criterion after a fixed 20k steps, an early-stopping rule, or permission to extend training. The selected checkpoint and treatment of failed seeds are not specified.

30. **“A and B (50/50).”** This could mean exactly balanced batches or independent case sampling with probability one half. “B never seen” also leaves open whether A-only sampling uses a changed prior, rejection sampling, or a separate generator.

31. **“Seeds: 3 pretrains.”** The randomization controlled by each seed, finetuning branch seeds, and whether arms share sample streams are unspecified.

32. **“Fixed eval set … with oracle beliefs and both target conventions precomputed.”** “Both” needs naming. The generation seed, probe training data, validation splits, and whether this set is also used for P1 acceptance or probe tuning are unstated.

33. **“marginalize to x and y.”** BOS treatment and normalization are undefined. It is also unclear whether marginal loss is computed from the joint distribution before or after conditioning on data-token output.

34. **“each minus its oracle floor on the same sequences.”** The oracle must be named for every reported metric. “Same sequences” does not specify the same positions, same target convention, or case-revealed versus posterior-mixture prediction.

35. **“B-excursion: max over finetune.”** The position aggregation, inclusion of step zero, use of sampled checkpoints versus training steps, and definition of sustained “return” are missing.

36. **“At each checkpoint and position, least-squares fit.”** It is unclear whether \(f\) is fitted per sequence, per case, or over all examples; whether there is one coefficient per position; how fits are weighted; and how degenerate denominators are handled.

37. **“report f_A and f_B averaged over t ≥ 16.”** The averaging order, endpoint, treatment of undefined values, and treatment of clipping are unspecified. No goodness-of-fit measure is stated.

38. **“pretrain end = (0,0); ideal anchored end = (0,1).”** These read as exact coordinates, although the pretrained model is approximate and the finite-prefix ideal anchored predictor uses a nondegenerate posterior. The distinction between schematic and quantitative claims is missing.

39. **“Probes (ridge, standardized, held-out R²).”** The ridge penalty, its selection procedure, feature versus target standardization, intercept, train/test split unit, pooling across positions, and multioutput \(R^2\) convention are unspecified.

40. **“HMM1 belief (3-dim), case-stream belief (6-dim), logit w_A.”** Their temporal alignment with captured activations is unstated. The six-dimensional belief’s block ordering and its inclusion of component uncertainty should be explicit. Logit clipping is unspecified.

41. **“principal subspace … normalized overlap.”** The retained rank, singular-value threshold, exact normalization, coordinate system after standardization, and handling of the belief’s sum-to-one constraint are missing.

42. **“HMM1 probe fit at pretrain end, applied at finetune end.”** It is unclear whether the scaler, intercept, and target normalization are frozen along with the weights, or whether anything is refitted.

43. **“best fit is σ-permuted. Record which.”** It is unclear whether the probe is fixed, what is permuted, whether only identity and σ are compared or all six permutations, and how ties or uniformly poor fits are handled.

44. **“earliest layer where logit-w_A R² > 0.9.”** “Layer” could mean block or sublayer. The checkpoint, case population, position range, and fallback if no layer qualifies are missing.

45. **“project the w direction out.”** The projection metric, centering, conversion from standardized probe weights to activation coordinates, affected positions, intervention duration, and downstream recomputation procedure are undefined.

46. **“f_A vs f_B in (b) already measures the switch’s causal effect.”** Section 4(b) does not specify paired x-streams or an intervention protocol. “Switch” could refer to the true component, inferred posterior, or neural representation.

47. **“f_A(t) … vs oracle w_A(t)” / “Spearman > 0.9.”** It is unclear whether posterior values are averaged over true A sequences, pooled across cases, or compared per example. The handling of \(t=0\), tied values, and saturated posterior estimates is unstated.

48. **“FWH check” / “Factored-World inductive bias.”** FWH is used before expansion, and neither the hypothesis nor the theorem or empirical claim being invoked is described or cited.

49. **“at some sublayer” / “overlap … > 0.8 and transfer R² … > 0.8.”** Layer-selection rules are missing. It is unclear whether all P2 conditions must hold at the same sublayer and whether thresholds apply to every seed or an aggregate.

50. **“within 0.03 of 0.853” / “within 0.02 of 0.848.”** Position ranges and whether “within” means absolute difference or only excess loss are unstated. These constants are not reconciled with the oracle table or the promise to use sequence-matched floors.

51. **“Falsifiers of (†).”** The document does not classify intermediate overlaps, moderate transfer degradation, mixed seeds, or correct losses with ambiguous probes. It also does not distinguish failure of the mechanism from failure of optimization or measurement.

52. **“≈ 1.36 and stays,” “≈ f_B → 1,” and “along the diagonal.”** No numerical tolerances, persistence interval, or trajectory-distance criterion makes these predictions adjudicable.

53. **“Hubinger-style shielding” / “would upgrade (†).”** The referenced idea is unexplained and uncited. “Upgrade” does not specify which claim would change or what evidential standard applies.

54. **“~65% confidence” / “~80% confidence.”** The forecast event is unclear: a single seed, all three seeds, their average, or the entire conjunction of predictions. The source of the probabilities is unstated.

55. **“Shows: … the gate is built from the existing situational-awareness variable.”** This section reads as an established result rather than a conditional interpretation. It does not explain which measurements would establish “built from” as opposed to correlated with.

56. **“behavior … stays at floor throughout (modulo the transient).”** The qualification removes the literal meaning of “throughout.” No maximum acceptable transient duration or damage is given.

57. **The file map is not an execution specification.** Names such as `train.py --phase pretrain|anchored|unanchored` do not explain how these phases map to `flip_target()`’s `'pre'`, `'A'`, and `'all'` strings, which parameters override `Process()` defaults, or which code version is frozen for the preregistration.
tokens used
18,560
**(A) ADVERSARIAL DESIGN CRITIQUE**

1. **The executable defaults specify a different experiment.** `Process()` uses `hmm1=(0.7, 0.10)`, `mess3A=(0.6, 0.15)`, and `mess3B=(0.66, 0.5)`, whereas §2 specifies `(0.90, 0.10)`, `(0.85, 0.15)`, and `(0.85, 0.50)`. Unless every caller overrides the defaults, training and evaluation will have different signal strengths, flip costs, and case-identification rates from those preregistered. Failure or success could then be attributed to the mechanism when the actual intervention was a parameter change.

2. **P2 does not identify the claimed mechanism.** Good A/B losses and \(f_A\approx1,f_B\approx0\) establish context-dependent prediction. They do not establish that the model preserves and gates one shared tracker. Separate predictors, a redundant representation, a context-conditioned update rule, or a short-history approximation could produce these outputs. Calling that endpoint “shared tracker preserved, readout gated” overinterprets behavioral success.

3. **The experiment does not establish that pretraining produced a shared circuit.** Similar A/B probe row spaces establish similar linear decodability, not shared computation. Separate circuits can write into the same residual directions; a shared circuit can write into different directions depending on context. “Shared ≈ 1 (as at pretrain end)” assumes the baseline conclusion that the experiment needs to demonstrate.

4. **P2 can pass while an unused tracker survives.** The network could retain the pretrained belief representation as a byproduct while using newly learned circuitry for predictions. That yields high transfer \(R^2\), high A/B overlap, and correct losses without preserved *functional use* of the tracker. None of the proposed tests intervenes on the alleged tracker and measures its necessity in both contexts.

5. **“GD’s cheapest resolution” is neither defined nor measured.** The experiment measures no computational cost, parameter movement, representational capacity, or optimization preference across matched implementations. A 120-dimensional residual stream has ample room for multiple representations of a two-dimensional belief. Successful gating would not show that duplication was expensive or that gradient descent selected gating because it was cheapest.

6. **The flip statistic is only identifiable away from a degeneracy.** For normalized three-way probabilities, let \(d=P^\sigma-P\). The single-example least-squares solution is
   \[
   \hat f=\operatorname{clip}_{[0,1]}
   \frac{(q-P)^\top d}{d^\top d}.
   \]
   For this transposition,
   \[
   d=(P_1-P_0,\;P_0-P_1,\;0).
   \]
   When \(P_0=P_1\), every \(f\) gives the same prediction. This occurs exactly at the uniform prior, including the first prediction. Near equality, the estimate is unstable. Thus an \(f_A(t)\) curve starting at \(t=0\), and its proposed Spearman correlation, are not defined without a convention for uninformative positions.

7. **S₃ symmetry does not destroy every behavioral estimate of \(f\), but it destroys the proposed interpretation of internal permutations.** Fixed observed symbol labels make \(P\) and \(P^\sigma\) distinguishable on informative histories. However, latent-state labels have a permutation gauge: a permuted belief representation with a compensating decoder can implement precisely the same predictor. A probe reporting a “σ-relabeled tracker” cannot by itself establish that tracking changed rather than coordinates changed.

8. **An \(f\) value is a projection, not evidence of a mixture mechanism.** The fit ignores errors orthogonal to \(P^\sigma-P\), including errors in symbol 2. A badly calibrated predictor can have the expected \(f\) while lying far from the mixture segment. Other changes to the predictor can also project onto this direction. Without reporting fit residuals and predictive quality alongside \(f\), diagonal motion can be misread as a shared readout permutation.

9. **The aggregation choice can materially change \(f\).** Fitting one coefficient across examples weights histories by \(\|P^\sigma-P\|^2\); averaging separately fitted coefficients weights examples equally, including nearly uninformative ones. Clipping adds boundary bias. Under ideal per-history gating, a pooled fit estimates a signal-weighted average of \(w\), not necessarily the mean posterior plotted in §4(e). The proposed sharpening comparison can fail—or appear to pass—because of the estimator.

10. **The ridge probes cannot reliably distinguish shared from duplicated circuitry.** Belief coordinates sum to one, so the centered HMM1 target has rank at most two. Ridge solutions depend on residual covariance, regularization, feature scaling, and redundant correlated features. Two case-specific fits can choose different decoding directions for the same underlying representation, or common directions for duplicated representations. High overlap can be a false positive for sharing; low overlap can be a false positive for duplication.

11. **Transfer \(R^2\) confounds preservation with coordinate stability.** An invertible rotation or rescaling can destroy a fixed pretrained probe’s performance while leaving all relevant information and computation intact. Conversely, retaining a linearly decodable copy gives high transfer even if the network stops using it. The “corrupted vs preserved” test measures compatibility with an old decoder, not circuit integrity.

12. **The cross-case probe does not locate a gate.** A high unpermuted B-to-A probe score does not exclude upstream gating: both original and permuted beliefs may coexist. A high permuted score does not establish upstream gating: context-dependent basis changes can produce it. Gates can also be distributed across layers. Moreover, §4(c) says both readout and upstream gating support (†), although (†) specifically claims gating how a shared circuit’s output is used. That broadens the hypothesis after naming a more specific mechanism.

13. **P3’s initial premise does not follow from independence.** “Pretraining has x ⊥ case” does not imply “there is no \(w\times b\) interaction.” A standard transformer with joint token embeddings, attention, MLPs, and layer normalization is not constrained to a factored implementation. It may already contain nonlinear case–x features or an unused gate. Low overlap between linear probe spaces would not prove their absence.

14. **A transient B-loss excursion has many explanations besides “readout permutes first.”** The finetuning objective changes abruptly; Adam moments reset; finite batches produce competing gradients; joint output parameters may couple x and y; and calibration or representation drift may temporarily harm B. A transient shared output bias can move \(f_B\) without any tracker damage or gate construction. Observing the predicted curve would not isolate the proposed causal sequence.

15. **The excursion measurement is biased toward finding an excursion and may miss its actual peak.** The maximum across checkpoints selects positive evaluation noise. It is not baseline-subtracted, so initial pretrained excess loss counts toward the excursion. Conversely, checkpoints every 25 steps can miss a short transient. “Return” is undefined: one threshold crossing, sustained recovery, and endpoint recovery are different outcomes. The result could be declared consistent with P3 through measurement choices.

16. **P4 is largely predetermined at the population optimum, although its finite-training trajectory is not.** Under A-only training, the optimal x-predictor is globally \(P^\sigma\): x is independent of y and every A target is flipped. For these positive-emission processes, every finite y-history possible under B is also possible under A. Therefore, an exact population optimum under A must also flip on B histories. P4’s endpoint mostly tests whether optimization approaches that solution on histories that can be rare under A. It is not a clean test of an unconstrained choice between shielding and corruption.

17. **A P4 “shielding” result would not establish “no incentive at all.”** Initialization, implicit regularization, small parameter movement, finite training, and low probability of B-like histories under A can preserve B behavior. B was also explicitly trained during pretraining. Thus \(f_B<0.3\) could reflect incomplete optimization or retained pretrained behavior, rather than a special structural shielding mechanism.

18. **The two arms change more than the presence of a B anchor.** A is half of anchored training but all of unanchored training. At the same step count, unanchored gets twice as many A examples. Its y-distribution and optimal y-predictor also change: the training prior becomes \(P(A)=1\). Differences can therefore arise from A exposure, gradient weighting, and case-stream adaptation, rather than from preserving the shared tracker under conflict.

19. **Target-only flipping creates a teacher-forcing asymmetry.** In finetuning, the model predicts a flipped A symbol but subsequently receives the original, unflipped symbol as input. This is a valid supervised transformation task, but it is not ordinary next-token modeling of the flipped stream. During autonomous rollout, feeding emitted symbols back would give a different input convention. S₃ invariance of the fully flipped sequence distribution does not remove this distinction: it relabels histories too, whereas this task explicitly does not.

20. **Training-case knowledge and model-case uncertainty must remain distinct.** Target construction uses the true sequence case, but the model sees only the prefix. Identical early prefixes can receive different target conventions. Consequently, “zero change in competence” does not mean zero increase in task difficulty: uncertainty about the case adds irreducible x-target uncertainty. The quoted Bayes-gated loss already exceeds the case-revealed loss for this reason.

21. **The target-only flip also breaks conditional x/y independence for the joint output.** Before flipping, x and y predictions factor. After flipping, the correct joint predictor is
   \[
   Q(z,y\mid h)
   =w\,P^\sigma(z\mid h_x)\,R_A(y\mid h_y)
   +(1-w)\,P(z\mid h_x)\,R_B(y\mid h_y).
   \]
   This generally differs from the product of the two marginal predictors. The model trains on joint CE, while much of the analysis uses marginal CEs. Correct marginals can conceal incorrect joint dependence; adding marginal oracle losses does not give the finetuning joint Bayes floor. A gate-like output pattern may partly implement this joint dependence.

22. **The BOS convention leaves both loss accounting and \(f\) ambiguous.** A ten-way softmax assigns some probability to BOS. Summing its nine data-token probabilities into x/y marginals produces subnormalized distributions. Renormalizing them conditions on “not BOS” and removes a real prediction error; leaving them unnormalized changes the meaning of marginal CE and the mixture fit. The oracle has zero BOS mass and remains a valid lower bound, but finite-model excess loss includes a \(-\log(1-q_{\mathrm{BOS}})\) contribution. That contribution must not be misread as tracker or gate failure.

23. **A population Bayes predictor is not necessarily a per-case or finite-sample loss floor.** The posterior-mixture predictor minimizes expected loss under the mixture distribution. It need not minimize loss conditional on true A alone, and its realized loss is not a hard lower bound on every finite evaluation sample. Negative oracle-relative differences are possible. Calling every such quantity a “floor” invites incorrect pass/fail interpretation.

24. **The oracle table is internally inconsistent.** The stated case-known y average, `0.9844`, is incompatible with its listed A/B values `.9951` and `.9837` under a 50/50 mixture: their average is `.9894`, close to the *case-uncertain* entry. P2 also uses a B x-floor of `0.848`, whereas the table gives a Bayes-gated B loss of `.8538` and a case-oracle loss of `.8465`. These may refer to different estimands, but the document does not say so. Thresholds tied to these numbers are not currently reproducible.

25. **Non-ergodicity makes the task unusually favorable and limits the sharpening claim.** The component never changes, so all accumulated evidence remains useful forever. The model need not handle context switches, stale evidence, or uncertainty recovery. A gate that succeeds here need not handle changing situations. Also, the direct sum has a family of stationary distributions indexed by component weight; the 50/50 mixture is a chosen prior, not something stationarity determines.

26. **A median sharpening time does not establish a resolved regime.** “Median wrong-case mass < 0.01 at t = 13” leaves nearly half the sequences potentially less resolved, including confidently wrong ones. The later mean threshold at \(t=34\) already signals a tail. Calling all \(t\ge16\) “resolved” overstates what these summaries establish. Residual gate error could simply be Bayes uncertainty.

27. **The 50/50 prior and fixed length allow shortcuts.** The prior makes the initial ideal mixture exactly one half, and learned positional embeddings allow the model to learn an average position-dependent schedule. Within A, a generic rise with position could correlate strongly with mean \(w_A(t)\) without tracking individual histories. Fixed length 64 tests neither extrapolation nor robustness to different priors. P5’s position-level correlation is particularly vulnerable to this shortcut.

28. **“Information rate 0.367 nats/token” is insufficient to justify the timing claims.** Evidence rates are generally directional: A-vs-B and B-vs-A need not match. An asymptotic rate does not determine finite-prefix posterior quantiles, especially with latent-state uncertainty. Without its definition and separate component results, the scalar can obscure asymmetric or heavy-tailed identification.

29. **The proposed data intervention does not isolate a causal posterior switch.** Replacing an A y-stream with a B y-stream while holding x fixed is an in-distribution intervention on y, provided the streams are sampled correctly. But it changes all y features, not only \(w\). Ordinary \(f_A\) versus \(f_B\) averages in §4(b) do not specify this paired experiment. A difference establishes sensitivity to case-associated y information, not mediation through the probed posterior variable.

30. **Projecting out a ridge direction is not a clean intervention on \(w\).** It can remove correlated x/y features, alter activation scale, and create off-distribution states. The posterior may remain encoded in other directions or be reconstructed later. Removing centered log-odds would suggest movement toward \(w=0.5\), not specifically toward an unflipped prediction. Degraded \(f_A\) can therefore be collateral damage; unchanged \(f_A\) can reflect redundancy. Neither outcome identifies the gate.

31. **The logit probe may be numerically undefined.** `beliefs()` stores ordinary probabilities. With accumulating evidence, `wA` can round to exactly 0 or 1, making its logit infinite. In my numerical check with the stated parameters, some A posterior masses rounded to 1 by length 64. Clipping is therefore operationally necessary for such data, and its threshold changes the target variance and \(R^2\).

32. **The confidence claims are not calibrated as stated.** “~65% confidence” and “~80% confidence” are subjective forecasts with no calibration evidence. Three pretraining seeds cannot establish corresponding success probabilities. The preregistration also lacks a rule for combining seeds, uncertainty intervals, multiplicity handling across layers/checkpoints, and treatment of borderline thresholds. Thousands of correlated token positions do not replace independent training runs.

33. **P1 permits selection that weakens the falsification story.** “Any of these fails → fix the model/training … amendment, not a result” allows repeated redesign until favorable preconditions are found. That can define a conditional experiment, but failed attempts remain evidence about how readily the mechanism arises. P2 also leaves broad intermediate outcomes unclassified, and “losses never reach floor” could mean inadequate optimization rather than a falsified mechanism. Section 6 states the preferred conclusion before those alternatives have been separated.

34. **The broader analogy adds an untested assumption.** A supervised B loss directly rewards B competence. A KL-to-reference penalty does so only on contexts where it is evaluated, and it constrains outputs rather than explicitly preserving a circuit. The toy cannot establish that this anchor exists or operates similarly in the motivating incident.

35. **The core operator and flip code is internally consistent; the remaining implementation hazards are specific.** I checked the stated parameterization: transition rows normalize, uniform state distributions are stationary, the operators obey simultaneous S₃ relabeling, and the state-stay probabilities are `.8`, `.7`, and `0` for x, A, and B. Filtering uses the stated row-vector convention. `SIGMA[SIGMA]` is the identity, so `px[..., SIGMA]` correctly pushes forward this transposition. `ideal_predictors()["bayes"]` is correct for the anchored **x-marginal**, and its case-oracle alternative is correct given the case. However:
   - `flip_target()` silently treats every arm string other than `'pre'` and `'all'` as A-conditional flipping. Passing the documented training name `'pretrain'` would incorrectly flip A.
   - `sample()` does not add BOS despite the document’s sequence convention; the caller must do so.
   - Beliefs and predictions have length \(L+1\), targets length \(L\); callers must use predictions `[:, :L]`, with the matching transformer positions.
   - `ce()` clips at \(10^{-12}\), so it is a capped numerical loss, not exact CE for zero/tiny probabilities.
   - Parameter ranges and shape compatibility are not validated.

   These checks establish internal consistency, not agreement with the unprovided Shai appendix or correctness of the other scripts.

**(B) FRESH-READER CLARITY**

1. **“Self-inoculation toy” / “Re-grounding.”** Neither “self-inoculation” nor what is being re-grounded is defined. The title assumes project history.

2. **“the July 2026 OpenAI/Hugging Face swarm incident” / “nostalgebraist’s hypothesis.”** No citation, description of the relevant observations, or account of which part of the hypothesis motivates this experiment is supplied.

3. **“eval-like A, deployment-like B.”** The labels are introduced, but their analogy to the actual generating processes is unexplained. Nothing intrinsic to sticky versus leaving-state HMMs makes one evaluation-like.

4. **“one shared circuit (‘goodness’).”** “Circuit,” “shared,” and “goodness” have no operational definitions. It is unclear whether sharing means common parameters, activations, subspaces, computations, or causal dependence.

5. **“GD’s cheapest resolution.”** “Cheapest” is undefined, and the text later uses AdamW rather than plain GD. It is unclear whether GD means gradient-based training generally.

6. **“already-present situational-awareness variable w.”** This could mean the exact oracle posterior, a linearly decodable estimate, or a causally used network variable. Those roles are not distinguished.

7. **“KL-to-reference penalty acting in every context.”** The distribution over those contexts and the connection between this penalty and the explicit B loss are unstated.

8. **“shared x-tracker, context-inferred case.”** This shorthand precedes the full definition of x, case, and the tracker. “Case” should be distinguished from a sequence, component, and observed context.

9. **“x, y ∈ {0,1,2}” followed by “Mess3(α = 0.90, x = 0.10)” and “y = 1−2x.”** Both letters are overloaded between observed streams and operator parameters. A reader must repeatedly infer which meaning applies.

10. **“Mess3.”** The process family is not explained in the document. The operator matrices, parameter domains, latent states, and roles of \(\alpha\) and the transition parameter are delegated to code or an external appendix.

11. **“operators as in Shai et al. 2026 App. C.1.”** No full reference or link identifies the work. The later reference to Appendix H.2.2 has the same problem.

12. **“Stationary = uniform.”** It is unclear whether this refers to latent states, emitted symbols, individual components, or the six-state direct sum. For the direct sum, uniformity also requires the chosen equal component weights.

13. **“x ⊥ y. The case is inferable only from the y-stream.”** The independence statement does not specify whole streams versus same-position variables, and the text does not explicitly say that the true case is withheld from the network while available to target construction and evaluation.

14. **“Sequence = BOS then L = 64 data tokens.”** The relationship among 65 input slots, 64 loss-bearing predictions, and the unused prediction after the last data token is unstated. `process.py` itself returns only data tokens.

15. **“applied to the x-TARGET only.”** A concrete input/label alignment is missing. It is unclear until reconstructed whether the label predicted after observing \(x_t\) is \(\sigma(x_{t+1})\), and which position predicts the first token.

16. **“identical marginal statistics and identical Bayes floor.”** “Marginal” could mean one-token frequencies or the entire transformed sequence law. “Bayes floor” could condition on original history, transformed history, or the true case. These are different claims.

17. **“zero change in competence.”** Competence has no measurable definition, and the phrase does not acknowledge the additional case uncertainty in the transformed prediction task.

18. **“averages over positions t ≥ 8.”** The indexing convention is missing: is \(t\) the target index, input index, or number of observed data tokens? The upper endpoint is also omitted.

19. **“x floor,” “case-uncertain / case-known,” and “joint (x+y) floor.”** Each needs its conditioning information, target convention, and aggregation population specified. It is unclear whether the parenthetical A/B y numbers describe the known-case or uncertain-case predictor.

20. **“Bayes-gated predictor w·P^σ + (1−w)·P.”** \(P\) is used before its explicit definition in §4(b). Its dependence on history and time is suppressed, and it is not immediately stated that this is only an x-marginal predictor.

21. **“corrupted net on B.”** This labels a fully flipped ideal predictor as a corrupted network, although no internal corruption is part of its definition.

22. **“median wrong-case mass” / “mean < 0.01.”** The random quantity should be specified as \(1-w_A\) on A and \(w_A\) on B. It is unclear whether the statistics pool cases equally or are required separately within each case.

23. **“information rate 0.367 nats/token.”** The quantity, direction, estimator, and finite-length versus asymptotic interpretation are undefined.

24. **“the Simplex-standard Mess3” / “the non-ergodic post’s pair.”** These are unexplained project references. “Learnable signal” also lacks a definition, presumably but not explicitly uniform CE minus oracle CE.

25. **“t ∈ [16, 64] as the resolved regime.”** The indexing conflicts with the code’s \(L\) targets and \(L+1\) beliefs unless an explicit mapping is given. “Resolved” lacks a posterior criterion applying to individual sequences.

26. **“GPT-2-style pre-LN causal transformer.”** Reproducibility details remain open: activation function, dropout, biases, embedding/output tying, final normalization, initialization, and precise capture locations relative to residual addition and normalization.

27. **“next-token CE on the joint 9-way token” with “vocab 10.”** It is unclear whether BOS is masked from the output softmax, included as a tenth outcome, or removed and renormalized for training.

28. **“lr 5e-4 constant after 200 warmup steps” / “Same optimizer settings in both phases.”** The warmup starting rate and schedule are unspecified; whether warmup restarts at finetuning is unclear. Adam betas and epsilon are omitted.

29. **“stop criterion checked post hoc.”** It is unclear whether this is an acceptance criterion after a fixed 20k steps, an early-stopping rule, or permission to extend training. The selected checkpoint and treatment of failed seeds are not specified.

30. **“A and B (50/50).”** This could mean exactly balanced batches or independent case sampling with probability one half. “B never seen” also leaves open whether A-only sampling uses a changed prior, rejection sampling, or a separate generator.

31. **“Seeds: 3 pretrains.”** The randomization controlled by each seed, finetuning branch seeds, and whether arms share sample streams are unspecified.

32. **“Fixed eval set … with oracle beliefs and both target conventions precomputed.”** “Both” needs naming. The generation seed, probe training data, validation splits, and whether this set is also used for P1 acceptance or probe tuning are unstated.

33. **“marginalize to x and y.”** BOS treatment and normalization are undefined. It is also unclear whether marginal loss is computed from the joint distribution before or after conditioning on data-token output.

34. **“each minus its oracle floor on the same sequences.”** The oracle must be named for every reported metric. “Same sequences” does not specify the same positions, same target convention, or case-revealed versus posterior-mixture prediction.

35. **“B-excursion: max over finetune.”** The position aggregation, inclusion of step zero, use of sampled checkpoints versus training steps, and definition of sustained “return” are missing.

36. **“At each checkpoint and position, least-squares fit.”** It is unclear whether \(f\) is fitted per sequence, per case, or over all examples; whether there is one coefficient per position; how fits are weighted; and how degenerate denominators are handled.

37. **“report f_A and f_B averaged over t ≥ 16.”** The averaging order, endpoint, treatment of undefined values, and treatment of clipping are unspecified. No goodness-of-fit measure is stated.

38. **“pretrain end = (0,0); ideal anchored end = (0,1).”** These read as exact coordinates, although the pretrained model is approximate and the finite-prefix ideal anchored predictor uses a nondegenerate posterior. The distinction between schematic and quantitative claims is missing.

39. **“Probes (ridge, standardized, held-out R²).”** The ridge penalty, its selection procedure, feature versus target standardization, intercept, train/test split unit, pooling across positions, and multioutput \(R^2\) convention are unspecified.

40. **“HMM1 belief (3-dim), case-stream belief (6-dim), logit w_A.”** Their temporal alignment with captured activations is unstated. The six-dimensional belief’s block ordering and its inclusion of component uncertainty should be explicit. Logit clipping is unspecified.

41. **“principal subspace … normalized overlap.”** The retained rank, singular-value threshold, exact normalization, coordinate system after standardization, and handling of the belief’s sum-to-one constraint are missing.

42. **“HMM1 probe fit at pretrain end, applied at finetune end.”** It is unclear whether the scaler, intercept, and target normalization are frozen along with the weights, or whether anything is refitted.

43. **“best fit is σ-permuted. Record which.”** It is unclear whether the probe is fixed, what is permuted, whether only identity and σ are compared or all six permutations, and how ties or uniformly poor fits are handled.

44. **“earliest layer where logit-w_A R² > 0.9.”** “Layer” could mean block or sublayer. The checkpoint, case population, position range, and fallback if no layer qualifies are missing.

45. **“project the w direction out.”** The projection metric, centering, conversion from standardized probe weights to activation coordinates, affected positions, intervention duration, and downstream recomputation procedure are undefined.

46. **“f_A vs f_B in (b) already measures the switch’s causal effect.”** Section 4(b) does not specify paired x-streams or an intervention protocol. “Switch” could refer to the true component, inferred posterior, or neural representation.

47. **“f_A(t) … vs oracle w_A(t)” / “Spearman > 0.9.”** It is unclear whether posterior values are averaged over true A sequences, pooled across cases, or compared per example. The handling of \(t=0\), tied values, and saturated posterior estimates is unstated.

48. **“FWH check” / “Factored-World inductive bias.”** FWH is used before expansion, and neither the hypothesis nor the theorem or empirical claim being invoked is described or cited.

49. **“at some sublayer” / “overlap … > 0.8 and transfer R² … > 0.8.”** Layer-selection rules are missing. It is unclear whether all P2 conditions must hold at the same sublayer and whether thresholds apply to every seed or an aggregate.

50. **“within 0.03 of 0.853” / “within 0.02 of 0.848.”** Position ranges and whether “within” means absolute difference or only excess loss are unstated. These constants are not reconciled with the oracle table or the promise to use sequence-matched floors.

51. **“Falsifiers of (†).”** The document does not classify intermediate overlaps, moderate transfer degradation, mixed seeds, or correct losses with ambiguous probes. It also does not distinguish failure of the mechanism from failure of optimization or measurement.

52. **“≈ 1.36 and stays,” “≈ f_B → 1,” and “along the diagonal.”** No numerical tolerances, persistence interval, or trajectory-distance criterion makes these predictions adjudicable.

53. **“Hubinger-style shielding” / “would upgrade (†).”** The referenced idea is unexplained and uncited. “Upgrade” does not specify which claim would change or what evidential standard applies.

54. **“~65% confidence” / “~80% confidence.”** The forecast event is unclear: a single seed, all three seeds, their average, or the entire conjunction of predictions. The source of the probabilities is unstated.

55. **“Shows: … the gate is built from the existing situational-awareness variable.”** This section reads as an established result rather than a conditional interpretation. It does not explain which measurements would establish “built from” as opposed to correlated with.

56. **“behavior … stays at floor throughout (modulo the transient).”** The qualification removes the literal meaning of “throughout.” No maximum acceptable transient duration or damage is given.

57. **The file map is not an execution specification.** Names such as `train.py --phase pretrain|anchored|unanchored` do not explain how these phases map to `flip_target()`’s `'pre'`, `'A'`, and `'all'` strings, which parameters override `Process()` defaults, or which code version is frozen for the preregistration.
