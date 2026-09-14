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

    # Defaults fixed by scan_params.py / oracle_sim.py (see PREREG.md §2). (alpha, x) each.
    def __init__(self, hmm1=(0.90, 0.10), mess3A=(0.85, 0.15), mess3B=(0.85, 0.50), pA=0.5):
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
        if arm == 'A':
            return np.where(case[:, None] == 0, f, xs)
        raise ValueError(f"unknown arm {arm!r}")

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
