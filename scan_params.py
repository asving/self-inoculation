"""Scan Mess3 parameters for (i) HMM1: learnable signal + flip cost; (ii) case pairs:
per-token information rate about the case (symmetrized KL rate between the two
components' predictive distributions) and each component's floor.
Writes sim/scan_single.csv, sim/scan_pairs.csv; prints the best candidates."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import itertools, sys
import numpy as np
from process import mess3, emission, sample_hmm, filter_beliefs, ce, SIGMA

rng = np.random.default_rng(0)
N, L, T0 = 3000, 40, 8
alphas = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
xs_ = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]
grid = list(itertools.product(alphas, xs_))
U = np.ones(3) / 3

# ---- singles ------------------------------------------------------------------------
single = {}
seqs = {}
for a, x in grid:
    T = mess3(a, x); E = emission(T)
    s, _ = sample_hmm(T, U, N, L, rng); seqs[(a, x)] = s
    b = filter_beliefs(T, U, s); p = b[:, :L] @ E
    floor = ce(p, s)[:, T0:].mean()
    flipcost = ce(p, SIGMA[s])[:, T0:].mean() - floor       # unflipped predictor vs flipped target
    hstate = -(E * np.log(E)).sum(1).mean()                 # entropy given exact state
    single[(a, x)] = dict(floor=floor, flipcost=flipcost, hstate=hstate)
with open("sim/scan_single.csv", "w") as f:
    f.write("alpha,x,floor,flipcost,h_state\n")
    for (a, x), r in single.items():
        f.write(f"{a},{x},{r['floor']:.4f},{r['flipcost']:.4f},{r['hstate']:.4f}\n")

# ---- pairs: symmetrized KL rate -----------------------------------------------------
def kl_rate(pa, pb):
    """E_A[log P_A - log P_B] per token on A-sequences, t>=T0."""
    TA, TB = mess3(*pa), mess3(*pb)
    s = seqs[pa]
    la = ce(filter_beliefs(TA, U, s)[:, :L] @ emission(TA), s)
    lb = ce(filter_beliefs(TB, U, s)[:, :L] @ emission(TB), s)
    return (lb - la)[:, T0:].mean()
rows = []
for pa, pb in itertools.combinations(grid, 2):
    r = 0.5 * (kl_rate(pa, pb) + kl_rate(pb, pa))
    rows.append((pa, pb, r))
with open("sim/scan_pairs.csv", "w") as f:
    f.write("alphaA,xA,alphaB,xB,sym_kl_rate,floorA,floorB\n")
    for pa, pb, r in rows:
        f.write(f"{pa[0]},{pa[1]},{pb[0]},{pb[1]},{r:.4f},{single[pa]['floor']:.4f},{single[pb]['floor']:.4f}\n")

print("SINGLE Mess3 (t>=8):  alpha  x    floor   flipcost  H(z|state)   [uniform 1.0986]")
for (a, x), r in sorted(single.items(), key=lambda kv: kv[1]['floor']):
    print(f"  {a:4.2f} {x:4.2f}  {r['floor']:.4f}  {r['flipcost']:.4f}   {r['hstate']:.4f}")

print("\nPAIRS with sym KL rate in [0.15, 0.6] nats/token and both floors < 1.00, sorted by rate:")
cands = [(pa, pb, r) for pa, pb, r in rows if 0.15 <= r <= 0.6 and single[pa]['floor'] < 1.0 and single[pb]['floor'] < 1.0]
for pa, pb, r in sorted(cands, key=lambda z: z[2]):
    print(f"  A={pa}  B={pb}  rate {r:.3f}  floors {single[pa]['floor']:.3f}/{single[pb]['floor']:.3f}  "
          f"~t(1%)={np.log(99)/r:.0f}")
print(f"\n(reference: non-ergodic post pair A=(0.6,0.15) B=(0.66,0.5) is not on grid; nearest (0.6,0.15)-(0.7,0.5) rate = "
      f"{[r for pa,pb,r in rows if pa==(0.6,0.15) and pb==(0.7,0.5)][0]:.3f})")
