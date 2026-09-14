"""CPU oracle simulation (no training). Answers, before any GPU run:
  1. How fast does the case posterior wA sharpen with position?  -> context length
  2. Bayes floors per token component (x, y), per case, per position.
  3. Under the flip (x-target flipped iff case A): CE vs position of the four ideal
     x-predictors (bayes-gated / case-oracle / unflipped / flipped), split by case.
     These are the loss levels the pretrained, anchored-finetuned and corrupted nets
     should sit at.
  4. Belief geometries of HMM1, Mess3A, Mess3B (2-simplex scatter) for distinctness.
Writes sim/oracle_summary.json and sim/*.png.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from process import Process, ce

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--L", type=int, default=64)
ap.add_argument("--hmm1", type=float, nargs=2, default=[0.90, 0.10], metavar=("ALPHA", "X"))
ap.add_argument("--mess3A", type=float, nargs=2, default=[0.85, 0.15])
ap.add_argument("--mess3B", type=float, nargs=2, default=[0.85, 0.50])
ap.add_argument("--out", default="sim")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed)

P = Process(hmm1=tuple(args.hmm1), mess3A=tuple(args.mess3A), mess3B=tuple(args.mess3B))
d = P.sample(args.n, args.L, rng)
xs, ys, case = d["xs"], d["ys"], d["case"]
bx, bc, wA = P.beliefs(xs, ys)
px, py = P.predictive(bx, bc)
L = args.L
t = np.arange(L)                       # predictive for target at position t uses beliefs[:, t]
A, B = case == 0, case == 1

# ---------- 1. sharpening of the case posterior --------------------------------------
def q(a, qs=(0.1, 0.5, 0.9)):
    return np.quantile(a, qs, axis=0)
wA_A, wA_B = q(wA[A]), q(wA[B])         # (3, L+1)
err_A = 1 - wA[A]; err_B = wA[B]        # posterior mass on the wrong component
med_err = np.median(np.concatenate([err_A, err_B]), axis=0)
mean_err = np.mean(np.concatenate([err_A, err_B]), axis=0)
def first_below(v, thr):
    idx = np.where(v < thr)[0]
    return int(idx[0]) if len(idx) else None
sharp = {f"median_err<{thr}": first_below(med_err, thr) for thr in (0.2, 0.1, 0.05, 0.01)}
sharp.update({f"mean_err<{thr}": first_below(mean_err, thr) for thr in (0.2, 0.1, 0.05, 0.01)})
# log-likelihood-ratio slope = information rate about the case, nats/token
llr = np.log(np.clip(wA, 1e-12, 1)) - np.log(np.clip(1 - wA, 1e-12, 1))
llr_signed = np.where(A[:, None], llr, -llr)       # positive = toward the truth
slope = np.polyfit(np.arange(4, min(24, L)), llr_signed[:, 4:min(24, L)].mean(0), 1)[0]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for name, qq, c in (("true A", wA_A, "C0"), ("true B", wA_B, "C3")):
    ax[0].plot(qq[1], c, label=f"{name} median"); ax[0].fill_between(np.arange(L + 1), qq[0], qq[2], color=c, alpha=.2)
ax[0].set(xlabel="position t (tokens seen)", ylabel="w_A = P(case A | y_<t)", title="case posterior sharpening")
ax[0].legend()
ax[1].semilogy(med_err, label="median"); ax[1].semilogy(mean_err, label="mean")
ax[1].set(xlabel="position t", ylabel="posterior mass on wrong case", title=f"info rate ≈ {slope:.3f} nats/token")
ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{args.out}/sharpening.png", dpi=130); plt.close(fig)

# ---------- 2. Bayes floors per component -----------------------------------------
ce_x = ce(px[:, :L], xs)                       # (n, L)  x_t predicted from x_<t
ce_y = ce(py[:, :L], ys)                       # (n, L)  y_t predicted from y_<t (case-uncertain)
# case-known y floor: filter within the true component only
from process import filter_beliefs, emission
byA = filter_beliefs(P.TA, np.ones(3) / 3, ys); byB = filter_beliefs(P.TB, np.ones(3) / 3, ys)
pyk = np.where(A[:, None, None], byA @ emission(P.TA), byB @ emission(P.TB))
ce_yk = ce(pyk[:, :L], ys)
floors = dict(
    x_all=float(ce_x[:, 8:].mean()), x_pos=ce_x.mean(0).tolist(),
    y_all=float(ce_y[:, 8:].mean()), y_caseA=float(ce_y[A][:, 8:].mean()), y_caseB=float(ce_y[B][:, 8:].mean()),
    y_pos=ce_y.mean(0).tolist(), y_caseknown_all=float(ce_yk[:, 8:].mean()), y_caseknown_pos=ce_yk.mean(0).tolist(),
    joint_all=float((ce_x + ce_y)[:, 8:].mean()),
    uniform=float(np.log(3)),
)

# ---------- 3. flip task: ideal x-predictors ---------------------------------------
tgt = P.flip_target(xs, case, "A")             # flipped iff case A
preds = P.ideal_predictors(px[:, :L], wA[:, :L], case)
ce_pred = {k: ce(v, tgt) for k, v in preds.items()}        # (n, L) each
flip_tab = {}
for k, v in ce_pred.items():
    flip_tab[k] = dict(all=float(v[:, 8:].mean()), caseA=float(v[A][:, 8:].mean()), caseB=float(v[B][:, 8:].mean()),
                       pos_A=v[A].mean(0).tolist(), pos_B=v[B].mean(0).tolist())
# what the UNANCHORED-arm corrupted net does on case B under the *unflipped* B target:
ce_B_unflipped_target_flippedpred = float(ce(P.flip_dist(px[B][:, :L]), xs[B])[:, 8:].mean())
flip_tab["corrupted_on_true_B_task"] = ce_B_unflipped_target_flippedpred

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for i, (cname, m) in enumerate((("case A (target flipped)", A), ("case B (target unflipped)", B))):
    for k, ls in (("bayes", "-"), ("oracle", "--"), ("unflipped", ":"), ("flipped", "-.")):
        ax[i].plot(ce_pred[k][m].mean(0), ls, label=k)
    ax[i].axhline(np.log(3), color="gray", lw=.8, label="uniform")
    ax[i].set(title=cname, xlabel="position t", ylabel="x-CE (nats)" if i == 0 else None)
    ax[i].grid(alpha=.3)
ax[0].legend(fontsize=8)
fig.suptitle("anchored finetune task: ideal x-predictors")
fig.tight_layout(); fig.savefig(f"{args.out}/flip_predictors.png", dpi=130); plt.close(fig)

# ---------- 4. belief geometries ----------------------------------------------------
def simplex2d(b):
    v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    return b @ v
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
geoms = (("HMM1 (x stream)", bx[:, 8:].reshape(-1, 3)),
         ("Mess3A (case A y)", (bc[A][:, 8:, :3] / np.clip(bc[A][:, 8:, :3].sum(-1, keepdims=True), 1e-12, None)).reshape(-1, 3)),
         ("Mess3B (case B y)", (bc[B][:, 8:, 3:] / np.clip(bc[B][:, 8:, 3:].sum(-1, keepdims=True), 1e-12, None)).reshape(-1, 3)))
for a, (name, b) in zip(ax, geoms):
    sub = b[rng.choice(len(b), min(60000, len(b)), replace=False)]
    pts = simplex2d(sub)
    a.scatter(pts[:, 0], pts[:, 1], s=.3, alpha=.4, c=sub @ np.array([0, 1, 2]), cmap="viridis")
    a.set(title=name, xticks=[], yticks=[]); a.set_aspect("equal")
fig.tight_layout(); fig.savefig(f"{args.out}/geometries.png", dpi=130); plt.close(fig)

summary = dict(params=P.params, n=args.n, L=L, sharpening=sharp, info_rate_nats_per_token=float(slope),
               wA_median_trueA=wA_A[1].tolist(), wA_median_trueB=wA_B[1].tolist(),
               floors=floors, flip_task=flip_tab)
json.dump(summary, open(f"{args.out}/oracle_summary.json", "w"), indent=1)

print(f"params {P.params}")
print(f"case posterior: median wrong-mass <0.1 at t={sharp['median_err<0.1']}, <0.01 at t={sharp['median_err<0.01']}; "
      f"mean <0.1 at t={sharp['mean_err<0.1']}, <0.01 at t={sharp['mean_err<0.01']}; info rate {slope:.3f} nats/token")
print(f"floors (t>=8): x {floors['x_all']:.4f} | y {floors['y_all']:.4f} (A {floors['y_caseA']:.4f}, B {floors['y_caseB']:.4f}) "
      f"| y case-known {floors['y_caseknown_all']:.4f} | joint {floors['joint_all']:.4f} | uniform {np.log(3):.4f}")
print("flip task x-CE (t>=8):  predictor   all     caseA   caseB")
for k in ("bayes", "oracle", "unflipped", "flipped"):
    r = flip_tab[k]; print(f"  {k:10s} {r['all']:.4f}  {r['caseA']:.4f}  {r['caseB']:.4f}")
print(f"  corrupted net on the true (unflipped) case-B task: {ce_B_unflipped_target_flippedpred:.4f}")
