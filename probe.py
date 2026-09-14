"""Probes and interventions on one checkpoint (PREREG §4(c),(d),(e); amendments A4, A6).

  python probe.py --ckpt runs/s0/pretrain/model.pt --out runs/s0/pretrain/probe.json --save_ref runs/s0/pretrain/probe_ref.npz
  python probe.py --ckpt runs/s0/anchored/model.pt --out runs/s0/anchored/probe.json --ref runs/s0/pretrain/probe_ref.npz

Per sublayer capture (emb, L1attn, L1mlp, ..., L4mlp), positions t = 1..63 (t = data tokens seen;
resid[:, t] <-> oracle beliefs[:, t]):
  ridge probes  (held-out R^2)  for HMM1 belief (3), case-stream belief (6), logit w_A (1),
                fit on A-contexts, on B-contexts, and pooled;
  subspace overlap  HMM1(A-fit) vs HMM1(B-fit)  [shared vs duplicated, supporting];
                    HMM1(pooled) vs case(pooled)   [FWH check];
  transfer          reference probe (from --ref) applied here: R^2 vs bx and vs sigma-relabeled bx;
  ablation          project out the HMM1 subspace / the w direction at capture l, re-run the rest of the
                    net, report x-CE, y-CE, f in A- and B-contexts (necessity in both contexts);
  per-history f     at t in [3,12]: single-sequence f_n vs oracle w_A,n (Spearman) — tracking vs schedule.
"""
import argparse, json
import numpy as np
import torch
torch.set_num_threads(8)
from scipy.stats import spearmanr
from process import Process, SIGMA, ce as np_ce
from model import GPT, BOS

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--ref', default=None, help='probe_ref.npz from the pretrain endpoint (transfer tests)')
ap.add_argument('--save_ref', default=None)
ap.add_argument('--n', type=int, default=1536, help='sequences per case')
ap.add_argument('--L', type=int, default=64)
ap.add_argument('--seed', type=int, default=777)
ap.add_argument('--hmm1', type=float, nargs=2, default=[0.90, 0.10])
ap.add_argument('--layers', type=int, default=4); ap.add_argument('--d', type=int, default=120); ap.add_argument('--heads', type=int, default=4)
args = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
L = args.L
CAPS = ['emb'] + [f'L{l}{s}' for l in range(1, args.layers + 1) for s in ('attn', 'mlp')]

# ---------------- data + oracle -----------------------------------------------------------
P = Process(hmm1=tuple(args.hmm1))
rng = np.random.default_rng(args.seed)
# PAIRED contexts: the SAME x-stream under a case-A y-stream and under a case-B y-stream (x is independent of
# the case, so both are in-distribution). Any A-vs-B difference in x-behaviour is then attributable to the case.
from process import sample_hmm
xs_shared, _ = sample_hmm(P.T1, P.pi1, args.n, L, rng)
data = {}
for cname, pA in (('A', 1.0), ('B', 0.0)):
    d = Process(hmm1=tuple(args.hmm1), pA=pA).sample(args.n, L, rng)
    xs = xs_shared; ys = d['ys']
    bx, bc, wA = P.beliefs(xs, ys)
    px, py = P.predictive(bx, bc)
    inp = torch.from_numpy(np.concatenate([np.full((args.n, 1), BOS), 3 * xs + ys], 1)).to(dev)
    data[cname] = dict(inp=inp, xs=xs, ys=ys, bx=bx, bc=bc, wA=wA, px=px[:, :L], pxf=P.flip_dist(px[:, :L]), py=py[:, :L])

model = GPT(d=args.d, n_layers=args.layers, n_heads=args.heads, n_ctx=L + 1).to(dev)
model.load_state_dict(torch.load(args.ckpt, map_location=dev)); model.eval()


@torch.no_grad()
def run(inp, edits=None):
    """Forward with optional edits={capture_index: fn(resid)->resid} applied at those captures.
    Returns logits, resid list (post-edit values)."""
    edits = edits or {}
    B, T = inp.shape
    x = model.wte(inp) + model.wpe(torch.arange(T, device=inp.device))
    if 0 in edits: x = edits[0](x)
    resid = [x]; k = 0
    for blk in model.blocks:
        x, x_mid, _ = blk(x)
        k += 1
        if k in edits: x_mid = edits[k](x_mid); x = x_mid + blk.mlp(blk.ln2(x_mid))
        resid.append(x_mid); k += 1
        if k in edits: x = edits[k](x)
        resid.append(x)
    return model.unembed(model.ln_f(x)), resid


def collect(cname, edits=None):
    out_logits, out_resid = [], None
    for i in range(0, args.n, 512):
        lg, rs = run(data[cname]['inp'][i:i + 512], edits)
        out_logits.append(lg[:, :-1].float().softmax(-1).cpu().numpy())
        rs = [r[:, :L].float().cpu().numpy() for r in rs]         # positions 0..63
        out_resid = rs if out_resid is None else [np.concatenate([a, b]) for a, b in zip(out_resid, rs)]
    return np.concatenate(out_logits), out_resid


def behav(q, e):
    """x-CE (flipped/unflipped targets), y-CE, pooled f per position from the 10-way softmax q (n, L, 10)."""
    q9 = q[..., :9]; q9 = q9 / q9.sum(-1, keepdims=True)
    qx = q9.reshape(*q9.shape[:2], 3, 3).sum(-1); qy = q9.reshape(*q9.shape[:2], 3, 3).sum(-2)
    D = e['pxf'] - e['px']; dq = qx - e['px']
    num = (dq * D).sum(-1).sum(0); den = (D * D).sum(-1).sum(0); tot = (dq * dq).sum(-1).sum(0); ok = den > 1e-3 * len(q)
    f = np.where(ok, num / np.where(ok, den, 1), np.nan)
    f_expl = np.where(ok, (num ** 2 / np.where(ok, den, 1)) / np.clip(tot, 1e-12, None), np.nan)
    return dict(x_ce_unflipped=np_ce(qx, e['xs']).mean(0), x_ce_flipped=np_ce(qx, SIGMA[e['xs']]).mean(0),
                y_ce=np_ce(qy, e['ys']).mean(0), f=f, f_expl=f_expl, qx=qx)


# ---------------- ridge utilities ----------------------------------------------------------
TPOS = np.arange(1, L)          # positions 1..63


def rows(resid_l, target, n_train=30000, n_test=10000, seed=0):
    X = resid_l[:, TPOS].reshape(-1, resid_l.shape[-1]); Y = target[:, TPOS].reshape(-1, target.shape[-1])
    idx = np.random.default_rng(seed).permutation(len(X))
    return X[idx[:n_train]], Y[idx[:n_train]], X[idx[n_train:n_train + n_test]], Y[idx[n_train:n_train + n_test]]


def ridge(Xtr, Ytr, Xte, Yte, lams=(1.0, 10.0, 100.0)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; ym = Ytr.mean(0)
    Z = (Xtr - mu) / sd; Zt = (Xte - mu) / sd
    best = None
    for lam in lams:
        W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Ytr - ym))
        pred = Zt @ W + ym
        r2 = 1 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
        if best is None or r2 > best['r2']: best = dict(r2=float(r2), W=W, mu=mu, sd=sd, ym=ym, lam=lam)
    return best


def apply(pr, X):
    return ((X - pr['mu']) / pr['sd']) @ pr['W'] + pr['ym']


def r2(Y, pred):
    return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def read_dirs(pr, k):
    """orthonormal basis (d x k) of the raw-activation directions the DECODER reads (top-k left singular vectors of W/sd)."""
    U, S, _ = np.linalg.svd(pr['W'] / pr['sd'][:, None], full_matrices=False)
    return U[:, :k]


def encoder_dirs(H, Bel, k):
    """ENCODER image: regress activations on the latent, H ~ Bel @ A + c; return orthonormal basis (d x k) of the
    row space of A (the directions along which the latent moves the activation) and the activation mean.
    Validated as the causal subspace in comp_icl/hier_rrxor/encdec_probe.py (decoder readout is not)."""
    Bc = Bel - Bel.mean(0); Hc = H - H.mean(0)
    A = np.linalg.lstsq(Bc, Hc, rcond=None)[0]                   # (k_lat, d)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[:k].T, H.mean(0)


def overlap(U1, U2):
    return float((np.linalg.norm(U1.T @ U2) ** 2) / min(U1.shape[1], U2.shape[1]))


def logit(w):
    w = np.clip(w, 1e-4, 1 - 1e-4); return np.log(w / (1 - w))


# ---------------- main -----------------------------------------------------------------------
q, resid = {}, {}
for c in ('A', 'B'):
    q[c], resid[c] = collect(c)
base = {c: behav(q[c], data[c]) for c in ('A', 'B')}
res = dict(ckpt=args.ckpt, captures=CAPS, base={c: {k: v.tolist() for k, v in base[c].items() if k != 'qx'} for c in ('A', 'B')}, probes={}, overlap={}, transfer={}, ablation={}, per_history={})

targets = {'hmm1': lambda c: data[c]['bx'][:, :L], 'case': lambda c: data[c]['bc'][:, :L], 'logit_w': lambda c: logit(data[c]['wA'][:, :L])[..., None]}
fits = {}
for li, cap in enumerate(CAPS):
    res['probes'][cap] = {}
    for tname, tf in targets.items():
        for ctx in ('A', 'B', 'pool'):
            if ctx == 'pool':
                Xtr, Ytr, Xte, Yte = [np.concatenate(z) for z in zip(rows(resid['A'][li], tf('A')), rows(resid['B'][li], tf('B')))]
            else:
                Xtr, Ytr, Xte, Yte = rows(resid[ctx][li], tf(ctx))
            pr = ridge(Xtr, Ytr, Xte, Yte)
            fits[(cap, tname, ctx)] = pr
            res['probes'][cap][f'{tname}_{ctx}'] = pr['r2']
    # overlaps
    UA, UB = read_dirs(fits[(cap, 'hmm1', 'A')], 2), read_dirs(fits[(cap, 'hmm1', 'B')], 2)
    Uh, Uc = read_dirs(fits[(cap, 'hmm1', 'pool')], 2), read_dirs(fits[(cap, 'case', 'pool')], 5)
    res['overlap'][cap] = dict(hmm1_A_vs_B=overlap(UA, UB), hmm1_vs_case=overlap(Uh, Uc))

# ---------------- transfer from reference probe ---------------------------------------------
if args.ref:
    ref = np.load(args.ref, allow_pickle=True)['fits'].item()
    for li, cap in enumerate(CAPS):
        pr = ref[(cap, 'hmm1', 'pool')]
        out = {}
        for c in ('A', 'B'):
            X = resid[c][li][:, TPOS].reshape(-1, args.d); Y = data[c]['bx'][:, TPOS].reshape(-1, 3)
            pred = apply(pr, X)
            out[f'{c}_r2'] = r2(Y, pred)
            out[f'{c}_r2_sigma'] = r2(Y[:, SIGMA], pred)             # relabeled-tracker hypothesis
        prw = ref[(cap, 'logit_w', 'pool')]
        for c in ('A', 'B'):
            X = resid[c][li][:, TPOS].reshape(-1, args.d); Y = logit(data[c]['wA'][:, TPOS]).reshape(-1, 1)
            out[f'{c}_w_r2'] = r2(Y, apply(prw, X))
        res['transfer'][cap] = out

# ---------------- ablations ------------------------------------------------------------------
def ablate_fn(U, mu):
    Ut = torch.from_numpy(U).float().to(dev); m = torch.from_numpy(mu).float().to(dev)
    def fn(h):
        hc = h - m
        return h - (hc @ Ut) @ Ut.T
    return fn


T16 = slice(16, L)
# encoder-image directions per capture, pooled over contexts (positions 1..63)
enc = {}
for li, cap in enumerate(CAPS):
    H = np.concatenate([resid[c][li][:, TPOS].reshape(-1, args.d) for c in ('A', 'B')])
    Bx = np.concatenate([data[c]['bx'][:, TPOS].reshape(-1, 3) for c in ('A', 'B')])
    Bc = np.concatenate([data[c]['bc'][:, TPOS].reshape(-1, 6) for c in ('A', 'B')])
    enc[(cap, 'hmm1')] = encoder_dirs(H, Bx, 2)          # x-tracker image (2-dim)
    enc[(cap, 'case')] = encoder_dirs(H, Bc, 5)          # case-stream image (5-dim: two scaled 2-simplices + the scale w)


def summar(b, b0, rel):
    return dict(
        x_unfl=float(b['x_ce_unflipped'][T16].mean()), x_fl=float(b['x_ce_flipped'][T16].mean()), y=float(b['y_ce'][T16].mean()),
        d_x_unfl=float(b['x_ce_unflipped'][T16].mean() - b0['x_ce_unflipped'][T16].mean()),
        d_x_fl=float(b['x_ce_flipped'][T16].mean() - b0['x_ce_flipped'][T16].mean()),
        d_y=float(b['y_ce'][T16].mean() - b0['y_ce'][T16].mean()),
        f=float(np.nanmean(b['f'][T16])), f0=float(np.nanmean(b0['f'][T16])),
        f_expl=float(np.nanmean(b['f_expl'][T16])), rel_dnorm=rel)


def erase(edits, label):
    entry = {}
    for c in ('A', 'B'):
        qa, rs = collect(c, edits=edits)
        rel = float(np.mean([np.linalg.norm(rs[k][:, TPOS] - resid[c][k][:, TPOS]) / np.linalg.norm(resid[c][k][:, TPOS]) for k in edits]))
        entry[c] = summar(behav(qa, data[c]), base[c], rel)
    res['ablation'][label] = entry


def patch_case(li, label):
    """Transplant the case-subspace component of the paired OTHER-context run into this run at capture li.
    A<-B: if the gate reads the case subspace at/before li, x-behaviour in A should become B-like (f_A -> f_B)."""
    U, mu = enc[(CAPS[li], 'case')]
    Ut = torch.from_numpy(U).float().to(dev); m = torch.from_numpy(mu).float().to(dev)
    entry = {}
    for c, other in (('A', 'B'), ('B', 'A')):
        donor = torch.from_numpy(resid[other][li]).float().to(dev)          # (n, L, d) positions 0..63
        qa_all = []
        for i in range(0, args.n, 512):
            dn = donor[i:i + 512]
            def fn(h, dn=dn):
                hc = h - m; dc = dn - m
                out = h.clone()
                out[:, :L] = h[:, :L] - (hc[:, :L] @ Ut) @ Ut.T + (dc @ Ut) @ Ut.T
                return out
            lg, _ = run(data[c]['inp'][i:i + 512], {li: fn})
            qa_all.append(lg[:, :-1].float().softmax(-1).cpu().numpy())
        b = behav(np.concatenate(qa_all), data[c])
        entry[c] = summar(b, base[c], float('nan'))
        entry[c]['f_other_ctx'] = float(np.nanmean(base[other]['f'][T16]))
    res['ablation'][label] = entry


for li, cap in enumerate(CAPS):
    erase({li: ablate_fn(*enc[(cap, 'hmm1')])}, f'{cap}:hmm1')
    erase({li: ablate_fn(*enc[(cap, 'case')])}, f'{cap}:case')
    patch_case(li, f'{cap}:casePATCH')
erase({li: ablate_fn(*enc[(cap, 'hmm1')]) for li, cap in enumerate(CAPS)}, 'ALL:hmm1')
erase({li: ablate_fn(*enc[(cap, 'case')]) for li, cap in enumerate(CAPS)}, 'ALL:case')
prd = fits[('L4mlp', 'hmm1', 'pool')]
erase({CAPS.index('L4mlp'): ablate_fn(read_dirs(prd, 2), prd['mu'])}, 'L4mlp:hmm1_DECODER')

# ---------------- per-history f vs w (P5, A4) ---------------------------------------------------
for c in ('A', 'B'):
    e = data[c]; qx = base[c]['qx']; D = e['pxf'] - e['px']; dq = qx - e['px']
    fn_ = (dq * D).sum(-1) / np.clip((D * D).sum(-1), 1e-12, None); ok = (D * D).sum(-1) > 0.02
    out = {}
    for t in range(3, 13):
        m = ok[:, t]
        if m.sum() > 50:
            rho = spearmanr(fn_[m, t], e['wA'][m, t]).correlation
            out[t] = dict(n=int(m.sum()), spearman_f_vs_wA=float(rho), f_mean=float(fn_[m, t].mean()), wA_mean=float(e['wA'][m, t].mean()))
    res['per_history'][c] = out

json.dump(res, open(args.out, 'w'), indent=1)
if args.save_ref:
    np.savez(args.save_ref, fits=np.array(fits, dtype=object))

# ---------------- console -----------------------------------------------------------------------
print(f"ckpt {args.ckpt}")
print("capture   R2 hmm1(A/B/pool)      R2 case(pool)  R2 logit_w(pool)  | ovl hmm1 A-B  hmm1-case")
for cap in CAPS:
    p = res['probes'][cap]; o = res['overlap'][cap]
    print(f"  {cap:6s}  {p['hmm1_A']:.3f}/{p['hmm1_B']:.3f}/{p['hmm1_pool']:.3f}      {p['case_pool']:.3f}        {p['logit_w_pool']:.3f}        |   {o['hmm1_A_vs_B']:.3f}       {o['hmm1_vs_case']:.3f}")
if args.ref:
    print("transfer from ref (pool hmm1 probe): capture  A_r2  A_r2_sigma  B_r2  B_r2_sigma | w: A B")
    for cap in CAPS:
        t = res['transfer'][cap]
        print(f"  {cap:6s}  {t['A_r2']:.3f}  {t['A_r2_sigma']:.3f}  {t['B_r2']:.3f}  {t['B_r2_sigma']:.3f} | {t['A_w_r2']:.3f} {t['B_w_r2']:.3f}")
print("interventions (t>=16): site:kind  ctx  d_x_unfl  d_x_fl   d_y    f0->f (f_expl)  |dresid|/|resid|  [casePATCH: f of other ctx]")
for lab, ent in res['ablation'].items():
    for c, v in ent.items():
        extra = f"  other-ctx f {v['f_other_ctx']:.2f}" if 'f_other_ctx' in v else ""
        print(f"  {lab:20s} {c}   {v['d_x_unfl']:+.3f}   {v['d_x_fl']:+.3f}  {v['d_y']:+.3f}   {v['f0']:.2f}->{v['f']:.2f} ({v['f_expl']:.2f})   {v['rel_dnorm']:.3f}{extra}")
print("per-history Spearman(f_n, w_A,n) at t=3..12:")
for c in ('A', 'B'):
    print(f"  {c}: " + "  ".join(f"t{t}:{v['spearman_f_vs_wA']:+.2f}(n={v['n']})" for t, v in res['per_history'][c].items()))
