"""Figures for the blog's results section (history-heavy arm, runs_h).
 1. pretrain_shared_circuit.png : (left) HMM1-belief decoders fit in one situation, scored in the other;
                                  (right) steering along the encoder image fit in one situation, applied in the other.
 2. aonly_snapshots.png / mixed_snapshots.png : x loss by position vs the ideal unswapped (flip 0) and fully swapped
                                  (flip 1) predictors, with model snapshots across finetuning.
"""
import json, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_num_threads(4)
from process import Process, SIGMA, ce as np_ce, sample_hmm
from model import GPT, BOS
H = (0.7, 0.03); L = 64; RUNS = 'runs_h'; OUT = 'figs_h'; dev = 'cpu'
CAPS = ['emb', 'block 1', 'block 2', 'block 3', 'block 4']; CAP_IDX = [0, 2, 4, 6, 8]   # post-MLP residuals
P = Process(hmm1=H)

def load(path):
    m = GPT(d=120, n_layers=4, n_heads=4, n_ctx=L + 1); m.load_state_dict(torch.load(path, map_location=dev)); m.eval(); return m

@torch.no_grad()
def run(model, inp, edits=None):
    edits = edits or {}; B, T = inp.shape
    x = model.wte(inp) + model.wpe(torch.arange(T)); k = 0
    if 0 in edits: x = edits[0](x)
    resid = [x]
    for blk in model.blocks:
        x, x_mid, _ = blk(x); k += 1
        if k in edits: x_mid = edits[k](x_mid); x = x_mid + blk.mlp(blk.ln2(x_mid))
        resid.append(x_mid); k += 1
        if k in edits: x = edits[k](x)
        resid.append(x)
    return model.unembed(model.ln_f(x)), resid

def xmarg(logits):
    q = logits[:, :-1].softmax(-1).numpy()[..., :9]; q = q / q.sum(-1, keepdims=True)
    return q.reshape(*q.shape[:2], 3, 3).sum(-1)

# ---------------- 1. shared circuit at pretrain ---------------------------------------------
N = 1536; rng = np.random.default_rng(4242)
xs, _ = sample_hmm(P.T1, P.pi1, N, L, rng)
data = {}
for c, pA in (('A', 1.0), ('B', 0.0)):
    ys = Process(hmm1=H, pA=pA).sample(N, L, rng)['ys']
    bx, bc, wA = P.beliefs(xs, ys); px, _ = P.predictive(bx, bc)
    data[c] = dict(inp=torch.from_numpy(np.concatenate([np.full((N, 1), BOS), 3 * xs + ys], 1)), bx=bx[:, :L], px=px[:, :L])
TP = np.arange(1, L)
def ridge_fit(X, Y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6; ym = Y.mean(0); Z = (X - mu) / sd
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Y - ym)); return dict(W=W, mu=mu, sd=sd, ym=ym)
def ridge_r2(pr, X, Y):
    pred = ((X - pr['mu']) / pr['sd']) @ pr['W'] + pr['ym']; return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
def enc_dirs(Hm, Bel, k=2):
    A = np.linalg.lstsq(Bel - Bel.mean(0), Hm - Hm.mean(0), rcond=None)[0]; _, _, Vt = np.linalg.svd(A, full_matrices=False); return Vt[:k].T

dec = {k: [] for k in ('A→A', 'A→B', 'B→B', 'B→A')}; steer = {k: [] for k in ('A→A', 'A→B', 'B→B', 'B→A')}
for s in ('s0', 's1', 's2'):
    m = load(f'{RUNS}/{s}/pretrain/model.pt')
    res = {c: run(m, data[c]['inp'])[1] for c in 'AB'}
    q0 = {c: xmarg(run(m, data[c]['inp'])[0]) for c in 'AB'}
    for name, key in zip(CAPS, CAP_IDX):
        Hs = {c: res[c][key][:, :L].numpy() for c in 'AB'}
        half = N // 2
        Xtr = {c: Hs[c][:half][:, TP].reshape(-1, 120) for c in 'AB'}; Ytr = {c: data[c]['bx'][:half][:, TP].reshape(-1, 3) for c in 'AB'}
        Xte = {c: Hs[c][half:][:, TP].reshape(-1, 120) for c in 'AB'}; Yte = {c: data[c]['bx'][half:][:, TP].reshape(-1, 3) for c in 'AB'}
        pr = {c: ridge_fit(Xtr[c], Ytr[c]) for c in 'AB'}
        for fit in 'AB':
            for test in 'AB':
                dec[f'{fit}→{test}'].append((name, ridge_r2(pr[fit], Xte[test], Yte[test])))
        # steering: donor-value patch along the encoder image fit in `fit`, applied in `test`
        U = {c: enc_dirs(Xtr[c], Ytr[c]) for c in 'AB'}
        perm = np.random.default_rng(7).permutation(N)
        for fit in 'AB':
            Ut = torch.from_numpy(U[fit]).float()
            for test in 'AB':
                h = res[test][key]; hd = h[perm]
                def fn(hh, hd=hd, Ut=Ut): return hh + ((hd - hh) @ Ut) @ Ut.T
                q1 = xmarg(run(m, data[test]['inp'], {key: fn})[0])
                Pt, Pd = data[test]['px'], data[test]['px'][perm]
                D = (Pd - Pt)[:, 16:]; dq = (q1 - q0[test])[:, 16:]
                slope = float((dq * D).sum() / (D * D).sum())
                steer[f'{fit}→{test}'].append((name, slope))
    print(s, 'done')
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(len(CAPS)); w = 0.2
for i, (k, c) in enumerate((('A→A', 'C0'), ('A→B', 'C0'), ('B→B', 'C3'), ('B→A', 'C3'))):
    vals = np.array([[v for n, v in dec[k] if n == cap] for cap in CAPS]); hatch = '' if k[0] == k[-1] else '//'
    ax[0].bar(x + (i - 1.5) * w, vals.mean(1), w, yerr=vals.std(1), color=c, alpha=.9 if not hatch else .45, hatch=hatch, label=f"fit in {k[0]}, tested in {k[-1]}")
    vals = np.array([[v for n, v in steer[k] if n == cap] for cap in CAPS])
    ax[1].bar(x + (i - 1.5) * w, vals.mean(1), w, yerr=vals.std(1), color=c, alpha=.9 if not hatch else .45, hatch=hatch, label=f"image fit in {k[0]}, steered in {k[-1]}")
ax[0].set(title="reading HMM1's posterior out of the residual stream", ylabel="held-out R²", ylim=(0, 1.05)); ax[0].set_xticks(x); ax[0].set_xticklabels(CAPS); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis='y')
ax[1].set(title="steering HMM1's posterior along the encoder image", ylabel="fraction of the predicted change in the x prediction", ylim=(0, 1.15)); ax[1].set_xticks(x); ax[1].set_xticklabels(CAPS); ax[1].axhline(1, color='k', lw=.6, ls=':'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis='y')
fig.suptitle("after pretraining: the same HMM1 tracker serves both situations (3 seeds, mean ± sd)")
fig.tight_layout(); fig.savefig(f'{OUT}/pretrain_shared_circuit.png', dpi=130); print("saved pretrain_shared_circuit.png")
json.dump(dict(dec=dec, steer=steer), open(f'{OUT}/pretrain_shared_circuit.json', 'w'))

# ---------------- 2. snapshots across finetuning --------------------------------------------
erng = np.random.default_rng(999); ideal = {}
for c, pA in (('A', 1.0), ('B', 0.0)):
    d = Process(hmm1=H, pA=pA).sample(4096, L, erng); bx, bc, wA = P.beliefs(d['xs'], d['ys']); px, _ = P.predictive(bx, bc); px = px[:, :L]
    xs_, pxf = d['xs'], P.flip_dist(px)
    ideal[c] = dict(flip0_orig=np_ce(px, xs_).mean(0), flip1_orig=np_ce(pxf, xs_).mean(0), flip0_swap=np_ce(px, SIGMA[xs_]).mean(0), flip1_swap=np_ce(pxf, SIGMA[xs_]).mean(0))
SNAPS = [0, 50, 75, 100, 125, 150, 200, 10000]
for arm, fname, title in (('unanchored', 'aonly_snapshots.png', 'finetuning on situation A only'), ('anchored', 'mixed_snapshots.png', 'mixed finetuning (situations A and B)')):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    cmap = plt.get_cmap('viridis'); norm = matplotlib.colors.LogNorm(vmin=25, vmax=10000)
    for j, (c, task_key, i0, i1, ttl) in enumerate((('A', 'x_ce_flipped', 'flip0_swap', 'flip1_swap', 'situation A, loss on the swapped task'),
                                                     ('B', 'x_ce_unflipped', 'flip0_orig', 'flip1_orig', 'situation B, loss on the original task'))):
        for s in ('s0',):
            j_ = json.load(open(f'{RUNS}/{s}/{arm}/train.json')); logs = {e['step']: e for e in j_['log']}
            for st in SNAPS:
                if st in logs:
                    ax[j].plot(np.array(logs[st]['eval'][c][task_key])[1:], color=cmap(norm(max(st, 25))), lw=1.4, alpha=.95, label=f"step {st}")
        ax[j].plot(ideal[c][i0][1:], 'k-', lw=2.2, label='ideal, unswapped (flip 0)')
        ax[j].plot(ideal[c][i1][1:], 'k--', lw=2.2, label='ideal, fully swapped (flip 1)')
        ax[j].axhline(np.log(3), color='gray', lw=.7, ls=':', label='uniform guessing')
        ax[j].set(title=ttl, xlabel='position in sequence'); ax[j].grid(alpha=.3)
    ax[0].set_ylabel('x loss (nats per token)'); ax[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{title}: model snapshots across training (seed 0) against the two ideal predictors")
    fig.tight_layout(); fig.savefig(f'{OUT}/{fname}', dpi=130); print("saved", fname)
