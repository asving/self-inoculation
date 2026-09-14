"""Is the situation a single axis? On the mixed-finetuned nets: erase / steer along the A-vs-B mean-difference
direction at each capture, and compare with the 5-dim case-subspace patch. Paired data (same x-stream)."""
import json, numpy as np, torch
torch.set_num_threads(4)
from process import Process, SIGMA, sample_hmm
from model import GPT, BOS
H = (0.7, 0.03); L = 64; N = 1536; P = Process(hmm1=H); dev = 'cpu'
CAPS = ['emb'] + [f'L{l}{s}' for l in range(1, 5) for s in ('attn', 'mlp')]
rng = np.random.default_rng(777); xs, _ = sample_hmm(P.T1, P.pi1, N, L, rng); data = {}
for c, pA in (('A', 1.0), ('B', 0.0)):
    ys = Process(hmm1=H, pA=pA).sample(N, L, rng)['ys']; bx, bc, wA = P.beliefs(xs, ys); px, _ = P.predictive(bx, bc)
    data[c] = dict(inp=torch.from_numpy(np.concatenate([np.full((N, 1), BOS), 3 * xs + ys], 1)), px=px[:, :L], pxf=P.flip_dist(px[:, :L]), bc=bc[:, :L], wA=wA[:, :L])
def load(p):
    m = GPT(d=120, n_layers=4, n_heads=4, n_ctx=L + 1); m.load_state_dict(torch.load(p, map_location=dev)); m.eval(); return m
@torch.no_grad()
def run(m, inp, edits=None):
    edits = edits or {}; B, T = inp.shape; x = m.wte(inp) + m.wpe(torch.arange(T)); k = 0
    if 0 in edits: x = edits[0](x)
    resid = [x]
    for blk in m.blocks:
        x, x_mid, _ = blk(x); k += 1
        if k in edits: x_mid = edits[k](x_mid); x = x_mid + blk.mlp(blk.ln2(x_mid))
        resid.append(x_mid); k += 1
        if k in edits: x = edits[k](x)
        resid.append(x)
    return m.unembed(m.ln_f(x)), resid
def fstat(logits, c):
    q = logits[:, :-1].softmax(-1).numpy()[..., :9]; q = q / q.sum(-1, keepdims=True); qx = q.reshape(*q.shape[:2], 3, 3).sum(-1)
    e = data[c]; D = e['pxf'] - e['px']; dq = qx - e['px']; num = (dq * D)[:, 16:].sum(); den = (D * D)[:, 16:].sum(); return float(num / den)
def enc_dirs(Hm, Bel, k):
    A = np.linalg.lstsq(Bel - Bel.mean(0), Hm - Hm.mean(0), rcond=None)[0]; _, _, Vt = np.linalg.svd(A, full_matrices=False); return Vt[:k].T
TP = slice(16, L); out = {}
print("mixed-finetuned nets. f_A after intervention in situation-A runs (baseline ~0.98); f_B after intervention in situation-B runs (baseline ~0.01)")
print("capture | erase mean-diff  shift by mean-diff (A->B / B->A)  shift x2 | erase enc-w(1d)  shift enc-w | 5d case PATCH | logit-w R2 via mean-diff dir")
for s in ('s0', 's1', 's2'):
    m = load(f'runs_h/{s}/anchored/model.pt'); res = {c: run(m, data[c]['inp'])[1] for c in 'AB'}
    base = {c: fstat(run(m, data[c]['inp'])[0], c) for c in 'AB'}; out[s] = {'base': base}
    print(f"[{s}] baseline f_A {base['A']:.2f}, f_B {base['B']:.2f}")
    for k, cap in enumerate(CAPS):
        hA, hB = res['A'][k][:, :L].numpy(), res['B'][k][:, :L].numpy()
        muA, muB = hA[:, TP].reshape(-1, 120).mean(0), hB[:, TP].reshape(-1, 120).mean(0); d = muA - muB; u = d / np.linalg.norm(d); mu = (muA + muB) / 2
        # how well does the mean-diff direction read logit-w?
        Hp = np.concatenate([hA[:, 1:].reshape(-1, 120), hB[:, 1:].reshape(-1, 120)]); w = np.clip(np.concatenate([data['A']['wA'][:, 1:].ravel(), data['B']['wA'][:, 1:].ravel()]), 1e-4, 1 - 1e-4); lw = np.log(w / (1 - w))
        proj = (Hp - mu) @ u; a, b = np.polyfit(proj, lw, 1); r2 = 1 - ((lw - (a * proj + b)) ** 2).sum() / ((lw - lw.mean()) ** 2).sum()
        # encoder direction for logit-w (1-d) and 5-d case subspace
        Hall = np.concatenate([hA[:, 1:].reshape(-1, 120), hB[:, 1:].reshape(-1, 120)]); uw = enc_dirs(Hall, lw[:, None], 1)[:, 0]
        Uc = enc_dirs(Hall, np.concatenate([data['A']['bc'][:, 1:].reshape(-1, 6), data['B']['bc'][:, 1:].reshape(-1, 6)]), 5)
        ut, dt, mut, uwt, Uct = (torch.from_numpy(v).float() for v in (u, d, mu, uw, Uc))
        row = {}
        for c, other, sign in (('A', 'B', -1.0), ('B', 'A', +1.0)):
            donor = torch.from_numpy(res[other][k].numpy())
            def erase(h): return h - ((h - mut) @ ut)[..., None] * ut
            def shift(h, f=1.0): return h + sign * f * dt
            def erase_w(h): return h - ((h - mut) @ uwt)[..., None] * uwt
            def shift_w(h):   # move along enc-w by the mean difference of the projections
                return h + sign * float(abs((muA - muB) @ uw)) * uwt
            def patch(h, dn=donor):
                o = h.clone(); o[:, :L] = h[:, :L] - ((h[:, :L] - mut) @ Uct) @ Uct.T + ((dn[:, :L] - mut) @ Uct) @ Uct.T; return o
            row[c] = [fstat(run(m, data[c]['inp'], {k: fn})[0], c) for fn in (erase, shift, lambda h: shift(h, 2.0), erase_w, shift_w, patch)]
        out[s][cap] = dict(row=row, r2_meandiff=float(r2))
        print(f"  {cap:6s} |   {row['A'][0]:.2f}/{row['B'][0]:.2f}        {row['A'][1]:.2f}/{row['B'][1]:.2f}          {row['A'][2]:.2f}/{row['B'][2]:.2f}   |   {row['A'][3]:.2f}/{row['B'][3]:.2f}      {row['A'][4]:.2f}/{row['B'][4]:.2f}    |  {row['A'][5]:.2f}/{row['B'][5]:.2f}     |  {r2:.2f}")
json.dump(out, open('runs_h/meandiff_test.json', 'w'), indent=1)
