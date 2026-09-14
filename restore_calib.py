"""Post-Codex follow-ups on the anchored endpoints (CPU):
 1. weight RESTORATION: put pretrained weights back into one parameter group of the finetuned net (or put
    finetuned weights of one group into the pretrained net) and re-measure f_A, f_B, x-gaps. Localizes
    which weight changes are necessary / sufficient for the gate.
 2. per-history CALIBRATION of the single-sequence flip fraction f_n against the oracle posterior w_A,n
    (binned means, slope, mean |f-w|), plus Spearman for every t in 3..12.
 3. position-curve Spearman of f_A(t) vs oracle mean w_A(t), t in 1..20, from train.json.
 4. P1 check: logit-w probe R^2 at L4mlp restricted to t>=8 on the pretrain checkpoint.
"""
import json, argparse, numpy as np, torch
torch.set_num_threads(4)
_ap = argparse.ArgumentParser(); _ap.add_argument('--runs', default='runs'); _ap.add_argument('--hmm1', type=float, nargs=2, default=[0.90, 0.10]); _a = _ap.parse_args(); RUNS = _a.runs
from scipy.stats import spearmanr
from process import Process, SIGMA, ce as np_ce, sample_hmm
from model import GPT, BOS
dev = 'cpu'; L = 64; N = 2048
P = Process(hmm1=tuple(_a.hmm1)); rng = np.random.default_rng(778)
xs, _ = sample_hmm(P.T1, P.pi1, N, L, rng)
data = {}
for c, pA in (('A', 1.0), ('B', 0.0)):
    ys = Process(hmm1=tuple(_a.hmm1), pA=pA).sample(N, L, rng)['ys']
    bx, bc, wA = P.beliefs(xs, ys); px, py = P.predictive(bx, bc)
    ideal = P.ideal_predictors(px[:, :L], wA[:, :L], np.zeros(N, int) if c == 'A' else np.ones(N, int))
    tgt = SIGMA[xs] if c == 'A' else xs
    data[c] = dict(inp=torch.from_numpy(np.concatenate([np.full((N, 1), BOS), 3 * xs + ys], 1)), xs=xs, ys=ys,
                   px=px[:, :L], pxf=P.flip_dist(px[:, :L]), wA=wA[:, :L], tgt=tgt, bc=bc[:, :L],
                   ref=np_ce(ideal['bayes'], tgt).mean(0))
T16 = slice(16, L)

@torch.no_grad()
def qx_of(model, c):
    out = []
    for i in range(0, N, 512):
        q = model(data[c]['inp'][i:i + 512])[:, :-1].softmax(-1).numpy()[..., :9]
        q = q / q.sum(-1, keepdims=True); out.append(q.reshape(*q.shape[:2], 3, 3).sum(-1))
    return np.concatenate(out)

def behav(model):
    r = {}
    for c in 'AB':
        e = data[c]; qx = qx_of(model, c); D = e['pxf'] - e['px']; dq = qx - e['px']
        num = (dq * D).sum(-1).sum(0); den = (D * D).sum(-1).sum(0); ok = den > 1e-3 * N
        f = np.where(ok, num / np.where(ok, den, 1), np.nan)
        r[c] = dict(f=float(np.nanmean(f[T16])), gap=float(np_ce(qx, e['tgt'])[:, T16].mean() - e['ref'][T16].mean()), qx=qx)
    return r

def load(path):
    m = GPT(d=120, n_layers=4, n_heads=4, n_ctx=L + 1); m.load_state_dict(torch.load(path, map_location=dev)); m.eval(); return m

GROUPS = {'embed': lambda k: k.startswith('wte') or k.startswith('wpe'),
          'blocks1-2': lambda k: k.startswith('blocks.0.') or k.startswith('blocks.1.'),
          'block3': lambda k: k.startswith('blocks.2.'), 'block4': lambda k: k.startswith('blocks.3.'),
          'blocks3-4': lambda k: k.startswith('blocks.2.') or k.startswith('blocks.3.'),
          'final(ln_f+unembed)': lambda k: k.startswith('ln_f') or k.startswith('unembed')}
res = {}
import os
for s in ('s0', 's1', 's2'):
    if not os.path.exists(f'{RUNS}/{s}/anchored/model.pt'):
        print(f'[{s}] anchored endpoint missing, skipping'); continue
    pre = torch.load(f'{RUNS}/{s}/pretrain/model.pt', map_location=dev); anc = torch.load(f'{RUNS}/{s}/anchored/model.pt', map_location=dev)
    m = load(f'{RUNS}/{s}/anchored/model.pt'); base = behav(m)
    rows = {'anchored (none restored)': base}
    for g, sel in GROUPS.items():
        sd = {k: (pre[k] if sel(k) else anc[k]) for k in anc}; m.load_state_dict(sd); rows[f'restore {g} -> pretrain'] = behav(m)
    for g, sel in GROUPS.items():
        sd = {k: (anc[k] if sel(k) else pre[k]) for k in anc}; m.load_state_dict(sd); rows[f'pretrain + finetuned {g} only'] = behav(m)
    print(f"\n[{s}] restoration (t>=16):  f_A   f_B  | gap_A(flipped task)  gap_B(unflipped task)")
    for k, v in rows.items():
        print(f"  {k:38s} {v['A']['f']:.3f} {v['B']['f']:.3f} | {v['A']['gap']:+.3f}  {v['B']['gap']:+.3f}")
    res[s] = {k: {c: {kk: vv for kk, vv in v[c].items() if kk != 'qx'} for c in 'AB'} for k, v in rows.items()}
    # ---- calibration on the anchored endpoint (paired, pooled contexts)
    m.load_state_dict(anc)
    fs, ws = [], []
    print(f"[{s}] per-history Spearman(f_n, w_A,n), pooled A+B contexts:")
    line = []
    for t in range(3, 13):
        f_t, w_t = [], []
        for c in 'AB':
            e = data[c]; qx = base[c]['qx']; D = e['pxf'][:, t] - e['px'][:, t]; d2 = (D * D).sum(-1); ok = d2 > 0.02
            f_t.append(((qx[:, t] - e['px'][:, t]) * D).sum(-1)[ok] / d2[ok]); w_t.append(e['wA'][ok, t])
        f_t, w_t = np.concatenate(f_t), np.concatenate(w_t); fs.append(f_t); ws.append(w_t)
        line.append(f"t{t}:{spearmanr(f_t, w_t).correlation:+.2f}")
    print("  " + "  ".join(line))
    F, W = np.concatenate(fs), np.concatenate(ws)
    bins = np.array([0, .1, .3, .5, .7, .9, 1.0001]); idx = np.digitize(W, bins) - 1
    cal = [(f"[{bins[b]:.1f},{bins[b+1]:.1f})", int((idx == b).sum()), float(W[idx == b].mean()), float(F[idx == b].mean()), float(np.clip(F[idx == b], 0, 1).mean())) for b in range(6) if (idx == b).sum() > 30]
    slope, icpt = np.polyfit(W, F, 1)
    print(f"[{s}] calibration pooled t=3..12: slope {slope:.3f} intercept {icpt:+.3f} mean|f-w| {np.abs(F - W).mean():.3f} (clipped f: {np.abs(np.clip(F,0,1) - W).mean():.3f}); n={len(F)}")
    print("  bin        n     mean w   mean f   mean clip(f)")
    for b in cal: print(f"  {b[0]:10s} {b[1]:5d}   {b[2]:.3f}    {b[3]:.3f}    {b[4]:.3f}")
    res[s]['calibration'] = dict(slope=float(slope), intercept=float(icpt), mae=float(np.abs(F - W).mean()), bins=cal, spearman_by_t=line)
    # ---- position-curve Spearman from train.json
    j = json.load(open(f'{RUNS}/{s}/anchored/train.json')); ev = j['log'][-1]['eval']; wa = np.array(j['floors']['A']['wA_mean'])
    fA = np.array(ev['A']['f'], float); tt = np.arange(1, 21)
    rho = spearmanr(fA[tt], wa[tt]).correlation; mae = np.abs(fA[tt] - wa[tt]).mean()
    print(f"[{s}] position curve t=1..20: Spearman(f_A(t), mean w_A(t)) = {rho:+.3f}, mean|f_A(t)-w_A(t)| = {mae:.3f}")
    res[s]['position'] = dict(spearman=float(rho), mae=float(mae))
# ---- P1: logit-w R^2 at L4mlp for t>=8, pretrain s0
_s = next(s for s in ('s0','s1','s2') if os.path.exists(f'{RUNS}/{s}/pretrain/model.pt')); m = load(f'{RUNS}/{_s}/pretrain/model.pt')
Hs, Ys = [], []
with torch.no_grad():
    for c in 'AB':
        for i in range(0, N, 512):
            _, resid, _ = m(data[c]['inp'][i:i + 512], return_resid=True)
            Hs.append(resid[-1][:, 8:L].reshape(-1, 120).numpy()); w = np.clip(data[c]['wA'][i:i + 512, 8:L], 1e-4, 1 - 1e-4); Ys.append(np.log(w / (1 - w)).reshape(-1, 1))
H, Y = np.concatenate(Hs), np.concatenate(Ys); perm = np.random.default_rng(0).permutation(len(H)); tr, te = perm[:40000], perm[40000:50000]
mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-6; Z = (H[tr] - mu) / sd; Zt = (H[te] - mu) / sd; ym = Y[tr].mean(0)
best = max((1 - ((Y[te] - (Zt @ np.linalg.solve(Z.T @ Z + lam * np.eye(120), Z.T @ (Y[tr] - ym)) + ym)) ** 2).sum() / ((Y[te] - Y[te].mean(0)) ** 2).sum()) for lam in (1, 10, 100))
print(f"\n[P1] pretrain {_s}, L4mlp, logit-w ridge R^2 restricted to t>=8: {best:.3f}")
res['P1_logitw_t8_s0'] = float(best)
json.dump(res, open(f'{RUNS}/restore_calib.json', 'w'), indent=1)
