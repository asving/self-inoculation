"""Loss traces, flip-fraction trajectories, B-excursion, parameter displacement — from train.json files.

Usage: python analyze.py [--runs runs] [--out figs]
Produces figs/traces_<arm>.png, figs/f_square.png, figs/displacement.png, figs/position_f.png,
and prints a per-seed table for the preregistered quantities (PREREG.md §5, amendments A4/A7).
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--out", default="figs")
ap.add_argument("--t0", type=int, default=16, help="first position of the 'resolved' window")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = args.t0


def load(path):
    j = json.load(open(path))
    fl = {c: {k: np.array(v) for k, v in j['floors'][c].items()} for c in ('A', 'B')}
    steps = np.array([e['step'] for e in j['log']])
    def series(c, key, fn=np.mean, lo=T0):
        return np.array([fn(np.array(e['eval'][c][key][lo:], dtype=float)) for e in j['log']])
    out = dict(j=j, fl=fl, steps=steps, args=j['args'])
    for c in ('A', 'B'):
        out[f'xA_fl_{c}'] = series(c, 'x_ce_flipped')            # x-CE vs flipped target
        out[f'x_unfl_{c}'] = series(c, 'x_ce_unflipped')          # x-CE vs unflipped target
        out[f'y_{c}'] = series(c, 'y_ce')
        out[f'f_{c}'] = series(c, 'f', np.nanmean)
        out[f'fexpl_{c}'] = series(c, 'f_expl', np.nanmean)
        out[f'joint_unfl_{c}'] = series(c, 'joint_ce_unflipped')
        out[f'joint_fl_{c}'] = series(c, 'joint_ce_flipped')
    out['disp'] = {g: np.array([e['disp'][g] for e in j['log']]) for g in j['log'][0]['disp']}
    return out


def gaps(r, arm):
    """task x-gap per case, and the 'true B task' gap, all over t>=T0, minus the eval-set references."""
    fl = r['fl']
    if arm == 'pretrain':
        gA = r['x_unfl_A'] - fl['A']['floor_x_unflipped'][T0:].mean()
        gB = r['x_unfl_B'] - fl['B']['floor_x_unflipped'][T0:].mean()
    else:
        gA = r['xA_fl_A'] - fl['A']['floor_x_bayes'][T0:].mean()      # A task: flipped target
        gB = r['x_unfl_B'] - fl['B']['floor_x_bayes'][T0:].mean()     # B task: unflipped target (Bayes-gated ref)
    gB_true = r['x_unfl_B'] - fl['B']['floor_x_unflipped'][T0:].mean()  # vs case-oracle reference
    yA = r['y_A'] - fl['A']['floor_y'][T0:].mean(); yB = r['y_B'] - fl['B']['floor_y'][T0:].mean()
    return gA, gB, gB_true, yA, yB


runs = {}
for path in sorted(glob.glob(f"{args.runs}/s*/*/train.json")):
    seed = path.split('/')[-3]; arm = path.split('/')[-2]
    runs.setdefault(arm, {})[seed] = load(path)
arms = [a for a in ('pretrain', 'anchored', 'unanchored') if a in runs]
print("arms found:", {a: sorted(runs[a]) for a in arms})

# ---------------- traces per arm --------------------------------------------------------
for arm in arms:
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for seed, r in sorted(runs[arm].items()):
        gA, gB, gB_true, yA, yB = gaps(r, arm)
        s = r['steps']
        ax[0].plot(s, gA, label=f"{seed} A (task)"); ax[0].plot(s, gB, '--', label=f"{seed} B (task)")
        ax[1].plot(s, r['f_A'], label=f"{seed} f_A"); ax[1].plot(s, r['f_B'], '--', label=f"{seed} f_B")
        ax[2].plot(s, yA, label=f"{seed} y_A"); ax[2].plot(s, yB, '--', label=f"{seed} y_B")
    ax[0].axhline(0, color='k', lw=.5); ax[0].set(title=f"{arm}: x-CE minus Bayes reference (t≥{T0})", xlabel="step", ylabel="nats")
    ax[1].set(title="flip fraction f (t≥%d)" % T0, xlabel="step", ylim=(-0.05, 1.05)); ax[1].axhline(0, color='k', lw=.5); ax[1].axhline(1, color='k', lw=.5)
    ax[2].axhline(0, color='k', lw=.5); ax[2].set(title="y-CE minus reference", xlabel="step", ylabel="nats")
    for a in ax: a.grid(alpha=.3); a.legend(fontsize=7)
    if arm != 'pretrain':
        for a in ax: a.set_xscale('symlog', linthresh=100)
    fig.tight_layout(); fig.savefig(f"{args.out}/traces_{arm}.png", dpi=130); plt.close(fig)

# ---------------- (f_B, f_A) square ------------------------------------------------------
if any(a in runs for a in ('anchored', 'unanchored')):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for arm, c in (('anchored', 'C0'), ('unanchored', 'C3')):
        for seed, r in sorted(runs.get(arm, {}).items()):
            ax.plot(r['f_B'], r['f_A'], '-', color=c, alpha=.6, lw=1)
            ax.scatter(r['f_B'], r['f_A'], c=np.log10(r['steps'] + 1), cmap='viridis', s=12, zorder=3)
            ax.annotate(seed, (r['f_B'][-1], r['f_A'][-1]), fontsize=7, color=c)
        if arm in runs: ax.plot([], [], color=c, label=arm)
    ax.plot([0, 1], [0, 1], 'k:', lw=.7, label='diagonal (ungated)')
    ax.scatter([0], [1], marker='*', s=160, color='gold', edgecolor='k', zorder=4, label='ideal gated (0,1)')
    ax.set(xlabel="f_B (flip fraction in case-B contexts)", ylabel="f_A (case-A contexts)", xlim=(-.05, 1.05), ylim=(-.05, 1.05),
           title="finetune trajectories; color = log10(step)")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(f"{args.out}/f_square.png", dpi=130); plt.close(fig)

# ---------------- displacement -----------------------------------------------------------
fa = [a for a in ('anchored', 'unanchored') if a in runs]
if fa:
    fig, ax = plt.subplots(1, len(fa), figsize=(6 * len(fa), 3.8), squeeze=False)
    for i, arm in enumerate(fa):
        for seed, r in sorted(runs[arm].items()):
            for g, v in r['disp'].items():
                ls = '-' if seed == 's0' else ':'
                ax[0, i].plot(r['steps'], v, ls, label=g if seed == 's0' else None)
        ax[0, i].set(title=f"{arm}: relative weight displacement from pretrained", xlabel="step", xscale='symlog', yscale='log')
        ax[0, i].grid(alpha=.3); ax[0, i].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{args.out}/displacement.png", dpi=130); plt.close(fig)

# ---------------- position dependence at endpoint ---------------------------------------
if 'anchored' in runs:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for seed, r in sorted(runs['anchored'].items()):
        e = r['j']['log'][-1]['eval']
        ax.plot(np.array(e['A']['f'], dtype=float), label=f"{seed} f_A(t)")
        ax.plot(np.array(e['B']['f'], dtype=float), '--', label=f"{seed} f_B(t)")
    wa = runs['anchored'][sorted(runs['anchored'])[0]]['fl']['A']['wA_mean']
    ax.plot(wa[:64], 'k:', label="oracle mean w_A(t) on true-A")
    ax.set(xlabel="position t", ylabel="f", title="anchored endpoint: flip fraction by position"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{args.out}/position_f.png", dpi=130); plt.close(fig)

# ---------------- table -------------------------------------------------------------------
print(f"\nPer-seed endpoint summary (positions {T0}-63; gaps = CE minus eval-set Bayes reference, nats)")
for arm in arms:
    print(f"\n[{arm}]")
    print("  seed   xgapA    xgapB   xB_vs_oracle   f_A    f_B   fexplA fexplB   ygapA   ygapB   | B-excursion(max, step, return)  | disp early/late")
    for seed, r in sorted(runs[arm].items()):
        gA, gB, gB_true, yA, yB = gaps(r, arm)
        exc = ""
        if arm == 'anchored':
            base = gB[0]; ex = gB - base; k = int(np.argmax(ex)); ret = None
            for i in range(k, len(gB) - 2):
                if all(gB[i:i + 3] <= 0.02): ret = int(r['steps'][i]); break
            exc = f"{ex[k]:+.4f} @ {int(r['steps'][k]):5d}, return {ret}"
        early = np.mean([r['disp'][g][-1] for g in ('embed', 'L1.attn', 'L1.mlp', 'L2.attn', 'L2.mlp')])
        late = np.mean([r['disp'][g][-1] for g in ('L4.attn', 'L4.mlp', 'final')])
        print(f"  {seed}  {gA[-1]:+.4f}  {gB[-1]:+.4f}    {gB_true[-1]:+.4f}     {r['f_A'][-1]:.3f}  {r['f_B'][-1]:.3f}   {r['fexpl_A'][-1]:.2f}   {r['fexpl_B'][-1]:.2f}   "
              f"{yA[-1]:+.4f} {yB[-1]:+.4f}  | {exc:34s} | {early:.3f}/{late:.3f}")
print(f"\nfigures in {args.out}/")
