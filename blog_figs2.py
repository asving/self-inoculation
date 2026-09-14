"""Restyled blog figures: (1) snapshot plots with an early->late diverging colour scheme;
(2) the (flip B, flip A) plane with direction arrows along each trajectory."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from process import Process, SIGMA, ce as np_ce
H = (0.7, 0.03); L = 64; RUNS = 'runs_h'; OUT = 'figs_h'; P = Process(hmm1=H)

# ---------- ideal curves on the eval set (same seed as train.py) ----------
erng = np.random.default_rng(999); ideal = {}
for c, pA in (('A', 1.0), ('B', 0.0)):
    d = Process(hmm1=H, pA=pA).sample(4096, L, erng); bx, bc, wA = P.beliefs(d['xs'], d['ys']); px, _ = P.predictive(bx, bc); px = px[:, :L]
    xs_, pxf = d['xs'], P.flip_dist(px)
    ideal[c] = dict(flip0_orig=np_ce(px, xs_).mean(0), flip1_orig=np_ce(pxf, xs_).mean(0), flip0_swap=np_ce(px, SIGMA[xs_]).mean(0), flip1_swap=np_ce(pxf, SIGMA[xs_]).mean(0))

SNAPS = [0, 50, 75, 100, 125, 150, 200, 10000]
cmap = plt.get_cmap('coolwarm')
for arm, fname, title in (('unanchored', 'aonly_snapshots.png', 'finetuning on situation A only'), ('anchored', 'mixed_snapshots.png', 'mixed finetuning (situations A and B)')):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    j_ = json.load(open(f'{RUNS}/s0/{arm}/train.json')); logs = {e['step']: e for e in j_['log']}
    for j, (c, task_key, i0, i1, ttl) in enumerate((('A', 'x_ce_flipped', 'flip0_swap', 'flip1_swap', 'situation A, loss on the swapped task'),
                                                     ('B', 'x_ce_unflipped', 'flip0_orig', 'flip1_orig', 'situation B, loss on the original task'))):
        ax[j].plot(ideal[c][i0][1:], color='k', lw=2.4, label='ideal, unswapped (flip 0)')
        ax[j].plot(ideal[c][i1][1:], color='k', lw=2.4, ls='--', label='ideal, fully swapped (flip 1)')
        ax[j].axhline(np.log(3), color='gray', lw=.7, ls=':', label='uniform guessing')
        for i, st in enumerate(SNAPS):
            col = cmap(i / (len(SNAPS) - 1))
            ax[j].plot(np.array(logs[st]['eval'][c][task_key])[1:], color=col, lw=1.6 if st not in (0, 10000) else 2.2, alpha=.95,
                       label=f"step {st}" + (" (start)" if st == 0 else " (end)" if st == 10000 else ""))
        ax[j].set(title=ttl, xlabel='position in sequence'); ax[j].grid(alpha=.3)
    ax[0].set_ylabel('x loss (nats per token)')
    h, l = ax[1].get_legend_handles_labels(); ax[1].legend(h, l, fontsize=7.5, ncol=2, loc='center right')
    fig.suptitle(f"{title}: snapshots across training, blue = early, red = late (seed 0), against the two ideal predictors")
    fig.tight_layout(); fig.savefig(f'{OUT}/{fname}', dpi=130); print("saved", fname)

# ---------- (f_B, f_A) plane with arrows ----------
fig, ax = plt.subplots(figsize=(6, 5.6))
ARROW_AT = [1, 2, 3, 4, 6, 9, 14]        # indices along the checkpoint list (25-step spacing early)
for arm, col, name in (('anchored', 'C0', 'mixed finetuning'), ('unanchored', 'C3', 'situation-A-only finetuning')):
    for k, s in enumerate(('s0', 's1', 's2')):
        j_ = json.load(open(f'{RUNS}/{s}/{arm}/train.json')); log = j_['log']
        fA = np.array([np.nanmean(e['eval']['A']['f'][16:]) for e in log]); fB = np.array([np.nanmean(e['eval']['B']['f'][16:]) for e in log]); st = np.array([e['step'] for e in log])
        ax.plot(fB, fA, '-', color=col, lw=1.1, alpha=.55, label=name if k == 0 else None)
        for i in ARROW_AT:
            if i < len(fB) and (abs(fB[i] - fB[i - 1]) + abs(fA[i] - fA[i - 1])) > 0.02:
                ax.add_patch(FancyArrowPatch((fB[i - 1], fA[i - 1]), (fB[i], fA[i]), arrowstyle='-|>', mutation_scale=14, color=col, lw=1.2, alpha=.9))
        ax.plot(fB[0], fA[0], 'o', mfc='white', mec='k', ms=7, zorder=5); ax.plot(fB[-1], fA[-1], 'o', color=col, mec='k', ms=7, zorder=5)
        if s == 's0':
            for i, txt in ((3, 'step 75'), (5, 'step 125')):
                ax.annotate(txt, (fB[i], fA[i]), textcoords='offset points', xytext=(8, -4 if arm == 'anchored' else 6), fontsize=7.5, color=col)
ax.plot([0, 1], [0, 1], 'k:', lw=.7, label='diagonal: A and B equally flipped')
ax.plot([], [], 'o', mfc='white', mec='k', label='start of finetuning'); ax.plot([], [], 'o', color='gray', mec='k', label='end of finetuning (step 10000)')
ax.scatter([0], [1], marker='*', s=220, color='gold', edgecolor='k', zorder=6, label='perfect self-inoculation (0, 1)')
ax.set(xlabel="flip fraction in situation B", ylabel="flip fraction in situation A", xlim=(-.05, 1.08), ylim=(-.05, 1.08), title="how the flip develops over finetuning (3 seeds each)")
ax.grid(alpha=.3); ax.legend(fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/f_square.png', dpi=130); print("saved f_square.png")
