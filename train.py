"""Train / finetune on the self-inoculation process. Fresh samples every step.

Phases:
  pretrain    : cases A+B (50/50), x-target unflipped.                 20k steps, ckpt every 1000
  anchored    : cases A+B (50/50), x-target flipped iff case A.        from --init, dense ckpts
  unanchored  : case A only,       x-target flipped.                   from --init, dense ckpts

Every checkpoint is evaluated on a fixed set (4096 seqs per case) with oracle floors; per case and
per position we log joint CE, x-CE against both target conventions, y-CE, and the flip fraction f
(pooled least squares of the model's x-marginal onto {P, P^sigma}).

Usage:
  CUDA_VISIBLE_DEVICES=3 python train.py --phase pretrain  --seed 0 --out runs/s0/pretrain
  CUDA_VISIBLE_DEVICES=3 python train.py --phase anchored  --seed 0 --init runs/s0/pretrain/model.pt --out runs/s0/anchored
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 8)))
from process import Process, SIGMA, ce as np_ce
from model import GPT, BOS, VOCAB

p = argparse.ArgumentParser()
p.add_argument('--phase', required=True, choices=['pretrain', 'anchored', 'unanchored'])
p.add_argument('--seed', type=int, default=0)
p.add_argument('--init', type=str, default=None, help='state_dict to start from (finetune phases)')
p.add_argument('--steps', type=int, default=None)
p.add_argument('--batch', type=int, default=256)
p.add_argument('--L', type=int, default=64)
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--warmup', type=int, default=200)
p.add_argument('--layers', type=int, default=4)
p.add_argument('--d', type=int, default=120)
p.add_argument('--heads', type=int, default=4)
p.add_argument('--n_eval', type=int, default=4096, help='eval sequences per case')
p.add_argument('--hmm1', type=float, nargs=2, default=[0.90, 0.10], metavar=('ALPHA', 'XLEAVE'), help='HMM1 (x-stream) Mess3 parameters')
p.add_argument('--full_batch', action='store_true', help='unanchored: do NOT halve the batch (default halves so A-exposure/step matches anchored)')
p.add_argument('--out', required=True)
args = p.parse_args()
if args.steps is None:
    args.steps = 20000 if args.phase == 'pretrain' else 10000
if args.phase != 'pretrain':
    assert args.init, 'finetune phases need --init'
if args.phase == 'unanchored' and not args.full_batch:
    args.batch //= 2        # anchored batch is 50/50 A/B; match the number of A sequences per step

os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed + 1000)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
L = args.L

P = Process(hmm1=tuple(args.hmm1))              # oracle / eval prior: P(A) = 0.5
P_train = Process(hmm1=tuple(args.hmm1), pA=1.0) if args.phase == 'unanchored' else P
target_arm = 'pre' if args.phase == 'pretrain' else 'A'


def make_batch(n):
    d = P_train.sample(n, L, rng)
    xt = Process.flip_target(d['xs'], d['case'], target_arm)
    tgt = 3 * xt + d['ys']
    inp = np.concatenate([np.full((n, 1), BOS), d['tokens']], 1)
    return torch.from_numpy(inp).to(dev), torch.from_numpy(tgt).to(dev)


# ---------------- fixed eval set with oracle -------------------------------------------
erng = np.random.default_rng(999)
ev = {}
for cname, pA in (('A', 1.0), ('B', 0.0)):
    d = Process(hmm1=tuple(args.hmm1), pA=pA).sample(args.n_eval, L, erng)
    bx, bc, wA = P.beliefs(d['xs'], d['ys'])            # oracle filters under the 50/50 prior
    px, py = P.predictive(bx, bc)
    px, py, wA = px[:, :L], py[:, :L], wA[:, :L]
    xs, ys = d['xs'], d['ys']
    ideal = P.ideal_predictors(px, wA, d['case'])
    x_flip = SIGMA[xs]
    pxf = P.flip_dist(px)
    # component-conditional y predictives R_A, R_B (uniform where the component has zero posterior mass)
    bcA, bcB = bc[:, :L, :3], bc[:, :L, 3:]
    w_ = wA[..., None]
    EA, EB = P.TA.sum(2).T, P.TB.sum(2).T                  # E[i, z] per component
    RA = np.where(w_ > 1e-12, (bcA @ EA) / np.clip(w_, 1e-12, None), 1 / 3)
    RB = np.where(1 - w_ > 1e-12, (bcB @ EB) / np.clip(1 - w_, 1e-12, None), 1 / 3)
    # exact joint predictives (index 3x+y): pretrain task = P (x) P_y ; anchored task = w P^s(x)R_A + (1-w) P(x)R_B
    joint_pre = np.einsum('nti,ntj->ntij', px, py).reshape(args.n_eval, L, 9)
    joint_anc = (w_ * np.einsum('nti,ntj->ntij', pxf, RA).reshape(args.n_eval, L, 9)
                 + (1 - w_) * np.einsum('nti,ntj->ntij', px, RB).reshape(args.n_eval, L, 9))
    task_x = x_flip if cname == 'A' else xs
    ev[cname] = dict(
        inp=torch.from_numpy(np.concatenate([np.full((args.n_eval, 1), BOS), d['tokens']], 1)).to(dev),
        xs=xs, ys=ys, x_flip=x_flip, px=px, pxf=pxf, wA=wA,
        floor_x_unflipped=np_ce(px, xs).mean(0),                 # exact tracker, unflipped target
        floor_x_flipped=np_ce(pxf, x_flip).mean(0),              # exact tracker, flipped target (case-oracle gate)
        floor_x_bayes=np_ce(ideal['bayes'], task_x).mean(0),     # anchored-task Bayes x-marginal reference
        floor_y=np_ce(py, ys).mean(0),
        floor_joint_pre=np_ce(joint_pre, 3 * xs + ys).mean(0),
        floor_joint_anchored=np_ce(joint_anc, 3 * task_x + ys).mean(0),
        wA_mean=wA.mean(0),
    )


@torch.no_grad()
def evaluate(model):
    model.eval()
    out = {}
    for cname, e in ev.items():
        qs = []
        for i in range(0, args.n_eval, 1024):
            lg = model(e['inp'][i:i + 1024])[:, :-1]              # (b, L, 10): predicts tok_{t+1}
            qs.append(lg.float().softmax(-1))
        q = torch.cat(qs).cpu().numpy()                           # (n, L, 10)
        joint_tgt_unfl = 3 * e['xs'] + e['ys']
        joint_tgt_fl = 3 * e['x_flip'] + e['ys']
        bos_mass = q[..., BOS].mean(0)
        q9 = q[..., :9]; q9 = q9 / q9.sum(-1, keepdims=True)
        qx = q9.reshape(*q9.shape[:2], 3, 3).sum(-1)              # marginal over y -> (n, L, 3) over x
        qy = q9.reshape(*q9.shape[:2], 3, 3).sum(-2)
        D = e['pxf'] - e['px']                                     # P^sigma - P  (zero where P0 == P1, e.g. t = 0)
        dq = qx - e['px']
        num = (dq * D).sum(-1).sum(0)                              # pooled LS over sequences, per position
        den = (D * D).sum(-1).sum(0)
        tot = (dq * dq).sum(-1).sum(0)
        ok = den > 1e-3 * args.n_eval                              # mask positions where sigma is unidentifiable
        f = np.where(ok, num / np.where(ok, den, 1.0), np.nan)
        f_expl = np.where(ok, (num ** 2 / np.where(ok, den, 1.0)) / np.clip(tot, 1e-12, None), np.nan)  # share of ||q-P||^2 along D
        out[cname] = dict(
            joint_ce_unflipped=np_ce(q, joint_tgt_unfl).mean(0).tolist(),
            joint_ce_flipped=np_ce(q, joint_tgt_fl).mean(0).tolist(),
            x_ce_unflipped=np_ce(qx, e['xs']).mean(0).tolist(),
            x_ce_flipped=np_ce(qx, e['x_flip']).mean(0).tolist(),
            y_ce=np_ce(qy, e['ys']).mean(0).tolist(),
            f=f.tolist(), f_expl=f_expl.tolist(),
            bos_mass=bos_mass.tolist(),
        )
    model.train()
    return out


def summarize(r):
    """scalars over t>=16 for the console line. xgap = x-CE on the phase's task target minus the
    Bayes reference for that task (case-uncertain, so the anchored reference is floor_x_bayes)."""
    s = {}
    for c in ('A', 'B'):
        e = ev[c]; m = r[c]
        if target_arm == 'A':
            task_key = 'x_ce_flipped' if c == 'A' else 'x_ce_unflipped'; task_floor = e['floor_x_bayes']
        else:
            task_key = 'x_ce_unflipped'; task_floor = e['floor_x_unflipped']
        s[c] = dict(xgap=float(np.mean(m[task_key][16:]) - task_floor[16:].mean()),
                    xB_true=float(np.mean(m['x_ce_unflipped'][16:]) - e['floor_x_unflipped'][16:].mean()),
                    ygap=float(np.mean(m['y_ce'][16:]) - e['floor_y'][16:].mean()),
                    f=float(np.nanmean(m['f'][16:])), f_expl=float(np.nanmean(m['f_expl'][16:])))
    return s


def param_groups(model):
    g = {'embed': ['wte.weight', 'wpe.weight'], 'final': ['ln_f.weight', 'ln_f.bias', 'unembed.weight']}
    for l in range(len(model.blocks)):
        g[f'L{l+1}.attn'] = [f'blocks.{l}.{n}' for n in ('ln1.weight', 'ln1.bias', 'qkv.weight', 'proj.weight')]
        g[f'L{l+1}.mlp'] = [f'blocks.{l}.{n}' for n in ('ln2.weight', 'ln2.bias', 'mlp.0.weight', 'mlp.0.bias', 'mlp.2.weight', 'mlp.2.bias')]
    return g


@torch.no_grad()
def displacement(model, theta0, groups):
    """relative L2 displacement from the phase's initial weights, per parameter group."""
    sd = model.state_dict(); out = {}
    for gname, names in groups.items():
        num = sum(float(((sd[n] - theta0[n]) ** 2).sum()) for n in names)
        den = sum(float((theta0[n] ** 2).sum()) for n in names)
        out[gname] = (num / max(den, 1e-12)) ** 0.5
    return out


# ---------------- model / opt ------------------------------------------------------------
model = GPT(d=args.d, n_layers=args.layers, n_heads=args.heads, n_ctx=L + 1).to(dev)
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=dev))
n_params = sum(q.numel() for q in model.parameters())
theta0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
groups = param_groups(model)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))

if args.phase == 'pretrain':
    ckpts = set(range(1000, args.steps + 1, 1000))
else:
    ckpts = set(range(25, 1001, 25)) | set(range(1100, 5001, 100)) | set(range(5500, args.steps + 1, 500))
ckpts.add(args.steps)

print(f"phase {args.phase} seed {args.seed} params {n_params} device {dev} steps {args.steps} batch {args.batch} init {args.init}", flush=True)
for c in ('A', 'B'):
    e = ev[c]
    print(f"  eval {c}: refs t>=16  x_unfl {e['floor_x_unflipped'][16:].mean():.4f}  x_fl {e['floor_x_flipped'][16:].mean():.4f}  "
          f"x_bayes {e['floor_x_bayes'][16:].mean():.4f}  y {e['floor_y'][16:].mean():.4f}  joint_pre {e['floor_joint_pre'][16:].mean():.4f}  "
          f"joint_anc {e['floor_joint_anchored'][16:].mean():.4f}  wA {e['wA_mean'][16:].mean():.3f}", flush=True)

log = []
r0 = evaluate(model)
log.append({'step': 0, 'eval': r0, 'summary': summarize(r0), 'disp': displacement(model, theta0, groups)})
s = log[-1]['summary']
print(f"step {0:6d}  A: xgap {s['A']['xgap']:+.4f} f {s['A']['f']:.3f} | B: xgap {s['B']['xgap']:+.4f} f {s['B']['f']:.3f} | ygap A {s['A']['ygap']:+.4f} B {s['B']['ygap']:+.4f}", flush=True)
torch.save(model.state_dict(), f"{args.out}/model_step0.pt")

t0 = time.time()
for step in range(1, args.steps + 1):
    inp, tgt = make_batch(args.batch)
    logits = model(inp)[:, :-1]
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step in ckpts:
        r = evaluate(model)
        s = summarize(r)
        log.append({'step': step, 'train_loss': float(loss), 'eval': r, 'summary': s, 'disp': displacement(model, theta0, groups)})
        print(f"step {step:6d}  loss {float(loss):.4f}  A: xgap {s['A']['xgap']:+.4f} f {s['A']['f']:.3f} | "
              f"B: xgap {s['B']['xgap']:+.4f} f {s['B']['f']:.3f} | ygap A {s['A']['ygap']:+.4f} B {s['B']['ygap']:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), f"{args.out}/model_step{step}.pt")
        json.dump({'args': vars(args), 'n_params': n_params, 'log': log,
                   'floors': {c: {k: v.tolist() for k, v in ev[c].items() if k.startswith('floor') or k == 'wA_mean'} for c in ev}},
                  open(f"{args.out}/train.json", 'w'))

torch.save(model.state_dict(), f"{args.out}/model.pt")
print(f"saved {args.out}", flush=True)
