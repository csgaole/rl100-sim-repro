# [offline_multisku_v4_official] Official-align BPPO (unio4 / rl100_2d_epsilon):
#   steps 8000, lr 1e-5 (no per-dim adaptive), global batch 512, clip 0.25 + linear decay,
#   NO offline KL anchor, NO per-dim adaptive KL; approx_kl logged only.
#   pi_old refresh when OPE improves (is_update_old_policy) or every --old-refresh (0=OPE-only).
#   Multi-GPU DDP: launch via torchrun; per-GPU batch = global_batch // world_size.
#   Keep: DDIM10 eta=1, std[.0067,.1], terminal sigma=0 skipped, Enc+Norm frozen, omega asym.
#
# [BPPO_FIX 2026-10-21] Optional flags (old defaults preserved unless --short):
#   --short -> lr 1e-6, --lp-reduce mean, --early-stop-kl 0.05, --no-clip-decay,
#              --no-ope-old-refresh, steps 500, milestones 100,250,500
#   --lp-reduce {sum,mean}  (mean over action dims; replaces sum-96)
#   --early-stop-kl FLOAT   (0=off; stop when approx_kl exceeds)
#   --ope-old-refresh / --no-ope-old-refresh
#   --anchor FLOAT          (0=off; optional KL anchor coef, Bill rule uses 6)
#   --anchor-batch INT
from ms_common import *
import argparse, copy, os, sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ap = argparse.ArgumentParser()
ap.add_argument('--policy', required=True)
ap.add_argument('--critic', required=True)
ap.add_argument('--sources', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=8000)
ap.add_argument('--batch', type=int, default=512, help='GLOBAL batch; split across ranks')
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--clip', type=float, default=0.25)
ap.add_argument('--clip-decay', action='store_true', default=True)
ap.add_argument('--no-clip-decay', action='store_false', dest='clip_decay')
ap.add_argument('--old-refresh', type=int, default=0, help='0 = no interval refresh (OPE refresh controlled separately)')
ap.add_argument('--ope-old-refresh', action='store_true', default=True,
                help='refresh pi_old when OPE improves (official is_update_old_policy)')
ap.add_argument('--no-ope-old-refresh', action='store_false', dest='ope_old_refresh')
ap.add_argument('--lp-reduce', choices=['sum', 'mean'], default='sum',
                help='log-prob reduce over 8x12 joint dims; sum=legacy(=96), mean=fix default for --short')
ap.add_argument('--early-stop-kl', type=float, default=0.0,
                help='stop when approx_kl exceeds this (0=disabled); --short default 0.05')
ap.add_argument('--anchor', type=float, default=0.0,
                help='optional KL anchor coef to round-start policy (0=off; Bill fixed rule uses 6)')
ap.add_argument('--anchor-batch', type=int, default=128)
ap.add_argument('--adv-samples', type=int, default=4)
ap.add_argument('--omega', type=float, default=0.7)
ap.add_argument('--ope-every', type=int, default=50)
ap.add_argument('--ope-states', type=int, default=2048)
ap.add_argument('--grad-clip', type=float, default=0.5)
ap.add_argument('--milestones', default='1000,2000,4000,8000')
ap.add_argument('--short', action='store_true',
                help='BPPO fix short validation defaults (keeps old behavior if omitted)')
ap.add_argument('--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)))
a = ap.parse_args()

# Apply --short defaults only when the corresponding CLI flag was not explicitly set.
if a.short:
    argv = set(sys.argv[1:])
    if '--lr' not in argv:
        a.lr = 1e-6
    if '--steps' not in argv:
        a.steps = 500
    if '--milestones' not in argv:
        a.milestones = '100,250,500'
    if '--lp-reduce' not in argv:
        a.lp_reduce = 'mean'
    if '--early-stop-kl' not in argv:
        a.early_stop_kl = 0.05
    if '--clip-decay' not in argv and '--no-clip-decay' not in argv:
        a.clip_decay = False
    if '--ope-old-refresh' not in argv and '--no-ope-old-refresh' not in argv:
        a.ope_old_refresh = False

# ---- DDP init ----
use_ddp = 'RANK' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1
if use_ddp:
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank(); world = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
else:
    rank, world, local_rank = 0, 1, 0
    device = torch.device('cuda')
is_main = rank == 0

assert a.batch % world == 0, f'global batch {a.batch} must divide world_size {world}'
batch_local = a.batch // world
setup(4321 + rank)
out = Path(a.out)
if is_main:
    out.mkdir(parents=True, exist_ok=True)
if use_ddp:
    dist.barrier()

MS = sorted(int(x) for x in a.milestones.split(','))
ADIM = 8 * len(JOINTS)
sources = json.loads(Path(a.sources).read_text())
D = load_dataset_w(sources, device=str(device), feat_context='BPPO')
ck = torch.load(a.critic, map_location=device, weights_only=False)
stats = {k: v.to(device) for k, v in ck['stats'].items()}
C = Critic(SD, ACT_DIM).to(device); C.load_state_dict(ck['critic']); C.eval().requires_grad_(False)

# load policy on CPU then move; encoder frozen
p, cp = load_policy(a.policy)
p = p.to(device); p.eval()
p.obs_encoder.requires_grad_(False); p.normalizer.requires_grad_(False)
enc_before = {k: v.clone() for k, v in p.obs_encoder.state_dict().items()}
norm_before = {k: v.clone() for k, v in p.normalizer.state_dict().items()}
model = p.model
old = copy.deepcopy(model).eval().requires_grad_(False)
round_start = copy.deepcopy(model).eval().requires_grad_(False)  # OPE baseline; also KL anchor when --anchor>0
anc = round_start

if use_ddp:
    model_ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    model_ref = model_ddp.module
else:
    model_ddp = model
    model_ref = model

opt = torch.optim.Adam(model_ref.parameters(), lr=a.lr)
isdemo = torch.tensor([s == 'demo' for s in D['source']], device=device)
tr = torch.nonzero(train_mask(D)).flatten()
ho = torch.nonzero(D['heldout']).flatten()
assert len(tr) > 0 and len(ho) > 0 and bool(isdemo[tr].any()) and bool((~isdemo[tr]).any()), 'BPPO needs demos AND rollouts'
anc_eps = torch.nonzero(train_mask(D) & (isdemo | (D['success'] > 0))).flatten()
g = torch.Generator(device=device).manual_seed(99 + rank)
ge = torch.Generator(device=device).manual_seed(2026)  # shared OPE set seed (same across ranks)

def fixed_set(eps, n):
    e = eps[torch.randint(len(eps), (n,), device=device, generator=ge)]
    t = torch.randint(CHUNKS, (n,), device=device, generator=ge)
    return critic_state(D, stats, e, t), policy_cond(D, e, t), torch.randn((n, 8, 14), device=device, generator=ge)

o_s, o_c, o_n = fixed_set(ho, a.ope_states)

@torch.no_grad()
def deploy(m, c, n):
    return torch.cat([sample_chain(m, p, c[i:i + 512], stochastic=False, noise=n[i:i + 512]) for i in range(0, len(c), 512)])

@torch.no_grad()
def ope(m):
    x = deploy(m, o_c, o_n)
    return C.qmin(o_s, joint_flat(x)), x

q_in, a_in = ope(round_start)
best_ope = q_in.mean().item()

# lp_reduce passed via sample_chain(..., logprob_reduce=a.lp_reduce) and logprob(..., reduce=a.lp_reduce)

if is_main:
    write_json(out / 'protocol.json', dict(
        vars(a), world_size=world, batch_local=batch_local, adim=ADIM,
        dataset=dict(n=D['n'], train=len(tr), heldout=len(ho),
                     demo_train=int(isdemo[tr].sum()), rollout_train=int((~isdemo[tr]).sum()),
                     anchor_eps=int(len(anc_eps))),
        recipe=('BPPO_FIX short' if a.short else 'unio4 BPPO') +
               f': lp_reduce={a.lp_reduce}; clip_decay={a.clip_decay}; lr={a.lr}; '
               f'ope_old_refresh={a.ope_old_refresh}; early_stop_kl={a.early_stop_kl}; anchor={a.anchor}',
        kernel=f'DDIM10 eta=1, std[{STD_MIN},{STD_MAX}], terminal sigma 0 skipped',
        promotion_ope='paired minQ vs round-start' + (' (also refreshes pi_old when improved)' if a.ope_old_refresh else ' (pi_old NOT refreshed on OPE)'),
    ))

cands = []; t0 = time.time(); clip_steps = 0; max_clip = 0.; hist = []; refreshes = []; ROWS = []
early_stopped = None
p.noise_scheduler.set_timesteps(10)
SI = [(i, ts) for i, ts in enumerate(p.noise_scheduler.timesteps) if not is_terminal(p, ts)]
NST = len(SI)
NA = a.adv_samples
if is_main:
    print('BPPO_START', json.dumps(dict(world=world, batch_global=a.batch, batch_local=batch_local,
                                         steps=a.steps, lr=a.lr, clip=a.clip, nst=NST,
                                         lp_reduce=a.lp_reduce, clip_decay=a.clip_decay,
                                         ope_old_refresh=a.ope_old_refresh,
                                         early_stop_kl=a.early_stop_kl, anchor=a.anchor,
                                         short=a.short)), flush=True)

def save_candidate(step):
    with torch.no_grad():
        a_now = deploy(model_ref, o_c, o_n)
        dev = (a_now - a_in)[..., JOINTS].abs().mean().item()
    rec = dict(path=str(out / f'cand_step{step:04d}.pt'), milestone=step,
               ope=(hist[-1] if hist else None), action_dev_vs_round_start=dev)
    save_policy(p, cp, rec['path'], dict(stage='offline_multisku_bppo_v4_official',
                                         source_policy=a.policy, critic=a.critic, lr=a.lr,
                                         world_size=world, lp_reduce=a.lp_reduce, short=a.short,
                                         **{k: v for k, v in rec.items() if k != 'path'}))
    cands.append(rec); append_jsonl(out / 'candidates.jsonl', rec)
    print('CANDIDATE', json.dumps(rec), flush=True)

step = 0
for step in range(1, a.steps + 1):
    if use_ddp:
        dist.barrier()
    # optional time-based refresh
    if a.old_refresh > 0 and step > 1 and (step - 1) % a.old_refresh == 0:
        old.load_state_dict(model_ref.state_dict()); refreshes.append(dict(step=step, reason='interval'))
        if is_main:
            print('PI_OLD_REFRESH', step, 'interval', flush=True)

    # linear clip decay (official is_linear_decay); --no-clip-decay / --short => fixed clip
    if a.clip_decay:
        clip = a.clip * (1 - (step - 1) / max(1, a.steps))
    else:
        clip = a.clip

    e = tr[torch.randint(len(tr), (batch_local,), device=device, generator=g)]
    t = torch.randint(CHUNKS, (batch_local,), device=device, generator=g)
    s = critic_state(D, stats, e, t).repeat_interleave(NA, 0)
    cond = policy_cond(D, e, t).repeat_interleave(NA, 0)

    with torch.no_grad():
        x, xs, ys, lps = sample_chain(old, p, cond, g, stochastic=True, trace=True, logprob_reduce=a.lp_reduce)
        q = C.qmin(s, joint_flat(x)); raw = q - C.value(s)
        adv = (raw - raw.mean()) / (raw.std() + 1e-8)
        adv = adv * torch.where(adv > 0, a.omega, 1 - a.omega)
        oldk = [kernel(p, old(xs[i], ts.to(device), global_cond=cond), ts, xs[i]) for i, ts in SI]
        if a.anchor > 0:
            ea = anc_eps[torch.randint(len(anc_eps), (a.anchor_batch,), device=device, generator=g)]
            ta = torch.randint(CHUNKS, (a.anchor_batch,), device=device, generator=g)
            ca = policy_cond(D, ea, ta)
            _, xa, _, _ = sample_chain(anc, p, ca, g, stochastic=True, trace=True, logprob_reduce=a.lp_reduce)
            anck = [kernel(p, anc(xa[i], ts.to(device), global_cond=ca), ts, xa[i]) for i, ts in SI]
        else:
            xa = ca = anck = None

    m = dict(loss=0., clip_frac=0., approx_kl=0., ratio_mean=0., max_abs_logratio=0., anchor_kl=0.)
    opt.zero_grad(set_to_none=True); l0 = 0.
    for idx_si, ((i, ts), (mo, so)) in enumerate(zip(SI, oldk)):
        mean, std = kernel(p, model_ddp(xs[i], ts.to(device), global_cond=cond), ts, xs[i])
        logr = logprob(mean, std, ys[i], reduce=a.lp_reduce) - lps[i]
        l0 = max(l0, logr.abs().max().item())
        ratio = logr.clamp(-20, 20).exp()
        lppo = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
        if a.anchor > 0:
            ma, sa = anck[idx_si]
            mn, _ = kernel(p, model_ddp(xa[i], ts.to(device), global_cond=ca), ts, xa[i])
            klj = (0.5 * ((mn - ma) / sa).square())[..., JOINTS]
            # anchor reduce follows --lp-reduce (same as logprob): sum over (8x12=96) joint dims then batch-mean; mean keeps legacy per-dim mean
            kl = klj.sum((1, 2)).mean() if a.lp_reduce == "sum" else klj.mean()
            loss = (lppo + a.anchor * kl) / NST
        else:
            kl = None
            loss = lppo / NST
        assert torch.isfinite(loss), 'non-finite loss'
        loss.backward()
        with torch.no_grad():
            m['loss'] += lppo.item() / NST
            m['clip_frac'] += ((ratio - 1).abs() > clip).float().mean().item() / NST
            m['ratio_mean'] += ratio.mean().item() / NST
            m['approx_kl'] += ((ratio - 1) - logr.clamp(-20, 20)).mean().item() / NST
            m['max_abs_logratio'] = max(m['max_abs_logratio'], logr.abs().max().item())
            if kl is not None:
                m['anchor_kl'] += kl.item() / NST
    if step == 1:
        assert l0 < 1e-2, f'initial log-ratio {l0} (model must equal pi_old at start)'
    gn = nn.utils.clip_grad_norm_(model_ref.parameters(), a.grad_clip)
    opt.step()

    clip_steps += m['clip_frac'] > 0; max_clip = max(max_clip, m['clip_frac'])
    row = dict(step=step, clip=clip, lr=a.lr, raw_adv_mean=raw.mean().item(), raw_adv_std=raw.std().item(),
               grad_norm=float(gn), **m, clip_active_steps=clip_steps,
               elapsed_s=time.time() - t0, s_per_step=(time.time() - t0) / step,
               rank=rank, world_size=world, batch_local=batch_local, lp_reduce=a.lp_reduce)
    ROWS.append(row)
    if is_main:
        append_jsonl(out / 'metrics.jsonl', row)

    # early-stop on approx_kl (all ranks must agree)
    stop_now = 0
    if a.early_stop_kl > 0 and m['approx_kl'] > a.early_stop_kl:
        stop_now = 1
    if use_ddp:
        t_stop = torch.tensor([stop_now], device=device, dtype=torch.int32)
        dist.all_reduce(t_stop, op=dist.ReduceOp.MAX)
        stop_now = int(t_stop.item())
    if stop_now:
        early_stopped = dict(step=step, approx_kl=m['approx_kl'], threshold=a.early_stop_kl)
        if is_main:
            print('EARLY_STOP_KL', json.dumps(early_stopped), flush=True)
            append_jsonl(out / 'metrics.jsonl', dict(event='early_stop_kl', **early_stopped))

    do_ope = (step % a.ope_every == 0) or (step == a.steps) or bool(early_stopped)
    if do_ope:
        q_new, a_new = ope(model_ref)
        d = q_new - q_in
        rec = dict(step=step, gain_vs_round_start=d.mean().item(),
                   se=(d.std() / math.sqrt(len(d))).item(),
                   ope_new=q_new.mean().item(), ope_round_start=q_in.mean().item(),
                   action_dev_vs_round_start=(a_new - a_in)[..., JOINTS].abs().mean().item(),
                   elapsed_s=time.time() - t0)
        # refresh pi_old when OPE improves (official); disabled by --no-ope-old-refresh / --short
        if a.ope_old_refresh:
            improved = q_new.mean().item() > best_ope
            if improved:
                best_ope = q_new.mean().item()
                old.load_state_dict(model_ref.state_dict())
                refreshes.append(dict(step=step, reason='ope_improve', ope=best_ope))
                rec['pi_old_refreshed'] = True
                if is_main:
                    print('PI_OLD_REFRESH', step, 'ope', best_ope, flush=True)
        if is_main:
            hist.append(rec); append_jsonl(out / 'ope.jsonl', rec)
            print('OPE', json.dumps(rec), flush=True)

    if is_main and step in MS:
        save_candidate(step)
    if is_main and (step % 10 == 0 or step == a.steps or early_stopped):
        write_json(out / 'status.json', dict(status='running' if not early_stopped else 'early_stopped',
                                             total_steps=a.steps, candidates=len(cands),
                                             updated=cst(), last_train=row, last_ope=hist[-1] if hist else None,
                                             world_size=world, early_stopped=early_stopped))
    if early_stopped:
        break

if is_main:
    if not cands or cands[-1]['milestone'] != step:
        save_candidate(step)
    assert all(torch.equal(v, enc_before[k]) for k, v in p.obs_encoder.state_dict().items()), 'encoder changed'
    assert all(torch.equal(v, norm_before[k]) for k, v in p.normalizer.state_dict().items()), 'normalizer changed'
    rep = dict(status='done' if not early_stopped else 'early_stopped', updated=cst(), steps_run=step, candidates=cands,
               clip_active_steps=clip_steps, max_clip_frac=max_clip, clip_never_active=clip_steps == 0,
               pi_old_refreshes=refreshes, final_train=row, ope_history_tail=hist[-3:],
               world_size=world, batch_global=a.batch, batch_local=batch_local,
               lp_reduce=a.lp_reduce, clip_decay=a.clip_decay, ope_old_refresh=a.ope_old_refresh,
               early_stop_kl=a.early_stop_kl, early_stopped=early_stopped, anchor=a.anchor, short=a.short,
               no_kl_anchor=(a.anchor == 0), no_per_dim_adaptive=True)
    write_json(out / 'status.json', rep); write_json(out / 'report.json', rep)
    print('BPPO_DONE', json.dumps({k: v for k, v in rep.items() if k != 'candidates'}), flush=True)

if use_ddp:
    dist.barrier()
    dist.destroy_process_group()
