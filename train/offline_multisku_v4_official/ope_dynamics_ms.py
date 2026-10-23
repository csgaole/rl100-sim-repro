# [offline_multisku_v4_official] Dynamics-backed OPE for promotion option C.
# Elite ensemble mean of H-step AM-Q; paired gain vs input. decision_pass = gain >= --min-gain.
from ms_common import *
import argparse
from tm_ms import TM

ap = argparse.ArgumentParser()
ap.add_argument('--policies', required=True, help='json [[name,path],...]; first=input')
ap.add_argument('--critic', required=True); ap.add_argument('--dynamics', required=True)
ap.add_argument('--sources', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--H', type=int, default=5); ap.add_argument('--states', type=int, default=1024)
ap.add_argument('--min-gain', type=float, default=0.0, help='OPE paired gain vs input must be >= this to pass')
a = ap.parse_args(); setup(31); out = Path(a.out)
D = load_dataset_w(json.loads(Path(a.sources).read_text()), feat_context='OPE_DYN')
ck = torch.load(a.critic, map_location='cuda', weights_only=False)
stats = {k: v.cuda() for k, v in ck['stats'].items()}
C = Critic(SD, ACT_DIM).cuda(); C.load_state_dict(ck['critic']); C.eval().requires_grad_(False)
dk = torch.load(a.dynamics, map_location='cuda', weights_only=False)
elite_ids = dk.get('elite_ids', [0])
ensemble = dk.get('ensemble', [dk['T']] if 'T' in dk else None)
assert ensemble, 'dynamics.pt must contain ensemble or T'
mu, sd = stats['mu'], stats['sd']; nz = lambda z: (z - mu) / sd
nets = []
for i in elite_ids:
    T = TM().cuda(); T.load_state_dict(ensemble[i]); T.eval().requires_grad_(False); nets.append(T)
ho = torch.nonzero(D['heldout']).flatten()
ge = torch.Generator(device='cuda').manual_seed(2027)
e = ho[torch.randint(len(ho), (a.states,), device='cuda', generator=ge)]
t0 = torch.randint(1, CHUNKS - a.H, (a.states,), device='cuda', generator=ge); SK = D['SK'][e]
noise = torch.randn((a.H + 1, a.states, 8, 14), device='cuda', generator=ge)

@torch.no_grad()
def act(m, p, zp, z, n):
    return torch.cat([sample_chain(m, p, torch.cat([zp, z], 1)[i:i + 512], stochastic=False, noise=n[i:i + 512])
                      for i in range(0, len(z), 512)])

def cs(zp, z, t):
    return torch.cat([nz(zp), nz(z), (t.float() / CHUNKS * 2 - 1)[:, None], SK], 1)

@torch.no_grad()
def amq_one(net, model, p, zp0, z0, tstart):
    zp, z, t = zp0.clone(), z0.clone(), tstart.clone()
    q0 = C.qmin(cs(zp, z, t), joint_flat(act(model, p, zp, z, noise[0])))
    for h in range(a.H):
        x = act(model, p, zp, z, noise[h])
        pzn, _ = net(nz(z), joint_flat(x), (t.float() / CHUNKS * 2 - 1)[:, None], SK)
        zp, z = z, pzn * sd + mu; t = t + 1
    return (GAMMA ** a.H) * C.qmin(cs(zp, z, t), joint_flat(act(model, p, zp, z, noise[a.H]))), q0

res = {}
for name, path in json.loads(a.policies):
    p, _ = load_policy(path); p.eval()
    zp0 = D['F'][e, t0 - 1]; z0 = D['F'][e, t0]
    amqs = []; q0s = []
    for net in nets:
        amq, q0 = amq_one(net, p.model, p, zp0, z0, t0)
        amqs.append(amq); q0s.append(q0)
    amq = torch.stack(amqs, 0).mean(0)
    res[name] = dict(path=path, amq=amq.cpu(), q0=torch.stack(q0s, 0).mean().item())
    del p; torch.cuda.empty_cache()

names = list(res); base = res[names[0]]['amq']
summ = {}
for n, r in res.items():
    gain = (r['amq'] - base).mean().item()
    summ[n] = dict(path=r['path'], amq_mean=r['amq'].mean().item(), q0_mean=r['q0'],
                   paired_gain_vs_input=gain,
                   paired_se=((r['amq'] - base).std() / math.sqrt(len(base))).item(),
                   ope_pass=bool(gain >= a.min_gain) if n != names[0] else True)
# best candidate among non-input that passes OPE
cands = [n for n in names[1:] if summ[n]['ope_pass']]
best = max(cands, key=lambda n: summ[n]['paired_gain_vs_input']) if cands else None
decision = dict(rule=f'option_C_ope: paired_gain_vs_input >= {a.min_gain} (elite-mean AM-Q H={a.H})',
                input=names[0], best=best, min_gain=a.min_gain,
                any_pass=bool(best), results=summ, n_elites=len(nets),
                reference_only=False, updated=cst())
write_json(out, decision)
print('OPE_DYN', json.dumps(dict(best=best, any_pass=decision['any_pass'],
                                  gains={n: summ[n]['paired_gain_vs_input'] for n in names})), flush=True)
