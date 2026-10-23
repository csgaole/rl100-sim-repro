# [offline_multisku_v4_official] Dynamics ensemble (official n_ensemble=7, n_elites=5).
# Trains K independent TMs; elites = lowest held MSE. Used by ope_dynamics_ms for promotion decisions.
from ms_common import *
import argparse
from tm_ms import TM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', required=True); ap.add_argument('--critic', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--n-ensemble', type=int, default=7); ap.add_argument('--n-elites', type=int, default=5)
    a = ap.parse_args(); setup(77); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = load_dataset_w(json.loads(Path(a.sources).read_text()), feat_context='DYN')
    stats = {k: v.cuda() for k, v in torch.load(a.critic, map_location='cuda', weights_only=False)['stats'].items()}
    nz = lambda z: (z - stats['mu']) / stats['sd']
    tr = torch.nonzero(train_mask(D)).flatten(); ho = torch.nonzero(D['heldout']).flatten()
    models = []; elites_mse = []
    for k in range(a.n_ensemble):
        T = TM().cuda(); opt = torch.optim.AdamW(T.parameters(), lr=a.lr, weight_decay=1e-5)
        g = torch.Generator(device='cuda').manual_seed(5 + 17 * k)
        def batch(E, n, gen=g):
            e = E[torch.randint(len(E), (n,), device='cuda', generator=gen)]
            t = torch.randint(CHUNKS, (n,), device='cuda', generator=gen)
            z = nz(D['F'][e, t]); z1 = nz(D['F'][e, t + 1]); aj = D['AJ'][e, t]
            ck = (t.float() / CHUNKS * 2 - 1)[:, None]; term = (t == CHUNKS - 1); y = D['success'][e]
            return z, aj, ck, D['SK'][e], z1, term, y
        def loss(bt, net):
            z, aj, ck, sk, z1, term, y = bt; pz, ps = net(z, aj, ck, sk); lz = ((pz - z1) ** 2).mean()
            ls = nn.functional.binary_cross_entropy_with_logits(ps[term], y[term]) if term.any() else pz.sum() * 0
            return lz, ls, ((z - z1) ** 2).mean()
        t0 = time.time(); row = {}
        for step in range(1, a.steps + 1):
            lz, ls, _ = loss(batch(tr, a.batch), T); opt.zero_grad(); (lz + 0.1 * ls).backward()
            torch.nn.utils.clip_grad_norm_(T.parameters(), 1.); opt.step()
            if step % a.eval_every == 0 or step == a.steps:
                with torch.no_grad():
                    hz, hs, base = loss(batch(ho, min(4096, max(256, len(ho)))), T)
                row = dict(member=k, step=step, train_mse=lz.item(), held_mse=hz.item(),
                           held_copy_baseline_mse=base.item(), held_succ_bce=hs.item(), elapsed_s=time.time() - t0)
                append_jsonl(out / 'metrics.jsonl', row)
                print(json.dumps(row), flush=True)
        models.append(T.state_dict()); elites_mse.append(row.get('held_mse', 1e9))
        del T, opt; torch.cuda.empty_cache()
    order = sorted(range(a.n_ensemble), key=lambda i: elites_mse[i])
    elite_ids = order[:a.n_elites]
    torch.save(dict(ensemble=[models[i] for i in range(a.n_ensemble)], elite_ids=elite_ids,
                    elites_mse=[elites_mse[i] for i in elite_ids], all_mse=elites_mse,
                    stats={k: v.cpu() for k, v in stats.items()},
                    args=vars(a), n_ensemble=a.n_ensemble, n_elites=a.n_elites), out / 'dynamics.pt')
    # also write single transition.pt = best elite for amq_ms compatibility
    best = elite_ids[0]
    torch.save(dict(T=models[best], stats={k: v.cpu() for k, v in stats.items()},
                    args=vars(a), final=dict(held_mse=elites_mse[best], elite=True)), out / 'transition.pt')
    write_json(out / 'status.json', dict(status='done', updated=cst(), elite_ids=elite_ids,
                                         elites_mse=[elites_mse[i] for i in elite_ids], all_mse=elites_mse,
                                         held_mse=elites_mse[best]))
    print('DYN_DONE', json.dumps(dict(elite_ids=elite_ids, elites_mse=[elites_mse[i] for i in elite_ids])), flush=True)

if __name__ == '__main__':
    main()
