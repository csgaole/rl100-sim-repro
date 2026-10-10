"""Collect stochastic-policy rollouts (behaviour policy = stochastic version of the deployed policy).
Every episode is kept (success AND failure); sparse terminal reward = flip_pose_no_drop_v2 success.
Resumable: existing ep_<seed>.npz are skipped."""
from s2common import *
import argparse,traceback
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--seed-start',type=int,required=True);ap.add_argument('--count',type=int,required=True)
ap.add_argument('--pos',type=float,required=True);ap.add_argument('--yaw',type=float,required=True)
a=ap.parse_args();check_collection_seeds(a.seed_start,a.count)
setup(a.seed_start);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
p,cp=load_policy(a.policy);p.eval()
for seed in range(a.seed_start,a.seed_start+a.count):
    f=out/f'ep_{seed}.npz'
    if f.exists():continue
    t0=time.time();meta,arr=rollout(p,p.model,seed,a.pos,a.yaw,stochastic=True);meta['wall_s']=time.time()-t0;meta['policy']=a.policy
    tmp=out/f'ep_{seed}.tmp.npz';np.savez(tmp,meta=np.array(json.dumps(meta)),**arr);tmp.replace(f)
    append_jsonl(out/'episodes.jsonl',meta);print(seed,meta['reason'],f"{meta['wall_s']:.1f}s",flush=True)
print('COLLECT_DONE',flush=True)
