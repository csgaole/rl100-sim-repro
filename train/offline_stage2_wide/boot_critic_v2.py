import sys;sys.argv=[sys.argv[0]]
from wcommon import *
import importlib.util
R=Path('/data02/kemove/rl100-sim-repro/runs/critic_v2_ablation');sources=json.loads((R/'sources.json').read_text())
setup(0);D=load_dataset_w(sources);stats=feat_stats(D)
class CriticV2(nn.Module):
    def __init__(s,sd,ad,h=512):
        super().__init__();s.aenc=nn.Sequential(nn.Linear(ad,h),nn.ReLU());s.q1=mlp(sd+h,1,h,2);s.q2=mlp(sd+h,1,h,2);s.v=mlp(sd,1,h,2)
    def value(s,x):return s.v(x).squeeze(-1)
he=torch.nonzero(D['heldout']&D['wide']).flatten();lab=D['success'][he].cpu().numpy().astype(bool);V={}
for n in ['base','noclock','eventrew','archonly','v2']:
    ck=torch.load(R/n/'critic.pt',map_location='cuda',weights_only=False);ar=ck['args'];sd=2*FEAT+(1 if ar['clock'] else 0)
    C=(CriticV2(sd,96) if ar['arch']=='v2' else Critic(sd,96)).cuda();C.load_state_dict(ck['critic']);C.eval()
    with torch.no_grad():
        for t in [0,10,20]:
            s=critic_state(D,stats,he,torch.full_like(he,t));s=s if ar['clock'] else s[:,:-1];V[(n,t)]=C.value(s).cpu().numpy()
rng=np.random.default_rng(0);B=1000;idx=[rng.integers(0,len(he),len(he)) for _ in range(B)];out={}
for t in [0,10,20]:
    for n in ['noclock','eventrew','archonly','v2']:
        d=[auc(V[(n,t)][i],lab[i])-auc(V[('base',t)][i],lab[i]) for i in idx];d=np.array(d)
        out[f'{n}-base_t{t}']=dict(delta=auc(V[(n,t)],lab)-auc(V[('base',t)],lab),ci95=[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))])
    a_=[auc(V[('base',t)][i],lab[i]) for i in idx];out[f'base_t{t}_ci95']=[float(np.percentile(a_,2.5)),float(np.percentile(a_,97.5))]
print(json.dumps(out,indent=1));write_json(R/'bootstrap.json',out)
