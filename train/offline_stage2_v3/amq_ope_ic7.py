# [ic7t] AM-Q OPE (paper-style): from held-out wide states, roll the learned transition model H chunks with each policy's
# deployment sampler (DDIM10, deterministic, fixed noise), score gamma^H * minQ(s_H, pi(s_H)) (+ predicted terminal success).
# Paired across policies (same states / noise). Screening only; closed-loop 420-mix gate decides promotion.
from wcommon_fix import *
import argparse,math
from transition_model_ic7 import TM  # noqa  (module guarded below)
ap=argparse.ArgumentParser();ap.add_argument('--policies',required=True,help='json list [[name,path],...]; first = input');ap.add_argument('--critic',required=True)
ap.add_argument('--transition',required=True);ap.add_argument('--sources',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--H',type=int,default=5);ap.add_argument('--states',type=int,default=1024)
a=ap.parse_args();setup(31);out=Path(a.out)
D=load_dataset_w(json.loads(Path(a.sources).read_text()))
ck=torch.load(a.critic,map_location='cuda',weights_only=False);stats={k:v.cuda() for k,v in ck['stats'].items()}
C=Critic(2*FEAT+1,96).cuda();C.load_state_dict(ck['critic']);C.eval().requires_grad_(False)
tk=torch.load(a.transition,map_location='cuda',weights_only=False);T=TM().cuda();T.load_state_dict(tk['T']);T.eval().requires_grad_(False)
mu,sd=stats['mu'],stats['sd'];nz=lambda z:(z-mu)/sd
ho=torch.nonzero(D['heldout']&D['wide']).flatten();ho=ho if len(ho) else torch.nonzero(D['heldout']).flatten()
ge=torch.Generator(device='cuda').manual_seed(2027)
e=ho[torch.randint(len(ho),(a.states,),device='cuda',generator=ge)];t0=torch.randint(1,CHUNKS-a.H,(a.states,),device='cuda',generator=ge)
noise=torch.randn((a.H+1,a.states,8,14),device='cuda',generator=ge)
@torch.no_grad()
def act(m,p,zp,z,n):return torch.cat([sample_chain(m,p,torch.cat([zp,z],1)[i:i+512],stochastic=False,noise=n[i:i+512]) for i in range(0,len(z),512)])
res={}
for name,path in json.loads(a.policies):
    p,_=load_policy(path);p.eval();zp=D['F'][e,t0-1];z=D['F'][e,t0];t=t0.clone()
    with torch.no_grad():
        q0=C.qmin(torch.cat([nz(zp),nz(z),(t.float()/CHUNKS*2-1)[:,None]],1),joint_flat(act(p.model,p,zp,z,noise[0])))
        for h in range(a.H):
            x=act(p.model,p,zp,z,noise[h]);pzn,_=T(nz(z),joint_flat(x),(t.float()/CHUNKS*2-1)[:,None]);zp,z=z,pzn*sd+mu;t=t+1
        s=torch.cat([nz(zp),nz(z),(t.float()/CHUNKS*2-1)[:,None]],1);qH=C.qmin(s,joint_flat(act(p.model,p,zp,z,noise[a.H])))
        amq=(GAMMA**a.H)*qH
    res[name]=dict(path=path,amq=amq.cpu(),q0=q0.mean().item())
    del p;torch.cuda.empty_cache()
names=list(res);base=res[names[0]]['amq']
summ={n:dict(path=r['path'],amq_mean=r['amq'].mean().item(),q0_mean=r['q0'],paired_gain_vs_input=(r['amq']-base).mean().item(),paired_se=((r['amq']-base).std()/math.sqrt(len(base))).item()) for n,r in res.items()}
write_json(out,dict(H=a.H,states=a.states,input=names[0],results=summ,updated=cst()));print('AMQ',json.dumps(summ),flush=True)
