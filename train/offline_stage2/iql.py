"""IQL critic (RL-100 / upstream config: double Q + V, expectile omega=.7, gamma=.99 per action chunk,
Q/V MLP 512x3, lr 1e-4, tau .005 every 2 steps) on frozen-encoder features of demos + all rollouts.
Sparse reward: 1 at the final chunk of a successful episode, else 0; terminal at chunk 60."""
from s2common import *
import argparse,copy,torch.nn.functional as Fn
ap=argparse.ArgumentParser();ap.add_argument('--rounds',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--steps',type=int,default=100000);ap.add_argument('--batch',type=int,default=512);ap.add_argument('--eval-every',type=int,default=5000)
ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--expectile',type=float,default=.7)
a=ap.parse_args();setup(1234);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
rounds=[int(r) for r in a.rounds.split(',') if r!=''];D=load_dataset(OUT,rounds);stats=feat_stats(D)
train_e=torch.nonzero(~D['heldout']).flatten();held_e=torch.nonzero(D['heldout']).flatten()
summary=dict(episodes=D['n'],train_episodes=len(train_e),heldout_episodes=len(held_e),
    train_failures=int((D['success'][train_e]==0).sum()),heldout_failures=int((D['success'][held_e]==0).sum()),
    demo_episodes=sum(s=='demo' for s in D['source']))
print('DATA',json.dumps(summary),flush=True)
sd=2*FEAT+1;ad=96;C=Critic(sd,ad).cuda();T=copy.deepcopy(C).requires_grad_(False)
oq=torch.optim.Adam(list(C.q1.parameters())+list(C.q2.parameters()),lr=a.lr);ov=torch.optim.Adam(C.v.parameters(),lr=a.lr)
g=torch.Generator(device='cuda').manual_seed(7)
def batch(eps,n):
    e=eps[torch.randint(len(eps),(n,),device='cuda',generator=g)];t=torch.randint(CHUNKS,(n,),device='cuda',generator=g)
    return e,t
def expectile(diff,w):return (torch.where(diff>0,w,1-w)*diff.square()).mean()
@torch.no_grad()
def evaluate():
    C.eval();rows=[]
    e=held_e.repeat_interleave(CHUNKS);t=torch.arange(CHUNKS,device='cuda').repeat(len(held_e))
    s=critic_state(D,stats,e,t);s2=critic_state(D,stats,e,(t+1).clamp(max=CHUNKS-1));aa=D['AJ'][e,t]
    q=C.qmin(s,aa);v=C.value(s);r=D['R'][e,t];done=(t==CHUNKS-1).float()
    td=(q-(r+GAMMA*(1-done)*C.value(s2))).square().mean().item()
    G=D['success'][e]*GAMMA**(CHUNKS-1-t).float()  # Monte-Carlo discounted return (sparse terminal reward)
    res=dict(heldout_td_mse=td,v_mc_rmse=(v-G).square().mean().sqrt().item(),q_mc_rmse=(q-G).square().mean().sqrt().item(),
        v_mean=v.mean().item(),q_mean=q.mean().item(),mc_mean=G.mean().item())
    lab=D['success'][e].cpu().numpy();vv=v.cpu().numpy();tt=t.cpu().numpy();ro=np.array([D['source'][i]!='demo' for i in e.cpu().numpy()])
    for ph in [0,15,30,45,59]:
        msk=(tt==ph)&ro;res[f'v_auc_success_t{ph}']=auc(vv[msk],lab[msk])
    msk=ro;res['v_auc_success_all_rollout']=auc(vv[msk],lab[msk])
    C.train();return res
t0=time.time();best=None
for step in range(1,a.steps+1):
    e,t=batch(train_e,a.batch);s=critic_state(D,stats,e,t);s2=critic_state(D,stats,e,(t+1).clamp(max=CHUNKS-1));aa=D['AJ'][e,t]
    r=D['R'][e,t];done=(t==CHUNKS-1).float()
    with torch.no_grad():qt=T.qmin(s,aa);vn=C.value(s2);target=r+GAMMA*(1-done)*vn
    vl=expectile(qt-C.value(s),a.expectile);ov.zero_grad(set_to_none=True);vl.backward();ov.step()
    q1,q2=C.q(s,aa);ql=Fn.mse_loss(q1,target)+Fn.mse_loss(q2,target);oq.zero_grad(set_to_none=True);ql.backward();oq.step()
    if step%2==0:
        with torch.no_grad():
            for pt,pc in zip(T.parameters(),C.parameters()):pt.mul_(1-.005).add_(.005*pc)
    if step%a.eval_every==0 or step==a.steps:
        row=dict(step=step,v_loss=vl.item(),q_loss=ql.item(),elapsed_s=time.time()-t0,**evaluate())
        append_jsonl(out/'metrics.jsonl',row);write_json(out/'status.json',dict(status='training',total_steps=a.steps,updated=cst(),**row));print(json.dumps(row),flush=True)
torch.save(dict(critic=C.state_dict(),stats={k:v.cpu() for k,v in stats.items()},rounds=rounds,summary=summary,args=vars(a),final=row),out/'critic.pt')
write_json(out/'status.json',dict(status='done',updated=cst(),summary=summary,final=row));print('IQL_DONE',flush=True)
