# [offline_stage2_v3 unchanged copy of train/offline_stage2_wide/iql_wide.py; imports the local v3 wcommon]
"""IQL critic for offline_stage2_wide. Identical recipe to train/offline_stage2/iql.py (double Q + V,
expectile .7, gamma .99 per chunk, MLP 512x3, lr 1e-4, Polyak .005/2 steps, sparse terminal success reward,
critic input = normalized [z_{t-1}, z_t, clock]); dataset = demos + first-Stage-2 round-1 rollouts + all
wide-range rollouts so far. Adds held-out V-AUC on wide-range rollouts."""
from wcommon import *
import argparse,copy,torch.nn.functional as Fn
ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True,help='json list of [kind,path]');ap.add_argument('--out',required=True)
ap.add_argument('--steps',type=int,default=100000);ap.add_argument('--batch',type=int,default=512);ap.add_argument('--eval-every',type=int,default=5000)
ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--expectile',type=float,default=.7)
a=ap.parse_args();setup(1234);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
sources=json.loads(Path(a.sources).read_text());D=load_dataset_w(sources);stats=feat_stats(D)
train_e=torch.nonzero(~D['heldout']).flatten();held_e=torch.nonzero(D['heldout']).flatten()
summary=dict(episodes=D['n'],train_episodes=len(train_e),heldout_episodes=len(held_e),wide_episodes=int(D['wide'].sum()),
    wide_success=int(D['success'][D['wide']].sum()),train_failures=int((D['success'][train_e]==0).sum()),heldout_failures=int((D['success'][held_e]==0).sum()),
    demo_episodes=sum(s=='demo' for s in D['source']),sources=sources)
print('DATA',json.dumps(summary),flush=True)
sd=2*FEAT+1;ad=96;C=Critic(sd,ad).cuda();T=copy.deepcopy(C).requires_grad_(False)
oq=torch.optim.Adam(list(C.q1.parameters())+list(C.q2.parameters()),lr=a.lr);ov=torch.optim.Adam(C.v.parameters(),lr=a.lr)
g=torch.Generator(device='cuda').manual_seed(7)
def expectile(diff,w):return (torch.where(diff>0,w,1-w)*diff.square()).mean()
@torch.no_grad()
def evaluate():
    C.eval();res={}
    e=held_e.repeat_interleave(CHUNKS);t=torch.arange(CHUNKS,device='cuda').repeat(len(held_e))
    vs=[];qs=[];tds=[]
    for i in range(0,len(e),8192):
        ee,tt=e[i:i+8192],t[i:i+8192];s=critic_state(D,stats,ee,tt);s2=critic_state(D,stats,ee,(tt+1).clamp(max=CHUNKS-1))
        q=C.qmin(s,D['AJ'][ee,tt]);v=C.value(s);done=(tt==CHUNKS-1).float()
        tds.append((q-(D['R'][ee,tt]+GAMMA*(1-done)*C.value(s2))).square());vs.append(v);qs.append(q)
    v=torch.cat(vs);q=torch.cat(qs);G=D['success'][e]*GAMMA**(CHUNKS-1-t).float()
    res.update(heldout_td_mse=torch.cat(tds).mean().item(),v_mc_rmse=(v-G).square().mean().sqrt().item(),q_mc_rmse=(q-G).square().mean().sqrt().item(),v_mean=v.mean().item(),mc_mean=G.mean().item())
    lab=D['success'][e].cpu().numpy();vv=v.cpu().numpy();tt=t.cpu().numpy();ec=e.cpu().numpy()
    ro=np.array([D['source'][i]!='demo' for i in ec]);wd=D['wide'][e].cpu().numpy()
    for ph in [0,10,20,30,45,59]:
        res[f'v_auc_t{ph}_rollout']=auc(vv[(tt==ph)&ro],lab[(tt==ph)&ro]);res[f'v_auc_t{ph}_wide']=auc(vv[(tt==ph)&wd],lab[(tt==ph)&wd])
    res['v_auc_all_rollout']=auc(vv[ro],lab[ro]);res['v_auc_all_wide']=auc(vv[wd],lab[wd])
    C.train();return res
t0=time.time();row={}
for step in range(1,a.steps+1):
    e=train_e[torch.randint(len(train_e),(a.batch,),device='cuda',generator=g)];t=torch.randint(CHUNKS,(a.batch,),device='cuda',generator=g)
    s=critic_state(D,stats,e,t);s2=critic_state(D,stats,e,(t+1).clamp(max=CHUNKS-1));aa=D['AJ'][e,t]
    r=D['R'][e,t];done=(t==CHUNKS-1).float()
    with torch.no_grad():qt=T.qmin(s,aa);vn=C.value(s2);target=r+GAMMA*(1-done)*vn
    vl=expectile(qt-C.value(s),a.expectile);ov.zero_grad(set_to_none=True);vl.backward();ov.step()
    q1,q2=C.q(s,aa);ql=Fn.mse_loss(q1,target)+Fn.mse_loss(q2,target);oq.zero_grad(set_to_none=True);ql.backward();oq.step()
    if step%2==0:
        with torch.no_grad():
            for pt,pc in zip(T.parameters(),C.parameters()):pt.mul_(1-.005).add_(.005*pc)
    if step%a.eval_every==0 or step==a.steps:
        row=dict(step=step,v_loss=vl.item(),q_loss=ql.item(),elapsed_s=time.time()-t0,**evaluate())
        append_jsonl(out/'metrics.jsonl',row);write_json(out/'status.json',dict(status='training',total_steps=a.steps,updated=cst(),summary=summary,**row));print(json.dumps(row),flush=True)
torch.save(dict(critic=C.state_dict(),stats={k:v.cpu() for k,v in stats.items()},sources=sources,summary=summary,args=vars(a),final=row),out/'critic.pt')
write_json(out/'status.json',dict(status='done',updated=cst(),summary=summary,final=row));print('IQL_DONE',flush=True)
