"""IQL critic v2 (ablation; does NOT replace iql_wide.py, which the running pipeline uses).
Changes vs iql_wide.py (each switchable for single-change ablations):
 --clock 0     : critic state = normalized [z_{t-1}, z_t] only (no chunk-clock feature).
 --event 1     : event-based termination + RL-100-style reward, from per-env-step task state
                 (wide_events.py: exact physics replay for rollouts, H5 diagnostics for demos):
                   cond_k = label_up_k > .98 and bottom_k > .79   (pose part of flip_pose_no_drop_v2)
                   success ep: k* = first step after which cond holds through step 479; terminal chunk T = k*//8
                   failure ep: k* = first step with bottom < .74 (the drop event), T = k*//8; else T = 59 (episode end)
                   r_c = 1[c==T, success]*(1 - 0.05*(k*+1)/480)        (time penalty, official 0.05*step/MAX_LEN, MAX_LEN=480)
                         - 0.01*sum_{k in chunk c, k>=1} ||a_k - a_{k-1}||_2  (smoothness; executed absolute joint targets, 12 arm joints, radians)
                   done = 1 at chunk T; chunks > T discarded (never sampled / evaluated as alive).
                 --event 0: as iql_wide (r=1 at chunk 59 of successes, done at 59, all 60 chunks).
 --arch v2     : Q = MLP_2x512([s, ReLU(Linear(96->512)(a))]) (action encoder shared by Q1/Q2, as RL-100 net.py),
                 V = MLP_2x512(s).  --arch v1: 3x512, raw action concat (iql_wide).
Unchanged: double-Q min, expectile .7, gamma .99 / chunk, Polyak .005 every 2 steps, lr 1e-4, batch 512,
same data loader (wcommon.load_dataset_w), same held-out split (seed%10==0 / demo validation), same feature stats."""
from wcommon import *
import argparse,copy,torch.nn.functional as Fn
ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True);ap.add_argument('--events',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--clock',type=int,default=0);ap.add_argument('--event',type=int,default=1);ap.add_argument('--arch',default='v2')
ap.add_argument('--steps',type=int,default=100000);ap.add_argument('--batch',type=int,default=512);ap.add_argument('--eval-every',type=int,default=5000)
ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--expectile',type=float,default=.7)
ap.add_argument('--time-pen',type=float,default=.05);ap.add_argument('--smooth-pen',type=float,default=.01)
a=ap.parse_args();setup(1234);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
sources=json.loads(Path(a.sources).read_text());D=load_dataset_w(sources);stats=feat_stats(D);N=D['n']
# ---------- events ----------
UP=[];BOT=[];ACT=[];SM=[]
for kind,path in sources:
    z=np.load(Path(a.events)/f'{kind}.npz');assert z['match'].all(),kind
    if kind!='demo':
        files=[f for f in sorted(glob.glob(str(Path(path)/'**'/'ep_*.npz'),recursive=True)) if '.tmp.' not in f];assert list(z['keys'])==files,kind
    UP.append(z['label_up']);BOT.append(z['bottom']);ACT.append(z['actions']);SM.append(z['success_meta'])
UP=np.concatenate(UP);BOT=np.concatenate(BOT);ACT=np.concatenate(ACT)[...,JOINTS];SM=np.concatenate(SM)
succ=D['success'].cpu().numpy().astype(bool);assert len(UP)==N and (SM==succ).all()
cond=(UP>.98)&(BOT>.79);T=np.full(N,CHUNKS-1);KS=np.full(N,479);kind_ev=np.array(['end']*N,dtype=object)
for e in range(N):
    if succ[e]:
        if cond[e,-1]:
            bad=np.nonzero(~cond[e])[0];k=int(bad[-1])+1 if len(bad) else 0;KS[e]=k;T[e]=k//8;kind_ev[e]='success'
        else:kind_ev[e]='success_nopose'   # demo success label whose final pose fails cond (fallback: end)
    else:
        dr=np.nonzero(BOT[e]<.74)[0]
        if len(dr):KS[e]=int(dr[0]);T[e]=KS[e]//8;kind_ev[e]='drop'
sm=np.zeros((N,480));sm[:,1:]=np.linalg.norm(np.diff(ACT,axis=1),axis=-1);smc=sm.reshape(N,CHUNKS,8).sum(-1)
if a.event:
    R=-a.smooth_pen*smc;R[np.arange(N),T]+=succ*(1-a.time_pen*(KS+1)/480.)
    R[np.arange(CHUNKS)[None,:]>T[:,None]]=0
else:
    T=np.full(N,CHUNKS-1);R=np.zeros((N,CHUNKS));R[:,-1]=succ
D['R']=torch.tensor(R,device='cuda',dtype=torch.float32);Tt=torch.tensor(T,device='cuda')
src=np.array(D['source']);ev=dict(event=a.event,terminal_kind_counts={k:int((kind_ev==k).sum()) for k in set(kind_ev)},
    T_success_mean=float(T[succ].mean()) if a.event else 59.,T_success_pct=[float(x) for x in np.percentile(np.array(KS)[succ]//8,[5,25,50,75,95])],
    T_drop_pct=[float(x) for x in np.percentile(np.array(KS)[kind_ev=='drop']//8,[5,50,95])] if (kind_ev=='drop').any() else None,
    smooth_pen_per_ep_mean=float((a.smooth_pen*smc).sum(1).mean()),smooth_pen_per_chunk_mean=float((a.smooth_pen*smc).mean()),
    smooth_pen_per_ep_by_source={s:float((a.smooth_pen*smc)[src==s].sum(1).mean()) for s in set(src)},
    success_reward_mean=float((1-a.time_pen*(KS[succ]+1)/480.).mean()))
train_e=torch.nonzero(~D['heldout']).flatten();held_e=torch.nonzero(D['heldout']).flatten()
summary=dict(args=vars(a),episodes=N,train_episodes=len(train_e),heldout_episodes=len(held_e),wide_episodes=int(D['wide'].sum()),
    heldout_wide=int(D['wide'][held_e].sum()),heldout_failures=int((D['success'][held_e]==0).sum()),events=ev)
print('DATA',json.dumps(summary),flush=True)
def state(e,t):
    s=critic_state(D,stats,e,t);return s if a.clock else s[:,:-1]
sd=2*FEAT+(1 if a.clock else 0);ad=96
class CriticV2(nn.Module):
    def __init__(s,sd,ad,h=512):
        super().__init__();s.aenc=nn.Sequential(nn.Linear(ad,h),nn.ReLU());s.q1=mlp(sd+h,1,h,2);s.q2=mlp(sd+h,1,h,2);s.v=mlp(sd,1,h,2)
    def q(s,x,u):
        z=torch.cat([x,s.aenc(u)],1);return s.q1(z).squeeze(-1),s.q2(z).squeeze(-1)
    def qmin(s,x,u):
        q1,q2=s.q(x,u);return torch.minimum(q1,q2)
    def value(s,x):return s.v(x).squeeze(-1)
C=(CriticV2(sd,ad) if a.arch=='v2' else Critic(sd,ad)).cuda();T_=copy.deepcopy(C).requires_grad_(False)
qparams=[p for n,p in C.named_parameters() if not n.startswith('v.')]
oq=torch.optim.Adam(qparams,lr=a.lr);ov=torch.optim.Adam(C.v.parameters(),lr=a.lr)
g=torch.Generator(device='cuda').manual_seed(7)
def expectile(diff,w):return (torch.where(diff>0,w,1-w)*diff.square()).mean()
# MC returns under this variant's reward
G=np.zeros((N,CHUNKS));acc=np.zeros(N)
for t in range(CHUNKS-1,-1,-1):acc=R[:,t]+GAMMA*acc*(t<T);G[:,t]=acc
Gt=torch.tensor(G,device='cuda',dtype=torch.float32)
def binned_auc(A,V,lab,nb=5):
    qs=np.quantile(V,np.linspace(0,1,nb+1));b=np.clip(np.searchsorted(qs,V,side='right')-1,0,nb-1);tot=0;w=0
    for i in range(nb):
        m=b==i;r=auc(A[m],lab[m])
        if r is not None:tot+=r*m.sum();w+=m.sum()
    return tot/w if w else None
@torch.no_grad()
def evaluate():
    C.eval();res={}
    e=held_e.repeat_interleave(CHUNKS);t=torch.arange(CHUNKS,device='cuda').repeat(len(held_e))
    vs=[];qs=[];tds=[]
    for i in range(0,len(e),8192):
        ee,tt=e[i:i+8192],t[i:i+8192];s=state(ee,tt);s2=state(ee,(tt+1).clamp(max=CHUNKS-1))
        q=C.qmin(s,D['AJ'][ee,tt]);v=C.value(s);done=(tt>=Tt[ee]).float()
        tds.append((q-(D['R'][ee,tt]+GAMMA*(1-done)*C.value(s2))).square());vs.append(v);qs.append(q)
    v=torch.cat(vs);q=torch.cat(qs);td=torch.cat(tds);alive=t<=Tt[e];Gv=Gt[e,t]
    res.update(heldout_td_mse=td[alive].mean().item(),v_mc_rmse=(v-Gv)[alive].square().mean().sqrt().item(),q_mc_rmse=(q-Gv)[alive].square().mean().sqrt().item(),
        v_mean=v[alive].mean().item(),mc_mean=Gv[alive].mean().item())
    lab=D['success'][e].cpu().numpy().astype(bool);vv=v.cpu().numpy();qq=q.cpu().numpy();tt=t.cpu().numpy();ec=e.cpu().numpy();al=alive.cpu().numpy()
    ro=np.array([D['source'][i]!='demo' for i in ec]);wd=D['wide'][e].cpu().numpy()
    for ph in [0,10,20,30,45,59]:
        m=(tt==ph)&al
        res[f'v_auc_t{ph}_rollout']=auc(vv[m&ro],lab[m&ro]);res[f'v_auc_t{ph}_wide']=auc(vv[m&wd],lab[m&wd]);res[f'n_alive_t{ph}_wide']=int((m&wd).sum())
    for ph in [0,10,20]:
        m=(tt==ph)&wd;res[f'v_auc_t{ph}_wide_full']=auc(vv[m],lab[m])
    res['v_auc_all_rollout']=auc(vv[ro&al],lab[ro&al]);res['v_auc_all_wide']=auc(vv[wd&al],lab[wd&al])
    A=qq-vv
    for lo,hi in [(0,10),(11,20),(0,30)]:
        m=wd&al&(tt>=lo)&(tt<=hi)
        res[f'adv_auc_t{lo}_{hi}_wide']=auc(A[m],lab[m]);res[f'adv_auc_vbin_t{lo}_{hi}_wide']=binned_auc(A[m],vv[m],lab[m])
        res[f'adv_std_t{lo}_{hi}_wide']=float(A[m].std())
    C.train();return res
t0=time.time();row={};Tf=Tt.float()
for step in range(1,a.steps+1):
    e=train_e[torch.randint(len(train_e),(a.batch,),device='cuda',generator=g)]
    t=(torch.rand(a.batch,device='cuda',generator=g)*(Tf[e]+1)).long().clamp(max=CHUNKS-1)
    t=torch.minimum(t,Tt[e])
    s=state(e,t);s2=state(e,(t+1).clamp(max=CHUNKS-1));aa=D['AJ'][e,t]
    r=D['R'][e,t];done=(t>=Tt[e]).float()
    with torch.no_grad():qt=T_.qmin(s,aa);vn=C.value(s2);target=r+GAMMA*(1-done)*vn
    vl=expectile(qt-C.value(s),a.expectile);ov.zero_grad(set_to_none=True);vl.backward();ov.step()
    q1,q2=C.q(s,aa);ql=Fn.mse_loss(q1,target)+Fn.mse_loss(q2,target);oq.zero_grad(set_to_none=True);ql.backward();oq.step()
    if step%2==0:
        with torch.no_grad():
            for pt,pc in zip(T_.parameters(),C.parameters()):pt.mul_(1-.005).add_(.005*pc)
    if step%a.eval_every==0 or step==a.steps:
        row=dict(step=step,v_loss=vl.item(),q_loss=ql.item(),elapsed_s=time.time()-t0,**evaluate())
        append_jsonl(out/'metrics.jsonl',row);write_json(out/'status.json',dict(status='training',total_steps=a.steps,updated=cst(),summary=summary,**row));print(json.dumps(row),flush=True)
torch.save(dict(critic=C.state_dict(),stats={k:v.cpu() for k,v in stats.items()},sources=sources,summary=summary,args=vars(a),final=row),out/'critic.pt')
write_json(out/'status.json',dict(status='done',updated=cst(),summary=summary,final=row));print('IQL_DONE',flush=True)
