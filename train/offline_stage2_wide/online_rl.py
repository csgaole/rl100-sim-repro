"""Sim ONLINE RL pilot (RL-100-style on-policy PPO on the diffusion denoising chain).
Iteration i: rollouts of the current policy (stochastic s2common kernel, K DDIM steps) with online_worker.py on
the 6 gate wide cells + nominal share (fresh seeds 670000+1000*i, disjoint from gate/heldout/nominal-gate/reserved)
-> event rewards (success: 1-0.05*(k*+1)/480 at terminal chunk T=k*//8; drop / early wrong_face / other failure: 0,
terminate) -> fresh-online V (2x512 MLP on frozen BC features [z_{t-1},z_t], no clock) refit on this iteration's
data -> GAE(gamma, lambda) -> PPO (clipped, per-denoising-step ratio of the step_mean Gaussian kernel, as
ppo_wide_v2.py) + KL anchor (default: to the START policy, i.e. a trust region; --anchor-ref stage1 for Stage-1) on demo + nominal states + PPO KL early stop -> checkpoint. Quick check (batch_worker.py,
deterministic 10-step deployment sampler) every --qc-every iterations on gate-seed subset 620000+[0,qc_n) x 6 cells
+ nominal 880000+[0,qc_n), paired vs the stored round-2-candidate / Stage-1 gate outcomes on the same seeds;
full gate (6x50 + nominal 100, paired McNemar) when quick wide >= start subset + --full-gate-margin.
Reuses: batch_worker.py (unchanged) for every evaluation; the pipeline's job/claim protocol (run_jobs/shards)
for both rollout and eval workers; s2common kernel/logprob/sample_chain/save_policy; wcommon init_pose / gate cells."""
from wcommon import *
HERE=Path(__file__).resolve().parent
import argparse,subprocess,shutil,traceback,copy,collections
import torch.nn.functional as Fn
ap=argparse.ArgumentParser()
ap.add_argument('--out',default=str(ROOT/'runs/online_rl_pilot'));ap.add_argument('--start',default=str(ROOT/'runs/offline_stage2_wide/round2/ppo/cand_step0150.pt'))
ap.add_argument('--iters',type=int,default=20);ap.add_argument('--eps',type=int,default=300);ap.add_argument('--nominal-frac',type=float,default=.23)
ap.add_argument('--K',type=int,default=10);ap.add_argument('--gpu',type=int,default=7);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--shard',type=int,default=25)
ap.add_argument('--seed-base',type=int,default=670000)
ap.add_argument('--lr',type=float,default=5e-8);ap.add_argument('--clip',type=float,default=.15);ap.add_argument('--epochs',type=int,default=2);ap.add_argument('--mb',type=int,default=512)
ap.add_argument('--train-last',type=int,default=0,help='0 = PPO loss on all K denoising steps; k = only the last k steps')
ap.add_argument('--anchor',type=float,default=3.);ap.add_argument('--anchor-ref',choices=['start','stage1'],default='start');ap.add_argument('--target-kl',type=float,default=.02);ap.add_argument('--anchor-batch',type=int,default=32);ap.add_argument('--grad-clip',type=float,default=1.)
ap.add_argument('--gamma',type=float,default=.99);ap.add_argument('--lam',type=float,default=.95);ap.add_argument('--time-pen',type=float,default=.05)
ap.add_argument('--v-lr',type=float,default=3e-4);ap.add_argument('--v-epochs',type=int,default=10);ap.add_argument('--v-warm-epochs',type=int,default=30)
ap.add_argument('--qc-every',type=int,default=2);ap.add_argument('--qc-n',type=int,default=20);ap.add_argument('--full-gate-margin',type=int,default=3)
ap.add_argument('--keep-last',type=int,default=3);ap.add_argument('--ddim-check',type=int,default=0,help='N>0: only run the K=5 vs K=10 stochastic-sampler check on N eps and exit')
ap.add_argument('--smoke',action='store_true')
ap.add_argument('--fix-encoder',action='store_true',help='freeze vision encoder (old behavior); default trains encoder+UNet')
ap.add_argument('--enc-mb',type=int,default=64,help='encoder re-encode micro-batch (samples) for live-encoder PPO')
ap.add_argument('--encoder-lr-scale',type=float,default=1.0,help='encoder LR = lr * scale when not --fix-encoder; 1.0 = single Adam(policy) like official; 0.1 matches some official online bash overrides')

a=ap.parse_args();OUT=Path(a.out);OUT.mkdir(parents=True,exist_ok=True);PY=sys.executable;setup(0)
LOG=OUT/'online.log';SP=OUT/'state.json'
def log(m):
    l=f'[{cst()}] {m}';print(l,flush=True)
    with open(LOG,'a') as f:f.write(l+'\n')
ST=json.loads(SP.read_text()) if SP.exists() else dict(started=cst(),iters={},best=None,gates={},args=vars(a))
ST.update(pid=os.getpid(),args=vars(a))
def save_state():write_json(SP,ST)
def status(phase,**kw):write_json(OUT/'status.json',dict(status='running',phase=phase,pid=os.getpid(),updated=cst(),iters_done=len([1 for v in ST['iters'].values() if v.get('done')]),planned_iters=a.iters,best=ST.get('best'),**kw))
def env_for():return dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),MUJOCO_EGL_DEVICE_ID=str(a.gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4',S2_OUT=str(OUT))
def run_jobs(script,d,spec,phase,result_dir):
    """Same protocol as wide_pipeline*.run_jobs: jobs.json + O_EXCL claims + done markers, resumable."""
    d.mkdir(parents=True,exist_ok=True);jf=d/'jobs.json';spec=dict(spec,out=str(d))
    if jf.exists():assert json.loads(jf.read_text())==spec,f'job spec changed in {d}'
    else:write_json(jf,spec)
    for c in (d/'claims').glob('*') if (d/'claims').exists() else []:
        if not (d/'done'/c.name).exists():c.unlink()
    jobs=spec['jobs'];t0=time.time()
    for attempt in range(4):
        for c in (d/'claims').glob('*') if (d/'claims').exists() else []:
            if not (d/'done'/c.name).exists():c.unlink()
        todo=[j for j in jobs if not (d/'done'/j['id']).exists()]
        if not todo:break
        (d/'logs').mkdir(exist_ok=True);n=min(a.workers,len(todo))
        ps=[subprocess.Popen([PY,'-u',HERE/script,'--jobs',jf,'--wid',str(i)],cwd=str(HERE),env=env_for(),stdout=open(d/'logs'/f'worker_{i}.log','a'),stderr=subprocess.STDOUT) for i in range(n)]
        while any(q.poll() is None for q in ps):
            nd=len(list((d/'done').glob('*')));status(phase,jobs_done=nd,jobs_total=len(jobs),worker_pids=[q.pid for q in ps],attempt=attempt);time.sleep(10 if not a.smoke else 3)
        rcs=[q.returncode for q in ps]
        if all(r==0 for r in rcs) or attempt==3:continue
        # e.g. 2026-10-10 22:20 CST: another user's JAX job preallocated ~68GB on the same GPU while 8 eval workers were
        # creating EGL/CUDA contexts -> all workers died by signal with empty logs. Log return codes and retry (resumable).
        log(f'{phase}: worker return codes {rcs} (negative = killed by signal) attempt {attempt}; retrying unfinished jobs in 60s');time.sleep(60)
    assert len(list((d/'done').glob('*')))==len(jobs),f'{phase}: workers failed repeatedly, see {d}/logs'
    return time.time()-t0
def shards(eps,prefix,**kw):return [dict(id=f'{prefix}_{i:04d}',episodes=eps[i:i+a.shard],**kw) for i in range(0,len(eps),a.shard)]
def cell_eps(seed0,n,cells,nominal_name=True):
    E=[]
    for pos,yaw in cells:
        for s in range(seed0,seed0+n):
            dx,dy,yw=init_pose(s,pos/1000,yaw);E.append(dict(seed=s,dx=dx,dy=dy,yaw_deg=yw,cell='nominal' if pos==5 else f'p{pos}_y{yaw}',pos_half_mm=pos,yaw_half_deg=yaw))
    return E
def train_eps(it,n):
    base=a.seed_base+1000*it;check_seeds(base,base+n-1);rng=np.random.default_rng(base);E=[]
    for s in range(base,base+n):
        if rng.random()<a.nominal_frac:pos,yaw=5,2.
        else:pos,yaw=GATE_CELLS[rng.integers(len(GATE_CELLS))]
        dx,dy,yw=init_pose(s,pos/1000,yaw);E.append(dict(seed=s,dx=dx,dy=dy,yaw_deg=yw,cell='nominal' if pos==5 else f'p{pos}_y{yaw}',pos_half_mm=pos,yaw_half_deg=yaw))
    return E
def load_rollouts(d):
    global ROLL_OBS
    M=[];C=[];X=[];Y=[];LP=[];L=[];OB=[]
    for f in sorted((d/'data').glob('*.npz')):
        if '.tmp' in f.name:continue
        z=np.load(f);M+=json.loads(str(z['meta']));C.append(z['cond']);X.append(z['xs']);Y.append(z['ys']);LP.append(z['lp']);L.append(z['ep_len']);OB.append({k:z[k] for k in z.files if k.startswith('rgb_') or k=='prop'} if 'prop' in z.files else None)
    ROLL_OBS=pack_rollout_obs(OB,np.concatenate(L)) if L else None
    return M,np.concatenate(C),np.concatenate(X),np.concatenate(Y),np.concatenate(LP),np.concatenate(L)
def rollout_stats(M):
    out={}
    for m in M:
        c=out.setdefault(m['cell'],dict(eps=0,success=0,fails={}));c['eps']+=1;c['success']+=m['success']
        if not m['success']:c['fails'][m['reason']]=c['fails'].get(m['reason'],0)+1
    w=[m for m in M if m['cell']!='nominal'];n=[m for m in M if m['cell']=='nominal']
    return dict(by_cell=out,wide=f"{sum(m['success'] for m in w)}/{len(w)}",nominal=f"{sum(m['success'] for m in n)}/{len(n)}",
        wide_rate=sum(m['success'] for m in w)/max(len(w),1),nominal_rate=sum(m['success'] for m in n)/max(len(n),1),
        fails={r:sum(1 for m in M if not m['success'] and m['reason']==r) for r in {m['reason'] for m in M if not m['success']}},mean_chunks=float(np.mean([m['chunks_run'] for m in M])))

# ---------------- DDIM K=5 vs K=10 check ----------------
if a.ddim_check:
    n=max(1,a.ddim_check//6);E=[]
    for pos,yaw in GATE_CELLS:
        for s in range(668000,668000+n):
            dx,dy,yw=init_pose(s,pos/1000,yaw);E.append(dict(seed=s,dx=dx,dy=dy,yaw_deg=yw,cell=f'p{pos}_y{yaw}',pos_half_mm=pos,yaw_half_deg=yaw))
    check_seeds(668000,668000+n-1);R={}
    for K in [10,5]:
        d=OUT/'ddim_check'/f'K{K}';wall=run_jobs('online_worker.py',d,dict(policy=a.start,K=K,jobs=shards(E,'c')),f'ddim_check:K{K}',d)
        M=load_rollouts(d)[0];k=sum(m['success'] for m in M);R[K]=dict(success=k,episodes=len(M),rate=k/len(M),wall_s=round(wall,1),s_per_ep_aggregate=round(wall/len(M),2),stats=rollout_stats(M))
        log(f'DDIM check K={K}: {k}/{len(M)} wall {wall:.0f}s ({wall/len(M):.2f} s/ep aggregate, {a.workers} workers)')
    R['drop_pts']=100*(R[10]['rate']-R[5]['rate']);R['speedup']=R[10]['wall_s']/R[5]['wall_s'];write_json(OUT/'ddim_check.json',R);log(f'DDIM check: drop {R["drop_pts"]:.1f} pts, speedup x{R["speedup"]:.2f}');sys.exit(0)

# ---------------- models ----------------
p,cp=load_policy(a.start);model=configure_online_policy(p,fix_encoder=a.fix_encoder);ROLL_OBS=None
p1,_=load_policy(STAGE1);ref=p1.model.eval().requires_grad_(False)
start_model=copy.deepcopy(model).eval().requires_grad_(False)
anc=start_model if a.anchor_ref=='start' else ref  # Stage-1 anchor: gradient ~16x the PG gradient at the start policy (KL~7), would drag the policy back to Stage-1
class VNet(nn.Module):
    def __init__(s):super().__init__();s.n=nn.Sequential(nn.Linear(2*FEAT,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,1))
    def forward(s,x):return s.n(x).squeeze(-1)
V=VNet().cuda();vopt=torch.optim.Adam(V.parameters(),lr=a.v_lr);opt,enc_opt=make_policy_optimizers(p,a.lr,fix_encoder=a.fix_encoder,encoder_lr_scale=a.encoder_lr_scale);log(f'policy opt: fix_encoder={a.fix_encoder} encoder_lr_scale={a.encoder_lr_scale} enc_opt={"yes" if enc_opt is not None else "no"}');VS=None
DD=load_dataset_w([['demo',str(DEMO_DIR)]]);demo_tr=torch.nonzero(~DD['heldout']).flatten();demo_ho=torch.nonzero(DD['heldout']).flatten()
gfix=torch.Generator(device='cuda').manual_seed(2026)
def demo_cond(idx,t):return policy_cond(DD,idx,t)
he=demo_ho[torch.randint(len(demo_ho),(512,),device='cuda',generator=gfix)];hd=torch.randint(CHUNKS,(512,),device='cuda',generator=gfix)
FIX_DEMO=demo_cond(he,hd);FIX_NOISE=torch.randn((512,8,14),device='cuda',generator=gfix);FIX_ROLL=None
@torch.no_grad()
def deploy(m,c,n):return sample_chain(m,p,c,stochastic=False,noise=n)
def jdev(x,y):return (x-y)[...,JOINTS].abs().mean().item()
@torch.no_grad()
def devs():
    out={}
    for nm,c in [('demo',FIX_DEMO)]+([('rollout',FIX_ROLL)] if FIX_ROLL is not None else []):
        x=deploy(model,c,FIX_NOISE[:len(c)]);out[f'dev_vs_start_{nm}']=jdev(x,deploy(start_model,c,FIX_NOISE[:len(c)]));out[f'dev_vs_stage1_{nm}']=jdev(x,deploy(ref,c,FIX_NOISE[:len(c)]))
    return out
TRAINER=OUT/'trainer_last.pt';RESUMED_IT=0;RESUMED_TRAIN=None
if TRAINER.exists():
    ck=torch.load(TRAINER,map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);V.load_state_dict(ck['V']);opt.load_state_dict(ck['opt']);vopt.load_state_dict(ck['vopt']);(enc_opt.load_state_dict(ck['enc_opt']) if enc_opt is not None and ck.get('enc_opt') else None)
    VS={k:v.cuda() for k,v in ck['VS'].items()} if ck['VS'] else None;FIX_ROLL=ck['FIX_ROLL'].cuda() if ck.get('FIX_ROLL') is not None else None;RESUMED_IT=int(ck['it']);RESUMED_TRAIN=ck.get('train');log(f'resumed trainer state after iter {ck["it"]}')
def ckpt_path(it):return Path(a.start) if it==0 else OUT/'ckpt'/f'it{it:03d}.pt'
V2=ROOT/'runs/offline_stage2_wide_v2'  # same reference gate results forced_eval.py pairs against (start = round-2 candidate)
G_R2=json.loads((V2/'gate/start/result.json').read_text());G_S1=json.loads((V2/'gate/stage1/result.json').read_text())
def tail(k,n):return sum(math.comb(n,i) for i in range(k,n+1))/2**n if n else 1.
def paired(c,r,nominal):
    ks=[k for k in c if k.startswith('nominal')==nominal and k in r];g=sum(c[k] and not r[k] for k in ks);l=sum(r[k] and not c[k] for k in ks)
    return dict(cand=sum(c[k] for k in ks),ref=sum(r[k] for k in ks),n=len(ks),delta=g-l,gained=g,lost=l,p_better=tail(g,g+l),p_worse=tail(l,g+l))
def evaluate(policy,name,n,nn_):
    d=OUT/'eval'/name;rp=d/'result.json'
    if rp.exists():return json.loads(rp.read_text())
    E=cell_eps(GATE_SEED0,n,GATE_CELLS)+cell_eps(NOMINAL['seed0'],nn_,[(5,2.)])
    wall=run_jobs('batch_worker.py',d,dict(policy=str(policy),jobs=[dict(j,kind='eval',mode='det') for j in shards(E,'e')]),f'eval:{name}',d)
    rows=[]
    for f in sorted((d/'rows').glob('*.jsonl')):rows+=[json.loads(l) for l in open(f)]
    out={f"{r['cell']}:{r['seed']}":bool(r['success']) for r in rows};cells={}
    for r in rows:x=cells.setdefault(r['cell'],[0,0]);x[0]+=bool(r['success']);x[1]+=1
    fails={}
    for r in rows:
        if not r['success']:f=fails.setdefault(r['cell'],{});f[r['reason']]=f.get(r['reason'],0)+1
    res=dict(policy=str(policy),wide=sum(v[0] for c,v in cells.items() if c!='nominal'),wide_n=sum(v[1] for c,v in cells.items() if c!='nominal'),nominal=cells['nominal'][0],nominal_n=cells['nominal'][1],
        cells=cells,fails=fails,outcomes=out,wall_s=round(wall,1),vs_r2cand=dict(wide=paired(out,G_R2['outcomes'],False),nominal=paired(out,G_R2['outcomes'],True)),
        vs_stage1=dict(wide=paired(out,G_S1['outcomes'],False),nominal=paired(out,G_S1['outcomes'],True)))
    write_json(rp,res);return res
def fmt(r):return f"wide {r['wide']}/{r['wide_n']} (r2 subset {r['vs_r2cand']['wide']['ref']}, d {r['vs_r2cand']['wide']['delta']:+d} p {r['vs_r2cand']['wide']['p_better']:.3g}; S1 {r['vs_stage1']['wide']['ref']}, d {r['vs_stage1']['wide']['delta']:+d}) nominal {r['nominal']}/{r['nominal_n']} (r2 {r['vs_r2cand']['nominal']['ref']}, S1 {r['vs_stage1']['nominal']['ref']})"

def report():
    L=['# Sim online RL pilot (on-policy PPO on the diffusion chain)\n',f'_Auto-generated {cst()}; state `{SP}`, log `{LOG}`._\n',
       f'- start `{a.start}`; anchor ref {a.anchor_ref} (w={a.anchor}), target KL {a.target_kl} on demo+nominal states; K={a.K} DDIM (train), eval = deterministic 10-step deployment sampler (batch_worker.py)',
       f'- {a.eps} eps/iter ({a.nominal_frac:.0%} nominal, rest uniform over 6 gate cells), seeds {a.seed_base}+1000*it; reward success 1-{a.time_pen}*(k*+1)/480, failures 0 (drop / early wrong_face terminate)',
       f'- V: 2x512 MLP on [z_(t-1),z_t] (no clock), refit each iter ({a.v_epochs} ep; {a.v_warm_epochs} MC warm-up ep on iter 1); GAE gamma {a.gamma} lambda {a.lam}',
       f'- PPO: lr {a.lr}, fix_encoder={a.fix_encoder}, encoder_lr_scale={a.encoder_lr_scale}, clip {a.clip}, {a.epochs} epochs, mb {a.mb}, train steps {"all" if not a.train_last else "last "+str(a.train_last)}, grad clip {a.grad_clip}',
       f'- quick check every {a.qc_every} iters: 6 cells x {a.qc_n} (620000+) + nominal {a.qc_n} (880000+), paired vs stored r2-cand / Stage-1 outcomes; full gate if wide >= r2 subset + {a.full_gate_margin}\n',
       '| it | train wide | train nominal | fails | mean chunks | V EV (pre/post) | adv std | clip frac | approx KL | anchor KL | dev vs start / S1 (demo) | quick check | wall s (collect/train/total) |','|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for it,v in sorted(ST['iters'].items(),key=lambda x:int(x[0])):
        if not v.get('done'):continue
        r=v['rollouts'];t=collections.defaultdict(lambda:float('nan'),v['train']);q=v.get('quick')
        L.append(f"| {it} | {r['wide']} | {r['nominal']} | {r['fails']} | {r['mean_chunks']:.1f} | {t['v_ev_pre']:.3f}/{t['v_ev_post']:.3f} | {t['adv_std_raw']:.4f} | {t['clip_frac']:.3f} | {t['approx_kl']:.4f} | {t['anchor_kl']:.3f} | {t['dev_vs_start_demo']:.4f}/{t['dev_vs_stage1_demo']:.4f} | {fmt(q) if q else ''} | {v['wall']['collect']:.0f}/{v['wall']['train']:.0f}/{v['wall']['total']:.0f} |")
    if ST['gates']:L.append('\n## Full gates\n```json\n'+json.dumps({k:{kk:vv for kk,vv in g.items() if kk!='outcomes'} for k,g in ST['gates'].items()},indent=1)+'\n```')
    L.append(f"\n- best (quick check): {ST.get('best')}");(OUT/'REPORT.md').write_text('\n'.join(L)+'\n')

def gae(M,C,L):
    """returns per-sample reward, done, episode id, chunk idx (samples are concatenated per episode in order)"""
    rew=np.zeros(len(C),np.float32);done=np.zeros(len(C),np.float32);o=0;epi=np.zeros(len(C),np.int64)
    for e,(m,l) in enumerate(zip(M,L)):
        epi[o:o+l]=e;done[o+l-1]=1
        if m['success']:rew[o+l-1]=1-a.time_pen*(m['k_star']+1)/480.
        o+=l
    return rew,done,epi
def vpred(Cn):
    with torch.no_grad():return torch.cat([V(Cn[i:i+4096]) for i in range(0,len(Cn),4096)])
def fit_v(Cn,tgt,epochs):
    n=len(Cn);ls=[]
    for ep in range(epochs):
        perm=torch.randperm(n,device='cuda')
        for i in range(0,n,512):
            j=perm[i:i+512];l=Fn.mse_loss(V(Cn[j]),tgt[j]);vopt.zero_grad();l.backward();vopt.step();ls.append(l.item())
    return float(np.mean(ls[-max(1,n//512):]))
def ev(pred,tgt):return float(1-(tgt-pred).var()/tgt.var().clamp_min(1e-8))
def train_iter(it,M,C,X,Y,LP,L):
    global VS,FIX_ROLL
    LIVE=(not a.fix_encoder) and ROLL_OBS is not None
    if not a.fix_encoder and ROLL_OBS is None:log('WARNING: encoder unfrozen but rollouts have no stored obs -> stored cond, NO encoder gradient')
    t0=time.time();rew,done,epi=gae(M,C,L);Ct=torch.tensor(C,device='cuda').float()
    if VS is None:VS=dict(mu=Ct.mean(0),sd=Ct.std(0).clamp_min(1e-3))
    Cn=(Ct-VS['mu'])/VS['sd']
    if FIX_ROLL is None:FIX_ROLL=Ct[torch.randperm(len(Ct),generator=torch.Generator().manual_seed(0))[:512].cuda()]
    R=torch.tensor(rew,device='cuda');Dn=torch.tensor(done,device='cuda');E=torch.tensor(epi,device='cuda')
    # MC returns (for V warm-up + logging)
    mcn=np.zeros(len(rew),np.float32);acc=0.
    for i in range(len(rew)-1,-1,-1):
        acc=rew[i]+(0. if done[i] else a.gamma*acc);mcn[i]=acc
    mc=torch.tensor(mcn,device='cuda')
    warm=None
    if int(it)==1 or not ST.get('v_warmed'):warm=fit_v(Cn,mc,a.v_warm_epochs);ST['v_warmed']=True
    v=vpred(Cn).cpu().numpy();A=np.zeros(len(rew),np.float32);last=0.
    for i in range(len(rew)-1,-1,-1):
        nv=0. if done[i] else v[i+1];delta=rew[i]+a.gamma*nv-v[i];last=delta+(0. if done[i] else a.gamma*a.lam*last);A[i]=last
    ret=torch.tensor(A+v,device='cuda');ev_pre=ev(torch.tensor(v,device='cuda'),ret);ev_pre_mc=ev(torch.tensor(v,device='cuda'),mc)
    vloss=fit_v(Cn,ret,a.v_epochs);ev_post=ev(vpred(Cn),ret)
    At=torch.tensor(A,device='cuda');adv=(At-At.mean())/(At.std()+1e-8)
    Xt=torch.tensor(X,device='cuda');Yt=torch.tensor(Y,device='cuda');LPt=torch.tensor(LP,device='cuda')
    p.noise_scheduler.set_timesteps(a.K);TS=[t.to('cuda') for t in p.noise_scheduler.timesteps];steps=list(range(a.K))[-a.train_last:] if a.train_last else list(range(a.K))
    nom_idx=torch.tensor([i for i,e in enumerate(epi) if M[e]['cell']=='nominal'],device='cuda')
    m=dict(init_logratio_absmean=None,init_logratio_max=None,pg_loss=0.,clip_frac=0.,approx_kl=0.,anchor_kl=0.,grad_norm=0.,enc_grad_norm=0.,live_encoder=bool(LIVE),n_updates=0,max_abs_logratio=0.,early_stop=None);n=len(Ct);g=torch.Generator(device='cuda').manual_seed(1000+int(it))
    stop=False
    for ep in range(a.epochs):
        if stop:break
        perm=torch.randperm(n,device='cuda',generator=g)
        for i in range(0,n,a.mb):
            j=perm[i:i+a.mb];cond=live_cond(p,ROLL_OBS,j,a.enc_mb) if LIVE else Ct[j];aj=adv[j];zero_policy_opts(opt,enc_opt);cf=0.;ak=0.;p.noise_scheduler.set_timesteps(a.K)
            for s in steps:
                mean,std=kernel(p,model(Xt[j,s],TS[s],global_cond=cond),TS[s],Xt[j,s]);logr=logprob(mean,std,Yt[j,s])-LPt[j,s]
                if m['init_logratio_absmean'] is None:m['init_logratio_absmean']=logr.abs().mean().item();m['init_logratio_max']=logr.abs().max().item()
                ratio=logr.clamp(-20,20).exp();lpg=-torch.minimum(ratio*aj,ratio.clamp(1-a.clip,1+a.clip)*aj).mean()/len(steps);lpg.backward()
                with torch.no_grad():cf+=((ratio-1).abs()>a.clip).float().mean().item()/len(steps);ak+=((ratio-1)-logr.clamp(-20,20)).mean().item()/len(steps);m['pg_loss']+=lpg.item();m['max_abs_logratio']=max(m['max_abs_logratio'],logr.abs().max().item())
            ed=demo_tr[torch.randint(len(demo_tr),(a.anchor_batch,),device='cuda',generator=g)];cd=demo_cond(ed,torch.randint(CHUNKS,(a.anchor_batch,),device='cuda',generator=g))
            if len(nom_idx):cd=torch.cat([cd,Ct[nom_idx[torch.randint(len(nom_idx),(a.anchor_batch,),device='cuda',generator=g)]]])
            with torch.no_grad():_,xd,_,_=sample_chain(anc,p,cd,g,stochastic=True,trace=True)  # 10-step chain (as ppo_wide_v2 anchor)
            T10=[t.to('cuda') for t in p.noise_scheduler.timesteps];kls=0.
            for s in range(len(xd)):
                with torch.no_grad():mr,sr=kernel(p,anc(xd[s],T10[s],global_cond=cd),T10[s],xd[s])
                mn,_=kernel(p,model(xd[s],T10[s],global_cond=cd),T10[s],xd[s]);kl=(.5*((mn-mr)/sr).square())[...,JOINTS].mean();(a.anchor*kl).backward();kls+=kl.item()
            if ak>a.target_kl and m['n_updates']>0:  # standard PPO KL early stop (measured on this minibatch before stepping)
                m['early_stop']=dict(epoch=ep,minibatch=i//a.mb,approx_kl=ak);zero_policy_opts(opt,enc_opt);stop=True;break
            if LIVE:m['enc_grad_norm']+=encoder_backward(p,ROLL_OBS,j,cond,a.enc_mb)
            gn=nn.utils.clip_grad_norm_(policy_grad_params(p,a.fix_encoder),a.grad_clip);step_policy_opts(opt,enc_opt)
            m['clip_frac']+=cf;m['approx_kl']+=ak;m['anchor_kl']+=kls;m['grad_norm']+=float(gn);m['n_updates']+=1
    for k in ['pg_loss','clip_frac','approx_kl','anchor_kl','grad_norm','enc_grad_norm']:m[k]/=max(m['n_updates'],1)
    return dict(m,samples=n,v_loss=vloss,v_warm_loss=warm,v_ev_pre=ev_pre,v_ev_pre_vs_mc=ev_pre_mc,v_ev_post=ev_post,adv_mean_raw=float(A.mean()),adv_std_raw=float(A.std()),
        adv_pos_frac=float((A>0).mean()),ret_mean=float(mcn.mean()),**devs(),train_wall_s=time.time()-t0)

def prune():
    keep={int(k) for k in sorted([int(k) for k,v in ST['iters'].items() if v.get('done')])[-a.keep_last:]}
    if ST.get('best'):keep.add(int(ST['best']['it']))
    for k in ST['gates']:keep.add(int(k))
    for f in (OUT/'ckpt').glob('it*.pt'):
        if int(f.stem[2:]) not in keep:f.unlink()
def main():
    log(f'online RL start pid={os.getpid()} gpu={a.gpu} workers={a.workers} K={a.K} eps/iter={a.eps} iters={a.iters} smoke={a.smoke} fix_encoder={a.fix_encoder} encoder_lr_scale={a.encoder_lr_scale}');save_state()
    (OUT/'ckpt').mkdir(exist_ok=True)
    for it in range(1,a.iters+1):
        I=ST['iters'].setdefault(str(it),{})
        if I.get('done'):continue
        t0=time.time();pol=ckpt_path(it-1);assert pol.exists(),pol
        d=OUT/'rollouts'/f'it{it:03d}';E=train_eps(it,a.eps)
        tc=run_jobs('online_worker.py',d,dict(policy=str(pol),K=a.K,jobs=shards(E,'r'),**({'store_obs':True} if not a.fix_encoder else {})),f'it{it}:collect',d)
        M,C,X,Y,LP,L=load_rollouts(d);I['rollouts']=rollout_stats(M);log(f'it{it} collect {I["rollouts"]["wide"]} wide, {I["rollouts"]["nominal"]} nominal, fails {I["rollouts"]["fails"]}, mean chunks {I["rollouts"]["mean_chunks"]:.1f}, {tc:.0f}s');save_state()
        if RESUMED_IT>=it and ckpt_path(it).exists():
            # crash after this iteration's update (e.g. in its full gate): never re-train on the same data
            tr=I.get('train') or RESUMED_TRAIN or dict(note='train stats lost in crash; see online.log');I['train']=tr;log(f'it{it}: update already done before restart, skipping training');save_state()
        else:
          status(f'it{it}:train');tr=train_iter(it,M,C,X,Y,LP,L);I['train']=tr
          save_policy(p,cp,ckpt_path(it),dict(stage='online_rl_pilot',iteration=it,start=a.start,anchor_ref=a.anchor_ref))
          torch.save(dict(train=tr,it=it,model=model.state_dict(),V=V.state_dict(),opt=opt.state_dict(),vopt=vopt.state_dict(),enc_opt=(enc_opt.state_dict() if enc_opt is not None else None),VS={k:v.cpu() for k,v in VS.items()},FIX_ROLL=FIX_ROLL.cpu()),str(TRAINER)+'.tmp');os.replace(str(TRAINER)+'.tmp',TRAINER);save_state()
          log(f'it{it} train: V EV pre {tr["v_ev_pre"]:.3f} post {tr["v_ev_post"]:.3f}, adv std {tr["adv_std_raw"]:.4f}, clip frac {tr["clip_frac"]:.3f}, approx_kl {tr["approx_kl"]:.4f}, anchor_kl {tr["anchor_kl"]:.3f}, dev start/S1 demo {tr["dev_vs_start_demo"]:.4f}/{tr["dev_vs_stage1_demo"]:.4f}, {tr["train_wall_s"]:.0f}s')
        I['wall']=dict(collect=tc,train=tr.get('train_wall_s',float('nan')))
        if it%a.qc_every==0 or it==a.iters:
            status(f'it{it}:quick_check');q=evaluate(ckpt_path(it),f'quick_it{it:03d}',a.qc_n if not a.smoke else 2,a.qc_n if not a.smoke else 2);I['quick']={k:v for k,v in q.items() if k!='outcomes'}
            log(f'it{it} QUICK {fmt(q)}  ({q["wall_s"]:.0f}s)')
            key=(q['wide'],q['nominal'])
            if not ST.get('best') or key>(ST['best']['wide'],ST['best']['nominal']):ST['best']=dict(it=it,wide=q['wide'],nominal=q['nominal'],path=str(ckpt_path(it)))
            if q['vs_r2cand']['wide']['delta']>=a.full_gate_margin and not a.smoke:
                status(f'it{it}:full_gate');gr=evaluate(ckpt_path(it),f'gate_it{it:03d}',GATE_N,NOMINAL['n']);ST['gates'][str(it)]={k:v for k,v in gr.items() if k!='outcomes'};log(f'it{it} FULL GATE {fmt(gr)}')
        I['wall']['total']=time.time()-t0;I['done']=True;I['finished']=cst();save_state();prune();report()
    write_json(OUT/'status.json',dict(status='complete',updated=cst(),best=ST.get('best'),gates={k:dict(wide=g['wide'],nominal=g['nominal'],vs_r2cand=g['vs_r2cand'],vs_stage1=g['vs_stage1']) for k,g in ST['gates'].items()}))
    log('online RL complete');report()
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(OUT/'status.json',dict(status='failed',pid=os.getpid(),updated=cst(),error=repr(e),traceback=traceback.format_exc()));log(f'FAILED {e!r}');raise
