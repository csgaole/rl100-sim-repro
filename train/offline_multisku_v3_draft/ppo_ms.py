# [offline_multisku_v3_draft 2026-10-20] NEW file = v2 ppo_ms.py + Bill-approved report recommendations:
#  (1) NO hard KL early stop at 0.02: KL-ADAPTIVE lr (official uni_ppo AdaptiveScheduler: lr/1.5 if kl>2*target, *1.5 if kl<target/2,
#      clamped [lr/--lr-range, lr*--lr-range]) on PER-DIM KL = post-update approx_kl(pi_theta||pi_old) / (8 x 12 joint dims);
#      a SAFETY hard stop only if the rolling-mean pre-update per-dim KL > --kl-hard-mult x target.
#  (2) pi_old refreshed to the current policy every --old-refresh steps (default 50; BPPO is_update_old_policy).
#  (3) KL anchor coef 6 (Bill's fixed rule) to the ROUND-START policy (frozen, never refreshed), evaluated on a separate batch of
#      demo + SUCCESSFUL-rollout states (its own pi_anchor chain), per non-terminal step 0.5*((mu-mu_anc)/sigma_anc)^2 mean over joints.
#  (4) advantage: --adv-samples (default 4) pi_old chains per state, every chain is a PPO sample with A_k=minQ(s,a_k)-V(s), batch-normalised,
#      then official omega asymmetric weighting (w=omega if A>0 else 1-omega, default .7); critic V-AUC (rollout states) logged.
# Unchanged: DDIM10 eta=1 kernel, std clamp [.0067,.1], terminal sigma 0 skipped, per-step Gaussian logprob over 12 joints x 8,
#   clip .2->.1 linear, grad clip 1, Adam, encoder+normalizer frozen (asserted), all-data PPO states, OPE log only, milestone snapshots.
from ms_common import *
import argparse,copy
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True,help='round-start policy (= initial pi_old = KL anchor)');ap.add_argument('--critic',required=True);ap.add_argument('--sources',required=True)
ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=300);ap.add_argument('--batch',type=int,default=128,help='states per step (x --adv-samples chains)')
ap.add_argument('--lr',type=float,default=1e-6,help='INITIAL lr (adaptive)');ap.add_argument('--lr-range',type=float,default=10.,help='adaptive lr clamped to [lr/r, lr*r]')
ap.add_argument('--kl-target-dim',type=float,default=5e-4,help='per-dim KL target for the adaptive lr (official adaptive_kl .02 over ~28-40 dims)')
ap.add_argument('--kl-hard-mult',type=float,default=10.,help='safety stop: rolling-mean pre-update per-dim KL > mult x target');ap.add_argument('--kl-stop-window',type=int,default=10)
ap.add_argument('--old-refresh',type=int,default=50,help='pi_old <- current policy every N steps (0 = never)')
ap.add_argument('--clip',type=float,default=.2);ap.add_argument('--clip-final',type=float,default=.1)
ap.add_argument('--anchor',type=float,default=6.);ap.add_argument('--anchor-batch',type=int,default=128)
ap.add_argument('--adv-samples',type=int,default=4);ap.add_argument('--omega',type=float,default=.7)
ap.add_argument('--ope-every',type=int,default=25);ap.add_argument('--ope-states',type=int,default=2048)
ap.add_argument('--grad-clip',type=float,default=1.);ap.add_argument('--milestones',default='50,100,150,300')
a=ap.parse_args();setup(4321);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
MS=sorted(int(x) for x in a.milestones.split(','));ADIM=8*len(JOINTS)
sources=json.loads(Path(a.sources).read_text());D=load_dataset_w(sources)
ck=torch.load(a.critic,map_location='cuda',weights_only=False);stats={k:v.cuda() for k,v in ck['stats'].items()}
C=Critic(SD,ACT_DIM).cuda();C.load_state_dict(ck['critic']);C.eval().requires_grad_(False)
p,cp=load_policy(a.policy);p.eval();p.obs_encoder.requires_grad_(False);p.normalizer.requires_grad_(False)
enc_before={k:v.clone() for k,v in p.obs_encoder.state_dict().items()}
model=p.model
anc=copy.deepcopy(model).eval().requires_grad_(False)   # KL anchor = ROUND-START policy, frozen for the whole round
old=copy.deepcopy(model).eval().requires_grad_(False)   # pi_old, refreshed every --old-refresh steps
opt=torch.optim.Adam(model.parameters(),lr=a.lr);LR_MIN,LR_MAX=a.lr/a.lr_range,a.lr*a.lr_range;cur_lr=a.lr
isdemo=torch.tensor([s=='demo' for s in D['source']],device='cuda')
tr=torch.nonzero(train_mask(D)).flatten();ho=torch.nonzero(D['heldout']).flatten()
anc_eps=torch.nonzero(train_mask(D)&(isdemo|(D['success']>0))).flatten()
assert len(tr)>0 and len(ho)>0 and bool(isdemo[tr].any()) and bool((~isdemo[tr]).any()),'PPO data must contain demos AND rollouts'
g=torch.Generator(device='cuda').manual_seed(99);ge=torch.Generator(device='cuda').manual_seed(2026)
@torch.no_grad()
def v_auc():
    ro=torch.nonzero(~isdemo).flatten();res={}
    for ph in (0,20,40):
        t=torch.full((len(ro),),ph,device='cuda');v=torch.cat([C.value(critic_state(D,stats,ro[i:i+8192],t[i:i+8192])) for i in range(0,len(ro),8192)])
        res[f'v_auc_t{ph}']=auc(v.cpu().numpy(),D['success'][ro].cpu().numpy()>0)
    return res
VAUC=v_auc();print('CRITIC_V_AUC',json.dumps(VAUC),flush=True)
def fixed_set(eps,n):
    e=eps[torch.randint(len(eps),(n,),device='cuda',generator=ge)];t=torch.randint(CHUNKS,(n,),device='cuda',generator=ge)
    return critic_state(D,stats,e,t),policy_cond(D,e,t),torch.randn((n,8,14),device='cuda',generator=ge)
o_s,o_c,o_n=fixed_set(ho,a.ope_states)
@torch.no_grad()
def deploy(m,c,n):return torch.cat([sample_chain(m,p,c[i:i+512],stochastic=False,noise=n[i:i+512]) for i in range(0,len(c),512)])
@torch.no_grad()
def ope(m):x=deploy(m,o_c,o_n);return C.qmin(o_s,joint_flat(x)),x
q_in,a_in=ope(anc)
write_json(out/'protocol.json',dict(vars(a),adim=ADIM,critic_v_auc=VAUC,dataset=dict(n=D['n'],train=len(tr),heldout=len(ho),anchor_eps=len(anc_eps),demo_train=int(isdemo[tr].sum()),rollout_train=int((~isdemo[tr]).sum()),train_all=TRAIN_ALL),
    pi_old=f'refreshed to current policy every {a.old_refresh} steps',anchor_ref='round-start policy (frozen) on demo+success states',
    lr=f'KL-adaptive (official AdaptiveScheduler) on per-dim post-update KL, target {a.kl_target_dim}, range [{LR_MIN:g},{LR_MAX:g}]',
    advantage=f'{a.adv_samples} pi_old chains/state, A=minQ(s,a_k)-V(s), batch-normalised, omega {a.omega}',ope='log only (no accept/revert)'))
cands=[];t0=time.time();clip_steps=0;max_clip=0.;hist=[];KLH=[];ROWS=[];kl_stop=None;refreshes=[]
def save_candidate(step):
    with torch.no_grad():a_now=deploy(model,o_c,o_n);dev=(a_now-a_in)[...,JOINTS].abs().mean().item()
    rec=dict(path=str(out/f'cand_step{step:04d}.pt'),milestone=step,ope=(hist[-1] if hist else None),action_dev_vs_round_start=dev,lr=cur_lr)
    save_policy(p,cp,rec['path'],dict(stage='offline_multisku_ppo_v3_draft',source_policy=a.policy,critic=a.critic,anchor=a.anchor,lr0=a.lr,**{k:v for k,v in rec.items() if k!='path'}))
    cands.append(rec);append_jsonl(out/'candidates.jsonl',rec);print('CANDIDATE',json.dumps(rec),flush=True)
p.noise_scheduler.set_timesteps(10);SI=[(i,ts) for i,ts in enumerate(p.noise_scheduler.timesteps) if not is_terminal(p,ts)];NST=len(SI);print('stochastic steps',NST,'adim',ADIM,flush=True)
NA=a.adv_samples
for step in range(1,a.steps+1):
    if a.old_refresh>0 and step>1 and (step-1)%a.old_refresh==0:old.load_state_dict(model.state_dict());refreshes.append(step);KLH.clear();print('PI_OLD_REFRESH',step,flush=True)
    clip=a.clip+(a.clip_final-a.clip)*(step-1)/max(1,a.steps-1)
    e=tr[torch.randint(len(tr),(a.batch,),device='cuda',generator=g)];t=torch.randint(CHUNKS,(a.batch,),device='cuda',generator=g)
    s=critic_state(D,stats,e,t).repeat_interleave(NA,0);cond=policy_cond(D,e,t).repeat_interleave(NA,0)
    with torch.no_grad():
        x,xs,ys,lps=sample_chain(old,p,cond,g,stochastic=True,trace=True)          # NA behaviour chains per state from pi_old
        q=C.qmin(s,joint_flat(x));raw=q-C.value(s);adv=(raw-raw.mean())/(raw.std()+1e-8)
        adv=adv*torch.where(adv>0,a.omega,1-a.omega)                               # official omega asymmetric weighting
        qs=q.view(a.batch,NA);within=qs.std(1).mean().item()
        oldk=[kernel(p,old(xs[i],ts.to('cuda'),global_cond=cond),ts,xs[i]) for i,ts in SI]
        ea=anc_eps[torch.randint(len(anc_eps),(a.anchor_batch,),device='cuda',generator=g)];ta=torch.randint(CHUNKS,(a.anchor_batch,),device='cuda',generator=g);ca=policy_cond(D,ea,ta)
        _,xa,_,_=sample_chain(anc,p,ca,g,stochastic=True,trace=True);anck=[kernel(p,anc(xa[i],ts.to('cuda'),global_cond=ca),ts,xa[i]) for i,ts in SI]
    m=dict(loss=0.,anchor_kl=0.,clip_frac=0.,approx_kl=0.,ratio_mean=0.,max_abs_logratio=0.,post_clip_frac=0.,post_approx_kl=0.,post_max_abs_logratio=0.)
    opt.zero_grad(set_to_none=True);l0=0.
    for (i,ts),(mo,so),(ma,sa) in zip(SI,oldk,anck):
        mean,std=kernel(p,model(xs[i],ts.to('cuda'),global_cond=cond),ts,xs[i])
        logr=logprob(mean,std,ys[i])-lps[i];l0=max(l0,logr.abs().max().item())
        ratio=logr.clamp(-20,20).exp()
        lppo=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
        mn,_=kernel(p,model(xa[i],ts.to('cuda'),global_cond=ca),ts,xa[i]);kl=(.5*((mn-ma)/sa).square())[...,JOINTS].mean()
        loss=(lppo+a.anchor*kl)/NST
        assert torch.isfinite(loss),'non-finite loss'
        loss.backward()
        with torch.no_grad():
            m['loss']+=lppo.item()/NST;m['anchor_kl']+=kl.item();m['clip_frac']+=((ratio-1).abs()>clip).float().mean().item()/NST;m['ratio_mean']+=ratio.mean().item()/NST
            m['approx_kl']+=((ratio-1)-logr.clamp(-20,20)).mean().item()/NST;m['max_abs_logratio']=max(m['max_abs_logratio'],logr.abs().max().item())
    if step==1:assert l0<1e-2,f'initial log-ratio {l0} (model must equal pi_old at round start)'
    gn=nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip)
    for gr in opt.param_groups:gr['lr']=cur_lr
    opt.step()
    with torch.no_grad():
        for i,ts in SI:
            mean,std=kernel(p,model(xs[i],ts.to('cuda'),global_cond=cond),ts,xs[i]);logr=(logprob(mean,std,ys[i])-lps[i]).clamp(-20,20);ratio=logr.exp()
            m['post_clip_frac']+=((ratio-1).abs()>clip).float().mean().item()/NST;m['post_approx_kl']+=((ratio-1)-logr).mean().item()/NST;m['post_max_abs_logratio']=max(m['post_max_abs_logratio'],logr.abs().max().item())
    kd_pre=m['approx_kl']/ADIM;kd_post=m['post_approx_kl']/ADIM;lr_used=cur_lr
    if kd_post>2*a.kl_target_dim:cur_lr=max(cur_lr/1.5,LR_MIN)
    elif kd_post<.5*a.kl_target_dim:cur_lr=min(cur_lr*1.5,LR_MAX)
    clip_steps+=m['clip_frac']>0;max_clip=max(max_clip,m['clip_frac'])
    row=dict(step=step,clip=clip,lr=lr_used,next_lr=cur_lr,kl_dim_pre=kd_pre,kl_dim_post=kd_post,raw_adv_mean=raw.mean().item(),raw_adv_std=raw.std().item(),q_within_state_std=within,
             grad_norm=float(gn),**m,clip_active_steps=clip_steps,elapsed_s=time.time()-t0,s_per_step=(time.time()-t0)/step)
    KLH.append(kd_pre);klm=sum(KLH[-a.kl_stop_window:])/len(KLH[-a.kl_stop_window:]);row['kl_dim_pre_rollmean']=klm
    append_jsonl(out/'metrics.jsonl',row);ROWS.append(row)
    if step%a.ope_every==0 or step==a.steps:
        q_new,a_new=ope(model);d=q_new-q_in
        rec=dict(step=step,gain_vs_round_start=d.mean().item(),se=(d.std()/math.sqrt(len(d))).item(),ope_new=q_new.mean().item(),ope_round_start=q_in.mean().item(),
                 action_dev_vs_round_start=(a_new-a_in)[...,JOINTS].abs().mean().item(),elapsed_s=time.time()-t0,lr=cur_lr)
        hist.append(rec);append_jsonl(out/'ope.jsonl',rec);print('OPE',json.dumps(rec),flush=True)
    if step in MS:save_candidate(step)
    write_json(out/'status.json',dict(status='running',total_steps=a.steps,candidates=len(cands),updated=cst(),last_train=row,last_ope=hist[-1] if hist else None))
    if a.kl_hard_mult>0 and len(KLH)>=a.kl_stop_window and klm>a.kl_hard_mult*a.kl_target_dim:
        kl_stop=dict(step=step,kl_dim_rollmean=klm,window=a.kl_stop_window,threshold=a.kl_hard_mult*a.kl_target_dim,kind='safety_hard_stop')
        print('KL_SAFETY_STOP',json.dumps(kl_stop),flush=True);append_jsonl(out/'kl_early_stop.jsonl',kl_stop);break
if not cands or cands[-1]['milestone']!=step:save_candidate(step)
assert all(torch.equal(v,enc_before[k]) for k,v in p.obs_encoder.state_dict().items()),'encoder changed'
def _mm(k):v=[r_[k] for r_ in ROWS];return dict(max=max(v),mean=sum(v)/len(v),last=v[-1])
rep=dict(status='done',updated=cst(),steps_run=step,candidates=cands,clip_active_steps=clip_steps,max_clip_frac=max_clip,clip_never_active=clip_steps==0,
         kl_early_stop=kl_stop,pi_old_refreshes=refreshes,critic_v_auc=VAUC,
         kl_summary={k:_mm(k) for k in ('approx_kl','anchor_kl','clip_frac','post_approx_kl','kl_dim_pre','kl_dim_post','lr','raw_adv_std')},final_train=row,ope_history_tail=hist[-3:])
write_json(out/'status.json',rep);write_json(out/'report.json',rep);print('PPO_DONE',json.dumps(rep),flush=True)
