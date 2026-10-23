# [v3fix3] ppo_wide_fix2 + milestone snapshots of CURRENT weights (closed-loop selects; OPE is log-only)
# [v3fix2] ppo_wide_fix + one optimizer step per batch (loss mean over 9 sigma>0 steps), lr default 3e-7, post-update ratio stats
# [v3fix] ppo_wide using fixed kernel; terminal (sigma=0) step skipped in PPO and anchor
# [offline_stage2_v3 unchanged copy of train/offline_stage2_wide/ppo_wide.py; imports the local v3 wcommon]
"""Offline PPO on the diffusion denoising MDP (same formulation as train/offline_stage2/ppo.py: K=10 DDIM
steps, per-step Gaussian likelihood around the upstream step_mean with std clamp [.0067,S2_STD_MAX], PPO clip
on the summed per-step log-ratio of the 12 joint dims, chunk advantage A=minQ(s,a0)-V(s) from IQL with a0 drawn
from the old stochastic policy, old policy refreshed every outer step, encoder frozen).
Wide-range changes (Stage-2-wide):
 * PPO states are sampled ONLY from wide-range rollouts (train split); advantage normalised within that batch.
 * Nominal anchor: on demo states, KL( new step kernel || stage-input step kernel ) summed over the 10 steps
   (Gaussian with shared std => 0.5*((mu_new-mu_ref)/std)^2, mean over joint dims), weight --anchor.
 * OPE (paired minQ of deployment-sampler actions on fixed held-out WIDE states) is only a sanity filter:
   accept if gain>2SE and gain>min_gain (default 0). The real closed-loop gate (pipeline) decides promotion.
 * Candidate snapshots of the last OPE-accepted policy are saved at milestone steps for the real gate."""
from wcommon_fix import *
import argparse,copy
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--critic',required=True);ap.add_argument('--sources',required=True)
ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=600);ap.add_argument('--batch',type=int,default=96);ap.add_argument('--demo-batch',type=int,default=32)
ap.add_argument('--lr',type=float,default=3e-7);ap.add_argument('--clip',type=float,default=.2);ap.add_argument('--clip-final',type=float,default=.1)
ap.add_argument('--anchor',type=float,default=1.);ap.add_argument('--revert-after',type=int,default=4)
ap.add_argument('--ope-every',type=int,default=25);ap.add_argument('--ope-states',type=int,default=2048);ap.add_argument('--patience',type=int,default=12)
ap.add_argument('--min-gain',type=float,default=0.);ap.add_argument('--grad-clip',type=float,default=1.);ap.add_argument('--milestones',default='150,300,450,600')  # ic5 r2+: 50,100,150
ap.add_argument('--accept-any',action='store_true',help='SMOKE ONLY')
a=ap.parse_args();setup(4321);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
MS=sorted(int(x) for x in a.milestones.split(','))
sources=json.loads(Path(a.sources).read_text());D=load_dataset_w(sources)
ck=torch.load(a.critic,map_location='cuda',weights_only=False);stats={k:v.cuda() for k,v in ck['stats'].items()}
C=Critic(2*FEAT+1,96).cuda();C.load_state_dict(ck['critic']);C.eval().requires_grad_(False)
p,cp=load_policy(a.policy);p.eval();p.obs_encoder.requires_grad_(False);p.normalizer.requires_grad_(False)
enc_before={k:v.clone() for k,v in p.obs_encoder.state_dict().items()}
model=p.model;old=copy.deepcopy(model).eval().requires_grad_(False);acc_model=copy.deepcopy(model).eval().requires_grad_(False)
ref=copy.deepcopy(model).eval().requires_grad_(False)   # stage-input policy (anchor + deviation reference)
opt=torch.optim.Adam(model.parameters(),lr=a.lr)
isdemo=torch.tensor([s=='demo' for s in D['source']],device='cuda')
wide_tr=torch.nonzero(~D['heldout']&D['wide']).flatten();wide_ho=torch.nonzero(D['heldout']&D['wide']).flatten()
demo_tr=torch.nonzero(~D['heldout']&isdemo).flatten();demo_ho=torch.nonzero(D['heldout']&isdemo).flatten()
assert len(wide_tr)>0 and len(wide_ho)>0 and len(demo_tr)>0
g=torch.Generator(device='cuda').manual_seed(99);ge=torch.Generator(device='cuda').manual_seed(2026)
def fixed_set(eps,n):
    e=eps[torch.randint(len(eps),(n,),device='cuda',generator=ge)];t=torch.randint(CHUNKS,(n,),device='cuda',generator=ge)
    return critic_state(D,stats,e,t),policy_cond(D,e,t),torch.randn((n,8,14),device='cuda',generator=ge)
o_s,o_c,o_n=fixed_set(wide_ho,a.ope_states);d_s,d_c,d_n=fixed_set(demo_ho,512)
@torch.no_grad()
def deploy(m,c,n):
    return torch.cat([sample_chain(m,p,c[i:i+512],stochastic=False,noise=n[i:i+512]) for i in range(0,len(c),512)])
@torch.no_grad()
def ope(m):
    x=deploy(m,o_c,o_n);return C.qmin(o_s,joint_flat(x)),x
q_ref,a_ref=ope(acc_model);q_in=q_ref.clone();a_in=a_ref.clone();d_in=deploy(ref,d_c,d_n);qd_in=C.qmin(d_s,joint_flat(d_in))
write_json(out/'protocol.json',dict(vars(a),dataset=dict(n=D['n'],wide_train=len(wide_tr),wide_heldout=len(wide_ho),demo_train=len(demo_tr)),
    kernel=f'upstream step_mean mean, std clamp [{STD_MIN},{STD_MAX}]',advantage='minQ(s,a0)-V(s) on wide-range rollout states, batch-normalised',
    anchor='0.5*((mu_new-mu_ref)/std)^2 mean over joint dims, summed over 10 denoising steps, demo states, ref=stage-input policy',
    ope='sanity filter only: paired minQ, deployment actions, held-out wide states; accept if gain>max(2SE,min_gain)'))
n_acc=0;since=0;t0=time.time();accepted=[];hist=[];cands=[];last_acc_step=0;mi=0
def save_candidate(step):
    # [fix3] snapshot CURRENT policy at milestone for closed-loop selection (OPE no longer gates which weights are saved)
    with torch.no_grad():
        dd=deploy(model,d_c,d_n);dev_demo=(dd-d_in)[...,JOINTS].abs().mean().item()
        a_now=deploy(model,o_c,o_n);dev_wide=(a_now-a_in)[...,JOINTS].abs().mean().item()
    rec=dict(path=str(out/f'cand_step{step:04d}.pt'),milestone=step,accepted_step=last_acc_step,ope=(accepted[-1] if accepted else None),
             action_dev_demo_states=dev_demo,action_dev_wide_states=dev_wide)
    save_policy(p,cp,rec['path'],dict(stage='offline_stage2_wide_ppo_fix3',source_policy=a.policy,critic=a.critic,**{k:v for k,v in rec.items() if k!='path'}))
    cands.append(rec);append_jsonl(out/'candidates.jsonl',rec);print('CANDIDATE',json.dumps(rec),flush=True)
p.noise_scheduler.set_timesteps(10);NST=sum(not is_terminal(p,t) for t in p.noise_scheduler.timesteps);print('stochastic steps',NST,flush=True)
for step in range(1,a.steps+1):
    clip=a.clip+(a.clip_final-a.clip)*(step-1)/max(1,a.steps-1)
    old.load_state_dict(model.state_dict())
    e=wide_tr[torch.randint(len(wide_tr),(a.batch,),device='cuda',generator=g)];t=torch.randint(CHUNKS,(a.batch,),device='cuda',generator=g)
    s=critic_state(D,stats,e,t);cond=policy_cond(D,e,t)
    ed=demo_tr[torch.randint(len(demo_tr),(a.demo_batch,),device='cuda',generator=g)];td=torch.randint(CHUNKS,(a.demo_batch,),device='cuda',generator=g);cd=policy_cond(D,ed,td)
    with torch.no_grad():
        x,xs,ys,lps=sample_chain(old,p,cond,g,stochastic=True,trace=True)
        raw=C.qmin(s,joint_flat(x))-C.value(s);adv=(raw-raw.mean())/(raw.std()+1e-8)
        _,xd,_,_=sample_chain(ref,p,cd,g,stochastic=True,trace=True)
    m=dict(loss=0.,anchor_kl=0.,clip_frac=0.,approx_kl=0.,max_abs_logratio=0.,post_clip_frac=0.,post_approx_kl=0.,post_max_abs_logratio=0.)
    # [fix2] ONE optimizer step per batch: loss averaged over the NST positive-variance denoising steps
    SI=[(i,ts) for i,ts in enumerate(p.noise_scheduler.timesteps) if not is_terminal(p,ts)]
    with torch.no_grad():refk=[kernel(p,ref(xd[i],ts.to('cuda'),global_cond=cd),ts,xd[i]) for i,ts in SI]
    opt.zero_grad(set_to_none=True);tot=0.;l0=0.
    for (i,ts),(mr,sr) in zip(SI,refk):
        eps=model(xs[i],ts.to('cuda'),global_cond=cond);mean,std=kernel(p,eps,ts,xs[i])
        logr=logprob(mean,std,ys[i])-lps[i];l0=max(l0,logr.abs().max().item())
        ratio=logr.clamp(-20,20).exp()
        lppo=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
        mn,_=kernel(p,model(xd[i],ts.to('cuda'),global_cond=cd),ts,xd[i])
        kl=(.5*((mn-mr)/sr).square())[...,JOINTS].mean()
        loss=(lppo+a.anchor*kl)/NST
        assert torch.isfinite(loss),'non-finite loss'
        loss.backward();tot+=loss.item()
        with torch.no_grad():
            m['loss']+=lppo.item()/NST;m['anchor_kl']+=kl.item();m['clip_frac']+=((ratio-1).abs()>clip).float().mean().item()/NST
            m['approx_kl']+=((ratio-1)-logr.clamp(-20,20)).mean().item()/NST;m['max_abs_logratio']=max(m['max_abs_logratio'],logr.abs().max().item())
    if step==1:assert l0<1e-2,f'initial log-ratio {l0}'
    gn=nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip);opt.step()
    with torch.no_grad():  # post-update ratio on the same batch = true policy shift per update
        for i,ts in SI:
            mean,std=kernel(p,model(xs[i],ts.to('cuda'),global_cond=cond),ts,xs[i]);logr=(logprob(mean,std,ys[i])-lps[i]).clamp(-20,20);ratio=logr.exp()
            m['post_clip_frac']+=((ratio-1).abs()>clip).float().mean().item()/NST;m['post_approx_kl']+=((ratio-1)-logr).mean().item()/NST;m['post_max_abs_logratio']=max(m['post_max_abs_logratio'],logr.abs().max().item())
    row=dict(step=step,clip=clip,initial_max_abs_logratio=l0,raw_adv_mean=raw.mean().item(),raw_adv_std=raw.std().item(),grad_norm=float(gn),**m,elapsed_s=time.time()-t0,s_per_step=(time.time()-t0)/step)
    if True:append_jsonl(out/'metrics.jsonl',row)
    if step%a.ope_every==0 or step==a.steps:
        q_new,a_new=ope(model);d=q_new-q_ref;gain=d.mean().item();se=(d.std()/math.sqrt(len(d))).item()
        acc=(gain>max(2*se,a.min_gain)) or a.accept_any
        rec=dict(step=step,gain_vs_old=gain,se=se,accepted=bool(acc),ope_new=q_new.mean().item(),ope_old=q_ref.mean().item(),gain_vs_stage_input=(q_new-q_in).mean().item(),
            action_dev_vs_old=(a_new-a_ref)[...,JOINTS].abs().mean().item(),action_dev_vs_stage_input=(a_new-a_in)[...,JOINTS].abs().mean().item(),elapsed_s=time.time()-t0)
        if acc:
            n_acc+=1;since=0;acc_model.load_state_dict(model.state_dict());q_ref=q_new;a_ref=a_new;last_acc_step=step;accepted.append(rec)
        else:
            since+=1
            if since%a.revert_after==0:
                model.load_state_dict(acc_model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=a.lr);rec['reverted_to_last_accepted']=True
        hist.append(rec);append_jsonl(out/'ope.jsonl',rec);print('OPE',json.dumps(rec),flush=True)
        write_json(out/'status.json',dict(status='running',total_steps=a.steps,accepted=n_acc,evals_since_accept=since,candidates=len(cands),updated=cst(),last_train=row,last_ope=rec))
    while mi<len(MS) and step>=MS[mi]:save_candidate(MS[mi]);mi+=1
    if since>=a.patience:print('PATIENCE_STOP',flush=True);break
    if step%10==0:write_json(out/'status.json',dict(status='running',total_steps=a.steps,accepted=n_acc,evals_since_accept=since,candidates=len(cands),updated=cst(),last_train=row,last_ope=hist[-1] if hist else None))
save_candidate(step)
assert all(torch.equal(v,enc_before[k]) for k,v in p.obs_encoder.state_dict().items()),'encoder changed'
rep=dict(status='done',updated=cst(),steps_run=step,accepted=n_acc,candidates=cands,stop_reason='patience' if since>=a.patience else 'max_steps',ope_history_tail=hist[-3:])
write_json(out/'status.json',rep);write_json(out/'report.json',rep);print('PPO_DONE',json.dumps(rep),flush=True)
