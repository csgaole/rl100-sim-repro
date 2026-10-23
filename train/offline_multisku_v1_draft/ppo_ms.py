# [offline_multisku_v1_draft] NEW file, multi-SKU copy of offline_stage2_v3_nominal_20261017/ppo_wide_fix3.py (STEP 3: offline PPO).
# Rule fixes vs base:
#  * pi_old = ROUND-START policy, frozen for the whole round (RL-100 off-policy offline PPO). Base refreshed old<-model EVERY step
#    with one optimizer step per batch, so the pre-update ratio was always exactly 1 and the clip never acted. Now the ratio
#    pi_theta/pi_old grows as theta drifts from the round start, so clip(1-eps,1+eps) really binds; clip activity is logged and reported.
#  * KL anchor to the round-start policy (= pi_old) with coef --anchor (pipeline passes 6 every round). Computed on the PPO batch
#    itself (all-data states), per non-terminal denoising step: 0.5*((mu_theta-mu_old)/sigma)^2, mean over joint dims (same scale
#    as base). Base: coef 1 (r1) / 3 (r2+) on a separate demo-only batch vs the stage-input policy.
#  * PPO states = ALL accumulated data (demos + every rollout, success and fail; MS_TRAIN_ALL=1) - base used wide-flagged rollouts only.
#  * OPE (paired minQ, deployment sampler) is LOG ONLY: no accept/revert/patience (base reverted weights on OPE rejections).
#  * critic state incl. sku_cond; batch default 256 (base 96 + 32 demo).
# Unchanged: K=10 DDIM eta=1 kernel, std clamp [.0067,S2_STD_MAX=.1], terminal step (sigma=0) skipped in loss/KL, per-step
#   Gaussian log-prob over 12 joint dims, chunk advantage A=minQ(s,a0)-V(s) with a0 ~ pi_old (batch-normalised), clip .2 -> .1
#   linear, grad clip 1, Adam, encoder + normalizer frozen (asserted), milestone snapshots of CURRENT weights for the gate.
from ms_common import *
import argparse,copy
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True,help='round-start policy (= pi_old = KL anchor)');ap.add_argument('--critic',required=True);ap.add_argument('--sources',required=True)
ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=300);ap.add_argument('--batch',type=int,default=256)
ap.add_argument('--lr',type=float,default=3e-7);ap.add_argument('--clip',type=float,default=.2);ap.add_argument('--clip-final',type=float,default=.1)
ap.add_argument('--anchor',type=float,default=6.);ap.add_argument('--ope-every',type=int,default=25);ap.add_argument('--ope-states',type=int,default=2048)
ap.add_argument('--grad-clip',type=float,default=1.);ap.add_argument('--milestones',default='150,300')
a=ap.parse_args();setup(4321);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
MS=sorted(int(x) for x in a.milestones.split(','))
sources=json.loads(Path(a.sources).read_text());D=load_dataset_w(sources)
ck=torch.load(a.critic,map_location='cuda',weights_only=False);stats={k:v.cuda() for k,v in ck['stats'].items()}
C=Critic(SD,ACT_DIM).cuda();C.load_state_dict(ck['critic']);C.eval().requires_grad_(False)
p,cp=load_policy(a.policy);p.eval();p.obs_encoder.requires_grad_(False);p.normalizer.requires_grad_(False)
enc_before={k:v.clone() for k,v in p.obs_encoder.state_dict().items()}
model=p.model
old=copy.deepcopy(model).eval().requires_grad_(False)   # pi_old = round-start policy: FIXED for the round, also the KL anchor reference
opt=torch.optim.Adam(model.parameters(),lr=a.lr)
isdemo=torch.tensor([s=='demo' for s in D['source']],device='cuda')
tr=torch.nonzero(train_mask(D)).flatten();ho=torch.nonzero(D['heldout']).flatten()
assert len(tr)>0 and len(ho)>0 and bool(isdemo[tr].any()) and bool((~isdemo[tr]).any()),'PPO data must contain demos AND rollouts'
g=torch.Generator(device='cuda').manual_seed(99);ge=torch.Generator(device='cuda').manual_seed(2026)
def fixed_set(eps,n):
    e=eps[torch.randint(len(eps),(n,),device='cuda',generator=ge)];t=torch.randint(CHUNKS,(n,),device='cuda',generator=ge)
    return critic_state(D,stats,e,t),policy_cond(D,e,t),torch.randn((n,8,14),device='cuda',generator=ge)
o_s,o_c,o_n=fixed_set(ho,a.ope_states)
@torch.no_grad()
def deploy(m,c,n):return torch.cat([sample_chain(m,p,c[i:i+512],stochastic=False,noise=n[i:i+512]) for i in range(0,len(c),512)])
@torch.no_grad()
def ope(m):x=deploy(m,o_c,o_n);return C.qmin(o_s,joint_flat(x)),x
q_in,a_in=ope(old)
write_json(out/'protocol.json',dict(vars(a),dataset=dict(n=D['n'],train=len(tr),heldout=len(ho),demo_train=int(isdemo[tr].sum()),rollout_train=int((~isdemo[tr]).sum()),train_all=TRAIN_ALL),
    pi_old='round-start policy, fixed for the whole round',kernel=f'DDIM10 eta=1 step mean, std clamp [{STD_MIN},{STD_MAX}], terminal sigma 0 (skipped)',
    advantage='minQ(s,a0)-V(s), a0~pi_old, all-data states, batch-normalised',anchor=f'{a.anchor} x 0.5*((mu_theta-mu_old)/sigma)^2 mean over joint dims, summed over non-terminal steps / NST, same PPO batch',
    ope='log only (no accept/revert)'))
cands=[];mi=0;t0=time.time();clip_steps=0;max_clip=0.;hist=[]
def save_candidate(step):
    with torch.no_grad():a_now=deploy(model,o_c,o_n);dev=(a_now-a_in)[...,JOINTS].abs().mean().item()
    rec=dict(path=str(out/f'cand_step{step:04d}.pt'),milestone=step,ope=(hist[-1] if hist else None),action_dev_vs_round_start=dev)
    save_policy(p,cp,rec['path'],dict(stage='offline_multisku_ppo_draft',source_policy=a.policy,critic=a.critic,anchor=a.anchor,lr=a.lr,**{k:v for k,v in rec.items() if k!='path'}))
    cands.append(rec);append_jsonl(out/'candidates.jsonl',rec);print('CANDIDATE',json.dumps(rec),flush=True)
p.noise_scheduler.set_timesteps(10);SI=[(i,ts) for i,ts in enumerate(p.noise_scheduler.timesteps) if not is_terminal(p,ts)];NST=len(SI);print('stochastic steps',NST,flush=True)
for step in range(1,a.steps+1):
    clip=a.clip+(a.clip_final-a.clip)*(step-1)/max(1,a.steps-1)
    e=tr[torch.randint(len(tr),(a.batch,),device='cuda',generator=g)];t=torch.randint(CHUNKS,(a.batch,),device='cuda',generator=g)
    s=critic_state(D,stats,e,t);cond=policy_cond(D,e,t)
    with torch.no_grad():
        x,xs,ys,lps=sample_chain(old,p,cond,g,stochastic=True,trace=True)          # behaviour chain from the FIXED pi_old
        raw=C.qmin(s,joint_flat(x))-C.value(s);adv=(raw-raw.mean())/(raw.std()+1e-8)
        oldk=[kernel(p,old(xs[i],ts.to('cuda'),global_cond=cond),ts,xs[i]) for i,ts in SI]
    m=dict(loss=0.,anchor_kl=0.,clip_frac=0.,approx_kl=0.,ratio_mean=0.,max_abs_logratio=0.,post_clip_frac=0.,post_approx_kl=0.,post_max_abs_logratio=0.)
    opt.zero_grad(set_to_none=True);l0=0.
    for (i,ts),(mo,so) in zip(SI,oldk):
        eps=model(xs[i],ts.to('cuda'),global_cond=cond);mean,std=kernel(p,eps,ts,xs[i])
        logr=logprob(mean,std,ys[i])-lps[i];l0=max(l0,logr.abs().max().item())
        ratio=logr.clamp(-20,20).exp()
        lppo=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
        kl=(.5*((mean-mo)/so).square())[...,JOINTS].mean()
        loss=(lppo+a.anchor*kl)/NST
        assert torch.isfinite(loss),'non-finite loss'
        loss.backward()
        with torch.no_grad():
            m['loss']+=lppo.item()/NST;m['anchor_kl']+=kl.item();m['clip_frac']+=((ratio-1).abs()>clip).float().mean().item()/NST;m['ratio_mean']+=ratio.mean().item()/NST
            m['approx_kl']+=((ratio-1)-logr.clamp(-20,20)).mean().item()/NST;m['max_abs_logratio']=max(m['max_abs_logratio'],logr.abs().max().item())
    if step==1:assert l0<1e-2,f'initial log-ratio {l0} (model must equal pi_old at round start)'
    gn=nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip);opt.step()
    with torch.no_grad():
        for i,ts in SI:
            mean,std=kernel(p,model(xs[i],ts.to('cuda'),global_cond=cond),ts,xs[i]);logr=(logprob(mean,std,ys[i])-lps[i]).clamp(-20,20);ratio=logr.exp()
            m['post_clip_frac']+=((ratio-1).abs()>clip).float().mean().item()/NST;m['post_approx_kl']+=((ratio-1)-logr).mean().item()/NST;m['post_max_abs_logratio']=max(m['post_max_abs_logratio'],logr.abs().max().item())
    clip_steps+=m['clip_frac']>0;max_clip=max(max_clip,m['clip_frac'])
    row=dict(step=step,clip=clip,raw_adv_mean=raw.mean().item(),raw_adv_std=raw.std().item(),grad_norm=float(gn),**m,clip_active_steps=clip_steps,elapsed_s=time.time()-t0,s_per_step=(time.time()-t0)/step)
    append_jsonl(out/'metrics.jsonl',row)
    if step%a.ope_every==0 or step==a.steps:
        q_new,a_new=ope(model);d=q_new-q_in
        rec=dict(step=step,gain_vs_round_start=d.mean().item(),se=(d.std()/math.sqrt(len(d))).item(),ope_new=q_new.mean().item(),ope_round_start=q_in.mean().item(),
                 action_dev_vs_round_start=(a_new-a_in)[...,JOINTS].abs().mean().item(),elapsed_s=time.time()-t0,log_only=True)
        hist.append(rec);append_jsonl(out/'ope.jsonl',rec);print('OPE',json.dumps(rec),flush=True)
    while mi<len(MS) and step>=MS[mi]:save_candidate(MS[mi]);mi+=1
    if step%10==0:write_json(out/'status.json',dict(status='running',total_steps=a.steps,candidates=len(cands),updated=cst(),last_train=row,last_ope=hist[-1] if hist else None))
if not cands or cands[-1]['milestone']!=step:save_candidate(step)
assert all(torch.equal(v,enc_before[k]) for k,v in p.obs_encoder.state_dict().items()),'encoder changed'
rep=dict(status='done',updated=cst(),steps_run=step,candidates=cands,clip_active_steps=clip_steps,max_clip_frac=max_clip,clip_never_active=clip_steps==0,
         final_train=row,ope_history_tail=hist[-3:])
write_json(out/'status.json',rep);write_json(out/'report.json',rep);print('PPO_DONE',json.dumps(rep),flush=True)
