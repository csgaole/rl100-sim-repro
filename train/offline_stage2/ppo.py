"""Offline policy improvement (RL-100 Stage-2 / upstream 'unio4' BPPO-style):
the K=10 DDIM denoising steps are an MDP; each step's Gaussian likelihood ratio new/old is optimised
with the PPO clipped objective, all steps share the chunk advantage A=min(Q1,Q2)(s,a0)-V(s) from IQL
(batch-normalised, upstream 'scalar_iql'), actions a0 sampled from the OLD stochastic policy at dataset
states. Old policy is replaced only when the OPE gate accepts (upstream is_update_old_policy).
OPE gate (deviation: Q-based, no learned dynamics / AM-Q): on a fixed held-out state set, paired
deployment (mean-sampler, shared initial noise) actions of new vs old policy are scored by minQ;
accept if mean improvement > 2 standard errors and > min_gain."""
from s2common import *
import argparse,copy
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--critic',required=True);ap.add_argument('--rounds',required=True)
ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=400);ap.add_argument('--batch',type=int,default=128)
ap.add_argument('--lr',type=float,default=1e-6);ap.add_argument('--clip',type=float,default=.25);ap.add_argument('--clip-final',type=float,default=.1)
ap.add_argument('--refresh',choices=['step','accept'],default='step');ap.add_argument('--revert-after',type=int,default=4)
ap.add_argument('--ope-every',type=int,default=25);ap.add_argument('--ope-states',type=int,default=2048);ap.add_argument('--patience',type=int,default=12)
ap.add_argument('--min-gain',type=float,default=1e-3);ap.add_argument('--grad-clip',type=float,default=1.);ap.add_argument('--accept-any',action='store_true',help='SMOKE ONLY')
a=ap.parse_args()
# Hard cap (measured ~4-9 s/outer step on shared GPUs); override with S2_PPO_MAX_STEPS.
a.steps=min(a.steps,int(os.environ.get('S2_PPO_MAX_STEPS','400')));setup(4321);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
rounds=[int(r) for r in a.rounds.split(',') if r!=''];D=load_dataset(OUT,rounds)
ck=torch.load(a.critic,map_location='cuda',weights_only=False);stats={k:v.cuda() for k,v in ck['stats'].items()}
C=Critic(2*FEAT+1,96).cuda();C.load_state_dict(ck['critic']);C.eval().requires_grad_(False)
p,cp=load_policy(a.policy);p.eval();p.obs_encoder.requires_grad_(False);p.normalizer.requires_grad_(False)
enc_before={k:v.clone() for k,v in p.obs_encoder.state_dict().items()}
model=p.model;old=copy.deepcopy(model).eval().requires_grad_(False);acc_model=copy.deepcopy(model).eval().requires_grad_(False)
opt=torch.optim.Adam(model.parameters(),lr=a.lr)
train_e=torch.nonzero(~D['heldout']).flatten();held_e=torch.nonzero(D['heldout']).flatten()
g=torch.Generator(device='cuda').manual_seed(99)
# fixed paired OPE set: held-out episodes, all phases
ge=torch.Generator(device='cuda').manual_seed(2026)
oe=held_e[torch.randint(len(held_e),(a.ope_states,),device='cuda',generator=ge)];ot=torch.randint(CHUNKS,(a.ope_states,),device='cuda',generator=ge)
o_s=critic_state(D,stats,oe,ot);o_c=policy_cond(D,oe,ot);o_noise=torch.randn((a.ope_states,8,14),device='cuda',generator=ge)
@torch.no_grad()
def ope(m):
    q=[];acts=[]
    for i in range(0,a.ope_states,512):
        x=sample_chain(m,p,o_c[i:i+512],stochastic=False,noise=o_noise[i:i+512]);acts.append(x);q.append(C.qmin(o_s[i:i+512],joint_flat(x)))
    return torch.cat(q),torch.cat(acts)
q_ref,a_ref=ope(acc_model);q_stage_in=q_ref.clone();a_stage_in=a_ref.clone()
v_ope=C.value(o_s)
write_json(out/'status.json',dict(status='running',step=0,total_steps=a.steps,updated=cst(),ope_ref=q_ref.mean().item()))
proto=dict(vars(a),denoising_steps=10,ratio='sum of per-dim log-ratios over 12 joint dims x 8 steps (grippers excluded)',
    advantage='minQ(s,a0)-V(s), batch-normalised',kernel=f'upstream step_mean (eta=1) mean, std clamp [{STD_MIN},{STD_MAX}]',
    encoder='frozen',old_policy_update=('every outer step (standard PPO snapshot)' if a.refresh=='step' else 'on OPE acceptance (upstream)'),revert='model<-last accepted + optimizer reset after every revert_after consecutive rejections',clip_schedule='linear clip->clip_final over steps',
    ope='paired minQ of deployment (mean) actions on fixed held-out states; accept if gain>max(2SE,min_gain)')
write_json(out/'protocol.json',proto)
n_acc=0;since=0;t0=time.time();accepted=[];hist=[]
for step in range(1,a.steps+1):
    clip=a.clip+(a.clip_final-a.clip)*(step-1)/max(1,a.steps-1)
    if a.refresh=='step':old.load_state_dict(model.state_dict())
    e=train_e[torch.randint(len(train_e),(a.batch,),device='cuda',generator=g)];t=torch.randint(CHUNKS,(a.batch,),device='cuda',generator=g)
    s=critic_state(D,stats,e,t);cond=policy_cond(D,e,t)
    with torch.no_grad():
        x,xs,ys,lps=sample_chain(old,p,cond,g,stochastic=True,trace=True)
        raw=C.qmin(s,joint_flat(x))-C.value(s);adv=(raw-raw.mean())/(raw.std()+1e-8)
    m=dict(loss=0.,clip_frac=0.,approx_kl=0.,max_abs_logratio=0.)
    for i,ts in enumerate(p.noise_scheduler.timesteps):
        eps=model(xs[i],ts.to('cuda'),global_cond=cond);mean,std=kernel(p,eps,ts,xs[i])
        logr=logprob(mean,std,ys[i])-lps[i]
        if i==0:
            l0=logr.abs().max().item()
            if step==1:assert l0<1e-2,f'initial log-ratio {l0}'
        ratio=logr.clamp(-20,20).exp()
        loss=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
        assert torch.isfinite(loss),'non-finite loss'
        opt.zero_grad(set_to_none=True);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip);opt.step()
        with torch.no_grad():
            m['loss']+=loss.item()/10;m['clip_frac']+=((ratio-1).abs()>clip).float().mean().item()/10
            m['approx_kl']+=((ratio-1)-logr.clamp(-20,20)).mean().item()/10;m['max_abs_logratio']=max(m['max_abs_logratio'],logr.abs().max().item())
    row=dict(step=step,clip=clip,initial_max_abs_logratio=l0,raw_adv_mean=raw.mean().item(),raw_adv_std=raw.std().item(),grad_norm=float(gn),**m,elapsed_s=time.time()-t0,s_per_step=(time.time()-t0)/step)
    if step%10==0 or step==1:append_jsonl(out/'metrics.jsonl',row)
    if step%a.ope_every==0 or step==a.steps:
        q_new,a_new=ope(model);d=q_new-q_ref;gain=d.mean().item();se=(d.std()/math.sqrt(len(d))).item()
        acc=gain>max(2*se,a.min_gain) or a.accept_any
        dev_old=(a_new-a_ref)[...,JOINTS].abs().mean().item();dev_in=(a_new-a_stage_in)[...,JOINTS].abs().mean().item()
        rec=dict(step=step,gain_vs_old=gain,se=se,accepted=bool(acc),ope_new=q_new.mean().item(),ope_old=q_ref.mean().item(),ope_stage_input=q_stage_in.mean().item(),
            gain_vs_stage_input=(q_new-q_stage_in).mean().item(),adv_vs_V=(q_new-v_ope).mean().item(),action_dev_vs_old=dev_old,action_dev_vs_stage_input=dev_in,elapsed_s=time.time()-t0)
        if acc:
            n_acc+=1;since=0;acc_model.load_state_dict(model.state_dict());q_ref=q_new;a_ref=a_new
            if a.refresh=='accept':old.load_state_dict(model.state_dict())
            save_policy(p,cp,out/'accepted_latest.pt',dict(stage='offline_stage2_ppo',source_policy=a.policy,critic=a.critic,step=step,accepted_index=n_acc,ope=rec));accepted.append(rec)
        else:
            since+=1
            if since%a.revert_after==0:
                model.load_state_dict(acc_model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=a.lr)
                if a.refresh=='accept':old.load_state_dict(acc_model.state_dict())
                rec['reverted_to_last_accepted']=True
        hist.append(rec);append_jsonl(out/'ope.jsonl',rec);print('OPE',json.dumps(rec),flush=True)
        write_json(out/'status.json',dict(status='running',total_steps=a.steps,accepted=n_acc,evals_since_accept=since,updated=cst(),last_train=row,last_ope=rec))
        if since>=a.patience:print('PATIENCE_STOP',flush=True);break
    if step%10==0:write_json(out/'status.json',dict(status='running',total_steps=a.steps,accepted=n_acc,evals_since_accept=since,updated=cst(),last_train=row,last_ope=hist[-1] if hist else None))
assert all(torch.equal(v,enc_before[k]) for k,v in p.obs_encoder.state_dict().items()),'encoder changed'
rep=dict(status='done',updated=cst(),steps_run=step,accepted=n_acc,last_accepted=accepted[-1] if accepted else None,
    candidate=str(out/'accepted_latest.pt') if n_acc else None,stop_reason='patience' if since>=a.patience else 'max_steps')
write_json(out/'status.json',rep);write_json(out/'report.json',rep);print('PPO_DONE',json.dumps(rep),flush=True)
