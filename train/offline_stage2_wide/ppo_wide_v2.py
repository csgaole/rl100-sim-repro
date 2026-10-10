"""Offline PPO v2 (Stage-2-wide continuation with nominal protection). Same diffusion-denoising PPO as ppo_wide.py
(K=10 DDIM steps, per-step Gaussian likelihood around upstream step_mean, std clamp [.0067,S2_STD_MAX], PPO clip on
summed per-step log-ratio of the 12 joint dims, A=minQ(s,a0)-V(s) from IQL, old policy refreshed every outer step,
encoder frozen). v2 changes:
 * Start policy = --policy (incumbent, e.g. round-2 candidate), not Stage-1.
 * Advantage batch = --batch wide-range rollout states + --nom-batch NOMINAL-range rollout states (+-5mm/+-2deg,
   collected with the current policy); advantages normalised separately within each group, then concatenated.
 * Nominal anchor (stronger, fixed reference = STAGE-1): KL(new step kernel || Stage-1 step kernel) summed over the
   10 denoising steps on demo states (--demo-batch) + nominal rollout states (--nom-anchor-batch), weight --anchor.
   No anchor on wide states (as before).
 * OPE (paired minQ, deployment actions, held-out WIDE states) is only a sanity filter for snapshotting candidates;
   nominal-state OPE and action deviation vs Stage-1 on demo/nominal states are logged. Real gate decides."""
from wcommon import *
import argparse,copy
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--anchor-ref',default=str(STAGE1))
ap.add_argument('--critic',required=True);ap.add_argument('--sources',required=True)
ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=600);ap.add_argument('--batch',type=int,default=96)
ap.add_argument('--nom-batch',type=int,default=32);ap.add_argument('--demo-batch',type=int,default=32);ap.add_argument('--nom-anchor-batch',type=int,default=32)
ap.add_argument('--lr',type=float,default=2e-6);ap.add_argument('--clip',type=float,default=.2);ap.add_argument('--clip-final',type=float,default=.1)
ap.add_argument('--anchor',type=float,default=3.);ap.add_argument('--revert-after',type=int,default=4)
ap.add_argument('--ope-every',type=int,default=25);ap.add_argument('--ope-states',type=int,default=2048);ap.add_argument('--patience',type=int,default=12)
ap.add_argument('--min-gain',type=float,default=0.);ap.add_argument('--grad-clip',type=float,default=1.);ap.add_argument('--milestones',default='150,300,450,600')
ap.add_argument('--accept-any',action='store_true',help='SMOKE ONLY')
ap.add_argument('--force-save-steps',default='',help='comma list: save the CURRENT policy at these steps regardless of OPE (diagnostic)')
ap.add_argument('--no-revert',action='store_true',help='never revert to last OPE-accepted weights (raw PPO trajectory)')
ap.add_argument('--no-patience',action='store_true',help='never stop early on OPE patience')
a=ap.parse_args();setup(4321);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
MS=sorted(int(x) for x in a.milestones.split(','));FORCE={int(x) for x in a.force_save_steps.split(',') if x}
sources=json.loads(Path(a.sources).read_text());D=load_dataset_w(sources)
ck=torch.load(a.critic,map_location='cuda',weights_only=False);stats={k:v.cuda() for k,v in ck['stats'].items()}
C=Critic(2*FEAT+1,96).cuda();C.load_state_dict(ck['critic']);C.eval().requires_grad_(False)
p,cp=load_policy(a.policy);p.eval();p.obs_encoder.requires_grad_(False);p.normalizer.requires_grad_(False)
p1,_=load_policy(a.anchor_ref);p1.eval()
enc_before={k:v.clone() for k,v in p.obs_encoder.state_dict().items()}
assert all(torch.equal(v.cpu(),p1.obs_encoder.state_dict()[k].cpu()) for k,v in enc_before.items()),'encoder differs from anchor reference (features would be inconsistent)'
model=p.model;old=copy.deepcopy(model).eval().requires_grad_(False);acc_model=copy.deepcopy(model).eval().requires_grad_(False)
inp=copy.deepcopy(model).eval().requires_grad_(False)          # stage-input policy (deviation reference)
ref=p1.model.cuda().eval().requires_grad_(False);del p1         # Stage-1 policy (nominal anchor reference)
opt=torch.optim.Adam(model.parameters(),lr=a.lr)
isdemo=torch.tensor([s=='demo' for s in D['source']],device='cuda')
wide_tr=torch.nonzero(~D['heldout']&D['wide']).flatten();wide_ho=torch.nonzero(D['heldout']&D['wide']).flatten()
nom_tr=torch.nonzero(~D['heldout']&D['nominal']).flatten();nom_ho=torch.nonzero(D['heldout']&D['nominal']).flatten()
demo_tr=torch.nonzero(~D['heldout']&isdemo).flatten();demo_ho=torch.nonzero(D['heldout']&isdemo).flatten()
assert len(wide_tr)>0 and len(wide_ho)>0 and len(demo_tr)>0 and len(nom_tr)>0,(len(wide_tr),len(wide_ho),len(demo_tr),len(nom_tr))
if len(nom_ho)==0:nom_ho=nom_tr   # smoke only (tiny data)
g=torch.Generator(device='cuda').manual_seed(99);ge=torch.Generator(device='cuda').manual_seed(2026)
def fixed_set(eps,n):
    e=eps[torch.randint(len(eps),(n,),device='cuda',generator=ge)];t=torch.randint(CHUNKS,(n,),device='cuda',generator=ge)
    return critic_state(D,stats,e,t),policy_cond(D,e,t),torch.randn((n,8,14),device='cuda',generator=ge)
o_s,o_c,o_n=fixed_set(wide_ho,a.ope_states);d_s,d_c,d_n=fixed_set(demo_ho,512);n_s,n_c,n_n=fixed_set(nom_ho,512)
@torch.no_grad()
def deploy(m,c,n):
    return torch.cat([sample_chain(m,p,c[i:i+512],stochastic=False,noise=n[i:i+512]) for i in range(0,len(c),512)])
@torch.no_grad()
def ope(m):
    x=deploy(m,o_c,o_n);return C.qmin(o_s,joint_flat(x)),x
q_ref,a_ref=ope(acc_model);q_in=q_ref.clone();a_in=a_ref.clone()
with torch.no_grad():
    d_s1=deploy(ref,d_c,d_n);n_s1=deploy(ref,n_c,n_n);d_inp=deploy(inp,d_c,d_n);n_inp=deploy(inp,n_c,n_n)
    qn_inp=C.qmin(n_s,joint_flat(n_inp))
def jdev(x,y):return (x-y)[...,JOINTS].abs().mean().item()
dev0=dict(input_vs_stage1_demo=jdev(d_inp,d_s1),input_vs_stage1_nominal=jdev(n_inp,n_s1))
print('INPUT_DEV',json.dumps(dev0),flush=True)
write_json(out/'protocol.json',dict(vars(a),dataset=dict(n=D['n'],wide_train=len(wide_tr),wide_heldout=len(wide_ho),nominal_train=len(nom_tr),nominal_heldout=len(nom_ho),demo_train=len(demo_tr)),
    input_deviation=dev0,kernel=f'upstream step_mean mean, std clamp [{STD_MIN},{STD_MAX}]',
    advantage='minQ(s,a0)-V(s); batch = wide rollout states + nominal rollout states, normalised per group',
    anchor='0.5*((mu_new-mu_stage1)/std)^2 mean over joint dims, summed over 10 denoising steps, on demo + nominal rollout states, ref=STAGE-1',
    ope='sanity filter only: paired minQ, deployment actions, held-out wide states; accept if gain>max(2SE,min_gain); nominal OPE logged'))
n_acc=0;since=0;t0=time.time();accepted=[];hist=[];cands=[];last_acc_step=0;saved_acc_step=-1;mi=0
def save_candidate(step):
    global saved_acc_step
    if last_acc_step==0 or last_acc_step==saved_acc_step:return
    cur={k:v.clone() for k,v in model.state_dict().items()};model.load_state_dict(acc_model.state_dict())
    with torch.no_grad():
        dd=deploy(model,d_c,d_n);nn_=deploy(model,n_c,n_n)
        rec=dict(path=str(out/f'cand_step{step:04d}.pt'),milestone=step,accepted_step=last_acc_step,ope=accepted[-1],
            action_dev_demo_states=jdev(dd,d_inp),action_dev_wide_states=jdev(a_ref,a_in),
            action_dev_vs_stage1_demo=jdev(dd,d_s1),action_dev_vs_stage1_nominal=jdev(nn_,n_s1),
            input_dev_vs_stage1_demo=dev0['input_vs_stage1_demo'],input_dev_vs_stage1_nominal=dev0['input_vs_stage1_nominal'],
            ope_nominal_gain_vs_input=(C.qmin(n_s,joint_flat(nn_))-qn_inp).mean().item())
    save_policy(p,cp,rec['path'],dict(stage='offline_stage2_wide_v2_ppo',source_policy=a.policy,anchor_ref=a.anchor_ref,critic=a.critic,**{k:v for k,v in rec.items() if k!='path'}))
    model.load_state_dict(cur);saved_acc_step=last_acc_step;cands.append(rec);append_jsonl(out/'candidates.jsonl',rec);print('CANDIDATE',json.dumps(rec),flush=True)
def norm(x):return (x-x.mean())/(x.std()+1e-8)
for step in range(1,a.steps+1):
    clip=a.clip+(a.clip_final-a.clip)*(step-1)/max(1,a.steps-1)
    old.load_state_dict(model.state_dict())
    ew=wide_tr[torch.randint(len(wide_tr),(a.batch,),device='cuda',generator=g)];en=nom_tr[torch.randint(len(nom_tr),(a.nom_batch,),device='cuda',generator=g)]
    e=torch.cat([ew,en]);t=torch.randint(CHUNKS,(len(e),),device='cuda',generator=g)
    s=critic_state(D,stats,e,t);cond=policy_cond(D,e,t)
    ed=demo_tr[torch.randint(len(demo_tr),(a.demo_batch,),device='cuda',generator=g)];ena=nom_tr[torch.randint(len(nom_tr),(a.nom_anchor_batch,),device='cuda',generator=g)]
    ea=torch.cat([ed,ena]);ta=torch.randint(CHUNKS,(len(ea),),device='cuda',generator=g);cd=policy_cond(D,ea,ta)
    with torch.no_grad():
        x,xs,ys,lps=sample_chain(old,p,cond,g,stochastic=True,trace=True)
        raw=C.qmin(s,joint_flat(x))-C.value(s);nb=a.batch;adv=torch.cat([norm(raw[:nb]),norm(raw[nb:])])
        _,xd,_,_=sample_chain(ref,p,cd,g,stochastic=True,trace=True)
    m=dict(loss=0.,anchor_kl=0.,anchor_kl_demo=0.,anchor_kl_nominal=0.,clip_frac=0.,clip_frac_nominal=0.,approx_kl=0.,max_abs_logratio=0.)
    for i,ts in enumerate(p.noise_scheduler.timesteps):
        eps=model(xs[i],ts.to('cuda'),global_cond=cond);mean,std=kernel(p,eps,ts,xs[i])
        logr=logprob(mean,std,ys[i])-lps[i]
        if i==0:
            l0=logr.abs().max().item()
            if step==1:assert l0<1e-2,f'initial log-ratio {l0}'
        ratio=logr.clamp(-20,20).exp()
        lppo=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
        with torch.no_grad():mr,sr=kernel(p,ref(xd[i],ts.to('cuda'),global_cond=cd),ts,xd[i])
        mn,_=kernel(p,model(xd[i],ts.to('cuda'),global_cond=cd),ts,xd[i])
        klv=(.5*((mn-mr)/sr).square())[...,JOINTS].mean((1,2));kl=klv.mean()
        loss=lppo+a.anchor*kl
        assert torch.isfinite(loss),'non-finite loss'
        opt.zero_grad(set_to_none=True);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip);opt.step()
        with torch.no_grad():
            cf=((ratio-1).abs()>clip).float()
            m['loss']+=lppo.item()/10;m['anchor_kl']+=kl.item();m['anchor_kl_demo']+=klv[:a.demo_batch].mean().item();m['anchor_kl_nominal']+=klv[a.demo_batch:].mean().item()
            m['clip_frac']+=cf.mean().item()/10;m['clip_frac_nominal']+=cf[nb:].mean().item()/10
            m['approx_kl']+=((ratio-1)-logr.clamp(-20,20)).mean().item()/10;m['max_abs_logratio']=max(m['max_abs_logratio'],logr.abs().max().item())
    row=dict(step=step,clip=clip,initial_max_abs_logratio=l0,raw_adv_mean_wide=raw[:nb].mean().item(),raw_adv_std_wide=raw[:nb].std().item(),
        raw_adv_mean_nominal=raw[nb:].mean().item(),raw_adv_std_nominal=raw[nb:].std().item(),grad_norm=float(gn),**m,elapsed_s=time.time()-t0,s_per_step=(time.time()-t0)/step)
    if step%10==0 or step==1:append_jsonl(out/'metrics.jsonl',row)
    if step%a.ope_every==0 or step==a.steps:
        q_new,a_new=ope(model);d=q_new-q_ref;gain=d.mean().item();se=(d.std()/math.sqrt(len(d))).item()
        with torch.no_grad():nx=deploy(model,n_c,n_n);dx_=deploy(model,d_c,d_n);qn=C.qmin(n_s,joint_flat(nx))
        acc=(gain>max(2*se,a.min_gain)) or a.accept_any
        rec=dict(step=step,gain_vs_old=gain,se=se,accepted=bool(acc),ope_new=q_new.mean().item(),ope_old=q_ref.mean().item(),gain_vs_stage_input=(q_new-q_in).mean().item(),
            ope_nominal_gain_vs_input=(qn-qn_inp).mean().item(),ope_nominal_se=((qn-qn_inp).std()/math.sqrt(len(qn))).item(),
            action_dev_vs_old=jdev(a_new,a_ref),action_dev_vs_stage_input=jdev(a_new,a_in),
            action_dev_vs_stage1_demo=jdev(dx_,d_s1),action_dev_vs_stage1_nominal=jdev(nx,n_s1),elapsed_s=time.time()-t0)
        if acc:
            n_acc+=1;since=0;acc_model.load_state_dict(model.state_dict());q_ref=q_new;a_ref=a_new;last_acc_step=step;accepted.append(rec)
        else:
            since+=1
            if since%a.revert_after==0 and not a.no_revert:
                model.load_state_dict(acc_model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=a.lr);rec['reverted_to_last_accepted']=True
        hist.append(rec);append_jsonl(out/'ope.jsonl',rec);print('OPE',json.dumps(rec),flush=True)
        write_json(out/'status.json',dict(status='running',total_steps=a.steps,accepted=n_acc,evals_since_accept=since,candidates=len(cands),updated=cst(),last_train=row,last_ope=rec))
    while mi<len(MS) and step>=MS[mi]:save_candidate(MS[mi]);mi+=1
    if step in FORCE:
        with torch.no_grad():
            qf,af=ope(model);nx=deploy(model,n_c,n_n);dx_=deploy(model,d_c,d_n);qn=C.qmin(n_s,joint_flat(nx));dd=qf-q_in;dn=qn-qn_inp
        rec=dict(path=str(out/f'forced_step{step:04d}.pt'),step=step,forced=True,ope_gain_vs_input=dd.mean().item(),ope_gain_vs_input_se=(dd.std()/math.sqrt(len(dd))).item(),
            ope_nominal_gain_vs_input=dn.mean().item(),ope_nominal_se=(dn.std()/math.sqrt(len(dn))).item(),action_dev_vs_stage_input=jdev(af,a_in),
            action_dev_vs_stage1_demo=jdev(dx_,d_s1),action_dev_vs_stage1_nominal=jdev(nx,n_s1),n_ope_accepted_so_far=n_acc,elapsed_s=time.time()-t0)
        save_policy(p,cp,rec['path'],dict(stage='offline_stage2_wide_v2_ppo_forced',source_policy=a.policy,anchor_ref=a.anchor_ref,critic=a.critic,**{k:v for k,v in rec.items() if k!='path'}))
        append_jsonl(out/'forced.jsonl',rec);print('FORCED',json.dumps(rec),flush=True)
    if since>=a.patience and not a.no_patience:print('PATIENCE_STOP',flush=True);break
    if step%10==0:write_json(out/'status.json',dict(status='running',total_steps=a.steps,accepted=n_acc,evals_since_accept=since,candidates=len(cands),updated=cst(),last_train=row,last_ope=hist[-1] if hist else None))
save_candidate(step)
assert all(torch.equal(v,enc_before[k]) for k,v in p.obs_encoder.state_dict().items()),'encoder changed'
rep=dict(status='done',updated=cst(),steps_run=step,accepted=n_acc,candidates=cands,input_deviation=dev0,stop_reason='patience' if since>=a.patience else 'max_steps',ope_history_tail=hist[-3:])
write_json(out/'status.json',rep);write_json(out/'report.json',rep);print('PPO_DONE',json.dumps(rep),flush=True)
