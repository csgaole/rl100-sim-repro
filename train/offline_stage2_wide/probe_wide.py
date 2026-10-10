"""Linear-probe reference for early discriminability (critic-v2 ablation): L2 logistic regression on the
z-scored frozen BC feature [z_{t-1}, z_t] at chunk t (t=0: z_0 twice -> use z_0 only) -> eventual success.
Train = same training episodes as the critic (all sources), weight decay picked by 5-fold CV AUC on train,
reported on the same held-out episodes (wide subset and all rollouts)."""
from wcommon import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True);ap.add_argument('--out',required=True);ap.add_argument('--ts',default='0,10,20');ap.add_argument('--wds',default='1e-4,1e-3,1e-2,1e-1,1');a=ap.parse_args()
setup(0);sources=json.loads(Path(a.sources).read_text());D=load_dataset_w(sources);stats=feat_stats(D)
tr=torch.nonzero(~D['heldout']).flatten();he=torch.nonzero(D['heldout']).flatten();y=D['success']
ro=torch.tensor([s!='demo' for s in D['source']],device='cuda')
def feats(e,t):
    tt=torch.full_like(e,t);s=critic_state(D,stats,e,tt)[:,:-1]
    return s[:,FEAT:] if t==0 else s
def fit(X,Y,wd,steps=300):
    w=torch.zeros(X.shape[1],device='cuda',requires_grad=True);b=torch.zeros(1,device='cuda',requires_grad=True)
    opt=torch.optim.LBFGS([w,b],max_iter=steps,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();l=torch.nn.functional.binary_cross_entropy_with_logits(X@w+b,Y)+wd*w.square().sum();l.backward();return l
    opt.step(closure);return w.detach(),b.detach()
res={}
for t in [int(x) for x in a.ts.split(',')]:
    X=feats(tr,t);Y=y[tr];perm=torch.randperm(len(tr),generator=torch.Generator().manual_seed(0)).cuda();folds=perm.chunk(5);cv={}
    for wd in [float(x) for x in a.wds.split(',')]:
        sc=[]
        for f in folds:
            m=torch.ones(len(tr),dtype=torch.bool,device='cuda');m[f]=False;w,b=fit(X[m],Y[m],wd);sc.append(auc((X[f]@w+b).cpu().numpy(),Y[f].cpu().numpy()))
        cv[wd]=float(np.mean(sc))
    best=max(cv,key=cv.get);w,b=fit(X,Y,best);Xh=feats(he,t);p=(Xh@w+b).cpu().numpy();lab=y[he].cpu().numpy()
    wm=D['wide'][he].cpu().numpy();rm=ro[he].cpu().numpy()
    res[f't{t}']=dict(cv_auc=cv,wd=best,heldout_auc_wide=auc(p[wm],lab[wm]),heldout_auc_rollout=auc(p[rm],lab[rm]),train_auc=auc((X@w+b).cpu().numpy(),Y.cpu().numpy()))
    print(t,json.dumps(res[f't{t}']),flush=True)
write_json(Path(a.out),res);print('PROBE_DONE')
