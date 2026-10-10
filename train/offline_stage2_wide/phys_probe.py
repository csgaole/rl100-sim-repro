"""Physical-state probe, step 2: does privileged sim state predict eventual success earlier/better than the frozen
BC features? Episode-level split (held-out = seed % 5 == 0), wide-range rollouts (wide_round1/2 Stage-1, v2_round1
round-2 cand). Feature sets at chunk boundary t: (a) BC [z_{t-1},z_t] (t=0: z_0), (b) physical state at t,
(c) init perturbation params, (e) physical + init, (d) BC + physical. Models: L2 logistic (wd by 5-fold CV) and
small MLP (2x256, early stop). Metrics on held-out: AUC + bootstrap 95% CI; within-cell (16 init cells) AUC; AUC
within quintile bins of the init-only model's predicted difficulty; success vs wrong_face; det-sampler subset;
feature importance (standardised logistic coefs, grouped permutation importance)."""
import os,sys,json,time,argparse
from pathlib import Path
import numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
ap=argparse.ArgumentParser();ap.add_argument('--phys',required=True);ap.add_argument('--out',required=True);ap.add_argument('--ts',default='0,5,10,15,20,30,40')
ap.add_argument('--boot',type=int,default=1000);ap.add_argument('--wds',default='1e-4,1e-3,1e-2,1e-1,1');a=ap.parse_args()
torch.manual_seed(0);np.random.seed(0);dev='cuda';torch.backends.cuda.matmul.allow_tf32=False
t00=time.time();Z=np.load(a.phys,allow_pickle=False);rows=json.loads(str(Z['rows']));PH=Z['phys'];NAMES=list(Z['names'])
assert Z['match'].all(),'replay mismatch'
keep=[i for i,r in enumerate(rows) if r['wide'] and not r['nominal']];rows=[rows[i] for i in keep];PH=PH[keep]
TS=[int(x) for x in a.ts.split(',')];need=sorted({max(t-1,0) for t in TS}|set(TS))
BC={t:np.zeros((len(rows),1564),np.float32) for t in need}
for i,r in enumerate(rows):
    f=np.load(r['file'])['feats']
    for t in need:BC[t][i]=f[t]
print('loaded',len(rows),f'{time.time()-t00:.0f}s',flush=True)
y=np.array([bool(r['success']) for r in rows],np.float32);reason=np.array([r['reason'] for r in rows])
seed=np.array([r['seed'] for r in rows]);he=seed%5==0;tr=~he;det=np.array([r['sampler']=='det' for r in rows])
cell=np.array([r['cell'] for r in rows]);src=np.array([r['source'] for r in rows])
dx=np.array([r['dx'] for r in rows]);dy=np.array([r['dy'] for r in rows]);yw=np.array([r['yaw_deg'] for r in rows])
INIT=np.stack([dx,dy,yw,abs(dx),abs(dy),abs(yw),(dy>0).astype(float),np.hypot(dx,dy),dx*dy,dx*yw,dy*yw,np.array([r['pos_half_mm'] for r in rows]),np.array([r['yaw_half_deg'] for r in rows])],1).astype(np.float32)
INIT_N=['dx','dy','yaw','|dx|','|dy|','|yaw|','dy>0','r','dx*dy','dx*yaw','dy*yaw','pos_half','yaw_half']
def aucf(s,l):
    l=np.asarray(l,bool);n1=l.sum();n0=len(l)-n1
    if n1==0 or n0==0:return float('nan')
    u,inv=np.unique(np.asarray(s),return_inverse=True);cnt=np.bincount(inv);start=np.cumsum(cnt)-cnt;rk=(start+(cnt+1)/2.)[inv]
    return float((rk[l].sum()-n1*(n1+1)/2)/(n1*n0))
def strat_auc(s,l,g):
    num=den=0.
    for c in np.unique(g):
        m=g==c;l1=l[m].astype(bool);n1=l1.sum();n0=m.sum()-n1
        if n1==0 or n0==0:continue
        num+=aucf(s[m],l1)*n1*n0;den+=n1*n0
    return num/den if den else float('nan')
rng=np.random.default_rng(0)
def boot(fn,idx_n):
    v=[fn(rng.integers(0,idx_n,idx_n)) for _ in range(a.boot)];return [float(np.nanpercentile(v,2.5)),float(np.nanpercentile(v,97.5))]
def standardize(Xtr,Xhe):
    mu=Xtr.mean(0);sd=Xtr.std(0);k=sd>1e-6;return (Xtr[:,k]-mu[k])/sd[k],(Xhe[:,k]-mu[k])/sd[k],k
def fit_lr(X,Y,wd,steps=300):
    w=torch.zeros(X.shape[1],device=dev,requires_grad=True);b=torch.zeros(1,device=dev,requires_grad=True)
    opt=torch.optim.LBFGS([w,b],max_iter=steps,line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad();l=Fn.binary_cross_entropy_with_logits(X@w+b,Y)+wd*w.square().sum();l.backward();return l
    opt.step(cl);return w.detach(),b.detach()
def logistic(Xtr,Ytr,Xhe):
    X=torch.tensor(Xtr,device=dev);Y=torch.tensor(Ytr,device=dev);H=torch.tensor(Xhe,device=dev)
    perm=torch.randperm(len(X),generator=torch.Generator().manual_seed(0));folds=perm.chunk(5);cv={}
    for wd in [float(x) for x in a.wds.split(',')]:
        sc=[]
        for f in folds:
            m=torch.ones(len(X),dtype=torch.bool);m[f]=False;w,b=fit_lr(X[m.to(dev)],Y[m.to(dev)],wd);sc.append(aucf((X[f.to(dev)]@w+b).cpu().numpy(),Ytr[f.numpy()]))
        cv[wd]=float(np.mean(sc))
    wd=max(cv,key=cv.get);w,b=fit_lr(X,Y,wd);return (H@w+b).cpu().numpy(),dict(wd=wd,cv=cv,w=w.cpu().numpy(),b=float(b))
def mlp(Xtr,Ytr,Xhe,seed=0):
    g=torch.Generator().manual_seed(seed);n=len(Xtr);p=torch.randperm(n,generator=g).numpy();nv=int(.15*n);va,tt=p[:nv],p[nv:]
    X=torch.tensor(Xtr,device=dev);Y=torch.tensor(Ytr,device=dev);H=torch.tensor(Xhe,device=dev);torch.manual_seed(seed)
    net=nn.Sequential(nn.Dropout(.1),nn.Linear(X.shape[1],256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,1)).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-2);best=(-1,None);bad=0
    for ep in range(300):
        net.train();q=tt[torch.randperm(len(tt),generator=g).numpy()]
        for i in range(0,len(q),256):
            j=torch.tensor(q[i:i+256],device=dev);l=Fn.binary_cross_entropy_with_logits(net(X[j]).squeeze(-1),Y[j]);opt.zero_grad();l.backward();opt.step()
        net.eval()
        with torch.no_grad():s=aucf(net(X[torch.tensor(va,device=dev)]).squeeze(-1).cpu().numpy(),Ytr[va])
        if s>best[0]+1e-4:best=(s,{k:v.clone() for k,v in net.state_dict().items()});bad=0
        else:
            bad+=1
            if bad>=25:break
    net.load_state_dict(best[1]);net.eval()
    with torch.no_grad():return net(H).squeeze(-1).cpu().numpy(),dict(val_auc=best[0],epochs=ep+1),net
def evaluate(s,lab=None,mask=None):
    lab=y[he] if lab is None else lab;m=np.ones(len(s),bool) if mask is None else mask;s=s[m];l=lab[m];c=cell[he][m];d=det[he][m];db=dbin[m]
    n=len(s);return dict(auc=aucf(s,l),ci=boot(lambda i:aucf(s[i],l[i]),n),within_cell=strat_auc(s,l,c),within_cell_ci=boot(lambda i:strat_auc(s[i],l[i],c[i]),n),
        within_difficulty_quintile=strat_auc(s,l,db),within_difficulty_ci=boot(lambda i:strat_auc(s[i],l[i],db[i]),n),within_difficulty_decile=strat_auc(s,l,dbin10[m]),det_only=aucf(s[d],l[d]),n=int(n),pos=int(l.sum()))
R=dict(n_wide=len(rows),n_train=int(tr.sum()),n_heldout=int(he.sum()),success_rate=float(y.mean()),reasons={k:int((reason==k).sum()) for k in np.unique(reason)},
       sources={k:int((src==k).sum()) for k in np.unique(src)},det_frac=float(det.mean()),results={})
# (c) init-only model -> difficulty bins (quintiles of held-out predicted prob)
Xi_tr,Xi_he,_=standardize(INIT[tr],INIT[he]);s_init,info_init=logistic(Xi_tr,y[tr],Xi_he)
s_init_mlp,_,_=mlp(Xi_tr,y[tr],Xi_he)
# difficulty bins = quintiles of the STRONGER init-only model (MLP) held-out score; deciles also reported
dbin=np.digitize(s_init_mlp,np.quantile(s_init_mlp,[.2,.4,.6,.8]));dbin10=np.digitize(s_init_mlp,np.quantile(s_init_mlp,np.arange(.1,1,.1)))
PRED={'init_lr':s_init,'init_mlp':s_init_mlp}
R['init']=dict(logistic=evaluate(s_init),mlp=evaluate(s_init_mlp),coef={n:float(c) for n,c in zip(INIT_N,info_init['w'])})
print('INIT',json.dumps({k:round(v['auc'],3) for k,v in R['init'].items() if isinstance(v,dict) and 'auc' in v}),flush=True)
GROUPS={'parcel_pose':lambda n:n.startswith('parcel_pos') or n.startswith('parcel_R') or n in('label_up','bottom'),'parcel_vel':lambda n:'vel_' in n and n.startswith('parcel'),
        'arm_q':lambda n:n.startswith('q_'),'arm_qvel':lambda n:n.startswith('qvel_'),'tcp_pose':lambda n:n.startswith('tcp') and 'minus' not in n,'tcp_rel_parcel':lambda n:'minus_parcel' in n,
        'contacts':lambda n:n.startswith('cforce') or n.startswith('ccount') or n.startswith('outfeed'),'running_aggr':lambda n:n.startswith('run_')}
for t in TS:
    T=dict();P=PH[:,t];Bc=BC[t] if t==0 else np.concatenate([BC[t-1],BC[t]],1)
    sets={'a_BC':Bc,'b_phys':P,'c_init':INIT,'e_phys+init':np.concatenate([P,INIT],1),'d_BC+phys':np.concatenate([Bc,P],1)}
    for nm,X in sets.items():
        Xtr,Xhe,k=standardize(X[tr].astype(np.float32),X[he].astype(np.float32))
        sl,inf=logistic(Xtr,y[tr],Xhe);sm,minf,net=mlp(Xtr,y[tr],Xhe)
        PRED[f't{t}_{nm}_lr']=sl;PRED[f't{t}_{nm}_mlp']=sm
        T[nm]=dict(dim=int(k.sum()),logistic=evaluate(sl),mlp=evaluate(sm),wd=inf['wd'],mlp_val_auc=minf['val_auc'])
        if nm=='b_phys':
            pn=[n for n,kk in zip(NAMES,k) if kk];w=inf['w'];o=np.argsort(-abs(w))[:12];T[nm]['top_coef']=[(pn[i],round(float(w[i]),3)) for i in o]
            base=aucf(sm,y[he]);H=torch.tensor(Xhe,device=dev);imp={}
            for gname,fn in GROUPS.items():
                cols=[i for i,n in enumerate(pn) if fn(n)]
                if not cols:continue
                drops=[]
                for rep in range(5):
                    Hp=H.clone();pi=torch.randperm(len(H),generator=torch.Generator().manual_seed(rep)).to(dev);Hp[:,cols]=H[pi][:,cols]
                    with torch.no_grad():drops.append(base-aucf(net(Hp).squeeze(-1).cpu().numpy(),y[he]))
                imp[gname]=round(float(np.mean(drops)),4)
            T[nm]['perm_importance_mlp']=dict(sorted(imp.items(),key=lambda x:-x[1]))
            # success vs wrong_face only (drops excluded)
            mtr=tr&np.isin(reason,['success','wrong_face']);mhe=he&np.isin(reason,['success','wrong_face'])
            Xa,Xb,_=standardize(P[mtr].astype(np.float32),P[mhe].astype(np.float32));s2,_=logistic(Xa,y[mtr],Xb);s3,_,_=mlp(Xa,y[mtr],Xb)
            l2=y[mhe];c2=cell[mhe];n2=len(s2)
            T[nm]['succ_vs_wrongface']=dict(logistic=aucf(s2,l2),logistic_ci=boot(lambda i:aucf(s2[i],l2[i]),n2),mlp=aucf(s3,l2),mlp_ci=boot(lambda i:aucf(s3[i],l2[i]),n2),
                within_cell_logistic=strat_auc(s2,l2,c2),within_cell_mlp=strat_auc(s3,l2,c2),n=int(n2),wrong_face=int((l2==0).sum()))
        print(f't{t} {nm}: LR {T[nm]["logistic"]["auc"]:.3f} {T[nm]["logistic"]["ci"]} cell {T[nm]["logistic"]["within_cell"]:.3f} diff {T[nm]["logistic"]["within_difficulty_quintile"]:.3f} | MLP {T[nm]["mlp"]["auc"]:.3f} cell {T[nm]["mlp"]["within_cell"]:.3f} diff {T[nm]["mlp"]["within_difficulty_quintile"]:.3f} ({time.time()-t00:.0f}s)',flush=True)
    R['results'][f't{t}']=T
    Path(a.out).write_text(json.dumps(R,indent=1,default=float));np.savez(Path(a.out).with_suffix('.preds.npz'),y=y[he],cell=cell[he],det=det[he],reason=reason[he],seed=seed[he],src=src[he],**PRED)
print('PROBE_DONE',f'{time.time()-t00:.0f}s',flush=True)
