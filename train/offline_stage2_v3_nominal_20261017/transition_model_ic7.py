# [ic7t] Transition model T(z_t, a_t, clock) -> z_{t+1} (residual, normalized feature space) + terminal-success head.
# Trained on the same data as the IQL critic (step 1 of RL-100 Alg.1). Used only for AM-Q OPE screening.
from wcommon_fix import *
import argparse
class TM(nn.Module):
    def __init__(s,h=1024):
        super().__init__();s.f=mlp(FEAT+96+1,h,h,2);s.dz=nn.Linear(h,FEAT);s.succ=nn.Linear(h,1)
    def forward(s,zn,aj,clock):
        h=torch.relu(s.f(torch.cat([zn,aj,clock],1)));return zn+s.dz(h),s.succ(h).squeeze(-1)
ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True);ap.add_argument('--critic',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--steps',type=int,default=20000);ap.add_argument('--batch',type=int,default=1024);ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--eval-every',type=int,default=2000)
def main():
    a=ap.parse_args()    ;setup(77);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    D=load_dataset_w(json.loads(Path(a.sources).read_text()))
    stats={k:v.cuda() for k,v in torch.load(a.critic,map_location='cuda',weights_only=False)['stats'].items()}  # same normalization as critic
    T=TM().cuda();opt=torch.optim.AdamW(T.parameters(),lr=a.lr,weight_decay=1e-5)
    nz=lambda z:(z-stats['mu'])/stats['sd']
    tr=torch.nonzero(~D['heldout']).flatten();ho=torch.nonzero(D['heldout']).flatten();g=torch.Generator(device='cuda').manual_seed(5)
    def batch(E,n):
        e=E[torch.randint(len(E),(n,),device='cuda',generator=g)];t=torch.randint(CHUNKS,(n,),device='cuda',generator=g)
        z=nz(D['F'][e,t]);z1=nz(D['F'][e,t+1]);aj=D['AJ'][e,t];ck=(t.float()/CHUNKS*2-1)[:,None];term=(t==CHUNKS-1);y=D['success'][e]
        return z,aj,ck,z1,term,y
    def loss(bt):
        z,aj,ck,z1,term,y=bt;pz,ps=T(z,aj,ck);lz=((pz-z1)**2).mean()
        ls=nn.functional.binary_cross_entropy_with_logits(ps[term],y[term]) if term.any() else pz.sum()*0
        return lz,ls,((z-z1)**2).mean()
    t0=time.time()
    for step in range(1,a.steps+1):
        lz,ls,_=loss(batch(tr,a.batch));opt.zero_grad();(lz+0.1*ls).backward();torch.nn.utils.clip_grad_norm_(T.parameters(),1.);opt.step()
        if step%a.eval_every==0 or step==a.steps:
            with torch.no_grad():hz,hs,base=loss(batch(ho,4096))
            row=dict(step=step,train_mse=lz.item(),held_mse=hz.item(),held_copy_baseline_mse=base.item(),held_succ_bce=hs.item(),elapsed_s=time.time()-t0)
            append_jsonl(out/'metrics.jsonl',row);write_json(out/'status.json',dict(status='training',updated=cst(),**row));print(json.dumps(row),flush=True)
    torch.save(dict(T=T.state_dict(),stats={k:v.cpu() for k,v in stats.items()},args=vars(a),final=row),out/'transition.pt')
    write_json(out/'status.json',dict(status='done',updated=cst(),**row));print('TM_DONE',json.dumps(row),flush=True)

if __name__=='__main__':main()
