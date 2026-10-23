# [official-align] Dynamics 7/5 ensemble DEFERRED (reference-only; cost/benefit low until AM-Q gates promote).
# [offline_multisku_v1_draft] NEW file, multi-SKU copy of offline_stage2_v3_nominal_20261017/transition_model_ic7.py (STEP 2: transition model,
#   REFERENCE ONLY - feeds the logged AM-Q number, never a decision). Changes: input += raw sku_cond (4); same data as the critic
#   (demos + all success/fail rollouts); MS_TRAIN_ALL; --batch default 4096 (base 1024).
from ms_common import *
import argparse
class TM(nn.Module):
    def __init__(s,h=1024):
        super().__init__();s.f=mlp(FEAT+ACT_DIM+1+SKU_DIM,h,h,2);s.dz=nn.Linear(h,FEAT);s.succ=nn.Linear(h,1)
    def forward(s,zn,aj,clock,sk):
        h=torch.relu(s.f(torch.cat([zn,aj,clock,sk],1)));return zn+s.dz(h),s.succ(h).squeeze(-1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True);ap.add_argument('--critic',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=20000);ap.add_argument('--batch',type=int,default=4096);ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--eval-every',type=int,default=2000)
    a=ap.parse_args();setup(77);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    D=load_dataset_w(json.loads(Path(a.sources).read_text()), feat_context='TM')
    stats={k:v.cuda() for k,v in torch.load(a.critic,map_location='cuda',weights_only=False)['stats'].items()}  # same normalization as critic
    T=TM().cuda();opt=torch.optim.AdamW(T.parameters(),lr=a.lr,weight_decay=1e-5)
    nz=lambda z:(z-stats['mu'])/stats['sd']
    tr=torch.nonzero(train_mask(D)).flatten();ho=torch.nonzero(D['heldout']).flatten();g=torch.Generator(device='cuda').manual_seed(5)
    def batch(E,n):
        e=E[torch.randint(len(E),(n,),device='cuda',generator=g)];t=torch.randint(CHUNKS,(n,),device='cuda',generator=g)
        z=nz(D['F'][e,t]);z1=nz(D['F'][e,t+1]);aj=D['AJ'][e,t];ck=(t.float()/CHUNKS*2-1)[:,None];term=(t==CHUNKS-1);y=D['success'][e]
        return z,aj,ck,D['SK'][e],z1,term,y
    def loss(bt):
        z,aj,ck,sk,z1,term,y=bt;pz,ps=T(z,aj,ck,sk);lz=((pz-z1)**2).mean()
        ls=nn.functional.binary_cross_entropy_with_logits(ps[term],y[term]) if term.any() else pz.sum()*0
        return lz,ls,((z-z1)**2).mean()
    t0=time.time();row={}
    for step in range(1,a.steps+1):
        lz,ls,_=loss(batch(tr,a.batch));opt.zero_grad();(lz+0.1*ls).backward();torch.nn.utils.clip_grad_norm_(T.parameters(),1.);opt.step()
        if step%a.eval_every==0 or step==a.steps:
            with torch.no_grad():hz,hs,base=loss(batch(ho,4096))
            row=dict(step=step,train_mse=lz.item(),held_mse=hz.item(),held_copy_baseline_mse=base.item(),held_succ_bce=hs.item(),heldout_in_sample=TRAIN_ALL,elapsed_s=time.time()-t0)
            append_jsonl(out/'metrics.jsonl',row);write_json(out/'status.json',dict(status='training',updated=cst(),**row));print(json.dumps(row),flush=True)
    torch.save(dict(T=T.state_dict(),stats={k:v.cpu() for k,v in stats.items()},args=vars(a),final=row),out/'transition.pt')
    write_json(out/'status.json',dict(status='done',updated=cst(),**row));print('TM_DONE',json.dumps(row),flush=True)
if __name__=='__main__':main()
