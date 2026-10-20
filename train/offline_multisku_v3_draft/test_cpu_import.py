# [offline_multisku_v1_draft] CPU-ONLY smoke test (no GPU, no env rollout, no training). Run with CUDA_VISIBLE_DEVICES="".
# Checks: all modules import; start checkpoint builds a multi-SKU BCPolicy on CPU (strict load); encoder output dim == FEAT and the
# last 4 dims are exactly the normalized sku_cond (policy consumes SKU); critic/TM dims incl. sku; DDIM kernel terminal sigma 0 and
# std clamp; PPO step-1 log-ratio == 0 with pi_old == model; dataset loader on a tiny synthetic demo+rollout set (CPU tensors).
import os,sys,json,tempfile
assert os.environ.get('CUDA_VISIBLE_DEVICES','x')=='','run with CUDA_VISIBLE_DEVICES=""'
os.environ.setdefault('S2_FIX_FINAL','1');os.environ.setdefault('S2_FLOOR_FINAL','0');os.environ.setdefault('S2_STD_MAX','0.1')
tmp=tempfile.mkdtemp(prefix='_cpu_test_',dir=os.path.dirname(os.path.abspath(__file__)));os.environ['S2_OUT']=tmp
import numpy as np,torch
import ms_common as M
import tm_ms
assert not torch.cuda.is_available()
from bc_model_multisku import BCPolicy
cp=torch.load(str(M.START_POLICY),map_location='cpu',weights_only=False)
p=BCPolicy(dict(agent_pos=np.zeros((2,28),np.float32),action=np.zeros((2,14),np.float32),sku_cond=np.zeros((2,4),np.float32)),pretrained=False)
p.load_state_dict(cp['ema'],strict=True);p.eval();tab=cp['config']['sku_cond_table'];assert sorted(tab)==M.SKUS
print('encoder low_dim keys',p.obs_encoder.low_dim_keys,'rgb keys',p.obs_encoder.rgb_keys)
with torch.no_grad():
    outs={}
    for s in M.SKUS:
        obs={c:torch.rand(2,3,240,320) for c in M.CAMS};obs['agent_pos']=torch.zeros(2,28);obs['sku_cond']=torch.tensor([tab[s]]*2)
        torch.manual_seed(0);obs={c:(torch.rand(2,3,240,320) if c in M.CAMS else v) for c,v in obs.items()}
        f=p.obs_encoder(p.normalizer.normalize(obs));outs[s]=f
    assert f.shape[-1]==M.FEAT,f.shape
    for s in M.SKUS:
        nk=p.normalizer['sku_cond'].normalize(torch.tensor([tab[s]]*2));assert torch.allclose(outs[s][:,-4:],nk,atol=1e-5),(s,outs[s][:,-4:],nk)
    diff=(outs['SKU5A']-outs['SKU5B']).abs();assert diff[:,:-4].max()<1e-5 and diff[:,-4:].max()>1e-3
    print('FEAT',f.shape[-1],'sku dims = last 4 (normalized sku_cond) OK; global_cond_dim',2*M.FEAT)
    # kernel / chain
    sc=p.noise_scheduler;sc.set_timesteps(10);cond=torch.cat([outs['SKU5C'][0:1],outs['SKU5C'][1:2]],1).repeat(3,1)
    x,xs,ys,lps=M.sample_chain(p.model,p,cond,torch.Generator().manual_seed(1),stochastic=True,trace=True)
    for i,t in enumerate(sc.timesteps):
        mean,std=M.kernel(p,p.model(xs[i],t,global_cond=cond),t,xs[i])
        if M.is_terminal(p,t):assert float(std.max())==0.
        else:assert M.STD_MIN-1e-6<=float(std.min()) and float(std.max())<=M.STD_MAX+1e-6
        if not M.is_terminal(p,t):
            lr=(M.logprob(mean,std,ys[i])-lps[i]).abs().max().item();assert lr<1e-4,lr
    print('kernel: terminal sigma 0, clamp ok, step-1 log-ratio 0 ok; stochastic steps',sum(not M.is_terminal(p,t) for t in sc.timesteps))
# critic / TM dims and synthetic dataset
from pathlib import Path
d=Path(tmp)/'data';d.mkdir(parents=True)
for split,n in (('train',5),('validation',5)):
    np.savez(d/f'demo_{split}.npz',feats=np.random.randn(n,61,M.FEAT).astype(np.float32),actions=np.random.randn(n,60,8,14).astype(np.float32)*.3,success=np.ones(n,bool),
             ids=np.arange(n),sku=np.array(M.SKUS[:n]),sku_cond=np.array([tab[s] for s in M.SKUS[:n]],np.float32))
rd=Path(tmp)/'round0'/'collect'/'eps';rd.mkdir(parents=True)
for i in range(10):
    s=M.SKUS[i%5];meta=dict(seed=2100000+i,sku=s,sku_cond=tab[s],success=bool(i%2),heldout=(i//5)%10==0)
    np.savez(rd/f'ep_{2100000+i}.npz',meta=np.array(json.dumps(meta)),feats=np.random.randn(61,M.FEAT).astype(np.float32),actions=np.random.randn(60,8,14).astype(np.float32)*.3)
D=M.load_dataset_w([['demo',str(d)],['round0',str(rd)]],device='cpu');st=M.feat_stats(D)
e=torch.arange(D['n']);t=torch.full((D['n'],),5);s=M.critic_state(D,st,e,t);assert s.shape==(D['n'],M.SD)
C=M.Critic(M.SD,M.ACT_DIM);q=C.qmin(s,D['AJ'][e,t]);assert q.shape==(D['n'],)
T=tm_ms.TM();z,_=T(s[:,M.FEAT:2*M.FEAT],D['AJ'][e,t],s[:,2*M.FEAT:2*M.FEAT+1],D['SK'][e]);assert z.shape==(D['n'],M.FEAT)
print('dataset n',D['n'],'train_all',M.TRAIN_ALL,'train',int(M.train_mask(D).sum()),'heldout',int(D['heldout'].sum()),'critic SD',M.SD,'OK')
import shutil;shutil.rmtree(tmp);print('CPU_TEST_PASS')
