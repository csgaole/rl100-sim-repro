"""[offline_stage2_v3] Encode the P3 v3 training demos with the FROZEN v3 (no-VIB) encoder -> OUT/data/demo_{train,validation}.npz.
Demos = sku09 expanded_v1 cache (old300+expanded900: train 960 / validation 120) + wide_v1 cache (train 715 / validation 89), all READ-ONLY.
Test splits are never read. actions = normalized actions/effective with the policy normalizer (BC label). All demos are successes."""
from wcommon import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument("--policy",default=str(STAGE1));ap.add_argument("--limit",type=int,default=0);a=ap.parse_args()
setup(0);p,cp=load_policy(a.policy);p.eval()
SKU=Path("/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/data")
CACHES=[("expanded_v1",SKU/"expanded_v1/cache"),("wide_v1",SKU/"p3_cache_wide_v1")]
(OUT/"data").mkdir(parents=True,exist_ok=True)
for split in ["train","validation"]:
    Fs=[];As=[];ids=[]
    for name,C in CACHES:
        info=json.loads((C/f"{split}_complete.json").read_text());cid=info["ids"][:a.limit] if a.limit else info["ids"]
        rgb=np.load(C/f"{split}_rgb.npy",mmap_mode="r");state=np.load(C/f"{split}_state.npy");action=np.load(C/f"{split}_action.npy")
        F=np.zeros((len(cid),61,FEAT),np.float32)
        for e in range(len(cid)):
            F[e]=encode_frames(p,{**{c:np.asarray(rgb[e,:,k]) for k,c in enumerate(CAMS)},"proprio":state[e]}).cpu().numpy()
            if e%100==0:print("DEMO",split,name,e,len(cid),flush=True)
        with torch.no_grad():A=p.normalizer["action"].normalize(torch.from_numpy(action[:len(cid)]).cuda()).cpu().numpy()
        Fs.append(F);As.append(A.astype(np.float32));ids+=[f"{name}:{i}" for i in cid]
    F=np.concatenate(Fs);A=np.concatenate(As);succ=np.ones(len(ids),bool)
    tmp=OUT/"data"/f"demo_{split}.tmp.npz";np.savez(tmp,feats=F,actions=A,success=succ,ids=np.array(ids));tmp.replace(OUT/"data"/f"demo_{split}.npz")
    print("DEMO_DONE",split,len(ids),F.shape,flush=True)
