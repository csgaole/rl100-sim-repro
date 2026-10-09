"""[rl100-sim-repro copy of stage_gate.py] CHANGE vs historical run: development eval uses --sampler upstream (DDIM10, eta=1),
the sampler under which bc_epoch252 scored 98/100. The historical run used the legacy default (eta=0) and scored 0/100 at every gate.
Development-only selection and fixed early stopping protocol."""
import json,subprocess,sys,shutil,hashlib
from pathlib import Path
def update_progress(state,successes,loss):
    if not state:return dict(success_reference=successes,loss_reference=loss,stale=0),True,True
    s=dict(state);success_improved=successes>=s['success_reference']+2;loss_improved=loss<=s['loss_reference']*.99
    if success_improved:s['success_reference']=successes
    if loss_improved:s['loss_reference']=loss
    s['stale']=0 if success_improved or loss_improved else s['stale']+1
    return s,success_improved,loss_improved

def run_stage(out,epoch):
    here=Path(__file__).resolve().parent;out=Path(out);folder=out/'development';folder.mkdir(exist_ok=True)
    frozen=folder/f'candidate_through_{epoch:04d}.pt';result=folder/f'eval_{epoch:04d}.json'
    if not frozen.exists():shutil.copy2(out/'best.pt',frozen)
    sha=hashlib.sha256(frozen.read_bytes()).hexdigest()
    if not result.exists():
        with open(folder/f'eval_{epoch:04d}.log','a') as log:
            subprocess.run([sys.executable,str(here/'evaluate_bc.py'),'--checkpoint',str(frozen),'--output',str(result),'--seed-start','880000','--episodes','100','--sampler','upstream'],stdout=log,stderr=subprocess.STDOUT,check=True)
    r=json.loads(result.read_text());assert r['checkpoint_sha256']==sha and r['episodes']==100 and r['seed_start']==880000 and r['sampler']=='upstream'
    path=folder/'decision.json';old=json.loads(path.read_text()) if path.exists() else {}
    if old.get('epoch')==epoch:return old
    progress,si,li=update_progress(old.get('progress'),r['successes'],r['validation_loss'])
    selected=old.get('selected')
    if selected is None or (r['successes'],-r['validation_loss'])>(selected['successes'],-selected['validation_loss']):
        selected=dict(r,stage_epoch=epoch)
        shutil.copy2(frozen,out/'selected_development.tmp');(out/'selected_development.tmp').replace(out/'selected_development.pt')
    decision=dict(epoch=epoch,progress=progress,success_improved=si,loss_improved=li,
        stop=progress['stale']>=3,selected=selected,current=r,protocol='development_v2_upstream_sampler')
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(decision,indent=2));tmp.replace(path)
    (folder/f'decision_{epoch:04d}.json').write_text(json.dumps(decision,indent=2))
    return decision
