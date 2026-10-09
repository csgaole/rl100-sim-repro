"""Wait for Stage-1 training to finish, then evaluate selected_development.pt on 100 fresh conditions
(910000-910099, upstream DDIM10 eta=1; same protocol as bc_epoch252's 98/100) and write final_eval.json.
Never touches reserved final seeds 900000-900999."""
import json,time,subprocess,sys,os
from pathlib import Path
here=Path(__file__).resolve().parent;OUT=here.parents[1]/'runs/bc_stage1'
pid=int((OUT/'train.pid').read_text())
def alive(p):
    try:os.kill(p,0);return True
    except OSError:return False
while alive(pid):time.sleep(120)
st=json.loads((OUT/'status.json').read_text())
if st.get('status')!='complete' or not (OUT/'selected_development.pt').exists():
    (OUT/'finalize_status.json').write_text(json.dumps(dict(status='skipped',reason='training not complete',train_status=st),indent=2));sys.exit(1)
dest=OUT/'final_eval_fresh910000.json'
with open(OUT/'final_eval.log','w') as log:
    subprocess.run([sys.executable,'-u',str(here/'evaluate_bc.py'),'--checkpoint',str(OUT/'selected_development.pt'),'--output',str(dest),'--seed-start','910000','--episodes','100','--sampler','upstream'],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=here)
r=json.loads(dest.read_text())
with open(OUT/'REPORT.md','a') as f:
    f.write(f"\n\n## 自动追加：训练完成后的最终闭环评估（finalize.py，{time.strftime('%Y-%m-%d %H:%M %Z')}）\n\n"
            f"- 训练状态：{st.get('completion_reason')}，完成 epoch {st.get('epochs_completed')}，step {st.get('step')}\n"
            f"- 选中检查点：runs/bc_stage1/selected_development.pt（epoch {r['checkpoint_epoch']}，val bc_loss {r['validation_loss']}）\n"
            f"- 新鲜条件 910000–910099，upstream DDIM10 eta=1：**{r['successes']}/{r['episodes']}**，Wilson95 {r['wilson95']}\n"
            f"- 失败分类：{r['failure_counts']}；对照 bc_epoch252 同协议 98/100\n")
(OUT/'finalize_status.json').write_text(json.dumps(dict(status='done',result=r),indent=2))
