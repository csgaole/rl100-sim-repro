import re,sys
from pathlib import Path
src=Path('/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/src/train_bc_v3_novib.py').read_text()
def rep(a,b,cnt=1):
    global src
    assert src.count(a)==cnt,(a,src.count(a));src=src.replace(a,b)
src='"""[NOMINAL 2026-10-17 copy of sku09 src/train_bc_v3_novib.py] data restricted to original300 (expanded_v1 ids<300 = first 240 train / 30 val; identical to rl100-sim-repro data/demos/sku09-original300 splits), NO wide, NO expanded cells; nominal p5_y2 dev gate on 880000-880099 (nominal_gate.py). Loss/model/optimizer/EMA unchanged; --seed/--lr added."""\n'+src
rep("import os\n","import os,sys\nsys.path.insert(0,'/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/src')\nsys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))\n")
rep("import p3_dev_gate_v3 as p3_dev_gate","import nominal_gate")
rep("RUN_FAMILY='p3_v3_novib_absolute'","RUN_FAMILY='p3_v3_novib_absolute'  # kept for loader compatibility; config.data_subset marks nominal")
rep("def subset_groups(val,train,cells,smoke):\n    g={}","def subset_groups(val,train,cells,smoke):\n    rng=np.random.default_rng(7);g={'val_old300':list(range(len(val['ids']))),'train_old300_10ep':sorted(rng.choice(len(train['ids']),10,replace=False).tolist())}\n    idx={k:np.concatenate([np.arange(e*60,e*60+60) for e in v]) for k,v in g.items()}\n    if smoke:idx={k:v[:8] for k,v in idx.items()}\n    return idx\ndef _old_subset_groups(val,train,cells,smoke):\n    g={}")
rep("    ap.add_argument('--eval-gpus',default='1',help='physical GPU indices for dev gates (from {0,1,5,6,7})')","    ap.add_argument('--eval-gpus',default='',help='physical GPU for nominal gate (own GPU)');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--lr',type=float,default=2e-4)")
rep("torch.manual_seed(42);np.random.seed(42);random.seed(42)","torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)")
rep("    train=merge(load_data('train',args.cache_dir),load_wide('train',args.wide_cache_dir));val=merge(load_data('validation',args.cache_dir),load_wide('validation',args.wide_cache_dir))\n    assert len(train['ids'])==960+715 and len(val['ids'])==120+89,(len(train['ids']),len(val['ids']))",
"    def o300(d,n):\n        assert all(i<300 for i in d['ids'][:n]) and all(i>=300 for i in d['ids'][n:])\n        return dict(rgb=d['rgb'][:n],state=d['state'][:n],action=d['action'][:n],n=n*60,ids=list(d['ids'][:n]))\n    train=o300(load_data('train',args.cache_dir),240);val=o300(load_data('validation',args.cache_dir),30)\n    _s=json.loads(Path('/data02/kemove/rl100-sim-repro/data/demos/sku09-original300/splits.json').read_text());assert sorted(train['ids'])==sorted(_s['train']) and sorted(val['ids'])==sorted(_s['validation'])\n    assert len(train['ids'])==240 and len(val['ids'])==30")
rep("lr=2e-4,betas=(.95,.999),eps","lr=args.lr,betas=(.95,.999),eps")
rep("lr=2e-4,betas=[.95,.999]","lr=args.lr,betas=[.95,.999]")
rep("seed=42,epochs=epochs","seed=args.seed,data_subset='original300_only_nominal_p5y2(train240/val30;no wide;no expanded cells)',epochs=epochs")
rep("dev_protocol=p3_dev_gate.PROTOCOL","dev_protocol=nominal_gate.PROTOCOL")
rep("        source_sha256={d:sha256(src/d) for d in deps if (src/d).exists()},","        source_sha256={d:sha256(src/d) for d in deps if (src/d).exists()},nominal_trainer_sha256=sha256(Path(__file__).resolve()),")
rep("    src=Path(__file__).resolve().parent\n","    src=Path('/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/src')\n")
rep("""        decision=p3_dev_gate.run_stage(OUT,completed_epoch,step,cand,eval_gpus=[int(x) for x in args.eval_gpus.split(',') if x!=''],
            procs_per_gpu=args.eval_procs_per_gpu,mirror=args.mirror_dir,min_stop_epoch=args.min_stop_epoch,smoke=args.smoke)""",
"""        decision=nominal_gate.run_stage(OUT,completed_epoch,step,cand,gpu=int(args.eval_gpus),procs=args.eval_procs_per_gpu,min_stop_epoch=args.min_stop_epoch,smoke=args.smoke,mirror=args.mirror_dir)""")
rep("order=np.random.default_rng(420000+epoch)","order=np.random.default_rng(420000+1000*(args.seed-42)+epoch)")
rep("test_episodes=120,","test_episodes=30,")
Path(sys.argv[1]).write_text(src)
