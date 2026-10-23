# [ic6 diag] closed-loop 18-episode smoke mix gate of one checkpoint, same seeds as the pipeline smoke gate.
import sys,json
pol,name,gpus=sys.argv[1],sys.argv[2],sys.argv[3]
sys.argv=['wide_pipeline_ic6.py','--smoke','--gpus',gpus,'--per-gpu',sys.argv[4] if len(sys.argv)>4 else '2','--rounds','1','--skip-final']
import wide_pipeline_ic6 as W
r=W.gate(pol,name)
print('GATE_ONE',name,f"{r['wide_successes']}/{r['wide_episodes']}",'nominal',f"{r['nominal_successes']}/{r['nominal_episodes']}",flush=True)
