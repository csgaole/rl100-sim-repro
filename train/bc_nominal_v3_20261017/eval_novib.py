import sys
sys.path.insert(0,'/data02/kemove/rl100-sim-repro/train/offline_stage2_v3')
import novib_patch  # exact offline V3_START_POLICY loader path (strict no-VIB load)
import evaluate_bc
assert evaluate_bc.BCPolicy.__module__=='bc_model_novib'
evaluate_bc.main()
