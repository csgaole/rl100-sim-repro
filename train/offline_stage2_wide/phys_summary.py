"""Summarize runs/phys_probe/probe.json -> REPORT.md table."""
import json,sys
from pathlib import Path
R=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2])
L=['# Physical-state probe (wide rollouts)\n',f"Episodes {R['n_wide']} (train {R['n_train']}, held-out {R['n_heldout']}, seed%5==0); success {R['success_rate']:.3f}; reasons {R['reasons']}; sources {R['sources']}; det frac {R['det_frac']:.2f}\n",
   'AUC on held-out [bootstrap 95% CI]; cell = AUC within the 16 init cells; diff = AUC within quintiles of init-only MLP predicted difficulty; det = det-sampler subset.\n',
   f"Init-only: LR {R['init']['logistic']['auc']:.3f} {[round(x,3) for x in R['init']['logistic']['ci']]}, MLP {R['init']['mlp']['auc']:.3f} {[round(x,3) for x in R['init']['mlp']['ci']]}\n",
   '| t | set | dim | LR AUC [CI] | LR cell | LR diff [CI] | MLP AUC [CI] | MLP cell | MLP diff [CI] | MLP det |','|---|---|---|---|---|---|---|---|---|---|']
f=lambda v:f'{v:.3f}';ci=lambda c:f'[{c[0]:.3f},{c[1]:.3f}]'
for t,T in R['results'].items():
    for nm,v in T.items():
        lr,ml=v['logistic'],v['mlp']
        L.append(f"| {t} | {nm} | {v['dim']} | {f(lr['auc'])} {ci(lr['ci'])} | {f(lr['within_cell'])} | {f(lr['within_difficulty_quintile'])} {ci(lr['within_difficulty_ci'])} | {f(ml['auc'])} {ci(ml['ci'])} | {f(ml['within_cell'])} | {f(ml['within_difficulty_quintile'])} {ci(ml['within_difficulty_ci'])} | {f(ml['det_only'])} |")
L.append('\n## Physical-only extras\n')
for t,T in R['results'].items():
    b=T['b_phys'];sw=b['succ_vs_wrongface']
    L.append(f"- {t}: success vs wrong_face (n={sw['n']}, wf={sw['wrong_face']}): LR {sw['logistic']:.3f} {ci(sw['logistic_ci'])}, MLP {sw['mlp']:.3f} {ci(sw['mlp_ci'])}, within-cell LR {sw['within_cell_logistic']:.3f} MLP {sw['within_cell_mlp']:.3f}")
    L.append(f"  - perm importance (MLP, AUC drop): {b['perm_importance_mlp']}")
    L.append(f"  - top |coef| (LR, standardized): {b['top_coef']}")
out.write_text('\n'.join(L)+'\n');print('\n'.join(L))
