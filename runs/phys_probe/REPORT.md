# Physical-state probe (wide rollouts)

Episodes 5625 (train 4500, held-out 1125, seed%5==0); success 0.685; reasons {'drop': 299, 'success': 3854, 'wrong_face': 1472}; sources {'v2_round1': 1125, 'wide_round1': 3000, 'wide_round2': 1500}; det frac 0.68

AUC on held-out [bootstrap 95% CI]; cell = AUC within the 16 init cells; diff = AUC within quintiles of init-only MLP predicted difficulty; det = det-sampler subset.

Init-only: LR 0.752 [0.718, 0.781], MLP 0.779 [0.75, 0.808]

| t | set | dim | LR AUC [CI] | LR cell | LR diff [CI] | MLP AUC [CI] | MLP cell | MLP diff [CI] | MLP det |
|---|---|---|---|---|---|---|---|---|---|
| t0 | a_BC | 1556 | 0.781 [0.751,0.811] | 0.756 | 0.607 [0.570,0.644] | 0.774 [0.745,0.804] | 0.748 | 0.581 [0.538,0.618] | 0.802 |
| t0 | b_phys | 40 | 0.666 [0.631,0.702] | 0.663 | 0.527 [0.484,0.566] | 0.780 [0.748,0.809] | 0.758 | 0.604 [0.565,0.644] | 0.805 |
| t0 | c_init | 13 | 0.752 [0.721,0.781] | 0.721 | 0.567 [0.526,0.610] | 0.779 [0.751,0.807] | 0.756 | 0.587 [0.548,0.627] | 0.802 |
| t0 | e_phys+init | 53 | 0.758 [0.727,0.786] | 0.726 | 0.578 [0.542,0.619] | 0.781 [0.752,0.808] | 0.757 | 0.608 [0.566,0.645] | 0.806 |
| t0 | d_BC+phys | 1596 | 0.781 [0.753,0.812] | 0.757 | 0.609 [0.572,0.646] | 0.777 [0.744,0.804] | 0.754 | 0.592 [0.550,0.630] | 0.802 |
| t5 | a_BC | 3128 | 0.785 [0.753,0.813] | 0.760 | 0.637 [0.600,0.674] | 0.775 [0.746,0.806] | 0.755 | 0.620 [0.584,0.658] | 0.817 |
| t5 | b_phys | 89 | 0.741 [0.713,0.773] | 0.716 | 0.593 [0.553,0.630] | 0.771 [0.740,0.800] | 0.745 | 0.645 [0.605,0.685] | 0.802 |
| t5 | c_init | 13 | 0.752 [0.722,0.783] | 0.721 | 0.567 [0.526,0.605] | 0.779 [0.749,0.808] | 0.756 | 0.587 [0.547,0.626] | 0.802 |
| t5 | e_phys+init | 102 | 0.769 [0.739,0.797] | 0.741 | 0.623 [0.583,0.663] | 0.785 [0.754,0.811] | 0.761 | 0.650 [0.611,0.688] | 0.813 |
| t5 | d_BC+phys | 3217 | 0.785 [0.755,0.814] | 0.762 | 0.636 [0.601,0.674] | 0.774 [0.744,0.802] | 0.748 | 0.617 [0.579,0.651] | 0.814 |
| t10 | a_BC | 3128 | 0.786 [0.757,0.812] | 0.764 | 0.643 [0.603,0.681] | 0.787 [0.756,0.816] | 0.772 | 0.630 [0.592,0.669] | 0.817 |
| t10 | b_phys | 97 | 0.719 [0.686,0.751] | 0.705 | 0.617 [0.579,0.655] | 0.764 [0.732,0.793] | 0.741 | 0.612 [0.571,0.650] | 0.797 |
| t10 | c_init | 13 | 0.752 [0.722,0.783] | 0.721 | 0.567 [0.527,0.605] | 0.779 [0.750,0.807] | 0.756 | 0.587 [0.544,0.626] | 0.802 |
| t10 | e_phys+init | 110 | 0.767 [0.738,0.797] | 0.736 | 0.620 [0.581,0.659] | 0.781 [0.750,0.809] | 0.760 | 0.626 [0.587,0.664] | 0.814 |
| t10 | d_BC+phys | 3225 | 0.785 [0.753,0.813] | 0.764 | 0.645 [0.607,0.684] | 0.785 [0.755,0.813] | 0.770 | 0.640 [0.601,0.675] | 0.813 |
| t15 | a_BC | 3128 | 0.832 [0.806,0.857] | 0.817 | 0.734 [0.701,0.767] | 0.844 [0.818,0.867] | 0.835 | 0.756 [0.724,0.788] | 0.867 |
| t15 | b_phys | 100 | 0.787 [0.758,0.815] | 0.778 | 0.687 [0.652,0.723] | 0.825 [0.796,0.850] | 0.812 | 0.723 [0.689,0.758] | 0.854 |
| t15 | c_init | 13 | 0.752 [0.721,0.783] | 0.721 | 0.567 [0.525,0.611] | 0.779 [0.747,0.809] | 0.756 | 0.587 [0.546,0.628] | 0.802 |
| t15 | e_phys+init | 113 | 0.798 [0.770,0.825] | 0.789 | 0.682 [0.647,0.717] | 0.827 [0.798,0.851] | 0.817 | 0.719 [0.685,0.751] | 0.856 |
| t15 | d_BC+phys | 3228 | 0.833 [0.809,0.858] | 0.818 | 0.734 [0.699,0.768] | 0.843 [0.818,0.869] | 0.834 | 0.755 [0.722,0.787] | 0.866 |
| t20 | a_BC | 3128 | 0.880 [0.859,0.901] | 0.876 | 0.828 [0.802,0.855] | 0.886 [0.866,0.906] | 0.883 | 0.839 [0.810,0.868] | 0.912 |
| t20 | b_phys | 102 | 0.832 [0.807,0.859] | 0.825 | 0.759 [0.724,0.793] | 0.871 [0.850,0.891] | 0.860 | 0.807 [0.773,0.838] | 0.903 |
| t20 | c_init | 13 | 0.752 [0.720,0.781] | 0.721 | 0.567 [0.526,0.606] | 0.779 [0.750,0.808] | 0.756 | 0.587 [0.546,0.630] | 0.802 |
| t20 | e_phys+init | 115 | 0.834 [0.807,0.858] | 0.826 | 0.748 [0.713,0.781] | 0.857 [0.832,0.880] | 0.852 | 0.779 [0.746,0.811] | 0.881 |
| t20 | d_BC+phys | 3230 | 0.885 [0.864,0.907] | 0.880 | 0.834 [0.806,0.863] | 0.883 [0.859,0.903] | 0.880 | 0.830 [0.800,0.857] | 0.906 |
| t30 | a_BC | 3128 | 0.999 [0.998,1.000] | 0.999 | 0.998 [0.997,1.000] | 0.998 [0.996,1.000] | 0.998 | 0.996 [0.991,0.999] | 0.998 |
| t30 | b_phys | 106 | 0.995 [0.987,0.999] | 0.995 | 0.992 [0.983,0.999] | 0.995 [0.988,0.999] | 0.996 | 0.996 [0.992,0.999] | 0.999 |
| t30 | c_init | 13 | 0.752 [0.723,0.782] | 0.721 | 0.567 [0.524,0.605] | 0.779 [0.748,0.807] | 0.756 | 0.587 [0.546,0.628] | 0.802 |
| t30 | e_phys+init | 119 | 0.995 [0.988,0.999] | 0.995 | 0.992 [0.982,0.999] | 0.991 [0.981,0.999] | 0.992 | 0.992 [0.984,0.998] | 0.999 |
| t30 | d_BC+phys | 3234 | 0.996 [0.989,1.000] | 0.996 | 0.995 [0.988,1.000] | 0.999 [0.997,1.000] | 0.998 | 0.997 [0.995,0.999] | 0.999 |
| t40 | a_BC | 3128 | 0.999 [0.996,1.000] | 0.999 | 0.997 [0.990,1.000] | 0.998 [0.995,1.000] | 0.998 | 0.997 [0.991,1.000] | 0.998 |
| t40 | b_phys | 105 | 1.000 [1.000,1.000] | 1.000 | 1.000 [1.000,1.000] | 0.997 [0.993,1.000] | 0.997 | 0.996 [0.988,1.000] | 0.998 |
| t40 | c_init | 13 | 0.752 [0.722,0.784] | 0.721 | 0.567 [0.526,0.604] | 0.779 [0.749,0.806] | 0.756 | 0.587 [0.545,0.627] | 0.802 |
| t40 | e_phys+init | 118 | 0.999 [0.997,1.000] | 0.999 | 0.998 [0.994,1.000] | 0.997 [0.991,1.000] | 0.997 | 0.998 [0.993,1.000] | 1.000 |
| t40 | d_BC+phys | 3233 | 1.000 [0.999,1.000] | 1.000 | 0.999 [0.998,1.000] | 1.000 [0.999,1.000] | 0.999 | 0.999 [0.998,1.000] | 1.000 |

## Physical-only extras

- t0: success vs wrong_face (n=1074, wf=312): LR 0.651 [0.611,0.690], MLP 0.756 [0.724,0.789], within-cell LR 0.653 MLP 0.734
  - perm importance (MLP, AUC drop): {'tcp_rel_parcel': 0.1708, 'parcel_pose': 0.0958, 'arm_q': 0.0, 'tcp_pose': 0.0, 'running_aggr': 0.0}
  - top |coef| (LR, standardized): [['tcp0_minus_parcel_y', 0.039], ['tcp1_minus_parcel_y', 0.039], ['parcel_pos_y', -0.039], ['parcel_R10', 0.029], ['parcel_R01', 0.029], ['parcel_pos_x', -0.008], ['tcp1_minus_parcel_x', 0.008], ['tcp0_minus_parcel_x', 0.008], ['parcel_R11', 0.001], ['parcel_R00', -0.001], ['parcel_pos_z', -0.0], ['tcp1_pos_x', -0.0]]
- t5: success vs wrong_face (n=1074, wf=312): LR 0.710 [0.677,0.744], MLP 0.738 [0.705,0.769], within-cell LR 0.690 MLP 0.719
  - perm importance (MLP, AUC drop): {'tcp_rel_parcel': 0.1115, 'parcel_pose': 0.0655, 'parcel_vel': 0.0492, 'arm_qvel': 0.0241, 'tcp_pose': 0.015, 'arm_q': 0.0147, 'running_aggr': 0.0092, 'contacts': 0.006}
  - top |coef| (LR, standardized): [['parcel_linvel_y', 0.628], ['tcp0_minus_parcel_y', 0.283], ['parcel_pos_y', -0.283], ['tcp1_minus_parcel_y', 0.282], ['q_5', -0.208], ['ccount_transfer_table', -0.181], ['parcel_R01', 0.148], ['parcel_R10', 0.148], ['parcel_R21', -0.116], ['parcel_R02', 0.113], ['qvel_1', 0.107], ['outfeed_contact', -0.106]]
- t10: success vs wrong_face (n=1074, wf=312): LR 0.685 [0.650,0.721], MLP 0.758 [0.727,0.788], within-cell LR 0.669 MLP 0.744
  - perm importance (MLP, AUC drop): {'tcp_rel_parcel': 0.0967, 'parcel_pose': 0.0751, 'arm_qvel': 0.0206, 'arm_q': 0.0191, 'tcp_pose': 0.0122, 'parcel_vel': 0.0078, 'contacts': 0.0057, 'running_aggr': -0.0044}
  - top |coef| (LR, standardized): [['q_14', 0.397], ['parcel_linvel_y', 0.39], ['tcp1_R11', 0.368], ['parcel_R02', -0.329], ['q_15', 0.298], ['ccount_arm1_grip', -0.273], ['parcel_R10', 0.258], ['q_7', -0.249], ['outfeed_contact', -0.237], ['parcel_R01', 0.236], ['run_min_bottom', 0.234], ['parcel_R21', 0.212]]
- t15: success vs wrong_face (n=1074, wf=312): LR 0.753 [0.722,0.782], MLP 0.780 [0.750,0.808], within-cell LR 0.745 MLP 0.770
  - perm importance (MLP, AUC drop): {'tcp_rel_parcel': 0.0829, 'tcp_pose': 0.0531, 'parcel_pose': 0.0508, 'arm_q': 0.038, 'contacts': 0.0357, 'arm_qvel': 0.0323, 'running_aggr': 0.0127, 'parcel_vel': 0.0095}
  - top |coef| (LR, standardized): [['qvel_8', -0.522], ['qvel_0', -0.509], ['parcel_linvel_x', 0.482], ['parcel_angvel_z', 0.403], ['parcel_R11', 0.342], ['parcel_pos_y', -0.341], ['parcel_R00', -0.341], ['parcel_R01', 0.34], ['tcp1_minus_parcel_y', 0.336], ['q_14', 0.328], ['run_min_bottom', 0.327], ['ccount_arm1_grip', 0.306]]
- t20: success vs wrong_face (n=1074, wf=312): LR 0.807 [0.777,0.836], MLP 0.860 [0.835,0.883], within-cell LR 0.805 MLP 0.854
  - perm importance (MLP, AUC drop): {'tcp_rel_parcel': 0.0909, 'parcel_pose': 0.0714, 'arm_qvel': 0.0546, 'tcp_pose': 0.0518, 'parcel_vel': 0.0513, 'contacts': 0.0289, 'arm_q': 0.0229, 'running_aggr': 0.0153}
  - top |coef| (LR, standardized): [['qvel_0', 1.751], ['bottom', -1.44], ['tcp0_minus_parcel_z', -0.963], ['tcp0_R02', 0.935], ['parcel_pos_z', 0.81], ['tcp1_minus_parcel_z', -0.795], ['parcel_R10', 0.764], ['run_max_label_up', -0.708], ['tcp0_R10', 0.669], ['tcp0_R21', -0.645], ['q_6', -0.569], ['tcp0_minus_parcel_x', -0.565]]
- t30: success vs wrong_face (n=1074, wf=312): LR 0.995 [0.988,1.000], MLP 0.992 [0.982,1.000], within-cell LR 0.996 MLP 0.994
  - perm importance (MLP, AUC drop): {'parcel_pose': 0.2397, 'parcel_vel': 0.0128, 'running_aggr': 0.0072, 'tcp_rel_parcel': 0.0026, 'contacts': 0.002, 'tcp_pose': 0.001, 'arm_qvel': 0.0005, 'arm_q': 0.0002}
  - top |coef| (LR, standardized): [['label_up', 1.461], ['parcel_R22', 1.461], ['parcel_angvel_y', 1.044], ['parcel_R11', -1.041], ['parcel_R01', -1.018], ['parcel_linvel_x', 0.775], ['tcp1_minus_parcel_y', 0.591], ['parcel_pos_y', -0.589], ['tcp0_minus_parcel_y', 0.554], ['run_max_label_up', 0.537], ['ccount_transfer_table', 0.465], ['parcel_R20', 0.461]]
- t40: success vs wrong_face (n=1074, wf=312): LR 1.000 [1.000,1.000], MLP 0.999 [0.996,1.000], within-cell LR 1.000 MLP 0.999
  - perm importance (MLP, AUC drop): {'parcel_pose': 0.2395, 'running_aggr': 0.0027, 'contacts': 0.0009, 'tcp_rel_parcel': 0.0008, 'parcel_vel': 0.0001, 'tcp_pose': 0.0001, 'arm_q': 0.0, 'arm_qvel': 0.0}
  - top |coef| (LR, standardized): [['label_up', 0.729], ['parcel_R22', 0.729], ['parcel_R01', -0.62], ['parcel_R02', 0.586], ['parcel_R21', 0.514], ['run_max_label_up', 0.5], ['ccount_transfer_table', 0.332], ['parcel_R11', -0.329], ['run_drop', -0.201], ['parcel_R20', 0.192], ['parcel_R12', 0.189], ['bottom', 0.16]]

## Data provenance
- Existing data had NO per-chunk physical state: ep_*.npz store only frozen BC features (61x1564) + normalized action chunks; critic_v2_ablation/events/*.npz store only per-step label_up/bottom/stable + executed actions.
- Nothing re-collected with the policy: all 6000 rollouts of wide_round1 (3000, Stage-1), wide_round2 (1500, Stage-1), v2_round1 (1500, round-2 cand) were REPLAYED physics-only from stored init pose + stored actions (phys_replay.py, same mechanism as wide_events.py). 6000/6000 reproduce success label and terminal label_up/bottom exactly (|d|<1e-6); 43 s wall with 48 procs. Probes use the 5625 wide episodes (nominal v2 episodes excluded).
- Physical state per chunk boundary c=0..60 (107 dims): parcel pos/rotmat/label_up/bottom/lin+ang vel, 16 arm+gripper q and qvel, both TCP pos/rotmat, TCP-parcel offsets, contact normal force + count by partner (gripper / other link per arm, table, infeed, outfeed, floor), outfeed flags, running max label_up / min bottom / peak gripper force / contact steps / drop.

## Verdict
- Physical state does NOT raise early predictability. At t0-t10 every feature set lands at AUC ~0.76-0.79 (BC 0.78-0.79, physical 0.72-0.78, init-only MLP 0.779, BC+phys 0.78-0.79); nothing approaches 0.85. Within-difficulty AUC (quintiles of the init-only MLP) is 0.58-0.65 for all sets vs 0.587 for the init model itself, overlapping CIs; within-cell AUC tracks overall AUC. Success-vs-wrong_face from physical state at t5/t10: 0.71-0.76.
- The frozen BC features are therefore not missing outcome-relevant information: at t<=10 they carry about what the init pose carries, and privileged physics adds nothing measurable. Early-state blindness is not a feature/critic-input problem.
- The outcome is decided during chunks ~10-25 (AUC 0.79 at t10 -> 0.84 t15 -> 0.88 t20 -> ~1.0 at t30, where parcel orientation/label_up dominate). Before that, the most informative physical variables are the init-pose proxies (TCP-parcel y offset, parcel y, parcel yaw); parcel velocity, arm q/qvel and contacts add little.
- Det-sampler episodes (68%) are deterministic given init pose + noise seed, yet are not predictable early either (det AUC 0.80-0.82 at t0-10). So early outcome is effectively unpredictable from state: it depends on the policy's subsequent actions during the flip (sensitive contact dynamics plus sampler noise), not on hidden state at t<=10. Credit assignment has to come from the t10-25 decision window, i.e. action-conditioned / on-policy signal (online RL) or policy/BC changes, not from a better early-state critic.
