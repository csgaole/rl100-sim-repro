"""Versioned task success; velocity settling is reported as quality, not reward."""
import json, math

SUCCESS_VERSION = 'flip_pose_no_drop_v2'
LEGACY_VERSION = 'flip_stability_v1'
LABEL_UP_MIN = .98
BOTTOM_MIN_M = .79

def assess_task(final_label_up, final_bottom_m, dropped, complete, stable):
    up=float(final_label_up);bottom=float(final_bottom_m)
    if not math.isfinite(up) or not math.isfinite(bottom):
        raise ValueError('Non-finite terminal task state')
    flags=[bool(v) for v in stable]
    strict=bool(complete and len(flags)==480 and all(flags[-48:]) and not dropped)
    success=bool(complete and not dropped and up>LABEL_UP_MIN and bottom>BOTTOM_MIN_M)
    strict_reason='success' if strict else 'drop' if dropped else 'wrong_face' if up<=LABEL_UP_MIN else 'unsettled'
    reason='drop' if dropped else 'incomplete' if not complete else 'wrong_face' if up<=LABEL_UP_MIN else 'below_height' if bottom<=BOTTOM_MIN_M else 'success'
    return dict(success=success,task_success=success,reason=reason,
        success_criteria_version=SUCCESS_VERSION,strict_success=strict,strict_reason=strict_reason,
        stability_quality='stable' if strict else 'unsettled_success' if success else 'task_failure',
        stable_steps_last_48=sum(flags[-48:]),stability_window_steps=48,
        final_label_up=up,final_bottom_m=bottom)

def assess_rollout_h5(f, metadata=None):
    """Read privileged terminal state for labeling; never modify the source H5."""
    m=metadata if metadata is not None else json.loads(f.attrs['metadata_json'])
    n=int(m['chunks']);stable=f['stable'][:]
    if 'parcel_motion' not in f:
        raise ValueError('Terminal physical state missing: replay required for relabeling')
    pose=f['parcel_motion'][-1]
    # Stored rotation is R[:, :2], row-major. R22 = cross(R[:,0],R[:,1])[2].
    up=float(pose[3])*float(pose[6])-float(pose[5])*float(pose[4])
    bottom=float(pose[15])
    dropped=bool((f['chunk_bottom_min'][:]<.74).any())
    result=assess_task(up,bottom,dropped,n==60,stable)
    result.update(label_source='recorded_terminal_parcel_motion',recorded_success=bool(m['success']),
        recorded_reason=m['reason'],recorded_success_criteria_version=m.get('success_criteria_version',LEGACY_VERSION))
    # Rotation serialization is float32. Check legacy classification unless
    # threshold roundoff alone might change its face comparison.
    if abs(up-LABEL_UP_MIN)>1e-6 and m.get('success_criteria_version',LEGACY_VERSION)==LEGACY_VERSION:
        assert result['strict_success']==bool(m['success'])
        assert result['strict_reason']==m['reason']
    return result
