import numpy as np
import pytest
from logistics_sorting_sim import LogisticsSortingEnv
from logistics_sorting_sim.task import OutcomeTracker


def test_seed_reset_and_chunk_contract():
    with LogisticsSortingEnv(render=False) as e:
        obs, _ = e.reset(123)
        initial = e.physics.data.qpos.copy()
        action = np.tile(obs['proprio'][:14], (8, 1))
        _, reward, term, trunc, info = e.step(action)
        after = e.physics.data.qpos.copy()
        assert info['executed_ticks'] == 8
        assert e.physics.data.time == pytest.approx(8/240)
        assert reward == 0 and not term and not trunc
        obs2, _ = e.reset(123)
        assert np.array_equal(initial, e.physics.data.qpos)
        e.step(action)
        assert np.array_equal(after, e.physics.data.qpos)
        e.reset(124)
        assert not np.array_equal(initial, e.physics.data.qpos)
        assert obs2['proprio'].shape == (28,)
        assert set(obs2) == {'rgb', 'proprio', 'timestamp', 'rgb_timestamps'}


def test_watchdog_partial_chunk_and_reset_required():
    with LogisticsSortingEnv(render=False) as e:
        obs, _ = e.reset()
        action = obs['proprio'][:14]
        e.step(action)  # offset chunk alignment by one tick
        while True:
            _, reward, term, trunc, info = e.step(np.tile(action, (8, 1)))
            if term or trunc:
                break
        assert info['reason'] == 'watchdog'
        assert trunc and not term and reward == 0
        assert info['executed_ticks'] == 7
        assert e.physics.data.time == pytest.approx(10)
        with pytest.raises(RuntimeError):
            e.step(action)


def test_success_requires_transport_and_continuity():
    t = OutcomeTracker()
    s = dict(bottom=.8, label_up=1., outfeed_contact=True, outfeed_inside=True,
             forces=[0., 0.], vy=-.45, position=[0., -.6, .84], trailing_y=-.5)
    for i in range(100):
        assert t.update(s, i/240) is None  # upright alone is insufficient
    s['outfeed_contact'] = False
    assert t.update(s, 1) is None
    assert t.stable == 0
    s['outfeed_contact'] = True
    for i in range(47):
        s['position'][1] = -.6-i*.003
        s['trailing_y'] = s['position'][1]+.05
        assert t.update(s, 2+i/240) is None
    s['position'][1] = -.75
    s['trailing_y'] = -.7
    assert t.update(s, 2.2) == 'success'
    s['bottom'] = .73
    assert OutcomeTracker().update(s, 10) == 'drop'


@pytest.mark.parametrize('sku', ['SKU02','SKU09'])
def test_scenes_and_invalid_inputs(sku):
    with LogisticsSortingEnv(sku=sku, render=False) as e:
        for bad in [np.zeros(13), np.zeros((9,14)), np.full(14,np.nan)]:
            with pytest.raises(ValueError):
                e.step(bad)
