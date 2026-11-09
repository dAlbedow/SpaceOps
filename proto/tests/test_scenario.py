import numpy as np

from spaceops import scenario as sc
from spaceops import sensing as se
from spaceops.constants import GM_EARTH, R_EARTH

_RES = {}


def duel():
    if "r" not in _RES:
        _RES["r"] = sc.run_duel(seed=1)
    return _RES["r"]


def test_duel_is_deterministic_for_a_seed():
    a = duel()
    b = sc.run_duel(seed=1)
    assert np.array_equal(a.pos_b, b.pos_b) and np.array_equal(a.temp_b, b.temp_b)
    assert a.salvo["A"]["mean_leakers"] == b.salvo["A"]["mean_leakers"]


def test_ship_a_stays_on_its_circular_orbit_and_b_moves():
    r = duel()
    assert np.allclose(np.linalg.norm(r.pos_a, axis=1), 8.0e6, rtol=1e-12)
    assert np.linalg.norm(r.pos_b[-1] - r.pos_b[0]) > 1.0e7


def test_hot_loud_ship_is_seen_far_beyond_the_dark_ship():
    r = duel()
    assert r.passive_range_a_sees_b[0] > 20 * r.passive_range_b_sees_a[0]
    assert r.a_detects_b.sum() > 10 * max(1, r.b_detects_a.sum())


def test_dark_ship_stays_cold_and_burning_ship_runs_hot():
    r = duel()
    assert r.temp_a.max() < 700.0  # cold-run clock never expired
    assert r.temp_b.max() > r.temp_b[0] + 100.0  # the burn heats B


def test_planet_blocks_line_of_sight_part_of_the_time():
    r = duel()
    assert 0.1 < r.blocked.mean() < 0.9
    # blocked steps never produce a detection
    assert not r.a_detects_b[r.blocked].any()


def test_dark_ship_gets_fire_control_and_shoots_first():
    r = duel()
    assert "A" in r.salvo and "B" in r.salvo
    assert r.salvo["A"]["launch_t"] < r.salvo["B"]["launch_t"]
    assert r.a_quality.max() == int(se.TrackQuality.FIRE_CONTROL)


def test_salvo_results_are_valid_distributions_and_weaker_defense_leaks_more():
    r = duel()
    for res in r.salvo.values():
        assert abs(res["distribution"].sum() - 1.0) < 1e-12
        assert 0.0 <= res["p_any_leak"] <= 1.0
    # A fires into B's strong defense, B fires into A's weak one
    assert r.salvo["B"]["mean_leakers"] > r.salvo["A"]["mean_leakers"]


def test_events_are_time_ordered_and_closest_approach_is_inside_launch_range():
    r = duel()
    times = [e[0] for e in r.events]
    assert times == sorted(times)
    assert r.range_ab.min() < sc.LAUNCH_RANGE_M


def test_ship_b_never_goes_inside_the_planet():
    r = duel()
    assert np.linalg.norm(r.pos_b, axis=1).min() > R_EARTH
