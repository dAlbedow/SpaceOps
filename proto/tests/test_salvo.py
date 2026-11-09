import numpy as np

from spaceops import salvo as sv


def test_hit_probability_rayleigh_matches_monte_carlo():
    rng = np.random.default_rng(5)
    sigma, r = 40.0, 50.0
    miss = np.linalg.norm(rng.normal(0, sigma, (200000, 2)), axis=1)
    assert abs((miss <= r).mean() - sv.hit_probability(sigma, r)) < 0.005
    assert sv.hit_probability(0.0, 1.0) == 1.0
    assert abs(sv.hit_probability(1.0, 1.1774) - 0.5) < 1e-3  # median miss = 1.1774 sigma


def test_saturation_estimate_design_doc_example():
    # 4 channels, 100 s window, 8 s per engagement, pk 0.7 -> about 35 kills
    assert abs(sv.saturation_estimate(4, 100.0, 8.0, 0.7) - 35.0) < 1e-12


def test_independent_leak_closed_form():
    assert abs(sv.independent_leak_probability(6, 0.34, 3) - 0.869) < 1e-3
    # one missile, one shot: leak prob = 1 - ph
    assert abs(sv.independent_leak_probability(1, 0.34, 1) - 0.66) < 1e-12


def test_single_round_mc_matches_closed_form_with_even_shots():
    # one volley of 6k shots spread evenly (coordination 1) is the independent case
    for k in (3, 4):
        r = sv.volley_leak_mc(6, 0.34, volleys=1, shots_per_volley=6 * k, coordination=1.0,
                              trials=20000, seed=k)
        assert abs(r.p_any_leak - sv.independent_leak_probability(6, 0.34, k)) < 0.015


def test_published_leak_figures_6_missiles_ph_034():
    # Naval Gazing, Aurora 2.2+ Missile Warfare: 58 / 24 / 7 percent for 3 / 4 / 5 shots per missile
    published = {3: 0.58, 4: 0.24, 5: 0.07}
    for n, target in published.items():
        r = sv.volley_leak_mc(6, 0.34, volleys=n, trials=20000, seed=100 + n)
        assert abs(r.p_any_leak - target) < 0.04, (n, r.p_any_leak, target)


def test_published_figures_lie_between_even_and_random_spreading():
    for n, target in {3: 0.58, 4: 0.24, 5: 0.07}.items():
        even = sv.volley_leak_mc(6, 0.34, n, coordination=1.0, trials=8000, seed=n).p_any_leak
        rand = sv.volley_leak_mc(6, 0.34, n, coordination=0.0, trials=8000, seed=n).p_any_leak
        assert even < target < rand


def test_leak_distribution_is_a_probability_distribution_and_monotone_in_shots():
    prev = 1.1
    for n in (1, 2, 3, 4, 5):
        r = sv.volley_leak_mc(6, 0.34, n, trials=4000, seed=7)
        assert abs(r.distribution.sum() - 1.0) < 1e-12
        assert r.p_any_leak < prev
        prev = r.p_any_leak


def test_engagement_timeline_saturation():
    rng = np.random.default_rng(11)

    def run(n, trials=300):
        out = [sv.engage_timeline(rng.uniform(100.0, 200.0, n), 4, 8.0, 0.7,
                                  first_engage_range_s=100.0, rng=rng) for _ in range(trials)]
        return (np.mean([o["engaged"] for o in out]), np.mean([o["killed"] for o in out]),
                np.mean([o["leakers"] for o in out]))

    # effective window = 100 s engagement range + 100 s arrival spread: capacity about 4*200/8 = 100
    e20, k20, l20 = run(20)
    assert e20 == 20 and abs(k20 / 20 - 0.7) < 0.05  # everything engaged, losses are just pk
    e120, k120, l120 = run(120)
    assert e120 < 100 and k120 < 0.7 * e120 + 3  # saturated: not everything can be engaged
    assert l120 / 120 > 0.35 and l120 / 120 > 2 * l20 / 20 * 0.7  # leak fraction jumps


def test_launch_clear_check():
    launcher = np.zeros(3)
    target = np.array([1.0e6, 0.0, 0.0])
    ally_in_way = (np.array([5.0e5, 10.0, 0.0]), 100.0)
    ally_aside = (np.array([5.0e5, 5.0e4, 0.0]), 100.0)
    assert not sv.launch_is_clear(launcher, target, [ally_in_way])
    assert sv.launch_is_clear(launcher, target, [ally_aside])
    assert sv.launch_is_clear(launcher, target, [])
