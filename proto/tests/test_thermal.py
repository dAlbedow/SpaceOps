import numpy as np

from spaceops import thermal as th
from spaceops.constants import SIGMA_SB, T_CMB


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b))


def test_radiator_area_table_from_design_doc():
    # single-sided, emissivity 0.9, area per GW; design doc section 5 table
    expected = {500: 313_000, 800: 47_800, 1000: 19_600, 1500: 3_900, 2000: 1_200}
    for T, area in expected.items():
        a = th.radiator_area_for_power(1e9, T, 0.9)
        assert rel(a, area) < 0.03, (T, a, area)


def test_radiator_flux_matches_stefan_boltzmann():
    assert rel(th.radiator_flux(1000.0, 0.9), 0.9 * SIGMA_SB * 1e12) < 1e-12
    # fourth-power law: doubling T gives 16x flux
    assert rel(th.radiator_flux(2000.0), 16 * th.radiator_flux(1000.0)) < 1e-12


def test_equilibrium_temperature_inverts_flux():
    area = 5.0e4
    T = th.equilibrium_temperature(1e9, area, 0.9, T_env=0.0, sides=1)
    assert rel(0.9 * SIGMA_SB * area * T**4, 1e9) < 1e-12


def _node(C=2.0e11 / 400.0, T0=300.0, area=5.0e4, state=th.RadiatorState.EXTENDED):
    rad = th.Radiators(area=area, sides=2, state=state)
    return th.ThermalNode(C=C, T=T0, radiators=rad, T_safe=700.0, T_damage=900.0)


def test_simulation_converges_to_closed_form_equilibrium():
    n = _node(T0=300.0, area=2.0e4)
    Q = 3.0e8
    for _ in range(4000):
        n.step(Q, 10.0)
    T_eq = th.equilibrium_temperature(Q, n.radiators.effective_area(), 0.9, n.T_env, sides=2)
    assert abs(n.T - T_eq) / T_eq < 1e-4, (n.T, T_eq)


def test_energy_conservation_over_a_run():
    n = _node(T0=300.0)
    Q = 4.0e8
    E0 = n.stored_energy(0.0)
    for _ in range(500):
        n.step(Q, 5.0)
    dE = n.stored_energy(0.0) - E0
    assert rel(n.energy_in, dE + n.energy_radiated) < 1e-9


def test_implicit_euler_is_stable_with_huge_steps():
    n = _node(T0=300.0, area=1.0e3, C=1.0e6)
    for _ in range(50):
        n.step(1e9, 1.0e6)
    assert np.isfinite(n.T) and n.T > 0


def test_sink_endurance_numbers_from_design_doc():
    # 100 t sink at 2 MJ/kg holds 2e11 J: 33 minutes at 100 MW, 200 s at 1 GW
    cap = th.ThermalNode.sink_capacity(1.0e5, 2.0e6)
    assert rel(cap, 2.0e11) < 1e-12
    assert abs(cap / 1.0e8 / 60.0 - 33.3) < 0.1
    assert abs(cap / 1.0e9 - 200.0) < 1e-9


def test_cold_run_endurance_equals_energy_budget_with_no_radiation():
    C = 2.0e11 / 400.0  # 400 K working range -> 2e11 J
    n = _node(C=C, T0=300.0, state=th.RadiatorState.RETRACTED)
    n.radiators.retracted_fraction = 0.0  # perfectly stowed: no leak at all
    n.T_safe = 700.0
    Q = 1.0e8
    # no radiation: t = C dT / Q = 2000 s exactly
    assert abs(n.time_to_temperature(700.0, Q) - 2000.0) < 1e-6
    t = 0.0
    while n.T < 700.0 and t < 1e5:
        n.step(Q, 1.0)
        t += 1.0
    assert abs(t - 2000.0) <= 1.0, t


def test_retracted_residual_radiation_lengthens_endurance_and_forecast_matches_sim():
    C = 2.0e11 / 400.0
    n = _node(C=C, T0=300.0, state=th.RadiatorState.RETRACTED)  # 1 percent residual area
    n.T_safe = 700.0
    Q = 1.0e8
    t_forecast = n.time_to_temperature(700.0, Q)
    # residual leak radiates ~12 percent of Q near 700 K, so endurance exceeds the 2000 s budget
    assert 2000.0 < t_forecast < 2300.0, t_forecast
    t = 0.0
    while n.T < 700.0 and t < 1e5:
        n.step(Q, 1.0)
        t += 1.0
    assert abs(t - t_forecast) / t_forecast < 0.01, (t, t_forecast)


def test_forecast_quadrature_matches_simulation_with_radiation():
    n = _node(T0=300.0, area=1.0e4)
    Q = 6.0e8
    f = n.forecast(Q)
    assert np.isfinite(f["time_to_safe_s"]) and f["time_to_safe_s"] > 0
    t = 0.0
    while n.T < n.T_safe:
        n.step(Q, 0.5)
        t += 0.5
        if t > 1e6:
            break
    assert abs(t - f["time_to_safe_s"]) / f["time_to_safe_s"] < 0.01


def test_no_limit_crossing_when_equilibrium_below_limit():
    n = _node(T0=300.0, area=5.0e4)
    f = n.forecast(1.0e8)
    assert f["time_to_safe_s"] == float("inf")
    assert f["equilibrium_K"] < n.T_safe


def test_auto_throttle_above_safe_temperature():
    n = _node(T0=800.0, area=1.0e4)
    n.T_safe = 700.0
    q = 5.0e9
    lim = n.throttle_limit(q)
    assert 0.0 < lim < 1.0
    assert rel(lim * q, n.radiated_power()) < 1e-12
    n.T = 500.0
    assert n.throttle_limit(q) == 1.0


def test_damage_flag_set_at_damage_temperature():
    n = _node(T0=880.0, area=10.0)
    n.step(1.0e10, 100.0)
    assert n.damaged


def test_power_allocation_priority_and_brownout():
    cons = [
        th.Consumer("lasers", 6.0e8, 0.6, priority=3),
        th.Consumer("drive", 8.0e8, 0.3, priority=1),
        th.Consumer("radar", 1.0e8, 0.8, priority=2),
    ]
    got = th.allocate(1.2e9, cons)
    assert got["drive"] == 8.0e8
    assert got["radar"] == 1.0e8
    assert abs(got["lasers"] - 3.0e8) < 1e-3  # gets what is left


def test_ship_energy_bookkeeping_closes():
    reactor = th.Reactor(thermal_power=3.0e9, efficiency=0.4)
    cons = [th.Consumer("drive", 6.0e8, 0.3, 1), th.Consumer("radar", 1.0e8, 0.8, 2),
            th.Consumer("lasers", 4.0e8, 0.6, 3)]
    got = th.allocate(reactor.electrical(), cons)
    heat, out = th.ship_heat_load(reactor, cons, got)
    unused = reactor.electrical() - sum(got.values())
    assert rel(reactor.thermal_power, heat + out + unused) < 1e-12
