import numpy as np

from spaceops import emcon as em
from spaceops import sensing as se
from spaceops import thermal as th


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b))


def node(T0=300.0, state=th.RadiatorState.EXTENDED, retracted_fraction=0.0):
    rad = th.Radiators(area=2.0e4, sides=2, state=state, retracted_fraction=retracted_fraction)
    return th.ThermalNode(C=2.0e11 / 400.0, T=T0, radiators=rad, T_safe=700.0, T_damage=900.0)


def test_policy_table_matches_design_doc():
    P = em.POLICIES
    assert P[em.Emcon.FULL].radar and P[em.Emcon.FULL].rf_link and P[em.Emcon.FULL].radiators_extended
    r = P[em.Emcon.RESTRICTED]
    assert not r.radar and r.laser_link and not r.rf_link and 0 < r.drive_limit < 1
    s = P[em.Emcon.SILENT]
    assert not (s.radar or s.laser_link or s.rf_link) and s.drive_limit == 0 and not s.radiators_extended
    d = P[em.Emcon.DARK]
    assert d.drive_limit == 0 and not d.radiators_extended and d.reactor_level < 0.1


def test_schedule_applies_in_order_and_sets_radiators():
    n = node()
    c = em.EmconController(schedule=[(100.0, em.Emcon.SILENT), (500.0, em.Emcon.FULL)])
    c.update(0.0, n)
    assert c.state == em.Emcon.FULL and n.radiators.state == th.RadiatorState.EXTENDED
    c.update(150.0, n)
    assert c.state == em.Emcon.SILENT and n.radiators.state == th.RadiatorState.RETRACTED
    c.update(400.0, n)
    assert c.state == em.Emcon.SILENT
    c.update(600.0, n)
    assert c.state == em.Emcon.FULL and n.radiators.state == th.RadiatorState.EXTENDED


def test_trigger_fires_once():
    n = node()
    c = em.EmconController(state=em.Emcon.DARK)
    c.triggers.append(em.Trigger("missile warning", lambda ctx: ctx.get("missile_warning", False),
                                 em.Emcon.FULL))
    c.update(0.0, n, {})
    assert c.state == em.Emcon.DARK
    c.update(1.0, n, {"missile_warning": True})
    assert c.state == em.Emcon.FULL
    c.update(2.0, n, {"missile_warning": False})  # does not revert
    assert c.state == em.Emcon.FULL


def test_endurance_meter_equals_budget_and_restores_state():
    n = node(state=th.RadiatorState.EXTENDED, retracted_fraction=0.0)
    assert abs(em.endurance_s(n, 1.0e8) - 2000.0) < 1e-6  # 2e11 J / 1e8 W
    assert n.radiators.state == th.RadiatorState.EXTENDED


def test_cold_run_clock_forces_radiators_out_at_endurance():
    n = node(retracted_fraction=0.0)
    c = em.EmconController(state=em.Emcon.SILENT)
    Q, dt, t = 1.0e8, 1.0, 0.0
    predicted = em.endurance_s(n, Q)
    while c.forced_extension_time is None and t < 5000:
        c.update(t, n)
        n.step(Q, dt)
        t += dt
    assert c.forced_extension_time is not None
    assert abs(c.forced_extension_time - predicted) <= 2.0
    assert n.radiators.state == th.RadiatorState.EXTENDED


def test_dark_ship_is_far_dimmer_than_full_burn():
    cold = node(T0=300.0, state=th.RadiatorState.RETRACTED, retracted_fraction=0.01)
    hot = node(T0=800.0, state=th.RadiatorState.EXTENDED)
    i_cold = em.ship_ir_intensity(cold)
    i_hot = em.ship_ir_intensity(hot, plume_w=1.0e9)
    assert i_hot > 1.0e3 * i_cold
    # and detectable much farther
    nei = 1e-14
    assert se.passive_range_for_snr(i_hot, 10, nei) > 30 * se.passive_range_for_snr(i_cold, 10, nei)


def test_forced_radiator_spike_is_brighter_than_the_cold_hull():
    n = node(T0=700.0, state=th.RadiatorState.RETRACTED, retracted_fraction=0.0)
    before = em.ship_ir_intensity(n)
    c = em.EmconController(state=em.Emcon.SILENT)
    c.update(0.0, n)
    assert c.forced_extension_time == 0.0
    assert em.ship_ir_intensity(n) > 100 * before


def test_esm_hears_radar_farther_than_radar_sees():
    p, g, lam, rcs = 1.0e6, 1.0e4, 0.03, 100.0
    r_radar = se.radar_range_for_snr(p, g, lam, rcs, 13.0)
    r_esm = em.esm_intercept_range(p, g, sensitivity_w_m2=1e-12)
    assert r_esm > 5 * r_radar
    # R^-2 law: 4x power doubles intercept range
    assert rel(em.esm_intercept_range(4 * p, g, 1e-12) / r_esm, 2.0) < 1e-12
