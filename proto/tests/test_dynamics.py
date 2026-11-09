import numpy as np

from spaceops import dynamics as dyn
from spaceops.constants import GM_EARTH


def approx(a, b, rel=1e-9, abs_=0.0):
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_)


def test_rocket_equation_known_value():
    # ve = 500 km/s, mass ratio 2 -> dv = 500 ln 2 = 346.574 km/s (design doc section 4)
    dv = dyn.delta_v(500e3, 2000.0, 1000.0)
    assert approx(dv, 500e3 * np.log(2.0))
    assert abs(dv - 346_574.0) < 1.0


def test_rocket_equation_round_trip():
    ve, m0, dv = 9000.0, 1.2e6, 3500.0
    m1 = dyn.final_mass(ve, m0, dv)
    assert approx(dyn.delta_v(ve, m0, m1), dv, rel=1e-12)
    assert approx(dyn.propellant_for_dv(ve, m0, dv), m0 - m1)


def test_design_doc_drive_table():
    # 1,000 t ship at 0.05 g: thrust 490,332.5 N. Jet power = F ve / 2.
    thrust = 1.0e6 * 0.05 * 9.80665
    cases = {4.4e3: 1.08e9, 9e3: 2.2e9, 50e3: 1.23e10, 500e3: 1.23e11}
    for ve, expected in cases.items():
        p = dyn.jet_power(thrust, ve)
        assert abs(p - expected) / expected < 0.03, (ve, p, expected)
    # propellant flow for the fusion case is about 0.98 kg/s
    assert abs(dyn.mass_flow(thrust, 500e3) - 0.98) < 0.01


def test_burn_closed_form_matches_rk4():
    ve, thrust = 8.0e4, 3.0e5
    body = dyn.Body(dyn.v3(1e6, -2e6, 3e5), dyn.v3(100.0, 50.0, -20.0), 4.0e5, 6.0e5,
                    thrust_max=thrust, ve=ve)
    d = dyn.v3(1.0, 2.0, 2.0)
    t = 1200.0
    x, v, m = dyn.burn_fixed_direction(body.pos, body.vel, body.mass, thrust, ve, d, t)
    num = dyn.propagate(body, t, dt=2.0, throttle=1.0, direction=d)
    assert np.allclose(num.pos, x, rtol=0, atol=1e-3), num.pos - x
    assert np.allclose(num.vel, v, rtol=0, atol=1e-6)
    assert approx(num.mass, m, rel=1e-12)
    # velocity change obeys the rocket equation exactly
    dv_vec = v - body.vel
    assert approx(float(np.linalg.norm(dv_vec)), dyn.delta_v(ve, body.mass, m), rel=1e-12)


def test_acceleration_rises_as_mass_burns():
    body = dyn.Body(dyn.v3(0, 0, 0), dyn.v3(0, 0, 0), 1e5, 9e5, thrust_max=1e6, ve=1e4)
    assert approx(body.accel_max(), 1.0)
    after = dyn.propagate(body, 600.0, dt=1.0, throttle=1.0, direction=dyn.v3(1, 0, 0))
    assert after.accel_max() > 1.0


def test_propellant_runs_out_and_thrust_stops():
    body = dyn.Body(dyn.v3(0, 0, 0), dyn.v3(0, 0, 0), 1000.0, 100.0, thrust_max=500.0, ve=2000.0)
    # tank empties after 100 kg * 2000 m/s / 500 N = 400 s
    out = dyn.propagate(body, 1000.0, dt=5.0, throttle=1.0, direction=dyn.v3(0, 1, 0))
    assert out.prop_mass == 0.0
    expected_dv = dyn.delta_v(2000.0, 1100.0, 1000.0)
    assert approx(out.vel[1], expected_dv, rel=1e-9)


def test_coast_is_straight_line():
    p, v = dyn.coast(dyn.v3(1, 2, 3), dyn.v3(10, -5, 0.5), 100.0)
    assert np.allclose(p, [1001, -498, 53])
    assert np.allclose(v, [10, -5, 0.5])


def test_flip_and_burn_closed_form():
    d, a = 3.0e8, 0.5
    t, vpk, dv = dyn.flip_and_burn(d, a)
    assert approx(t, 2 * np.sqrt(d / a))
    assert approx(dv, a * t)
    assert approx(vpk, dv / 2)
    # check by integrating: distance covered in two half-burns equals d
    half = t / 2
    assert approx(2 * 0.5 * a * half**2, d)


def test_lateral_displacement_small_at_long_range():
    # design doc: 0.05 g for 60 s moves a target about 900 m off its line
    assert abs(dyn.lateral_displacement(0.05 * 9.80665, 60.0) - 882.6) < 1.0


def test_closest_approach():
    rel_p = dyn.v3(1000.0, 100.0, 0.0)
    rel_v = dyn.v3(-100.0, 0.0, 0.0)
    t, d = dyn.closest_approach(rel_p, rel_v)
    assert approx(t, 10.0) and approx(d, 100.0)
    t2, d2 = dyn.closest_approach(rel_p, -rel_v)
    assert t2 < 0 and approx(d2, np.linalg.norm(rel_p))


def test_energy_conserved_in_gravity_coast():
    # circular orbit around Earth integrated by RK4 for one period
    r = 7.0e6
    v = np.sqrt(GM_EARTH / r)
    body = dyn.Body(dyn.v3(r, 0, 0), dyn.v3(0, v, 0), 100.0, 0.0)
    grav = dyn.point_mass_gravity(GM_EARTH)
    e0 = 0.5 * v * v - GM_EARTH / r
    period = 2 * np.pi * np.sqrt(r**3 / GM_EARTH)
    out = dyn.propagate(body, period, dt=1.0, gravity=grav)
    e1 = 0.5 * float(np.dot(out.vel, out.vel)) - GM_EARTH / float(np.linalg.norm(out.pos))
    assert abs(e1 - e0) / abs(e0) < 1e-9
    # back at the start after one period
    assert np.linalg.norm(out.pos - body.pos) < 1.0
