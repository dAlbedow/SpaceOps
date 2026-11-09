import numpy as np
from scipy.integrate import solve_ivp

from spaceops import orbits as ob
from spaceops.constants import GM_EARTH, R_EARTH, SIGMA_SB, T_CMB


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b))


def test_kepler_period_leo():
    r = 7.0e6
    assert abs(ob.kepler_period(r, GM_EARTH) - 5828.5) < 1.0
    # Kepler's third law: T^2 proportional to a^3
    t1, t2 = ob.kepler_period(7e6, GM_EARTH), ob.kepler_period(14e6, GM_EARTH)
    assert rel(t2 / t1, 2 ** 1.5) < 1e-12


def test_solve_kepler_satisfies_equation():
    for e in (0.0, 0.1, 0.5, 0.9):
        for M in np.linspace(0.0, 6.2, 13):
            E = ob.solve_kepler(M, e)
            assert abs(E - e * np.sin(E) - np.mod(M, 2 * np.pi)) < 1e-12


def test_circular_orbit_radius_and_speed_constant():
    orb = ob.circular_orbit(7.0e6, GM_EARTH, phase0=0.7, inc=0.5, raan=1.0)
    for t in np.linspace(0, 6000, 25):
        p, v = orb.state_local(t)
        assert rel(np.linalg.norm(p), 7.0e6) < 1e-12
        assert rel(np.linalg.norm(v), np.sqrt(GM_EARTH / 7.0e6)) < 1e-12
        assert abs(np.dot(p, v)) < 1e-6 * np.linalg.norm(p) * np.linalg.norm(v)


def _ode_state(orb, t_end):
    p0, v0 = orb.state_local(0.0)

    def f(t, y):
        r = y[:3]
        return np.concatenate([y[3:], -GM_EARTH * r / np.linalg.norm(r) ** 3])

    sol = solve_ivp(f, (0.0, t_end), np.concatenate([p0, v0]), method="DOP853",
                    rtol=1e-12, atol=1e-6)
    return sol.y[:3, -1], sol.y[3:, -1]


def test_analytic_orbit_matches_independent_ode_solution():
    orb = ob.KeplerOrbit(a=1.2e7, e=0.35, inc=0.6, raan=0.4, argp=1.1, M0=0.3, gm=GM_EARTH)
    for t_end in (900.0, 5000.0, 20000.0):
        p, v = orb.state_local(t_end)
        po, vo = _ode_state(orb, t_end)
        assert np.linalg.norm(p - po) < 1e-3, np.linalg.norm(p - po)
        assert np.linalg.norm(v - vo) < 1e-6


def test_vis_viva_and_energy_conservation():
    orb = ob.KeplerOrbit(a=1.2e7, e=0.35, inc=0.2, gm=GM_EARTH)
    energy = -GM_EARTH / (2 * orb.a)
    for t in np.linspace(0, orb.period(), 17):
        p, v = orb.state_local(t)
        e_now = 0.5 * np.dot(v, v) - GM_EARTH / np.linalg.norm(p)
        assert rel(e_now, energy) < 1e-12
        assert rel(np.linalg.norm(v), ob.vis_viva_speed(np.linalg.norm(p), orb.a, GM_EARTH)) < 1e-12


def test_orbit_returns_after_one_period():
    orb = ob.KeplerOrbit(a=9.0e6, e=0.2, inc=0.3, raan=0.1, argp=0.2, M0=1.0, gm=GM_EARTH)
    p0, v0 = orb.state_local(0.0)
    p1, v1 = orb.state_local(orb.period())
    assert np.linalg.norm(p1 - p0) < 1e-3 and np.linalg.norm(v1 - v0) < 1e-6


def test_angular_momentum_constant():
    orb = ob.KeplerOrbit(a=1.1e7, e=0.4, inc=0.9, raan=0.3, argp=2.0, gm=GM_EARTH)
    h0 = np.cross(*orb.state_local(0.0))
    for t in (500.0, 3000.0, 9000.0):
        p, v = orb.state_local(t)
        assert np.linalg.norm(np.cross(p, v) - h0) / np.linalg.norm(h0) < 1e-12


def test_segment_sphere_blocking_cases():
    c, R = np.zeros(3), 1000.0
    # straight through the centre
    assert ob.segment_blocked_by_sphere([-5000, 0, 0], [5000, 0, 0], c, R)
    # passing well outside
    assert not ob.segment_blocked_by_sphere([-5000, 1500, 0], [5000, 1500, 0], c, R)
    # just inside / just outside the limb
    assert ob.segment_blocked_by_sphere([-5000, 999, 0], [5000, 999, 0], c, R)
    assert not ob.segment_blocked_by_sphere([-5000, 1001, 0], [5000, 1001, 0], c, R)
    # segment ends before reaching the sphere
    assert not ob.segment_blocked_by_sphere([-5000, 0, 0], [-2000, 0, 0], c, R)
    # an endpoint inside counts as blocked
    assert ob.segment_blocked_by_sphere([0, 0, 0], [5000, 0, 0], c, R)


def test_ship_hidden_behind_moon_is_blocked():
    moon = ob.Sphere.fixed("moon", 1.7374e6, 4.9e12, [0, 0, 0])
    a = np.array([-1.0e7, 0, 0])
    b_hidden = np.array([1.0e7, 0, 0])
    b_open = np.array([1.0e7, 5.0e6, 0])
    assert not ob.line_of_sight_clear(a, b_hidden, 0.0, [moon])
    assert ob.line_of_sight_clear(a, b_open, 0.0, [moon])


def test_cylindrical_shadow_geometry():
    c, R, s = np.zeros(3), 1000.0, np.array([1.0, 0, 0])
    assert ob.in_cylindrical_shadow([-5000, 0, 0], c, R, s)
    assert not ob.in_cylindrical_shadow([-5000, 1500, 0], c, R, s)
    assert not ob.in_cylindrical_shadow([5000, 0, 0], c, R, s)  # sunward side


def test_leo_eclipse_fraction_matches_closed_form():
    r = 7.0e6
    orb = ob.circular_orbit(r, GM_EARTH)
    sun = np.array([1.0, 0.0, 0.0])
    n = 20000
    inside = 0
    T = orb.period()
    for k in range(n):
        p, _ = orb.state_local(T * (k + 0.5) / n)
        inside += ob.in_cylindrical_shadow(p, np.zeros(3), R_EARTH, sun)
    frac = inside / n
    assert abs(frac - ob.circular_eclipse_fraction(r, R_EARTH)) < 2e-4, frac


def test_planet_disk_geometry():
    d, R = 1.0e7, 1.0e6
    assert rel(ob.angular_radius(d, R), np.arcsin(0.1)) < 1e-12
    # small-angle limit of the solid angle is pi (R/d)^2
    d_far = 1.0e9
    assert rel(ob.disk_solid_angle(d_far, R), np.pi * (R / d_far) ** 2) < 1e-5
    assert rel(ob.blackbody_radiance(300.0), SIGMA_SB * 300.0**4 / np.pi) < 1e-12


def test_target_in_front_of_planet_disk_sees_bright_background():
    planet = ob.Sphere.fixed("planet", 1.0e6, 1e13, [1.0e8, 0, 0], ir_temperature=280.0)
    obs = np.zeros(3)
    in_front = np.array([5.0e7, 0.0, 0.0])  # nearer than the planet, on the axis
    off_disk = np.array([5.0e7, 2.0e7, 0.0])
    behind = np.array([2.0e8, 0.0, 0.0])  # beyond the planet: blocked, not backlit
    bg = ob.background_radiance(obs, in_front, [planet], 0.0)
    assert rel(bg, ob.blackbody_radiance(280.0)) < 1e-12
    assert ob.background_radiance(obs, off_disk, [planet], 0.0) == ob.blackbody_radiance(T_CMB)
    assert not ob.line_of_sight_clear(obs, behind, 0.0, [planet])


def test_moving_body_centre_and_two_body_gravity_field():
    earth = ob.Sphere.fixed("earth", R_EARTH, GM_EARTH, [0, 0, 0])
    g = ob.gravity_field([earth])
    pos = np.array([7.0e6, 0.0, 0.0])
    a = g(pos, 0.0)
    assert rel(np.linalg.norm(a), GM_EARTH / 7.0e6**2) < 1e-12
    assert a[0] < 0
