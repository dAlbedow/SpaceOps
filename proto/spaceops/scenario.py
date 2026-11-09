"""Two-ship duel around a planet: the M0 integration scenario.

Ship A ("Shade") sits dark in a low circular orbit: cold hull, radiators retracted, passive
sensor only. Ship B ("Salvo") falls inward from far out with its drive burning, radiators
out and radar on, so it is loud in every channel. Each side sees the other only through
retarded-time observations, line of sight past the planet, and the planet's infrared disk.
Deterministic for a given seed. Documented simplifications: passive IR gives a range with a
2 percent error (angle-only ranging is not modeled), one planet, constant-velocity tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import emcon as em
from . import orbits as ob
from . import salvo as sv
from . import sensing as se
from . import thermal as th
from .constants import GM_EARTH, R_EARTH
from .dynamics import Body, propagate, unit

NEI = 1.0e-9  # W/m^2 noise-equivalent irradiance of the passive IR sensors
PIXEL_SR = 1.0e-10
PASSIVE_SNR = 10.0
RADAR_SNR = 13.0
RADAR = dict(p_tx=1.0e7, gain=1.0e5, wavelength=0.03)
SHIP_RCS = 200.0
FIRE_LIMIT_M = 5.0e4  # largest 1-sigma semi-axis accepted for fire control
LAUNCH_RANGE_M = 1.2e7  # missile reach: 8 km/s average closing speed, about 25 minutes of flight


@dataclass
class DuelResult:
    t: np.ndarray
    pos_a: np.ndarray
    pos_b: np.ndarray
    temp_a: np.ndarray
    temp_b: np.ndarray
    range_ab: np.ndarray
    passive_range_a_sees_b: np.ndarray  # detection range of A's passive sensor against B (m)
    passive_range_b_sees_a: np.ndarray
    radar_range_b: float
    a_quality: np.ndarray  # TrackQuality of A's track on B (int)
    b_quality: np.ndarray
    a_detects_b: np.ndarray
    b_detects_a: np.ndarray
    blocked: np.ndarray  # planet blocks line of sight
    events: list[tuple[float, str]] = field(default_factory=list)
    salvo: dict = field(default_factory=dict)
    planet_radius: float = R_EARTH


B_START = np.array([3.0e7, 8.0e6, 0.0])
B_SPEED = 4.0e3
B_PERIAPSIS = 9.5e6  # design periapsis radius of the coasting orbit, m
A_PHASE0 = 2.74  # offline scan: closest approach about 1.97e6 m near t = 6730 s


def _b_initial_velocity() -> np.ndarray:
    """Velocity at B_START (in the orbital plane) of a bound orbit with periapsis B_PERIAPSIS."""
    r0 = float(np.linalg.norm(B_START))
    energy = 0.5 * B_SPEED**2 - GM_EARTH / r0
    vp = np.sqrt(2.0 * (energy + GM_EARTH / B_PERIAPSIS))
    v_t = B_PERIAPSIS * vp / r0
    v_r = -np.sqrt(B_SPEED**2 - v_t**2)
    ru = B_START / r0
    tu = np.cross([0.0, 0.0, 1.0], ru)
    return v_r * ru + v_t * tu


def _new_ship_b() -> Body:
    """Ship B: 2500 t dry, 500 t propellant, ve 50 km/s, 0.005 g at full thrust."""
    return Body(pos=B_START.copy(), vel=_b_initial_velocity(),
                dry_mass=2.5e6, prop_mass=5.0e5, thrust_max=1.2e5, ve=5.0e4)


def _interp_pos(t_hist, p_hist, n):
    def fn(tt):
        return np.array([np.interp(tt, t_hist[:n], p_hist[:n, i]) for i in range(3)])
    return fn


def run_duel(seed: int = 1, dt: float = 10.0, duration: float = 12000.0) -> DuelResult:
    rng = np.random.default_rng(seed)
    planet = ob.Sphere.fixed("planet", R_EARTH, GM_EARTH, [0, 0, 0], ir_temperature=280.0)
    grav = ob.gravity_field([planet])
    orbit_a = ob.circular_orbit(8.0e6, GM_EARTH, phase0=A_PHASE0, inc=0.0)

    body_b = _new_ship_b()
    node_a = th.ThermalNode(C=2.0e11 / 400.0, T=300.0, T_safe=700.0, T_damage=900.0,
                            radiators=th.Radiators(area=2.0e4, sides=2,
                                                   state=th.RadiatorState.RETRACTED,
                                                   retracted_fraction=0.0))
    node_b = th.ThermalNode(C=2.0e11 / 400.0, T=700.0, T_safe=900.0, T_damage=1100.0,
                            radiators=th.Radiators(area=2.0e4, sides=2,
                                                   state=th.RadiatorState.EXTENDED))
    ctrl_a = em.EmconController(state=em.Emcon.DARK)
    ctrl_b = em.EmconController(state=em.Emcon.FULL)

    n = int(duration / dt) + 1
    t = np.arange(n) * dt
    pos_a = np.zeros((n, 3))
    pos_b = np.zeros((n, 3))
    temp_a = np.zeros(n)
    temp_b = np.zeros(n)
    rng_ab = np.zeros(n)
    pr_a = np.zeros(n)
    pr_b = np.zeros(n)
    qa = np.zeros(n, dtype=int)
    qb = np.zeros(n, dtype=int)
    det_a = np.zeros(n, dtype=bool)
    det_b = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    events: list[tuple[float, str]] = []

    r_radar = se.radar_range_for_snr(rcs=SHIP_RCS, snr=RADAR_SNR, **RADAR)
    track_a: se.Track | None = None  # A's track on B
    track_b: se.Track | None = None
    launched = {"A": None, "B": None}
    burn_s = 3600.0

    for k in range(n):
        tk = t[k]
        pos_a[k] = orbit_a.position(tk)
        pos_b[k] = body_b.pos

        # heat and EMCON
        burning = tk < burn_s and body_b.prop_mass > 0
        q_b = 1.0e9 if burning else 4.0e8
        q_a = 1.5e7
        ctrl_a.update(tk, node_a)
        ctrl_b.update(tk, node_b)
        node_a.step(q_a, dt)
        node_b.step(q_b, dt)
        temp_a[k], temp_b[k] = node_a.T, node_b.T

        # signatures
        plume_w = 0.9e9 if burning else 0.0
        i_a = em.ship_ir_intensity(node_a, hull_area=2.0e3, hull_temp=290.0)
        i_b = em.ship_ir_intensity(node_b, hull_area=2.0e3, hull_temp=290.0, plume_w=plume_w)
        pr_a[k] = se.passive_range_for_snr(i_b, PASSIVE_SNR, NEI)  # A looking at B
        pr_b[k] = se.passive_range_for_snr(i_a, PASSIVE_SNR, NEI)  # B looking at A

        # light-delayed observation
        fn_b = _interp_pos(t, pos_b, k + 1)
        t_ret_b = se.retarded_time(pos_a[k], tk, fn_b) if k > 0 else tk
        seen_b = fn_b(max(t_ret_b, 0.0)) if k > 0 else pos_b[k]
        t_ret_a = se.retarded_time(pos_b[k], tk, orbit_a.position)
        seen_a = orbit_a.position(t_ret_a)
        rng_ab[k] = float(np.linalg.norm(pos_b[k] - pos_a[k]))

        clear = ob.line_of_sight_clear(pos_a[k], seen_b, tk, [planet])
        blocked[k] = not clear

        # A: passive detection of B
        if clear:
            r = float(np.linalg.norm(seen_b - pos_a[k]))
            bg = ob.background_radiance(pos_a[k], seen_b, [planet], tk)
            snr = se.passive_snr(i_b, r, NEI, bg, PIXEL_SR)
            if snr >= PASSIVE_SNR:
                det_a[k] = True
                R = se.measurement_covariance(pos_a[k], seen_b, 2e-5, 0.02 * r)
                z = seen_b + rng.multivariate_normal(np.zeros(3), R)
                if track_a is None:
                    track_a = se.Track.initiate(z, R, tk, q=0.01)
                else:
                    track_a.update(z, R, tk)
        if track_a is not None and not det_a[k]:
            track_a.predict(tk)

        # B: radar (if emitting) and passive IR detection of A
        if clear:
            r = float(np.linalg.norm(seen_a - pos_b[k]))
            z_cov = None
            if ctrl_b.policy().radar and se.radar_snr(r=r, rcs=SHIP_RCS, **RADAR) >= RADAR_SNR:
                z_cov = se.measurement_covariance(pos_b[k], seen_a, 1e-5, 20.0)
            else:
                bg = ob.background_radiance(pos_b[k], seen_a, [planet], tk)
                if se.passive_snr(i_a, r, NEI, bg, PIXEL_SR) >= PASSIVE_SNR:
                    z_cov = se.measurement_covariance(pos_b[k], seen_a, 2e-5, 0.02 * r)
            if z_cov is not None:
                det_b[k] = True
                z = seen_a + rng.multivariate_normal(np.zeros(3), z_cov)
                if track_b is None:
                    track_b = se.Track.initiate(z, z_cov, tk, q=0.01)
                else:
                    track_b.update(z, z_cov, tk)
        if track_b is not None and not det_b[k]:
            track_b.predict(tk)

        qa[k] = int(se.classify_track(track_a, fire_limit_m=FIRE_LIMIT_M))
        qb[k] = int(se.classify_track(track_b, fire_limit_m=FIRE_LIMIT_M))
        if k and qa[k] != qa[k - 1]:
            events.append((tk, f"A track on B -> {se.TrackQuality(qa[k]).name}"))
        if k and qb[k] != qb[k - 1]:
            events.append((tk, f"B track on A -> {se.TrackQuality(qb[k]).name}"))
        if k and det_a[k] != det_a[k - 1]:
            events.append((tk, "A passive contact " + ("gained" if det_a[k] else "lost")))
        if k and blocked[k] != blocked[k - 1]:
            events.append((tk, "planet " + ("blocks" if blocked[k] else "clears") + " line of sight"))

        # launches (friendly-fire check is trivial in a duel)
        if launched["A"] is None and qa[k] == se.TrackQuality.FIRE_CONTROL and rng_ab[k] < LAUNCH_RANGE_M:
            launched["A"] = tk
            events.append((tk, "A launches 6 missiles"))
        if launched["B"] is None and qb[k] == se.TrackQuality.FIRE_CONTROL and rng_ab[k] < LAUNCH_RANGE_M:
            launched["B"] = tk
            events.append((tk, "B launches 6 missiles"))

        # advance ship B
        if k + 1 < n:
            direction = unit(body_b.vel)  # prograde burn
            body_b = propagate(body_b, dt, dt / 5.0, throttle=1.0 if burning else 0.0,
                      direction=direction, gravity=grav, t0=tk)

    salvo = _resolve_salvos(launched, rng_ab, t, seed)
    for shooter, res in salvo.items():
        events.append((res["launch_t"], f"salvo from {shooter}: {res['mean_leakers']:.2f} mean "
                       f"leakers, P(any leak) = {res['p_any_leak']:.2f}"))
    events.sort()
    return DuelResult(t, pos_a, pos_b, temp_a, temp_b, rng_ab, pr_a, pr_b, r_radar, qa, qb,
                      det_a, det_b, blocked, events, salvo)


# defender point-defense suites: (channels, seconds per engagement, kill probability)
DEFENSE = {"A": (2, 6.0, 0.6), "B": (4, 6.0, 0.7)}  # the dark ship has the lighter suite


def _resolve_salvos(launched, rng_ab, t, seed: int) -> dict[str, dict]:
    """Each launching side fires 6 missiles into the other's point defense (time-resolved
    engagement, Monte Carlo over arrival jitter and kill rolls)."""
    out: dict[str, dict] = {}
    for shooter, t0 in launched.items():
        if t0 is None:
            continue
        defender = "B" if shooter == "A" else "A"
        channels, engage_s, pk = DEFENSE[defender]
        k0 = int(np.searchsorted(t, t0))
        flight = float(rng_ab[k0]) / 8.0e3  # average closing speed 8 km/s
        rng = np.random.default_rng(seed + 1000 + (0 if shooter == "A" else 1))
        leakers = []
        for _ in range(4000):
            arrivals = t0 + flight + np.arange(6) * 1.5 + rng.normal(0.0, 0.3, 6)
            leakers.append(sv.engage_timeline(arrivals, channels, engage_s, pk,
                                              first_engage_range_s=40.0, rng=rng)["leakers"])
        leakers = np.array(leakers)
        out[shooter] = {"defender": defender, "launch_t": float(t0), "flight_s": flight,
                        "p_any_leak": float((leakers > 0).mean()),
                        "mean_leakers": float(leakers.mean()),
                        "distribution": np.bincount(leakers, minlength=7) / len(leakers)}
    return out
