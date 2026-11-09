"""Newtonian flight: rocket equation, powered and coasting motion, flip-and-burn planning.

Coasting is propagated analytically. Powered flight with a fixed thrust direction has a
closed form (used as a test oracle); general powered flight under gravity uses RK4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .constants import G0

Vec = np.ndarray


def v3(x: float, y: float, z: float) -> Vec:
    return np.array([x, y, z], dtype=np.float64)


def unit(v: Vec) -> Vec:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero vector has no direction")
    return v / n


# ---------------------------------------------------------------- rocket equation

def delta_v(exhaust_velocity: float, m0: float, m1: float) -> float:
    """Tsiolkovsky: dv = ve ln(m0/m1)."""
    if m1 <= 0.0 or m0 < m1:
        raise ValueError("need m0 >= m1 > 0")
    return exhaust_velocity * float(np.log(m0 / m1))


def final_mass(exhaust_velocity: float, m0: float, dv: float) -> float:
    return m0 * float(np.exp(-dv / exhaust_velocity))


def propellant_for_dv(exhaust_velocity: float, m0: float, dv: float) -> float:
    return m0 - final_mass(exhaust_velocity, m0, dv)


def isp_to_ve(isp_s: float) -> float:
    return isp_s * G0


def jet_power(thrust: float, exhaust_velocity: float) -> float:
    """Kinetic power of the exhaust, P = F ve / 2 (W)."""
    return 0.5 * thrust * exhaust_velocity


def mass_flow(thrust: float, exhaust_velocity: float) -> float:
    return thrust / exhaust_velocity


def thrust_for_power(jet_power_w: float, exhaust_velocity: float) -> float:
    return 2.0 * jet_power_w / exhaust_velocity


# ---------------------------------------------------------------- analytic motion

def coast(pos: Vec, vel: Vec, t: float) -> tuple[Vec, Vec]:
    """Straight-line coast with no gravity."""
    return pos + vel * t, vel.copy()


def burn_fixed_direction(
    pos: Vec, vel: Vec, m0: float, thrust: float, ve: float, direction: Vec, t: float
) -> tuple[Vec, Vec, float]:
    """Closed-form constant-thrust burn along a fixed direction, no gravity.

    v(t) = v0 + ve ln(m0/m) d,   x(t) = x0 + v0 t + ve (t - (m/mdot) ln(m0/m)) d
    Returns (pos, vel, mass). Raises if the burn would exhaust the mass.
    """
    d = unit(direction)
    mdot = mass_flow(thrust, ve)
    m = m0 - mdot * t
    if m <= 0.0:
        raise ValueError("burn consumes more than the whole mass")
    ln_ratio = float(np.log(m0 / m))
    v = vel + ve * ln_ratio * d
    x = pos + vel * t + ve * (t - (m / mdot) * ln_ratio) * d
    return x, v, m


@dataclass
class Body:
    """A point-mass vehicle with a reaction-mass tank."""

    pos: Vec
    vel: Vec
    dry_mass: float
    prop_mass: float
    thrust_max: float = 0.0
    ve: float = 1.0
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def mass(self) -> float:
        return self.dry_mass + self.prop_mass

    def delta_v_remaining(self) -> float:
        if self.prop_mass <= 0.0:
            return 0.0
        return delta_v(self.ve, self.mass, self.dry_mass)

    def accel_max(self) -> float:
        return self.thrust_max / self.mass if self.mass > 0 else 0.0


GravityFn = Callable[[Vec, float], Vec]


def point_mass_gravity(gm: float, centre: Vec | None = None) -> GravityFn:
    c = np.zeros(3) if centre is None else np.asarray(centre, dtype=np.float64)

    def g(pos: Vec, t: float) -> Vec:
        r = pos - c
        d = float(np.linalg.norm(r))
        return -gm * r / d**3

    return g


def _deriv(state: np.ndarray, t: float, thrust: float, ve: float, direction: Vec,
           gravity: GravityFn | None) -> np.ndarray:
    pos, vel, m = state[0:3], state[3:6], state[6]
    acc = thrust * direction / m
    if gravity is not None:
        acc = acc + gravity(pos, t)
    return np.concatenate([vel, acc, [-thrust / ve if thrust > 0 else 0.0]])


def rk4_step(state: np.ndarray, t: float, dt: float, thrust: float, ve: float,
             direction: Vec, gravity: GravityFn | None) -> np.ndarray:
    k1 = _deriv(state, t, thrust, ve, direction, gravity)
    k2 = _deriv(state + 0.5 * dt * k1, t + 0.5 * dt, thrust, ve, direction, gravity)
    k3 = _deriv(state + 0.5 * dt * k2, t + 0.5 * dt, thrust, ve, direction, gravity)
    k4 = _deriv(state + dt * k3, t + dt, thrust, ve, direction, gravity)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def propagate(body: Body, duration: float, dt: float, throttle: float = 0.0,
              direction: Vec | None = None, gravity: GravityFn | None = None,
              t0: float = 0.0) -> Body:
    """Propagate a body with constant throttle and direction using RK4.

    Thrust is clipped when the propellant runs out. The input body is not modified.
    """
    d = unit(direction) if direction is not None else v3(1, 0, 0)
    thrust = throttle * body.thrust_max
    state = np.concatenate([body.pos, body.vel, [body.mass]])
    t = t0
    end = t0 + duration
    while t < end - 1e-12:
        h = min(dt, end - t)
        prop_left = state[6] - body.dry_mass
        th = thrust
        if th > 0.0:
            if prop_left <= 0.0:
                th = 0.0
            else:
                # shorten the step if the tank would run dry mid-step
                burn_time_left = prop_left * body.ve / th
                h = min(h, burn_time_left)
        state = rk4_step(state, t, h, th, body.ve, d, gravity)
        t += h
    out = Body(state[0:3].copy(), state[3:6].copy(), body.dry_mass,
               max(0.0, float(state[6]) - body.dry_mass), body.thrust_max, body.ve,
               body.name, dict(body.meta))
    return out


# ---------------------------------------------------------------- manoeuvre planning

def flip_and_burn(distance: float, accel: float, v0: float = 0.0) -> tuple[float, float, float]:
    """Rest-to-rest brachistochrone along a line (accelerate half way, flip, decelerate).

    Returns (total_time, peak_speed, total_delta_v). v0 must be 0 for the closed form.
    """
    if v0 != 0.0:
        raise ValueError("closed form assumes rest-to-rest")
    t_half = float(np.sqrt(distance / accel))
    return 2.0 * t_half, accel * t_half, 2.0 * accel * t_half


def lateral_displacement(accel: float, t: float) -> float:
    """Offset a target gains from its nominal line after thrusting sideways for t."""
    return 0.5 * accel * t * t


def closest_approach(rel_pos: Vec, rel_vel: Vec) -> tuple[float, float]:
    """Time and distance of closest approach for straight-line relative motion.

    t_ca may be negative (already receding). Distance is evaluated at t_ca clipped to >= 0.
    """
    vv = float(np.dot(rel_vel, rel_vel))
    if vv == 0.0:
        return 0.0, float(np.linalg.norm(rel_pos))
    t_ca = -float(np.dot(rel_pos, rel_vel)) / vv
    t_eval = max(t_ca, 0.0)
    return t_ca, float(np.linalg.norm(rel_pos + rel_vel * t_eval))
