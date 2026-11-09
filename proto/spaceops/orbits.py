"""Analytic orbits, gravity, occultation, eclipse and planetary infrared background.

Bodies follow Keplerian two-body orbits (no n-body). A dominant body's gravity acts on ships
and missiles; the Sun is an optional second term. Spheres block line of sight; cylindrical
shadows model eclipse. Planets are also bright infrared backgrounds that mask signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .constants import SIGMA_SB, T_CMB

Vec = np.ndarray


# ---------------------------------------------------------------- Kepler

def kepler_period(a: float, gm: float) -> float:
    return float(2.0 * np.pi * np.sqrt(a**3 / gm))


def vis_viva_speed(r: float, a: float, gm: float) -> float:
    return float(np.sqrt(gm * (2.0 / r - 1.0 / a)))


def solve_kepler(M: float, e: float, tol: float = 1e-14) -> float:
    """Solve M = E - e sin E for E (elliptic, 0 <= e < 1) by Newton iteration."""
    M = float(np.mod(M, 2.0 * np.pi))
    E = M if e < 0.8 else np.pi
    for _ in range(100):
        d = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= d
        if abs(d) < tol:
            break
    return float(E)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


@dataclass
class KeplerOrbit:
    """Two-body elliptic orbit around a point that may itself move (parent_pos function)."""

    a: float
    e: float
    inc: float = 0.0
    raan: float = 0.0
    argp: float = 0.0
    M0: float = 0.0  # mean anomaly at epoch
    gm: float = 3.986004418e14
    epoch: float = 0.0
    parent_pos: Callable[[float], Vec] | None = None

    def mean_motion(self) -> float:
        return float(np.sqrt(self.gm / self.a**3))

    def period(self) -> float:
        return kepler_period(self.a, self.gm)

    def state_local(self, t: float) -> tuple[Vec, Vec]:
        """Position and velocity relative to the parent at time t."""
        n = self.mean_motion()
        M = self.M0 + n * (t - self.epoch)
        E = solve_kepler(M, self.e)
        cE, sE = np.cos(E), np.sin(E)
        b = self.a * np.sqrt(1.0 - self.e**2)
        r_pf = np.array([self.a * (cE - self.e), b * sE, 0.0])
        r = self.a * (1.0 - self.e * cE)
        # velocity in the perifocal frame: (a^2 n / r) * (-sin E, sqrt(1-e^2) cos E, 0)
        v_pf = (n * self.a * self.a / r) * np.array([-sE, np.sqrt(1.0 - self.e**2) * cE, 0.0])
        R = _rot_z(self.raan) @ _rot_x(self.inc) @ _rot_z(self.argp)
        return R @ r_pf, R @ v_pf

    def position(self, t: float) -> Vec:
        p, _ = self.state_local(t)
        if self.parent_pos is not None:
            p = p + self.parent_pos(t)
        return p

    def state(self, t: float) -> tuple[Vec, Vec]:
        p, v = self.state_local(t)
        if self.parent_pos is not None:
            pp = self.parent_pos(t)
            h = 1.0
            pv = (self.parent_pos(t + h) - self.parent_pos(t - h)) / (2 * h)
            return p + pp, v + pv
        return p, v


def circular_orbit(radius: float, gm: float, phase0: float = 0.0, inc: float = 0.0,
                   raan: float = 0.0, epoch: float = 0.0,
                   parent_pos: Callable[[float], Vec] | None = None) -> KeplerOrbit:
    return KeplerOrbit(a=radius, e=0.0, inc=inc, raan=raan, argp=0.0, M0=phase0, gm=gm,
                       epoch=epoch, parent_pos=parent_pos)


# ---------------------------------------------------------------- bodies and gravity

@dataclass
class Sphere:
    """A spherical body with a (possibly moving) centre."""

    name: str
    radius: float
    gm: float
    centre_fn: Callable[[float], Vec]
    ir_temperature: float = 250.0  # effective radiating temperature of the visible disk, K
    albedo: float = 0.3
    meta: dict = field(default_factory=dict)

    def centre(self, t: float) -> Vec:
        return self.centre_fn(t)

    @staticmethod
    def fixed(name: str, radius: float, gm: float, centre: Vec, **kw) -> "Sphere":
        c = np.asarray(centre, dtype=np.float64)
        return Sphere(name, radius, gm, lambda t, c=c: c, **kw)


def gravity_field(spheres: list[Sphere]) -> Callable[[Vec, float], Vec]:
    """Acceleration from the listed bodies (point masses) at (pos, t)."""

    def g(pos: Vec, t: float) -> Vec:
        a = np.zeros(3)
        for s in spheres:
            r = pos - s.centre(t)
            d = float(np.linalg.norm(r))
            a -= s.gm * r / d**3
        return a

    return g


# ---------------------------------------------------------------- occultation

def segment_blocked_by_sphere(p1: Vec, p2: Vec, centre: Vec, radius: float) -> bool:
    """True if the straight segment p1-p2 passes through the sphere's interior.

    Endpoints inside the sphere count as blocked. A grazing line (distance == radius) is not.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    d = p2 - p1
    L2 = float(np.dot(d, d))
    if L2 == 0.0:
        return float(np.linalg.norm(p1 - c)) < radius
    s = float(np.clip(np.dot(c - p1, d) / L2, 0.0, 1.0))
    closest = p1 + s * d
    return float(np.linalg.norm(closest - c)) < radius


def line_of_sight_clear(p1: Vec, p2: Vec, t: float, spheres: list[Sphere]) -> bool:
    for s in spheres:
        if segment_blocked_by_sphere(p1, p2, s.centre(t), s.radius):
            return False
    return True


# ---------------------------------------------------------------- eclipse

def in_cylindrical_shadow(pos: Vec, centre: Vec, radius: float, sun_dir: Vec) -> bool:
    """True if pos is inside the planet's cylindrical umbra.

    sun_dir is the unit vector from the planet toward the Sun. The shadow region is behind
    the planet (negative along sun_dir) and within one planet radius of the axis.
    """
    s = np.asarray(sun_dir, dtype=np.float64)
    s = s / np.linalg.norm(s)
    r = np.asarray(pos, dtype=np.float64) - np.asarray(centre, dtype=np.float64)
    along = float(np.dot(r, s))
    if along >= 0.0:
        return False
    perp = float(np.linalg.norm(r - along * s))
    return perp < radius


def circular_eclipse_fraction(r_orbit: float, R_planet: float) -> float:
    """Fraction of a circular orbit spent in cylindrical shadow (Sun in the orbit plane)."""
    return float(np.arcsin(R_planet / r_orbit) / np.pi)


# ---------------------------------------------------------------- planetary IR background

def angular_radius(distance_to_centre: float, radius: float) -> float:
    return float(np.arcsin(min(1.0, radius / distance_to_centre)))


def disk_solid_angle(distance_to_centre: float, radius: float) -> float:
    """Solid angle of a spherical body seen from outside (exact cone formula)."""
    return float(2.0 * np.pi * (1.0 - np.cos(angular_radius(distance_to_centre, radius))))


def blackbody_radiance(T: float, emissivity: float = 1.0) -> float:
    """Total radiance of a grey diffuse surface, W m^-2 sr^-1 = eps sigma T^4 / pi."""
    return emissivity * SIGMA_SB * T**4 / np.pi


def target_against_planet_disk(obs: Vec, target: Vec, centre: Vec, radius: float) -> bool:
    """True if the target is seen *in front of* the planet's disk (planet behind the target).

    That target sits against a bright infrared background. A target hidden behind the planet
    is blocked instead (see segment_blocked_by_sphere).
    """
    obs = np.asarray(obs, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    to_t = target - obs
    to_c = c - obs
    dist_c = float(np.linalg.norm(to_c))
    if dist_c <= radius:
        return False
    if float(np.dot(to_t, to_c)) <= 0.0:
        return False
    cosang = float(np.dot(to_t, to_c) / (np.linalg.norm(to_t) * dist_c))
    ang = float(np.arccos(np.clip(cosang, -1.0, 1.0)))
    if ang >= angular_radius(dist_c, radius):
        return False
    # the line of sight through the target hits the planet; the target is in front of it
    # exactly when the segment observer -> target does not already enter the sphere
    return not segment_blocked_by_sphere(obs, target, c, radius)


def background_radiance(obs: Vec, target: Vec, spheres: list[Sphere], t: float,
                        deep_space: float | None = None) -> float:
    """Infrared background radiance (W m^-2 sr^-1) behind a target as seen from obs.

    Deep space defaults to the cosmic microwave background (about 1e-6 W m^-2 sr^-1).
    """
    best = blackbody_radiance(T_CMB) if deep_space is None else deep_space
    for s in spheres:
        if target_against_planet_disk(obs, target, s.centre(t), s.radius):
            best = max(best, blackbody_radiance(s.ir_temperature))
    return best
