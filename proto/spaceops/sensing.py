"""Signatures, SNR detection, light delay and tracking.

Everything is closed form where possible so tests can check it:
  radar:   SNR = Pt G^2 lam^2 sigma / ((4 pi)^3 R^4 k T B F L)   (R^-4)
  passive: SNR = (I / R^2) / NEI,  I in W/sr                       (R^-2)
Targets are observed at the retarded time (light delay), never at the present time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

import numpy as np

from .constants import C, K_B, SIGMA_SB

Vec = np.ndarray


# ---------------------------------------------------------------- signatures

def ir_intensity(T: float, area: float, emissivity: float = 0.9) -> float:
    """Radiant intensity (W/sr) of a grey Lambertian surface, normal view: eps sigma T^4 A / pi."""
    return emissivity * SIGMA_SB * T**4 * area / np.pi


def plume_intensity(waste_power_w: float) -> float:
    """Isotropic-equivalent intensity (W/sr) of a drive plume radiating waste_power_w."""
    return waste_power_w / (4.0 * np.pi)


def rcs_sphere(radius: float) -> float:
    """Optical-regime radar cross section of a conducting sphere: pi r^2 (m^2)."""
    return float(np.pi * radius**2)


def reflected_optical_intensity(radius: float, geometric_albedo: float, sun_irradiance: float,
                                in_shadow: bool = False) -> float:
    """Reflected sunlight from a sphere at full phase (W/sr): p E r^2, where p is the
    geometric albedo and E the solar irradiance at the target (W/m^2). Zero in eclipse."""
    if in_shadow:
        return 0.0
    return float(geometric_albedo * sun_irradiance * radius**2)


# ---------------------------------------------------------------- detection

def radar_snr(p_tx: float, gain: float, wavelength: float, rcs: float, r: float,
              bandwidth: float = 1.0e6, noise_temp: float = 500.0, noise_fig: float = 2.0,
              loss: float = 2.0) -> float:
    """Single-pulse-equivalent radar SNR (linear)."""
    num = p_tx * gain**2 * wavelength**2 * rcs
    den = (4.0 * np.pi) ** 3 * r**4 * K_B * noise_temp * bandwidth * noise_fig * loss
    return float(num / den)


def radar_range_for_snr(p_tx: float, gain: float, wavelength: float, rcs: float, snr: float,
                        bandwidth: float = 1.0e6, noise_temp: float = 500.0,
                        noise_fig: float = 2.0, loss: float = 2.0) -> float:
    """Closed-form range at which radar_snr == snr."""
    num = p_tx * gain**2 * wavelength**2 * rcs
    den = (4.0 * np.pi) ** 3 * snr * K_B * noise_temp * bandwidth * noise_fig * loss
    return float((num / den) ** 0.25)


def passive_snr(intensity: float, r: float, nei: float, background_radiance: float = 0.0,
                pixel_sr: float = 1.0e-10) -> float:
    """Passive IR SNR. NEI is the noise-equivalent irradiance (W/m^2); the background adds
    irradiance noise equal to radiance times the pixel solid angle, in quadrature."""
    noise = float(np.hypot(nei, background_radiance * pixel_sr))
    return float((intensity / r**2) / noise)


def passive_range_for_snr(intensity: float, snr: float, nei: float) -> float:
    """Closed-form passive range (no background) at which SNR == snr."""
    return float(np.sqrt(intensity / (snr * nei)))


def detection_probability(snr: float, threshold: float = 5.0) -> float:
    """Probability of detection against a threshold for a Gaussian noise model
    (unit-variance noise, signal = snr): P(n + snr > threshold)."""
    from scipy.special import ndtr
    return float(ndtr(snr - threshold))


# ---------------------------------------------------------------- light delay

def retarded_time(observer_pos: Vec, t_obs: float, target_pos_fn: Callable[[float], Vec],
                  iterations: int = 6) -> float:
    """Emission time t_r < t_obs with |x_target(t_r) - x_obs| = c (t_obs - t_r).

    Fixed-point iteration converges geometrically with ratio v/c.
    """
    obs = np.asarray(observer_pos, dtype=np.float64)
    t_r = t_obs
    for _ in range(iterations):
        d = float(np.linalg.norm(np.asarray(target_pos_fn(t_r)) - obs))
        t_r = t_obs - d / C
    return float(t_r)


def light_delay(observer_pos: Vec, t_obs: float, target_pos_fn: Callable[[float], Vec]) -> float:
    return t_obs - retarded_time(observer_pos, t_obs, target_pos_fn)


# ---------------------------------------------------------------- tracking

class TrackQuality(IntEnum):
    NONE = 0
    DETECTION = 1  # something is there (bearing / rough position)
    TRACK = 2  # good enough to maneuver against
    FIRE_CONTROL = 3  # good enough to launch / guide


def measurement_covariance(observer_pos: Vec, target_pos: Vec, sigma_angle: float,
                           sigma_range: float) -> np.ndarray:
    """3x3 position covariance: sigma_range along the line of sight, sigma_angle*R across it."""
    d = np.asarray(target_pos, dtype=np.float64) - np.asarray(observer_pos, dtype=np.float64)
    R = float(np.linalg.norm(d))
    u = d / R
    P_los = np.outer(u, u)
    P_perp = np.eye(3) - P_los
    return sigma_range**2 * P_los + (sigma_angle * R) ** 2 * P_perp


@dataclass
class Track:
    """Constant-velocity Kalman filter on (position, velocity), with process noise q (m/s^2)."""

    x: np.ndarray  # 6
    P: np.ndarray  # 6x6
    t: float
    q: float = 0.05
    n_updates: int = 0

    @staticmethod
    def initiate(pos: Vec, R_pos: np.ndarray, t: float, vel_sigma: float = 1.0e4,
                 q: float = 0.05) -> "Track":
        x = np.concatenate([np.asarray(pos, dtype=np.float64), np.zeros(3)])
        P = np.zeros((6, 6))
        P[:3, :3] = R_pos
        P[3:, 3:] = np.eye(3) * vel_sigma**2
        return Track(x=x, P=P, t=t, q=q)

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt == 0.0:
            return
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        # continuous white-noise acceleration
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * dt**3 / 3.0
        Q[:3, 3:] = np.eye(3) * dt**2 / 2.0
        Q[3:, :3] = Q[:3, 3:]
        Q[3:, 3:] = np.eye(3) * dt
        Q *= self.q**2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, z: Vec, R_pos: np.ndarray, t: float) -> None:
        self.predict(t)
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        S = H @ self.P @ H.T + R_pos
        K = self.P @ H.T @ np.linalg.inv(S)
        y = np.asarray(z, dtype=np.float64) - H @ self.x
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_pos @ K.T  # Joseph form
        self.n_updates += 1

    def position(self) -> Vec:
        return self.x[:3].copy()

    def velocity(self) -> Vec:
        return self.x[3:].copy()

    def position_covariance(self) -> np.ndarray:
        return self.P[:3, :3].copy()

    def ellipsoid_semi_axes(self, n_sigma: float = 1.0) -> np.ndarray:
        """Semi-axes (m) of the n-sigma position error ellipsoid, largest first."""
        w = np.linalg.eigvalsh(self.position_covariance())
        return n_sigma * np.sqrt(np.maximum(w, 0.0))[::-1]

    def mahalanobis_sq(self, truth_pos: Vec) -> float:
        e = np.asarray(truth_pos, dtype=np.float64) - self.x[:3]
        return float(e @ np.linalg.solve(self.position_covariance(), e))


def classify_track(track: Track | None, track_limit_m: float = 5.0e5,
                   fire_limit_m: float = 5.0e3, min_updates: int = 3) -> TrackQuality:
    """Quality from the largest 1-sigma semi-axis of the position error ellipsoid."""
    if track is None:
        return TrackQuality.NONE
    if track.n_updates < min_updates:
        return TrackQuality.DETECTION
    big = float(track.ellipsoid_semi_axes()[0])
    if big <= fire_limit_m:
        return TrackQuality.FIRE_CONTROL
    if big <= track_limit_m:
        return TrackQuality.TRACK
    return TrackQuality.DETECTION
