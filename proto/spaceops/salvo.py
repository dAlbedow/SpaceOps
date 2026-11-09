"""Missiles, point defense and salvo Monte Carlo.

Pieces:
  * hit_probability: 2-D Gaussian miss distance against a lethal radius (closed form).
  * saturation_estimate: the first-order planning formula shown to the player.
  * volley_leak_mc: salvo vs layered defense fired in volleys, with a `coordination`
    parameter for how well shots are spread over surviving missiles (design doc section 9).
  * engage_timeline: time-resolved defender channels against arriving missiles.
  * launch_is_clear: friendly-fire check at launch (no debris is modeled, so this is where
    stray munitions are prevented).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .orbits import segment_blocked_by_sphere


# ---------------------------------------------------------------- terminal accuracy

def hit_probability(miss_sigma: float, lethal_radius: float) -> float:
    """P(miss distance <= lethal_radius) for isotropic 2-D Gaussian error of std dev
    miss_sigma per axis (Rayleigh distribution): 1 - exp(-r^2 / (2 sigma^2))."""
    if miss_sigma <= 0.0:
        return 1.0
    return float(1.0 - np.exp(-(lethal_radius**2) / (2.0 * miss_sigma**2)))


# ---------------------------------------------------------------- planning estimate

def saturation_estimate(channels: int, window_s: float, engage_s: float, pk: float) -> float:
    """Expected kills by a defender in a window: channels * (window / engage time) * pk."""
    return float(channels * (window_s / engage_s) * pk)


# ---------------------------------------------------------------- volley Monte Carlo

# Calibrated, not derived: the published example (6 missiles, Ph 0.34, 3/4/5 shots per missile
# leak 58/24/7 percent) states no formula. Perfectly even shot spreading gives about
# 50/18/5 and random spreading about 66/28/8; 0.4 reproduces the published figures.
COORDINATION = 0.4

@dataclass
class LeakResult:
    p_any_leak: float
    mean_leakers: float
    distribution: np.ndarray  # P(k leakers), k = 0..n_missiles


def volley_leak_mc(n_missiles: int, ph: float, volleys: int, shots_per_volley: int | None = None,
                   coordination: float = COORDINATION, trials: int = 20000, seed: int = 0) -> LeakResult:
    """Salvo of n_missiles against a defender who fires `volleys` rounds of
    `shots_per_volley` shots (default n_missiles), looking between rounds.

    Each shot is assigned to a surviving missile: with probability `coordination` to the
    missile with the fewest shots so far this volley (ties random), else to a random survivor.
    Each shot kills its missile with probability ph. Overkill is wasted. A leaker is a
    missile alive after the last volley.
    """
    rng = np.random.default_rng(seed)
    spv = n_missiles if shots_per_volley is None else shots_per_volley
    counts = np.zeros(n_missiles + 1)
    for _ in range(trials):
        alive = n_missiles
        for _v in range(volleys):
            if alive == 0:
                break
            assigned = np.zeros(alive, dtype=int)
            hits = np.zeros(alive, dtype=bool)
            for _s in range(spv):
                if rng.random() < coordination:
                    m = assigned.min()
                    cand = np.flatnonzero(assigned == m)
                    j = int(rng.choice(cand))
                else:
                    j = int(rng.integers(alive))
                assigned[j] += 1
                if rng.random() < ph:
                    hits[j] = True
            alive -= int(hits.sum())
        counts[alive] += 1
    dist = counts / trials
    return LeakResult(p_any_leak=float(1.0 - dist[0]),
                      mean_leakers=float((dist * np.arange(n_missiles + 1)).sum()),
                      distribution=dist)


def independent_leak_probability(n_missiles: int, ph: float, shots_per_missile: int) -> float:
    """Closed form for a single non-adaptive round of fixed shots per missile:
    P(at least one survives) = 1 - (1 - (1-ph)^k)^n."""
    s = (1.0 - ph) ** shots_per_missile
    return float(1.0 - (1.0 - s) ** n_missiles)


# ---------------------------------------------------------------- time-resolved defense

def engage_timeline(arrival_times: np.ndarray, channels: int, engage_s: float, pk: float,
                    first_engage_range_s: float, rng: np.random.Generator) -> dict:
    """Defender channels engage missiles in arrival order.

    A missile can be engaged from (arrival - first_engage_range_s) until it arrives. A channel
    that becomes free takes the earliest-arriving engageable missile not yet engaged; the
    engagement lasts engage_s and succeeds with probability pk. A missile still alive at its
    arrival time leaks. Missiles are engaged at most once (no re-shoot) in this simple model.
    """
    arrivals = np.sort(np.asarray(arrival_times, dtype=float))
    free_at = np.zeros(channels)
    killed = 0
    engaged = 0
    for a in arrivals:
        ch = int(np.argmin(free_at))
        start = max(free_at[ch], a - first_engage_range_s)
        end = start + engage_s
        if end <= a:  # finishes before it arrives
            free_at[ch] = end
            engaged += 1
            if rng.random() < pk:
                killed += 1
    n = len(arrivals)
    return {"incoming": n, "engaged": engaged, "killed": killed, "leakers": n - killed}


# ---------------------------------------------------------------- launch safety

def launch_is_clear(launcher_pos: np.ndarray, aim_point: np.ndarray,
                    friendlies: list[tuple[np.ndarray, float]], margin: float = 1.0) -> bool:
    """True if the straight launch path misses every friendly (position, radius) sphere.

    Missiles that miss are removed after leaving the engagement volume, so the only
    friendly-fire hazard is the launch path itself; the check is made once at launch.
    """
    for pos, radius in friendlies:
        if segment_blocked_by_sphere(launcher_pos, aim_point, pos, radius * margin):
            return False
    return True
