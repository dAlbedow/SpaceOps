"""EMCON states 0-3, scheduling, triggers, the cold-run clock and ESM intercept geometry.

State table (design doc section 7):
  0 Full        all sensors/links, free drive, radiators extended
  1 Restricted  passive + LPI bursts, laser links, limited burns, radiators extended
  2 Silent      passive only, no links, coast, radiators retracted (heat stored)
  3 Dark        minimal passive, no links, drive off, radiators retracted, reactor idle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

import numpy as np

from . import sensing
from .thermal import RadiatorState, ThermalNode


class Emcon(IntEnum):
    FULL = 0
    RESTRICTED = 1
    SILENT = 2
    DARK = 3


@dataclass(frozen=True)
class Policy:
    radar: bool
    laser_link: bool
    rf_link: bool
    drive_limit: float  # max throttle 0..1
    radiators_extended: bool
    reactor_level: float


POLICIES: dict[Emcon, Policy] = {
    Emcon.FULL: Policy(True, True, True, 1.0, True, 1.0),
    Emcon.RESTRICTED: Policy(False, True, False, 0.3, True, 1.0),
    Emcon.SILENT: Policy(False, False, False, 0.0, False, 1.0),
    Emcon.DARK: Policy(False, False, False, 0.0, False, 0.05),
}


@dataclass
class Trigger:
    """Switch to `to_state` when predicate(ctx) is true (one-shot)."""

    name: str
    predicate: Callable[[dict], bool]
    to_state: Emcon
    fired: bool = False


@dataclass
class EmconController:
    state: Emcon = Emcon.FULL
    schedule: list[tuple[float, Emcon]] = field(default_factory=list)  # (time, state)
    triggers: list[Trigger] = field(default_factory=list)
    forced_extension_time: float | None = None  # when the sink filled
    log: list[tuple[float, str]] = field(default_factory=list)
    _next: int = 0  # index of the next unapplied schedule entry

    def _set(self, t: float, s: Emcon, why: str) -> None:
        if s != self.state:
            self.log.append((t, f"{self.state.name}->{s.name}: {why}"))
            self.state = s
            if s in (Emcon.FULL, Emcon.RESTRICTED):
                self.forced_extension_time = None

    def policy(self) -> Policy:
        return POLICIES[self.state]

    def update(self, t: float, node: ThermalNode, ctx: dict | None = None) -> Policy:
        """Apply the schedule, triggers and the heat-sink rule; set the node's radiator state."""
        ctx = ctx or {}
        ordered = sorted(self.schedule)
        while self._next < len(ordered) and ordered[self._next][0] <= t:
            ts, s = ordered[self._next]
            self._next += 1
            self._set(t, s, f"scheduled at {ts:.0f}s")
        for tr in self.triggers:
            if not tr.fired and tr.predicate(ctx):
                tr.fired = True
                self._set(t, tr.to_state, f"trigger {tr.name}")
        pol = self.policy()
        extend = pol.radiators_extended
        if not extend and node.T >= node.T_safe:
            # cold-run clock expired: the ship must radiate, a bright predictable spike
            extend = True
            if self.forced_extension_time is None:
                self.forced_extension_time = t
                self.log.append((t, "heat sink full: radiators forced out"))
        node.radiators.state = RadiatorState.EXTENDED if extend else RadiatorState.RETRACTED
        return pol


def endurance_s(node: ThermalNode, heat_load_w: float) -> float:
    """The cold-run endurance meter: seconds until the safe temperature with radiators retracted."""
    saved = node.radiators.state
    node.radiators.state = RadiatorState.RETRACTED
    t = node.time_to_temperature(node.T_safe, heat_load_w)
    node.radiators.state = saved
    return t


def ship_ir_intensity(node: ThermalNode, hull_area: float = 2.0e3, hull_temp: float = 290.0,
                      plume_w: float = 0.0) -> float:
    """Infrared intensity (W/sr): hot radiators (effective area, all sides) + crewed hull + plume."""
    rad = node.radiators
    a_rad = rad.effective_area() * rad.sides
    return (sensing.ir_intensity(node.T, a_rad, rad.emissivity)
            + sensing.ir_intensity(hull_temp, hull_area)
            + sensing.plume_intensity(plume_w))


def esm_intercept_range(p_tx: float, gain: float, sensitivity_w_m2: float) -> float:
    """Range at which a one-way radar emission reaches an ESM receiver at its sensitivity.

    Power density = P G / (4 pi R^2), so intercept range scales as R^-2 while the radar's
    own detection range scales as R^-4: the emitter is heard long before it sees.
    """
    return float(np.sqrt(p_tx * gain / (4.0 * np.pi * sensitivity_w_m2)))
