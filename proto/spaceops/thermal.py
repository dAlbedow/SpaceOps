"""Power bus, heat and radiators.

Single lumped coolant-loop node (hull + coolant + heat sink) with a temperature T and a
thermal mass C. Heat enters from the reactor conversion loss and from consumers; it leaves
only through radiators (Stefan-Boltzmann) or is stored by raising T. The node is advanced
with implicit Euler (unconditionally stable) and the time to a temperature limit is
computed by quadrature, which is also the test oracle for the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import integrate

from .constants import SIGMA_SB, T_CMB


# ---------------------------------------------------------------- radiators

def radiator_flux(T: float, emissivity: float = 0.9, T_env: float = 0.0) -> float:
    """Net radiated flux per unit area of one radiating face (W/m^2)."""
    return emissivity * SIGMA_SB * (T**4 - T_env**4)


def radiator_area_for_power(power_w: float, T: float, emissivity: float = 0.9,
                            sides: int = 1, T_env: float = 0.0) -> float:
    """Area (m^2, planform) needed to reject power_w at temperature T."""
    return power_w / (sides * radiator_flux(T, emissivity, T_env))


def equilibrium_temperature(Q: float, area: float, emissivity: float = 0.9,
                            T_env: float = 0.0, sides: int = 1) -> float:
    """Temperature at which radiated power equals heat input Q."""
    k = sides * emissivity * SIGMA_SB * area
    return float((Q / k + T_env**4) ** 0.25)


class RadiatorState(Enum):
    RETRACTED = "retracted"
    EXTENDED = "extended"
    DAMAGED = "damaged"


@dataclass
class Radiators:
    area: float  # planform area per side when extended, m^2
    emissivity: float = 0.9
    sides: int = 2
    retracted_fraction: float = 0.01  # residual effective area when retracted
    damage_fraction: float = 1.0  # fraction of area still working (1 = undamaged)
    state: RadiatorState = RadiatorState.EXTENDED

    def effective_area(self) -> float:
        if self.state is RadiatorState.RETRACTED:
            return self.area * self.retracted_fraction
        return self.area * self.damage_fraction

    def k(self) -> float:
        """Radiated power = k (T^4 - T_env^4)."""
        return self.sides * self.emissivity * SIGMA_SB * self.effective_area()

    def power(self, T: float, T_env: float = T_CMB) -> float:
        return self.k() * (T**4 - T_env**4)


# ---------------------------------------------------------------- bus

@dataclass
class Reactor:
    thermal_power: float  # W at 100 percent
    efficiency: float = 0.40  # thermal to electrical
    level: float = 1.0

    def electrical(self) -> float:
        return self.thermal_power * self.level * self.efficiency

    def conversion_waste_heat(self) -> float:
        return self.thermal_power * self.level * (1.0 - self.efficiency)


@dataclass
class Consumer:
    name: str
    power: float  # requested electrical power, W
    heat_fraction: float  # fraction of input power that ends up as ship heat
    priority: int  # lower number is served first
    enabled: bool = True


def allocate(supply: float, consumers: list[Consumer]) -> dict[str, float]:
    """Grant power in priority order; the first consumer that does not fit gets the rest."""
    granted: dict[str, float] = {}
    left = supply
    for c in sorted(consumers, key=lambda c: c.priority):
        want = c.power if c.enabled else 0.0
        g = min(want, max(left, 0.0))
        granted[c.name] = g
        left -= g
    return granted


def ship_heat_load(reactor: Reactor, consumers: list[Consumer],
                   granted: dict[str, float]) -> tuple[float, float]:
    """Return (heat_into_ship_W, power_leaving_ship_W).

    Energy bookkeeping: reactor thermal power = heat + outgoing power + unused electrical.
    Outgoing power is the non-heat part of consumer input (jet kinetic energy, laser beam,
    radar and comms emission).
    """
    heat = reactor.conversion_waste_heat()
    out = 0.0
    for c in consumers:
        g = granted[c.name]
        heat += c.heat_fraction * g
        out += (1.0 - c.heat_fraction) * g
    return heat, out


# ---------------------------------------------------------------- thermal node

@dataclass
class ThermalNode:
    """Lumped loop: thermal mass C (J/K), temperature T (K), limits and radiators."""

    C: float
    T: float
    radiators: Radiators
    T_safe: float = 900.0
    T_damage: float = 1100.0
    T_env: float = T_CMB
    damaged: bool = False
    energy_radiated: float = 0.0
    energy_in: float = 0.0

    @staticmethod
    def sink_capacity(sink_mass: float, joules_per_kg: float = 2.0e6) -> float:
        return sink_mass * joules_per_kg

    def step(self, Q: float, dt: float) -> float:
        """Implicit Euler step with heat input Q (W). Returns the new temperature."""
        k = self.radiators.k()
        C = self.C
        T0 = self.T
        Tn = T0
        for _ in range(60):
            f = C * (Tn - T0) / dt - Q + k * (Tn**4 - self.T_env**4)
            df = C / dt + 4.0 * k * Tn**3
            step = f / df
            Tn -= step
            if abs(step) < 1e-9 * max(Tn, 1.0):
                break
        p_rad = k * (Tn**4 - self.T_env**4)
        self.energy_radiated += p_rad * dt
        self.energy_in += Q * dt
        self.T = Tn
        if Tn >= self.T_damage:
            self.damaged = True
        return Tn

    def stored_energy(self, T_ref: float) -> float:
        return self.C * (self.T - T_ref)

    def radiated_power(self) -> float:
        return self.radiators.power(self.T, self.T_env)

    def throttle_limit(self, Q_demand: float) -> float:
        """Fraction of demanded heat load allowed once above the safe temperature."""
        if self.T < self.T_safe or Q_demand <= 0.0:
            return 1.0
        return float(min(1.0, max(0.0, self.radiated_power() / Q_demand)))

    # -- forecast
    def time_to_temperature(self, T_target: float, Q: float) -> float:
        """Seconds until T reaches T_target at constant Q (inf if it never does)."""
        if self.T >= T_target:
            return 0.0
        k = self.radiators.k()
        net_at_target = Q - k * (T_target**4 - self.T_env**4)
        if net_at_target <= 0.0:
            return float("inf")

        def integrand(T: float) -> float:
            return self.C / (Q - k * (T**4 - self.T_env**4))

        val, _ = integrate.quad(integrand, self.T, T_target, epsabs=1e-6, epsrel=1e-10, limit=200)
        return float(val)

    def forecast(self, Q: float) -> dict[str, float]:
        return {
            "time_to_safe_s": self.time_to_temperature(self.T_safe, Q),
            "time_to_damage_s": self.time_to_temperature(self.T_damage, Q),
            "equilibrium_K": (equilibrium_temperature(Q, self.radiators.effective_area(),
                                                      self.radiators.emissivity, self.T_env,
                                                      self.radiators.sides)
                              if self.radiators.k() > 0 else float("inf")),
        }
