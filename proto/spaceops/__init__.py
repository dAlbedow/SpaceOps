"""SpaceOps prototype (milestone M0): physics, thermal, sensing, EMCON and salvo models.

All quantities are SI (metres, seconds, kilograms, watts, kelvin) in float64.
Randomness is seeded and local; no model reads the wall clock.
"""

__all__ = [
    "constants",
    "dynamics",
    "thermal",
    "orbits",
    "sensing",
    "emcon",
    "salvo",
    "scenario",
]
