"""IMD intensity classes and the Dvorak T-number <-> wind table.

Shared by the Colab training notebook and the backend so both use identical
definitions. Winds are in knots.
"""
import numpy as np

# IMD classification (3-min sustained wind, knots). Lower bounds, inclusive.
IMD_CLASSES = [
    ("D", "Depression", 17),
    ("DD", "Deep Depression", 28),
    ("CS", "Cyclonic Storm", 34),
    ("SCS", "Severe Cyclonic Storm", 48),
    ("VSCS", "Very Severe Cyclonic Storm", 64),
    ("ESCS", "Extremely Severe Cyclonic Storm", 90),
    ("SuCS", "Super Cyclonic Storm", 120),
]
IMD_CODES = [c[0] for c in IMD_CLASSES]
IMD_NAMES = [c[1] for c in IMD_CLASSES]
_IMD_LOWER = np.array([c[2] for c in IMD_CLASSES], dtype=float)

# TCIR / JTWC labels are 1-min sustained winds; IMD classes use 3-min winds.
# WMO guidance (Harper et al. 2010) gives ~0.93 for 1-min -> longer averaging
# periods at sea. Applied only when assigning IMD classes from 1-min winds.
WIND_1MIN_TO_IMD = 0.93

# Dvorak Current Intensity (CI/T-number) -> max sustained wind (1-min, kt).
DVORAK_T = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
DVORAK_WIND = np.array([25, 25, 30, 35, 45, 55, 65, 77, 90, 102, 115, 127, 140, 155, 170], dtype=float)


def imd_class_index(wind_imd_kt):
    """Index into IMD_CLASSES for a 3-min wind. Winds below 17 kt map to D (0)."""
    w = np.asarray(wind_imd_kt, dtype=float)
    idx = np.searchsorted(_IMD_LOWER, w, side="right") - 1
    return np.clip(idx, 0, len(IMD_CLASSES) - 1)


def imd_class_from_1min(wind_1min_kt):
    return imd_class_index(np.asarray(wind_1min_kt, dtype=float) * WIND_1MIN_TO_IMD)


def t_to_wind(t):
    """Dvorak T-number -> 1-min wind (kt), piecewise linear on the Dvorak table."""
    return np.interp(np.asarray(t, dtype=float), DVORAK_T, DVORAK_WIND)


def wind_to_t(wind_kt):
    """1-min wind (kt) -> Dvorak T-number. The table is flat at 25 kt for T1.0-1.5,
    so the inverse uses T1.5 upward and clamps weaker winds to T1.0-1.5 linearly."""
    w = np.asarray(wind_kt, dtype=float)
    t = np.interp(w, DVORAK_WIND[1:], DVORAK_T[1:])
    weak = w < 25
    return np.where(weak, np.clip(1.0 + 0.5 * w / 25.0, 1.0, 1.5), t)


def round_t(t):
    """Dvorak T-numbers are reported in 0.5 steps."""
    return np.round(np.asarray(t, dtype=float) * 2) / 2
