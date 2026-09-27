"""ERA5 monthly-mean environment at storm positions: SST, 200-850 hPa vertical
wind shear and 600 hPa relative humidity.

Uses the *previous* month's mean (lag_months=1) so every value would have been
available operationally at forecast time - no look-ahead into the storm's own month.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERA5_DIR = os.path.join(ROOT, "data", "raw", "era5")
ENV_FEATURES = ["sst_c", "shear_ms", "rh600"]

_cache = {}


def _open(name):
    ds = xr.open_dataset(os.path.join(ERA5_DIR, name))
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    return ds


def load_env_grid():
    if "grid" in _cache:
        return _cache["grid"]
    grid = xr.Dataset({"sst_c": _open("era5_sst_monthly.nc")["sst"] - 273.15})
    if os.path.exists(os.path.join(ERA5_DIR, "era5_pl_monthly.nc")):
        pl = _open("era5_pl_monthly.nc")
        lev = "pressure_level" if "pressure_level" in pl.dims else "level"
        u, v = pl["u"], pl["v"]
        shear = np.hypot(u.sel({lev: 200}) - u.sel({lev: 850}), v.sel({lev: 200}) - v.sel({lev: 850}))
        grid["shear_ms"] = shear.drop_vars(lev, errors="ignore")
        grid["rh600"] = pl["r"].sel({lev: 600}).drop_vars(lev, errors="ignore")
    grid = grid.drop_vars([c for c in grid.coords if c not in ("time", "latitude", "longitude")])
    grid["time"] = pd.to_datetime(grid.time.values).to_period("M").to_timestamp()
    # SST is NaN over land: fill from nearest ocean cells so near-coast storms still get a value
    grid["sst_c"] = grid.sst_c.ffill("longitude").bfill("longitude").ffill("latitude").bfill("latitude")
    _cache["grid"] = grid.load()
    return _cache["grid"]


def available():
    return os.path.exists(os.path.join(ERA5_DIR, "era5_sst_monthly.nc"))


def available_features():
    return [k for k in ENV_FEATURES if k in load_env_grid()] if available() else []


def env_at(lat, lon, time, lag_months=1):
    """Nearest-gridpoint environment for arrays of lat, lon, time."""
    g = load_env_grid()
    month = (pd.to_datetime(pd.Series(time)).dt.to_period("M") - lag_months).dt.to_timestamp()
    month = month.clip(g.time.values.min(), g.time.values.max())
    pts = g.sel(time=xr.DataArray(month.values, dims="p"),
                latitude=xr.DataArray(np.asarray(lat, float), dims="p"),
                longitude=xr.DataArray(np.asarray(lon, float), dims="p"), method="nearest")
    return pd.DataFrame({k: pts[k].values for k in available_features()})
