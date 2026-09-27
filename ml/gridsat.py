"""NOAA GridSat-B1 geostationary IR / water-vapour scenes over the North Indian Ocean.

GridSat is on a 0.07 deg grid with latitude ascending (row 0 = south). TCIR's IR
channel is GridSat itself (a crop matches with correlation 1.000), so 201x201
crops from these scenes are exactly the images the intensity model was trained on.
"""
import io

import numpy as np

URL = ("https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/"
       "{y}/GRIDSAT-B1.{y}.{m:02d}.{d:02d}.{h:02d}.v02r01.nc")
BOX = (0.0, 30.0, 40.0, 100.0)  # lat0, lat1, lon0, lon1
RES = 0.07
IR_RANGE = (180.0, 310.0)
WV_RANGE = (190.0, 260.0)
FILL = -31999


def url_for(ts):
    return URL.format(y=ts.year, m=ts.month, d=ts.day, h=ts.hour)


def bt_to_u8(bt, rng):
    lo, hi = rng
    out = np.clip((hi - bt) / (hi - lo), 0, 1) * 255
    return np.where(np.isfinite(bt), out, 0).astype(np.uint8)


def u8_to_bt(u8, rng):
    lo, hi = rng
    return hi - np.asarray(u8, np.float32) / 255.0 * (hi - lo)


def _read(var, i0, i1, j0, j1):
    raw = var[0, i0:i1, j0:j1].astype(np.float32)
    return np.where(raw == FILL, np.nan, raw * 0.01 + 200.0)


def fetch_scene(ts, session=None, timeout=300):
    """Download one GridSat file and return the NIO box as uint8 IR and WV (cold = bright) + axes."""
    import h5py
    import requests
    get = (session or requests).get
    r = get(url_for(ts), timeout=timeout)
    r.raise_for_status()
    with h5py.File(io.BytesIO(r.content), "r") as f:
        lat, lon = f["lat"][:], f["lon"][:]
        i0, i1 = np.searchsorted(lat, BOX[0]), np.searchsorted(lat, BOX[1])
        j0, j1 = np.searchsorted(lon, BOX[2]), np.searchsorted(lon, BOX[3])
        ir = _read(f["irwin_cdr"], i0, i1, j0, j1)
        wv = _read(f["irwvp"], i0, i1, j0, j1)
    return {"ir": bt_to_u8(ir, IR_RANGE), "wv": bt_to_u8(wv, WV_RANGE),
            "lat": lat[i0:i1].astype(np.float32), "lon": lon[j0:j1].astype(np.float32)}


def scene_axes():
    lat = np.round(np.arange(BOX[0] + RES / 2, BOX[1], RES), 3).astype(np.float32)
    lon = np.round(np.arange(BOX[2] + RES / 2, BOX[3], RES), 3).astype(np.float32)
    return lat, lon


def latlon_to_ij(lat, lon, lat_axis, lon_axis):
    return (np.abs(lat_axis - lat).argmin(), np.abs(lon_axis - lon).argmin())


def crop(img, i, j, size=201):
    """Centred size x size crop with edge padding (img rows = latitude ascending)."""
    h = size // 2
    pad = np.pad(img, ((h, h), (h, h)), mode="edge")
    return pad[i:i + size, j:j + size]
