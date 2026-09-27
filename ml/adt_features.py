"""ADT-lite features from a storm-centred IR brightness-temperature image.

Simplified versions of the quantities the Advanced Dvorak Technique reads off
IR imagery (eye temperature, coldest surrounding cloud ring, cold-cloud cover,
asymmetry). They are computed directly from brightness temperature, not
learned, so they can be used as RI-model inputs without label leakage.
"""
import numpy as np

FEATURE_NAMES = [
    "eye_bt", "cloud_bt", "eye_contrast", "min_bt", "mean_bt_200",
    "cold_frac_60", "cold_frac_70", "asym_std", "radial_warming",
]
SCENE_TYPES = ["EYE", "CDO", "CURVED_BAND", "SHEAR", "DISORGANIZED"]

_mask_cache = {}


def _radius_km(h, w, km_per_px):
    key = (h, w, round(km_per_px, 4))
    if key not in _mask_cache:
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = (h - 1) / 2, (w - 1) / 2
        r = np.hypot(yy - cy, xx - cx) * km_per_px
        az = np.arctan2(yy - cy, xx - cx)
        _mask_cache[key] = (r, az)
    return _mask_cache[key]


def adt_features(bt, km_per_px):
    """bt: (B, H, W) or (H, W) brightness temperature in K, storm at centre.
    Returns (B, len(FEATURE_NAMES)) float32 array."""
    bt = np.asarray(bt, dtype=np.float32)
    single = bt.ndim == 2
    if single:
        bt = bt[None]
    b, h, w = bt.shape
    r, az = _radius_km(h, w, km_per_px)
    flat = bt.reshape(b, -1)

    def sel(mask):
        return flat[:, mask.ravel()]

    eye_bt = sel(r <= 40).max(1)
    # ADT cloud region: warmest pixel on each ring 24-136 km, coldest such ring.
    ring_width = max(km_per_px, 4.0)
    ring_max = [sel((r >= r0) & (r < r0 + ring_width)).max(1)
                for r0 in np.arange(24, 136, ring_width)]
    cloud_bt = np.min(np.stack(ring_max, 1), 1)
    in200 = sel(r <= 200)
    min_bt = in200.min(1)
    mean_200 = in200.mean(1)
    cold60 = (in200 < 213.15).mean(1)
    cold70 = (in200 < 203.15).mean(1)
    ann = (r >= 50) & (r <= 200)
    sectors = np.stack([sel(ann & (az >= a) & (az < a + np.pi / 4)).mean(1)
                        for a in np.linspace(-np.pi, np.pi, 8, endpoint=False)], 1)
    asym = sectors.std(1)
    outer = sel((r >= 150) & (r <= 300))
    inner = sel(r <= 100)
    radial = (outer.mean(1) if outer.shape[1] else mean_200) - inner.mean(1)

    out = np.stack([eye_bt, cloud_bt, eye_bt - cloud_bt, min_bt, mean_200,
                    cold60, cold70, asym, radial], 1).astype(np.float32)
    return out[0] if single else out


def scene_type(f):
    """Heuristic Dvorak-style scene type from one feature vector."""
    d = dict(zip(FEATURE_NAMES, f))
    if d["eye_contrast"] >= 10 and d["cloud_bt"] <= 233:
        return "EYE"
    if d["cold_frac_60"] >= 0.5 and d["asym_std"] < 12:
        return "CDO"
    if d["asym_std"] >= 15 and d["cold_frac_60"] >= 0.1:
        return "SHEAR"
    if d["cold_frac_60"] >= 0.15:
        return "CURVED_BAND"
    return "DISORGANIZED"
