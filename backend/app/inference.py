"""Loads every trained model once and exposes inference helpers."""
import base64
import io
import json
import os
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ml import era5_env
from ml.adt_features import FEATURE_NAMES, adt_features, scene_type
from ml.cyclone_scale import IMD_CODES, IMD_NAMES, imd_class_index, round_t, t_to_wind, WIND_1MIN_TO_IMD
from ml.ibtracs import LEADS_H, build_features, load_ibtracs
from ml.train_track import TrackNet, cone_radius_km, sequence_inputs, to_latlon, POS_SCALE, WIND_SCALE

from .config import WEIGHTS

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))


def _json(name):
    return json.load(open(os.path.join(WEIGHTS, name)))


# ---------- tabular models (RI, genesis) ----------
class CalibratedLGBM:
    def __init__(self, model_file, cal_file, features):
        self.booster = lgb.Booster(model_file=os.path.join(WEIGHTS, model_file))
        cal = _json(cal_file)
        self.a, self.b = cal["coef"][0], cal["intercept"][0]
        self.features = features
        self._explainer = None

    def prob(self, df):
        p = np.clip(self.booster.predict(df[self.features]), 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-(self.a * np.log(p / (1 - p)) + self.b)))

    def explain(self, row_df, top=5):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.booster)
        sv = self._explainer.shap_values(row_df[self.features])
        sv = (sv[1] if isinstance(sv, list) else sv)[0]
        order = np.argsort(-np.abs(sv))[:top]
        return [{"feature": self.features[i], "value": _num(row_df[self.features[i]].iloc[0]),
                 "impact": float(sv[i])} for i in order]


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


@lru_cache
def ri_model():
    m = _json("ri_metrics.json")
    sel = m["selected"]
    model = CalibratedLGBM("ri_model.txt", "ri_calibration.json", m["variants"][sel]["features"])
    model.threshold = m["variants"][sel]["val"]["threshold"]
    tr = m["splits"]["train"]
    model.base_rate = tr["ri"] / tr["rows"]
    return model


@lru_cache
def genesis_model():
    cal = _json("genesis_calibration.json")
    return CalibratedLGBM("genesis_model.txt", "genesis_calibration.json", cal["features"])


# ---------- storm feature table ----------
@lru_cache
def storm_table():
    f = build_features(load_ibtracs()).reset_index(drop=True)
    f["wind_imd_eff"] = f.wind_imd.fillna(f.wind * WIND_1MIN_TO_IMD)
    if era5_env.available():
        env = era5_env.env_at(f.lat.values, f.lon.values, f.time.values)
        f[list(env.columns)] = env.values
    return f


# ---------- track model ----------
@lru_cache
def track_model():
    ck = torch.load(os.path.join(WEIGHTS, "track_model.pt"), map_location="cpu", weights_only=False)
    m = TrackNet()
    m.load_state_dict(ck["state_dict"])
    m.k_per_lead = np.array(ck["k_per_lead"])
    return m


def track_forecast(rows: pd.DataFrame, n=50):
    """rows: one or more feature rows. Returns per-row list of lead dicts."""
    m = track_model()
    seq, st = sequence_inputs(rows)
    torch.manual_seed(0)
    m.train()
    with torch.no_grad():
        samples = torch.stack([m(torch.from_numpy(seq), torch.from_numpy(st)) for _ in range(n)]).numpy()
    mean = samples.mean(0)
    radius = cone_radius_km(samples) * m.k_per_lead[None, :]
    lat, lon = to_latlon(rows, mean)
    wind_samples = rows.wind.values[None, :, None] + samples[..., 2] * WIND_SCALE
    # IMD-scale forecast = current IMD wind + predicted change (x0.93), so the forecast starts from the
    # official intensity even when JTWC and IMD disagree (common early in a storm's life)
    imd_now = rows.wind_imd_eff.values if "wind_imd_eff" in rows else rows.wind.values * WIND_1MIN_TO_IMD
    imd_now = np.where(np.isfinite(imd_now), imd_now, rows.wind.values * WIND_1MIN_TO_IMD)
    out = []
    for r in range(len(rows)):
        leads = []
        for i, h in enumerate(LEADS_H):
            w = float(np.clip(wind_samples[:, r, i].mean(), 10, 185))
            w_imd = float(np.clip(imd_now[r] + (w - rows.wind.values[r]) * WIND_1MIN_TO_IMD, 10, 170))
            leads.append({"lead_h": h, "lat": float(lat[r, i]), "lon": float(lon[r, i]),
                          "cone_radius_km": float(radius[r, i]), "wind_kt": w,
                          "wind_sd_kt": float(wind_samples[:, r, i].std()),
                          "wind_imd_kt": w_imd, "imd_class": IMD_CODES[int(imd_class_index(w_imd))]})
        out.append(leads)
    return out


def track_ensemble(row: pd.DataFrame, n=50):
    """Calibrated ensemble for one feature row: members' spread around the mean is scaled by the same
    k_lead factors as the 67% cone. Returns lat, lon, wind_imd arrays of shape (n, leads)."""
    m = track_model()
    seq, st = sequence_inputs(row)
    torch.manual_seed(1)
    m.train()
    with torch.no_grad():
        s = torch.stack([m(torch.from_numpy(seq), torch.from_numpy(st)) for _ in range(n)]).numpy()[:, 0]  # (n, leads, 3)
    mean = s.mean(0, keepdims=True)
    k = m.k_per_lead[None, :, None]
    disp = mean[..., :2] + k * (s[..., :2] - mean[..., :2])
    ref = pd.DataFrame({"lat": np.repeat(row.lat.values, n), "lon": np.repeat(row.lon.values, n)})
    lat, lon = to_latlon(ref, disp)
    w1 = float(row.wind.iloc[0])
    imd_now = float(row.wind_imd_eff.iloc[0]) if np.isfinite(row.wind_imd_eff.iloc[0]) else w1 * WIND_1MIN_TO_IMD
    wind_imd = np.clip(imd_now + s[..., 2] * WIND_SCALE * WIND_1MIN_TO_IMD, 10, 170)
    return lat, lon, wind_imd


def pressure_from_wind(v_kt):
    """Atkinson-Holliday wind-pressure relationship (illustrative)."""
    v = max(float(v_kt), 0.0)
    return 1010.0 - (v / 6.7) ** (1 / 0.644)


def imd_class_info(wind_imd):
    i = int(imd_class_index(wind_imd))
    return {"code": IMD_CODES[i], "name": IMD_NAMES[i], "index": i}


# ---------- satellite intensity model ----------
class IntensityNet(nn.Module):
    def __init__(self, arch, n_cls):
        super().__init__()
        import timm
        self.backbone = timm.create_model(arch, pretrained=False, in_chans=1, num_classes=0)
        d = self.backbone.num_features
        self.drop = nn.Dropout(0.3)
        self.head_v, self.head_t, self.head_c = nn.Linear(d, 1), nn.Linear(d, 1), nn.Linear(d, n_cls)

    def heads(self, pooled):
        f = self.drop(pooled)
        return self.head_v(f).squeeze(1), self.head_t(f).squeeze(1), self.head_c(f)

    def forward(self, x):
        return self.heads(self.backbone(x))


@lru_cache
def intensity_model():
    ck = torch.load(os.path.join(WEIGHTS, "intensity_model.pt"), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = IntensityNet(cfg["arch"], len(cfg["classes"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m.cfg = cfg
    return m


def _to_input(u8):
    return (torch.as_tensor(u8).float()[None, None] / 255.0 - 0.5) / 0.25


def u8_to_bt(u8, cfg):
    return cfg["bt_hi"] - np.asarray(u8, np.float32) / 255.0 * (cfg["bt_hi"] - cfg["bt_lo"])


def analyze_image(u8, adt=None, calibrate_insat=False, mc=20):
    """u8: (S,S) grayscale IR, cold = bright, storm-centred."""
    m = intensity_model()
    cfg = m.cfg
    x = _to_input(u8)
    views = torch.cat([torch.rot90(x, k, (2, 3)) for k in range(4)])
    with torch.no_grad():
        m.eval()
        v, t, c = m(views)
        v_det = float(v.mean() * cfg["v_sd"] + cfg["v_mu"])
        t_det = float(t.mean() * cfg["t_sd"] + cfg["t_mu"])
        probs = c.softmax(1).mean(0).numpy()
        # MC dropout on the heads only: a single backbone pass is shared
        pooled = m.backbone(views)
        m.drop.train()
        mc_v = torch.stack([m.heads(pooled)[0].mean() for _ in range(mc)]) * cfg["v_sd"] + cfg["v_mu"]
        m.drop.eval()
    wind = v_det
    if calibrate_insat and cfg.get("insat_calibration"):
        a, b = cfg["insat_calibration"]
        wind = a * v_det + b
    t_num = float(np.clip(t_det, 1.0, 8.0))
    if adt is None:
        adt = dict(zip(FEATURE_NAMES, adt_features(u8_to_bt(u8, cfg), cfg["crop_deg"] * 111.2 / u8.shape[0]).tolist()))
    cls_head = int(np.argmax(probs))
    dvorak_wind = float(t_to_wind(t_num))
    return {
        "cnn_wind_kt": wind, "cnn_wind_raw_kt": v_det, "cnn_wind_sd_kt": float(mc_v.std()),
        "insat_calibrated": bool(calibrate_insat),
        "t_number": float(round_t(t_num)), "t_number_raw": t_num, "dvorak_wind_kt": dvorak_wind,
        "class_probs": {IMD_CODES[i]: float(p) for i, p in enumerate(probs)},
        "class_head": {"code": IMD_CODES[cls_head], "name": IMD_NAMES[cls_head]},
        "class_from_wind": imd_class_info(wind * WIND_1MIN_TO_IMD),
        "class_from_dvorak": imd_class_info(dvorak_wind * WIND_1MIN_TO_IMD),
        "pressure_est_hpa": pressure_from_wind(wind * WIND_1MIN_TO_IMD),
        "scene_type": scene_type(np.array([adt[k] for k in FEATURE_NAMES])),
        "adt_features": {k: float(adt[k]) for k in FEATURE_NAMES},
        "gradcam_png": gradcam_png(u8),
    }


def gradcam_png(u8):
    """Grad-CAM for the wind head over the last conv feature map, blended on the grayscale image."""
    m = intensity_model()
    x = _to_input(u8).requires_grad_(False)
    m.eval()
    fmap = m.backbone.forward_features(x)
    fmap.retain_grad()
    pooled = m.backbone.forward_head(fmap)
    v = m.head_v(pooled).sum()
    m.zero_grad()
    v.backward()
    w = fmap.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=u8.shape, mode="bicubic", align_corners=False)[0, 0].detach().numpy()
    cam = np.clip(cam / (cam.max() + 1e-6), 0, 1)
    import matplotlib
    rgb = matplotlib.colormaps["inferno"](cam)[..., :3]
    gray = np.repeat((np.asarray(u8, np.float32) / 255.0)[..., None], 3, 2)
    alpha = (cam[..., None] ** 1.2) * 0.65
    blend = (gray * (1 - alpha) + rgb * alpha)
    return _png_b64((blend * 255).astype(np.uint8))


def gray_png(u8):
    return _png_b64(np.asarray(u8, np.uint8))


def _png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).resize((320, 320), Image.BICUBIC).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_uploaded_image(data: bytes, size):
    """INSAT-style JPEG/PNG (grayscale IR, cold = bright) -> (size,size) uint8. Removes a red centre marker."""
    a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(np.float32)
    gray = a.mean(2)
    marker = (a[..., 0] > 150) & (a[..., 1] < 110) & (a[..., 2] < 110)
    if marker.any():
        g = torch.from_numpy(np.where(marker, 0, gray).astype(np.float32))[None, None]
        mk = torch.from_numpy((~marker).astype(np.float32))[None, None]
        k = torch.ones(1, 1, 9, 9)
        fill = (F.conv2d(g, k, padding=4) / F.conv2d(mk, k, padding=4).clamp(min=1))[0, 0].numpy()
        gray = np.where(marker, fill, gray)
    t = torch.from_numpy(gray.astype(np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode="area")[0, 0].round().clamp(0, 255).byte().numpy()
