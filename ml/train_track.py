"""Track + intensity forecaster with a calibrated uncertainty cone.

LSTM over the last 18 h (4 six-hourly fixes) -> east/north displacement and wind
change at +6/12/24/48/72 h. Monte-Carlo dropout gives a per-forecast spread; the
cone radius is spread x k_lead, where k_lead is set on validation seasons so ~67%
of real positions fall inside (the same 2/3 convention as NHC cones).
Compared against a persistence baseline (last-12 h motion extrapolated, wind held).
Run: python -m ml.train_track
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

from ml.ibtracs import (LEADS_H, KM_PER_DEG, build_features, displacement_km, haversine_km,
                        load_ibtracs, split_by_season)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "models", "weights")
PAST = [-18, -12, -6, 0]
STATIC = ["lat", "lon", "is_arabian_sea", "month_sin", "month_cos", "log_dist2land"]
POS_SCALE, WIND_SCALE = 100.0, 10.0  # km/100, kt/10
MC_SAMPLES = 50
CONE_COVERAGE = 0.67


def sequence_inputs(f):
    """(N, 4, 3) past dx, dy (relative to now, /100 km) and wind (/100 kt) + (N, 6) static features."""
    steps = []
    for h in PAST:
        sfx = "" if h == 0 else f"_m{-h}"
        dx, dy = displacement_km(f.lat.values, f.lon.values, f[f"lat{sfx}"].values, f[f"lon{sfx}"].values)
        steps.append(np.stack([dx / POS_SCALE, dy / POS_SCALE, f[f"wind{sfx}"].values / 100.0], 1))
    seq = np.stack(steps, 1).astype(np.float32)
    st = f[STATIC].values.astype(np.float32).copy()
    st[:, 0] /= 30.0
    st[:, 1] = (st[:, 1] - 80.0) / 20.0
    st[:, 5] /= 7.0
    return seq, st


def targets(f):
    ys = []
    for h in LEADS_H:
        dx, dy = displacement_km(f.lat.values, f.lon.values, f[f"lat_p{h}"].values, f[f"lon_p{h}"].values)
        ys.append(np.stack([dx / POS_SCALE, dy / POS_SCALE, (f[f"wind_p{h}"].values - f.wind.values) / WIND_SCALE], 1))
    return np.stack(ys, 1).astype(np.float32)  # (N, leads, 3)


class TrackNet(nn.Module):
    def __init__(self, hidden=64, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden, num_layers=2, batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(hidden + len(STATIC), 128), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(128, 128), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(128, len(LEADS_H) * 3))

    def forward(self, seq, st):
        h, _ = self.lstm(seq)
        return self.head(torch.cat([h[:, -1], st], 1)).view(-1, len(LEADS_H), 3)


def mc_predict(model, seq, st, n=MC_SAMPLES):
    model.train()  # keep dropout active
    with torch.no_grad():
        s, t = torch.from_numpy(seq), torch.from_numpy(st)
        return torch.stack([model(s, t) for _ in range(n)]).numpy()  # (n, N, leads, 3)


def build_dataset():
    f = build_features(load_ibtracs())
    ok = (f.wind >= 20) & f[[f"lat_m{-h}" for h in PAST if h] + [f"wind_m{-h}" for h in PAST if h]].notna().all(1)
    ok &= f.lat_p6.notna() & f.dist2land.notna()
    f = f[ok].reset_index(drop=True)
    f["split"] = split_by_season(f.season)
    return f


def persistence(f):
    """Extrapolate last-12h motion; hold wind constant."""
    out = np.zeros((len(f), len(LEADS_H), 3), np.float32)
    for i, h in enumerate(LEADS_H):
        out[:, i, 0] = f.u_motion.values * h / POS_SCALE
        out[:, i, 1] = f.v_motion.values * h / POS_SCALE
    return out


def to_latlon(f, disp):
    """disp (..., leads, >=2) in scaled km -> lat, lon arrays."""
    lat0, lon0 = f.lat.values[:, None], f.lon.values[:, None]
    lat = lat0 + disp[..., 1] * POS_SCALE / KM_PER_DEG
    lon = lon0 + disp[..., 0] * POS_SCALE / (KM_PER_DEG * np.cos(np.radians((lat0 + lat) / 2)))
    return lat, lon


def errors(f, pred):
    lat, lon = to_latlon(f, pred)
    res = {}
    for i, h in enumerate(LEADS_H):
        tl, tn, tw = f[f"lat_p{h}"].values, f[f"lon_p{h}"].values, f[f"wind_p{h}"].values
        m = ~np.isnan(tl)
        e = haversine_km(lat[m, i], lon[m, i], tl[m], tn[m])
        wm = m & ~np.isnan(tw)
        we = np.abs(f.wind.values[wm] + pred[wm, i, 2] * WIND_SCALE - tw[wm])
        res[f"{h}h"] = {"n": int(m.sum()), "track_err_km": float(e.mean()), "track_err_median_km": float(np.median(e)),
                        "wind_mae_kt": float(we.mean())}
    return res


def train(f, seed=0, epochs=150):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = f[f.split == "train"], f[f.split == "val"]
    s_tr, t_tr = (torch.from_numpy(a) for a in sequence_inputs(tr))
    y_tr = torch.from_numpy(targets(tr))
    s_va, t_va = (torch.from_numpy(a) for a in sequence_inputs(va))
    y_va = torch.from_numpy(targets(va))
    model = TrackNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def loss_fn(p, y):
        m = ~torch.isnan(y)
        return nn.functional.huber_loss(p[m], torch.nan_to_num(y)[m], delta=1.0)

    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(s_tr))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            loss = loss_fn(model(s_tr[b], t_tr[b]), y_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(s_va, t_va), y_va).item()
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def cone_radius_km(samples):
    """Per-forecast MC spread (km) for each lead: RMS distance of samples from their mean."""
    mean = samples.mean(0)
    d = np.hypot(samples[..., 0] - mean[..., 0], samples[..., 1] - mean[..., 1]) * POS_SCALE
    return np.sqrt((d ** 2).mean(0))  # (N, leads)


def main():
    f = build_dataset()
    print("samples:", f.split.value_counts().to_dict(), "storms:", f.groupby("split").sid.nunique().to_dict())
    model = train(f)
    va, te = f[f.split == "val"].reset_index(drop=True), f[f.split == "test"].reset_index(drop=True)
    res, k_lead = {}, []
    samp_va = mc_predict(model, *sequence_inputs(va))
    mean_va, spread_va = samp_va.mean(0), cone_radius_km(samp_va)
    lat, lon = to_latlon(va, mean_va)
    for i, h in enumerate(LEADS_H):
        m = va[f"lat_p{h}"].notna().values
        e = haversine_km(lat[m, i], lon[m, i], va[f"lat_p{h}"].values[m], va[f"lon_p{h}"].values[m])
        k_lead.append(float(np.quantile(e / np.maximum(spread_va[m, i], 1.0), CONE_COVERAGE)))

    samp_te = mc_predict(model, *sequence_inputs(te))
    mean_te, spread_te = samp_te.mean(0), cone_radius_km(samp_te)
    lat, lon = to_latlon(te, mean_te)
    coverage, radius = {}, {}
    for i, h in enumerate(LEADS_H):
        m = te[f"lat_p{h}"].notna().values
        e = haversine_km(lat[m, i], lon[m, i], te[f"lat_p{h}"].values[m], te[f"lon_p{h}"].values[m])
        r = spread_te[m, i] * k_lead[i]
        coverage[f"{h}h"], radius[f"{h}h"] = float((e <= r).mean()), float(np.median(r))
    res["model_test"] = errors(te, mean_te)
    res["persistence_test"] = errors(te, persistence(te))
    res["model_val"] = errors(va, mean_va)
    res["skill_vs_persistence_pct"] = {
        h: {"track": 100 * (1 - res["model_test"][h]["track_err_km"] / res["persistence_test"][h]["track_err_km"]),
            "wind": 100 * (1 - res["model_test"][h]["wind_mae_kt"] / res["persistence_test"][h]["wind_mae_kt"])}
        for h in res["model_test"]}
    res["cone"] = {"target_coverage": CONE_COVERAGE, "k_per_lead": dict(zip([f"{h}h" for h in LEADS_H], k_lead)),
                   "test_coverage": coverage, "median_radius_km_test": radius}
    res["samples"] = f.split.value_counts().to_dict()

    os.makedirs(OUT, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "leads_h": LEADS_H, "past_h": PAST, "static": STATIC,
                "k_per_lead": k_lead, "mc_samples": MC_SAMPLES}, os.path.join(OUT, "track_model.pt"))
    json.dump(res, open(os.path.join(OUT, "track_metrics.json"), "w"), indent=1)
    print(f"{'lead':>5} {'model km':>9} {'persist km':>10} {'skill%':>7} {'wind MAE':>9} {'persist':>8} {'cone cov':>9} {'radius':>7}")
    for h in res["model_test"]:
        a, b = res["model_test"][h], res["persistence_test"][h]
        print(f"{h:>5} {a['track_err_km']:9.0f} {b['track_err_km']:10.0f} {res['skill_vs_persistence_pct'][h]['track']:7.1f} "
              f"{a['wind_mae_kt']:9.1f} {b['wind_mae_kt']:8.1f} {coverage[h]:9.2f} {radius[h]:7.0f}")


if __name__ == "__main__":
    main()
