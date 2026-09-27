"""Held-out North Indian Ocean satellite frames (TCIR 2015-2016) for the image backtest."""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from ml.ibtracs import haversine_km

from .config import WEIGHTS
from .inference import storm_table


@lru_cache
def frames():
    z = np.load(os.path.join(WEIGHTS, "nio_eval_frames.npz"))
    meta = pd.DataFrame({"k": np.arange(len(z["idx"])), "idx": z["idx"], "tcir_sid": z["sid"],
                         "time": pd.to_datetime(z["time"]), "lat": z["lat"], "lon": z["lon"], "vmax": z["vmax"]})
    feats = pd.read_parquet(os.path.join(WEIGHTS, "frame_features.parquet")).set_index("idx")
    meta = meta.join(feats.drop(columns=["sid", "basin", "time", "lat", "lon", "vmax"]), on="idx")
    meta["name"], meta["ibtracs_sid"] = _match_names(meta)
    seq = os.path.join(os.environ.get("ADV_WEIGHTS_DIR", WEIGHTS), "seq_predictions.parquet")
    if os.path.exists(seq):  # ConvLSTM 24 h outlook from the advanced notebook, keyed by TCIR frame index
        sp = pd.read_parquet(seq)[["idx", "dv24_pred", "ri_prob", "dv24_obs"]].drop_duplicates("idx").set_index("idx")
        meta = meta.join(sp, on="idx")
    return z["images"], meta


def _match_names(meta):
    """Match each TCIR storm to the IBTrACS storm with the closest positions at shared times."""
    ib = storm_table()[["sid", "name", "time", "lat", "lon"]]
    names, sids = {}, {}
    for tsid, g in meta.groupby("tcir_sid"):
        j = g[["time", "lat", "lon"]].merge(ib, on="time", suffixes=("", "_ib"))
        if j.empty:
            names[tsid], sids[tsid] = tsid, None
            continue
        j["d"] = haversine_km(j.lat, j.lon, j.lat_ib, j.lon_ib)
        best = j.groupby(["sid", "name"]).d.median().idxmin()
        ok = j.groupby(["sid", "name"]).d.median().min() < 300
        names[tsid], sids[tsid] = (best[1], best[0]) if ok else (tsid, None)
    return meta.tcir_sid.map(names), meta.tcir_sid.map(sids)


def storm_list():
    _, m = frames()
    g = m.groupby(["tcir_sid", "name", "ibtracs_sid"], dropna=False).agg(
        start=("time", "min"), end=("time", "max"), frames=("k", "size"), peak_kt=("vmax", "max")).reset_index()
    g["held_out"] = g.start.dt.year >= 2016
    return g.sort_values("start")
