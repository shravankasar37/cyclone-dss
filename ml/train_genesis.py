"""Cyclogenesis early warning: will a depression (below Cyclonic Storm strength)
become a Cyclonic Storm (IMD >= 34 kt) within the next 72 h?

Rows are 6-hourly fixes of systems that are still below CS strength, from the
first 48 h of each system's life. IMD 3-min winds are used where available,
otherwise JTWC 1-min x 0.93. Seasons: train <=2016, val 2017-19, test 2020-25.
Run: python -m ml.train_genesis
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ml import era5_env
from ml.cyclone_scale import WIND_1MIN_TO_IMD
from ml.ibtracs import build_features, load_ibtracs, split_by_season

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "models", "weights")
CS_KT, HORIZON_H = 34, 72
FEATURES = ["wind_imd_eff", "dw6", "dw12", "lat", "lon", "u_motion", "v_motion", "log_dist2land",
            "is_arabian_sea", "month_sin", "month_cos", "age_h"]


def future_max(d, col, hours):
    out = np.full(len(d), np.nan)
    for _, g in d.groupby("sid"):
        t = g.time.values.astype("datetime64[h]").astype(np.int64)
        v = g[col].values
        for j, i in enumerate(g.index):
            m = (t > t[j]) & (t <= t[j] + hours)
            if m.any():
                out[i] = np.nanmax(v[m]) if np.isfinite(v[m]).any() else np.nan
    return out


def build_dataset():
    f = build_features(load_ibtracs()).reset_index(drop=True)
    f["wind_imd_eff"] = f.wind_imd.fillna(f.wind * WIND_1MIN_TO_IMD)
    f["future_max_imd"] = future_max(f, "wind_imd_eff", HORIZON_H)
    d = f[(f.wind_imd_eff < CS_KT) & (f.age_h <= 48) & f.future_max_imd.notna() & (f.dist2land > 0)].copy()
    d["y"] = (d.future_max_imd >= CS_KT).astype(int)
    d["split"] = split_by_season(d.season)
    feats = list(FEATURES)
    if era5_env.available():
        env = era5_env.env_at(d.lat.values, d.lon.values, d.time.values)
        d[list(env.columns)] = env.values
        feats += list(env.columns)
    return d.reset_index(drop=True), feats


def main():
    d, feats = build_dataset()
    tr, va, te = (d[d.split == s] for s in ["train", "val", "test"])
    print("rows (developing):", {s: f"{len(x)} ({int(x.y.sum())})" for s, x in [("train", tr), ("val", va), ("test", te)]},
          "systems:", d.groupby("split").sid.nunique().to_dict())
    params = dict(objective="binary", learning_rate=0.02, num_leaves=7, min_child_samples=20, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, metric="average_precision", verbose=-1, seed=0)
    from sklearn.model_selection import GroupKFold
    cv = lgb.cv(params, lgb.Dataset(tr[feats], tr.y), 3000, folds=list(GroupKFold(5).split(tr, tr.y, tr.sid)),
                callbacks=[lgb.early_stopping(200, verbose=False)])
    rounds = len(next(iter(cv.values())))
    m = lgb.train(params, lgb.Dataset(tr[feats], tr.y), rounds)
    raw = lambda x: m.predict(x[feats])  # noqa: E731
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    cal = LogisticRegression().fit(logit(raw(va))[:, None], va.y)
    prob = lambda x: cal.predict_proba(logit(raw(x))[:, None])[:, 1]  # noqa: E731

    def ev(x):
        p = prob(x)
        clim = np.full(len(x), tr.y.mean())
        # system-level: first fix of each system only (the earliest warning)
        first = x.assign(p=p).sort_values("time").groupby("sid").head(1)
        return {"n": int(len(x)), "developing_rate": float(x.y.mean()), "roc_auc": float(roc_auc_score(x.y, p)),
                "pr_auc": float(average_precision_score(x.y, p)),
                "brier_skill_vs_climatology": float(1 - brier_score_loss(x.y, p) / brier_score_loss(x.y, clim)),
                "accuracy_at_0.5": float(((p >= 0.5) == x.y).mean()),
                "first_fix_systems": int(len(first)), "first_fix_roc_auc": float(roc_auc_score(first.y, first.p)) if first.y.nunique() > 1 else None,
                "first_fix_accuracy_at_0.5": float(((first.p >= 0.5) == first.y).mean())}

    res = {"features": feats, "rounds": rounds, "val": ev(va), "test": ev(te),
           "feature_gain": dict(zip(feats, m.feature_importance("gain").round(1).tolist()))}
    first = te.assign(p=prob(te)).sort_values("time").groupby("sid").head(1)
    res["test_systems_first_fix"] = first[["season", "name", "time", "lat", "lon", "wind_imd_eff", "p", "y"]].assign(
        p=lambda x: x.p.round(3), time=lambda x: x.time.astype(str)).to_dict("records")
    m.save_model(os.path.join(OUT, "genesis_model.txt"))
    json.dump({"coef": cal.coef_.ravel().tolist(), "intercept": cal.intercept_.tolist(), "features": feats},
              open(os.path.join(OUT, "genesis_calibration.json"), "w"))
    json.dump(res, open(os.path.join(OUT, "genesis_metrics.json"), "w"), indent=1, default=str)
    t = res["test"]
    print(f"test ROC-AUC {t['roc_auc']:.3f}  PR-AUC {t['pr_auc']:.3f} (rate {t['developing_rate']:.2f})  BSS {t['brier_skill_vs_climatology']:+.3f}  "
          f"acc {t['accuracy_at_0.5']:.3f} | first-fix systems {t['first_fix_systems']} AUC {t['first_fix_roc_auc']} acc {t['first_fix_accuracy_at_0.5']:.3f}")
    print(first[["season", "name", "wind_imd_eff", "p", "y"]].to_string(index=False))


if __name__ == "__main__":
    main()
