"""Rapid-intensification (RI) model: P(wind rises >= 30 kt in the next 24 h).

Two feature sets, same rows, so the value of multi-source fusion is measured:
  track     - best-track history only (intensity trend, motion, position, season)
  track_env - track + ERA5 environment (SST, vertical shear, mid-level humidity)

Imbalance (~15:1 in the NIO record) is handled with class weighting
(scale_pos_weight = neg/pos); the decision threshold is chosen on validation
seasons (2017-19) and reported on unseen test seasons (2020-25).
Run: python -m ml.train_ri
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score)

from ml import era5_env
from ml.ibtracs import build_features, load_ibtracs, split_by_season

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "models", "weights")
RI_KT = 30

TRACK_FEATURES = ["wind", "dw6", "dw12", "dw24", "max_wind_so_far", "age_h", "lat", "lon",
                  "u_motion", "v_motion", "log_dist2land", "is_arabian_sea", "month_sin", "month_cos"]


def build_dataset():
    f = build_features(load_ibtracs())
    d = f[(f.wind >= 25) & f.dw24_future.notna() & (f.dist2land > 0)].copy()
    d["y"] = (d.dw24_future >= RI_KT).astype(int)
    d["split"] = split_by_season(d.season)
    if era5_env.available():
        env = era5_env.env_at(d.lat.values, d.lon.values, d.time.values)
        d[list(env.columns)] = env.values
    return d.reset_index(drop=True)


def evaluate(y, p, thr):
    pred = p >= thr
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    clim = np.full_like(p, y.mean(), dtype=float)
    bs, bs_ref = brier_score_loss(y, p), brier_score_loss(y, clim)
    return {"n": int(len(y)), "ri_cases": int(y.sum()),
            "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(bs), "brier_skill_vs_climatology": float(1 - bs / bs_ref),
            "threshold": float(thr), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "false_alarms": fp, "misses": fn, "hits": tp}


def train_variant(d, feats):
    tr, va, te = (d[d.split == s] for s in ["train", "val", "test"])
    pos_w = (tr.y == 0).sum() / max((tr.y == 1).sum(), 1)
    params = dict(objective="binary", learning_rate=0.02, num_leaves=7, min_child_samples=30,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  scale_pos_weight=pos_w, metric="average_precision", verbose=-1, seed=0)
    # number of rounds from storm-grouped 5-fold CV on the training seasons (val set is too small to early-stop on)
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(5).split(tr, tr.y, tr.sid))
    cv = lgb.cv(params, lgb.Dataset(tr[feats], tr.y), 3000, folds=folds,
                callbacks=[lgb.early_stopping(200, verbose=False)])
    rounds = len(next(iter(cv.values())))
    m = lgb.train(params, lgb.Dataset(tr[feats], tr.y), rounds)
    m.best_iteration = rounds
    # class weighting inflates probabilities; recalibrate to true frequencies with Platt scaling on val
    from sklearn.linear_model import LogisticRegression
    raw_va = m.predict(va[feats], num_iteration=m.best_iteration)
    cal = LogisticRegression().fit(np.log(raw_va / (1 - raw_va + 1e-9) + 1e-9)[:, None], va.y)

    def prob(x):
        r = m.predict(x[feats], num_iteration=m.best_iteration)
        return cal.predict_proba(np.log(r / (1 - r + 1e-9) + 1e-9)[:, None])[:, 1]

    p_va, p_te = prob(va), prob(te)
    prec, rec, thr = precision_recall_curve(va.y, p_va)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best_thr = float(thr[np.argmax(f1[:-1])])
    return m, cal, prob, {"features": feats, "pos_weight": float(pos_w), "best_iteration": int(m.best_iteration),
                          "val": evaluate(va.y.values, p_va, best_thr), "test": evaluate(te.y.values, p_te, best_thr)}


def main():
    d = build_dataset()
    counts = d.groupby("split").y.agg(["size", "sum"]).to_dict("index")
    print("rows/RI cases per split:", counts)
    variants = {"track": TRACK_FEATURES}
    if era5_env.available():
        variants["track_env"] = TRACK_FEATURES + era5_env.available_features()
    results, models = {}, {}
    for name, feats in variants.items():
        m, cal, prob, r = train_variant(d, feats)
        results[name], models[name] = r, (m, cal, prob)
        t = r["test"]
        print(f"{name:10s} test PR-AUC {t['pr_auc']:.3f} (base rate {t['ri_cases']/t['n']:.3f})  ROC-AUC {t['roc_auc']:.3f}  "
              f"BSS {t['brier_skill_vs_climatology']:+.3f}  P {t['precision']:.2f} R {t['recall']:.2f}  "
              f"hits {t['hits']} FA {t['false_alarms']} miss {t['misses']}")
    best = max(results, key=lambda k: results[k]["val"]["pr_auc"])
    m, cal, prob = models[best]
    imp = dict(zip(results[best]["features"], m.feature_importance("gain").round(1).tolist()))
    import shap
    te = d[d.split == "test"]
    sv = shap.TreeExplainer(m).shap_values(te[results[best]["features"]])
    sv = sv[1] if isinstance(sv, list) else sv
    shap_mean = dict(zip(results[best]["features"], np.abs(sv).mean(0).round(4).tolist()))

    # per-storm view of test RI events for the pitch
    te = te.assign(p=prob(te))
    storms = (te.groupby(["season", "name"]).agg(ri_obs=("y", "sum"), max_p=("p", "max")).reset_index()
              .query("ri_obs > 0").sort_values("season"))
    thr = results[best]["val"]["threshold"]
    storms["flagged"] = storms.max_p >= thr
    all_storms = te.groupby(["season", "name"]).agg(ri_obs=("y", "sum"), max_p=("p", "max"))
    false_alarm_storms = all_storms.query("ri_obs == 0 and max_p >= @thr").reset_index()
    print(f"test storms: {len(all_storms)}, with RI: {len(storms)}, RI storms flagged: {int(storms.flagged.sum())}, "
          f"non-RI storms falsely flagged: {len(false_alarm_storms)} of {int((all_storms.ri_obs == 0).sum())}")

    os.makedirs(OUT, exist_ok=True)
    m.save_model(os.path.join(OUT, "ri_model.txt"), num_iteration=m.best_iteration)
    json.dump({"coef": cal.coef_.ravel().tolist(), "intercept": cal.intercept_.tolist()},
              open(os.path.join(OUT, "ri_calibration.json"), "w"))
    json.dump({"selected": best, "ri_threshold_kt_24h": RI_KT, "splits": {k: {"rows": int(v["size"]), "ri": int(v["sum"])} for k, v in counts.items()},
               "imbalance_neg_per_pos": float((d.y == 0).sum() / d.y.sum()), "variants": results,
               "feature_gain": imp, "shap_mean_abs_test": shap_mean,
               "test_false_alarm_storms": false_alarm_storms.assign(max_p=false_alarm_storms.max_p.round(3)).to_dict("records"),
               "test_ri_storms": storms.assign(max_p=storms.max_p.round(3)).to_dict("records")},
              open(os.path.join(OUT, "ri_metrics.json"), "w"), indent=1, default=str)
    print("selected:", best, "| threshold", round(thr, 3))
    print(storms.to_string(index=False))


if __name__ == "__main__":
    main()
