# %% [markdown]
# # PS 26070 — Cyclone intensity model (Colab / Kaggle GPU)
#
# Trains the satellite-image intensity model: **EfficientNet-B0** on IR brightness
# temperature, with three heads — max wind (kt), **Dvorak T-number**, and **IMD class**
# (D → SuCS). Runs two experiments so we can report the cross-basin transfer gain:
#
# * **A — NIO-only:** ImageNet init → train on North Indian Ocean storms only.
# * **B — Cross-basin:** pretrain on Central Pacific + Southern Hemisphere storms → fine-tune on NIO.
#
# Data: TCIR (Chen et al. 2018, CPAC+IO+SH subset) + INSAT-3D Kaggle set (external test).
# Split is **by storm and by year** (NIO ≤2014 train, 2015 val, 2016+ test) so no storm leaks
# between train and test.
#
# **How to run:** Runtime → Change runtime type → **T4 GPU**. Add Colab secrets
# `KAGGLE_USERNAME` and `KAGGLE_KEY` (🔑 icon, enable notebook access). Then Runtime → **Run all**.
# Everything is saved to `MyDrive/cyclone26070/`. If the session disconnects, just Run all
# again — finished stages are skipped and training resumes from the last epoch.

# %%
# 1. Environment
import os, sys, subprocess, glob, json, math, time
IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle/working")
LOCAL_TEST = os.environ.get("CYCLONE_LOCAL_TEST") == "1"
if IN_COLAB or IN_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "kaggle", "tables"], check=True)

import numpy as np, pandas as pd, h5py, torch
import torch.nn as nn, torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not LOCAL_TEST:
    assert DEVICE == "cuda", "No GPU. Runtime -> Change runtime type -> T4 GPU, then Run all again."
    print("GPU:", torch.cuda.get_device_name(0))

if "IMD_CODES" not in globals():  # local run: import shared modules from the repo
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
    sys.path.insert(0, os.getcwd())
    from ml.cyclone_scale import *  # noqa: F401,F403
    from ml.adt_features import *  # noqa: F401,F403

# %%
# 2. Credentials and output folder
if IN_COLAB:
    from google.colab import userdata, drive
    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"]:
        try:
            v = userdata.get(k)
            if v:
                os.environ[k] = v.strip()
        except Exception:
            pass
    drive.mount("/content/drive")
    OUT, WORK = "/content/drive/MyDrive/cyclone26070", "/content/work"
elif IN_KAGGLE:
    OUT, WORK = "/kaggle/working/cyclone26070", "/kaggle/working/work"
else:
    OUT, WORK = os.environ.get("OUT", "./colab_out"), os.environ.get("WORK", "./colab_work")
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
print("Saving results to", OUT)

# %%
# 3. Download datasets (TCIR ~4.4 GB zip -> ~15 GB; INSAT-3D ~45 MB)
TCIR_SLUG = "lomashrelia/tcir-cpac-io-sh-h5-file"
INSAT_SLUG = "sshubam/insat3d-infrared-raw-cyclone-images-20132021"

def find_one(pattern, roots):
    for r in roots:
        hits = sorted(glob.glob(os.path.join(r, "**", pattern), recursive=True))
        if hits:
            return hits[0]
    return None

def kaggle_get(slug, dest):
    if not (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
            or os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))):
        raise RuntimeError("Kaggle credentials missing: add KAGGLE_USERNAME and KAGGLE_KEY in Colab secrets.")
    print("Downloading", slug, "...")
    subprocess.run(["kaggle", "datasets", "download", slug, "-p", dest, "--unzip", "-q"], check=True)

ROOTS = ["/kaggle/input", WORK, os.environ.get("DATA_DIR", WORK)]
TCIR = os.environ.get("TCIR_PATH") or find_one("TCIR*.h5", ROOTS)
if not TCIR:
    kaggle_get(TCIR_SLUG, os.path.join(WORK, "tcir"))
    TCIR = find_one("TCIR*.h5", ROOTS)
INSAT_DIR = os.environ.get("INSAT_DIR") or (os.path.dirname(find_one("insat_3d_ds*.csv", ROOTS) or "") or None)
if not INSAT_DIR:
    kaggle_get(INSAT_SLUG, os.path.join(WORK, "insat3d"))
    INSAT_DIR = os.path.dirname(find_one("insat_3d_ds*.csv", ROOTS))
print("TCIR:", TCIR, f"({os.path.getsize(TCIR)/1e9:.1f} GB)")
print("INSAT:", INSAT_DIR)

# %%
# 4. Inspect TCIR
try:
    info = pd.read_hdf(TCIR, key="info", mode="r")
except KeyError:
    with h5py.File(TCIR, "r") as hf:
        raise RuntimeError(f"No 'info' table in TCIR file. Keys: {list(hf.keys())} - send this to Claude.")
with h5py.File(TCIR, "r") as hf:
    MSHAPE = hf["matrix"].shape
    print("matrix", MSHAPE, hf["matrix"].dtype, "chunks", hf["matrix"].chunks)
info = info.reset_index(drop=True)
print(info.columns.tolist())
print(info.head(3))
print(info["data_set"].value_counts())
assert len(info) == MSHAPE[0], "info rows and image count differ"

# %%
# 5. Preprocess: IR brightness temperature -> storm-centred grayscale (cold = bright), ADT-lite features
IMG = 160                     # model input size
BT_LO, BT_HI = 180.0, 310.0   # K; display range for IR -> grayscale
KM_PER_PX = 0.07 * 111.2      # TCIR grid is 0.07 deg
CROP_DEG = MSHAPE[1] * 0.07   # ~14 deg box; backend crops live imagery the same way
N = len(info)
basin = info["data_set"].astype(str).str.strip().values
X_PATH = os.path.join(WORK, f"X_{IMG}.npy")
F_PATH = os.path.join(WORK, "adt_feats.npy")
B_PATH = os.path.join(WORK, "bad.npy")

def bt_to_u8(bt):
    return (np.clip((BT_HI - bt) / (BT_HI - BT_LO), 0, 1) * 255).astype(np.uint8)

if not os.path.exists(B_PATH):
    X = np.lib.format.open_memmap(X_PATH, mode="w+", dtype=np.uint8, shape=(N, IMG, IMG))
    feats = np.zeros((N, len(FEATURE_NAMES)), np.float32)
    bad = np.zeros(N, bool)
    CH = 256
    t0 = time.time()
    with h5py.File(TCIR, "r") as hf:
        M = hf["matrix"]
        for s in range(0, N, CH):
            e = min(N, s + CH)
            bt = M[s:e, :, :, 0].astype(np.float32)
            bt[(bt < 150) | (bt > 340)] = np.nan
            nan_frac = np.isnan(bt).mean((1, 2))
            bad[s:e] = nan_frac > 0.1
            med = np.nanmedian(np.where(np.isnan(bt), np.nan, bt).reshape(e - s, -1), 1) if (nan_frac > 0).any() else None
            if med is not None:
                med = np.where(np.isnan(med), 280.0, med)
                bt = np.where(np.isnan(bt), med[:, None, None], bt)
            sh = basin[s:e] == "SH"
            bt[sh] = bt[sh, ::-1, :]  # mirror SH storms so all rotate counter-clockwise like NIO storms
            feats[s:e] = adt_features(bt, KM_PER_PX)
            u8 = torch.from_numpy(bt_to_u8(bt)).float()[:, None]
            X[s:e] = F.interpolate(u8, size=(IMG, IMG), mode="area")[:, 0].round().byte().numpy()
            if (s // CH) % 20 == 0:
                print(f"  {e}/{N} frames  {time.time()-t0:.0f}s")
    X.flush(); np.save(F_PATH, feats); np.save(B_PATH, bad)
X = np.load(X_PATH, mmap_mode="r"); feats = np.load(F_PATH); bad = np.load(B_PATH)
print("bad frames dropped:", int(bad.sum()), "of", N)

# %%
# 6. Labels and storm-level splits
vmax = pd.to_numeric(info["Vmax"], errors="coerce").values.astype(np.float32)
t_str = info["time"].astype(str).str.slice(0, 10)
when = pd.to_datetime(t_str, format="%Y%m%d%H", errors="coerce")
year = when.dt.year.fillna(pd.to_numeric(info["ID"].astype(str).str.slice(0, 4), errors="coerce")).values
sid = (info["ID"].astype(str) + "_" + pd.Series(basin)).values
valid = (~bad) & np.isfinite(vmax) & (vmax > 0) & np.isfinite(year)

split = np.full(N, "", dtype=object)
io = basin == "IO"
split[io & (year <= 2014)] = "io_train"
split[io & (year == 2015)] = "io_val"
split[io & (year >= 2016)] = "io_test"
other = ~io
h = pd.util.hash_array(sid.astype(str)) % 10
split[other & (h != 0)] = "x_train"
split[other & (h == 0)] = "x_val"
split[~valid] = ""
print(pd.Series(split[split != ""]).value_counts())
print("NIO storms per split:", {s: len(set(sid[split == s])) for s in ["io_train", "io_val", "io_test"]})

tnum = wind_to_t(vmax).astype(np.float32)
cls = imd_class_from_1min(np.nan_to_num(vmax)).astype(np.int64)
V_MU, V_SD, T_MU, T_SD = 60.0, 30.0, 4.0, 1.5
idx = {s: np.where(split == s)[0] for s in ["io_train", "io_val", "io_test", "x_train", "x_val"]}

# %%
# 7. Model, GPU augmentation, training loop
import timm
ARCH = "efficientnet_b0"

class IntensityNet(nn.Module):
    def __init__(self, pretrained=True, drop=0.3, n_cls=len(IMD_CODES)):
        super().__init__()
        self.backbone = timm.create_model(ARCH, pretrained=pretrained, in_chans=1, num_classes=0)
        d = self.backbone.num_features
        self.drop = nn.Dropout(drop)
        self.head_v, self.head_t, self.head_c = nn.Linear(d, 1), nn.Linear(d, 1), nn.Linear(d, n_cls)
    def forward(self, x):
        f = self.drop(self.backbone(x))
        return self.head_v(f).squeeze(1), self.head_t(f).squeeze(1), self.head_c(f)

X_GPU = torch.from_numpy(np.ascontiguousarray(X)).to(DEVICE)          # uint8, ~1 GB for 40k frames
V_T = torch.from_numpy(np.nan_to_num(vmax)).to(DEVICE)
T_T = torch.from_numpy(tnum).to(DEVICE)
C_T = torch.from_numpy(cls).to(DEVICE)

def to_input(u8):
    return (u8.float()[:, None] / 255.0 - 0.5) / 0.25

def augment(x):
    """Rotation (any angle - keeps rotation direction), small shift/scale (centre-fix error), brightness."""
    b = x.shape[0]
    ang = torch.rand(b, device=x.device) * 2 * math.pi
    sc = 1 + (torch.rand(b, device=x.device) - 0.5) * 0.1
    tx, ty = ((torch.rand(2, b, device=x.device) - 0.5) * 0.08)
    c, s = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([c, -s, tx], 1), torch.stack([s, c, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    return x * (1 + 0.1 * torch.randn(b, 1, 1, 1, device=x.device)) + 0.05 * torch.randn(b, 1, 1, 1, device=x.device)

@torch.no_grad()
def predict(model, ids, tta=True, bs=512):
    model.eval()
    vs, ts, ps = [], [], []
    for i in range(0, len(ids), bs):
        x = to_input(X_GPU[torch.as_tensor(ids[i:i+bs], device=DEVICE)])
        views = [torch.rot90(x, k, (2, 3)) for k in range(4)] if tta else [x]
        outs = [model(v) for v in views]
        vs.append(torch.stack([o[0] for o in outs]).mean(0) * V_SD + V_MU)
        ts.append(torch.stack([o[1] for o in outs]).mean(0) * T_SD + T_MU)
        ps.append(torch.stack([o[2].softmax(1) for o in outs]).mean(0))
    return torch.cat(vs).cpu().numpy(), torch.cat(ts).cpu().numpy(), torch.cat(ps).cpu().numpy()

def evaluate(model, ids):
    from sklearn.metrics import f1_score, confusion_matrix
    pv, pt, pp = predict(model, ids)
    tv, tc = vmax[ids], cls[ids]
    err = pv - tv
    cls_head = pp.argmax(1)
    cls_from_v = imd_class_from_1min(pv)
    cls_from_t = imd_class_from_1min(t_to_wind(pt))
    labels = list(range(len(IMD_CODES)))
    return {
        "n": int(len(ids)), "storms": int(len(set(sid[ids]))),
        "mae_kt": float(np.abs(err).mean()), "rmse_kt": float(np.sqrt((err ** 2).mean())), "bias_kt": float(err.mean()),
        "t_number_mae": float(np.abs(pt - tnum[ids]).mean()),
        "dvorak_wind_mae_kt": float(np.abs(t_to_wind(pt) - tv).mean()),
        "class_acc_head": float((cls_head == tc).mean()),
        "class_acc_from_wind": float((cls_from_v == tc).mean()),
        "class_acc_from_dvorak": float((cls_from_t == tc).mean()),
        "macro_f1_head": float(f1_score(tc, cls_head, labels=labels, average="macro", zero_division=0)),
        "per_class_f1_head": dict(zip(IMD_CODES, f1_score(tc, cls_head, labels=labels, average=None, zero_division=0).round(3).tolist())),
        "class_support": dict(zip(IMD_CODES, np.bincount(tc, minlength=len(labels)).tolist())),
        "confusion_head": confusion_matrix(tc, cls_head, labels=labels).tolist(),
        "mae_by_class_kt": {IMD_CODES[k]: float(np.abs(err[tc == k]).mean()) for k in labels if (tc == k).any()},
        "head_agreement_wind_vs_dvorak": float((cls_from_v == cls_from_t).mean()),
    }

def train_stage(tag, tr_ids, va_ids, epochs, lr, init=None, bs=128):
    best_path, last_path = os.path.join(OUT, f"{tag}_best.pt"), os.path.join(OUT, f"{tag}_last.pt")
    model = IntensityNet(pretrained=init is None and not LOCAL_TEST).to(DEVICE)
    if init is not None:
        model.load_state_dict(init.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(tr_ids) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    start, best = 0, float("inf")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=DEVICE, weights_only=False)
        if ck.get("done"):
            print(f"[{tag}] already trained - loading best checkpoint")
            model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=False)["state_dict"])
            return model
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"[{tag}] resuming at epoch {start}")
    # sample weights: rarer intensities (strong storms) are drawn more often
    bins = np.clip((vmax[tr_ids] // 10).astype(int), 0, 20)
    w = 1 / np.sqrt(np.bincount(bins, minlength=21)[bins])
    w_t = torch.as_tensor(w / w.sum(), device=DEVICE)
    tr_t = torch.as_tensor(tr_ids, device=DEVICE)
    cw = torch.as_tensor(1 / np.sqrt(np.bincount(cls[tr_ids], minlength=len(IMD_CODES)) + 1), device=DEVICE, dtype=torch.float32)
    ce = nn.CrossEntropyLoss(weight=cw / cw.mean(), label_smoothing=0.05)
    for ep in range(start, epochs):
        model.train(); t0 = time.time(); tot = 0.0
        order = tr_t[torch.multinomial(w_t, steps * bs, replacement=True)]
        for i in range(steps):
            b = order[i * bs:(i + 1) * bs]
            x = augment(to_input(X_GPU[b]))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                pv, pt, pc = model(x)
                loss = (F.huber_loss(pv.float(), (V_T[b] - V_MU) / V_SD, delta=0.5)
                        + 0.5 * F.huber_loss(pt.float(), (T_T[b] - T_MU) / T_SD, delta=0.5)
                        + 0.3 * ce(pc.float(), C_T[b]))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        m = evaluate(model, va_ids)
        flag = ""
        if m["mae_kt"] < best:
            best = m["mae_kt"]; flag = "  *best*"
            torch.save({"state_dict": model.state_dict()}, best_path)
        torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "done": ep == epochs - 1}, last_path)
        print(f"[{tag}] ep {ep+1}/{epochs} loss {tot/steps:.3f}  val MAE {m['mae_kt']:.2f} kt  "
              f"RMSE {m['rmse_kt']:.2f}  cls-acc {m['class_acc_head']:.3f}  {time.time()-t0:.0f}s{flag}")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=False)["state_dict"])
    return model

EP = {"base": 25, "pre": 12, "ft": 12} if not LOCAL_TEST else {"base": 1, "pre": 1, "ft": 1}

# %%
# 8. Experiment A: NIO-only baseline
model_a = train_stage("A_nio_only", idx["io_train"], idx["io_val"], EP["base"], lr=1e-3)

# %%
# 9. Experiment B: cross-basin pretraining (CPAC + SH) -> fine-tune on NIO
model_pre = train_stage("B_pretrain_xbasin", idx["x_train"], idx["x_val"], EP["pre"], lr=1e-3)
model_b = train_stage("B_finetune_nio", idx["io_train"], idx["io_val"], EP["ft"], lr=3e-4, init=model_pre)

# %%
# 10. Compare on the held-out NIO test storms (2016 onward) and pick the model by validation MAE
results = {}
for name, m in [("A_nio_only", model_a), ("B_cross_basin", model_b)]:
    results[name] = {"val": evaluate(m, idx["io_val"]), "test": evaluate(m, idx["io_test"])}
    r = results[name]["test"]
    print(f"{name:14s} test MAE {r['mae_kt']:.2f} kt  RMSE {r['rmse_kt']:.2f}  class-acc {r['class_acc_head']:.3f}  "
          f"macro-F1 {r['macro_f1_head']:.3f}  T-num MAE {r['t_number_mae']:.2f}")
gain = (results["A_nio_only"]["test"]["mae_kt"] - results["B_cross_basin"]["test"]["mae_kt"]) / results["A_nio_only"]["test"]["mae_kt"]
print(f"Cross-basin transfer: {gain*100:+.1f}% test-MAE change vs NIO-only")
BEST = "B_cross_basin" if results["B_cross_basin"]["val"]["mae_kt"] <= results["A_nio_only"]["val"]["mae_kt"] else "A_nio_only"
best_model = model_b if BEST == "B_cross_basin" else model_a
print("Selected (by validation MAE):", BEST)

# %%
# 11. External test: INSAT-3D Kaggle images (different sensor/processing - measures domain shift)
from PIL import Image
def load_insat(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    gray = a.mean(2)
    marker = (a[..., 0] > 150) & (a[..., 1] < 110) & (a[..., 2] < 110)   # red centre cross
    if marker.any():
        g = torch.from_numpy(np.where(marker, 0, gray))[None, None]
        m = torch.from_numpy((~marker).astype(np.float32))[None, None]
        k = torch.ones(1, 1, 9, 9)
        fill = (F.conv2d(g, k, padding=4) / F.conv2d(m, k, padding=4).clamp(min=1))[0, 0].numpy()
        gray = np.where(marker, fill, gray)
    t = torch.from_numpy(gray)[None, None]
    return F.interpolate(t, size=(IMG, IMG), mode="area")[0, 0].round().clamp(0, 255).byte()

insat_csv = find_one("insat_3d_ds*.csv", [INSAT_DIR])
raw_dir = os.path.dirname(find_one("*.jp*g", [os.path.join(INSAT_DIR, "insat3d_raw_cyclone_ds")]) or "")
lab = pd.read_csv(insat_csv)
imgs, ys = [], []
for _, row in lab.iterrows():
    p = os.path.join(raw_dir, str(row["img_name"]))
    if not os.path.exists(p):
        alt = glob.glob(os.path.splitext(p)[0] + ".*")
        p = alt[0] if alt else None
    if p:
        imgs.append(load_insat(p)); ys.append(float(row["label"]))
ys = np.array(ys, np.float32)
xi = torch.stack(imgs).to(DEVICE)
best_model.eval()
with torch.no_grad():
    x = to_input(xi)
    pv = torch.stack([best_model(torch.rot90(x, k, (2, 3)))[0] for k in range(4)]).mean(0).cpu().numpy() * V_SD + V_MU
from sklearn.model_selection import KFold
cal = np.zeros_like(pv)
for tr, te in KFold(5, shuffle=True, random_state=0).split(pv):
    a, b = np.polyfit(pv[tr], ys[tr], 1)
    cal[te] = a * pv[te] + b
insat_metrics = {"n": int(len(ys)), "zero_shot_mae_kt": float(np.abs(pv - ys).mean()),
                 "zero_shot_bias_kt": float((pv - ys).mean()),
                 "linear_calibrated_5fold_mae_kt": float(np.abs(cal - ys).mean()),
                 "corr": float(np.corrcoef(pv, ys)[0, 1]),
                 "calibration_full_fit": [float(v) for v in np.polyfit(pv, ys, 1)]}
print("INSAT-3D external test:", json.dumps(insat_metrics, indent=1))

# %%
# 12. Export weights, metrics, features for the RI model, and NIO test frames for the backtest demo
config = {"arch": ARCH, "img_size": IMG, "in_chans": 1, "bt_lo": BT_LO, "bt_hi": BT_HI, "crop_deg": CROP_DEG,
          "km_per_px_native": KM_PER_PX, "v_mu": V_MU, "v_sd": V_SD, "t_mu": T_MU, "t_sd": T_SD,
          "classes": IMD_CODES, "wind_units": "kt, 1-min sustained (JTWC); IMD class uses x0.93",
          "sh_mirrored": True, "selected_experiment": BEST, "trained_at": time.strftime("%Y-%m-%d %H:%M"),
          "insat_calibration": insat_metrics["calibration_full_fit"]}
torch.save({"state_dict": best_model.state_dict(), "config": config}, os.path.join(OUT, "intensity_model.pt"))

metrics = {"experiments": results, "cross_basin_test_mae_gain_pct": gain * 100, "selected": BEST,
           "insat3d_external": insat_metrics,
           "split_sizes": {k: int(len(v)) for k, v in idx.items()},
           "split_storms": {k: int(len(set(sid[v]))) for k, v in idx.items()},
           "config": config}
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)

keep = np.where(valid)[0]
cnn_v, cnn_t, _ = predict(best_model, keep)
ff = pd.DataFrame(feats[keep], columns=FEATURE_NAMES)
ff.insert(0, "idx", keep); ff.insert(1, "sid", sid[keep]); ff.insert(2, "basin", basin[keep])
ff.insert(3, "time", when.values[keep]); ff.insert(4, "lat", info["lat"].values[keep]); ff.insert(5, "lon", info["lon"].values[keep])
ff["vmax"] = vmax[keep]; ff["mslp"] = pd.to_numeric(info["MSLP"], errors="coerce").values[keep] if "MSLP" in info else np.nan
ff["split"] = split[keep]; ff["cnn_vmax"] = cnn_v; ff["cnn_tnum"] = cnn_t
ff.to_parquet(os.path.join(OUT, "frame_features.parquet"), index=False)

demo = np.concatenate([idx["io_val"], idx["io_test"]])
np.savez_compressed(os.path.join(OUT, "nio_eval_frames.npz"), images=np.asarray(X[demo]), idx=demo,
                    sid=sid[demo].astype(str), time=when.values[demo].astype("datetime64[s]").astype(str),
                    lat=info["lat"].values[demo], lon=info["lon"].values[demo], vmax=vmax[demo])

# %%
# 13. Plots for the slide deck
import matplotlib; matplotlib.use("Agg") if LOCAL_TEST else None
import matplotlib.pyplot as plt
pv, _, pp = predict(best_model, idx["io_test"])
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(vmax[idx["io_test"]], pv, s=6, alpha=0.5)
ax[0].plot([0, 160], [0, 160], "k--", lw=1); ax[0].set_xlabel("Best-track Vmax (kt)"); ax[0].set_ylabel("Predicted (kt)")
ax[0].set_title(f"NIO held-out test (2016+) - MAE {results[BEST]['test']['mae_kt']:.1f} kt")
cm = np.array(results[BEST]["test"]["confusion_head"], float); cm = cm / cm.sum(1, keepdims=True).clip(min=1)
ax[1].imshow(cm, cmap="Blues", vmin=0, vmax=1); ax[1].set_xticks(range(7), IMD_CODES); ax[1].set_yticks(range(7), IMD_CODES)
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("True"); ax[1].set_title("IMD class confusion (row-normalised)")
for i in range(7):
    for j in range(7):
        ax[1].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "intensity_eval.png"), dpi=150)
print("\nDONE. Download these from", OUT, ":")
for f in ["intensity_model.pt", "metrics.json", "frame_features.parquet", "nio_eval_frames.npz", "intensity_eval.png"]:
    p = os.path.join(OUT, f); print(f"  {f:28s} {os.path.getsize(p)/1e6:7.1f} MB" if os.path.exists(p) else f"  {f} MISSING")
