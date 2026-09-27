# %% [markdown]
# # PS 26070 — Advanced models (Colab / Kaggle GPU)
#
# Trains three models in one run. Every part caches to Google Drive, so if Colab disconnects just
# **Runtime → Run all** again and it resumes.
#
# * **A · Multi-channel intensity** — IR + water vapour + passive microwave (TCIR), with random channel dropout so it
#   still works when WV/PMW are missing. One model, evaluated in IR-only / IR+WV / IR+WV+PMW modes.
# * **B · ConvLSTM image-sequence model** — last 8 frames (21 h) of IR+WV → 24 h wind change and rapid-intensification
#   probability. Compared against the same network without images.
# * **C · Full-basin detector** — scans the whole Bay of Bengal + Arabian Sea (NOAA GridSat-B1 IR+WV, 0–30N 40–100E)
#   and finds cyclone / depression centres. Tested on 2017–2024 seasons it never saw.
# * **D · Demo scenes** — 3-hourly basin scenes for Ockhi 2017, Fani 2019, Amphan 2020, Tauktae 2021, Dana 2024, Vardah 2016.
#
# **Setup:** Runtime → Change runtime type → **T4 GPU**. Secrets `KAGGLE_USERNAME`, `KAGGLE_KEY` (same as before).
# Expect ~1.5–2 h. Results go to `MyDrive/cyclone26070/advanced/`.

# %%
# 1. Environment
import os, sys, subprocess, glob, json, math, time, io
IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle/working")
LOCAL_TEST = os.environ.get("CYCLONE_LOCAL_TEST") == "1"
if IN_COLAB or IN_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "kaggle", "tables"], check=True)

import numpy as np, pandas as pd, h5py, torch, requests
import torch.nn as nn, torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not LOCAL_TEST:
    assert DEVICE == "cuda", "No GPU. Runtime -> Change runtime type -> T4 GPU, then Run all again."
    print("GPU:", torch.cuda.get_device_name(0))
AMP = DEVICE == "cuda"

if "IMD_CODES" not in globals():  # local run: import shared modules from the repo
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
    sys.path.insert(0, os.getcwd())
    from ml.cyclone_scale import *  # noqa: F401,F403
    from ml.adt_features import *  # noqa: F401,F403
    from ml.gridsat import *  # noqa: F401,F403
    from ml.nets import *  # noqa: F401,F403

# %%
# 2. Credentials, Drive, output folders
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
    OUT, WORK = "/content/drive/MyDrive/cyclone26070/advanced", "/content/work"
elif IN_KAGGLE:
    OUT, WORK = "/kaggle/working/advanced", "/kaggle/working/work"
else:
    OUT, WORK = os.environ.get("OUT", "./adv_out"), os.environ.get("WORK", "./adv_work")
CACHE = os.path.join(OUT, "gridsat_nio")
for d in [OUT, WORK, CACHE]:
    os.makedirs(d, exist_ok=True)
print("Saving results to", OUT)

def done(name):
    return os.path.exists(os.path.join(OUT, name))

# %%
# 3. Data: TCIR (Kaggle) + IBTrACS North Indian Ocean (NOAA)
TCIR_SLUG = "lomashrelia/tcir-cpac-io-sh-h5-file"
def find_one(pattern, roots):
    for r in roots:
        hits = sorted(glob.glob(os.path.join(r, "**", pattern), recursive=True))
        if hits:
            return hits[0]
    return None

TCIR = os.environ.get("TCIR_PATH") or find_one("TCIR*.h5", ["/kaggle/input", WORK])
need_tcir = not (done("mc_intensity_model.pt") and done("seq_model.pt"))
if need_tcir and not TCIR:
    print("Downloading TCIR (~4.4 GB) ...")
    subprocess.run(["kaggle", "datasets", "download", TCIR_SLUG, "-p", os.path.join(WORK, "tcir"), "--unzip", "-q"], check=True)
    TCIR = find_one("TCIR*.h5", [WORK])

IB_PATH = os.environ.get("IBTRACS_PATH") or os.path.join(WORK, "ibtracs_ni.csv")
if not os.path.exists(IB_PATH):
    r = requests.get("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
                     "v04r01/access/csv/ibtracs.NI.list.v04r01.csv", timeout=300)
    open(IB_PATH, "wb").write(r.content)
ib = pd.read_csv(IB_PATH, skiprows=[1], low_memory=False, keep_default_na=False, na_values=[" ", ""],
                 usecols=["SID", "NAME", "SEASON", "ISO_TIME", "LAT", "LON", "USA_WIND", "NEWDELHI_WIND", "TRACK_TYPE"])
ib = ib[ib.TRACK_TYPE == "main"].copy()
ib["time"] = pd.to_datetime(ib.ISO_TIME)
for c in ["LAT", "LON", "USA_WIND", "NEWDELHI_WIND"]:
    ib[c] = pd.to_numeric(ib[c], errors="coerce")
ib["wind_imd"] = ib.NEWDELHI_WIND.fillna(ib.USA_WIND * WIND_1MIN_TO_IMD)
ib = ib[(ib.time.dt.minute == 0) & (ib.time.dt.hour % 3 == 0) & ib.wind_imd.notna()]
print("IBTrACS NI rows:", len(ib))

# %%
# 4. TCIR preprocessing: IR, WV, PMW channels (+ availability) at 160 px, and IR/WV at 64 px for the sequence model
IMG, SEQ_IMG = 160, 64
if need_tcir:
    info = pd.read_hdf(TCIR, key="info", mode="r").reset_index(drop=True)
    N = len(info)
    basin = info["data_set"].astype(str).str.strip().values
    XP, MP = os.path.join(WORK, "X3.npy"), os.path.join(WORK, "masks.npy")
    if not os.path.exists(MP):
        with h5py.File(TCIR, "r") as hf:
            M = hf["matrix"]
            samp = M[:: max(1, N // 400), :, :, 3].astype(np.float32)
            samp = samp[np.isfinite(samp) & (samp >= 0)]
            PMW_MAX = float(np.percentile(samp, 99.5)) if samp.size else 1.0
            print(f"PMW channel: median {np.median(samp) if samp.size else 0:.2f}, p99.5 {PMW_MAX:.2f} (used as scale)")
            X = np.lib.format.open_memmap(XP, mode="w+", dtype=np.uint8, shape=(N, 3, IMG, IMG))
            masks = np.zeros((N, 3), bool)  # ir_ok, wv_ok, pmw_ok
            CH = 256
            for s in range(0, N, CH):
                e = min(N, s + CH)
                m = M[s:e].astype(np.float32)          # (b, 201, 201, 4): IR, WV, VIS, PMW
                sh = basin[s:e] == "SH"
                m[sh] = m[sh, ::-1]                     # mirror SH storms to NH rotation
                ir, wv, pmw = m[..., 0], m[..., 1], m[..., 3]
                ir = np.where((ir > 150) & (ir < 340), ir, np.nan)
                wv = np.where((wv > 150) & (wv < 340), wv, np.nan)
                pmw = np.where(np.isfinite(pmw) & (pmw >= 0), pmw, np.nan)
                masks[s:e, 0] = np.isnan(ir).mean((1, 2)) < 0.1
                masks[s:e, 1] = np.isnan(wv).mean((1, 2)) < 0.1
                masks[s:e, 2] = np.isnan(pmw).mean((1, 2)) < 0.5
                u8 = np.stack([bt_to_u8(np.where(np.isnan(ir), np.nanmedian(ir) if np.isfinite(ir).any() else 280, ir), IR_RANGE),
                               bt_to_u8(np.nan_to_num(wv, nan=WV_RANGE[1]), WV_RANGE),
                               (np.clip(np.nan_to_num(pmw) / PMW_MAX, 0, 1) * 255).astype(np.uint8)], 1)
                X[s:e] = F.interpolate(torch.from_numpy(u8).float(), size=(IMG, IMG), mode="area").round().byte().numpy()
                if (s // CH) % 20 == 0:
                    print(f"  {e}/{N}")
            X.flush(); np.save(MP, masks); json.dump({"pmw_max": PMW_MAX}, open(os.path.join(WORK, "pmw.json"), "w"))
    X = np.load(XP, mmap_mode="r"); masks = np.load(MP); PMW_MAX = json.load(open(os.path.join(WORK, "pmw.json")))["pmw_max"]
    print("channel availability  IR %.2f  WV %.2f  PMW %.2f" % tuple(masks.mean(0)))

    vmax = pd.to_numeric(info["Vmax"], errors="coerce").values.astype(np.float32)
    when = pd.to_datetime(info["time"].astype(str).str.slice(0, 10), format="%Y%m%d%H", errors="coerce")
    year = when.dt.year.values
    sid = (info["ID"].astype(str) + "_" + pd.Series(basin)).values
    valid = masks[:, 0] & np.isfinite(vmax) & (vmax > 0) & np.isfinite(year)
    split = np.full(N, "", dtype=object)
    io_ = basin == "IO"
    split[io_ & (year <= 2014)] = "io_train"; split[io_ & (year == 2015)] = "io_val"; split[io_ & (year >= 2016)] = "io_test"
    hsh = pd.util.hash_array(sid.astype(str)) % 10
    split[~io_ & (hsh != 0)] = "x_train"; split[~io_ & (hsh == 0)] = "x_val"
    split[~valid] = ""
    idx = {s: np.where(split == s)[0] for s in ["io_train", "io_val", "io_test", "x_train", "x_val"]}
    tnum = wind_to_t(vmax).astype(np.float32)
    cls = imd_class_from_1min(np.nan_to_num(vmax)).astype(np.int64)
    print({k: len(v) for k, v in idx.items()})

# %%
# 5. Part A — multi-channel intensity model with channel dropout
V_MU, V_SD, T_MU, T_SD = 60.0, 30.0, 4.0, 1.5
EPA = {"pre": 10, "ft": 12} if not LOCAL_TEST else {"pre": 1, "ft": 1}

def rotate(x):
    b = x.shape[0]
    ang = torch.rand(b, device=x.device) * 2 * math.pi
    sc = 1 + (torch.rand(b, device=x.device) - 0.5) * 0.1
    tx, ty = (torch.rand(2, b, device=x.device) - 0.5) * 0.08
    c, s = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([c, -s, tx], 1), torch.stack([s, c, ty], 1)], 1)
    return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)

def mc_input(u8, m, mode="train"):
    """u8 (B,3,S,S) uint8 on GPU, m (B,3) bool -> (B,5,S,S). mode: train | ir | ir+wv | ir+wv+pmw"""
    x = u8.float() / 255.0
    wv_ok, pmw_ok = m[:, 1].float(), m[:, 2].float()
    if mode == "train":
        wv_ok = wv_ok * (torch.rand_like(wv_ok) > 0.3)
        pmw_ok = pmw_ok * (torch.rand_like(pmw_ok) > 0.5)
    elif mode == "ir":
        wv_ok, pmw_ok = wv_ok * 0, pmw_ok * 0
    elif mode == "ir+wv":
        pmw_ok = pmw_ok * 0
    wm, pm = wv_ok[:, None, None, None], pmw_ok[:, None, None, None]
    ones = torch.ones_like(x[:, :1])
    return torch.cat([(x[:, :1] - 0.5) / 0.25, (x[:, 1:2] - 0.5) / 0.25 * wm, (x[:, 2:3] - 0.5) / 0.25 * pm,
                      ones * wm, ones * pm], 1)

if not done("mc_intensity_model.pt"):
    X_GPU = torch.from_numpy(np.ascontiguousarray(X)).to(DEVICE)
    M_GPU = torch.from_numpy(masks).to(DEVICE)
    V_T, T_T, C_T = (torch.from_numpy(a).to(DEVICE) for a in [np.nan_to_num(vmax), tnum, cls])

    @torch.no_grad()
    def mc_predict(model, ids, mode, bs=256):
        model.eval(); vs, ts, ps = [], [], []
        for i in range(0, len(ids), bs):
            b = torch.as_tensor(ids[i:i + bs], device=DEVICE)
            x = mc_input(X_GPU[b], M_GPU[b], mode)
            outs = [model(torch.rot90(x, k, (2, 3))) for k in range(4)]
            vs.append(torch.stack([o[0] for o in outs]).mean(0) * V_SD + V_MU)
            ts.append(torch.stack([o[1] for o in outs]).mean(0) * T_SD + T_MU)
            ps.append(torch.stack([o[2].softmax(1) for o in outs]).mean(0))
        return torch.cat(vs).cpu().numpy(), torch.cat(ts).cpu().numpy(), torch.cat(ps).cpu().numpy()

    def mc_eval(model, ids, mode):
        from sklearn.metrics import f1_score
        pv, pt, pp = mc_predict(model, ids, mode)
        e = pv - vmax[ids]
        return {"n": int(len(ids)), "mae_kt": float(np.abs(e).mean()), "rmse_kt": float(np.sqrt((e ** 2).mean())),
                "bias_kt": float(e.mean()), "t_number_mae": float(np.abs(pt - tnum[ids]).mean()),
                "class_acc": float((pp.argmax(1) == cls[ids]).mean()),
                "macro_f1": float(f1_score(cls[ids], pp.argmax(1), labels=list(range(7)), average="macro", zero_division=0)),
                "wv_available": float(masks[ids, 1].mean()), "pmw_available": float(masks[ids, 2].mean())}

    def train_mc(tag, tr_ids, va_ids, epochs, lr, init=None, bs=96):
        best_p, last_p = os.path.join(OUT, f"{tag}_best.pt"), os.path.join(OUT, f"{tag}_last.pt")
        model = MultiChannelIntensityNet(pretrained=init is None and not LOCAL_TEST).to(DEVICE)
        if init is not None:
            model.load_state_dict(init.state_dict())
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        steps = max(1, len(tr_ids) // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.15)
        scaler = torch.amp.GradScaler(enabled=AMP)
        start, best = 0, 1e9
        if os.path.exists(last_p):
            ck = torch.load(last_p, map_location=DEVICE, weights_only=False)
            if ck.get("done"):
                model.load_state_dict(torch.load(best_p, map_location=DEVICE, weights_only=False)["state_dict"]); return model
            model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            start, best = ck["epoch"] + 1, ck["best"]
        bins = np.clip((vmax[tr_ids] // 10).astype(int), 0, 20)
        w = torch.as_tensor(1 / np.sqrt(np.bincount(bins, minlength=21)[bins]), device=DEVICE)
        tr_t = torch.as_tensor(tr_ids, device=DEVICE)
        cw = torch.as_tensor(1 / np.sqrt(np.bincount(cls[tr_ids], minlength=7) + 1), device=DEVICE, dtype=torch.float32)
        ce = nn.CrossEntropyLoss(weight=cw / cw.mean(), label_smoothing=0.05)
        for ep in range(start, epochs):
            model.train(); t0 = time.time(); tot = 0.0
            order = tr_t[torch.multinomial(w, steps * bs, replacement=True)]
            for i in range(steps):
                b = order[i * bs:(i + 1) * bs]
                x = rotate(mc_input(X_GPU[b], M_GPU[b], "train"))
                x = torch.cat([x[:, :3] * (1 + 0.1 * torch.randn(len(b), 1, 1, 1, device=DEVICE)), x[:, 3:]], 1)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                    pv, pt, pc = model(x)
                    loss = (F.huber_loss(pv.float(), (V_T[b] - V_MU) / V_SD, delta=0.5)
                            + 0.5 * F.huber_loss(pt.float(), (T_T[b] - T_MU) / T_SD, delta=0.5) + 0.3 * ce(pc.float(), C_T[b]))
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item()
            m = mc_eval(model, va_ids, "ir+wv+pmw")
            if m["mae_kt"] < best:
                best = m["mae_kt"]; torch.save({"state_dict": model.state_dict()}, best_p)
            torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": ep, "best": best, "done": ep == epochs - 1}, last_p)
            print(f"[{tag}] ep {ep+1}/{epochs} loss {tot/steps:.3f} val MAE {m['mae_kt']:.2f} kt  {time.time()-t0:.0f}s")
        model.load_state_dict(torch.load(best_p, map_location=DEVICE, weights_only=False)["state_dict"])
        return model

    pre = train_mc("MC_pretrain", idx["x_train"], idx["x_val"], EPA["pre"], 1e-3)
    mc_model = train_mc("MC_finetune", idx["io_train"], idx["io_val"], EPA["ft"], 3e-4, init=pre)
    mc_metrics = {mode: {"val": mc_eval(mc_model, idx["io_val"], mode), "test": mc_eval(mc_model, idx["io_test"], mode)}
                  for mode in ["ir", "ir+wv", "ir+wv+pmw"]}
    p_ir = mc_predict(mc_model, idx["io_test"], "ir")[0]
    p_all = mc_predict(mc_model, idx["io_test"], "ir+wv+pmw")[0]
    print(f"extra channels change the wind estimate by {np.abs(p_all - p_ir).mean():.2f} kt on average")
    pmw_ids = idx["io_test"][masks[idx["io_test"], 2]]
    if len(pmw_ids):
        mc_metrics["pmw_frames_only"] = {m: mc_eval(mc_model, pmw_ids, m) for m in ["ir", "ir+wv", "ir+wv+pmw"]}
    for mode, r in mc_metrics.items():
        if "test" in r:
            print(f"{mode:10s} test MAE {r['test']['mae_kt']:.2f} kt  RMSE {r['test']['rmse_kt']:.2f}  class acc {r['test']['class_acc']:.3f}")
    cfg = {"channels": MC_CHANNELS, "img_size": IMG, "ir_range": IR_RANGE, "wv_range": WV_RANGE, "pmw_max": PMW_MAX,
           "v_mu": V_MU, "v_sd": V_SD, "t_mu": T_MU, "t_sd": T_SD, "classes": IMD_CODES, "crop_px": 201, "res_deg": 0.07}
    torch.save({"state_dict": mc_model.state_dict(), "config": cfg}, os.path.join(OUT, "mc_intensity_model.pt"))
    json.dump(mc_metrics, open(os.path.join(OUT, "metrics_multichannel.json"), "w"), indent=1)
    del X_GPU
    torch.cuda.empty_cache() if AMP else None

# %%
# 6. Part B — ConvLSTM on the last 8 frames (21 h): 24 h wind change + rapid intensification
T_SEQ, STEP_H = 8, 3
EPB = 15 if not LOCAL_TEST else 1
if not done("seq_model.pt"):
    t_h = (when.values.astype("datetime64[h]").astype(np.int64))
    key = pd.Series(np.arange(N), index=pd.MultiIndex.from_arrays([sid, t_h]))
    key = key[~key.index.duplicated()]
    def lookup(offset_h):
        k = pd.MultiIndex.from_arrays([sid, t_h + offset_h])
        return key.reindex(k).values  # float with NaN
    past = np.stack([lookup(-STEP_H * j) for j in range(T_SEQ - 1, -1, -1)], 1)  # (N, T) oldest -> now
    fut = lookup(24)
    ok = valid & np.isfinite(past).all(1) & np.isfinite(fut)
    ok[ok] &= valid[past[ok].astype(int)].all(1) & valid[fut[ok].astype(int)]
    seq_ids = np.where(ok)[0]
    P = past[seq_ids].astype(int)
    dv24 = vmax[fut[seq_ids].astype(int)] - vmax[seq_ids]
    ri = (dv24 >= 30).astype(np.float32)
    scal = np.stack([vmax[seq_ids] / 100.0, (vmax[seq_ids] - vmax[P[:, 0]]) / 30.0], 1).astype(np.float32)
    sp = split[seq_ids]
    print("sequences:", pd.Series(sp).value_counts().to_dict(), "RI rate", ri.mean().round(3))
    assert len(seq_ids) > (100 if not LOCAL_TEST else 10), "Too few 8-frame sequences - send this output to Claude."

    X64 = torch.cat([F.interpolate(torch.from_numpy(np.ascontiguousarray(X[s:s + 2048, :2])).float(), size=(SEQ_IMG, SEQ_IMG),
                                   mode="area").round().byte() for s in range(0, N, 2048)]).to(DEVICE)
    WVOK = torch.from_numpy(masks[:, 1]).to(DEVICE)
    P_T, S_T = torch.from_numpy(P).to(DEVICE), torch.from_numpy(scal).to(DEVICE)
    DV_T, RI_T = torch.from_numpy(dv24.astype(np.float32)).to(DEVICE), torch.from_numpy(ri).to(DEVICE)

    def seq_batch(rows, aug=False):
        fr = X64[P_T[rows]].float() / 255.0                        # (B,T,2,64,64)
        fr[:, :, 1] *= WVOK[P_T[rows]].float()[..., None, None]     # zero missing WV
        fr = (fr - 0.5) / 0.25
        if aug:
            k = int(torch.randint(0, 4, (1,)))
            fr = torch.rot90(fr, k, (3, 4))
        return fr, S_T[rows]

    def seq_eval(model, rows):
        from sklearn.metrics import average_precision_score, roc_auc_score
        model.eval(); pd_, pr = [], []
        with torch.no_grad():
            for i in range(0, len(rows), 256):
                r = torch.as_tensor(rows[i:i + 256], device=DEVICE)
                d, lg = model(*seq_batch(r))
                pd_.append(d.float().cpu() * 20); pr.append(torch.sigmoid(lg.float()).cpu())
        d, p = torch.cat(pd_).numpy(), torch.cat(pr).numpy()
        y, t = ri[rows], dv24[rows]
        res = {"n": int(len(rows)), "ri_cases": int(y.sum()), "dv24_mae_kt": float(np.abs(d - t).mean()),
               "persistence_dv24_mae_kt": float(np.abs(t).mean())}
        if 0 < y.sum() < len(y):
            res.update(ri_pr_auc=float(average_precision_score(y, p)), ri_roc_auc=float(roc_auc_score(y, p)), ri_base_rate=float(y.mean()))
        return res, d, p

    def train_seq(tag, use_images):
        rows_tr = np.where(np.isin(sp, ["x_train", "io_train"]))[0]
        rows_va = np.where(np.isin(sp, ["x_val", "io_val"]))[0]
        model = SeqIntensityNet(use_images=use_images).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        bs = 64; steps = max(1, len(rows_tr) // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPB * steps, pct_start=0.15)
        pos_w = torch.tensor((1 - ri[rows_tr].mean()) / max(ri[rows_tr].mean(), 1e-3), device=DEVICE)
        best, best_state = 1e9, None
        for ep in range(EPB):
            model.train(); t0 = time.time()
            perm = torch.as_tensor(np.random.permutation(rows_tr), device=DEVICE)
            for i in range(steps):
                r = perm[i * bs:(i + 1) * bs]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                    d, lg = model(*seq_batch(r, aug=True))
                    loss = F.huber_loss(d.float(), DV_T[r] / 20, delta=0.5) + 0.5 * F.binary_cross_entropy_with_logits(lg.float(), RI_T[r], pos_weight=pos_w)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            m, _, _ = seq_eval(model, rows_va)
            score = m["dv24_mae_kt"] - 5 * m.get("ri_pr_auc", 0)
            if score < best:
                best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"[{tag}] ep {ep+1}/{EPB} val dv24 MAE {m['dv24_mae_kt']:.2f} kt  RI PR-AUC {m.get('ri_pr_auc', float('nan')):.3f}  {time.time()-t0:.0f}s")
        model.load_state_dict(best_state)
        return model

    seq_img = train_seq("ConvLSTM", True)
    seq_scalar = train_seq("scalars_only", False)
    seq_metrics = {}
    for name, mdl in [("convlstm_images", seq_img), ("scalars_only", seq_scalar)]:
        seq_metrics[name] = {s: seq_eval(mdl, np.where(np.isin(sp, g))[0])[0]
                             for s, g in [("val_all", ["x_val", "io_val"]), ("nio_val", ["io_val"]), ("nio_test", ["io_test"])]}
        t = seq_metrics[name]["nio_test"]
        print(f"{name:16s} NIO test dv24 MAE {t['dv24_mae_kt']:.2f} (persistence {t['persistence_dv24_mae_kt']:.2f})  RI PR-AUC {t.get('ri_pr_auc', float('nan')):.3f}")
    rows_nio = np.where(np.isin(sp, ["io_val", "io_test"]))[0]
    _, d, p = seq_eval(seq_img, rows_nio)
    pd.DataFrame({"idx": seq_ids[rows_nio], "time": when.values[seq_ids[rows_nio]], "sid": sid[seq_ids[rows_nio]],
                  "dv24_pred": d, "ri_prob": p, "dv24_obs": dv24[rows_nio]}).to_parquet(os.path.join(OUT, "seq_predictions.parquet"), index=False)
    torch.save({"state_dict": seq_img.state_dict(), "config": {"t_seq": T_SEQ, "step_h": STEP_H, "img": SEQ_IMG,
                "ir_range": IR_RANGE, "wv_range": WV_RANGE, "scalars": ["vmax/100", "dv_past21h/30"]}},
               os.path.join(OUT, "seq_model.pt"))
    json.dump(seq_metrics, open(os.path.join(OUT, "metrics_convlstm.json"), "w"), indent=1)
    del X64
    torch.cuda.empty_cache() if AMP else None

# %%
# 7. GridSat scene lists: detector training/test scenes + 3-hourly demo storms
MAX_SCENES = int(os.environ.get("MAX_SCENES", "900" if not LOCAL_TEST else "40"))
GRIDSAT_GAP = (pd.Timestamp("2023-04-05"), pd.Timestamp("2024-01-01"))   # not yet in the GridSat archive
rng = np.random.default_rng(0)
act = ib[(ib.wind_imd >= 17) & ib.LAT.between(0, 30) & ib.LON.between(40, 100)]
syn = act[act.time.dt.hour.isin([0, 12])]
syn_times = pd.Series(sorted(syn.time.unique()))
yr = syn_times.dt.year
def one_per_day(ts):
    return ts[ts.dt.hour == np.where(ts.dt.day % 2 == 0, 0, 12)]
train_t = one_per_day(syn_times[(yr >= 2005) & (yr <= 2015)])
val_t = syn_times[yr == 2016]
test_t = one_per_day(syn_times[(yr >= 2017) & (yr <= 2024)])
test_t = test_t[~test_t.between(*GRIDSAT_GAP)]
busy = set()
for t in act.time:
    for dh in range(-24, 25, 12):
        busy.add(t + pd.Timedelta(hours=dh))
def negatives(y0, y1, n):
    days = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="12h")
    days = days[~days.isin(list(busy)) & ~((days >= GRIDSAT_GAP[0]) & (days < GRIDSAT_GAP[1]))]
    return pd.Series(sorted(rng.choice(days, size=min(n, len(days)), replace=False)))
neg_tr, neg_te = (negatives(2005, 2015, 100), negatives(2017, 2024, 30)) if not LOCAL_TEST else (negatives(2005, 2015, 4), negatives(2017, 2024, 2))

DEMO = [("OCKHI", 2017), ("FANI", 2019), ("AMPHAN", 2020), ("TAUKTAE", 2021), ("DANA", 2024), ("VARDAH", 2016)]
DEMO_MAX = 72 if not LOCAL_TEST else 10
demo_rows = []
for name, season in DEMO:
    s = ib[(ib.NAME == name) & (ib.SEASON == season)]
    s = s[s.wind_imd >= 17]
    if s.empty:
        continue
    ts = pd.date_range(s.time.min().floor("3h") - pd.Timedelta(hours=21), s.time.max().floor("3h"), freq="3h")[:DEMO_MAX]
    demo_rows += [(t, name) for t in ts]
demo_t = pd.Series([t for t, _ in demo_rows])

def cap(s, n):
    n = max(int(n), 0)
    return s if len(s) <= n else s.iloc[np.sort(rng.choice(len(s), n, replace=False))]
budget = max(MAX_SCENES - len(demo_t), 0)
train_t, val_t, test_t = cap(train_t, int(budget * 0.45)), cap(val_t, int(budget * 0.1)), cap(test_t, int(budget * 0.3))
scene_split = pd.concat([pd.DataFrame({"time": train_t, "split": "train"}), pd.DataFrame({"time": neg_tr, "split": "train"}),
                         pd.DataFrame({"time": val_t, "split": "val"}), pd.DataFrame({"time": test_t, "split": "test"}),
                         pd.DataFrame({"time": neg_te, "split": "test"})]).drop_duplicates("time").reset_index(drop=True)
all_times = sorted(set(scene_split.time) | set(demo_t))
print("scenes:", scene_split.split.value_counts().to_dict(), "| demo:", len(demo_t), "| total to fetch:", len(all_times))

# %%
# 8. Download GridSat scenes (cached to Drive; parallel)
from concurrent.futures import ThreadPoolExecutor, as_completed
def cache_path(t):
    return os.path.join(CACHE, pd.Timestamp(t).strftime("%Y%m%d%H") + ".npz")

def fake_scene(t):
    lat, lon = scene_axes()
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    bt = 290 + np.random.default_rng(int(pd.Timestamp(t).value % 2**31)).normal(0, 3, yy.shape)
    for r in ib[ib.time == t].itertuples():
        d2 = (yy - r.LAT) ** 2 + (xx - r.LON) ** 2
        bt -= (r.wind_imd / 120) * 90 * np.exp(-d2 / 2.0)
    return {"ir": bt_to_u8(bt, IR_RANGE), "wv": bt_to_u8(bt - 20, WV_RANGE), "lat": lat, "lon": lon}

def get_scene(t):
    p = cache_path(t)
    if os.path.exists(p):
        return "cached"
    for attempt in range(3):
        try:
            sc = fake_scene(t) if LOCAL_TEST else fetch_scene(pd.Timestamp(t))
            np.savez_compressed(p, ir=sc["ir"], wv=sc["wv"], lat=sc["lat"], lon=sc["lon"])
            return "ok"
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return "missing"
        except Exception:
            time.sleep(3 * (attempt + 1))
    return "failed"

todo = [t for t in all_times if not os.path.exists(cache_path(t))]
print(f"{len(all_times) - len(todo)} cached, downloading {len(todo)} (~35 MB each, trimmed to the NIO box) ...")
t0, status = time.time(), {}
with ThreadPoolExecutor(12) as ex:
    futs = {ex.submit(get_scene, t): t for t in todo}
    for n, fu in enumerate(as_completed(futs), 1):
        status[futs[fu]] = fu.result()
        if n % 25 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el/60:.1f} min elapsed, ETA {el/n*(len(todo)-n)/60:.1f} min  {pd.Series(status).value_counts().to_dict()}")
available = [t for t in all_times if os.path.exists(cache_path(t))]
print("scenes available:", len(available), "of", len(all_times))

# %%
# 9. Part C — full-basin detector (CenterNet-style heatmaps at 1/4 resolution)
_z = np.load(cache_path(available[0])) if available else None
LAT_AX, LON_AX = (_z["lat"], _z["lon"]) if _z is not None else scene_axes()   # exact GridSat axes
H0, W0 = len(LAT_AX), len(LON_AX)
HP, WP = 448, 864                       # padded to a multiple of 32
STRIDE, SIGMA = 4, 1.5
EPC = int(os.environ.get("DET_EPOCHS", 60 if not LOCAL_TEST else 2))

def load_scene(t):
    z = np.load(cache_path(t))
    x = np.zeros((2, HP, WP), np.uint8)
    x[0, :H0, :W0], x[1, :H0, :W0] = z["ir"][:H0, :W0], z["wv"][:H0, :W0]
    return x

def truth_at(t):
    r = ib[(ib.time == t) & (ib.wind_imd >= 17) & ib.LAT.between(0, 30) & ib.LON.between(40, 100)]
    return [(0 if w >= 34 else 1, la, lo) for w, la, lo in zip(r.wind_imd, r.LAT, r.LON)]

def heat(truth):
    hm = np.zeros((2, HP // STRIDE, WP // STRIDE), np.float32)
    yy, xx = np.mgrid[0:HP // STRIDE, 0:WP // STRIDE]
    for c, la, lo in truth:
        cy, cx = (la - LAT_AX[0]) / RES / STRIDE, (lo - LON_AX[0]) / RES / STRIDE
        iy, ix = int(round(cy)), int(round(cx))
        if not (0 <= iy < hm.shape[1] and 0 <= ix < hm.shape[2]):
            continue
        hm[c] = np.maximum(hm[c], np.exp(-((yy - iy) ** 2 + (xx - ix) ** 2) / (2 * SIGMA ** 2)))  # peak exactly 1 at the centre cell
    return hm

def focal(logits, hm):
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = hm.eq(1).float()
    neg = 1 - pos
    pl = -torch.log(p) * (1 - p) ** 2 * pos
    nl = -torch.log(1 - p) * p ** 2 * (1 - hm) ** 4 * neg
    return (pl.sum() + nl.sum()) / pos.sum().clamp(min=1)

avail = set(available)
sc = scene_split[scene_split.time.isin(avail)].reset_index(drop=True)
if not done("detector.pt") and len(sc):
    S_X = torch.from_numpy(np.stack([load_scene(t) for t in sc.time]))          # CPU uint8
    S_T = [truth_at(t) for t in sc.time]
    S_H = torch.from_numpy(np.stack([heat(tr) for tr in S_T]))
    tr_rows, va_rows = np.where(sc.split == "train")[0], np.where(sc.split == "val")[0]
    det = BasinDetector(pretrained=not LOCAL_TEST).to(DEVICE)
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    CROP, BS = 256, 16
    steps = max(1, len(tr_rows) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPC * steps, pct_start=0.1)

    def sample_crop(i):
        tr = S_T[i]
        if tr and rng.random() < 0.6:
            _, la, lo = tr[rng.integers(len(tr))]
            cy, cx = int((la - LAT_AX[0]) / RES), int((lo - LON_AX[0]) / RES)
            y0 = int(np.clip(cy - CROP // 2 + rng.integers(-90, 90), 0, HP - CROP)); x0 = int(np.clip(cx - CROP // 2 + rng.integers(-90, 90), 0, WP - CROP))
        else:
            y0, x0 = int(rng.integers(0, HP - CROP)), int(rng.integers(0, WP - CROP))
        y0, x0 = y0 // STRIDE * STRIDE, x0 // STRIDE * STRIDE
        return (S_X[i, :, y0:y0 + CROP, x0:x0 + CROP],
                S_H[i, :, y0 // STRIDE:(y0 + CROP) // STRIDE, x0 // STRIDE:(x0 + CROP) // STRIDE])

    for ep in range(EPC):
        det.train(); t0 = time.time(); tot = 0.0
        perm = rng.permutation(tr_rows)
        for s in range(steps):
            rows = perm[s * BS:(s + 1) * BS]
            xs, hs = zip(*[sample_crop(i) for i in rows])
            x = (torch.stack(xs).to(DEVICE).float() / 255 - 0.5) / 0.25
            h = torch.stack(hs).to(DEVICE)
            k = int(rng.integers(0, 4))
            x, h = torch.rot90(x, k, (2, 3)), torch.rot90(h, k, (2, 3))
            x = x * (1 + 0.1 * torch.randn(len(rows), 1, 1, 1, device=DEVICE))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                loss = focal(det(x), h)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if (ep + 1) % 10 == 0 or ep == EPC - 1:
            print(f"[detector] ep {ep+1}/{EPC} loss {tot/steps:.3f}  {time.time()-t0:.0f}s")
    torch.save({"state_dict": det.state_dict()}, os.path.join(WORK, "detector_tmp.pt"))

    @torch.no_grad()
    def detect_all(rows):
        det.eval(); out = []
        for i in rows:
            x = (S_X[i:i + 1].to(DEVICE).float() / 255 - 0.5) / 0.25
            out.append(det(x)[0].float().cpu())
        return out

    def hav(la0, lo0, la1, lo1):
        p0, p1 = np.radians(la0), np.radians(la1)
        a = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(np.radians(lo1 - lo0) / 2) ** 2
        return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    def score(logit_list, rows, thr, match_km=200):
        tp = fp = fn = 0; errs = []; cls_ok = cls_n = 0
        for lg, i in zip(logit_list, rows):
            pk = heatmap_peaks(lg, thr)
            dets = sorted([(sc_, c, LAT_AX[0] + (y + 0.5) * STRIDE * RES, LON_AX[0] + (x + 0.5) * STRIDE * RES) for c, y, x, sc_ in pk], reverse=True)
            merged = []  # one detection per location (either class)
            for d in dets:
                if all(hav(d[2], d[3], m[2], m[3]) > 150 for m in merged):
                    merged.append(d)
            truth = list(S_T[i]); used = set()
            for d in merged:
                best = min(((hav(d[2], d[3], la, lo), k) for k, (c, la, lo) in enumerate(truth) if k not in used), default=None)
                if best and best[0] <= match_km:
                    used.add(best[1]); tp += 1; errs.append(best[0])
                    cls_n += 1; cls_ok += int(d[1] == truth[best[1]][0])
                else:
                    fp += 1
            fn += len(truth) - len(used)
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return {"threshold": thr, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "tp": tp, "fp": fp, "fn": fn,
                "median_loc_err_km": float(np.median(errs)) if errs else None, "class_acc_matched": cls_ok / max(cls_n, 1)}

    va_logits = detect_all(va_rows)
    sweep = [score(va_logits, va_rows, t) for t in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]]
    best_thr = max(sweep, key=lambda s: s["f1"])["threshold"]
    te_rows = np.where(sc.split == "test")[0]
    te_res = score(detect_all(te_rows), te_rows, best_thr)
    det_metrics = {"threshold": best_thr, "val_sweep": sweep, "test": te_res,
                   "test_scenes": int(len(te_rows)), "train_scenes": int(len(tr_rows)), "match_radius_km": 200}
    print("detector test:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in te_res.items()}))
    torch.save({"state_dict": det.state_dict(), "config": {"classes": DET_CLASSES, "threshold": best_thr, "stride": STRIDE,
                "pad": [HP, WP], "box": BOX, "res": RES, "ir_range": IR_RANGE, "wv_range": WV_RANGE}},
               os.path.join(OUT, "detector.pt"))
    json.dump(det_metrics, open(os.path.join(OUT, "metrics_detector.json"), "w"), indent=1, default=float)

# %%
# 10. Part D — export demo scenes (3-hourly basin IR+WV for the dashboard)
rows = [(t, n) for t, n in demo_rows if os.path.exists(cache_path(t))]
ir = np.stack([np.load(cache_path(t))["ir"] for t, _ in rows]).astype(np.uint8)
wv = np.stack([np.load(cache_path(t))["wv"] for t, _ in rows]).astype(np.uint8)
np.savez_compressed(os.path.join(OUT, "demo_scenes.npz"), ir=ir, wv=wv, time=np.array([str(t) for t, _ in rows]),
                    storm=np.array([n for _, n in rows]), lat=LAT_AX, lon=LON_AX)
print("demo scenes:", pd.Series([n for _, n in rows]).value_counts().to_dict())
print("\nDONE. Download these from", OUT, ":")
for f in ["mc_intensity_model.pt", "metrics_multichannel.json", "seq_model.pt", "metrics_convlstm.json", "seq_predictions.parquet",
          "detector.pt", "metrics_detector.json", "demo_scenes.npz"]:
    p = os.path.join(OUT, f)
    print(f"  {f:28s} {os.path.getsize(p)/1e6:7.1f} MB" if os.path.exists(p) else f"  {f} MISSING")
