"""Tiny synthetic file in the TCIR h5 layout (pandas 'info' table + 'matrix' N x 201 x 201 x 4)
used to smoke-test the Colab notebook on CPU."""
import sys, numpy as np, pandas as pd, h5py

out = sys.argv[1]
rng = np.random.default_rng(0)
rows = []
storms = [("IO", 2010), ("IO", 2012), ("IO", 2015), ("IO", 2016), ("IO", 2017),
          ("SH", 2008), ("SH", 2011), ("CPAC", 2009), ("CPAC", 2014), ("SH", 2013)] * 2
for k, (b, y) in enumerate(storms):
    for t in range(20):  # 3-hourly like real TCIR
        rows.append(dict(data_set=b, ID=f"{y}{k:02d}{b[0]}", lon=80 + rng.normal(), lat=15 + t * 0.2,
                         time=f"{y}05{10 + t // 8:02d}{(t % 8) * 3:02d}", Vmax=float(20 + 6 * t + rng.normal() * 5),
                         R35_4qAVG=np.nan, MSLP=1000 - 3 * t))
info = pd.DataFrame(rows)
info.to_hdf(out, key="info", mode="w", format="table")
yy, xx = np.mgrid[0:201, 0:201]
r = np.hypot(yy - 100, xx - 100)
with h5py.File(out, "a") as hf:
    m = hf.create_dataset("matrix", shape=(len(info), 201, 201, 4), dtype=np.float32)
    for i, v in enumerate(info.Vmax):
        bt = 290 - (v / 150) * 90 * np.exp(-(r / 40) ** 2) + rng.normal(0, 2, r.shape)
        if v > 80:
            bt[r < 3] = 280
        m[i, :, :, 0] = bt
        m[i, :, :, 1] = bt - 25
        m[i, :, :, 2] = np.nan
        m[i, :, :, 3] = np.clip((250 - bt) / 10, 0, None) if i % 2 else np.nan
print("wrote", out, len(info), "frames")
