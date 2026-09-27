"""Download ERA5 monthly means over the North Indian Ocean for the RI/genesis models:
SST, and u/v at 200 & 850 hPa (vertical wind shear) plus relative humidity at 600 hPa."""
import os, cdsapi

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "era5")
os.makedirs(OUT, exist_ok=True)
YEARS = [str(y) for y in range(1982, 2026)]
COMMON = {"product_type": ["monthly_averaged_reanalysis"], "year": YEARS,
          "month": [f"{m:02d}" for m in range(1, 13)], "time": ["00:00"],
          "area": [32, 40, -2, 102], "grid": [1.0, 1.0],
          "data_format": "netcdf", "download_format": "unarchived"}
c = cdsapi.Client()
jobs = [("reanalysis-era5-single-levels-monthly-means", {"variable": ["sea_surface_temperature"]}, "era5_sst_monthly.nc"),
        ("reanalysis-era5-pressure-levels-monthly-means",
         {"variable": ["u_component_of_wind", "v_component_of_wind", "relative_humidity"],
          "pressure_level": ["200", "600", "850"]}, "era5_pl_monthly.nc")]
for ds, extra, name in jobs:
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        print("have", name); continue
    c.retrieve(ds, {**COMMON, **extra}).download(path)
    print("saved", path, os.path.getsize(path) // 1024, "KB")
