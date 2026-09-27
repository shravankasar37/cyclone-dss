"""Pre-compute 0.05 deg land mask and population grids for landfall / exposure queries.

Inputs (data/raw/geo): Natural Earth 10m land polygons; WorldPop 2020 1 km population
for India, Bangladesh, Myanmar, Sri Lanka, Pakistan.
Output: data/processed/geo_grids.npz
"""
import json
import os

import numpy as np
import rasterio
import shapely
from shapely.geometry import box, shape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO = os.path.join(ROOT, "data", "raw", "geo")
OUT = os.path.join(ROOT, "data", "processed", "geo_grids.npz")
LAT0, LAT1, LON0, LON1, RES = -2.0, 32.0, 40.0, 102.0, 0.05
COUNTRIES = {"ind": "India", "bgd": "Bangladesh", "mmr": "Myanmar", "lka": "Sri Lanka", "pak": "Pakistan"}

ny, nx = round((LAT1 - LAT0) / RES), round((LON1 - LON0) / RES)
lat_c = LAT0 + (np.arange(ny) + 0.5) * RES
lon_c = LON0 + (np.arange(nx) + 0.5) * RES

# land mask
region = box(LON0, LAT0, LON1, LAT1)
feats = json.load(open(os.path.join(GEO, "ne_10m_land.geojson"), encoding="utf-8"))["features"]
land = shapely.union_all([shape(f["geometry"]).intersection(region) for f in feats if shape(f["geometry"]).intersects(region)])
shapely.prepare(land)
LON, LAT = np.meshgrid(lon_c, lat_c)
land_mask = shapely.contains_xy(land, LON.ravel(), LAT.ravel()).reshape(ny, nx)
print("land cells:", int(land_mask.sum()), "of", land_mask.size)

# population (sum of 1 km pixels into 0.05 deg cells), plus a country id grid
pop = np.zeros((ny, nx), np.float64)
country = np.zeros((ny, nx), np.int8)
for k, (code, name) in enumerate(COUNTRIES.items(), 1):
    with rasterio.open(os.path.join(GEO, f"{code}_pop_2020_1km.tif")) as src:
        a = src.read(1).astype(np.float64)
        a[(a < 0) | ~np.isfinite(a) | (a == src.nodata)] = 0
        rows, cols = np.nonzero(a)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        xs, ys = np.asarray(xs), np.asarray(ys)
        iy, ix = ((ys - LAT0) / RES).astype(int), ((xs - LON0) / RES).astype(int)
        ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
        np.add.at(pop, (iy[ok], ix[ok]), a[rows[ok], cols[ok]])
        country[iy[ok], ix[ok]] = k
        print(f"{name:10s} {a.sum()/1e6:8.1f} M people")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, land=land_mask, pop=pop.astype(np.float32), country=country,
                    lat=lat_c.astype(np.float32), lon=lon_c.astype(np.float32),
                    countries=np.array(list(COUNTRIES.values())), res=RES)
print("wrote", OUT, f"{os.path.getsize(OUT)/1e6:.1f} MB; total pop in grid {pop.sum()/1e6:.0f} M")
