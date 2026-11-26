"""Load the gridded daily gust maximum and rain sum (ERA5 through the Open-Meteo archive) and assign points to countries."""
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

COUNTRIES = {"DE": "Germany", "FR": "France", "BE": "Belgium", "NL": "Netherlands", "GB": "United Kingdom"}


def load_points(folder):
    gust, rain, coords = {}, {}, {}
    for f in sorted(Path(folder).glob("pt_*.parquet")):
        _, la, lo = f.stem.split("_")
        d = pd.read_parquet(f)
        key = (float(la), float(lo))
        gust[key] = d["gust_max"]
        rain[key] = d["rain_sum"]
        coords[key] = key
    G = pd.DataFrame(gust).sort_index(axis=1)
    R = pd.DataFrame(rain)[G.columns]
    return G, R


def assign_countries(columns, ne_zip):
    world = gpd.read_file(f"zip://{ne_zip}")
    names = {v: k for k, v in COUNTRIES.items()}
    world = world[world["NAME"].isin(names)].copy()
    world["code"] = world["NAME"].map(names)
    out = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
        wide = world.geometry.buffer(0.35)   # 0.35 degrees on purpose: coarse 110m coastlines, 1 degree points near the coast
    for la, lo in columns:
        hit = world[wide.contains(Point(lo, la))]
        if len(hit):
            inside = world[world.geometry.contains(Point(lo, la))]
            out[(la, lo)] = (inside if len(inside) else hit).iloc[0]["code"]
    return pd.Series(out)


def winter_mask(index, months=(10, 11, 12, 1, 2, 3)):
    return pd.DatetimeIndex(index).month.isin(months)


def distance_km(columns):
    la = np.radians(np.array([c[0] for c in columns]))
    lo = np.radians(np.array([c[1] for c in columns]))
    dla, dlo = la[:, None] - la[None], lo[:, None] - lo[None]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None] * np.sin(dlo / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))
