"""East Atlantic pattern and Atlantic overturning (AMOC) in CMIP6, historical and ssp585.

One definition for all models: December-February mean sea level pressure of each run is interpolated to the 2.5 degree ERA5 grid
(30-75N, 80W-40E) and projected on the ERA5 EOF patterns of 1951-2025 (EOF1 NAO-like, EOF2 EA-like; their correlations with the NOAA
CPC indices are in results/era5_eof_check.json), after removing the run's own 1985-2014 mean. Changes are 2071-2100 against 1985-2014 in
ERA5 standard deviations of the pattern's index. AMOC change: annual strength over the same two periods.

Inputs from scripts/download_cmip6.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "cmip6"
OUT = ROOT / "results"
ref = np.load(ROOT / "data" / "era5_djf_eof_reference.npz")
check = json.loads((OUT / "era5_eof_check.json").read_text())
SIGN = {"NAO": np.sign(check["EOF1"]["corr_CPC_NAO"]), "EA": np.sign(check["EOF2"]["corr_CPC_EA"])}
WEIGHT = np.cos(np.deg2rad(ref["lat"]))[:, None]
BASE, LATE = (1985, 2014), (2071, 2100)


def winters(path, years):
    """Complete December-February means (winter named after January) within `years`, on the reference grid, in hPa."""
    d = xr.open_dataset(path).psl.sortby("lat").sortby("lon")
    # a model grid may end short of the box edge by less than one cell: the nearest model value is used there
    d = d.interp(lat=np.clip(ref["lat"], float(d.lat.min()), float(d.lat.max())), lon=np.clip(ref["lon"], float(d.lon.min()), float(d.lon.max()))) / 100.0
    year = (d.time.dt.year + (d.time.dt.month == 12)).values
    d = d.assign_coords(w=("time", year))
    n = pd.Series(year).value_counts()
    keep = sorted(y for y in n.index[n == 3] if years[0] <= y <= years[1])
    return d.groupby("w").mean("time").sel(w=keep).transpose("w", "lat", "lon").values


def index(fields, k):
    return np.einsum("tij,ij->t", fields, ref["patterns"][k] * WEIGHT)


rows = []
for m in sorted({p.name.split("_")[0] for p in D.glob("*_historical_psl.nc")}):
    h, s = winters(D / f"{m}_historical_psl.nc", BASE), winters(D / f"{m}_ssp585_psl.nc", LATE)
    assert np.isfinite(h).all() and np.isfinite(s).all(), m
    clim = h.mean(axis=0)
    out = {"model": m, "winters_base": len(h), "winters_late": len(s)}
    for k, name in ((0, "NAO"), (1, "EA")):
        unit = ref["pcs"][:, k].std(ddof=1)   # the index of the ERA5 winters themselves
        ih, il = SIGN[name] * index(h - clim, k), SIGN[name] * index(s - clim, k)
        out[f"{name}_mean_change_sd"] = float(il.mean() / unit)
        out[f"{name}_model_sd_over_era5"] = float(ih.std(ddof=1) / unit)
        out[f"{name}_std_ratio_late_over_base"] = float(il.std(ddof=1) / ih.std(ddof=1))
    files = [D / f"{m}_{e}_amoc.csv" for e in ("historical", "ssp585")]
    if all(f.exists() for f in files):
        a = [pd.read_csv(f).replace([np.inf, -np.inf], np.nan) for f in files]
        base = a[0][a[0].year.between(*BASE)].amoc.mean()
        late = a[1][a[1].year.between(*LATE)].amoc.mean()
        sv = base / 1025.0 / 1e6   # msftmz is a mass transport in kg/s
        if np.isfinite(sv) and 5 < sv < 40:
            out.update({"amoc_base_sv": float(sv), "amoc_change_pct": float(100 * (late / base - 1))})
        else:
            out["amoc_excluded"] = f"base value not usable ({sv})"
    rows.append(out)
    print(out, flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "cmip6_ea_amoc.csv", index=False)
res = {"n_models": len(df)}
for name in ("EA", "NAO"):
    c = df[f"{name}_mean_change_sd"]
    res[name] = {"mean_change_sd": float(c.mean()), "range": [float(c.min()), float(c.max())], "models_positive": int((c > 0).sum()),
                 "model_sd_over_era5_mean": float(df[f"{name}_model_sd_over_era5"].mean()), "std_ratio_late_over_base_mean": float(df[f"{name}_std_ratio_late_over_base"].mean())}
d2 = df.dropna(subset=["amoc_change_pct"])
res["amoc"] = {"n_models": len(d2), "change_pct_mean": float(d2.amoc_change_pct.mean()), "change_pct_range": [float(d2.amoc_change_pct.min()), float(d2.amoc_change_pct.max())]}
for name in ("EA", "NAO"):
    r, p = spearmanr(d2.amoc_change_pct, d2[f"{name}_mean_change_sd"])
    res["amoc"][f"spearman_vs_{name}_shift"] = {"r": float(r), "p": float(p)}
(OUT / "cmip6_ea_amoc.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
