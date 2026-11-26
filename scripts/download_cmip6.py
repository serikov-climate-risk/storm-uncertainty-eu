"""CMIP6 inputs from the public Pangeo cloud archive (member r1i1p1f1, historical and ssp585): December-February monthly sea level
pressure over the North Atlantic, and the strength of the Atlantic overturning (annual mean of the monthly maximum over depth of msftmz
in the Atlantic basin at 26.5N). Writes data/cmip6/<model>_<experiment>_psl.nc and _amoc.csv; existing files are skipped.

Usage: python scripts/download_cmip6.py [k n]    (worker k of n)
"""
import sys
from pathlib import Path

import gcsfs
import pandas as pd
import xarray as xr

OUT = Path(__file__).resolve().parents[1] / "data" / "cmip6"
MODELS = ["ACCESS-CM2", "ACCESS-ESM1-5", "CESM2-WACCM", "CanESM5", "FGOALS-f3-L", "FGOALS-g3", "INM-CM4-8", "MPI-ESM1-2-HR",
          "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM", "NorESM2-MM"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cat_f = OUT / "pangeo-cmip6.csv"
    if not cat_f.exists():
        pd.read_csv("https://storage.googleapis.com/cmip6/pangeo-cmip6.csv").to_csv(cat_f, index=False)
    cat = pd.read_csv(cat_f)
    cat = cat[cat.member_id == "r1i1p1f1"]
    models = MODELS[int(sys.argv[1])::int(sys.argv[2])] if len(sys.argv) > 2 else MODELS
    fs = gcsfs.GCSFileSystem(token="anon")

    def store(model, exp, var, table):
        r = cat[(cat.source_id == model) & (cat.experiment_id == exp) & (cat.variable_id == var) & (cat.table_id == table)]
        return None if r.empty else r.sort_values("version").iloc[-1].zstore

    for model in models:
        for exp in ("historical", "ssp585"):
            f_psl, f_amoc = OUT / f"{model}_{exp}_psl.nc", OUT / f"{model}_{exp}_amoc.csv"
            t0, t1 = ("1950", "2014") if exp == "historical" else ("2015", "2100")
            if not f_psl.exists():
                try:
                    ds = xr.open_zarr(fs.get_mapper(store(model, exp, "psl", "Amon")), consolidated=True)
                    lon = ds.lon.where(ds.lon <= 180, ds.lon - 360) if float(ds.lon.max()) > 180 else ds.lon
                    ds = ds.assign_coords(lon=lon).sortby("lon") if lon.ndim == 1 else ds
                    sub = ds.psl.sel(time=slice(t0, t1)).sel(lat=slice(20, 80), lon=slice(-90, 40))
                    sub = sub.isel(time=sub.time.dt.month.isin([12, 1, 2]))
                    sub.load().to_netcdf(f_psl)
                    print("ok", f_psl.name, flush=True)
                except Exception as e:   # one missing model must not stop the others
                    print("fail psl", model, exp, str(e)[:150], flush=True)
            if not f_amoc.exists():
                try:
                    ds = xr.open_zarr(fs.get_mapper(store(model, exp, "msftmz", "Omon")), consolidated=True)
                    v = ds.msftmz.sel(time=slice(t0, t1))
                    if "basin" in v.dims:
                        src = ds["sector"] if "sector" in ds.coords else v.basin
                        names = [str(b).lower() for b in src.values]
                        v = v.isel(basin=[i for i, n in enumerate(names) if "atlantic" in n][0])
                    latname = [d for d in v.dims if d in ("lat", "rlat", "y", "latitude")][0]
                    v = v.sel({latname: 26.5}, method="nearest")
                    lev = [d for d in v.dims if d in ("lev", "olevel", "depth", "z")][0]
                    amoc = v.max(lev).resample(time="YS").mean().load()
                    pd.DataFrame({"year": amoc.time.dt.year.values, "amoc": amoc.values}).to_csv(f_amoc, index=False)
                    print("ok", f_amoc.name, flush=True)
                except Exception as e:
                    print("fail amoc", model, exp, str(e)[:150], flush=True)


if __name__ == "__main__":
    main()
