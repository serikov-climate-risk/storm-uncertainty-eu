"""Download the inputs of run_all.py that are not committed: daily gust maximum and rain sum at 284 grid points (ERA5 through the
Open-Meteo archive, 1991-2025) and the Natural Earth country outlines.

The free Open-Meteo access is rate limited. The script waits out short-term limits and stops at the daily limit; run it again later,
points already on disk are skipped.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "openmeteo"
NE = DATA / "ne_110m_admin_0_countries.zip"
NE_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
API = "https://archive-api.open-meteo.com/v1/archive"
BOXES = [(47.3, 55.0, 6.0, 15.0), (42.5, 51.0, -4.5, 8.0), (49.5, 53.5, 2.5, 7.0), (50.0, 58.5, -5.5, 1.7)]   # lat0, lat1, lon0, lon1


def grid():
    pts = set()
    for la0, la1, lo0, lo1 in BOXES:
        la = la0
        while la <= la1 + 1e-9:
            lo = lo0
            while lo <= lo1 + 1e-9:
                pts.add((round(la, 1), round(lo, 1)))
                lo += 1.0
            la += 1.0
    return sorted(pts)


def get(params):
    """The JSON answer, or None when the daily limit is reached or the short-term limit does not clear."""
    for attempt in range(12):
        r = requests.get(API, params=params, timeout=300)
        if r.status_code == 429:
            if "Daily" in r.text:
                return None
            time.sleep(60 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not NE.exists():
        r = requests.get(NE_URL, timeout=120)
        r.raise_for_status()
        NE.write_bytes(r.content)
    todo = [(la, lo) for la, lo in grid() if not (OUT / f"pt_{la:.1f}_{lo:.1f}.parquet").exists()]
    print(f"grid points: {len(grid())}, to download: {len(todo)}", flush=True)
    for la, lo in todo:
        j = get({"latitude": la, "longitude": lo, "start_date": "1991-01-01", "end_date": "2025-12-31",
                 "daily": "wind_gusts_10m_max,precipitation_sum", "timezone": "UTC",
                 "models": "era5"})   # explicit: the default of the API mixes in another model from 2017
        if j is None:
            print("rate limit reached, run the script again later")
            return 1
        d = j["daily"]
        pd.DataFrame({"gust_max": d["wind_gusts_10m_max"], "rain_sum": d["precipitation_sum"]},
                     index=pd.to_datetime(d["time"])).to_parquet(OUT / f"pt_{la:.1f}_{lo:.1f}.parquet")
        print(la, lo, flush=True)
        time.sleep(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
