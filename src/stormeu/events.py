"""Storm events and loss indices from daily gust footprints."""
import numpy as np
import pandas as pd

KNOWN = {"Anatol": "1999-12-03", "Lothar": "1999-12-26", "Jeanett": "2002-10-27", "Gudrun": "2005-01-08", "Kyrill": "2007-01-18",
         "Emma": "2008-03-01", "Klaus": "2009-01-24", "Xynthia": "2010-02-28", "Christian": "2013-10-28", "Xaver": "2013-12-05",
         "Niklas": "2015-03-31", "Burglind": "2018-01-03", "Friederike": "2018-01-18", "Sabine": "2020-02-10", "Eunice": "2022-02-18",
         "Ciaran": "2023-11-02"}


def exceedance(G, q=0.98, p=2.0):
    """Relative-threshold index per cell and day: max(0, v / v_q - 1) ** p, v_q the cell's own gust quantile (model A of the
    sibling repository unrisk-windstorm-uq, with the quantile and the power fixed)."""
    vq = G.quantile(q)
    return ((G / vq - 1).clip(lower=0)) ** p, vq


def country_index(L, countries):
    """Mean index over the cells of each country (uniform weights: no population grid was used)."""
    out = {c: L[cols].mean(axis=1) for c, cols in countries.groupby(countries).groups.items()}
    out["ALL"] = L[countries.index].mean(axis=1)
    return pd.DataFrame(out)


def define_events(day_index, min_share=None, gap=1, threshold=None):
    """Events: runs of days (with the default gap=1, strictly consecutive days) on which the domain index exceeds `threshold`.
    Returns a DataFrame with start, end, days, peak_day, sum and max of the daily index."""
    thr = float(day_index.quantile(0.95)) if threshold is None else threshold
    on = day_index[day_index > thr]
    rows, cur = [], None
    for t, v in on.items():
        if cur is not None and (t - cur["last"]).days <= gap + 0:
            cur["last"] = t
            cur["vals"].append((t, v))
        else:
            if cur is not None:
                rows.append(cur)
            cur = {"start": t, "last": t, "vals": [(t, v)]}
    if cur is not None:
        rows.append(cur)
    ev = pd.DataFrame([{"start": r["start"], "end": r["last"], "days": (r["last"] - r["start"]).days + 1,
                        "peak_day": max(r["vals"], key=lambda x: x[1])[0], "sum": sum(v for _, v in r["vals"]),
                        "max": max(v for _, v in r["vals"])} for r in rows])
    ev["season"] = np.where(ev["start"].dt.month >= 7, ev["start"].dt.year + 1, ev["start"].dt.year)
    return ev, thr


def check_known(ev, top=25):
    """Which named storms fall into the top `top` events of the domain by summed index?"""
    r = ev.sort_values("sum", ascending=False).reset_index(drop=True)
    res = {}
    for name, d in KNOWN.items():
        d = pd.Timestamp(d)
        hit = r[(r.start - pd.Timedelta(days=1) <= d) & (r.end + pd.Timedelta(days=1) >= d)]
        res[name] = int(hit.index[0] + 1) if len(hit) else None
    return res
