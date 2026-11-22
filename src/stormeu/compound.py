"""Compound wind and rain: joint exceedance of the cell's own gust and rain quantiles on the same day, relative to independence."""
import numpy as np
import pandas as pd


def compound_ratio(G, R, qg=0.98, qr=0.95, lag=0):
    """Observed frequency of (gust > its qg quantile and rain > its qr quantile of wet days) over the product of the margins.
    Rain quantiles are taken over all winter days. Returns per-cell frequency and the ratio to independence."""
    Gw, Rw = G, R.shift(lag) if lag else R
    eg = Gw > Gw.quantile(qg)
    er = Rw > Rw.quantile(qr)
    both = (eg & er).mean()
    return both, both / ((1 - qg) * (1 - qr))


def by_country(G, R, countries, qg=0.98, qr=0.95, n_boot=400, seed=5):
    """Mean ratio per country with a bootstrap over winters (resampling whole winters)."""
    rng = np.random.default_rng(seed)
    season = np.where(G.index.month >= 7, G.index.year + 1, G.index.year)
    seasons = np.unique(season)
    rows = []
    for c, cols in countries.groupby(countries).groups.items():
        cols = list(cols)
        _, ratio = compound_ratio(G[cols], R[cols], qg, qr)
        point = float(ratio.mean())
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(seasons, len(seasons))
            idx = np.concatenate([np.where(season == s)[0] for s in pick])
            Gb, Rb = G[cols].iloc[idx].reset_index(drop=True), R[cols].iloc[idx].reset_index(drop=True)
            boots.append(float(compound_ratio(Gb, Rb, qg, qr)[1].mean()))
        _, ratio_l1 = compound_ratio(G[cols], R[cols], qg, qr, lag=1)
        rows.append({"country": c, "cells": len(cols), "ratio_same_day": point, "ci_low": float(np.quantile(boots, 0.05)),
                     "ci_high": float(np.quantile(boots, 0.95)), "ratio_rain_previous_day": float(ratio_l1.mean())})
    return pd.DataFrame(rows)
