"""Temporal clustering of storm events: dispersion of seasonal counts, extremal index, inter-arrival times."""
import numpy as np
import pandas as pd
from scipy import stats


def seasonal_counts(ev, seasons, col="sum", q=0.8):
    """Number of events per winter above the q quantile of the event size (all winters, zeros included)."""
    big = ev[ev[col] > ev[col].quantile(q)]
    c = big.groupby("season").size().reindex(list(seasons), fill_value=0)
    return c


def dispersion_test(counts, n_boot=4000, seed=1):
    """Variance to mean ratio of the counts and a parametric-bootstrap p-value against the Poisson model (clustering gives ratio > 1)."""
    m, v = counts.mean(), counts.var(ddof=1)
    rng = np.random.default_rng(seed)
    sim = rng.poisson(m, size=(n_boot, len(counts)))
    ratio = sim.var(axis=1, ddof=1) / np.maximum(sim.mean(axis=1), 1e-9)
    return {"mean": float(m), "var_over_mean": float(v / m), "p_poisson": float((ratio >= v / m).mean()),
            "nb_size": float(m * m / (v - m)) if v > m else None, "winters": int(len(counts))}


def extremal_index(x, thr, run=3):
    """Runs estimator: exceedances separated by at least `run` days below the threshold start new clusters; theta = clusters / exceedances."""
    ex = (x > thr).to_numpy()
    n_ex = int(ex.sum())
    if n_ex == 0:
        return np.nan
    last, clusters = -10 ** 9, 0
    for i in np.where(ex)[0]:
        if i - last > run:
            clusters += 1
        last = i
    return clusters / n_ex


def interarrival(ev):
    """Days between event peaks within the same winter; coefficient of variation (1 for a Poisson process) and a KS test against the exponential."""
    d = ev.sort_values("peak_day")
    gaps = d.groupby("season").peak_day.diff().dt.days.dropna()
    gaps = gaps[gaps > 0]
    rate = 1.0 / gaps.mean()
    ks = stats.kstest(gaps, "expon", args=(0, 1 / rate))
    return {"n": int(len(gaps)), "mean_days": float(gaps.mean()), "cv": float(gaps.std(ddof=1) / gaps.mean()), "ks_p": float(ks.pvalue),
            "share_within_3_days": float((gaps <= 3).mean())}
