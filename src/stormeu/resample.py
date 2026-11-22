"""Importance resampling of loss-model parameters to eight observed German storm losses (GDV), for three structural variants,
and the effect on the 1-in-200 annual aggregate loss from a stochastic event set.

Models (on the German cells of the grid, uniform weights because no population grid was used):
A  relative threshold  l = max(0, v / v_q - 1) ** p
B  absolute threshold  l = max(0, (v - v0) / 10) ** p
C  as A with saturating damage  l = min(l_A, cap)   (the third structural variant)
Event loss = scale * mean over cells of l, in million EUR. Observation error 15% and structural discrepancy 25% on the log scale
(assumptions as in unrisk-windstorm-uq).
"""
import numpy as np
import pandas as pd

GDV = {"Kyrill": ("2007-01-18", 3350), "Jeanett": ("2002-10-27", 1340), "Ylenia/Zeynep": ("2022-02-18", 1250),
       "Friederike": ("2018-01-18", 1160), "Niklas": ("2015-03-31", 730), "Xynthia": ("2010-02-28", 730),
       "Sabine": ("2020-02-10", 600), "Emma": ("2008-03-01", 610)}
PRIOR = {"A": {"q": (0.95, 0.995), "p": (2.0, 4.0), "log10_scale": (3.0, 6.0)},
         "B": {"v0": (14.0, 30.0), "p": (2.0, 4.0), "log10_scale": (3.0, 6.0)},
         "C": {"q": (0.95, 0.995), "p": (1.0, 3.0), "cap": (0.05, 1.0), "log10_scale": (3.0, 6.0)}}
SIG = np.hypot(0.15, 0.25)
QGRID = np.array([0.90, 0.95, 0.97, 0.98, 0.99, 0.995])


def draw(model, n, rng):
    return {k: rng.uniform(lo, hi, n) for k, (lo, hi) in PRIOR[model].items()}


def loss_fn(model, th, F, vq_grid):
    """F: (events, cells) footprints; th: dict of arrays of length n; returns (n, events) losses."""
    n = len(next(iter(th.values())))
    out = np.empty((n, F.shape[0]))
    for i in range(n):
        if model in ("A", "C"):
            j = int(np.clip(np.searchsorted(QGRID, th["q"][i]) - 1, 0, len(QGRID) - 2))
            f = (th["q"][i] - QGRID[j]) / (QGRID[j + 1] - QGRID[j])
            vq = (1 - f) * vq_grid[j] + f * vq_grid[j + 1]
            raw = np.clip(F / vq - 1, 0, None) ** th["p"][i]
            if model == "C":
                raw = np.minimum(raw, th["cap"][i])
        else:
            raw = np.clip((F - th["v0"][i]) / 10.0, 0, None) ** th["p"][i]
        out[i] = raw.mean(axis=1) * 10 ** th["log10_scale"][i]
    return out


def importance_weights(L_storms, obs):
    """Gaussian likelihood on the log scale; returns normalised weights and the effective sample size."""
    z = (np.log(np.maximum(L_storms, 1e-9)) - np.log(obs)[None]) / SIG
    ll = -0.5 * (z ** 2).sum(axis=1)
    w = np.exp(ll - ll.max())
    w /= w.sum()
    return w, float(1.0 / (w ** 2).sum())


def aep_995(annual):
    return float(np.quantile(annual, 0.995))


def annual_loss(L_events, season_idx, n_seasons):
    a = np.zeros((L_events.shape[0], n_seasons))
    for s in range(n_seasons):
        m = season_idx == s
        if m.any():
            a[:, s] = L_events[:, m].sum(axis=1)
    return a


def importance_shape(L1, obs, lo=3.0, hi=6.0):
    """Shape parameters importance-weighted with the scale integrated out analytically.

    L1: (n, storms) modelled losses with unit scale. On the log scale the likelihood is Gaussian with sd SIG, and the log scale has a
    flat prior on [lo, hi] (log10). For each shape draw the best log scale is the mean of log(obs) - log(L1); its conditional
    distribution is normal with variance SIG^2 / m. Weights are the marginal likelihoods (flat prior on the scale is integrated over
    the part inside [lo, hi]); returns weights, effective sample size and one scale draw per shape draw."""
    from scipy.stats import norm

    m = obs.size
    ok = np.ones(len(L1), bool)
    lr = np.log(obs)[None] - np.log(np.maximum(L1, 1e-9))   # a storm with no modelled loss is heavily penalised, not excluded
    mu = lr.mean(axis=1)
    ss = ((lr - mu[:, None]) ** 2).sum(axis=1)
    sd = SIG / np.sqrt(m)
    a, b = lo * np.log(10), hi * np.log(10)
    mass = norm.cdf((b - mu) / sd) - norm.cdf((a - mu) / sd)
    ll = -0.5 * ss / SIG ** 2 + np.log(np.maximum(mass, 1e-300))
    ll = np.where(ok, ll, -np.inf)
    w = np.exp(ll - ll.max())
    w /= w.sum()
    return w, float(1.0 / (w ** 2).sum()), mu, sd, (a, b)


def draw_scale(mu, sd, bounds, rng):
    from scipy.stats import truncnorm

    a, b = bounds
    z = truncnorm.rvs((a - mu) / sd, (b - mu) / sd, loc=mu, scale=sd, random_state=rng)
    return z / np.log(10)
