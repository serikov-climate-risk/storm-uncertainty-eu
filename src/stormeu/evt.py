"""Non-stationary peaks-over-threshold model per grid cell: generalised Pareto exceedances of the daily gust with a log-scale that depends
on monthly circulation indices (NAO, EA), constant shape. Exceedances are declustered (runs: gaps of at most `gap` days form one cluster,
the cluster maximum is kept). Cells are fitted one by one; spatial dependence is handled by the copula step, so this is a set of marginal
models mapped in space, not a max-stable or latent spatial process.
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats


DAYS_PER_WINTER = 182.25   # October to March


def decluster(series: pd.Series, u: float, gap: int = 2) -> pd.Series:
    on = series[series > u]
    if on.empty:
        return on
    days = on.index
    new = np.r_[True, (days[1:] - days[:-1]).days > gap]
    group = np.cumsum(new)
    idx = on.groupby(group).idxmax()
    return on.loc[idx.to_numpy()]


def _nll(theta, y, X):
    xi = theta[0]
    ls = theta[1] + X @ theta[2:]
    s = np.exp(ls)
    z = 1 + xi * y / s
    if np.any(z <= 0) or not -0.5 < xi < 0.5:
        return 1e10
    if abs(xi) < 1e-6:
        return float(np.sum(ls + y / s))
    return float(np.sum(ls + (1 + 1 / xi) * np.log(z)))


def fit_cell(y: np.ndarray, X: np.ndarray) -> dict:
    """y: exceedances over the threshold (> 0); X: (n, k) covariates (centred and scaled). Returns shape, intercept, slopes, nll."""
    k = X.shape[1]
    x0 = np.r_[0.0, np.log(max(y.mean(), 1e-6)), np.zeros(k)]
    best = None
    for start_xi in (-0.1, 0.05):
        x0[0] = start_xi
        r = optimize.minimize(_nll, x0, args=(y, X), method="Nelder-Mead", options={"maxiter": 4000, "maxfev": 4000, "xatol": 1e-6, "fatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    return {"xi": float(best.x[0]), "b0": float(best.x[1]), "slopes": best.x[2:].astype(float), "nll": float(best.fun)}


def return_level(u, xi, scale, rate_per_winter, winters):
    """Gust level exceeded once in `winters` winters for a cluster rate `rate_per_winter`."""
    m = rate_per_winter * winters
    if abs(xi) < 1e-6:
        return u + scale * np.log(m)
    return u + scale / xi * (m ** xi - 1)


def fit_grid(G: pd.DataFrame, monthly: pd.DataFrame, q: float = 0.95, winters: int = 20) -> pd.DataFrame:
    """G: daily winter gusts (days x cells); monthly: NAO and EA indexed by month start. One row per cell with likelihood-ratio tests of
    the EA and NAO terms and the return level at EA = +1 and -1 standard deviation (NAO at its mean)."""
    cov = monthly[["NAO", "EA"]]
    cov = (cov - cov.loc[G.index.min().to_period("M").to_timestamp():G.index.max()].mean()) / cov.loc[G.index.min().to_period("M").to_timestamp():G.index.max()].std()
    rows = []
    for c in G.columns:
        s = G[c].dropna()
        u = float(s.quantile(q))
        peaks = decluster(s, u)
        X = cov.reindex(peaks.index.to_period("M").to_timestamp())
        ok = X.notna().all(axis=1).to_numpy()
        y = (peaks.to_numpy() - u)[ok]
        X = X.to_numpy()[ok]
        full = fit_cell(y, X)
        no_ea = fit_cell(y, X[:, [0]])
        no_nao = fit_cell(y, X[:, [1]])
        p_ea = float(stats.chi2.sf(max(2 * (no_ea["nll"] - full["nll"]), 0), 1))
        p_nao = float(stats.chi2.sf(max(2 * (no_nao["nll"] - full["nll"]), 0), 1))
        rate = len(y) / (ok.size and ok.mean() * len(s) / DAYS_PER_WINTER)   # clusters per winter of 182.25 days
        rl = {}
        for tag, ea in (("minus", -1.0), ("mean", 0.0), ("plus", 1.0)):
            rl[tag] = return_level(u, full["xi"], np.exp(full["b0"] + full["slopes"][1] * ea), rate, winters)
        rows.append({"lat": c[0], "lon": c[1], "threshold": u, "clusters": int(len(y)), "xi": full["xi"], "b_nao": full["slopes"][0],
                     "b_ea": full["slopes"][1], "p_nao": p_nao, "p_ea": p_ea, "rl_ea_minus": rl["minus"], "rl_mean": rl["mean"], "rl_ea_plus": rl["plus"]})
    return pd.DataFrame(rows)


def false_discovery(p: np.ndarray, alpha: float = 0.10) -> int:
    """Number of cells significant after the Benjamini-Hochberg correction (field significance)."""
    p = np.sort(np.asarray(p))
    m = len(p)
    ok = np.where(p <= alpha * np.arange(1, m + 1) / m)[0]
    return int(ok.max() + 1) if len(ok) else 0
