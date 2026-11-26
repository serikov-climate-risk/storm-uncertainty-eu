"""A spatial t-copula with distance decay and a day-to-day AR(1) in the normal scores, GPD-tailed margins per cell.

This is a statistical emulator of daily gust fields, NOT a max-stable process: asymptotic dependence is controlled by the degrees of
freedom only; the spatial fit is checked on held-out winters (joint exceedance of the 99th percentile by distance).
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats


class Margins:
    """Empirical quantile function below the tail quantile, generalised Pareto above, one per cell."""

    def __init__(self, G, tail_q=0.95):
        self.cols = list(G.columns)
        self.tail_q = tail_q
        self.sorted = {c: np.sort(G[c].dropna().to_numpy()) for c in self.cols}
        self.u0 = {c: float(np.quantile(self.sorted[c], tail_q)) for c in self.cols}
        self.gpd = {}
        for c in self.cols:
            ex = self.sorted[c][self.sorted[c] > self.u0[c]] - self.u0[c]
            xi, _, sc = stats.genpareto.fit(ex, floc=0)
            self.gpd[c] = (min(max(xi, -0.3), 0.5), sc)

    def ppf(self, U):
        """U: (days, cells) uniforms in the column order of self.cols."""
        out = np.empty_like(U)
        for j, c in enumerate(self.cols):
            s = self.sorted[c]
            u = U[:, j]
            body = u <= self.tail_q
            out[body, j] = np.interp(u[body], (np.arange(len(s)) + 0.5) / len(s), s)
            xi, sc = self.gpd[c]
            p = (1 - u[~body]) / (1 - self.tail_q)
            out[~body, j] = self.u0[c] + stats.genpareto.isf(np.clip(p, 1e-12, 1), xi, loc=0, scale=sc)
        return out


def normal_scores(G):
    r = G.rank(axis=0) / (G.notna().sum() + 1)
    return pd.DataFrame(stats.norm.ppf(r), index=G.index, columns=G.columns)


def fit_decay(Z, dist):
    """Fit corr(d) = (1 - nugget) * exp(-d / ell) to the empirical correlation of the normal scores."""
    C = np.corrcoef(Z.to_numpy().T)
    iu = np.triu_indices_from(C, 1)
    d, c = dist[iu], C[iu]

    def loss(p):
        ell, nug = p
        return float(np.mean((c - (1 - nug) * np.exp(-d / ell)) ** 2))

    res = optimize.minimize(loss, x0=[500.0, 0.05], bounds=[(30, 5000), (0, 0.5)])
    return {"ell_km": float(res.x[0]), "nugget": float(res.x[1]), "rmse": float(np.sqrt(res.fun))}


def lag1(Z):
    a = Z.to_numpy()
    return float(np.mean([np.corrcoef(a[:-1, j], a[1:, j])[0, 1] for j in range(a.shape[1])]))


def simulate_scores(R_chol, days, rho, nu, rng, n_seasons=1, kappa=0.0):
    """(n_seasons*days, cells) uniforms from the t copula with AR(1) normal scores.

    The chi-square mixing variable of the t copula follows its own AR(1) through a Gaussian transform (coefficient `kappa`), which keeps
    the t margins and gives stormy spells some persistence; kappa = 0 draws it independently every day."""
    n = R_chol.shape[0]
    out = np.empty((n_seasons, days, n))
    z = rng.standard_normal((n_seasons, n)) @ R_chol.T
    a = rng.standard_normal(n_seasons)
    for t in range(days):
        eps = rng.standard_normal((n_seasons, n)) @ R_chol.T
        z = rho * z + np.sqrt(1 - rho ** 2) * eps
        if np.isfinite(nu):
            a = kappa * a + np.sqrt(1 - kappa ** 2) * rng.standard_normal(n_seasons)
            w = stats.chi2.ppf(np.clip(stats.norm.cdf(a), 1e-12, 1 - 1e-12), nu)[:, None] / nu
            out[:, t] = stats.t.cdf(z / np.sqrt(w), nu)
        else:
            out[:, t] = stats.norm.cdf(z)
    return out.reshape(n_seasons * days, n)


def joint_exceedance_by_distance(X, dist, q=0.99, bins=(0, 150, 300, 600, 1000, 2000)):
    """P(both cells above their own q-quantile) for pairs in distance bins; X (days, cells)."""
    E = (X > np.quantile(X, q, axis=0)).astype(float)
    J = E.T @ E / len(E)
    iu = np.triu_indices_from(J, 1)
    d, j = dist[iu], J[iu]
    res = {}
    for a, b in zip(bins[:-1], bins[1:]):
        m = (d > a) & (d <= b)
        res[f"{a}-{b}"] = float(j[m].mean() / (1 - q)) if m.any() else np.nan     # conditional: P(second | first)
    return res
