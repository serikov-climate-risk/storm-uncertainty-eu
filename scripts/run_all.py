"""Windstorm and compound wind-rain uncertainty over Western Europe on a 1-degree ERA5 grid (daily gust maximum and rain sum)."""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stormeu import clustering, compound, copula, events, grid, resample  # noqa: E402

DATA = Path(os.environ.get("STORM_DATA", Path.home() / "z_data" / "openmeteo_w_europe"))
NE = Path(os.environ.get("NE_ZIP", Path.home() / "z_data" / "z7" / "ne.zip"))
CPC = Path.home() / "unrisk-windstorm-uq" / "data" / "cpc_tele_index.nh.txt"
N_WINTERS = int(os.environ.get("N_WINTERS", 3000))
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
rng = np.random.default_rng(11)
out = {}
t0 = time.time()

G, R = grid.load_points(DATA)
cty = grid.assign_countries(list(G.columns), NE)
G, R = G[cty.index], R[cty.index]
w = grid.winter_mask(G.index)
Gw, Rw = G[w], R[w]
out["grid"] = {"cells": int(G.shape[1]), "by_country": cty.value_counts().to_dict(), "days_winter": int(w.sum()), "period": [str(G.index[0].date()), str(G.index[-1].date())]}
print(out["grid"], flush=True)

# A. events and loss indices
L, vq = events.exceedance(Gw)
CI = events.country_index(L, cty)
ev, thr = events.define_events(CI["ALL"])
ev.to_csv(RES / "events.csv", index=False)
known = events.check_known(ev)
out["events"] = {"n": int(len(ev)), "threshold_daily_index": thr, "known_storm_rank_by_summed_index": known,
                 "known_in_top25": int(sum(v is not None and v <= 25 for v in known.values())), "known_total": len(known)}
# B. temporal clustering
seasons = sorted(ev.season.unique())
cnt = clustering.seasonal_counts(ev, range(int(Gw.index.year.min()) + 1, int(Gw.index.year.max()) + 1))
out["clustering"] = {"dispersion_top20pct_events": clustering.dispersion_test(cnt), "extremal_index_q95": clustering.extremal_index(CI["ALL"], CI["ALL"].quantile(0.95)),
                     "extremal_index_q99": clustering.extremal_index(CI["ALL"], CI["ALL"].quantile(0.99)), "interarrival": clustering.interarrival(ev)}
# C. circulation covariates: Poisson GLM of monthly exceedance days of each country on NAO and EA
rows = []
for line in CPC.read_text().splitlines():
    m = re.match(r"\s*(\d{4})\s+(\d{1,2})(.*)$", line)
    if m:
        v = re.findall(r"-?\d+\.\d\d", m.group(3))
        if len(v) >= 2:
            rows.append((pd.Timestamp(int(m.group(1)), int(m.group(2)), 1), float(v[0]), float(v[1])))
idx = pd.DataFrame(rows, columns=["t", "NAO", "EA"]).set_index("t")
glm = {}
for c in [c for c in CI.columns]:
    day_on = (CI[c] > CI[c].quantile(0.95))
    mth = day_on.groupby([day_on.index.year, day_on.index.month]).sum()
    mth.index = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in mth.index])
    X = idx.reindex(mth.index).dropna()
    y = mth.reindex(X.index)
    fit = sm.GLM(y, sm.add_constant((X - X.mean()) / X.std()), family=sm.families.Poisson()).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    glm[c] = {"NAO_rate_ratio_per_sd": float(np.exp(fit.params["NAO"])), "NAO_t": float(fit.tvalues["NAO"]),
              "EA_rate_ratio_per_sd": float(np.exp(fit.params["EA"])), "EA_t": float(fit.tvalues["EA"])}
out["circulation_glm"] = glm
# D. compound wind and rain
cb = compound.by_country(Gw, Rw, cty)
cb.to_csv(RES / "compound_by_country.csv", index=False)
out["compound"] = cb.round(2).to_dict("records")
print("A-D done", round(time.time() - t0), flush=True)

# E. spatial dependence: fit on 1991-2010, test on 2011-2025
dist = grid.distance_km(list(Gw.columns))
tr = Gw.index.year <= 2010
Ztr = copula.normal_scores(Gw[tr])
dec = copula.fit_decay(Ztr, dist)
rho = copula.lag1(Ztr)
Rm = (1 - dec["nugget"]) * np.exp(-dist / dec["ell_km"]) + dec["nugget"] * np.eye(len(dist))
Rm = (Rm + Rm.T) / 2 + 1e-6 * np.eye(len(dist))
chol = np.linalg.cholesky(Rm)
marg_tr = copula.Margins(Gw[tr])
obs_test = copula.joint_exceedance_by_distance(Gw[~tr].to_numpy(), dist)
errs = {}
for nu in (3, 6, 12, np.inf):
    Us = copula.simulate_scores(chol, 182, rho, nu, rng, n_seasons=60)
    Xs = marg_tr.ppf(Us)
    sim = copula.joint_exceedance_by_distance(Xs, dist)
    errs[str(nu)] = {"sim": sim, "mae_vs_test": float(np.nanmean([abs(sim[k] - obs_test[k]) for k in obs_test]))}
nu_best = min(errs, key=lambda k: errs[k]["mae_vs_test"])
out["dependence"] = {"fit": dec, "lag1_score_corr": rho, "observed_test_conditional_joint_exceedance_q99": obs_test, "by_nu": errs, "nu_selected": nu_best}
print("E done", nu_best, round(time.time() - t0), flush=True)

# F. stochastic event set from all years, loss by country, AEP and OEP, held against the historical record
Zall = copula.normal_scores(Gw)
marg = copula.Margins(Gw)
nu = float(nu_best)
dec_all = copula.fit_decay(Zall, dist)
Rm = (1 - dec_all["nugget"]) * np.exp(-dist / dec_all["ell_km"]) + dec_all["nugget"] * np.eye(len(dist))
chol = np.linalg.cholesky((Rm + Rm.T) / 2 + 1e-6 * np.eye(len(dist)))
rho_all = copula.lag1(Zall)
cols = list(Gw.columns)
de_idx = [i for i, c in enumerate(cols) if cty[c] == os.environ.get("RESAMPLE_COUNTRY", "DE")] or list(range(len(cols)))
vq_np = Gw.quantile(0.98).to_numpy()
annual_sim = {c: [] for c in CI.columns}
footprints, fp_season = [], []
thr_day = thr
for chunk in range(0, N_WINTERS, 100):
    ns = min(100, N_WINTERS - chunk)
    Us = copula.simulate_scores(chol, 182, rho_all, nu, rng, n_seasons=ns)
    X = marg.ppf(Us).reshape(ns, 182, -1)
    Ls = np.clip(X / vq_np - 1, 0, None) ** 2
    for s in range(ns):
        day = {c: Ls[s][:, [i for i, cc in enumerate(cols) if cty[cc] == c]].mean(axis=1) for c in CI.columns if c != "ALL"}
        day["ALL"] = Ls[s].mean(axis=1)
        for c in CI.columns:
            annual_sim[c].append(float(day[c].sum()))
        on = np.where(day["ALL"] > thr_day)[0]
        # events: consecutive days (gap 1) of the domain index above the threshold; footprint = max gust over the event days
        if len(on):
            groups = np.split(on, np.where(np.diff(on) > 1)[0] + 1)
            for g in groups:
                footprints.append(X[s][g][:, de_idx].max(axis=0))
                fp_season.append(chunk + s)
    print("sim", chunk + ns, round(time.time() - t0), flush=True)
hist_annual = {c: CI[c].groupby(np.where(CI.index.month >= 7, CI.index.year + 1, CI.index.year)).sum() for c in CI.columns}
res_aep = {}
for c in CI.columns:
    a = np.array(annual_sim[c])
    h = hist_annual[c]
    res_aep[c] = {"hist_mean_annual": float(h.mean()), "sim_mean_annual": float(a.mean()), "hist_max_annual": float(h.max()),
                  "sim_q98_annual_1in50": float(np.quantile(a, 0.98)), "sim_q995_annual_1in200": float(np.quantile(a, 0.995)),
                  "hist_q98_annual_1in50": float(h.quantile(0.98)), "sim_sd_over_hist_sd": float(a.std() / h.std())}
out["stochastic_set"] = {"winters": N_WINTERS, "events": len(footprints), "by_region": res_aep}
print("F done", round(time.time() - t0), flush=True)

# G. importance resampling of loss-model parameters to eight GDV storm losses (German cells)
Gde = Gw.iloc[:, de_idx]
vq_grid = np.nanquantile(Gde.to_numpy(), resample.QGRID, axis=0)
storms = {k: v for k, v in resample.GDV.items()}
Fst, obs, names = [], [], []
for k, (d, loss) in storms.items():
    d = pd.Timestamp(d)
    win = Gde.loc[d - pd.Timedelta(days=1):d + pd.Timedelta(days=1)]
    if len(win):
        Fst.append(win.max().to_numpy())
        obs.append(loss)
        names.append(k)
Fst, obs = np.vstack(Fst), np.array(obs, float)
FP = np.vstack(footprints) if footprints else np.empty((0, len(de_idx)))
fps = np.array(fp_season)
imp = {}
for model in ("A", "B", "C"):
    th = resample.draw(model, 20000, rng)
    th_post = dict(th)
    th_post["log10_scale"] = np.zeros(20000)
    L1 = resample.loss_fn(model, th_post, Fst, vq_grid)
    wts, ess, mu, sd, bnd = resample.importance_shape(L1, obs)
    pick = rng.choice(len(wts), size=300, p=wts)
    sc = resample.draw_scale(mu[pick], sd, bnd, rng)
    th_pick = {k: v[pick] for k, v in th.items()}
    th_pick["log10_scale"] = sc
    prior_pick = rng.choice(len(wts), size=300)
    th_prior = {k: v[prior_pick] for k, v in th.items()}
    res = {"ess": ess, "ess_share": ess / len(wts), "distinct_shape_draws_in_posterior_sample": int(len(set(pick.tolist())))}
    for tag, thx in (("posterior", th_pick), ("prior", th_prior)):
        Lev = resample.loss_fn(model, thx, FP, vq_grid) if len(FP) else np.zeros((300, 0))
        ann = resample.annual_loss(Lev, fps, N_WINTERS)
        aep = np.quantile(ann, 0.995, axis=1)
        res[tag] = {"aep_1in200_median": float(np.median(aep)), "aep_5": float(np.quantile(aep, 0.05)), "aep_95": float(np.quantile(aep, 0.95)),
                    "ratio_95_5": float(np.quantile(aep, 0.95) / max(np.quantile(aep, 0.05), 1e-9))}
    Lpost = resample.loss_fn(model, th_pick, Fst, vq_grid)
    res["modelled_storms_posterior_median"] = {n: float(np.median(Lpost[:, i])) for i, n in enumerate(names)}
    imp[model] = res
out["importance_resampling"] = {"storms_used": names, "obs_meur": obs.tolist(), "cells": int(len(de_idx)), "models": imp}
(RES / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("events", "clustering", "importance_resampling")}, indent=2, default=float)[:6000])
print("total seconds", round(time.time() - t0))
