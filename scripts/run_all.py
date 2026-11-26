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
from stormeu import clustering, compound, copula, events, evt, grid, resample  # noqa: E402

DATA = Path(os.environ.get("STORM_DATA", ROOT / "data" / "openmeteo"))
NE = Path(os.environ.get("NE_ZIP", ROOT / "data" / "ne_110m_admin_0_countries.zip"))
CPC = ROOT / "data" / "cpc_tele_index.nh.txt"
N_WINTERS = int(os.environ.get("N_WINTERS", 3000))
N_DRAWS = int(os.environ.get("N_DRAWS", 100000))   # importance-resampling draws per loss model (larger set against a low effective sample share)
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
# C2. non-stationary peaks-over-threshold per cell: GPD scale depending on NAO and EA
mg = evt.fit_grid(Gw, idx)
mg.to_csv(RES / "evt_cells.csv", index=False)
mg["country"] = [cty[(a_, b_)] for a_, b_ in zip(mg.lat, mg.lon)]
out["evt_covariates"] = {"cells": int(len(mg)), "xi_median": float(mg.xi.median()),
    "share_p_ea_below_005": float((mg.p_ea < 0.05).mean()), "share_p_nao_below_005": float((mg.p_nao < 0.05).mean()),
    "fdr10_cells_ea": evt.false_discovery(mg.p_ea.to_numpy()), "fdr10_cells_nao": evt.false_discovery(mg.p_nao.to_numpy()),
    "rl20_change_ea_plus_vs_minus_kmh_by_country": (mg.rl_ea_plus - mg.rl_ea_minus).groupby(mg.country).median().round(3).to_dict()}
print("C2 done", out["evt_covariates"], round(time.time() - t0), flush=True)
# D. compound wind and rain
cb = compound.by_country(Gw, Rw, cty)
cb.to_csv(RES / "compound_by_country.csv", index=False)
out["compound"] = cb.round(2).to_dict("records")
print("A-D done", round(time.time() - t0), flush=True)

# E. dependence. The spatial correlation is fitted to normal scores; the degrees of freedom of the t copula (nu) and the day-to-day
# persistence of its mixing variable (kappa) are chosen on the full record so that the simulated number of events per winter and the
# spread of the winter loss index match the observed ones. The spatial tail is then checked with a fit on 1991-2010 against 2011-2025.
dist = grid.distance_km(list(Gw.columns))
cols = list(Gw.columns)
vq_np = Gw.quantile(0.98).to_numpy()
thr_day = thr
by_country = {c: [i for i, cc in enumerate(cols) if cty[cc] == c] for c in CI.columns if c != "ALL"}
complete = range(int(Gw.index.year.min()) + 1, int(Gw.index.year.max()) + 1)   # whole October-March winters only
season = np.where(CI.index.month >= 7, CI.index.year + 1, CI.index.year)
hist_annual = {c: CI[c].groupby(season).sum().reindex(list(complete)) for c in CI.columns}
hist_events = int(ev.season.isin(list(complete)).sum()) / len(complete)


def setup(frame):
    Z = copula.normal_scores(frame)
    d = copula.fit_decay(Z, dist)
    Rm = (1 - d["nugget"]) * np.exp(-dist / d["ell_km"]) + d["nugget"] * np.eye(len(dist))
    return d, np.linalg.cholesky((Rm + Rm.T) / 2 + 1e-6 * np.eye(len(dist))), copula.lag1(Z), copula.Margins(frame)


dec_all, chol, rho_all, marg = setup(Gw)
de_idx = by_country[os.environ.get("RESAMPLE_COUNTRY", "DE")]


def make_set(nu, kappa, winters, gen):
    """Simulate winters; events are runs of days with the domain index above the observed threshold, footprint = max gust over the run."""
    annual = {c: [] for c in CI.columns}
    fps_, season_, days_on = [], [], 0
    for first in range(0, winters, 100):
        ns = min(100, winters - first)
        X = marg.ppf(copula.simulate_scores(chol, 182, rho_all, nu, gen, n_seasons=ns, kappa=kappa)).reshape(ns, 182, -1)
        Ls = np.clip(X / vq_np - 1, 0, None) ** 2
        for k in range(ns):
            day = {c: Ls[k][:, ix].mean(axis=1) for c, ix in by_country.items()}
            day["ALL"] = Ls[k].mean(axis=1)
            for c in CI.columns:
                annual[c].append(float(day[c].sum()))
            on = np.where(day["ALL"] > thr_day)[0]
            days_on += len(on)
            if len(on):
                for g in np.split(on, np.where(np.diff(on) > 1)[0] + 1):
                    fps_.append(X[k][g][:, de_idx].max(axis=0))
                    season_.append(first + k)
    return {"annual": {c: np.array(v) for c, v in annual.items()}, "footprints": np.vstack(fps_), "season": np.array(season_), "days_on": days_on,
            "events_per_winter": len(fps_) / winters, "sd_ratio": float(np.std(annual["ALL"]) / hist_annual["ALL"].std())}


calib = []
for nu_ in (6.0, 12.0):
    for kappa_ in (0.0, 0.5, 0.6, 0.7, 0.8):
        r = make_set(nu_, kappa_, 1000, np.random.default_rng(5))
        calib.append({"nu": nu_, "kappa": kappa_, "events_per_winter": r["events_per_winter"], "sd_ratio_winter_index": r["sd_ratio"],
                      "score": abs(r["events_per_winter"] / hist_events - 1) + abs(r["sd_ratio"] - 1)})
        print(calib[-1], flush=True)
best = min(calib, key=lambda r: r["score"])
nu, kappa = best["nu"], best["kappa"]

d_tr, chol_tr, rho_tr, marg_tr = setup(Gw[Gw.index.year <= 2010])
obs_test = copula.joint_exceedance_by_distance(Gw[Gw.index.year > 2010].to_numpy(), dist)
errs = {}
for nu_ in (3, 6, 12, np.inf):
    Xs = marg_tr.ppf(copula.simulate_scores(chol_tr, 182, rho_tr, nu_, np.random.default_rng(9), n_seasons=200, kappa=kappa))
    sim = copula.joint_exceedance_by_distance(Xs, dist)
    errs[str(nu_)] = {"sim": sim, "mae_vs_test": float(np.nanmean([abs(sim[k] - obs_test[k]) for k in obs_test]))}
out["dependence"] = {"fit_train": d_tr, "fit_all": dec_all, "lag1_score_corr": rho_all, "calibration_grid": calib, "nu_selected": nu, "kappa_selected": kappa,
                     "observed_test_conditional_joint_exceedance_q99": obs_test, "by_nu_at_selected_kappa": errs}
print("E done", nu, kappa, round(time.time() - t0), flush=True)

# F. stochastic event set from all years, held against the record; a second set without persistence shows the effect of that choice
main_set = make_set(nu, kappa, N_WINTERS, rng)
alt_set = make_set(12.0, 0.0, N_WINTERS, np.random.default_rng(23))
FP, fps = main_set["footprints"], main_set["season"]
res_aep = {}
for c in CI.columns:
    sim_a, h = main_set["annual"][c], hist_annual[c]
    res_aep[c] = {"hist_mean_annual": float(h.mean()), "sim_mean_annual": float(sim_a.mean()), "hist_max_annual": float(h.max()),
                  "sim_q98_annual_1in50": float(np.quantile(sim_a, 0.98)), "sim_q995_annual_1in200": float(np.quantile(sim_a, 0.995)),
                  "hist_q98_annual_1in50": float(h.quantile(0.98)), "sim_sd_over_hist_sd": float(sim_a.std() / h.std())}
freq = {"events_per_winter_hist": hist_events, "events_per_winter_sim": main_set["events_per_winter"],
        "share_days_above_threshold_hist": float((CI["ALL"] > thr_day).mean()), "share_days_above_threshold_sim": main_set["days_on"] / (182 * N_WINTERS),
        "days_per_event_hist": float((CI["ALL"] > thr_day).sum() / len(ev)), "days_per_event_sim": main_set["days_on"] / len(FP)}
out["stochastic_set"] = {"winters": N_WINTERS, "events": len(FP), "frequency_check": freq, "by_region": res_aep,
                         "without_persistence_nu12": {"events_per_winter": alt_set["events_per_winter"], "sd_ratio_winter_index": alt_set["sd_ratio"]}}
print(freq, flush=True)
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
imp = {}


def aep_summary(th_x, event_set):
    ann = resample.annual_loss(resample.loss_fn(model, th_x, event_set["footprints"], vq_grid), event_set["season"], N_WINTERS)
    aep = np.quantile(ann, 0.995, axis=1)
    return {"aep_1in200_median": float(np.median(aep)), "aep_5": float(np.quantile(aep, 0.05)), "aep_95": float(np.quantile(aep, 0.95)),
            "ratio_95_5": float(np.quantile(aep, 0.95) / max(np.quantile(aep, 0.05), 1e-9))}


for model in ("A", "B", "C"):
    th = resample.draw(model, N_DRAWS, rng)
    th_unit = dict(th, log10_scale=np.zeros(N_DRAWS))
    L1 = resample.loss_fn(model, th_unit, Fst, vq_grid)
    prior_pick = rng.choice(N_DRAWS, size=300)
    res = {"prior": aep_summary({k: v[prior_pick] for k, v in th.items()}, main_set)}
    # error on the log scale: fixed at SIG (assumed), or log-uniform on [0.15, 1.5] and weighted by the eight storms as well
    for tag, sig in (("fixed_error", None), ("inferred_error", np.exp(rng.uniform(np.log(0.15), np.log(1.5), N_DRAWS)))):
        wts, ess, mu, sd, bnd = resample.importance_shape(L1, obs, sig=sig)
        pick = rng.choice(N_DRAWS, size=300, p=wts)
        th_pick = {k: v[pick] for k, v in th.items()}
        th_pick["log10_scale"] = resample.draw_scale(mu[pick], sd[pick], bnd, rng)
        med = np.median(resample.loss_fn(model, th_pick, Fst, vq_grid), axis=0)
        r = aep_summary(th_pick, main_set)
        alt = aep_summary(th_pick, alt_set)
        r["without_persistence_nu12"] = {k: alt[k] for k in ("aep_1in200_median", "aep_5", "aep_95")}
        r.update({"ess": ess, "ess_share": ess / N_DRAWS, "distinct_shape_draws_in_posterior_sample": int(len(set(pick.tolist()))),
                  "log_rmse_of_posterior_median_storm_losses": resample.log_rmse(med, obs),
                  "modelled_storms_posterior_median": {n: float(med[i]) for i, n in enumerate(names)}})
        if sig is not None:
            q = np.quantile(sig[pick], [0.05, 0.5, 0.95])
            r["error_sd_log_5_50_95"] = [float(v) for v in q]
        res[tag] = r
    imp[model] = res
out["importance_resampling"] = {"storms_used": names, "obs_meur": obs.tolist(), "cells": int(len(de_idx)), "fixed_error_sd_log": float(resample.SIG),
                               "log_rmse_of_one_constant_for_all_storms": resample.log_rmse(np.full(len(obs), np.exp(np.log(obs).mean())), obs), "models": imp}
(RES / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("events", "clustering", "importance_resampling")}, indent=2, default=float)[:6000])
print("total seconds", round(time.time() - t0))
