"""Figures of the README from the files in results/."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
S = json.loads((RES / "summary.json").read_text())

# 1. East Atlantic effect on the 20-winter gust
m = pd.read_csv(RES / "evt_cells.csv")
d = m.rl_ea_plus - m.rl_ea_minus
lim = float(np.quantile(np.abs(d), 0.98))
fig, ax = plt.subplots(figsize=(6.4, 5.6))
sc = ax.scatter(m.lon, m.lat, c=d, cmap="RdBu_r", vmin=-lim, vmax=lim, s=52, marker="s")
sig = m.p_ea < 0.05
ax.scatter(m.lon[sig], m.lat[sig], s=8, c="black", label="EA term p < 0.05 (single cell)")
ax.set(xlabel="longitude", ylabel="latitude", title="20-winter gust: EA = +1 minus EA = -1 standard deviation")
ax.set_aspect(1.5)
ax.legend(loc="lower right", fontsize=8)
fig.colorbar(sc, ax=ax, label="km/h", shrink=0.8)
fig.tight_layout()
fig.savefig(FIG / "ea_return_level.png", dpi=150)

# 2. observed and modelled storm losses
imp = S["importance_resampling"]
names, obs = imp["storms_used"], np.array(imp["obs_meur"])
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(names))
ax.bar(x - 0.3, obs, 0.2, label="GDV", color="black")
for k, model in enumerate(("A", "B", "C")):
    med = imp["models"][model]["inferred_error"]["modelled_storms_posterior_median"]
    ax.bar(x - 0.1 + 0.2 * k, [med[n] for n in names], 0.2, label=f"model {model}")
ax.set_xticks(x, names, rotation=30, ha="right")
ax.set_ylabel("insured loss, million EUR")
ax.set_title("Eight storms: the gridded footprints do not rank the losses (posterior medians)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "storm_losses.png", dpi=150)

# 3. spatial dependence on held-out winters
dep = S["dependence"]
bins = list(dep["observed_test_conditional_joint_exceedance_q99"])
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(bins, [dep["observed_test_conditional_joint_exceedance_q99"][b] for b in bins], "ko-", label="observed 2011-2025")
for nu, v in dep["by_nu_at_selected_kappa"].items():
    ax.plot(bins, [v["sim"][b] for b in bins], "--", label=f"t copula, nu = {nu}" if nu != "inf" else "Gaussian copula")
ax.set(xlabel="distance between cells, km", ylabel="P(second cell above its q99 | first)", title="Fitted on 1991-2010, compared on 2011-2025")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "dependence.png", dpi=150)
print("figures written")
