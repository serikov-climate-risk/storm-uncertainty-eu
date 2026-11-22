import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormeu import clustering, compound, copula, events, grid, resample  # noqa: E402


def test_extremal_index_of_iid_is_near_one():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.standard_normal(20000))
    assert clustering.extremal_index(x, x.quantile(0.98)) > 0.8


def test_extremal_index_of_clustered_series_is_low():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(6000)
    x = pd.Series(np.repeat(base, 3))
    assert clustering.extremal_index(x, x.quantile(0.98)) < 0.6


def test_dispersion_flags_clustered_counts():
    c = pd.Series(np.random.default_rng(1).negative_binomial(2, 0.3, 60))
    assert clustering.dispersion_test(c)["var_over_mean"] > 1.5


def test_events_merge_consecutive_days():
    idx = pd.date_range("2000-01-01", periods=20)
    s = pd.Series(0.0, index=idx)
    s.iloc[3:6] = 1.0
    s.iloc[12] = 2.0
    ev, _ = events.define_events(s, threshold=0.5)
    assert len(ev) == 2 and ev.days.tolist() == [3, 1]


def test_copula_reproduces_correlation_decay():
    cols = [(50.0, 5.0 + i) for i in range(12)]
    d = grid.distance_km(cols)
    R = 0.9 * np.exp(-d / 400) + 0.1 * np.eye(12)
    chol = np.linalg.cholesky(R + 1e-8 * np.eye(12))
    rng = np.random.default_rng(2)
    U = copula.simulate_scores(chol, 182, 0.3, np.inf, rng, n_seasons=60)
    Z = pd.DataFrame(U).pipe(lambda u: pd.DataFrame(np.log(u / (1 - u))))
    fit = copula.fit_decay(Z, d)
    assert 200 < fit["ell_km"] < 800


def test_marginal_quantiles_match_data():
    rng = np.random.default_rng(3)
    G = pd.DataFrame({"a": rng.gumbel(15, 3, 5000)})
    m = copula.Margins(G)
    q = m.ppf(np.array([[0.5], [0.9], [0.99]]))
    assert abs(q[0, 0] - G.a.median()) < 0.3 and abs(q[2, 0] - G.a.quantile(0.99)) < 2.0


def test_importance_weights_prefer_the_true_shape():
    rng = np.random.default_rng(4)
    base = np.exp(rng.normal(0, 1, 8))
    obs = base * 1000
    L1 = np.vstack([base, base[::-1], np.exp(rng.normal(0, 1, 8))])
    w, ess, mu, sd, b = resample.importance_shape(L1, obs)
    assert w.argmax() == 0 and ess < 3


def test_compound_ratio_is_one_for_independent():
    rng = np.random.default_rng(5)
    G = pd.DataFrame(rng.standard_normal((400000, 3)))
    R = pd.DataFrame(rng.standard_normal((400000, 3)))
    _, ratio = compound.compound_ratio(G, R)
    assert abs(ratio.mean() - 1) < 0.1
