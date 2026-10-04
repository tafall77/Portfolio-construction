import numpy as np
import pandas as pd
import pytest

from pcon import allocation as A
from pcon import expectations as E
from pcon import metrics as M


def series(mu=0.0004, sd=0.01, n=2520, seed=0, start="2010-01-01"):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=pd.bdate_range(start, periods=n))


# ---- metrics -------------------------------------------------------------------------------------
def test_basic_metrics():
    r = pd.Series([0.1, -0.5, 0.2], index=pd.bdate_range("2026-01-01", periods=3))
    assert M.max_drawdown(r) == pytest.approx(-0.5)
    assert M.drawdown(r).iloc[-1] == pytest.approx(1.1 * 0.5 * 1.2 / 1.1 - 1)
    r = series()
    s = M.perf_summary(r)
    assert s["Sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert s["Total return"] == pytest.approx((1 + r).prod() - 1)


def test_underwater_periods_recovery():
    r = pd.Series([0.0, -0.1, 0.05, 0.1, -0.02], index=pd.bdate_range("2026-01-01", periods=5))
    t = M.underwater_periods(r)
    assert len(t) == 2
    assert bool(t.loc[0, "recovered"]) and not bool(t.loc[1, "recovered"])
    assert t.loc[0, "depth"] == pytest.approx(-0.1)


def test_psr_and_min_track_record_are_sensible():
    good, flat = series(mu=0.001, seed=1), series(mu=0.0, seed=2)
    assert M.psr(good) > 0.95 > M.psr(flat)
    assert M.min_track_record(good) < M.min_track_record(series(mu=0.0003, seed=1))


def test_trade_summary():
    rt = pd.DataFrame({"ret": [0.1, -0.05, 0.02], "pnl": [100, -50, 20], "days": [10, 5, 3],
                       "exit_date": pd.bdate_range("2026-01-01", periods=3)})
    s = M.trade_summary(rt)
    assert s["Win rate"] == pytest.approx(2 / 3)
    assert s["Profit factor"] == pytest.approx(120 / 50)


# ---- expectations --------------------------------------------------------------------------------
def test_live_drawn_from_backtest_sits_mid_distribution():
    bt = series(seed=3, n=4000)
    live = series(seed=4, n=252, start="2030-01-01")
    c = E.compare(live, bt)
    assert 0.05 < c.table.loc["Return", "Percentile"] < 0.95
    assert c.table.loc["Volatility", "Status"] == "In line"


def test_broken_strategy_is_flagged():
    bt = series(mu=0.001, seed=5, n=4000)
    live = series(mu=-0.004, seed=6, n=126, start="2030-01-01")
    c = E.compare(live, bt)
    assert c.table.loc["Return", "Status"] == "Below expectations"


def test_tracking_identical_is_zero_and_cone_shape():
    r = series(n=300)
    t = E.tracking(r, r)
    assert t["Tracking error"] == pytest.approx(0.0) and t["Implementation shortfall"] == pytest.approx(0.0)
    cn = E.cone(series(n=3000), pd.bdate_range("2031-01-01", periods=60))
    assert (cn["p5"] <= cn["p50"]).all() and (cn["p50"] <= cn["p95"]).all()
    assert cn["p95"].iloc[-1] - cn["p5"].iloc[-1] > cn["p95"].iloc[0] - cn["p5"].iloc[0]


def test_expectation_window_excludes_live_period():
    bt = series(n=1000)
    w = E.expectation_window(bt, "2011-01-01", "2012-06-01")
    assert w.index.min() >= pd.Timestamp("2011-01-01") and w.index.max() < pd.Timestamp("2012-06-01")


# ---- allocation ----------------------------------------------------------------------------------
@pytest.fixture
def R():
    rng = np.random.default_rng(9)
    n = 2000
    common = rng.normal(0.0003, 0.008, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"a": common + rng.normal(0.0002, 0.006, n),
                         "b": 0.5 * common + rng.normal(0.0001, 0.004, n),
                         "c": rng.normal(0.0004, 0.012, n)}, index=idx)


def test_portfolio_returns_rebalancing(R):
    w = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})
    daily = A.portfolio_returns(R, w, "D")
    assert np.allclose(daily, R @ w)
    monthly, contrib = A.portfolio_returns(R, w, "M", contributions=True)
    assert np.allclose(contrib.sum(axis=1), monthly)
    first_days = R.index.to_series().groupby(R.index.to_period("M")).first()
    assert np.allclose(monthly.loc[first_days.values], (R @ w).loc[first_days.values])
    # buy-and-hold over the whole window equals the weighted sum of each strategy's growth
    bh = A.portfolio_returns(R, w, "none")
    assert (1 + bh).prod() == pytest.approx(((1 + R).prod() * w).sum())


@pytest.mark.parametrize("method", list(A.METHODS))
def test_optimizers_valid_weights(R, method):
    w = A.optimize(R, method, None, 0.0, 0.6, pd.Series({"a": 0.5, "b": 0.25, "c": 0.25}), n_resamples=20)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all() and (w <= 0.6 + 1e-6).all()


def test_risk_parity_equalises_contributions(R):
    w = A.optimize(R, "risk_parity")
    rc = A.risk_contributions(R, w)["% of risk"]
    assert np.allclose(rc, 1 / 3, atol=0.01)


def test_walk_forward_is_causal(R):
    r1, w1 = A.walk_forward(R, "max_sharpe", lookback_days=500, step="Q")
    R2 = R.copy()
    cut = R.index[1200]
    R2.loc[cut:] = R2.loc[cut:] * -3          # change the future only
    r2, w2 = A.walk_forward(R2, "max_sharpe", lookback_days=500, step="Q")
    before = w1.index[w1.index <= cut]
    assert np.allclose(w1.loc[before], w2.loc[before])
    assert np.allclose(r1.loc[:R.index[1199]], r2.loc[:R.index[1199]])


def test_marginal_hurdle_logic(R):
    m = A.marginal(R, method="equal", test=False)
    assert set(m.index) == {"a", "b", "c"}
    for k, row in m.iterrows():
        assert row["Hurdle Sharpe"] == pytest.approx(row["Corr to rest"] * row["Sharpe without"])


def test_explore_covers_all_subsets(R):
    ex = A.explore(R, methods=("equal", "risk_parity"), walk_forward_days=500, n_resamples=10)
    assert set(ex["subset"]) == {"a", "b", "c", "a + b", "a + c", "b + c", "a + b + c"}
    assert ex["WF Sharpe"].notna().all()


def test_ledoit_wolf_is_psd(R):
    S = A.ledoit_wolf(R.to_numpy())
    assert np.all(np.linalg.eigvalsh(S) > 0)
