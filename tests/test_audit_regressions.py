"""Regression tests for the quant audit findings (each test reproduces a confirmed defect)."""
import numpy as np
import pandas as pd
import pytest

from pcon import allocation as A
from pcon import expectations as E
from pcon import metrics as M
from pcon.ledger import twr_returns


@pytest.fixture
def R():
    rng = np.random.default_rng(21)
    n = 1500
    common = rng.normal(0.0003, 0.008, n)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({"a": common + rng.normal(0.0002, 0.006, n),
                         "b": 0.5 * common + rng.normal(0.0001, 0.004, n),
                         "c": rng.normal(0.0004, 0.012, n)}, index=idx)


def _grid(f, hi, step=0.005):
    best = None
    for a in np.arange(0, 1 + 1e-9, step):
        for b in np.arange(0, 1 - a + 1e-9, step):
            w = np.array([a, b, 1 - a - b])
            if (w > hi + 1e-9).any():
                continue
            v = f(w)
            if best is None or v < best[0]:
                best = (v, w)
    return best


# ---- optimiser -----------------------------------------------------------------------------------
def test_max_sharpe_failing_case_reaches_optimum():
    vol = np.array([0.096, 0.157, 0.284])
    mu = np.array([0.235, 0.037, 0.005])
    C = np.array([[1, -0.334, -0.754], [-0.334, 1, 0.834], [-0.754, 0.834, 1]])
    cov = np.outer(vol, vol) * C
    w = A._max_sharpe(mu, cov, 0, 0.5)
    assert (w @ mu) / np.sqrt(w @ cov @ w) == pytest.approx(1.7633, abs=1e-3)   # was equal weight, SR 0.746


@pytest.mark.parametrize("hi", [1.0, 0.7, 0.5, 0.4])
def test_max_sharpe_and_min_variance_match_brute_force(R, hi):
    mu, cov = A.estimate(R)
    w = A.optimize(R, "max_sharpe", None, 0, hi).to_numpy()
    best = _grid(lambda v: -(v @ mu) / np.sqrt(v @ cov @ v), hi)
    assert -(w @ mu) / np.sqrt(w @ cov @ w) <= best[0] + 1e-4
    w = A.optimize(R, "min_variance", None, 0, hi).to_numpy()
    assert w @ cov @ w <= _grid(lambda v: v @ cov @ v, hi)[0] + 1e-9


def test_capped_risk_parity_respects_cap_and_equalises_free_assets():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = rng.integers(3, 7)
        L = rng.normal(size=(n, n))
        S = L @ L.T + 0.05 * np.eye(n)
        d = rng.uniform(0.05, 0.4, n)
        S = S / np.sqrt(np.outer(np.diag(S), np.diag(S))) * np.outer(d, d)
        hi = rng.uniform(1 / n + 0.02, 0.9)
        w = A._risk_budget(S, None, 0, hi)
        assert w.max() <= hi + 1e-12 and w.sum() == pytest.approx(1, abs=1e-12)
        rc = w * (S @ w) / (w @ S @ w)
        free = w < hi - 1e-6
        if free.sum() > 1:
            assert np.ptp(rc[free]) < 1e-3


def test_custom_risk_budgets(R):
    w = A.optimize(R, "risk_budget", budgets=pd.Series({"a": 0.2, "b": 0.3, "c": 0.5}))
    rc = A.risk_contributions(R, w)["% of risk"]
    assert rc.to_numpy() == pytest.approx([0.2, 0.3, 0.5], abs=1e-4)


def test_inverse_vol_with_cap_keeps_free_weights_proportional():
    # vols 0.4% / 1% / 2% daily: raw inverse-vol is [0.625, 0.25, 0.125]
    w = A._capped(1 / np.array([0.004, 0.01, 0.02]), 0, 0.4)
    assert w == pytest.approx([0.4, 0.4, 0.2])                # was [0.400, 0.363, 0.237]


def test_infeasible_cap_is_reported():
    assert A.feasible_bounds(3, 0, 0.2) == (0, pytest.approx(1 / 3), True)
    assert A.feasible_bounds(3, 0, 0.5)[2] is False


def test_frontier_cloud_respects_caps_and_reaches_capped_maximum(R):
    fr, cloud = A.efficient_frontier(R, None, 0, 0.5)
    mu, cov = A.estimate(R)
    lp_top = max(np.sort(mu)[::-1][:2] @ [0.5, 0.5], 0)
    assert fr["Excess return"].max() == pytest.approx(lp_top, abs=1e-6)
    # every random mix is inside the cap, so none beats the capped maximum Sharpe (the old uncapped cloud did)
    w = A.optimize(R, "max_sharpe", None, 0, 0.5).to_numpy()
    assert cloud["Sharpe"].max() <= (w @ mu) / np.sqrt(w @ cov @ w) + 1e-9


def test_min_cdar_has_lowest_cdar(R):
    def cdar(w, alpha=0.95):
        cum = np.cumsum(R.to_numpy() @ w)
        dd = np.maximum.accumulate(np.maximum(cum, 0)) - cum
        q = np.quantile(dd, alpha)
        return dd[dd >= q].mean()
    w_c = A.optimize(R, "min_cdar").to_numpy()
    for m in ("equal", "risk_parity", "min_variance", "max_sharpe"):
        assert cdar(w_c) <= cdar(A.optimize(R, m).to_numpy()) + 1e-4


# ---- statistics ----------------------------------------------------------------------------------
def test_marginal_zero_weight_and_overweighted_verdicts(R):
    R2 = R.assign(d=-R["c"] * 0.3 - 0.0005)                       # a loser max-Sharpe will not hold
    m = A.marginal(R2, method="max_sharpe", test=False)
    assert m.loc["d", "Weight"] == pytest.approx(0, abs=1e-6)
    assert m.loc["d", "Verdict"].startswith("Not held")
    assert np.isnan(m.loc["d", "Δ Sharpe"])
    m = A.marginal(R, method="target", target=pd.Series({"a": 0.05, "b": 0.05, "c": 0.9}), test=False)
    for k, row in m.iterrows():
        if row["Δ Sharpe"] <= 0 and row["Passes hurdle"]:
            assert row["Verdict"].startswith("Keep, at a smaller weight")


def test_sharpe_diff_identical_series_has_no_p_value(R):
    assert np.isnan(A.sharpe_diff_test(R["a"], R["a"])["p_not_better"])


def test_compare_rebases_cash_to_live_rate():
    # a pure T-bill strategy: backtest at 0% rates, live at 5% rates -> live Sharpe/return must look normal
    bt_idx = pd.bdate_range("2012-01-02", periods=2000)
    live_idx = pd.bdate_range("2026-01-02", periods=200)
    rf = pd.concat([pd.Series(0.0, index=bt_idx), pd.Series(0.05 / 252, index=live_idx)])
    rng = np.random.default_rng(1)
    bt = pd.Series(rng.normal(0.0004, 0.01, len(bt_idx)), index=bt_idx)
    live = pd.Series(rng.normal(0.0004, 0.01, len(live_idx)), index=live_idx) + 0.05 / 252
    c = E.compare(live, bt, rf, rf)
    assert 0.1 < c.table.loc["Return", "Percentile"] < 0.9
    assert 0.1 < c.table.loc["Sharpe", "Percentile"] < 0.9


def test_sharpe_percentile_too_early():
    rng = np.random.default_rng(2)
    bt = pd.Series(rng.normal(0.0004, 0.01, 2000), index=pd.bdate_range("2012-01-02", periods=2000))
    live = pd.Series(rng.normal(0.0004, 0.01, 40), index=pd.bdate_range("2026-01-02", periods=40))
    c = E.compare(live, bt)
    assert c.table.loc["Sharpe", "Status"] == "Too early"
    assert np.isfinite(c.table.loc["Return", "Percentile"])


def test_sharpe_consistency_counts_backtest_uncertainty():
    rng = np.random.default_rng(3)
    bt = pd.Series(rng.normal(0.0004, 0.01, 500), index=pd.bdate_range("2020-01-02", periods=500))
    live = pd.Series(rng.normal(0.0004, 0.01, 500), index=pd.bdate_range("2023-01-02", periods=500))
    s = E.sharpe_consistency(live, bt)
    assert s["Std error (difference)"] == pytest.approx(np.hypot(s["Std error (live)"], s["Std error (backtest)"]))


def test_drawdown_contributions_add_up_to_depth(R):
    w = pd.Series({"a": 0.4, "b": 0.3, "c": 0.3})
    dc = A.drawdown_contributions(R, w, "M")
    r = A.portfolio_returns(R, w, "M")
    assert dc["Contribution"].sum() == pytest.approx(M.drawdown_table(r, 1).loc[0, "depth"], abs=1e-10)


def test_stress_window_excludes_the_peak_day():
    idx = pd.bdate_range("2020-02-18", "2020-03-24")
    r = pd.Series(0.0, index=idx)
    r.loc["2020-02-19"] = 0.05                                     # the up-move INTO the peak close
    r.loc["2020-03-20"] = -0.10
    t = A.stress_test({"x": r}, {"crash": ("2020-02-19", "2020-03-23")})
    assert t.loc["crash", "x"] == pytest.approx(-0.10)
    late = pd.Series(0.0, index=pd.bdate_range("2020-02-25", "2020-03-24"))
    assert "x" not in A.stress_test({"x": late}, {"crash": ("2020-02-19", "2020-03-23")}).columns


def test_vol_target_is_capped_and_causal(R):
    r = R.mean(axis=1)
    scaled, expo = A.vol_target(r, 0.05)
    assert expo.max() <= 1.0 + 1e-12
    r2 = r.copy()
    r2.iloc[1000:] *= 5                                            # change the future only
    assert np.allclose(A.vol_target(r2, 0.05)[1].iloc[:1000], expo.iloc[:1000])


def test_diversification_metrics(R):
    d = A.diversification(R, pd.Series({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}), R.mean(axis=1))
    assert 1 <= d["Effective number of bets"] <= 3 and d["Diversification ratio"] >= 1
    assert 0 <= d["Share of variance from benchmark beta"] <= 1


# ---- accounting ----------------------------------------------------------------------------------
def test_same_day_close_and_withdrawal_is_not_minus_100pct():
    nav = pd.Series([100.0, 105.0, 0.0])                 # day 3: up 5% then everything withdrawn at the close
    fin = pd.Series([100.0, 0.0, 0.0])
    fout = pd.Series([0.0, 0.0, -110.25])
    r = twr_returns(nav, fin, fout)
    assert r.iloc[2] == pytest.approx(0.05)


def test_dust_and_unfunded_sleeves_have_no_returns():
    nav = pd.Series([0.0, 0.004, 0.008, 1000.0, 1010.0])  # fills before funding leave dust; then a deposit
    fin = pd.Series([0.0, 0.0, 0.0, 1000.0, 0.0])
    r = twr_returns(nav, fin, pd.Series(0.0, index=nav.index))
    assert r.iloc[:3].isna().all() and r.iloc[4] == pytest.approx(0.01)


def test_mwr_short_period_and_fully_withdrawn():
    ann, period, yrs = M.mwr(["2026-01-02", "2026-01-04"], [-100, 105])
    assert period == pytest.approx(0.05) and np.isfinite(ann)
    # money in for 1 year earning 10%, then fully withdrawn; measured over that year, not to today
    ann, period, yrs = M.mwr(["2025-01-01", "2026-01-01"], [-100, 110])
    assert ann == pytest.approx(0.10) and period == pytest.approx(0.10)


def test_stress_columns_keep_input_order():
    old = pd.Series(0.001, index=pd.bdate_range("2005-01-03", "2026-06-01"))
    new = pd.Series(0.001, index=pd.bdate_range("2012-01-02", "2026-06-01"))
    t = A.stress_test({"new": new, "old": old, "mix": new})
    assert list(t.columns) == ["new", "old", "mix"] and np.isnan(t.iloc[0, 0])
