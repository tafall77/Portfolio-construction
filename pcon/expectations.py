"""Expected vs actual: is the live record consistent with what the backtest promised?

Method
* The *expectation* is the strategy's backtest over its honest out-of-sample window (``expectation_start``
  in portfolio.yaml), up to the day before live trading started.
* Thousands of paths with the live record's length are drawn from it with a moving-block bootstrap
  (63-day blocks keep volatility clustering and the persistence of exposure regimes).
* Each live metric is placed in the distribution of the same metric across those paths. A percentile
  near 50 % means "exactly as expected"; below 5 % means the live result is worse than 95 % of what the
  backtest would normally produce over the same length of time.
* The model-vs-live comparison uses the backtest series on the *same* dates (re-export the notebook after
  going live) and isolates execution: slippage, missed or late signals, sizing errors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M

TD = M.TD
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def block_indices(n_obs: int, horizon: int, n_paths: int, block: int, seed: int = 7) -> np.ndarray:
    """Moving-block bootstrap with circular wrap: (n_paths, horizon) row indices into the sample."""
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n_obs))
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n_obs, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n_obs
    return idx.reshape(n_paths, -1)[:, :horizon]


def simulate(r: pd.Series, horizon: int, n_paths: int = 2000, block: int = 63, seed: int = 7) -> np.ndarray:
    x = M._clean(r).to_numpy()
    if len(x) < 20 or horizon < 1:
        return np.empty((0, max(horizon, 0)))
    return x[block_indices(len(x), horizon, n_paths, block, seed)]


def _path_stats(sims: np.ndarray, rf_daily: np.ndarray | float = 0.0) -> dict[str, np.ndarray]:
    eq = np.cumprod(1 + sims, axis=1)
    peak = np.maximum(np.maximum.accumulate(eq, axis=1), 1.0)
    dd = eq / peak - 1
    ex = sims - rf_daily
    sd = sims.std(axis=1, ddof=1) if sims.shape[1] > 1 else np.full(len(sims), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = ex.mean(axis=1) / ex.std(axis=1, ddof=1) * np.sqrt(TD)
    return {"Return": eq[:, -1] - 1, "Volatility": sd * np.sqrt(TD), "Sharpe": sr, "Max drawdown": dd.min(axis=1),
            "Current drawdown": dd[:, -1], "% positive days": (sims > 0).mean(axis=1)}


def percentile_of(value: float, dist: np.ndarray) -> float:
    d = dist[np.isfinite(dist)]
    if not np.isfinite(value) or len(d) == 0:
        return np.nan
    return float(((d < value).sum() + 0.5 * (d == value).sum()) / len(d))


def expectation_window(bt: pd.Series, start, live_start) -> pd.Series:
    """Backtest returns used as the expectation: honest OOS start -> the day before live trading."""
    r = M._clean(bt)
    if start is not None:
        r = r.loc[pd.Timestamp(start):]
    if live_start is not None:
        r = r.loc[:pd.Timestamp(live_start) - pd.Timedelta(days=1)]
    return r


def cone(expected: pd.Series, dates: pd.DatetimeIndex, n_paths: int = 2000, block: int = 63,
         quantiles=QUANTILES, start_value: float = 1.0) -> pd.DataFrame:
    """Quantiles of cumulative wealth along ``dates`` (wealth = ``start_value`` the day before ``dates[0]``)."""
    sims = simulate(expected, len(dates), n_paths, block)
    if sims.size == 0:
        return pd.DataFrame(index=dates)
    eq = start_value * np.cumprod(1 + sims, axis=1)
    q = np.quantile(eq, quantiles, axis=0).T
    return pd.DataFrame(q, index=dates, columns=[f"p{int(x * 100)}" for x in quantiles])


def forward_cone(expected: pd.Series, last_date, start_value: float, days: int = 252, n_paths: int = 2000,
                 block: int = 63) -> pd.DataFrame:
    dates = pd.bdate_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=days)
    return cone(expected, dates, n_paths, block, start_value=start_value)


def forward_risk(expected: pd.Series, days: int = 252, n_paths: int = 4000, block: int = 63) -> dict:
    """What to expect over the next ``days``: loss probability and drawdown quantiles."""
    sims = simulate(expected, days, n_paths, block)
    if sims.size == 0:
        return {}
    st = _path_stats(sims)
    return {"P(loss)": float((st["Return"] < 0).mean()),
            "Median return": float(np.median(st["Return"])),
            "5% worst return": float(np.quantile(st["Return"], 0.05)),
            "Median max drawdown": float(np.median(st["Max drawdown"])),
            "5% worst max drawdown": float(np.quantile(st["Max drawdown"], 0.05)),
            "1% worst max drawdown": float(np.quantile(st["Max drawdown"], 0.01))}


@dataclass
class Comparison:
    table: pd.DataFrame            # metric x [Expected, Low (5%), High (95%), Actual, Percentile, Status]
    distributions: dict            # metric -> simulated values (for histograms)
    horizon: int
    expected_profile: dict         # full-window backtest stats (annualised)
    sharpe_test: dict


GOOD_HIGH = {"Return": True, "Volatility": None, "Sharpe": True, "Max drawdown": True, "Current drawdown": True,
             "% positive days": True}


def _status(metric: str, pct: float) -> str:
    if not np.isfinite(pct):
        return "n/a"
    direction = GOOD_HIGH.get(metric)
    if direction is None:              # volatility: both tails are a surprise
        return "In line" if 0.05 <= pct <= 0.95 else ("Lower than expected" if pct < 0.05 else "Higher than expected")
    if pct < 0.05:
        return "Below expectations"
    if pct < 0.15:
        return "Watch"
    if pct > 0.95:
        return "Above expectations"
    return "In line"


def compare(live: pd.Series, expected: pd.Series, rf_live: pd.Series | None = None,
            rf_expected: pd.Series | None = None, n_paths: int = 3000, block: int = 63) -> Comparison | None:
    """Place the live record inside the distribution of same-length backtest paths."""
    live = M._clean(live)
    expected = M._clean(expected)
    if len(live) < 2 or len(expected) < 60:
        return None
    H = len(live)
    rf_l = (rf_live.reindex(live.index).fillna(0).to_numpy() if rf_live is not None else np.zeros(H))
    sims = simulate(expected, H, n_paths, block)
    dist = _path_stats(sims, rf_l[None, :])
    w = (1 + live).cumprod()
    dd = w / w.cummax().clip(lower=1.0) - 1
    actual = {"Return": w.iloc[-1] - 1, "Volatility": live.std() * np.sqrt(TD) if H > 1 else np.nan,
              "Sharpe": M.sharpe(live, rf_live), "Max drawdown": dd.min(), "Current drawdown": dd.iloc[-1],
              "% positive days": (live > 0).mean()}
    rows = {}
    for k, d in dist.items():
        d = d[np.isfinite(d)]
        pct = percentile_of(actual[k], d)
        rows[k] = {"Expected": np.median(d) if len(d) else np.nan,
                   "Low (5%)": np.quantile(d, 0.05) if len(d) else np.nan,
                   "High (95%)": np.quantile(d, 0.95) if len(d) else np.nan,
                   "Actual": actual[k], "Percentile": pct, "Status": _status(k, pct)}
    table = pd.DataFrame(rows).T
    profile = M.perf_summary(expected, rf_expected)
    return Comparison(table, dist, H, profile, sharpe_consistency(live, expected, rf_live, rf_expected))


def sharpe_consistency(live: pd.Series, expected: pd.Series, rf_live=None, rf_expected=None) -> dict:
    """Is the live Sharpe statistically different from the backtest Sharpe? And how long until we know?"""
    from scipy import stats
    live = M._clean(live)
    if len(live) < 10:
        return {}
    sr_bt = M.sharpe(expected, rf_expected)
    ex = (live - M._rf_for(live, rf_live)).to_numpy()
    sr_d = ex.mean() / ex.std(ddof=1) if ex.std(ddof=1) > 0 else np.nan
    se = M.sharpe_std_error(sr_d, len(ex), stats.skew(ex), stats.kurtosis(ex, fisher=False)) * np.sqrt(TD)
    sr_live = sr_d * np.sqrt(TD)
    z = (sr_live - sr_bt) / se if se > 0 else np.nan
    return {"Live Sharpe": sr_live, "Backtest Sharpe": sr_bt, "Std error (live)": se,
            "z-score": z, "p-value (two-sided)": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "PSR live (SR>0)": M.psr(live, rf_live),
            "PSR live (SR>backtest/2)": M.psr(live, rf_live, sr_bt / 2) if np.isfinite(sr_bt) else np.nan,
            "Years live": len(live) / TD,
            "Years needed to confirm SR>0 (95%)": M.min_track_record(expected, rf_expected, 0.0, 0.95)}


def tracking(live: pd.Series, model: pd.Series) -> dict | None:
    """Live vs model on the dates both exist: implementation shortfall and tracking error."""
    a, b = M._clean(live).align(M._clean(model), join="inner")
    if len(a) < 5:
        return None
    diff = a - b
    rel = (1 + a).cumprod() / (1 + b).cumprod() - 1
    beta = np.cov(a, b)[0, 1] / b.var() if b.var() > 0 else np.nan
    return {"days": len(a), "start": a.index[0], "end": a.index[-1],
            "Live return": (1 + a).prod() - 1, "Model return": (1 + b).prod() - 1,
            "Implementation shortfall": rel.iloc[-1], "Shortfall (ann.)": diff.mean() * TD,
            "Tracking error": diff.std() * np.sqrt(TD), "Correlation": a.corr(b), "Beta to model": beta,
            "Worst day vs model": diff.min(), "series": pd.DataFrame({"live": (1 + a).cumprod(),
                                                                      "model": (1 + b).cumprod(),
                                                                      "shortfall": rel})}


def health_checks(book) -> list[dict]:
    """Traffic-light checks across strategies, portfolio and data. Each: level (red/amber/green/info), scope, msg."""
    out: list[dict] = []
    cfg, L = book.cfg, book.ledger
    a = cfg.alerts
    from .backtests import tests_summary
    for sid, s in cfg.strategies.items():
        meta = book.backtest_meta.get(sid)
        if book.backtest_returns(sid) is not None and meta is None:
            out.append(dict(level="info", scope=s.short,
                            msg="The backtest export has no selection record: re-run the updated export cell so the "
                                "dashboard can confirm it is the configuration that passed the notebook's final tests."))
        elif meta is not None:
            n_pass, n_eval, failed = tests_summary(meta)
            if not meta.get("passed_selection", True):
                out.append(dict(level="amber", scope=s.short,
                                msg=f"Exported configuration ({meta.get('selected', '?')}) is NOT the notebook's final "
                                    "selection (override, or no candidate survived the selection gates)."))
            if failed:
                out.append(dict(level="amber", scope=s.short,
                                msg=f"{meta.get('selected', 'Exported configuration')} passed {n_pass}/{n_eval} final "
                                    f"tests; failed: {'; '.join(failed[:3])}{' ...' if len(failed) > 3 else ''}."))
        cmp_ = book.comparison(sid)
        live = book.live_returns().get(sid)
        if live is None or len(live.dropna()) < 2:
            out.append(dict(level="info", scope=s.short, msg="No live record yet."))
            continue
        if cmp_ is None:
            out.append(dict(level="info", scope=s.short, msg="No backtest expectation available (see Data)."))
            continue
        t = cmp_.table
        if len(live.dropna()) < 21:
            out.append(dict(level="info", scope=s.short,
                            msg=f"Only {len(live.dropna())} live days: percentiles are indicative only."))
        dd_p = t.loc["Max drawdown", "Percentile"]
        if np.isfinite(dd_p) and dd_p < 1 - a.drawdown_percentile:
            out.append(dict(level="red", scope=s.short,
                            msg=f"Max drawdown {t.loc['Max drawdown', 'Actual']:.1%} is worse than "
                                f"{1 - dd_p:.0%} of backtest paths of the same length "
                                f"(expected {t.loc['Max drawdown', 'Expected']:.1%})."))
        r_p = t.loc["Return", "Percentile"]
        if np.isfinite(r_p) and r_p < a.return_percentile:
            out.append(dict(level="red", scope=s.short,
                            msg=f"Live return {t.loc['Return', 'Actual']:.1%} sits at the {r_p:.0%} percentile "
                                f"of the expectation cone (median {t.loc['Return', 'Expected']:.1%})."))
        elif np.isfinite(r_p) and r_p < 0.15:
            out.append(dict(level="amber", scope=s.short,
                            msg=f"Live return in the bottom {r_p:.0%} of expected outcomes: watch."))
        exp_vol = cmp_.expected_profile.get("Volatility", np.nan)
        live_vol = t.loc["Volatility", "Actual"]
        if np.isfinite(exp_vol) and exp_vol > 0 and np.isfinite(live_vol) and len(live.dropna()) >= 21 \
                and live_vol / exp_vol > a.vol_ratio:
            out.append(dict(level="amber", scope=s.short,
                            msg=f"Live volatility {live_vol:.1%} is {live_vol / exp_vol:.1f}x the backtest's "
                                f"{exp_vol:.1%}: check position sizing."))
        tr = book.tracking(sid)
        if tr is not None and tr["days"] >= 10 and tr["Tracking error"] > a.tracking_error:
            out.append(dict(level="amber", scope=s.short,
                            msg=f"Tracking error vs model {tr['Tracking error']:.1%} (shortfall "
                                f"{tr['Implementation shortfall']:+.2%}): execution differs from the signals."))
        if not any(o["scope"] == s.short and o["level"] in ("red", "amber") for o in out):
            out.append(dict(level="green", scope=s.short, msg="Live record consistent with the backtest."))

    if not L.empty:
        w = L.weights().iloc[-1]
        tgt = cfg.target_weights()
        drift = (w.reindex(tgt.index).fillna(0) - tgt)
        big = drift[drift.abs() > a.weight_drift]
        if len(big):
            out.append(dict(level="amber", scope="Portfolio",
                            msg="Allocation drift: " + ", ".join(f"{cfg.label(k)} {v:+.1%}" for k, v in big.items())
                                + " vs target. See the rebalance list in Portfolio Construction."))
        lr = pd.DataFrame(book.live_returns()).dropna(how="all").tail(63)
        if lr.shape[1] > 1 and len(lr.dropna()) >= 40:
            c = lr.dropna().corr()
            pairs = [(i, j, c.loc[i, j]) for k, i in enumerate(c.index) for j in c.index[k + 1:]
                     if c.loc[i, j] > a.correlation]
            for i, j, v in pairs:
                out.append(dict(level="amber", scope="Portfolio",
                                msg=f"{cfg.label(i)} and {cfg.label(j)} correlation {v:.2f} over the last 63 days: "
                                    "diversification is lower than planned."))
        if not any(o["scope"] == "Portfolio" for o in out):
            out.append(dict(level="green", scope="Portfolio", msg="Allocation within drift bands."))
    for msg in book.data_warnings():
        out.append(dict(level="info", scope="Data", msg=msg))
    return out
