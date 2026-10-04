"""
rolling_momentum.py
===================

Research engine for a long/flat *rolling time-series momentum* strategy on
equity indices (S&P 500, Nasdaq-100).

Rule
----
At every close ``t`` and for a calendar lookback ``L`` (1 week, 30 days,
90 days, 1 year, ...):

    m_t(L) = P_t / P_{tau(t - L)} - 1        tau(x) = last trading day <= x
    s_t    = 1 if m_t > 0 else 0             (1 = long index, 0 = cash)
    w_{t+1} = s_t                            (decided at close t, held over t+1)

    r^strat_{t+1} = w_{t+1} * r_{t+1} + (1 - w_{t+1}) * rf_{t+1}
                    - c * |w_{t+1} - w_t|

The module is deliberately free of notebook state so every function can be
unit-tested (see ``tests/``) and audited for lookahead bias.

Sections
--------
1. Data acquisition & cleaning
2. Signals
3. Backtest accounting
4. Performance statistics
5. Statistical inference & overfitting diagnostics
   (PSR / DSR, PBO via CSCV, stationary bootstrap, circular-shift test,
   walk-forward re-optimisation)
6. Lookahead-bias audits
7. Plotting helpers
"""
from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329

# The user-specified candidate set. Lookbacks are *calendar* windows: on
# 27-Sep-2026 the 1Y window compares against the close on (or the last trading
# day before) 27-Sep-2025.
CANDIDATES: dict[str, pd.DateOffset] = {
    "1W": pd.DateOffset(weeks=1),
    "30D": pd.DateOffset(days=30),
    "90D": pd.DateOffset(days=90),
    "1Y": pd.DateOffset(years=1),
}


# =============================================================================
# 1. Data acquisition & cleaning
# =============================================================================
def _cache_name(ticker: str) -> str:
    return ticker.replace("^", "").replace("=", "_").replace("/", "_") + ".csv"


def download_close(ticker: str, start: str, end=None) -> pd.Series:
    """Download daily closes from Yahoo Finance via ``yfinance``.

    ``end`` defaults to today's date, which yfinance treats as *exclusive*, so
    an in-progress (partial) bar for today is never included -- a partial bar
    would leak intraday information that was not final at the close.
    """
    import yfinance as yf

    if end is None:
        end = pd.Timestamp.today().normalize()
    df = yf.download(ticker, start=start, end=end, auto_adjust=False,
                     progress=False, actions=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].astype(float)
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s.name = ticker
    return s


def load_close(ticker: str, start: str, cache_dir: str | Path = "data",
               refresh: bool = False, required: bool = True) -> pd.Series | None:
    """Load closes from the CSV cache, downloading (and caching) when needed.

    If a refresh is requested but the download fails, the cached file is used.
    Returns ``None`` for an unavailable optional series (``required=False``).
    """
    path = Path(cache_dir) / _cache_name(ticker)
    s = None
    if refresh or not path.exists():
        try:
            s = download_close(ticker, start)
            path.parent.mkdir(parents=True, exist_ok=True)
            s.rename("close").to_frame().to_csv(path, index_label="date")
        except Exception as exc:  # network / ticker problems
            if not path.exists():
                if required:
                    raise RuntimeError(
                        f"Could not download {ticker} and no cache at {path}: {exc}"
                    ) from exc
                warnings.warn(f"{ticker} unavailable ({exc}); continuing without it")
                return None
            warnings.warn(f"{ticker} download failed ({exc}); using cache {path}")
    if s is None:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
        s.name = ticker
    return s.loc[pd.Timestamp(start):]


def clean_close(s: pd.Series, positive: bool = True) -> tuple[pd.Series, dict]:
    """Sort, de-duplicate and drop invalid / missing observations.

    ``positive=True`` (prices) drops non-positive closes; ``positive=False``
    (yields) keeps zeros -- T-bill yields were ~0% in 2009-15 and 2020-21 --
    and drops only negative prints.
    Returns the cleaned series and a data-quality report.
    """
    n_raw = len(s)
    s = s.sort_index()
    dupes = int(s.index.duplicated().sum())
    s = s[~s.index.duplicated(keep="last")]
    s = s.replace([np.inf, -np.inf], np.nan)
    bad = (s <= 0) if positive else (s < 0)
    non_pos = int(bad.sum())
    s = s.where(~bad)
    missing = int(s.isna().sum())
    s = s.dropna()
    r = s.pct_change() if positive else s.diff()
    gaps = s.index.to_series().diff().dt.days
    unit = "{:+.2%}" if positive else "{:+.2f} pts"
    report = {
        "first date": s.index[0].date(),
        "last date": s.index[-1].date(),
        "rows (raw)": n_raw,
        "rows (clean)": len(s),
        "duplicate dates": dupes,
        ("non-positive closes" if positive else "negative values"): non_pos,
        "missing closes": missing,
        "calendar gaps > 5 days": int((gaps > 5).sum()),
        "largest daily rise": f"{unit.format(r.max())} ({r.idxmax().date()})",
        "largest daily fall": f"{unit.format(r.min())} ({r.idxmin().date()})",
        "|daily move| > 10%": int((r.abs() > 0.10).sum()) if positive else "n/a",
    }
    return s, report


@dataclass
class Market:
    """Aligned inputs for one market.

    close : price index used for the momentum signal (what a trader sees)
    ret   : daily total return used for P&L (total-return index where
            available, price return otherwise -- see ``meta``)
    cash  : daily return earned in cash, accrued from the *previous* close's
            T-bill yield (known at the time the position is taken)
    """

    name: str
    close: pd.Series
    ret: pd.Series
    cash: pd.Series
    meta: dict = field(default_factory=dict)


def build_market(name: str, price_close: pd.Series,
                 tr_close: pd.Series | None = None,
                 rf_yield_pct: pd.Series | None = None) -> Market:
    """Assemble signal prices, P&L returns and cash returns on one calendar.

    * Each day's return comes from a single source: the total-return index's
      one-day return when both today's and yesterday's TR closes exist,
      otherwise the price index's return. No day is double counted.
    * Cash accrues the T-bill yield observed at the *previous* close over the
      calendar days elapsed (weekends earn three days of carry).
    """
    idx = price_close.index
    px_ret = price_close.pct_change()
    meta: dict = {"price_ticker": price_close.name}

    if tr_close is not None and len(tr_close) > 1:
        tr = tr_close.reindex(idx)
        tr_ret = tr / tr.shift(1) - 1.0
        ret = tr_ret.where(tr_ret.notna(), px_ret)
        first_tr = tr_ret.first_valid_index()
        meta.update(tr_ticker=tr_close.name, tr_from=first_tr,
                    tr_share=float(tr_ret.notna().mean()))
    else:
        ret = px_ret
        meta.update(tr_ticker=None, tr_from=None, tr_share=0.0)

    if rf_yield_pct is not None and len(rf_yield_pct) > 0:
        y = rf_yield_pct.reindex(rf_yield_pct.index.union(idx)).ffill().reindex(idx)
        y = y.clip(lower=0.0, upper=25.0) / 100.0
        days = idx.to_series().diff().dt.days.fillna(1).to_numpy()
        prev_y = y.shift(1)  # yield known at close t-1 accrues over day t
        cash = (1.0 + prev_y) ** (days / 365.0) - 1.0
        meta.update(rf_ticker=rf_yield_pct.name,
                    rf_from=rf_yield_pct.first_valid_index())
        cash = cash.fillna(0.0)
    else:
        cash = pd.Series(0.0, index=idx)
        meta.update(rf_ticker=None, rf_from=None)

    return Market(name=name, close=price_close, ret=ret.rename("ret"),
                  cash=cash.rename("cash"), meta=meta)


# =============================================================================
# 2. Signals
# =============================================================================
def as_offset(lookback) -> pd.DateOffset:
    """Accept a DateOffset, a candidate label ('1Y') or an int (calendar days)."""
    if isinstance(lookback, pd.DateOffset):
        return lookback
    if isinstance(lookback, str):
        return CANDIDATES[lookback]
    return pd.DateOffset(days=int(lookback))


_REF_CACHE: dict = {}


def reference_positions(index: pd.DatetimeIndex, lookback) -> np.ndarray:
    """Integer position of the as-of reference close for every date.

    For date t the reference is the last trading day on or before t - L
    (``-1`` when t - L precedes the sample). This depends only on the
    calendar, never on prices.
    """
    off = as_offset(lookback)
    key = (hash(index.asi8.tobytes()), len(index), repr(off))
    if key not in _REF_CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            ref_dates = index - off
        if len(_REF_CACHE) > 512:
            _REF_CACHE.clear()
        _REF_CACHE[key] = np.searchsorted(index.values, ref_dates.values, side="right") - 1
    return _REF_CACHE[key]


def momentum(close: pd.Series, lookback) -> pd.Series:
    """Trailing calendar-window return ``P_t / P_{tau(t-L)} - 1``."""
    pos = reference_positions(close.index, lookback)
    px = close.to_numpy(dtype=float)
    ref = np.where(pos >= 0, px[np.clip(pos, 0, None)], np.nan)
    return pd.Series(px / ref - 1.0, index=close.index, name="momentum")


def signal(close: pd.Series, lookback, threshold: float = 0.0) -> pd.Series:
    """1.0 when trailing return > threshold, 0.0 otherwise, NaN in warm-up."""
    m = momentum(close, lookback)
    return (m > threshold).astype(float).where(m.notna()).rename("signal")


def reference_date(index: pd.DatetimeIndex, date, lookback) -> pd.Timestamp:
    """The reference trading day used by the rule on ``date`` (for display)."""
    target = pd.Timestamp(date) - as_offset(lookback)
    i = index.searchsorted(target, side="right") - 1
    return index[i] if i >= 0 else pd.NaT


# =============================================================================
# 3. Backtest accounting
# =============================================================================
def backtest(sig: pd.Series, ret: pd.Series, cash: pd.Series,
             cost_bps: float = 5.0, lag: int = 1) -> pd.DataFrame:
    """Daily long/flat backtest.

    ``lag`` is the number of closes between observing the signal and holding
    the position: ``lag=1`` means the signal from close t sets the position for
    the return from close t to close t+1. ``lag=0`` is the *leaky* (lookahead)
    version kept only to demonstrate the bias. Costs are charged in basis
    points per unit of turnover (one switch in or out = 1 unit).
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    pos = sig.shift(lag)
    p = pos.fillna(0.0)
    turnover = p.diff().abs().fillna(p.abs())
    gross = p * ret + (1.0 - p) * cash
    cost = turnover * cost_bps / 1e4
    return pd.DataFrame({
        "signal": sig, "pos": pos, "turnover": turnover,
        "asset": ret, "cash": cash, "gross": gross, "cost": cost,
        "net": gross - cost,
    })


def run_lookbacks(market: Market, lookbacks: Mapping[str, object],
                  cost_bps: float = 5.0, lag: int = 1) -> dict[str, pd.DataFrame]:
    """Backtest every lookback in ``lookbacks`` on one market."""
    return {
        name: backtest(signal(market.close, lb), market.ret, market.cash,
                       cost_bps=cost_bps, lag=lag)
        for name, lb in lookbacks.items()
    }


def window(obj, start=None, end=None):
    """Inclusive date slice that tolerates ``None`` bounds."""
    return obj.loc[pd.Timestamp(start) if start else None:
                   pd.Timestamp(end) if end else None]


def positions_to_returns(pos: np.ndarray, ret: np.ndarray, cash: np.ndarray,
                         cost_bps: float) -> np.ndarray:
    """Net daily returns for an already-lagged position array."""
    turnover = np.abs(np.diff(pos, prepend=pos[0]))
    return pos * ret + (1.0 - pos) * cash - turnover * cost_bps / 1e4


# =============================================================================
# 4. Performance statistics
# =============================================================================
def sharpe(excess: np.ndarray | pd.Series, annualize: bool = True) -> float:
    x = np.asarray(excess, dtype=float)
    x = x[~np.isnan(x)]
    sd = x.std(ddof=1)
    if len(x) < 2 or sd == 0:
        return np.nan
    sr = x.mean() / sd
    return sr * math.sqrt(TRADING_DAYS) if annualize else sr


def drawdown(r: pd.Series) -> pd.Series:
    wealth = (1.0 + r.fillna(0.0)).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    return pd.Series(wealth.to_numpy() / peak - 1.0, index=r.index)


def _longest_underwater_years(dd: pd.Series) -> float:
    under = (dd < 0).to_numpy()
    if not under.any():
        return 0.0
    dates = dd.index
    best, start = 0.0, None
    for i, u in enumerate(under):
        if u and start is None:
            start = i - 1 if i > 0 else i
        if (not u or i == len(under) - 1) and start is not None:
            end = i
            best = max(best, (dates[end] - dates[start]).days / 365.25)
            start = None
    return best


def perf_stats(r: pd.Series, rf: pd.Series, pos: pd.Series | None = None,
               turnover: pd.Series | None = None) -> pd.Series:
    """Standard performance & risk statistics for a daily return series."""
    r = r.dropna()
    rf = rf.reindex(r.index).fillna(0.0)
    n = len(r)
    years = n / TRADING_DAYS
    ex = r - rf
    wealth_end = float((1.0 + r).prod())
    cagr = wealth_end ** (1.0 / years) - 1.0
    vol = r.std(ddof=1) * math.sqrt(TRADING_DAYS)
    downside = math.sqrt((np.minimum(ex, 0.0) ** 2).mean()) * math.sqrt(TRADING_DAYS)
    dd = drawdown(r)
    mdd = dd.min()
    q05 = r.quantile(0.05)
    out = {
        "Start": r.index[0].date(),
        "End": r.index[-1].date(),
        "Years": years,
        "CAGR": cagr,
        "Ann. volatility": vol,
        "Sharpe": sharpe(ex),
        "Sortino": ex.mean() * TRADING_DAYS / downside if downside > 0 else np.nan,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Longest drawdown (yrs)": _longest_underwater_years(dd),
        "Skew": stats.skew(r),
        "Excess kurtosis": stats.kurtosis(r, fisher=True),
        "Daily VaR 95%": -q05,
        "Daily CVaR 95%": -r[r <= q05].mean(),
        "Worst day": r.min(),
        "Best day": r.max(),
        "Positive days": (r > 0).mean(),
    }
    if pos is not None:
        p = pos.reindex(r.index)
        out["Time in market"] = p.mean()
    if turnover is not None:
        out["Switches / year"] = turnover.reindex(r.index).sum() / years
    return pd.Series(out)


def stats_table(series: Mapping[str, pd.Series], rf: pd.Series,
                frames: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Side-by-side ``perf_stats`` for several return series."""
    frames = frames or {}
    cols, order = {}, []
    for name, r in series.items():
        f = frames.get(name)
        cols[name] = perf_stats(r, rf,
                                pos=None if f is None else f["pos"],
                                turnover=None if f is None else f["turnover"])
        order += [k for k in cols[name].index if k not in order]
    return pd.DataFrame(cols).reindex(order)


def annual_returns(r: pd.Series) -> pd.Series:
    return (1.0 + r).groupby(r.index.year).prod() - 1.0


def rolling_sharpe(excess: pd.Series, window_days: int = 756) -> pd.Series:
    m = excess.rolling(window_days).mean()
    s = excess.rolling(window_days).std()
    return m / s * math.sqrt(TRADING_DAYS)


def trade_stats(frame: pd.DataFrame) -> pd.Series:
    """Statistics over invested spells (consecutive days with position = 1)."""
    p = frame["pos"].fillna(0.0).to_numpy()
    asset = frame["asset"].fillna(0.0).to_numpy()
    cash = frame["cash"].fillna(0.0).to_numpy()
    spells, start = [], None
    for i, v in enumerate(p):
        if v > 0 and start is None:
            start = i
        if (v == 0 or i == len(p) - 1) and start is not None:
            end = i if v > 0 else i - 1
            a = np.prod(1 + asset[start:end + 1]) - 1
            c = np.prod(1 + cash[start:end + 1]) - 1
            spells.append((end - start + 1, a, a - c))
            start = None
    if not spells:
        return pd.Series({"Invested spells": 0})
    lengths, rets, rel = map(np.array, zip(*spells))
    wins = rel > 0
    return pd.Series({
        "Invested spells": len(spells),
        "Median spell (days)": float(np.median(lengths)),
        "Mean spell (days)": float(lengths.mean()),
        "Spells beating cash": wins.mean(),
        "Avg winning spell": rets[wins].mean() if wins.any() else np.nan,
        "Avg losing spell": rets[~wins].mean() if (~wins).any() else np.nan,
        "Best spell": rets.max(),
        "Worst spell": rets.min(),
    })


def capture_ratios(r: pd.Series, bench: pd.Series) -> pd.Series:
    """Monthly up/down capture versus a benchmark."""
    m = (1 + r).resample("ME").prod() - 1
    b = (1 + bench).resample("ME").prod() - 1
    up, dn = b > 0, b < 0
    diff = m - b
    differs = diff.abs() > 1e-9  # fully-invested months match the benchmark exactly
    return pd.Series({
        "Up capture": m[up].mean() / b[up].mean(),
        "Down capture": m[dn].mean() / b[dn].mean(),
        "Months differing from bench (share)": differs.mean(),
        "Monthly hit rate vs bench (differing months)": (diff[differs] > 0).mean(),
    })


def alpha_beta(r_ex: pd.Series, b_ex: pd.Series, hac_lags: int = 10) -> pd.Series:
    """CAPM regression of strategy excess on benchmark excess (Newey-West SE)."""
    import statsmodels.api as sm

    df = pd.concat([r_ex, b_ex], axis=1).dropna()
    X = sm.add_constant(df.iloc[:, 1].to_numpy())
    fit = sm.OLS(df.iloc[:, 0].to_numpy(), X).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return pd.Series({
        "Alpha (ann.)": fit.params[0] * TRADING_DAYS,
        "Alpha t-stat (HAC)": fit.tvalues[0],
        "Beta": fit.params[1],
        "Beta t-stat (HAC)": fit.tvalues[1],
        "R-squared": fit.rsquared,
    })


# =============================================================================
# 5. Statistical inference & overfitting diagnostics
# =============================================================================
def sharpe_std_error(sr: float, n: int, skew: float, kurt: float) -> float:
    """Std. error of a *non-annualised* Sharpe ratio under non-normal returns
    (Mertens 2002 / Bailey & Lopez de Prado 2012). ``kurt`` is Pearson
    (non-excess) kurtosis."""
    return math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12) / (n - 1))


def probabilistic_sharpe(excess: np.ndarray | pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true SR > ``sr_benchmark``); both in *daily* (non-annualised) units."""
    x = np.asarray(excess, dtype=float)
    x = x[~np.isnan(x)]
    sr = sharpe(x, annualize=False)
    se = sharpe_std_error(sr, len(x), stats.skew(x), stats.kurtosis(x, fisher=False))
    return float(stats.norm.cdf((sr - sr_benchmark) / se))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum of ``n_trials`` Sharpe estimates under a zero-skill
    null (Bailey & Lopez de Prado 2014, eq. 6). Daily units."""
    if n_trials <= 1:
        return 0.0
    n = float(n_trials)
    return math.sqrt(sr_variance) * (
        (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n)
        + EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    )


def deflated_sharpe(selected_excess: np.ndarray | pd.Series,
                    trial_sharpes_daily: np.ndarray,
                    n_trials: int | None = None) -> dict:
    """Deflated Sharpe Ratio: PSR against the Sharpe a zero-skill researcher
    would expect to find as the best of ``n_trials`` attempts."""
    trial = np.asarray(trial_sharpes_daily, dtype=float)
    n = len(trial) if n_trials is None else n_trials
    sr0 = expected_max_sharpe(n, float(np.var(trial, ddof=1)) if len(trial) > 1 else 0.0)
    return {
        "Trials": n,
        "Observed SR (ann.)": sharpe(selected_excess),
        "Null max SR (ann.)": sr0 * math.sqrt(TRADING_DAYS),
        "PSR (SR > 0)": probabilistic_sharpe(selected_excess, 0.0),
        "DSR": probabilistic_sharpe(selected_excess, sr0),
    }


def _sr_from_sums(s1: np.ndarray, s2: np.ndarray, n: int) -> np.ndarray:
    mean = s1 / n
    var = (s2 - n * mean ** 2) / (n - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return mean / np.sqrt(var)


def pbo_cscv(excess: pd.DataFrame, n_blocks: int = 16) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation (Bailey, Borwein, Lopez de Prado & Zhu 2017).

    ``excess`` is a T x N frame of daily excess returns, one column per
    configuration. The sample is cut into ``n_blocks`` contiguous blocks; for
    every way of choosing half the blocks as "in-sample" the best IS
    configuration is located and its rank in the complementary "out-of-sample"
    half is recorded. PBO is the share of splits in which the IS winner ranks
    at or below the OOS median.
    """
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    X = excess.dropna().to_numpy(dtype=float)
    T, N = X.shape
    m = T // n_blocks
    X = X[T - m * n_blocks:]
    blocks = X.reshape(n_blocks, m, N)
    s1, s2 = blocks.sum(axis=1), (blocks ** 2).sum(axis=1)
    half = n_blocks // 2
    logits, is_best, oos_best = [], [], []
    for combo in itertools.combinations(range(n_blocks), half):
        train = np.zeros(n_blocks, dtype=bool)
        train[list(combo)] = True
        sr_is = _sr_from_sums(s1[train].sum(0), s2[train].sum(0), m * half)
        sr_oos = _sr_from_sums(s1[~train].sum(0), s2[~train].sum(0), m * half)
        best = int(np.nanargmax(sr_is))
        rank = stats.rankdata(sr_oos)[best]  # 1 = worst ... N = best
        w = rank / (N + 1.0)
        logits.append(math.log(w / (1.0 - w)))
        is_best.append(sr_is[best])
        oos_best.append(sr_oos[best])
    logits = np.array(logits)
    is_best = np.array(is_best) * math.sqrt(TRADING_DAYS)
    oos_best = np.array(oos_best) * math.sqrt(TRADING_DAYS)
    slope, intercept = np.polyfit(is_best, oos_best, 1)
    return {
        "configs": N,
        "splits": len(logits),
        "pbo": float((logits <= 0).mean()),
        "logits": logits,
        "is_sharpe": is_best,
        "oos_sharpe": oos_best,
        "degradation_slope": float(slope),
        "degradation_intercept": float(intercept),
        "prob_oos_loss": float((oos_best < 0).mean()),
    }


def stationary_bootstrap_index(n: int, mean_block: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap resampling indices."""
    new_block = rng.random(n) < 1.0 / mean_block
    new_block[0] = True
    block_id = np.cumsum(new_block) - 1
    block_start_t = np.flatnonzero(new_block)
    starts = rng.integers(0, n, size=len(block_start_t))
    offset = np.arange(n) - block_start_t[block_id]
    return (starts[block_id] + offset) % n


def bootstrap_sharpe_difference(a_excess: pd.Series, b_excess: pd.Series,
                                n_boot: int = 2000, mean_block: float = 63,
                                seed: int = 0) -> dict:
    """Paired stationary-bootstrap test of SR(a) - SR(b) (annualised).

    The one-sided p-value tests H0: SR(a) <= SR(b) using the bootstrap
    distribution re-centred on zero.
    """
    df = pd.concat([a_excess, b_excess], axis=1).dropna()
    A, B = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    obs = sharpe(A) - sharpe(B)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ix = stationary_bootstrap_index(len(A), mean_block, rng)
        diffs[i] = sharpe(A[ix]) - sharpe(B[ix])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = (np.sum(diffs - obs >= obs) + 1) / (n_boot + 1)
    return {"SR difference": obs, "95% CI low": lo, "95% CI high": hi,
            "p-value (H0: no improvement)": p, "draws": diffs}


def circular_shift_test(frame: pd.DataFrame, cost_bps: float, n: int = 1000,
                        min_shift: int = 252, seed: int = 0) -> dict:
    """Timing-skill test: rotate the position series by random offsets.

    Rotation keeps exposure, switch frequency and spell lengths identical while
    destroying the alignment between signal and subsequent returns. If the
    real alignment is no better than random rotations, the strategy's edge is
    just its average exposure.
    """
    pos = frame["pos"].fillna(0.0).to_numpy()
    ret = frame["asset"].to_numpy()
    cash = frame["cash"].to_numpy()
    T = len(pos)
    min_shift = min(min_shift, T // 4)  # short samples (e.g. a 19-month OOS) still get a valid null
    actual = sharpe(positions_to_returns(pos, ret, cash, cost_bps) - cash)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, T - min_shift, size=n)
    null = np.array([
        sharpe(positions_to_returns(np.roll(pos, k), ret, cash, cost_bps) - cash)
        for k in shifts
    ])
    return {"actual": actual, "null": null,
            "p-value": (1 + np.sum(null >= actual)) / (n + 1),
            "null 95th pct": np.percentile(null, 95)}


def walk_forward(frames: Mapping[str, pd.DataFrame], start, end=None,
                 min_train_years: int = 5, train_years: int | None = None,
                 cost_bps: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annual walk-forward re-selection of the lookback.

    At each calendar year-end the lookback with the best trailing Sharpe
    (expanding window from ``start``, or a rolling ``train_years`` window) is
    chosen and its positions are used for the following year. Selection at
    year-end y only uses returns up to y, so the procedure is causal.
    """
    names = list(frames)
    any_f = frames[names[0]]
    pos = pd.DataFrame({k: f["pos"] for k, f in frames.items()})
    ex = pd.DataFrame({k: f["net"] - f["cash"] for k, f in frames.items()})
    pos, ex = window(pos, start, end), window(ex, start, end)
    dates = ex.index
    year_ends = dates.to_series().groupby(dates.year).max()
    first_refit = pd.Timestamp(start) + pd.DateOffset(years=min_train_years)
    refits = [d for d in year_ends if d >= first_refit - pd.Timedelta(days=7)]
    choice = pd.Series(index=dates, dtype=object)
    log = []
    for i, d in enumerate(refits):
        lo = pd.Timestamp(start) if train_years is None else d - pd.DateOffset(years=train_years)
        train = ex.loc[lo:d]
        scores = train.apply(sharpe)
        pick = scores.idxmax()
        nxt = refits[i + 1] if i + 1 < len(refits) else dates[-1]
        choice.loc[(dates > d) & (dates <= nxt)] = pick
        log.append({"refit date": d.date(), "chosen": pick,
                    **{f"train SR {k}": v for k, v in scores.items()}})
    choice = choice.dropna()
    comp_pos = pd.Series([pos.at[t, c] for t, c in choice.items()], index=choice.index)
    asset = any_f["asset"].reindex(choice.index)
    cash = any_f["cash"].reindex(choice.index)
    p = comp_pos.fillna(0.0)
    turnover = p.diff().abs().fillna(0.0)
    net = p * asset + (1 - p) * cash - turnover * cost_bps / 1e4
    bt = pd.DataFrame({"choice": choice, "pos": comp_pos, "turnover": turnover,
                       "asset": asset, "cash": cash, "net": net})
    return bt, pd.DataFrame(log)


# =============================================================================
# 6. Lookahead-bias audits
# =============================================================================
def _same(a: pd.Series, b: pd.Series) -> bool:
    a, b = a.align(b, join="inner")
    return bool(np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))


def truncation_check(close: pd.Series, lookbacks: Mapping[str, object],
                     cutoffs) -> pd.DataFrame:
    """Signals computed on data truncated at each cutoff must equal the
    full-sample signals up to that cutoff (no dependence on future rows)."""
    rows = []
    for name, lb in lookbacks.items():
        full = signal(close, lb)
        ok = all(_same(full.loc[:c], signal(close.loc[:c], lb)) for c in cutoffs)
        rows.append({"lookback": name, "cutoffs tested": len(cutoffs),
                     "identical": ok})
    return pd.DataFrame(rows).set_index("lookback")


def future_perturbation_check(market: Market, lookbacks: Mapping[str, object],
                              cutoffs, cost_bps: float, lag: int = 1,
                              seed: int = 0) -> pd.DataFrame:
    """Replace every price/return after each cutoff with random noise and
    confirm that signals, positions and strategy returns up to the cutoff do
    not change."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, lb in lookbacks.items():
        base = backtest(signal(market.close, lb), market.ret, market.cash, cost_bps, lag)
        ok = True
        for c in cutoffs:
            after = market.close.index > pd.Timestamp(c)
            noise = np.exp(np.cumsum(rng.normal(0, 0.03, after.sum())))
            close2 = market.close.copy()
            close2[after] = market.close[~after].iloc[-1] * noise
            ret2 = market.ret.copy()
            ret2[after] = rng.normal(0, 0.03, after.sum())
            alt = backtest(signal(close2, lb), ret2, market.cash, cost_bps, lag)
            for col in ("signal", "pos", "net"):
                ok &= _same(base[col].loc[:c], alt[col].loc[:c])
        rows.append({"lookback": name, "cutoffs tested": len(cutoffs),
                     "unchanged up to cutoff": ok})
    return pd.DataFrame(rows).set_index("lookback")


def same_bar_check(market: Market, lookback, lag: int, n_dates: int = 250,
                   shock: float = 0.05, seed: int = 0) -> float:
    """Share of test dates on which shocking *that day's* return changes the
    position held over that same day. A causal rule must return 0.

    The shock multiplies every price from day t onwards by ``1 + shock``,
    which alters only the day-t return.
    """
    rng = np.random.default_rng(seed)
    close = market.close
    sig0 = signal(close, lookback)
    pos0 = sig0.shift(lag)
    valid = np.flatnonzero(pos0.notna().to_numpy())
    test_ix = rng.choice(valid[1:], size=min(n_dates, len(valid) - 1), replace=False)
    px = close.to_numpy(dtype=float)
    changed = 0
    for i in test_ix:
        bumped = px.copy()
        bumped[i:] *= 1.0 + rng.choice([-shock, shock])
        pos1 = signal(pd.Series(bumped, index=close.index), lookback).shift(lag)
        changed += pos1.iloc[i] != pos0.iloc[i]
    return changed / len(test_ix)


# =============================================================================
# 7. Plotting helpers (matplotlib)
# =============================================================================
# Fixed entity -> colour mapping (colour follows the entity everywhere in the
# report). Categorical slots validated for colour-vision deficiency.
COLORS = {
    "1Y": "#2a78d6",       # slot 1 blue
    "90D": "#eb6834",      # slot 2 orange
    "30D": "#1baf7a",      # slot 3 aqua
    "1W": "#eda100",       # slot 4 yellow
    "Walk-forward": "#4a3aa7",  # slot 7 violet
    "Leaky": "#e34948",    # slot 8 red
    "benchmark": "#52514e",   # secondary ink: buy & hold of the same market
    "benchmark2": "#898781",  # muted ink: second benchmark
}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}


def set_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "figure.dpi": 110, "figure.figsize": (11, 4.6),
        "font.family": "sans-serif", "font.size": 10,
        "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK["secondary"],
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "text.color": INK["primary"], "lines.linewidth": 1.6,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def color_for(name: str, default: str = "#2a78d6") -> str:
    for key, col in COLORS.items():
        if name == key or name.startswith(key + " "):
            return col
    return default


def _zorder(name: str) -> int:
    return 2 if "B&H" in name else 3  # benchmarks sit behind strategies


def _money(v: float) -> str:
    return f"${v:,.0f}" if v >= 10 else f"${v:,.2g}"


def log_money_axis(ax) -> None:
    """Log y-axis with plain dollar labels (minor ticks labelled at 2x and 5x)."""
    import matplotlib.ticker as mtick

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: _money(v)))

    def minor(v, _):
        lead = int(round(v / 10 ** math.floor(math.log10(v))))
        return _money(v) if lead in (2, 5) else ""

    ax.yaxis.set_minor_formatter(mtick.FuncFormatter(minor))


def _spread_labels(ax, labels: list[tuple[float, str]], x, log: bool) -> None:
    """Place end-of-line labels, nudging them apart so none overlap."""
    if not labels:
        return
    f = (lambda v: math.log10(v)) if log else (lambda v: v)
    g = (lambda v: 10 ** v) if log else (lambda v: v)
    lo, hi = (f(v) for v in ax.get_ylim())
    gap = 0.045 * (hi - lo)
    pts = sorted((f(v), text) for v, text in labels)
    ys = [pts[0][0]]
    for y, _ in pts[1:]:
        ys.append(max(y, ys[-1] + gap))
    for y, (_, text) in zip(ys, pts):
        ax.annotate(text, (x, g(y)), xytext=(4, 0), textcoords="offset points",
                    color=INK["secondary"], fontsize=8, va="center",
                    annotation_clip=False)


def plot_wealth(series: Mapping[str, pd.Series], title: str, ax=None,
                colors: Mapping[str, str] | None = None, log: bool = True,
                label_ends: bool = True):
    """Growth of $1 on one (log) axis, with direct end labels."""
    import matplotlib.pyplot as plt

    import matplotlib.ticker as mtick

    colors = colors or {}
    ax = ax or plt.subplots()[1]
    ends, last_x, lo, hi = [], None, np.inf, -np.inf
    for name, r in series.items():
        w = (1 + r.fillna(0)).cumprod()
        c = colors.get(name, color_for(name))
        ax.plot(w.index, w.to_numpy(), color=c, label=name, lw=1.6, zorder=_zorder(name))
        ends.append((float(w.iloc[-1]), f"${w.iloc[-1]:,.2f}"))
        last_x = w.index[-1] if last_x is None else max(last_x, w.index[-1])
        lo, hi = min(lo, float(w.min())), max(hi, float(w.max()))
    log = log and hi / lo > 2.5  # a narrow range (short windows) reads better on a linear axis
    if log:
        log_money_axis(ax)
    else:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.2f}"))
    if label_ends:
        _spread_labels(ax, ends, last_x, log)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1" + (" (log scale)" if log else ""))
    ax.legend(loc="upper left")
    return ax


def plot_drawdowns(series: Mapping[str, pd.Series], title: str, ax=None,
                   colors: Mapping[str, str] | None = None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    colors = colors or {}
    ax = ax or plt.subplots(figsize=(11, 3.4))[1]
    for name, r in series.items():
        dd = drawdown(r)
        c = colors.get(name, color_for(name))
        ax.plot(dd.index, dd.to_numpy(), color=c, label=f"{name} (max {dd.min():.0%})",
                lw=1.2, zorder=_zorder(name))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left")
    return ax


# =============================================================================
# 8. Table formatting
# =============================================================================
_PCT_FIRST = ("PSR", "DSR", "PBO", "Prob")
_NOT_PCT = ("(yrs)", "(days)", "(bps)", "bps", "Sharpe", "SR", "t-stat", "Beta",
            "R-squared", "Skew", "kurtosis", "Years", "Switches", "Sortino",
            "Calmar", "slope", "Trials", "spells", "splits", "configs", "CI")
_PCT = ("CAGR", "volatility", "drawdown", "DD", "VaR", "day", "Positive",
        "Time in market", "beating", "spell", "capture", "hit rate", "Alpha",
        "return", "Return", "share", "Share", "changed")


def _fmt_value(key: str, v):
    if isinstance(v, (bool, np.bool_)):
        return "\u2713" if v else "\u2717"
    if isinstance(v, (int, np.integer)):
        return f"{v:,d}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "\u2013"
        if "p-value" in key:
            return f"{v:.3f}"
        if any(k in key for k in _PCT_FIRST):
            return f"{v:.1%}"
        if any(k in key for k in _NOT_PCT):
            return f"{v:,.0f}" if float(v).is_integer() and abs(v) > 1 else f"{v:,.2f}"
        if any(k in key for k in _PCT):
            return f"{v:.1%}"
        return f"{v:,.2f}"
    return v


def format_table(df: pd.DataFrame, by: str = "index") -> pd.DataFrame:
    """Human-readable copy of a results table. ``by`` says whether metric
    names sit in the index (one row per metric) or in the columns."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    out = df.astype(object).copy()
    for r in df.index:
        for c in df.columns:
            key = str(r if by == "index" else c)
            out.at[r, c] = _fmt_value(key, df.at[r, c])
    return out
