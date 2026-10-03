"""Performance, risk and trade statistics on daily return series.

All functions take *daily simple returns* (fractions) indexed by date. ``rf`` is the daily risk-free
return on the same dates (optional, defaults to 0). Annualisation uses 252 trading days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TD = 252
EULER_GAMMA = 0.5772156649015329


def _clean(r: pd.Series) -> pd.Series:
    return pd.Series(r, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def _rf_for(r: pd.Series, rf: pd.Series | None) -> pd.Series:
    if rf is None:
        return pd.Series(0.0, index=r.index)
    return rf.reindex(r.index).fillna(0.0)


def wealth(r: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1 + _clean(r)).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    w = wealth(r)
    return w / w.cummax().clip(lower=1.0) - 1


def cagr(r: pd.Series) -> float:
    r = _clean(r)
    if len(r) == 0:
        return np.nan
    tot = (1 + r).prod()
    yrs = len(r) / TD
    return tot ** (1 / yrs) - 1 if tot > 0 else -1.0


def sharpe(r: pd.Series, rf: pd.Series | None = None) -> float:
    r = _clean(r)
    ex = r - _rf_for(r, rf)
    sd = ex.std()
    return ex.mean() / sd * np.sqrt(TD) if len(ex) > 2 and sd > 0 else np.nan


def sortino(r: pd.Series, rf: pd.Series | None = None) -> float:
    r = _clean(r)
    ex = r - _rf_for(r, rf)
    dd = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return ex.mean() / dd * np.sqrt(TD) if dd > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    d = drawdown(r)
    return float(d.min()) if len(d) else np.nan


def underwater_periods(r: pd.Series) -> pd.DataFrame:
    """Every drawdown episode: start (peak), trough, end (recovery or NaT), depth, lengths in trading days."""
    r = _clean(r)
    if r.empty:
        return pd.DataFrame(columns=["peak", "trough", "recovery", "depth", "days_to_trough", "days_underwater",
                                     "recovered"])
    w = (1 + r).cumprod()
    peak = w.cummax().clip(lower=1.0)
    dd = w / peak - 1
    under = dd < -1e-12
    rows = []
    i, n, idx = 0, len(dd), dd.index
    vals = dd.to_numpy()
    while i < n:
        if under.iat[i]:
            j = i
            while j < n and under.iat[j]:
                j += 1
            seg = vals[i:j]
            t = i + int(np.argmin(seg))
            rows.append(dict(peak=idx[i - 1] if i > 0 else idx[i], trough=idx[t],
                             recovery=idx[j] if j < n else pd.NaT, depth=float(seg.min()),
                             days_to_trough=t - i + 1, days_underwater=j - i, recovered=j < n))
            i = j
        else:
            i += 1
    return pd.DataFrame(rows)


def drawdown_table(r: pd.Series, top: int = 5) -> pd.DataFrame:
    t = underwater_periods(r)
    return t.nsmallest(top, "depth").reset_index(drop=True) if len(t) else t


def monthly_returns(r: pd.Series) -> pd.Series:
    r = _clean(r)
    return (1 + r).groupby([r.index.year, r.index.month]).prod() - 1


def monthly_table(r: pd.Series) -> pd.DataFrame:
    """Year x month table of returns plus the calendar-year total."""
    r = _clean(r)
    if r.empty:
        return pd.DataFrame()
    m = monthly_returns(r).unstack()
    m = m.reindex(columns=range(1, 13))
    m.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    m["Year"] = (1 + r).groupby(r.index.year).prod() - 1
    m.index.name = None
    return m


def period_return(r: pd.Series, start=None, end=None) -> float:
    r = _clean(r).loc[start:end]
    return float((1 + r).prod() - 1) if len(r) else np.nan


def sharpe_std_error(sr_period: float, n: int, skew: float, kurt: float) -> float:
    """Standard error of a per-period Sharpe ratio (Mertens / Lo, non-normal returns; kurt = Pearson)."""
    return float(np.sqrt(max(1 - skew * sr_period + (kurt - 1) / 4 * sr_period ** 2, 1e-12) / max(n - 1, 1)))


def psr(r: pd.Series, rf: pd.Series | None = None, sr_benchmark_annual: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true Sharpe > benchmark Sharpe) given track length, skew and kurtosis."""
    r = _clean(r)
    ex = (r - _rf_for(r, rf)).to_numpy()
    if len(ex) < 10 or ex.std(ddof=1) == 0:
        return np.nan
    sr = ex.mean() / ex.std(ddof=1)
    se = sharpe_std_error(sr, len(ex), stats.skew(ex), stats.kurtosis(ex, fisher=False))
    return float(stats.norm.cdf((sr - sr_benchmark_annual / np.sqrt(TD)) / se))


def min_track_record(r: pd.Series, rf: pd.Series | None = None, sr_benchmark_annual: float = 0.0,
                     prob: float = 0.95) -> float:
    """Minimum Track Record Length (years) to be ``prob`` confident the true Sharpe beats the benchmark.

    Uses the observed Sharpe, skew and kurtosis of ``r`` (Bailey & Lopez de Prado, 2012).
    """
    r = _clean(r)
    ex = (r - _rf_for(r, rf)).to_numpy()
    if len(ex) < 10 or ex.std(ddof=1) == 0:
        return np.nan
    sr = ex.mean() / ex.std(ddof=1)
    srb = sr_benchmark_annual / np.sqrt(TD)
    if sr <= srb:
        return np.inf
    g3, g4 = stats.skew(ex), stats.kurtosis(ex, fisher=False)
    z = stats.norm.ppf(prob)
    n = 1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (z / (sr - srb)) ** 2
    return float(n / TD)


def expected_max_sharpe(n_trials: int, sr_std_annual: float) -> float:
    """Expected maximum Sharpe of ``n_trials`` skill-less strategies (for the Deflated Sharpe Ratio)."""
    if n_trials < 2:
        return 0.0
    return float(sr_std_annual * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                                  + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e))))


def xirr(dates, amounts) -> float:
    """Annualised money-weighted return (XIRR).

    ``amounts`` are from the investor's side: deposits negative, withdrawals and the final value positive.
    """
    from scipy.optimize import brentq
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    a = np.asarray(amounts, dtype=float)
    ok = np.isfinite(a) & (np.abs(a) > 0)
    d, a = d[ok], a[ok]
    if len(a) < 2 or not (a > 0).any() or not (a < 0).any():
        return np.nan
    t = np.asarray((d - d.min()).days, dtype=float) / 365.0          # Excel XIRR convention
    if t.max() <= 0:
        return np.nan
    f = lambda r: float(np.sum(a / (1 + r) ** t))
    try:
        return float(brentq(f, -0.9999, 1000.0, maxiter=500))
    except (ValueError, RuntimeError):
        return np.nan


def beta_alpha(r: pd.Series, bench: pd.Series, rf: pd.Series | None = None) -> tuple[float, float, float]:
    """(beta, annualised alpha, correlation) of excess returns vs the benchmark's excess returns."""
    r = _clean(r)
    b = pd.Series(bench, dtype=float).reindex(r.index)
    ok = b.notna()
    r, b = r[ok], b[ok]
    if len(r) < 10 or b.var() == 0:
        return np.nan, np.nan, np.nan
    f = _rf_for(r, rf)
    re, be = r - f, b - f
    beta = np.cov(re, be)[0, 1] / be.var()
    return float(beta), float((re.mean() - beta * be.mean()) * TD), float(np.corrcoef(r, b)[0, 1])


def capture(r: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """Up / down capture on monthly returns."""
    m = monthly_returns(r)
    bm = monthly_returns(pd.Series(bench, dtype=float).reindex(_clean(r).index))
    m, bm = m.align(bm, join="inner")
    up, dn = bm > 0, bm < 0
    upc = m[up].mean() / bm[up].mean() if up.sum() else np.nan
    dnc = m[dn].mean() / bm[dn].mean() if dn.sum() else np.nan
    return float(upc), float(dnc)


def _days_since_peak(dd: pd.Series) -> int:
    at_peak = np.flatnonzero(dd.to_numpy() >= -1e-12)
    return int(len(dd) - 1 - at_peak[-1]) if len(at_peak) else len(dd)


def perf_summary(r: pd.Series, rf: pd.Series | None = None, bench: pd.Series | None = None) -> dict:
    """The full metrics block shown in every tearsheet."""
    r = _clean(r)
    out: dict = {"Start": r.index[0] if len(r) else pd.NaT, "End": r.index[-1] if len(r) else pd.NaT,
                 "Days": len(r), "Years": len(r) / TD}
    if len(r) < 2:
        return out
    f = _rf_for(r, rf)
    ex = r - f
    w = (1 + r).cumprod()
    dd = w / w.cummax().clip(lower=1.0) - 1
    uw = underwater_periods(r)
    m = monthly_returns(r)
    var95 = r.quantile(0.05)
    var99 = r.quantile(0.01)
    sd = r.std()
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    out.update({
        "Total return": w.iloc[-1] - 1,
        "CAGR": cagr(r),
        "Volatility": sd * np.sqrt(TD),
        "Sharpe": sharpe(r, f),
        "Sortino": sortino(r, f),
        "Max drawdown": dd.min(),
        "Current drawdown": dd.iloc[-1],
        "Calmar": cagr(r) / abs(dd.min()) if dd.min() < 0 else np.nan,
        "Longest drawdown (days)": int(uw["days_underwater"].max()) if len(uw) else 0,
        "Days since peak": _days_since_peak(dd),
        "Ulcer index": float(np.sqrt((dd ** 2).mean())),
        "VaR 95% (1d)": var95,
        "CVaR 95% (1d)": r[r <= var95].mean(),
        "VaR 99% (1d)": var99,
        "Skew": float(stats.skew(r)),
        "Excess kurtosis": float(stats.kurtosis(r)),
        "Best day": r.max(),
        "Worst day": r.min(),
        "Best month": m.max() if len(m) else np.nan,
        "Worst month": m.min() if len(m) else np.nan,
        "% positive days": (r > 0).mean(),
        "% positive months": (m > 0).mean() if len(m) else np.nan,
        "Gain/pain ratio": gains / losses if losses > 0 else np.nan,
        "Tail ratio (95/5)": abs(r.quantile(0.95) / var95) if var95 < 0 else np.nan,
        "PSR (SR>0)": psr(r, f),
        "Excess return (ann.)": ex.mean() * TD,
    })
    if bench is not None and len(pd.Series(bench).dropna()):
        b, a, c = beta_alpha(r, bench, f)
        bb = pd.Series(bench, dtype=float).reindex(r.index).fillna(0.0)
        act = r - bb
        te = act.std() * np.sqrt(TD)
        up, dn = capture(r, bench)
        out.update({"Beta": b, "Alpha (ann.)": a, "Correlation": c, "Tracking error": te,
                    "Information ratio": act.mean() * TD / te if te > 0 else np.nan,
                    "Up capture": up, "Down capture": dn,
                    "Benchmark return": (1 + bb).prod() - 1})
    return out


def trade_summary(rt: pd.DataFrame) -> dict:
    """Round-trip statistics from ``Ledger.round_trips`` (or a backtest trade log with ret/pnl/days)."""
    if rt is None or len(rt) == 0:
        return {"Trades": 0}
    r = rt["ret"].astype(float)
    pnl = rt["pnl"].astype(float) if "pnl" in rt else r
    w, l = rt[r > 0], rt[r <= 0]
    streak = worst = 0
    order = rt.sort_values("exit_date")["ret"] if "exit_date" in rt else r
    for x in order:
        streak = streak + 1 if x <= 0 else 0
        worst = max(worst, streak)
    gross_win, gross_loss = pnl[r > 0].sum(), -pnl[r <= 0].sum()
    win_rate = len(w) / len(rt)
    avg_w = w["ret"].mean() if len(w) else 0.0
    avg_l = l["ret"].mean() if len(l) else 0.0
    return {"Trades": len(rt), "Win rate": win_rate, "Avg trade": r.mean(), "Median trade": r.median(),
            "Avg win": avg_w if len(w) else np.nan, "Avg loss": avg_l if len(l) else np.nan,
            "Payoff ratio": avg_w / abs(avg_l) if len(w) and len(l) and avg_l != 0 else np.nan,
            "Profit factor": gross_win / gross_loss if gross_loss > 0 else np.nan,
            "Expectancy": win_rate * avg_w + (1 - win_rate) * avg_l,
            "Best trade": r.max(), "Worst trade": r.min(), "Total P&L": pnl.sum() if "pnl" in rt else np.nan,
            "Avg holding (days)": rt["days"].mean() if "days" in rt else np.nan,
            "Max consecutive losses": worst}


def rolling_sharpe(r: pd.Series, rf: pd.Series | None = None, window: int = 126) -> pd.Series:
    r = _clean(r)
    ex = r - _rf_for(r, rf)
    return (ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(TD)).dropna()


def rolling_vol(r: pd.Series, window: int = 63) -> pd.Series:
    return (_clean(r).rolling(window).std() * np.sqrt(TD)).dropna()


def rolling_beta(r: pd.Series, bench: pd.Series, window: int = 126) -> pd.Series:
    r = _clean(r)
    b = pd.Series(bench, dtype=float).reindex(r.index)
    return (r.rolling(window).cov(b) / b.rolling(window).var()).dropna()


PCT_KEYS = {"Total return", "CAGR", "Volatility", "Max drawdown", "Current drawdown", "Ulcer index", "VaR 95% (1d)",
            "CVaR 95% (1d)", "VaR 99% (1d)", "Best day", "Worst day", "Best month", "Worst month", "% positive days",
            "% positive months", "PSR (SR>0)", "Excess return (ann.)", "Alpha (ann.)", "Tracking error",
            "Up capture", "Down capture", "Benchmark return", "Win rate", "Avg trade", "Median trade", "Avg win",
            "Avg loss", "Expectancy", "Best trade", "Worst trade", "Exposure", "Weight", "Return"}
INT_KEYS = {"Days", "Trades", "Longest drawdown (days)", "Days since peak", "Max consecutive losses"}


def fmt(key: str, v) -> str:
    """Human formatting for a metric value by name."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–" if not (isinstance(v, float) and np.isinf(v)) else "∞"
    if isinstance(v, pd.Timestamp):
        return f"{v:%Y-%m-%d}"
    if key in PCT_KEYS or key.endswith("%") or "return" in key.lower() and key not in INT_KEYS:
        return f"{v:.1%}" if abs(v) >= 0.001 or v == 0 else f"{v:.2%}"
    if key in INT_KEYS:
        return f"{v:,.0f}"
    if "P&L" in key or "($)" in key:
        return f"{v:,.0f}"
    if key == "Years":
        return f"{v:.2f}"
    return f"{v:,.2f}"


def summary_frame(series: dict[str, pd.Series], rf=None, bench=None, rows: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame({k: perf_summary(v, rf, bench) for k, v in series.items()})
    return df.reindex(rows) if rows else df


def format_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Format a metrics frame whose index holds metric names."""
    return df.apply(lambda row: row.map(lambda v: fmt(str(row.name), v)), axis=1)
