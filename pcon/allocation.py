"""Portfolio construction across strategies.

Answers three questions with the strategies' daily return series (backtest, optionally extended by live):

1. **Which strategies belong in the book?** Every subset is tested. For each strategy the add/remove
   analysis reports the Sharpe with and without it, and the textbook hurdle: adding a little of strategy
   *k* to portfolio *P* raises *P*'s Sharpe iff ``SR_k > corr(k, P) x SR_P``.
2. **How much capital each?** Equal weight, inverse volatility, equal risk contribution (risk parity),
   minimum variance, maximum diversification, maximum Sharpe, and a *resampled* maximum Sharpe (the
   average optimum over bootstrap resamples, which is much less sensitive to estimation error).
3. **Is the answer robust?** In-sample optimisation flatters every optimiser. The walk-forward test
   estimates weights on a trailing window only, holds them for the next quarter and stitches those
   out-of-sample quarters together. Sharpe differences come with a paired block-bootstrap p-value.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import metrics as M

TD = M.TD
METHODS = {
    "target": "Your target weights",
    "equal": "Equal weight",
    "inverse_vol": "Inverse volatility",
    "risk_parity": "Equal risk contribution",
    "min_variance": "Minimum variance",
    "max_diversification": "Maximum diversification",
    "max_sharpe": "Maximum Sharpe",
    "resampled": "Resampled max Sharpe",
}
FREQ = {"D": None, "W": "W", "M": "M", "Q": "Q", "A": "Y", "Y": "Y", "none": "none"}


# ------------------------------------------------------------------------------------------------
# data helpers
# ------------------------------------------------------------------------------------------------
def align(series: dict[str, pd.Series], start=None, end=None) -> pd.DataFrame:
    """Inner-joined daily returns (only dates where every strategy has a return)."""
    if not series:
        return pd.DataFrame()
    R = pd.concat({k: M._clean(v) for k, v in series.items()}, axis=1, join="inner").dropna()
    if start is not None:
        R = R.loc[pd.Timestamp(start):]
    if end is not None:
        R = R.loc[:pd.Timestamp(end)]
    return R


def _period_key(index: pd.DatetimeIndex, rebalance: str):
    f = FREQ.get(rebalance, rebalance)
    if f == "none":
        return np.zeros(len(index), dtype=int)
    return np.asarray(index.to_period(f).astype(str))


def portfolio_returns(R: pd.DataFrame, w: pd.Series, rebalance: str = "M",
                      contributions: bool = False):
    """Daily returns of a fixed-weight mix, rebalanced at the start of each period (weights drift inside).

    Any weight not allocated (sum < 1) sits in cash at 0 %. With ``contributions=True`` also returns each
    strategy's daily contribution (drifted weight x return), which sums to the portfolio return.
    """
    w = pd.Series(w, dtype=float).reindex(R.columns).fillna(0.0)
    cash_w = 1.0 - w.sum()
    if FREQ.get(rebalance, rebalance) is None:
        contrib = R * w
        r = contrib.sum(axis=1)
        return (r, contrib) if contributions else r
    key = _period_key(R.index, rebalance)
    H = ((1 + R).groupby(key).cumprod() * w.to_numpy()).to_numpy()   # sleeve values vs NAV at last rebalance
    prev = np.vstack([w.to_numpy()[None, :], H[:-1]])
    first = np.r_[True, key[1:] != key[:-1]]
    prev[first] = w.to_numpy()                                         # weights right after rebalancing
    prev_v = prev.sum(axis=1) + cash_w
    r = pd.Series((H.sum(axis=1) + cash_w) / prev_v - 1, index=R.index)
    if contributions:
        contrib = pd.DataFrame(prev / prev_v[:, None], index=R.index, columns=R.columns) * R
        return r, contrib
    return r


# ------------------------------------------------------------------------------------------------
# estimators and optimisers
# ------------------------------------------------------------------------------------------------
def ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance towards a scaled identity."""
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    d2 = ((S - F) ** 2).sum()
    # sum_t ||x_t x_t' - S||^2 = sum_t ||x_t||^4 - T ||S||^2
    b2 = (((Xc ** 2).sum(axis=1) ** 2).sum() - T * (S ** 2).sum()) / T ** 2
    b2 = min(b2, d2)
    s = b2 / d2 if d2 > 0 else 0.0
    return s * F + (1 - s) * S


def estimate(R: pd.DataFrame, rf: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Annualised excess mean vector and Ledoit-Wolf covariance."""
    X = R.to_numpy(float)
    f = rf.reindex(R.index).fillna(0.0).to_numpy()[:, None] if rf is not None else 0.0
    mu = (X - f).mean(axis=0) * TD
    cov = ledoit_wolf(X) * TD
    return mu, cov


def _bounds(n: int, lo: float, hi: float):
    lo = min(lo, 1.0 / n)
    hi = max(hi, 1.0 / n)
    return [(lo, hi)] * n


def _solve(obj, n: int, lo: float, hi: float, x0=None, jac=None) -> np.ndarray:
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    res = minimize(obj, x0, jac=jac, method="SLSQP", bounds=_bounds(n, lo, hi),
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x if res.success else x0, 0, None)
    return w / w.sum()


def _risk_parity(cov: np.ndarray, lo: float, hi: float) -> np.ndarray:
    n = len(cov)
    obj = lambda y: 0.5 * y @ cov @ y - np.log(y).sum() / n
    res = minimize(obj, np.full(n, 1.0 / np.sqrt(np.diag(cov)).mean()), method="L-BFGS-B",
                   bounds=[(1e-10, None)] * n)
    w = res.x / res.x.sum()
    if lo > 0 or hi < 1:                                   # box constraints: refine with SLSQP from the ERC point
        rc = lambda v: v * (cov @ v) / (v @ cov @ v)
        w = _solve(lambda v: ((rc(v) - 1.0 / n) ** 2).sum() * 1e4, n, lo, hi, np.clip(w, lo, hi))
    return w


def _ratio(a: np.ndarray, cov: np.ndarray):
    """Objective and gradient of -(w.a) / sqrt(w'Cw): max Sharpe (a = mu) or max diversification (a = vol)."""
    def f(w):
        return -(w @ a) / np.sqrt(w @ cov @ w)

    def g(w):
        s2 = w @ cov @ w
        s = np.sqrt(s2)
        return -(a * s - (w @ a) * (cov @ w) / s) / s2
    return f, g


def optimize(R: pd.DataFrame, method: str, rf: pd.Series | None = None, lo: float = 0.0, hi: float = 1.0,
             target: pd.Series | None = None, n_resamples: int = 100, seed: int = 11) -> pd.Series:
    """Long-only weights summing to 1 for the columns of ``R``."""
    cols = list(R.columns)
    n = len(cols)
    if n == 1:
        return pd.Series([1.0], index=cols)
    if method == "equal":
        return pd.Series(1.0 / n, index=cols)
    if method == "target":
        t = (target if target is not None else pd.Series(1.0, index=cols)).reindex(cols).fillna(0.0)
        return t / t.sum() if t.sum() > 0 else pd.Series(1.0 / n, index=cols)
    if method == "resampled":
        W = resampled_weights(R, rf, lo, hi, n_resamples, seed=seed)
        return W.mean().reindex(cols) / W.mean().sum()
    mu, cov = estimate(R, rf)
    vol = np.sqrt(np.diag(cov))
    if method == "inverse_vol":
        w = (1 / vol) / (1 / vol).sum()
        w = _solve(lambda v: ((v - w) ** 2).sum(), n, lo, hi, w) if (lo > 0 or hi < 1) else w
    elif method == "risk_parity":
        w = _risk_parity(cov, lo, hi)
    elif method == "min_variance" or (method == "max_sharpe" and (mu <= 0).all()):
        w = _solve(lambda v: v @ cov @ v, n, lo, hi, jac=lambda v: 2 * cov @ v)
    elif method == "max_diversification":
        f, g = _ratio(vol, cov)
        w = _solve(f, n, lo, hi, jac=g)
    elif method == "max_sharpe":
        f, g = _ratio(mu, cov)
        w = _solve(f, n, lo, hi, jac=g)
    else:
        raise ValueError(f"unknown method {method}")
    return pd.Series(w, index=cols)


def resampled_weights(R: pd.DataFrame, rf: pd.Series | None = None, lo: float = 0.0, hi: float = 1.0,
                      n: int = 200, block: int = 63, seed: int = 11) -> pd.DataFrame:
    """Max-Sharpe weights on ``n`` block-bootstrap resamples (Michaud-style resampled efficiency)."""
    from .expectations import block_indices
    T = len(R)
    idx = block_indices(T, T, n, block, seed)
    X = R.to_numpy(float)
    f = rf.reindex(R.index).fillna(0.0).to_numpy() if rf is not None else np.zeros(T)
    out = []
    for k in range(n):
        Rk = pd.DataFrame(X[idx[k]], columns=R.columns)
        fk = pd.Series(f[idx[k]])
        out.append(optimize(Rk, "max_sharpe", fk, lo, hi).to_numpy())
    return pd.DataFrame(out, columns=R.columns)


# ------------------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------------------
def stats_row(r: pd.Series, rf: pd.Series | None = None) -> dict:
    r = M._clean(r)
    if len(r) < 20:
        return {}
    m = M.monthly_returns(r)
    return {"CAGR": M.cagr(r), "Volatility": r.std() * np.sqrt(TD), "Sharpe": M.sharpe(r, rf),
            "Sortino": M.sortino(r, rf), "Max drawdown": M.max_drawdown(r),
            "Calmar": M.cagr(r) / abs(M.max_drawdown(r)) if M.max_drawdown(r) < 0 else np.nan,
            "Worst month": m.min(), "% positive months": (m > 0).mean()}


def walk_forward(R: pd.DataFrame, method: str, rf: pd.Series | None = None, lookback_days: int = 756,
                 step: str = "Q", rebalance: str = "M", lo: float = 0.0, hi: float = 1.0,
                 target: pd.Series | None = None, n_resamples: int = 50) -> tuple[pd.Series, pd.DataFrame]:
    """Out-of-sample returns of an allocation *method*: weights estimated on the trailing ``lookback_days``
    only, then held (rebalanced per ``rebalance``) through the next ``step`` period."""
    if len(R) <= lookback_days + 20:
        return pd.Series(dtype=float), pd.DataFrame()
    oos = R.iloc[lookback_days:]
    keys = _period_key(oos.index, step)
    rets, weights = [], {}
    for k in pd.unique(keys):
        block = oos[keys == k]
        first = R.index.get_loc(block.index[0])
        est = R.iloc[max(0, first - lookback_days):first]
        w = optimize(est, method, rf, lo, hi, target, n_resamples)
        weights[block.index[0]] = w
        rets.append(portfolio_returns(block, w, rebalance))
    return pd.concat(rets), pd.DataFrame(weights).T


def sharpe_diff_test(a: pd.Series, b: pd.Series, rf: pd.Series | None = None, n: int = 1000,
                     block: int = 63, seed: int = 3) -> dict:
    """Paired block bootstrap of SR(a) - SR(b): observed difference, 90 % interval, P(diff <= 0)."""
    from .expectations import block_indices
    x = pd.concat([M._clean(a), M._clean(b)], axis=1, join="inner").dropna()
    if len(x) < 60:
        return {}
    f = rf.reindex(x.index).fillna(0.0).to_numpy()[:, None] if rf is not None else 0.0
    X = x.to_numpy() - f
    idx = block_indices(len(X), len(X), n, block, seed)
    S = X[idx]
    sr = S.mean(axis=1) / S.std(axis=1, ddof=1) * np.sqrt(TD)
    d = sr[:, 0] - sr[:, 1]
    obs = X.mean(0) / X.std(0, ddof=1) * np.sqrt(TD)
    return {"diff": float(obs[0] - obs[1]), "lo": float(np.quantile(d, 0.05)), "hi": float(np.quantile(d, 0.95)),
            "p_not_better": float((d <= 0).mean())}


def explore(R: pd.DataFrame, rf: pd.Series | None = None, methods=("equal", "risk_parity", "max_sharpe"),
            rebalance: str = "M", lo: float = 0.0, hi: float = 1.0, target: pd.Series | None = None,
            walk_forward_days: int | None = 756, step: str = "Q", n_resamples: int = 50) -> pd.DataFrame:
    """Every non-empty subset of strategies x every method: in-sample and walk-forward statistics."""
    rows = []
    cols = list(R.columns)
    for size in range(1, len(cols) + 1):
        for sub in itertools.combinations(cols, size):
            Rs = R[list(sub)]
            for method in (methods if size > 1 else ("single",)):
                if method == "target" and target is not None and target.reindex(list(sub)).fillna(0).sum() <= 0:
                    continue
                w = optimize(Rs, method if size > 1 else "equal", rf, lo, hi, target, n_resamples)
                r = portfolio_returns(Rs, w, rebalance)
                row = {"subset": " + ".join(sub), "n": size, "method": method, **{c: w.get(c, 0.0) for c in cols}}
                row.update(stats_row(r, rf))
                if walk_forward_days:
                    if size == 1 or method in ("equal", "target"):
                        wf = r.iloc[walk_forward_days:] if len(r) > walk_forward_days + 20 else pd.Series(dtype=float)
                    else:
                        wf, _ = walk_forward(Rs, method, rf, walk_forward_days, step, rebalance, lo, hi, target,
                                             n_resamples)
                    s = stats_row(wf, rf)
                    row.update({"WF Sharpe": s.get("Sharpe", np.nan), "WF CAGR": s.get("CAGR", np.nan),
                                "WF Max drawdown": s.get("Max drawdown", np.nan)})
                rows.append(row)
    return pd.DataFrame(rows)


def marginal(R: pd.DataFrame, rf: pd.Series | None = None, method: str = "target", rebalance: str = "M",
             lo: float = 0.0, hi: float = 1.0, target: pd.Series | None = None, test: bool = True) -> pd.DataFrame:
    """Add/remove analysis for every strategy against the portfolio built from all the others."""
    cols = list(R.columns)
    w_all = optimize(R, method, rf, lo, hi, target)
    r_all = portfolio_returns(R, w_all, rebalance)
    s_all = stats_row(r_all, rf)
    rows = []
    for k in cols:
        rest = [c for c in cols if c != k]
        sk = stats_row(R[k], rf)
        row = {"strategy": k, "Sharpe alone": sk.get("Sharpe"), "CAGR alone": sk.get("CAGR"),
               "Max DD alone": sk.get("Max drawdown")}
        if rest:
            w_rest = optimize(R[rest], method, rf, lo, hi, target)
            r_rest = portfolio_returns(R[rest], w_rest, rebalance)
            s_rest = stats_row(r_rest, rf)
            rho = R[k].corr(r_rest)
            hurdle = rho * s_rest.get("Sharpe", np.nan)
            row.update({"Corr to rest": rho, "Hurdle Sharpe": hurdle,
                        "Passes hurdle": bool(sk.get("Sharpe", -np.inf) > hurdle),
                        "Sharpe without": s_rest.get("Sharpe"), "Sharpe with": s_all.get("Sharpe"),
                        "Δ Sharpe": s_all.get("Sharpe", np.nan) - s_rest.get("Sharpe", np.nan),
                        "Δ CAGR": s_all.get("CAGR", np.nan) - s_rest.get("CAGR", np.nan),
                        "Δ Max DD": s_all.get("Max drawdown", np.nan) - s_rest.get("Max drawdown", np.nan),
                        "Weight": w_all[k]})
            if test:
                t = sharpe_diff_test(r_all, r_rest, rf)
                row["P(no improvement)"] = t.get("p_not_better", np.nan)
            d = row["Δ Sharpe"]
            p = row.get("P(no improvement)", np.nan)
            if d > 0 and row["Passes hurdle"]:
                row["Verdict"] = "Keep: improves Sharpe" + (" (significant)" if p < 0.1 else " (not yet significant)")
            elif d > 0:
                row["Verdict"] = "Keep: small improvement"
            else:
                row["Verdict"] = "Removal candidate" + (" (significant)" if p > 0.9 else " (not significant)")
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")


def risk_contributions(R: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    """Share of portfolio volatility from each strategy (Euler decomposition, Ledoit-Wolf covariance)."""
    w = pd.Series(w, dtype=float).reindex(R.columns).fillna(0.0)
    cov = ledoit_wolf(R.to_numpy(float)) * TD
    wv = w.to_numpy()
    port_vol = np.sqrt(wv @ cov @ wv)
    mrc = cov @ wv / port_vol if port_vol > 0 else np.zeros(len(wv))
    rc = wv * mrc
    return pd.DataFrame({"Weight": wv, "Volatility": np.sqrt(np.diag(cov)), "Marginal risk": mrc,
                         "Risk contribution": rc, "% of risk": rc / port_vol if port_vol > 0 else np.nan},
                        index=R.columns)


def drawdown_contributions(R: pd.DataFrame, w: pd.Series, rebalance: str = "M") -> pd.DataFrame:
    """Each strategy's share of the loss in the portfolio's worst drawdown (peak to trough)."""
    r, c = portfolio_returns(R, w, rebalance, contributions=True)
    t = M.drawdown_table(r, 1)
    if t.empty:
        return pd.DataFrame()
    pk, tr = t.loc[0, "peak"], t.loc[0, "trough"]
    seg = c.loc[pk:tr].iloc[1:] if pk != tr else c.loc[pk:tr]
    loss = seg.sum()
    return pd.DataFrame({"Contribution": loss, "% of drawdown": loss / loss.sum() if loss.sum() else np.nan}) \
        .assign(peak=pk, trough=tr)


def efficient_frontier(R: pd.DataFrame, rf: pd.Series | None = None, lo: float = 0.0, hi: float = 1.0,
                       points: int = 30, n_random: int = 1500, seed: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frontier (min variance for each target return) plus a cloud of random long-only portfolios."""
    mu, cov = estimate(R, rf)
    n = len(mu)
    cols = list(R.columns)
    w_mv = optimize(R, "min_variance", rf, lo, hi).to_numpy()
    targets = np.linspace(w_mv @ mu, mu.max(), points)
    front = []
    for t in targets:
        res = minimize(lambda v: v @ cov @ v, w_mv, method="SLSQP", bounds=_bounds(n, lo, hi),
                       constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1},
                                    {"type": "eq", "fun": lambda v, t=t: v @ mu - t}])
        if res.success:
            v = res.x
            front.append({"Excess return": v @ mu, "Volatility": np.sqrt(v @ cov @ v),
                          **dict(zip(cols, v))})
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(n), n_random)
    cloud = pd.DataFrame({"Excess return": W @ mu, "Volatility": np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))})
    cloud["Sharpe"] = cloud["Excess return"] / cloud["Volatility"]
    fr = pd.DataFrame(front)
    if len(fr):
        fr["Sharpe"] = fr["Excess return"] / fr["Volatility"]
    return fr, cloud


def correlations(R: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if freq == "D":
        return R.corr()
    g = (1 + R).groupby(R.index.to_period(freq)).prod() - 1
    return g.corr()


def rolling_correlations(R: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    out = {}
    cols = list(R.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            out[f"{a} / {b}"] = R[a].rolling(window).corr(R[b])
    return pd.DataFrame(out).dropna(how="all")


def stress_correlation(R: pd.DataFrame, bench: pd.Series, q: float = 0.1) -> pd.DataFrame:
    """Correlation between strategies on the benchmark's worst ``q`` share of days (tail dependence)."""
    b = pd.Series(bench, dtype=float).reindex(R.index)
    bad = b <= b.quantile(q)
    return R[bad.to_numpy()].corr() if bad.sum() > 10 else pd.DataFrame()


STRESS_WINDOWS = {
    "GFC (Oct-07 to Mar-09)": ("2007-10-09", "2009-03-09"),
    "Euro crisis / US downgrade (2011)": ("2011-07-22", "2011-10-03"),
    "China deval. / oil (2015-16)": ("2015-08-10", "2016-02-11"),
    "Q4 2018 sell-off": ("2018-09-20", "2018-12-24"),
    "COVID crash (2020)": ("2020-02-19", "2020-03-23"),
    "COVID rebound (2020)": ("2020-03-24", "2020-08-31"),
    "2022 bear market": ("2022-01-03", "2022-10-12"),
    "2023 rally": ("2023-01-03", "2023-07-31"),
    "2025 tariff shock": ("2025-02-19", "2025-04-08"),
}


def stress_test(series: dict[str, pd.Series], windows: dict = STRESS_WINDOWS) -> pd.DataFrame:
    """Total return of each series in historical crisis / rebound windows (blank where no data)."""
    out = {}
    for name, (a, b) in windows.items():
        row = {}
        for k, r in series.items():
            r = M._clean(r)
            if len(r) and r.index[0] <= pd.Timestamp(a) + pd.Timedelta(days=7) and r.index[-1] >= pd.Timestamp(b):
                row[k] = (1 + r.loc[a:b]).prod() - 1
        out[name] = row
    return pd.DataFrame(out).T


def rebalance_orders(nav: pd.Series, target: pd.Series, labels: dict | None = None) -> pd.DataFrame:
    """Dollar transfers that bring each sleeve to its target weight of today's total NAV."""
    nav = nav.reindex(target.index).fillna(0.0)
    total = nav.sum()
    tgt_val = target * total
    df = pd.DataFrame({"Current $": nav, "Current weight": nav / total if total else np.nan,
                       "Target weight": target, "Target $": tgt_val, "Transfer $": tgt_val - nav})
    if labels:
        df.index = [labels.get(i, i) for i in df.index]
    return df
