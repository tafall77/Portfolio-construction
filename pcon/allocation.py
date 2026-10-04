"""Portfolio construction across strategies.

Answers three questions with the strategies' daily return series (backtest, optionally extended by live):

1. **Which strategies belong in the book?** Every subset is tested. For each strategy the add/remove
   analysis reports the Sharpe with and without it, and the textbook hurdle: adding a little of strategy
   *k* to portfolio *P* raises *P*'s Sharpe iff ``SR_k > corr(k, P) x SR_P``.
2. **How much capital each?**
   * Mean-variance (Markowitz): maximum Sharpe, minimum variance, and a resampled maximum Sharpe (Michaud).
   * Risk-based: inverse volatility, equal risk contribution and your own risk budgets (Roncalli), and
     maximum diversification (Choueifaty).
   * Drawdown-based: minimum conditional drawdown at risk (CDaR, Chekhlov-Uryasev-Zabarankin).
   * Portfolio level: volatility targeting with the rest in T-bills.
   Every optimiser is long-only with min / max weight bounds and uses a Ledoit-Wolf covariance.
3. **Is the answer robust?** In-sample optimisation flatters every optimiser. The walk-forward test
   estimates weights on a trailing window only, holds them for the next quarter and stitches those
   out-of-sample quarters together. Bootstrap error bars show how unstable each method's weights are,
   and Sharpe differences come with a paired block-bootstrap p-value.

The diversification diagnostics (beta to the benchmark, share of variance that is just market beta,
diversification ratio, effective number of bets) show how many independent bets the book really holds.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq, linprog, minimize

from . import metrics as M

TD = M.TD
METHODS = {
    "target": "Your target weights",
    "equal": "Equal weight",
    "inverse_vol": "Inverse volatility",
    "risk_parity": "Equal risk contribution",
    "risk_budget": "Your risk budgets",
    "min_variance": "Minimum variance",
    "max_diversification": "Maximum diversification",
    "min_cdar": "Minimum drawdown (CDaR 95%)",
    "max_sharpe": "Maximum Sharpe",
    "resampled": "Resampled max Sharpe",
}
FREQ = {"D": None, "W": "W", "M": "M", "Q": "Q", "A": "Y", "Y": "Y", "none": "none"}


class OptimizerWarning(UserWarning):
    """A solver could not reach a verified optimum; a feasible fallback was used."""


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
                      contributions: bool = False, cash: pd.Series | None = None):
    """Daily returns of a fixed-weight mix, rebalanced at the start of each period (weights drift inside).

    Any weight not allocated (sum < 1) sits in cash, earning ``cash`` (daily T-bill return) or 0 %. With
    ``contributions=True`` also returns each strategy's daily contribution (drifted weight x return).
    """
    w = pd.Series(w, dtype=float).reindex(R.columns).fillna(0.0)
    cash_w = 1.0 - w.sum()
    c = (cash.reindex(R.index).fillna(0.0) if cash is not None else pd.Series(0.0, index=R.index)).to_numpy()
    if FREQ.get(rebalance, rebalance) is None:
        contrib = R * w
        r = contrib.sum(axis=1) + cash_w * c
        return (r, contrib) if contributions else r
    key = _period_key(R.index, rebalance)
    H = ((1 + R).groupby(key).cumprod() * w.to_numpy()).to_numpy()   # sleeve values vs NAV at last rebalance
    C = pd.Series(1 + c).groupby(key).cumprod().to_numpy() * cash_w  # cash sleeve value
    prev = np.vstack([w.to_numpy()[None, :], H[:-1]])
    prev_c = np.r_[cash_w, C[:-1]]
    first = np.r_[True, key[1:] != key[:-1]]
    prev[first] = w.to_numpy()                                         # weights right after rebalancing
    prev_c[first] = cash_w
    prev_v = prev.sum(axis=1) + prev_c
    r = pd.Series((H.sum(axis=1) + C) / prev_v - 1, index=R.index)
    if contributions:
        contrib = pd.DataFrame(prev / prev_v[:, None], index=R.index, columns=R.columns) * R
        return r, contrib
    return r


# ------------------------------------------------------------------------------------------------
# estimators
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


# ------------------------------------------------------------------------------------------------
# bounds and generic solver
# ------------------------------------------------------------------------------------------------
def feasible_bounds(n: int, lo: float, hi: float) -> tuple[float, float, bool]:
    """Weight bounds actually usable for ``n`` assets fully invested.

    A cap below 1/n (or a floor above 1/n) has no solution; it is relaxed to 1/n and ``relaxed`` is True so
    callers can say so instead of silently reporting the requested cap.
    """
    lo2, hi2 = min(lo, 1.0 / n), max(hi, 1.0 / n)
    return lo2, hi2, bool(lo2 < lo - 1e-12 or hi2 > hi + 1e-12)


def _bounds(n: int, lo: float, hi: float):
    lo, hi, _ = feasible_bounds(n, lo, hi)
    return [(lo, hi)] * n


def _feasible(w: np.ndarray, lo: float, hi: float, tol: float = 1e-7) -> bool:
    return bool(np.all(np.isfinite(w)) and abs(w.sum() - 1) < 1e-6 and (w >= lo - tol).all()
                and (w <= hi + tol).all())


def _capped(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Weights proportional to ``scores`` inside [lo, hi], summing to 1 (iterative capping / water-filling).

    Weights that would break a bound are fixed at it and the rest is redistributed in proportion to the
    scores of the free assets, so free weights keep their relative sizes.
    """
    s = np.maximum(np.asarray(scores, dtype=float), 1e-300)
    n = len(s)
    lo, hi, _ = feasible_bounds(n, lo, hi)
    w = np.zeros(n)
    fixed = np.zeros(n, dtype=bool)
    for _ in range(2 * n + 2):
        rem = 1.0 - w[fixed].sum()
        free = ~fixed
        if not free.any():
            break
        w[free] = s[free] / s[free].sum() * rem
        over = free & (w > hi + 1e-12)
        under = free & (w < lo - 1e-12)
        if over.any():                                  # cap the largest first, then re-spread
            w[over] = hi
            fixed |= over
        elif under.any():
            w[under] = lo
            fixed |= under
        else:
            break
    return w / w.sum()


def _solve(obj, n: int, lo: float, hi: float, x0=None, jac=None) -> np.ndarray:
    """SLSQP over the capped simplex from several starts; keep the best *feasible* answer.

    A solver that stops with ``success=False`` often sits at the optimum anyway (common when two bounds
    bind), so any feasible end point is accepted and compared on the objective. Equal weight is never
    returned silently: if no start ends feasible an ``OptimizerWarning`` is raised and the best start is
    projected onto the bounds.
    """
    lo_, hi_, _ = feasible_bounds(n, lo, hi)
    starts = [np.full(n, 1.0 / n) if x0 is None else _capped(np.clip(x0, 1e-12, None), lo_, hi_)]
    for i in range(n):                                    # tilted starts towards each asset
        e = np.ones(n)
        e[i] = 4.0 * n
        starts.append(_capped(e, lo_, hi_))
    best, best_f = None, np.inf
    for s in starts:
        res = minimize(obj, s, jac=jac, method="SLSQP", bounds=_bounds(n, lo, hi),
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                     "jac": lambda w: np.ones_like(w)}],
                       options={"maxiter": 1000, "ftol": 1e-14})
        x = _capped(np.clip(res.x, 0, None), lo_, hi_)
        if _feasible(x, lo_, hi_) and obj(x) < best_f - 1e-15:
            best, best_f = x, obj(x)
    if best is None:
        warnings.warn("optimizer did not reach a feasible point; using the best start", OptimizerWarning)
        best = min(starts, key=obj)
    return best


# ------------------------------------------------------------------------------------------------
# optimisers
# ------------------------------------------------------------------------------------------------
def _ratio(a: np.ndarray, cov: np.ndarray):
    """Objective and gradient of -(w.a) / sqrt(w'Cw): max Sharpe (a = mu) or max diversification (a = vol)."""
    def f(w):
        return -(w @ a) / np.sqrt(w @ cov @ w)

    def g(w):
        s2 = w @ cov @ w
        s = np.sqrt(s2)
        return -(a * s - (w @ a) * (cov @ w) / s) / s2
    return f, g


def _min_variance(cov: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return _solve(lambda v: v @ cov @ v, len(cov), lo, hi, jac=lambda v: 2 * cov @ v)


def _max_sharpe(mu: np.ndarray, cov: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Maximum Sharpe via its convex reformulation (Cornuejols & Tutuncu): min y'Cy s.t. mu'y = 1,
    lo*sum(y) <= y_i <= hi*sum(y), y >= 0; then w = y / sum(y). Globally optimal whenever some feasible
    portfolio has a positive expected excess return; otherwise the minimum-variance portfolio is used."""
    n = len(mu)
    lo, hi, _ = feasible_bounds(n, lo, hi)
    lp = linprog(-mu, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(lo, hi)] * n, method="highs")
    if not lp.success or -lp.fun <= 1e-12:
        return _min_variance(cov, lo, hi)
    w0 = 0.5 * lp.x + 0.5 * _capped(np.ones(n), lo, hi)
    if mu @ w0 <= 1e-12:
        w0 = lp.x
    A = np.vstack([np.eye(n) - lo * np.ones((n, n)), hi * np.ones((n, n)) - np.eye(n)])
    res = minimize(lambda y: y @ cov @ y, w0 / (mu @ w0), jac=lambda y: 2 * cov @ y, method="SLSQP",
                   bounds=[(0, None)] * n,
                   constraints=[{"type": "eq", "fun": lambda y: mu @ y - 1.0, "jac": lambda y: mu},
                                {"type": "ineq", "fun": lambda y: A @ y, "jac": lambda y: A}],
                   options={"maxiter": 1000, "ftol": 1e-15})
    y = np.clip(res.x, 0, None)
    if y.sum() > 0:
        w = y / y.sum()
        if _feasible(w, lo, hi, 1e-6):
            return _capped(w, lo, hi)
    f, g = _ratio(mu, cov)                                # fallback: direct ratio, multi-start
    return _solve(f, n, lo, hi, jac=g)


def _risk_budget(cov: np.ndarray, budgets: np.ndarray | None, lo: float, hi: float) -> np.ndarray:
    """Risk budgeting (risk parity when budgets are equal) with weight bounds.

    Spinu (2013) / Richard & Roncalli (2019): for c > 0 the convex problem
    ``min 0.5 w'Cw - c sum b_i log w_i`` over lo <= w <= hi has a unique solution whose risk contributions
    are proportional to b among the assets not at a bound; c is root-found so that the weights sum to 1.
    """
    n = len(cov)
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float)
    b = np.maximum(b, 1e-12) / np.maximum(b, 1e-12).sum()
    lo, hi, _ = feasible_bounds(n, lo, hi)
    if n * hi - 1 < 1e-9 or 1 - n * lo < 1e-9:
        return np.full(n, 1.0 / n)
    opts = {"ftol": 1e-15, "gtol": 1e-13, "maxiter": 5000}

    def solve(c, bounds, x0):
        res = minimize(lambda y: 0.5 * y @ cov @ y - c * (b * np.log(y)).sum(), x0,
                       jac=lambda y: cov @ y - c * b / y, method="L-BFGS-B", bounds=bounds, options=opts)
        return res.x

    # unconstrained solution scales with sqrt(c): solve once, normalise, and keep it if the bounds hold
    y = solve(1.0, [(1e-12, None)] * n, np.full(n, 1.0 / np.sqrt(np.diag(cov)).mean()))
    w = y / y.sum()
    if (w >= lo - 1e-9).all() and (w <= hi + 1e-9).all():
        return w
    bounds = [(max(lo, 1e-12), hi)] * n
    state = {"x": np.clip(w, max(lo, 1e-12), hi)}

    def excess(lc):
        x = solve(np.exp(lc), bounds, state["x"])
        state["x"] = x
        return x.sum() - 1.0

    base = np.log(max(np.trace(cov) / n, 1e-12))
    a, z = base - 15, base + 15
    for _ in range(20):
        if excess(a) < 0:
            break
        a -= 10
    for _ in range(20):
        if excess(z) > 0:
            break
        z += 10
    lc = brentq(excess, a, z, xtol=1e-13, rtol=1e-13, maxiter=300)
    return _capped(solve(np.exp(lc), bounds, state["x"]), lo, hi)


def _min_cdar(X: np.ndarray, lo: float, hi: float, alpha: float = 0.95) -> np.ndarray:
    """Minimum Conditional Drawdown-at-Risk (Chekhlov, Uryasev & Zabarankin 2005) as a linear programme.

    Drawdowns are measured on uncompounded cumulative returns (the standard LP form). Variables: weights w,
    running peaks u_t, excess drawdowns z_t and the threshold zeta.
    """
    T, n = X.shape
    lo, hi, _ = feasible_bounds(n, lo, hi)
    Cum = np.cumsum(X, axis=0)
    I = sparse.identity(T, format="csr")
    Z = sparse.csr_matrix((T, T))
    # u_t >= cum_t . w          ->   cum_t . w - u_t <= 0
    b1 = sparse.hstack([sparse.csr_matrix(Cum), -I, Z, sparse.csr_matrix((T, 1))])
    # u_t >= u_{t-1}            ->   u_{t-1} - u_t <= 0
    D = sparse.diags([np.ones(T - 1), -np.ones(T - 1)], [0, 1], shape=(T - 1, T))
    b2 = sparse.hstack([sparse.csr_matrix((T - 1, n)), D, sparse.csr_matrix((T - 1, T)),
                        sparse.csr_matrix((T - 1, 1))])
    # z_t >= u_t - cum_t . w - zeta
    b3 = sparse.hstack([sparse.csr_matrix(-Cum), I, -I, -np.ones((T, 1))])
    A_ub = sparse.vstack([b1, b2, b3]).tocsr()
    c = np.r_[np.zeros(n), np.zeros(T), np.full(T, 1.0 / ((1 - alpha) * T)), 1.0]
    A_eq = sparse.hstack([sparse.csr_matrix(np.ones((1, n))), sparse.csr_matrix((1, 2 * T + 1))])
    bounds = [(lo, hi)] * n + [(0, None)] * T + [(0, None)] * T + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=[1.0], bounds=bounds,
                  method="highs")
    if not res.success:
        warnings.warn(f"min-CDaR LP failed ({res.message}); using minimum variance", OptimizerWarning)
        return _min_variance(np.cov(X.T) * TD, lo, hi)
    return _capped(np.clip(res.x[:n], 0, None), lo, hi)


def optimize(R: pd.DataFrame, method: str, rf: pd.Series | None = None, lo: float = 0.0, hi: float = 1.0,
             target: pd.Series | None = None, n_resamples: int = 100, seed: int = 11,
             budgets: pd.Series | None = None) -> pd.Series:
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
        w = W.mean().reindex(cols)
        return w / w.sum()
    if method == "min_cdar":
        return pd.Series(_min_cdar(R.to_numpy(float), lo, hi), index=cols)
    mu, cov = estimate(R, rf)
    vol = np.sqrt(np.diag(cov))
    if method == "inverse_vol":
        w = _capped(1 / vol, lo, hi)
    elif method == "risk_parity":
        w = _risk_budget(cov, None, lo, hi)
    elif method == "risk_budget":
        b = budgets.reindex(cols).fillna(0.0).to_numpy() if budgets is not None else None
        w = _risk_budget(cov, b if b is not None and b.sum() > 0 else None, lo, hi)
    elif method == "min_variance":
        w = _min_variance(cov, lo, hi)
    elif method == "max_diversification":
        f, g = _ratio(vol, cov)
        w = _solve(f, n, lo, hi, jac=g)
    elif method == "max_sharpe":
        w = _max_sharpe(mu, cov, lo, hi)
    else:
        raise ValueError(f"unknown method {method}")
    return pd.Series(w, index=cols)


def resampled_weights(R: pd.DataFrame, rf: pd.Series | None = None, lo: float = 0.0, hi: float = 1.0,
                      n: int = 200, block: int = 63, seed: int = 11, method: str = "max_sharpe",
                      budgets: pd.Series | None = None) -> pd.DataFrame:
    """Optimal weights on ``n`` block-bootstrap resamples (max Sharpe = Michaud-style resampled efficiency)."""
    from .expectations import block_indices
    T = len(R)
    idx = block_indices(T, T, n, block, seed)
    X = R.to_numpy(float)
    f = rf.reindex(R.index).fillna(0.0).to_numpy() if rf is not None else np.zeros(T)
    out = []
    for k in range(n):
        Rk = pd.DataFrame(X[idx[k]], columns=R.columns)
        fk = pd.Series(f[idx[k]])
        out.append(optimize(Rk, method, fk, lo, hi, budgets=budgets).to_numpy())
    return pd.DataFrame(out, columns=R.columns)


def weight_uncertainty(R: pd.DataFrame, methods=("risk_parity", "min_variance", "max_sharpe"),
                       rf: pd.Series | None = None, lo: float = 0.0, hi: float = 1.0, n: int = 100,
                       budgets: pd.Series | None = None, quantiles=(0.05, 0.5, 0.95)) -> pd.DataFrame:
    """Bootstrap quantiles of each method's weights: how much would they move on a different history?"""
    rows = []
    for m in methods:
        W = resampled_weights(R, rf, lo, hi, n, method=m, budgets=budgets)
        for c in R.columns:
            q = W[c].quantile(list(quantiles))
            rows.append({"method": m, "strategy": c, **{f"p{int(x * 100)}": v for x, v in zip(quantiles, q)}})
    return pd.DataFrame(rows)


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
                 target: pd.Series | None = None, n_resamples: int = 50,
                 budgets: pd.Series | None = None) -> tuple[pd.Series, pd.DataFrame]:
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
        w = optimize(est, method, rf, lo, hi, target, n_resamples, budgets=budgets)
        weights[block.index[0]] = w
        rets.append(portfolio_returns(block, w, rebalance))
    return pd.concat(rets), pd.DataFrame(weights).T


def weight_turnover(W: pd.DataFrame) -> float:
    """Average one-way turnover per re-estimation of a walk-forward weight history."""
    if W is None or len(W) < 2:
        return np.nan
    return float((W.diff().abs().sum(axis=1) / 2).iloc[1:].mean())


def sharpe_diff_test(a: pd.Series, b: pd.Series, rf: pd.Series | None = None, n: int = 1000,
                     block: int = 63, seed: int = 3) -> dict:
    """Paired block bootstrap of SR(a) - SR(b): observed difference, 90 % interval, P(diff <= 0).

    Identical series (e.g. a strategy at zero weight) carry no information: the p-value is NaN.
    """
    from .expectations import block_indices
    x = pd.concat([M._clean(a), M._clean(b)], axis=1, join="inner").dropna()
    if len(x) < 60:
        return {}
    f = rf.reindex(x.index).fillna(0.0).to_numpy()[:, None] if rf is not None else 0.0
    X = x.to_numpy() - f
    if np.allclose(X[:, 0], X[:, 1], atol=1e-14):
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p_not_better": np.nan}
    idx = block_indices(len(X), len(X), n, block, seed)
    S = X[idx]
    sr = S.mean(axis=1) / S.std(axis=1, ddof=1) * np.sqrt(TD)
    d = sr[:, 0] - sr[:, 1]
    obs = X.mean(0) / X.std(0, ddof=1) * np.sqrt(TD)
    return {"diff": float(obs[0] - obs[1]), "lo": float(np.quantile(d, 0.05)), "hi": float(np.quantile(d, 0.95)),
            "p_not_better": float((d < 0).mean() + 0.5 * (d == 0).mean())}


def explore(R: pd.DataFrame, rf: pd.Series | None = None, methods=("equal", "risk_parity", "max_sharpe"),
            rebalance: str = "M", lo: float = 0.0, hi: float = 1.0, target: pd.Series | None = None,
            walk_forward_days: int | None = 756, step: str = "Q", n_resamples: int = 50,
            budgets: pd.Series | None = None) -> pd.DataFrame:
    """Every non-empty subset of strategies x every method: in-sample and walk-forward statistics.

    ``Max weight used`` shows the cap actually applied: a cap below 1/n cannot hold for a small subset.
    """
    rows = []
    cols = list(R.columns)
    for size in range(1, len(cols) + 1):
        for sub in itertools.combinations(cols, size):
            Rs = R[list(sub)]
            hi_used = feasible_bounds(size, lo, hi)[1]
            for method in (methods if size > 1 else ("single",)):
                if method == "target" and target is not None and target.reindex(list(sub)).fillna(0).sum() <= 0:
                    continue
                w = optimize(Rs, method if size > 1 else "equal", rf, lo, hi, target, n_resamples, budgets=budgets)
                r = portfolio_returns(Rs, w, rebalance)
                row = {"subset": " + ".join(sub), "n": size, "method": method, **{c: w.get(c, 0.0) for c in cols},
                       "Max weight used": hi_used if size > 1 else 1.0}
                row.update(stats_row(r, rf))
                if walk_forward_days:
                    if size == 1 or method in ("equal", "target"):
                        wf = r.iloc[walk_forward_days:] if len(r) > walk_forward_days + 20 else pd.Series(dtype=float)
                    else:
                        wf, _ = walk_forward(Rs, method, rf, walk_forward_days, step, rebalance, lo, hi, target,
                                             n_resamples, budgets)
                    s = stats_row(wf, rf)
                    row.update({"WF Sharpe": s.get("Sharpe", np.nan), "WF CAGR": s.get("CAGR", np.nan),
                                "WF Max drawdown": s.get("Max drawdown", np.nan)})
                rows.append(row)
    return pd.DataFrame(rows)


def marginal(R: pd.DataFrame, rf: pd.Series | None = None, method: str = "target", rebalance: str = "M",
             lo: float = 0.0, hi: float = 1.0, target: pd.Series | None = None, test: bool = True,
             budgets: pd.Series | None = None) -> pd.DataFrame:
    """Add/remove analysis for every strategy against the portfolio built from all the others.

    Verdicts:
    * held and Δ Sharpe > 0: keep (significant or not yet);
    * held, Δ Sharpe <= 0 but it passes the hurdle: keep at a smaller weight (it is over-weighted);
    * held, Δ Sharpe <= 0 and fails the hurdle: removal candidate;
    * zero weight: not held; the hurdle says whether adding a little would help.
    """
    cols = list(R.columns)
    w_all = optimize(R, method, rf, lo, hi, target, budgets=budgets)
    r_all = portfolio_returns(R, w_all, rebalance)
    s_all = stats_row(r_all, rf)
    rows = []
    for k in cols:
        rest = [c for c in cols if c != k]
        sk = stats_row(R[k], rf)
        row = {"strategy": k, "Sharpe alone": sk.get("Sharpe"), "CAGR alone": sk.get("CAGR"),
               "Max DD alone": sk.get("Max drawdown")}
        if rest:
            w_rest = optimize(R[rest], method, rf, lo, hi, target, budgets=budgets)
            r_rest = portfolio_returns(R[rest], w_rest, rebalance)
            s_rest = stats_row(r_rest, rf)
            ex_k = R[k] - (rf.reindex(R.index).fillna(0.0) if rf is not None else 0.0)
            ex_rest = r_rest - (rf.reindex(R.index).fillna(0.0) if rf is not None else 0.0)
            rho = ex_k.corr(ex_rest)
            hurdle = rho * s_rest.get("Sharpe", np.nan)
            passes = bool(sk.get("Sharpe", -np.inf) > hurdle)
            held = w_all[k] > 1e-6
            row.update({"Corr to rest": rho, "Hurdle Sharpe": hurdle, "Passes hurdle": passes,
                        "Sharpe without": s_rest.get("Sharpe"), "Sharpe with": s_all.get("Sharpe"),
                        "Δ Sharpe": s_all.get("Sharpe", np.nan) - s_rest.get("Sharpe", np.nan) if held else np.nan,
                        "Δ CAGR": s_all.get("CAGR", np.nan) - s_rest.get("CAGR", np.nan) if held else np.nan,
                        "Δ Max DD": (s_all.get("Max drawdown", np.nan) - s_rest.get("Max drawdown", np.nan)
                                     if held else np.nan),
                        "Weight": w_all[k]})
            p = np.nan
            if test and held:
                p = sharpe_diff_test(r_all, r_rest, rf).get("p_not_better", np.nan)
            row["P(no improvement)"] = p
            d = row["Δ Sharpe"]
            if not held:
                row["Verdict"] = ("Not held: adding a little would raise Sharpe (passes hurdle)" if passes
                                  else "Not held: adding it would not help (fails hurdle)")
            elif d > 0:
                row["Verdict"] = ("Keep: improves Sharpe (passes hurdle)" if passes else "Keep: small improvement") + \
                    (" (significant)" if p < 0.1 else " (not yet significant)")
            elif passes:
                row["Verdict"] = "Keep, at a smaller weight (over-weighted; passes hurdle)"
            else:
                row["Verdict"] = "Removal candidate" + (" (significant)" if p > 0.9 else " (not significant)")
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")


# ------------------------------------------------------------------------------------------------
# risk and diversification
# ------------------------------------------------------------------------------------------------
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


def diversification(R: pd.DataFrame, w: pd.Series, bench: pd.Series | None = None) -> dict:
    """How many independent bets does the mix really hold?

    * Diversification ratio (Choueifaty): weighted average vol / portfolio vol (1 = no diversification).
    * Effective number of bets (Meucci 2009): exp(entropy) of the portfolio's variance across principal
      components; between 1 and n.
    * Effective number of risk contributors: 1 / sum(RC share^2).
    * Beta to the benchmark and the share of portfolio variance that is just benchmark beta.
    """
    w = pd.Series(w, dtype=float).reindex(R.columns).fillna(0.0).to_numpy()
    cov = ledoit_wolf(R.to_numpy(float)) * TD
    var = w @ cov @ w
    vol = np.sqrt(np.diag(cov))
    out = {"Diversification ratio": float(w @ vol / np.sqrt(var)) if var > 0 else np.nan}
    lam, E = np.linalg.eigh(cov)
    v = E.T @ w
    p = np.clip(v ** 2 * lam / var, 1e-300, None) if var > 0 else np.ones(len(w)) / len(w)
    p = p / p.sum()
    out["Effective number of bets"] = float(np.exp(-(p * np.log(p)).sum()))
    rc = w * (cov @ w) / var if var > 0 else np.full(len(w), np.nan)
    out["Effective risk contributors"] = float(1 / (rc ** 2).sum()) if var > 0 else np.nan
    if bench is not None and len(pd.Series(bench).dropna()):
        b = pd.Series(bench, dtype=float).reindex(R.index)
        ok = b.notna().to_numpy()
        X, bb = R.to_numpy(float)[ok], b.to_numpy()[ok]
        if len(bb) > 60 and bb.var() > 0:
            betas = np.array([np.cov(X[:, i], bb)[0, 1] / bb.var(ddof=1) for i in range(X.shape[1])])
            port = X @ w
            beta_p = float(w @ betas)
            out["Beta to benchmark"] = beta_p
            out["Share of variance from benchmark beta"] = float(min(1.0, beta_p ** 2 * bb.var(ddof=1)
                                                                     / port.var(ddof=1)))
            out["Strategy betas"] = dict(zip(R.columns, betas))
            out["Strategy R2 vs benchmark"] = {c: float(np.corrcoef(X[:, i], bb)[0, 1] ** 2)
                                               for i, c in enumerate(R.columns)}
    return out


def drawdown_contributions(R: pd.DataFrame, w: pd.Series, rebalance: str = "M") -> pd.DataFrame:
    """Each strategy's share of the loss in the portfolio's worst drawdown (peak to trough).

    Daily contributions are scaled by NAV relative to the peak, so they add up exactly to the depth.
    """
    r, c = portfolio_returns(R, w, rebalance, contributions=True)
    t = M.drawdown_table(r, 1)
    if t.empty:
        return pd.DataFrame()
    pk, tr = t.loc[0, "peak"], t.loc[0, "trough"]
    nav = (1 + r).cumprod()
    nav_prev = nav.shift(1).fillna(1.0)
    dd0 = nav.loc[pk] / max(nav.loc[:pk].max(), 1.0) - 1
    start_after_peak = dd0 >= -1e-12                     # a genuine pre-drawdown peak
    seg = c.loc[pk:tr].iloc[1:] if start_after_peak else c.loc[pk:tr]
    peak_nav = nav.loc[pk] if start_after_peak else 1.0   # otherwise the drawdown starts on day one
    scaled = seg.mul(nav_prev.loc[seg.index] / peak_nav, axis=0)
    loss = scaled.sum()
    return pd.DataFrame({"Contribution": loss, "% of drawdown": loss / loss.sum() if loss.sum() else np.nan}) \
        .assign(peak=pk, trough=tr)


def efficient_frontier(R: pd.DataFrame, rf: pd.Series | None = None, lo: float = 0.0, hi: float = 1.0,
                       points: int = 30, n_random: int = 1500, seed: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frontier (min variance for each target return) and a cloud of random mixes, both inside the bounds."""
    mu, cov = estimate(R, rf)
    n = len(mu)
    lo_, hi_, _ = feasible_bounds(n, lo, hi)
    cols = list(R.columns)
    w_mv = optimize(R, "min_variance", rf, lo, hi).to_numpy()
    lp = linprog(-mu, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(lo_, hi_)] * n, method="highs")
    top = -lp.fun if lp.success else mu.max()
    front = []
    for t in np.linspace(w_mv @ mu, top, points):
        res = minimize(lambda v: v @ cov @ v, w_mv, jac=lambda v: 2 * cov @ v, method="SLSQP",
                       bounds=[(lo_, hi_)] * n,
                       constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1},
                                    {"type": "eq", "fun": lambda v, t=t: v @ mu - t}],
                       options={"maxiter": 500, "ftol": 1e-14})
        v = res.x if res.success else (lp.x if abs(t - top) < 1e-12 and lp.success else None)
        if v is not None and _feasible(v / v.sum(), lo_, hi_, 1e-6):
            front.append({"Excess return": v @ mu, "Volatility": np.sqrt(v @ cov @ v), **dict(zip(cols, v))})
    rng = np.random.default_rng(seed)
    kept, tries = [], 0
    while sum(len(k) for k in kept) < n_random and tries < 60:
        W = rng.dirichlet(np.ones(n), n_random * 4)
        kept.append(W[((W <= hi_ + 1e-12) & (W >= lo_ - 1e-12)).all(axis=1)])
        tries += 1
    W = np.vstack(kept)[:n_random] if kept else np.empty((0, n))
    if len(W) < n_random // 4:                             # tight caps: random walk inside the polytope
        W = np.array([_capped(rng.dirichlet(np.ones(n)), lo_, hi_) for _ in range(n_random)])
    cloud = pd.DataFrame({"Excess return": W @ mu, "Volatility": np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))})
    cloud["Sharpe"] = cloud["Excess return"] / cloud["Volatility"]
    fr = pd.DataFrame(front)
    if len(fr):
        fr["Sharpe"] = fr["Excess return"] / fr["Volatility"]
    return fr, cloud


def vol_target(r: pd.Series, target_vol: float, rf: pd.Series | None = None, halflife: int = 21,
               cap: float = 1.0, rebalance: str = "M") -> tuple[pd.Series, pd.Series]:
    """Scale the whole book to a target volatility, the rest in T-bills.

    The exposure for each period is ``min(cap, target / sigma_hat)`` where sigma_hat is the EWMA volatility
    (half-life ``halflife`` days) known at the previous close; it is held for the period (monthly by default).
    ``cap = 1`` never borrows. Returns the scaled daily returns and the exposure series.
    """
    r = M._clean(r)
    c = rf.reindex(r.index).fillna(0.0) if rf is not None else pd.Series(0.0, index=r.index)
    sig = np.sqrt((r ** 2).ewm(halflife=halflife, min_periods=20).mean() * TD).shift(1)
    raw = (target_vol / sig).clip(upper=cap)
    key = pd.Series(_period_key(r.index, rebalance), index=r.index)
    scale = raw.groupby(key).transform("first").ffill().fillna(min(cap, 1.0))
    return (scale * r + (1 - scale) * c).rename(r.name), scale.rename("exposure")


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
    """Correlation between strategies on the benchmark's worst ``q`` share of days."""
    b = pd.Series(bench, dtype=float).reindex(R.index)
    bad = b <= b.quantile(q)
    return R[bad.to_numpy()].corr() if bad.sum() > 10 else pd.DataFrame()


# (from close, to close): returns are taken strictly after the first date, up to and including the second
STRESS_WINDOWS = {
    "GFC (Oct-07 to Mar-09)": ("2007-10-09", "2009-03-09"),
    "Euro crisis / US downgrade (2011)": ("2011-07-22", "2011-10-03"),
    "China deval. / oil (2015-16)": ("2015-08-10", "2016-02-11"),
    "Q4 2018 sell-off": ("2018-09-20", "2018-12-24"),
    "COVID crash (2020)": ("2020-02-19", "2020-03-23"),
    "COVID rebound (2020)": ("2020-03-23", "2020-08-31"),
    "2022 bear market": ("2022-01-03", "2022-10-12"),
    "2023 rally": ("2022-12-30", "2023-07-31"),
    "2025 tariff shock": ("2025-02-19", "2025-04-08"),
}


def stress_test(series: dict[str, pd.Series], windows: dict = STRESS_WINDOWS) -> pd.DataFrame:
    """Total return of each series from the close of the first date to the close of the second
    (blank unless the series already existed at the start of the window)."""
    out = {}
    for name, (a, b) in windows.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        row = {}
        for k, r in series.items():
            r = M._clean(r)
            if len(r) and r.index[0] <= a and r.index[-1] >= b:
                row[k] = float((1 + r[(r.index > a) & (r.index <= b)]).prod() - 1)
        out[name] = row
    return pd.DataFrame(out).T.reindex(columns=[k for k in series if any(k in row for row in out.values())])


def rebalance_orders(nav: pd.Series, target: pd.Series, labels: dict | None = None,
                     band: float | None = None) -> pd.DataFrame:
    """Dollar transfers that bring each sleeve to its target weight of today's total NAV.

    With ``band`` (tolerance-band rebalancing) nothing is traded while every sleeve is within ``band`` of its
    target; once one breaches, all sleeves go back to target.
    """
    nav = nav.reindex(target.index).fillna(0.0)
    total = nav.sum()
    tgt_val = target * total
    cur_w = nav / total if total else nav * np.nan
    df = pd.DataFrame({"Current $": nav, "Current weight": cur_w, "Target weight": target, "Target $": tgt_val,
                       "Transfer $": tgt_val - nav})
    if band is not None and total and (cur_w - target).abs().max() <= band:
        df["Transfer $"] = 0.0
    if labels:
        df.index = [labels.get(i, i) for i in df.index]
    return df
