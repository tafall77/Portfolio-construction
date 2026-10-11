"""Combinatorial purged cross-validation (CPCV) of the allocation methods, using skfolio.

Walk-forward tests each method on ONE history: the years after its first training window. CPCV (Lopez de Prado,
*Advances in Financial Machine Learning*, ch. 12) splits the common window into ``n_folds`` blocks, trains on
every combination of ``n_folds - n_test_folds`` blocks and tests on the rest, with a purge and an embargo around
each test block so no information leaks across the boundary. The test blocks recombine into several complete
out-of-sample histories (9 with the defaults), each spanning the whole window. That gives, per method:

* the spread of out-of-sample Sharpe ratios across histories: how much the result depends on which years the
  weights were estimated from;
* on how many histories it beats your target weights on the same days (a paired comparison).

Weights are held constant inside each test block (daily rebalanced), so the numbers compare methods with each
other rather than reproduce the monthly-rebalanced backtest exactly. Requires ``pip install skfolio``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import allocation as A


def available() -> bool:
    return A.HAS_SKFOLIO


def _estimator(method: str, lo: float, hi: float, target: pd.Series | None, budgets: pd.Series | None,
               n_resamples: int):
    from skfolio.optimization._base import BaseOptimization

    class PconMethod(BaseOptimization):
        """One of pcon's allocation methods as a skfolio / scikit-learn estimator."""

        def __init__(self, portfolio_params=None):
            super().__init__(portfolio_params=portfolio_params)

        def fit(self, X, y=None):
            R = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
            self.feature_names_in_ = np.asarray(R.columns, dtype=object)
            self.n_features_in_ = R.shape[1]
            self.weights_ = A.optimize(R, method, None, lo, hi, target, n_resamples,
                                       budgets=budgets).to_numpy(float)
            return self

    return PconMethod()


def cpcv_compare(R: pd.DataFrame, rf: pd.Series | None = None, methods: list[str] | None = None, lo: float = 0.0,
                 hi: float = 1.0, target: pd.Series | None = None, budgets: pd.Series | None = None,
                 n_folds: int = 10, n_test_folds: int = 2, purge: int = 5, embargo: int = 5,
                 n_resamples: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-method summary and the (methods x histories) matrix of annualised excess Sharpe ratios."""
    from skfolio.model_selection import CombinatorialPurgedCV, cross_val_predict

    X = R.sub(rf.reindex(R.index).fillna(0.0), axis=0) if rf is not None else R
    methods = [m for m in (methods or A.METHODS) if not (m == "risk_budget" and budgets is None)]
    cv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test_folds, purged_size=purge, embargo_size=embargo)
    sharpe, mdd = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in methods:
            pop = cross_val_predict(_estimator(m, lo, hi, target, budgets, n_resamples), X, cv=cv)
            sharpe[m] = [float(p.annualized_sharpe_ratio) for p in pop]
            mdd[m] = [float(p.max_drawdown) for p in pop]
    S = pd.DataFrame(sharpe).T
    S.columns = [f"history {k + 1}" for k in range(S.shape[1])]
    D = pd.DataFrame(mdd).T
    base = S.loc["target"] if "target" in S.index else None
    out = pd.DataFrame({
        "Median Sharpe": S.median(axis=1),
        "Worst Sharpe": S.min(axis=1),
        "Best Sharpe": S.max(axis=1),
        "Median max drawdown": -D.median(axis=1).abs(),
        "Beats your targets": (S.gt(base, axis=1).sum(axis=1) if base is not None else np.nan),
    })
    out.attrs.update(n_paths=S.shape[1], n_splits=cv.n_splits, n_folds=n_folds, n_test_folds=n_test_folds)
    return out, S
