"""Independent checks against skfolio (skipped when it is not installed)."""
import numpy as np
import pandas as pd
import pytest

from pcon import allocation as A

skfolio = pytest.importorskip("skfolio")
from skfolio import RiskMeasure  # noqa: E402
from skfolio.moments import LedoitWolf  # noqa: E402
from skfolio.optimization import MeanRisk, ObjectiveFunction, RiskBudgeting  # noqa: E402
from skfolio.prior import EmpiricalPrior  # noqa: E402


@pytest.fixture
def R():
    rng = np.random.default_rng(5)
    n = 1500
    common = rng.normal(0.0003, 0.008, n)
    return pd.DataFrame({"a": common + rng.normal(0.0003, 0.006, n), "b": 0.5 * common + rng.normal(0.0002, 0.004, n),
                         "c": rng.normal(0.0004, 0.012, n)}, index=pd.bdate_range("2018-01-01", periods=n))


def _prior():
    return EmpiricalPrior(covariance_estimator=LedoitWolf())


@pytest.mark.parametrize("hi", [1.0, 0.5])
def test_max_sharpe_and_min_variance_match_skfolio(R, hi):
    mu, cov = A.estimate(R)
    sr = lambda w: (w @ mu) / np.sqrt(w @ cov @ w)
    ours = A.optimize(R, "max_sharpe", None, 0, hi).to_numpy()
    theirs = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, prior_estimator=_prior(),
                      max_weights=hi).fit(R.to_numpy()).weights_
    assert sr(ours) >= sr(theirs) - 1e-4
    ours = A.optimize(R, "min_variance", None, 0, hi).to_numpy()
    theirs = MeanRisk(prior_estimator=_prior(), max_weights=hi).fit(R.to_numpy()).weights_
    assert ours @ cov @ ours <= theirs @ cov @ theirs * (1 + 1e-5)


def test_risk_parity_matches_skfolio(R):
    ours = A.optimize(R, "risk_parity").to_numpy()
    theirs = RiskBudgeting(risk_measure=RiskMeasure.VARIANCE, prior_estimator=_prior()).fit(R.to_numpy()).weights_
    assert ours == pytest.approx(theirs, abs=2e-3)


def test_hrp_and_min_cvar_respect_bounds(R):
    for m in ("hrp", "min_cvar"):
        w = A.optimize(R, m, None, 0.1, 0.5)
        assert w.sum() == pytest.approx(1) and w.min() >= 0.1 - 1e-6 and w.max() <= 0.5 + 1e-6


def test_cpcv_summary(R):
    from pcon import robustness as RB
    out, S = RB.cpcv_compare(R, methods=["target", "equal", "max_sharpe"], n_folds=6, n_test_folds=2,
                             target=pd.Series({"a": 0.6, "b": 0.2, "c": 0.2}))
    assert S.shape == (3, out.attrs["n_paths"]) and out.attrs["n_paths"] == 5
    assert out.loc["target", "Beats your targets"] == 0
    assert (out["Worst Sharpe"] <= out["Median Sharpe"]).all()
