import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_prices(root: Path, symbol: str, dates, close, dividends=None, splits=None):
    (root / "prices").mkdir(parents=True, exist_ok=True)
    n = len(dates)
    pd.DataFrame({"date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"), "close": close,
                  "dividends": dividends if dividends is not None else np.zeros(n),
                  "splits": splits if splits is not None else np.zeros(n)}).to_csv(
        root / "prices" / f"{symbol.replace('^', '_')}.csv", index=False)


@pytest.fixture
def workspace(tmp_path):
    """Small offline workspace: two strategies, a futures instrument, SPY calendar."""
    (tmp_path / "portfolio.yaml").write_text(
        "portfolio:\n  name: Test\n  benchmark: SPY\n  risk_free: 0.0\n  offline: true\n"
        "strategies:\n  a: {name: A, target_weight: 0.5}\n  b: {name: B, target_weight: 0.5}\n"
        "instruments:\n  ES: {multiplier: 50, type: future}\n")
    d = pd.bdate_range("2026-01-05", periods=10)
    write_prices(tmp_path, "SPY", d, np.linspace(100, 109, 10))
    # XYZ: Yahoo-style split-adjusted closes; 2-for-1 split on day 5, 0.5 dividend (post-split units) on day 3
    write_prices(tmp_path, "XYZ", d, [50, 51, 52, 53, 27, 27.5, 28, 28, 29, 30.0],
                 dividends=[0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0], splits=[0, 0, 0, 0, 2, 0, 0, 0, 0, 0])
    write_prices(tmp_path, "ES", d, np.linspace(5000, 5090, 10))
    (tmp_path / "trades.csv").write_text(
        "date,strategy,symbol,side,quantity,price,fees,note\n"
        "2026-01-05,a,XYZ,BUY,100,100,1,pre-split units\n"
        "2026-01-12,a,XYZ,SELL,200,28,1,post-split units\n"
        "2026-01-06,b,ES,BUY,1,5010,2,\n"
        "2026-01-14,b,ES,SELL,1,5070,2,\n")
    (tmp_path / "cashflows.csv").write_text(
        "date,strategy,type,amount,note\n2026-01-05,a,deposit,20000,\n2026-01-05,b,deposit,30000,\n")
    return tmp_path
