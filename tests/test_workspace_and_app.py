"""End-to-end: demo generation, Book, CLI, backtest loader, Yahoo adapter and the Streamlit app."""
import sys

import numpy as np
import pandas as pd
import pytest

from pcon.backtests import load_backtest
from pcon.book import Book
from pcon.cli import main
from pcon.config import load_config
from pcon.demo import build_demo
from pcon.prices import PriceStore
from pcon.workspace import init_workspace

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return build_demo(tmp_path_factory.mktemp("demo"))


def test_demo_book_is_consistent(demo):
    b = Book(demo)
    k = b.kpis()
    assert k["NAV"] > 0 and abs(k["NAV"] - k["Net invested"] - k["P&L (ITD)"]) < 1e-6
    assert set(b.live_returns()) == set(b.strategy_ids)
    for sid in b.strategy_ids:
        assert b.comparison(sid) is not None
        assert b.tracking(sid)["Correlation"] > 0.9
    assert b.comparison(None) is not None
    levels = {h["level"] for h in b.health()}
    assert levels <= {"red", "amber", "green", "info"}
    assert abs(b.rebalance_orders()["Transfer $"].sum()) < 1e-6


def test_cli(demo, capsys, tmp_path):
    main(["summary", "--workspace", str(demo)])
    assert "Health" in capsys.readouterr().out
    ws = init_workspace(tmp_path / "ws")
    main(["cash", "2026-10-01", "regime_filter", "deposit", "50000", "--workspace", str(ws)])
    main(["trade", "2026-10-02", "regime_filter", "QQQ", "BUY", "10", "600", "--fees", "1", "--workspace", str(ws)])
    cfg = load_config(ws)
    assert len(pd.read_csv(cfg.trades_path)) == 1 and len(pd.read_csv(cfg.cashflows_path)) == 1


def test_backtest_loader(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"date": ["2026-01-02", "2026-01-05"], "equity": [100, 101]}).to_csv(p, index=False)
    assert load_backtest(p)["ret"].iloc[0] == pytest.approx(0.01)
    pd.DataFrame({"date": ["2026-01-02", "2026-01-05"], "return": [1.0, 2.5]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_backtest(p)                      # looks like percent


def test_yahoo_adapter_shape(tmp_path, monkeypatch):
    """yfinance >= 0.2.5x returns (Price, Ticker) MultiIndex columns even for one ticker."""
    import yfinance as yf
    idx = pd.bdate_range("2026-01-05", periods=3, tz="America/New_York")
    cols = pd.MultiIndex.from_product([["Adj Close", "Close", "Dividends", "Stock Splits"], ["AAPL"]],
                                      names=["Price", "Ticker"])
    fake = pd.DataFrame(np.array([[1, 10, 0, 0], [1, 11, 0.2, 0], [1, 12, 0, 2]], float).repeat(1, axis=1),
                        index=idx, columns=cols)
    monkeypatch.setattr(yf, "download", lambda *a, **k: fake)
    init_workspace(tmp_path)
    cfg = load_config(tmp_path)
    h = PriceStore(cfg).history("AAPL", "2026-01-01")
    assert list(h["close"]) == [10, 11, 12] and h["splits"].iloc[-1] == 2 and h.index.tz is None
    assert (cfg.cache_dir / "prices" / "AAPL.csv").exists()


def test_streamlit_app_runs_on_demo(demo):
    from streamlit.testing.v1 import AppTest
    sys.argv = ["app.py", "--workspace", str(demo)]
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=600)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.tabs) == 8
