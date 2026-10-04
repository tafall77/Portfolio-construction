"""Orders: signals sized to the allocation, compared with the sleeves' holdings."""
import json

import numpy as np
import pandas as pd
import pytest

from pcon import orders as O
from pcon.book import Book
from pcon.config import load_config


def _w(weights, **kw):
    return {"kind": "weights", "as_of": pd.Timestamp("2026-01-14"), "weights": weights, **kw}


def test_weights_first_entry_and_band(workspace):
    cfg = load_config(workspace)
    p = O.plan_weights("a", _w({"SPY": 1.0}), 10_000, {}, {"SPY": 99.0}, cfg)
    assert p.orders[["symbol", "side", "quantity"]].values.tolist() == [["SPY", "BUY", 101]]
    # inside the band: no trade (the model only trades when the signal changes)
    assert O.plan_weights("a", _w({"SPY": 1.0}), 10_000, {"SPY": 95}, {"SPY": 100.0}, cfg).orders.empty
    # signal flips to cash: sell everything
    p = O.plan_weights("a", _w({}), 10_000, {"SPY": 95}, {"SPY": 100.0}, cfg)
    assert p.orders[["side", "quantity"]].values.tolist() == [["SELL", 95]]


def test_monthly_signal_waits_for_its_trade_date(workspace):
    cfg = load_config(workspace)
    sig = _w({"SPY": 1.0}, next_weights={"QQQ": 1.0}, next_trade="2026-02-02")
    before = O.plan_weights("a", sig, 10_000, {"SPY": 100}, {"SPY": 100.0, "QQQ": 50.0}, cfg, today="2026-01-20")
    assert before.orders.empty and "Scheduled" in before.notes[0]
    on = O.plan_weights("a", sig, 10_000, {"SPY": 100}, {"SPY": 100.0, "QQQ": 50.0}, cfg, today="2026-02-02")
    assert sorted(on.orders[["symbol", "side", "quantity"]].values.tolist()) == [["QQQ", "BUY", 200],
                                                                                 ["SPY", "SELL", 100]]
    assert not O.is_stale(sig, "2026-02-27") and O.is_stale(sig, "2026-03-02")
    assert O.is_stale(_w({}), "2026-01-19") and not O.is_stale(_w({}), "2026-01-15")


def test_stock_picks_exits_entries_and_limits(workspace):
    cfg = load_config(workspace)
    sig = {"kind": "stock_picks", "as_of": pd.Timestamp("2026-01-14"), "max_positions": 4, "position_weight": 0.25,
           "market_ok": True,
           "holdings": [{"symbol": "AAA", "entry_date": "2025-06-01", "last_close": 50.0, "take_profit": 60.0,
                         "exit": None},
                        {"symbol": "BBB", "entry_date": "2025-03-01", "last_close": 20.0, "take_profit": 30.0,
                         "exit": "F-Score below min"},
                        {"symbol": "CCC", "entry_date": "2025-09-01", "last_close": 10.0, "take_profit": 12.0,
                         "exit": None}],
           "buys": [{"symbol": "DDD", "weight": 0.25, "last_close": 33.0, "take_profit": 40.0}],
           "backups": [{"symbol": "EEE", "weight": 0.25, "take_profit": 9.0}]}
    held = {"AAA": 50, "BBB": 100, "ZZZ": 10}
    p = O.plan_stock_picks("s", sig, 10_000, held, {"ZZZ": 5.0}, cfg)
    o = p.orders.set_index(["symbol", "order"])
    assert o.loc[("AAA", "Limit, day"), "limit"] == 60.0                       # take-profit on a kept holding
    assert o.loc[("BBB", "Market on open"), "side"] == "SELL"                 # model exit
    assert o.loc[("ZZZ", "Market on open"), "side"] == "SELL"                 # not held by the model
    assert o.loc[("DDD", "Market on open"), "quantity"] == 75                 # floor(2500 / 33)
    assert any("CCC" in n for n in p.notes) and any("EEE" in n for n in p.notes)
    sig["market_ok"], sig["buys"] = False, []
    assert any("risk-off" in n for n in O.plan_stock_picks("s", sig, 10_000, held, {}, cfg).notes)


def test_net_orders_across_strategies():
    a = O.Plan("a", 0, O._frame([dict(strategy="a", symbol="SPY", side="BUY", quantity=10, order="At the close",
                                      limit=np.nan, price=100.0, value=1000.0, reason="")]), pd.DataFrame())
    b = O.Plan("b", 0, O._frame([dict(strategy="b", symbol="SPY", side="SELL", quantity=4, order="At the close",
                                      limit=np.nan, price=100.0, value=400.0, reason="")]), pd.DataFrame())
    n = O.net_orders([a, b])
    assert n[["symbol", "side", "quantity", "strategies"]].values.tolist() == [["SPY", "BUY", 6, "a, b"]]


def test_book_reads_signals_and_sizes_by_weight(workspace):
    d = workspace / "signals"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"kind": "weights", "as_of": "2026-01-16", "weights": {"SPY": 1.0}}))
    (d / "b.json").write_text(json.dumps({"kind": "weights", "as_of": "2026-01-16", "weights": {"SPY": 1.5}}))
    b = Book(workspace, end="2026-01-16")
    assert set(b.signals) == {"a"} and any("b.json" in w for w in b.data_warnings())
    plans = b.order_plans(pd.Series({"a": 0.5, "b": 0.5}), 20_000)
    assert plans[0].capital == pytest.approx(10_000)
    assert plans[0].orders["quantity"].iloc[0] == 10_000 // 109           # SPY closes at 109 in the fixture
