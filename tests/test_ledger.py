import numpy as np
import pandas as pd
import pytest

from pcon.config import load_config
from pcon.journal import JournalError, append_cashflow, append_trade, load_cashflows, load_marks, load_trades
from pcon.ledger import build_ledger, twr_returns
from pcon.prices import PriceStore


def ledger_for(ws, end="2026-01-16"):
    cfg = load_config(ws)
    return build_ledger(cfg, load_trades(cfg), load_cashflows(cfg), PriceStore(cfg, load_marks(cfg)), end)


def test_split_dividend_and_fees(workspace):
    L = ledger_for(workspace)
    # 100 raw shares @100 = 200 split-adjusted @50 -> sold 200 @28: -4400, fees 2, dividend 200 x 0.5
    assert L.nav["a"].iloc[-1] == pytest.approx(20000 - 4400 - 2 + 100)
    rt = L.round_trips.set_index("symbol")
    assert rt.loc["XYZ", "pnl"] == pytest.approx(-4402)
    assert rt.loc["XYZ", "entry_price"] == pytest.approx(50)
    assert L.position_pnl[("a", "XYZ")].sum() == pytest.approx(-4300)       # price P&L + dividend


def test_futures_booked_at_notional(workspace):
    L = ledger_for(workspace)
    assert L.nav["b"].iloc[-1] == pytest.approx(30000 + 60 * 50 - 4)
    held = L.dates[(L.dates >= "2026-01-06") & (L.dates < "2026-01-14")]
    assert (L.long_exposure.loc[held, "b"] > 250_000).all()                 # notional, not margin
    # margin-style cash: not debited by the notional, variation margin settles into it daily
    assert np.allclose(L.cash.loc[held, "b"], L.nav.loc[held, "b"])


def test_pnl_identity(workspace):
    L = ledger_for(workspace)
    # NAV change = external flows + position P&L (price + dividends) - fees, every day
    nav = L.nav.sum(axis=1)
    lhs = nav.diff().fillna(nav)
    rhs = L.flows.sum(axis=1) + L.position_pnl.sum(axis=1) - L.fees.sum(axis=1)
    assert np.allclose(lhs, rhs)


def test_twr_removes_flows():
    nav = pd.Series([100.0, 110.0, 160.0, 176.0])
    flows = pd.Series([100.0, 0.0, 50.0, 0.0])
    r = twr_returns(nav, flows)
    assert r.iloc[0] == pytest.approx(0.0)
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(0.0)          # the 50 deposit is not performance
    assert r.iloc[3] == pytest.approx(0.10)


def test_transfer_nets_out_at_portfolio_level(workspace):
    with open(workspace / "cashflows.csv", "a") as f:
        f.write("2026-01-08,a,transfer,-5000,\n2026-01-08,b,transfer,5000,\n")
    L = ledger_for(workspace)
    assert L.total.loc["2026-01-08", "flows"] == pytest.approx(0.0)
    assert L.flows.loc["2026-01-08", "b"] == pytest.approx(5000)


def test_unknown_strategy_and_bad_side(workspace):
    cfg = load_config(workspace)
    with pytest.raises(JournalError):
        append_trade(cfg, "2026-01-07", "nope", "SPY", "BUY", 1, 100)
    with pytest.raises(JournalError):
        append_trade(cfg, "2026-01-07", "a", "SPY", "HOLD", 1, 100)
    with open(workspace / "trades.csv", "a") as f:
        f.write("2026-01-07,zzz,SPY,BUY,1,100,0,\n")
    with pytest.raises(JournalError):
        load_trades(cfg)


def test_append_and_symbol_without_prices(workspace):
    cfg = load_config(workspace)
    append_trade(cfg, "2026-01-07", "a", "NOPX", "BUY", 10, 20.0, 0.5, "otc")
    append_cashflow(cfg, "2026-01-09", "a", "dividend", 3.0)
    L = ledger_for(workspace)
    assert any("NOPX" in w for w in L.warnings)
    op = L.open_positions.set_index("symbol")
    assert op.loc["NOPX", "last"] == pytest.approx(20.0)    # marked at the fill price
    assert L.income.loc["2026-01-09", "a"] == pytest.approx(3.0)


def test_marks_override_and_short_position(workspace):
    (workspace / "marks.csv").write_text("date,symbol,price\n2026-01-15,SPY,120\n")
    with open(workspace / "trades.csv", "a") as f:
        f.write("2026-01-13,a,SPY,SELL,10,107,0,short\n2026-01-16,a,SPY,BUY,10,109,0,cover\n")
    L = ledger_for(workspace)
    assert L.prices.loc["2026-01-15", "SPY"] == pytest.approx(120)
    rt = L.round_trips.set_index("symbol")
    assert rt.loc["SPY", "direction"] == "short"
    assert rt.loc["SPY", "pnl"] == pytest.approx(-20)


def test_position_flip_is_split(workspace):
    with open(workspace / "trades.csv", "a") as f:
        f.write("2026-01-07,a,SPY,BUY,10,102,0,\n2026-01-09,a,SPY,SELL,25,104,0,flip\n2026-01-13,a,SPY,BUY,15,106,0,\n")
    L = ledger_for(workspace)
    spy = L.round_trips[L.round_trips["symbol"] == "SPY"].sort_values("entry_date")
    assert list(spy["direction"]) == ["long", "short"]
    assert spy["pnl"].tolist() == pytest.approx([20.0, -30.0])
