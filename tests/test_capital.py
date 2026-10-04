import pandas as pd
import pytest

from pcon import metrics as M
from pcon.book import Book
from pcon.config import load_config
from pcon.journal import JournalError, append_cashflow, append_transfer, load_cashflows, save_cashflows, save_trades


def test_xirr_known_answers():
    assert M.xirr(["2025-01-01", "2026-01-01"], [-100, 110]) == pytest.approx(0.10)
    # money added just before a loss: money-weighted < time-weighted
    assert M.xirr(["2025-01-01", "2025-07-01", "2026-01-01"], [-100, -900, 950]) < 0
    assert pd.isna(M.xirr(["2025-01-01"], [-100]))


def test_capital_summary_and_transfers(workspace):
    cfg = load_config(workspace)
    append_transfer(cfg, "2026-01-08", "a", "b", 5000, "rebalance")
    append_cashflow(cfg, "2026-01-12", "b", "withdrawal", 1000)
    b = Book(workspace, end="2026-01-16")
    cs = b.capital_summary()
    tot = cs.loc["Total"]
    assert tot["Deposited"] == pytest.approx(50000)
    assert tot["Withdrawn"] == pytest.approx(1000)
    assert tot["Net invested"] == pytest.approx(49000)                 # transfers do not change the book's capital
    assert cs.loc["A", "Transfers out"] == pytest.approx(5000) and cs.loc["B", "Transfers in"] == pytest.approx(5000)
    assert cs.loc["A", "Net invested"] == pytest.approx(15000)
    assert tot["P&L"] == pytest.approx(tot["NAV"] - 49000)
    assert not any("net to" in w for w in b.data_warnings())
    fp = b.flows_by_period("M")
    assert fp.loc[pd.Period("2026-01", "M"), "Net new money"] == pytest.approx(49000)


def test_one_sided_transfer_is_flagged(workspace):
    cfg = load_config(workspace)
    append_cashflow(cfg, "2026-01-08", "a", "transfer", -5000)
    assert any("net to" in w for w in Book(workspace, end="2026-01-16").data_warnings())
    with pytest.raises(JournalError):
        append_transfer(cfg, "2026-01-08", "a", "a", 10)


def test_save_validates_and_keeps_backup(workspace):
    cfg = load_config(workspace)
    df = pd.read_csv(cfg.cashflows_path)
    df.loc[len(df)] = ["2026-01-09", "a", "deposit", 1000, "top-up"]
    save_cashflows(cfg, df)
    assert len(load_cashflows(cfg)) == 3 and cfg.cashflows_path.with_suffix(".csv.bak").exists()
    bad = df.copy()
    bad.loc[0, "type"] = "gift"
    with pytest.raises(JournalError):
        save_cashflows(cfg, bad)
    assert len(load_cashflows(cfg)) == 3                               # nothing written on a validation error
    tr = pd.read_csv(cfg.trades_path)
    tr.loc[0, "price"] = 101.0                                         # fix a typo in the first fill
    save_trades(cfg, tr)
    assert pd.read_csv(cfg.trades_path).loc[0, "price"] == 101.0


def test_first_deposit_on_a_weekend_after_the_last_trading_day(workspace):
    # SPY data ends Fri 2026-01-16; a fresh book whose only entry is a deposit dated Sunday 2026-01-18
    (workspace / "trades.csv").write_text("date,strategy,symbol,side,quantity,price,fees,note\n")
    (workspace / "cashflows.csv").write_text("date,strategy,type,amount,note\n2026-01-18,a,deposit,200,\n")
    b = Book(workspace, end="2026-01-18")
    assert not b.ledger.empty
    assert b.capital_summary().loc["Total", "NAV"] == pytest.approx(200)
