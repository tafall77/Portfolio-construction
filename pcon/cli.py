"""Command line: ``python -m pcon <command>``.

    init        create your workspace (default: data/)
    demo        generate the synthetic demo workspace (examples/demo/)
    trade       log a fill            python -m pcon trade 2026-10-05 sma_piotroski AAPL BUY 40 227.31 --fees 0.01
    cash        log a cash flow       python -m pcon cash 2026-10-01 regime_filter deposit 50000
    transfer    move money between sleeves   python -m pcon transfer 2026-11-02 sma_piotroski rolling_momentum 5000
    capital     deposits, withdrawals, net invested, money- and time-weighted returns per strategy
    summary     NAV, sleeves, health checks
    allocate    add/remove analysis and allocation methods (walk-forward)
    dashboard   start the Streamlit dashboard
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WS = ROOT / "data"


def _fmt_table(df: pd.DataFrame) -> str:
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        return df.to_string()


def cmd_init(a):
    from .workspace import init_workspace
    p = init_workspace(a.path)
    print(f"Workspace ready at {p.resolve()}\n"
          "Next: edit portfolio.yaml (target weights), export each strategy's backtest into backtests/ "
          "(strategy_exports/README.md), then log your deposits and fills.")


def cmd_demo(a):
    from .demo import build_demo
    p = build_demo(a.path, force=a.force)
    print(f"Demo workspace at {p.resolve()} (synthetic data)")


def cmd_trade(a):
    from .config import load_config
    from .journal import append_trade
    row = append_trade(load_config(a.workspace), a.date, a.strategy, a.symbol, a.side, a.quantity, a.price, a.fees,
                       a.note)
    print("added:", row)


def cmd_cash(a):
    from .config import load_config
    from .journal import append_cashflow
    row = append_cashflow(load_config(a.workspace), a.date, a.strategy, a.type, a.amount, a.note)
    print("added:", row)


def cmd_transfer(a):
    from .config import load_config
    from .journal import append_transfer
    for row in append_transfer(load_config(a.workspace), a.date, a.from_strategy, a.to_strategy, a.amount, a.note):
        print("added:", row)


def cmd_capital(a):
    from .book import Book
    b = Book(a.workspace)
    cs = b.capital_summary()
    if cs.empty:
        print("No cash flows yet.")
        return
    out = cs.drop(columns=["First flow"]).copy()
    for c in out.columns:
        out[c] = [(f"{v:+.2%}" if "return" in c or "(ann.)" in c else f"{v:,.0f}") if isinstance(v, (int, float)) and v == v else "-"
                  for v in out[c]]
    print(_fmt_table(out.T))
    fp = b.flows_by_period("M")
    if len(fp):
        print("\nDeposits / withdrawals by month:")
        print(_fmt_table(fp[(fp["Deposits"] != 0) | (fp["Withdrawals"] != 0)].round(0)))


def cmd_summary(a):
    from . import metrics as M
    from .book import Book
    b = Book(a.workspace)
    k = b.kpis()
    if k:
        print(f"{b.cfg.name}  as of {k['As of']:%Y-%m-%d}")
        print(f"NAV {k['NAV']:,.0f}  P&L {k['P&L (ITD)']:+,.0f}  TWR {k['ITD (TWR)']:+.2%}  "
              f"YTD {k['YTD']:+.2%}  MTD {k['MTD']:+.2%}  day {k['Day P&L']:+,.0f}")
        ls = b.live_summary()
        if not ls.empty:
            print(_fmt_table(M.format_frame(ls.reindex(["Total return", "CAGR", "Volatility", "Sharpe",
                                                        "Max drawdown", "Current drawdown", "Beta"]))))
        w = b.ledger.weights().iloc[-1]
        print("\nweights:", ", ".join(f"{b.label(s)} {v:.1%} (target {b.cfg.target_weights().get(s, 0):.0%})"
                                      for s, v in w.items()))
    else:
        print("No live record yet.")
    print("\nHealth:")
    for h in b.health():
        print(f"  [{h['level']:>5}] {h['scope']}: {h['msg']}")


def cmd_allocate(a):
    from . import allocation as A
    from .book import Book
    b = Book(a.workspace)
    R = b.alloc_matrix(a.source)
    if R.shape[1] < 2:
        sys.exit("Need at least two strategies with backtest returns in backtests/.")
    rf = b.rf.reindex(R.index).fillna(0.0)
    tgt = b.cfg.target_weights().reindex(R.columns).fillna(0.0)
    budgets = b.cfg.risk_budgets()
    budgets = budgets.reindex(R.columns).fillna(0.0) if budgets is not None else None
    al = b.cfg.allocation
    lo = al.min_weight
    print(f"Window {R.index[0]:%Y-%m-%d} -> {R.index[-1]:%Y-%m-%d} ({len(R)} days), source={a.source}\n")
    mg = A.marginal(R, rf, "target", al.rebalance, lo, al.max_weight, tgt, budgets=budgets)
    print("Add / remove analysis (at target weights):")
    print(_fmt_table(mg.round(3)))
    rows, W = {}, {}
    for m in A.METHODS:
        if m == "risk_budget" and budgets is None:
            continue
        w = A.optimize(R, m, rf, lo, al.max_weight, tgt, 100, budgets=budgets)
        W[m] = w
        s = A.stats_row(A.portfolio_returns(R, w, al.rebalance), rf)
        wf = A.stats_row(A.walk_forward(R, m, rf, int(al.walk_forward_lookback_years * 252), al.walk_forward_step,
                                        al.rebalance, lo, al.max_weight, tgt, 30, budgets)[0], rf) \
            if m not in ("equal", "target") else A.stats_row(A.portfolio_returns(R, w, al.rebalance)
                                                             .iloc[int(al.walk_forward_lookback_years * 252):], rf)
        rows[m] = {**{b.label(c): w[c] for c in R.columns}, "Sharpe": s.get("Sharpe"), "MaxDD": s.get("Max drawdown"),
                   "WF Sharpe": wf.get("Sharpe"), "WF MaxDD": wf.get("Max drawdown")}
    print("\nAllocation methods (WF = walk-forward, out of sample):")
    print(_fmt_table(pd.DataFrame(rows).T.round(3)))
    d = A.diversification(R, tgt, b.bench if len(b.bench) else None)
    print(f"\nAt your targets: {d['Effective number of bets']:.2f} effective bets, diversification ratio "
          f"{d['Diversification ratio']:.2f}"
          + (f", {d['Share of variance from benchmark beta']:.0%} of variance is {b.cfg.benchmark} beta"
             if "Share of variance from benchmark beta" in d else ""))


def cmd_dashboard(a):
    args = [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py")]
    if a.workspace:
        args += ["--", "--workspace", str(a.workspace)]
    subprocess.run(args, check=False)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m pcon", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init")
    s.add_argument("--path", default=str(DEFAULT_WS))
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("demo")
    s.add_argument("--path", default=str(ROOT / "examples" / "demo"))
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_demo)
    s = sub.add_parser("trade")
    for k in ("date", "strategy", "symbol", "side"):
        s.add_argument(k)
    s.add_argument("quantity", type=float)
    s.add_argument("price", type=float)
    s.add_argument("--fees", type=float, default=0.0)
    s.add_argument("--note", default="")
    s.add_argument("--workspace", default=str(DEFAULT_WS))
    s.set_defaults(fn=cmd_trade)
    s = sub.add_parser("cash")
    for k in ("date", "strategy", "type"):
        s.add_argument(k)
    s.add_argument("amount", type=float)
    s.add_argument("--note", default="")
    s.add_argument("--workspace", default=str(DEFAULT_WS))
    s.set_defaults(fn=cmd_cash)
    s = sub.add_parser("transfer")
    s.add_argument("date")
    s.add_argument("from_strategy")
    s.add_argument("to_strategy")
    s.add_argument("amount", type=float)
    s.add_argument("--note", default="")
    s.add_argument("--workspace", default=str(DEFAULT_WS))
    s.set_defaults(fn=cmd_transfer)
    for name, fn in (("summary", cmd_summary), ("allocate", cmd_allocate), ("capital", cmd_capital)):
        s = sub.add_parser(name)
        s.add_argument("--workspace", default=str(DEFAULT_WS))
        if name == "allocate":
            s.add_argument("--source", default="backtest", choices=["backtest", "backtest+live", "live"])
        s.set_defaults(fn=fn)
    s = sub.add_parser("dashboard")
    s.add_argument("--workspace", default=None)
    s.set_defaults(fn=cmd_dashboard)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
