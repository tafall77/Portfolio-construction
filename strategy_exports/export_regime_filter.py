# =================================================================================================
# Export for the Portfolio-construction dashboard  ->  backtests/regime_filter.csv (+ .meta.json)
# Paste as the LAST cell of regime_filter_backtest.ipynb (repo Regime-filter) and run it after
# "Restart Kernel and Run All" (the notebook downloads fresh FRED / Yahoo data by default).
#
# Which strategy is exported: the notebook's FINAL selection, never a hand-picked one.
#   * the allocation rule with the highest Sharpe on the untouched test window (2005 -> today), the same
#     pick the notebook's "Key findings" cell reports ("Highest test-window Sharpe");
#   * its final tests are saved next to the returns: look-ahead audit, engine causality, test-window Sharpe
#     and drawdown vs S&P 500 buy & hold, bootstrap significance, placebo timing test and, for the optimised
#     rules, Deflated Sharpe and PBO. The dashboard shows them.
# RULE_OVERRIDE lets you trade a different rule on purpose; the export then says so.
#
# What it writes: daily net returns of that rule (drifting weights, monthly fills one day after the
# signal, 5 bp per unit of turnover, cash at the T-bill rate), plus the equity exposure (S&P + Nasdaq
# weight). The rule is path-independent, so re-running after you go live extends the series over your
# live dates and the dashboard compares your fills with it day by day.
# =================================================================================================
import json
import os
from pathlib import Path

EXPORT_DIR = Path(os.environ.get("PCON_BACKTESTS", "../Portfolio-construction/data/backtests"))
RULE_OVERRIDE = None         # e.g. "Tiered 70/40 (user)" to export a rule other than the test-window winner

test_name = list(windows)[2]                      # 'Test (2005-01 to today)'
test_tbl = tables[test_name]
winner = test_tbl.loc["Sharpe", STRAT_NAMES].astype(float).idxmax()
RULE = RULE_OVERRIDE or winner
res = results[RULE]

bench = "S&P 500 B&H"
diff, lo5, hi95, p_not_better = boot_sharpe_diff(res["ret"].loc[test_start:], results[bench]["ret"].loc[test_start:])
fitted = RULE in ("Optimized Tiers", "Optimized + Trend")
placebo_p = None
if RULE in placebo:
    act, pl = placebo[RULE]
    placebo_p = float((pl >= act).mean())
tests = {
    "Look-ahead audit: truncated-data rebuild matches": bool(lookahead_ok),
    "Engine causality: future returns do not change the past": bool(causal_ok),
    f"Test window: Sharpe >= {bench}": bool(test_tbl.loc["Sharpe", RULE] >= test_tbl.loc["Sharpe", bench]),
    f"Test window: max drawdown shallower than {bench}": bool(test_tbl.loc["Max Drawdown", RULE]
                                                             > test_tbl.loc["Max Drawdown", bench]),
    "Test window: Sharpe gain significant (bootstrap P(diff <= 0) < 0.05)": bool(p_not_better < 0.05),
    "Placebo timing test: p < 0.10": None if placebo_p is None else bool(placebo_p < 0.10),
    "Deflated Sharpe > 0.95 (design window, 540 trials)": bool(dsr > 0.95) if fitted else None,
    "PBO < 0.5 (CSCV over the tier grid)": bool(pbo["pbo"] < 0.5) if fitted else None,
}
n_pass = sum(v is True for v in tests.values())
n_eval = sum(v is not None for v in tests.values())
meta = {
    "strategy": "regime_filter",
    "selected": RULE + (f" ({band_text(BEST)})" if fitted else ""),
    "selection_rule": "Allocation rule with the highest Sharpe on the untouched test window "
                      f"({test_name}); optimised tiers were fitted on 1986-2004 only.",
    "passed_selection": RULE == winner,
    "final_tests": tests,
    "verdict": (f"{RULE}: {n_pass}/{n_eval} final tests passed. Test-window Sharpe "
                f"{test_tbl.loc['Sharpe', RULE]:.2f} vs {test_tbl.loc['Sharpe', bench]:.2f} for {bench} "
                f"(difference {diff:+.2f}, 90% interval {lo5:+.2f} to {hi95:+.2f}); max drawdown "
                f"{test_tbl.loc['Max Drawdown', RULE]:.1%} vs {test_tbl.loc['Max Drawdown', bench]:.1%}."
                + ("" if RULE == winner else f" NOTE: exported by override; the test-window winner is {winner}.")),
    "oos_start": f"{test_start}-01",
    "source": "regime_filter_backtest.ipynb",
    "test_window_sharpe": {k: float(test_tbl.loc["Sharpe", k]) for k in STRAT_NAMES + [bench]},
}

expo = res["w"]["SPX"] + res["w"]["NDX"]
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
out = pd.DataFrame({"return": res["ret"], "exposure": expo}).dropna(subset=["return"]).rename_axis("date")
meta["data_end"] = f"{out.index[-1]:%Y-%m-%d}"
out.to_csv(EXPORT_DIR / "regime_filter.csv", date_format="%Y-%m-%d", float_format="%.8f")
(EXPORT_DIR / "regime_filter.meta.json").write_text(json.dumps(meta, indent=2))
print(f"{RULE}{' (test-window winner)' if RULE == winner else ' (OVERRIDE; winner is ' + winner + ')'}: "
      f"saved {len(out):,} days ({out.index[0]:%Y-%m-%d} -> {out.index[-1]:%Y-%m-%d}) to "
      f"{(EXPORT_DIR / 'regime_filter.csv').resolve()}")
print(meta["verdict"])
last = res["w"].iloc[-1]
print(f"Current weights: S&P {last['SPX']:.0%}, Nasdaq-100 {last['NDX']:.0%}, cash {last['CASH']:.0%}")

# ---- what to hold -> signals/regime_filter.json (read by the dashboard's Orders tab) ----------------------
# The backtest fills each month's decision at the close on the first trading day of the NEXT month
# (EXEC_LAG_DAYS after the month-end signal). So: hold the last filled decision now, and switch to the live
# reading (section "Live", decision month `now`) on the next monthly trade date.
TRADE_AS = {"SPX": "SPY", "NDX": "QQQ"}       # the symbols you trade for each sleeve (as logged in your journal)
SIGNALS_DIR = Path(os.environ.get("PCON_SIGNALS", EXPORT_DIR.parent / "signals"))
as_weights = lambda w: {TRADE_AS[k]: round(float(w[k]), 6) for k in TRADE_AS if float(w[k]) > 1e-9}
hold_now = res["target"].iloc[-1]
signal = {
    "strategy": "regime_filter", "kind": "weights", "selected": meta["selected"], "order": "At the close",
    "as_of": f"{(live_px if 'live_px' in globals() else rets['SPX']).dropna().index[-1]:%Y-%m-%d}",
    "generated": f"{pd.Timestamp.now():%Y-%m-%dT%H:%M:%S}", "source": meta["source"],
    "weights": as_weights(hold_now), "decision_month": str(res["target"].index[-1]),
    "execution": "Monthly. Trade at the close on the first trading day of the month; hold (weights drift) "
                 "until the next one. Cash = T-bills.",
}
last_px = {sym: float(prices[sym].dropna().iloc[-1]) for sym in TRADE_AS.values()
           if isinstance(prices.get(sym), pd.Series) and prices[sym].notna().any()}
if last_px:
    signal["prices"] = last_px                       # fallback when the dashboard cannot reach Yahoo
if "live_alloc" in globals() and RULE in live_alloc:
    nxt = pd.bdate_range((pd.Period(str(now), "M") + 1).start_time, periods=1)[0]
    signal.update(next_weights=as_weights(live_alloc[RULE]), next_trade=f"{nxt:%Y-%m-%d}",
                  next_decision_month=str(now))
else:
    print("NOTE: run the notebook's live section first to include next month's allocation.")
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
(SIGNALS_DIR / "regime_filter.json").write_text(json.dumps(signal, indent=2))
pretty = lambda w: ", ".join(f"{k} {v:.0%}" for k, v in w.items()) or "cash"
print(f"Signal: hold {pretty(signal['weights'])}"
      + (f"; from {signal['next_trade']}: {pretty(signal['next_weights'])}" if "next_trade" in signal else "")
      + f"  -> {(SIGNALS_DIR / 'regime_filter.json').resolve()}")
