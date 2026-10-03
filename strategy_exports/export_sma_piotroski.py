# =================================================================================================
# Export for the Portfolio-construction dashboard  ->  backtests/sma_piotroski.csv (+ .meta.json)
# Paste as the LAST cell of SMA_Piotroski_Backtest.ipynb (repo SMA_Pfscore) and run it after "Run All"
# (it needs section 8's diagnostics and, if enabled, section 9's look-ahead audit).
#
# Which strategy is exported: the notebook's FINAL selection, never a hand-picked one.
#   * the walk-forward OUT-OF-SAMPLE account (WF): every year it trades the configuration that won the
#     scorecard on the preceding in-sample years, under the current rules (the notebook advises against
#     switching to an ablation / ranking variant after seeing its table);
#   * its final tests are saved next to the returns: look-ahead audit, PSR, out-of-sample Sharpe vs SPY,
#     Deflated Sharpe, PBO, walk-forward efficiency and the bootstrap Sharpe interval. The dashboard shows them.
#
# Live tracking: once you trade the strategy, set LIVE_START to your first live day and re-run the notebook
# with REFRESH_DATA = True. From LIVE_START on, the export switches to a model account opened that same
# day with the walk-forward parameters, so it holds what your account should hold (same entries, same
# exits) and the dashboard can measure your execution against it day by day.
# =================================================================================================
import json
import os
from pathlib import Path

EXPORT_DIR = Path(os.environ.get("PCON_BACKTESTS", "../Portfolio-construction/data/backtests"))
LIVE_START = None            # e.g. "2026-10-05" = first day you traded this strategy live

ret, expo = WF["returns"], WF["exposure"]
if LIVE_START:
    cut = pd.Timestamp(LIVE_START)
    fresh = run_backtest(FEAT, schedule, start=cut)          # model account opened the day you went live
    ret = pd.concat([ret.loc[:cut - pd.Timedelta(days=1)], fresh["returns"]])
    expo = pd.concat([expo.loc[:cut - pd.Timedelta(days=1)], fresh["exposure"]])

oos_sr, bench_sr = perf_stats(OOS_R).get("Sharpe", float("nan")), perf_stats(B_OOS).get("Sharpe", float("nan"))
audit_ok = bool(audit["PASS"].all()) if "audit" in globals() and RUN_LOOKAHEAD_AUDIT else None
tests = {
    "Look-ahead audit: truncated-data rebuild identical": audit_ok,
    "PSR of the out-of-sample record > 0.95": bool(psr(OOS_R) > 0.95),
    f"Out-of-sample Sharpe >= {BENCHMARK} buy & hold": bool(oos_sr >= bench_sr),
    "Deflated Sharpe of the best configuration > 0.95": bool(dsr > 0.95),
    "PBO < 0.5 (CSCV over the configuration grid)": bool(pbo < 0.5),
    "Walk-forward efficiency >= 0.5": bool(WFE >= 0.5) if pd.notna(WFE) else None,
    "Bootstrap 5th-percentile Sharpe > 0": bool(ci.loc["Sharpe", "5%"] > 0),
}
n_pass = sum(v is True for v in tests.values())
n_eval = sum(v is not None for v in tests.values())
meta = {
    "strategy": "sma_piotroski",
    "selected": f"Walk-forward account; currently trading {label(schedule[-1][1])}",
    "selection_rule": f"Each out-of-sample year trades the configuration with the most scorecard points over the "
                      f"previous {WF_IS_YEARS} years ({len(LABELS)} configurations); the out-of-sample account is "
                      "the strategy.",
    "passed_selection": True,
    "final_tests": tests,
    "verdict": (f"{n_pass}/{n_eval} final tests passed. Out-of-sample Sharpe {oos_sr:.2f} vs {bench_sr:.2f} for "
                f"{BENCHMARK}; PSR {psr(OOS_R):.0%}, DSR {dsr:.0%}, PBO {pbo:.0%}, walk-forward efficiency {WFE:.2f}."),
    "oos_start": f"{OOS_R.index[0]:%Y-%m-%d}",
    "source": "SMA_Piotroski_Backtest.ipynb",
    "live_start": LIVE_START,
}

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
out = pd.DataFrame({"return": ret, "exposure": expo}).dropna(subset=["return"]).rename_axis("date")
meta["data_end"] = f"{out.index[-1]:%Y-%m-%d}"
out.to_csv(EXPORT_DIR / "sma_piotroski.csv", date_format="%Y-%m-%d", float_format="%.8f")
(EXPORT_DIR / "sma_piotroski.meta.json").write_text(json.dumps(meta, indent=2))
print(f"Saved {len(out):,} days ({out.index[0]:%Y-%m-%d} -> {out.index[-1]:%Y-%m-%d}) to "
      f"{(EXPORT_DIR / 'sma_piotroski.csv').resolve()}")
print(meta["selected"])
print(meta["verdict"])
