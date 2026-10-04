# =================================================================================================
# Export for the Portfolio-construction dashboard  ->  backtests/rolling_momentum.csv (+ .meta.json)
# Paste as the LAST cell of Rolling_Momentum_Report.ipynb (repo Rolling-momentum) and run it after
# "Run All" (it needs the section 11 scorecard). Set CONFIG["refresh_data"] = True in section 2 first so the
# series reaches the latest close.
#
# Which strategy is exported: the notebook's FINAL selection, never a hand-picked one.
#   * the lookback SELECTED in section 6 (survived all four in-sample gates, highest in-sample Sharpe among
#     survivors, then frozen), on the S&P 500 where it was selected and validated;
#   * the section 11 scorecard (9 checks) and verdict are saved next to the returns, so the dashboard shows
#     whether this configuration passed the final tests.
# Set MARKET = "Nasdaq-100" only if the scorecard's NQ transfer check (6) passed and you trade NQ / QQQ.
#
# What it writes: daily net returns of that rule (long/flat, T-bill cash leg, 5 bp per switch) from the
# 1990 pre-sample onward, plus the position (1 = long, 0 = cash) as "exposure". The rule is
# path-independent, so the export over your live dates is exactly what you should have earned.
# =================================================================================================
import json
import os
import re
from pathlib import Path

EXPORT_DIR = Path(os.environ.get("PCON_BACKTESTS", "../Portfolio-construction/data/backtests"))
MARKET = "S&P 500"           # "S&P 500" (SPY / ES), or "Nasdaq-100" (QQQ / NQ) if check 6 passed

frame = (spx_frames if MARKET == "S&P 500" else ndx_frames)[SELECTED]
out = (frame.loc[CONFIG["presample_start"]:, ["net", "pos"]]
       .rename(columns={"net": "return", "pos": "exposure"}).dropna(subset=["return"]).rename_axis("date"))

tests = {k: (None if v is None else bool(v)) for k, v in checks.items()}
nq_check = next((k for k in tests if k.startswith("6.")), None)
if MARKET != "S&P 500" and nq_check and not tests[nq_check]:
    print(f"WARNING: exporting {MARKET} although the scorecard's transfer check FAILED ({nq_check}).")
meta = {
    "strategy": "rolling_momentum",
    "selected": f"{SELECTED} lookback, long/flat {MARKET}",
    "selection_rule": "Lookback that survived all four in-sample gates (2000-2020) with the highest in-sample "
                      "Sharpe; frozen for all out-of-sample and cross-market tests.",
    "passed_selection": bool(SELECTED_SURVIVED),
    "final_tests": tests,
    "verdict": re.sub(r"\*\*", "", verdict).strip(),
    "oos_start": CONFIG["oos_start"],
    "market": MARKET,
    "source": "Rolling_Momentum_Report.ipynb",
    "data_end": f"{out.index[-1]:%Y-%m-%d}",
}

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(EXPORT_DIR / "rolling_momentum.csv", date_format="%Y-%m-%d", float_format="%.8f")
(EXPORT_DIR / "rolling_momentum.meta.json").write_text(json.dumps(meta, indent=2))
n_pass = sum(v is True for v in tests.values())
n_eval = sum(v is not None for v in tests.values())
print(f"{MARKET}, lookback {SELECTED} ({'survived the gates' if SELECTED_SURVIVED else 'DID NOT survive the gates'}): "
      f"saved {len(out):,} days ({out.index[0]:%Y-%m-%d} -> {out.index[-1]:%Y-%m-%d}) to "
      f"{(EXPORT_DIR / 'rolling_momentum.csv').resolve()}")
print(f"Final scorecard: {n_pass}/{n_eval} checks passed. Verdict: {meta['verdict'][:160]}")
print("Position for the next session:", "LONG" if frame["signal"].iloc[-1] == 1 else "CASH")
