# =================================================================================================
# Export for the Portfolio-construction dashboard  ->  backtests/rolling_momentum.csv
# Paste as the LAST cell of Rolling_Momentum_Report.ipynb (repo Rolling-momentum) and run it after
# "Run All". Set CONFIG["refresh_data"] = True in section 2 first so the series reaches the latest close.
#
# What it writes: daily net returns of the FROZEN lookback (SELECTED) on the market you trade, long/flat
# with the T-bill cash leg and 5 bp per switch, from the 1990 pre-sample onward, plus the position
# (1 = long, 0 = cash) as "exposure". The rule is path-independent, so the export over your live dates
# is exactly what you should have earned: the dashboard measures execution against it.
#
# In portfolio.yaml, `expectation_start: 2021-01-01` uses only the untouched out-of-sample years;
# 1990-01-01 (the pre-sample holdout plus in-sample) gives a longer, slightly optimistic expectation.
# =================================================================================================
import os
from pathlib import Path

EXPORT_DIR = Path(os.environ.get("PCON_BACKTESTS", "../Portfolio-construction/data/backtests"))
MARKET = "S&P 500"           # "S&P 500" (SPY / ES) or "Nasdaq-100" (QQQ / NQ)

frame = (spx_frames if MARKET == "S&P 500" else ndx_frames)[SELECTED]
out = (frame.loc[CONFIG["presample_start"]:, ["net", "pos"]]
       .rename(columns={"net": "return", "pos": "exposure"}).dropna(subset=["return"]).rename_axis("date"))
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(EXPORT_DIR / "rolling_momentum.csv", date_format="%Y-%m-%d", float_format="%.8f")
print(f"{MARKET}, lookback {SELECTED}: saved {len(out):,} days ({out.index[0]:%Y-%m-%d} -> "
      f"{out.index[-1]:%Y-%m-%d}) to {(EXPORT_DIR / 'rolling_momentum.csv').resolve()}")
print("Position for the next session:", "LONG" if frame["signal"].iloc[-1] == 1 else "CASH")
