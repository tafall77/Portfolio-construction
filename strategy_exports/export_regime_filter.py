# =================================================================================================
# Export for the Portfolio-construction dashboard  ->  backtests/regime_filter.csv
# Paste as the LAST cell of regime_filter_backtest.ipynb (repo Regime-filter) and run it after
# "Restart Kernel and Run All" (the notebook downloads fresh FRED / Yahoo data by default).
#
# What it writes: daily net returns of the allocation rule you actually trade (drifting weights, monthly
# fills one day after the signal, 5 bp per unit of turnover, cash at the T-bill rate), plus the equity
# exposure (S&P + Nasdaq weight). The rule is the same every month, so re-running after you go live
# extends the series over your live dates and the dashboard compares your fills with it day by day.
#
# In portfolio.yaml keep `expectation_start: 2005-01-01` for the optimised tiers (they were fitted on
# 1986-2004). If you trade 'Tiered 70/40 (user)' or 'Proportional 50/50' (no fitted parameters) you can
# set it to 1986-01-01 for a longer expectation.
# =================================================================================================
import os
from pathlib import Path

EXPORT_DIR = Path(os.environ.get("PCON_BACKTESTS", "../Portfolio-construction/data/backtests"))
RULE = "Optimized Tiers"     # one of: 'Tiered 70/40 (user)', 'Optimized Tiers', 'Optimized + Trend', 'Proportional 50/50'

res = results[RULE]
expo = res["w"]["SPX"] + res["w"]["NDX"]
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
out = pd.DataFrame({"return": res["ret"], "exposure": expo}).dropna(subset=["return"]).rename_axis("date")
out.to_csv(EXPORT_DIR / "regime_filter.csv", date_format="%Y-%m-%d", float_format="%.8f")
print(f"{RULE}: saved {len(out):,} days ({out.index[0]:%Y-%m-%d} -> {out.index[-1]:%Y-%m-%d}) to "
      f"{(EXPORT_DIR / 'regime_filter.csv').resolve()}")
last = res["w"].iloc[-1]
print(f"Current weights: S&P {last['SPX']:.0%}, Nasdaq-100 {last['NDX']:.0%}, cash {last['CASH']:.0%}")
