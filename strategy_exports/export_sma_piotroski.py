# =================================================================================================
# Export for the Portfolio-construction dashboard  ->  backtests/sma_piotroski.csv
# Paste as the LAST cell of SMA_Piotroski_Backtest.ipynb (repo SMA_Pfscore) and run it after "Run All".
#
# What it writes: the walk-forward OUT-OF-SAMPLE account (WF), i.e. the honest record, as daily net returns
# on the strategy's own capital, plus its invested share ("exposure").
#
# Live tracking: once you trade the strategy, set LIVE_START to your first live day and re-run the notebook
# with REFRESH_DATA = True. From LIVE_START on, the export switches to a model account opened that same
# day with the walk-forward parameters, so it holds what your account should hold (same entries, same
# exits) and the dashboard can measure your execution against it day by day.
# =================================================================================================
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

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
out = pd.DataFrame({"return": ret, "exposure": expo}).dropna(subset=["return"]).rename_axis("date")
out.to_csv(EXPORT_DIR / "sma_piotroski.csv", date_format="%Y-%m-%d", float_format="%.8f")
print(f"Saved {len(out):,} days ({out.index[0]:%Y-%m-%d} -> {out.index[-1]:%Y-%m-%d}) to "
      f"{(EXPORT_DIR / 'sma_piotroski.csv').resolve()}")
print(f"Walk-forward config now trading: {label(schedule[-1][1])}")
