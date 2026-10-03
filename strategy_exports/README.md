# Strategy exports

The dashboard never re-implements your strategies. Each research notebook exports one CSV of **daily net
returns on the strategy's own capital** into your workspace's `backtests/` folder:

```
date,return,exposure
2014-01-03,-0.00086111,0.12227053
```

| Strategy | Notebook (repo) | Paste this as the last cell | What it exports |
|---|---|---|---|
| `sma_piotroski` | `SMA_Piotroski_Backtest.ipynb` (SMA_Pfscore) | [`export_sma_piotroski.py`](export_sma_piotroski.py) | The walk-forward out-of-sample account (`WF`). With `LIVE_START` set, a model account opened on your first live day. |
| `regime_filter` | `regime_filter_backtest.ipynb` (Regime-filter) | [`export_regime_filter.py`](export_regime_filter.py) | The allocation rule you trade (`RULE`, default *Optimized Tiers*). |
| `rolling_momentum` | `Rolling_Momentum_Report.ipynb` (Rolling-momentum) | [`export_rolling_momentum.py`](export_rolling_momentum.py) | The frozen lookback (`SELECTED`) on the market you trade (`MARKET`). |

By default the cells write to `../Portfolio-construction/data/backtests/`, which works when the four repos are
cloned side by side. Set the `PCON_BACKTESTS` environment variable to write somewhere else.

## When to re-export

* **Once before going live**: this is the *expectation* (cones, percentiles, allocation analysis).
* **Regularly after going live** (weekly or monthly, with fresh data): the export then covers your live dates
  and the dashboard compares your real fills with the model on the same days (*Expected vs actual → Live vs
  model*). For the SMA strategy set `LIVE_START` so the model account holds what yours should hold.

Each cell was run against its notebook's own code (the SMA notebook with its synthetic test dataset, the
regime and momentum notebooks with synthetic cached series) to check the variable names and output format.
