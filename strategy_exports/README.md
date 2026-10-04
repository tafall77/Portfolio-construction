# Strategy exports

The dashboard never re-implements your strategies. Each research notebook exports one CSV of **daily net
returns on the strategy's own capital** into your workspace's `backtests/` folder:

```
date,return,exposure
2014-01-03,-0.00086111,0.12227053
```

| Strategy | Notebook (repo) | Paste this as the last cell | What it exports: the notebook's **final selection** |
|---|---|---|---|
| `sma_piotroski` | `SMA_Piotroski_Backtest.ipynb` (SMA_Pfscore) | [`export_sma_piotroski.py`](export_sma_piotroski.py) | The walk-forward out-of-sample account (`WF`). Each year it trades the configuration that won the scorecard on the previous 3 years, under the current rules. With `LIVE_START` set, a model account is opened on your first live day. |
| `regime_filter` | `regime_filter_backtest.ipynb` (Regime-filter) | [`export_regime_filter.py`](export_regime_filter.py) | The allocation rule with the **highest Sharpe on the untouched 2005-onward test window**, picked automatically. That's the notebook's own "Highest test-window Sharpe". `RULE_OVERRIDE` exports another rule on purpose, and the record then says so. |
| `rolling_momentum` | `Rolling_Momentum_Report.ipynb` (Rolling-momentum) | [`export_rolling_momentum.py`](export_rolling_momentum.py) | The frozen lookback `SELECTED`: it survived all four in-sample gates and has the best in-sample Sharpe among survivors, on the S&P 500 where it was validated. |

No cell picks a configuration by hand, and none re-optimises. Each cell also writes a **selection record**,
`backtests/<strategy>.meta.json`, containing:

* the exported configuration and the rule that selected it;
* the notebook's **final tests** as pass/fail/not applicable:
  * SMA: look-ahead audit, PSR, OOS Sharpe vs SPY, DSR, PBO, walk-forward efficiency, bootstrap interval;
  * regime: look-ahead audit, engine causality, test-window Sharpe and drawdown vs S&P 500, bootstrap significance, placebo test, and DSR and PBO for the optimised tiers;
  * momentum: the 9-check scorecard of section 11 plus the pre-sample holdout;
* the notebook's verdict and the out-of-sample start date.

Each cell also writes **`signals/<strategy>.json`** next to `backtests/`, for the dashboard's *Orders* tab:

| Strategy | Signal |
|---|---|
| `regime_filter` | The weights to hold now (the last monthly decision, as SPY / QQQ / cash), plus the live section's allocation for the next monthly trade date. Map the sleeves to other symbols with `TRADE_AS`. |
| `rolling_momentum` | SPY 100 % (or QQQ with `MARKET = "Nasdaq-100"`) when the selected lookback's return is positive at the last close, otherwise cash; plus the momentum value. |
| `sma_piotroski` | The model account's next open, using `run_backtest`'s own rules: the holdings, which of them exit and why, the entries that fill the free slots (ranking, sector cap, market filter, sizing), three backups, and each holding's take-profit level. With `LIVE_START` set it follows the account opened on your first live day. |

Re-run with fresh data before trading: the signal is only as recent as the last close in the notebook (set
`REFRESH_DATA = True` / `CONFIG["refresh_data"] = True`). `PCON_SIGNALS` overrides the folder.

The dashboard shows this record on each strategy (*Strategies* tab, *Journal & data → Data status*, and the
Overview scoreboard). It raises an amber health check when the exported configuration failed a final test or
is not the notebook's own pick. Failing a test is information, not an automatic veto: for example, a
momentum rule judged "a risk overlay, not an alpha source" can still earn its place in the book through
diversification. *Portfolio construction → add/remove* tests exactly that.

By default the cells write to `../Portfolio-construction/data/backtests/`, which works when the four repos are
cloned side by side. Set the `PCON_BACKTESTS` environment variable to write somewhere else.

## When to re-export

* **Once before going live**: this is the *expectation* (cones, percentiles, allocation analysis).
* **Regularly after going live** (weekly or monthly, with fresh data): the export then covers your live dates
  and the dashboard compares your real fills with the model on the same days (*Expected vs actual → Live vs
  model*). For the SMA strategy set `LIVE_START` so the model account holds what yours should hold.

Each cell was run at the end of a full execution of its notebook, using synthetic data in place of Yahoo, FRED
and SEC: the SMA notebook's own synthetic test dataset, and synthetic cached series for the other two. That
confirms the variable names, the automatic selection and both output files.
