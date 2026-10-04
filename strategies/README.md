# Strategies

The three research notebooks, as validated in their own repositories:

| Folder | Notebook | Research repo |
|---|---|---|
| `sma_piotroski/` | `SMA_Piotroski_Backtest.ipynb` | tafall77/sma_pfscore |
| `regime_filter/` | `regime_filter_backtest.ipynb` | tafall77/regime-filter |
| `rolling_momentum/` | `Rolling_Momentum_Report.ipynb` + `rolling_momentum.py` | tafall77/rolling-momentum |

They are stored here unchanged (outputs stripped). The dashboard's **Update signals** button and
`python -m pcon update` run their code cells headless (see `pcon/runner.py`): data refresh switched on, caches
under `<workspace>/cache/strategies/`, then the matching cell from `../strategy_exports/`, which writes the
final, test-passing configuration's returns, selection record and current signal into the workspace.

To change a strategy, change it in its research repo, re-validate it there, and copy the notebook back here.
