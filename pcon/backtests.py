"""Load the daily return series exported from each strategy's research notebook.

Standard format (see ``strategy_exports/``)::

    date,return,exposure
    2014-01-02,0.0012,0.80

``return`` is the strategy's daily net return on its own capital (fraction, not %). ``exposure`` (optional)
is the invested share of capital. A file with an ``equity``/``nav`` column instead of ``return`` also works.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RETURN_NAMES = ("return", "ret", "returns", "net", "strategy", "daily_return")
EQUITY_NAMES = ("equity", "nav", "value", "wealth")


def load_backtest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    idx = pd.DatetimeIndex(pd.to_datetime(df[date_col]))
    idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    rcol = next((c for c in RETURN_NAMES if c in df.columns), None)
    if rcol is not None:
        ret = pd.to_numeric(df[rcol], errors="coerce").to_numpy()
    else:
        ecol = next((c for c in EQUITY_NAMES if c in df.columns), None)
        if ecol is None:
            raise ValueError(f"{path.name}: need a 'return' (or 'equity') column, found {list(df.columns)}")
        eq = pd.to_numeric(df[ecol], errors="coerce")
        ret = (eq / eq.shift(1) - 1).to_numpy()
    out = pd.DataFrame({"ret": ret}, index=idx)
    if "exposure" in df.columns:
        out["exposure"] = pd.to_numeric(df["exposure"], errors="coerce").to_numpy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["ret"])
    if len(out) and out["ret"].abs().max() > 1.5:
        raise ValueError(f"{path.name}: returns above 150% in one day; the file looks like it is in percent. "
                         "Export fractions (0.01 = 1%).")
    return out


def load_all(cfg) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Backtest frames per strategy id plus warnings for missing / unreadable files."""
    out, warn = {}, []
    for sid, s in cfg.strategies.items():
        if s.backtest is None:
            warn.append(f"{s.short}: no backtest file configured; expected-vs-actual and allocation skip it.")
            continue
        if not s.backtest.exists():
            warn.append(f"{s.short}: backtest file {s.backtest.name} not found in backtests/ "
                        "(run the export cell in strategy_exports/).")
            continue
        try:
            out[sid] = load_backtest(s.backtest)
        except Exception as exc:
            warn.append(f"{s.short}: could not read {s.backtest.name}: {exc}")
    return out, warn
