"""Market data: daily closes, dividends and splits.

Source priority for each symbol:
1. ``<workspace>/prices/<SYMBOL>.csv`` (date, close[, dividends, splits]) -- local / demo data;
2. Yahoo Finance via ``yfinance``, cached in ``<workspace>/cache/prices`` and refreshed when older than
   ``price_max_age_hours``;
3. nothing -- the ledger then marks the position at your own fill prices and ``marks.csv``.

``marks.csv`` always wins over 1 and 2 on the dates it covers.

Yahoo's ``Close`` is split-adjusted (but not dividend-adjusted) for the *whole* history, so the ledger
restates your quantities into today's share units with the split history (see ``ledger.py``).
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PortfolioConfig

TD = 252


def _safe(symbol: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in symbol)


def _read_local(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    out = pd.DataFrame({"close": pd.to_numeric(df["close"], errors="coerce").to_numpy()},
                       index=pd.DatetimeIndex(pd.to_datetime(df[date_col])).tz_localize(None).normalize())
    for col, src in (("dividends", "dividends"), ("splits", "stock_splits"), ("splits", "splits")):
        if src in df.columns:
            out[col] = pd.to_numeric(df[src], errors="coerce").fillna(0.0).to_numpy()
    for col in ("dividends", "splits"):
        if col not in out:
            out[col] = 0.0
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["close"])


class PriceStore:
    """Loads and caches daily price histories; collects human-readable warnings for the dashboard."""

    def __init__(self, cfg: PortfolioConfig, marks: pd.DataFrame | None = None):
        self.cfg = cfg
        self.marks = marks if marks is not None else pd.DataFrame(columns=["date", "symbol", "price"])
        self.cache = cfg.cache_dir / "prices"
        self.warnings: list[str] = []
        self.sources: dict[str, str] = {}
        self._mem: dict[str, pd.DataFrame | None] = {}

    # ---------------------------------------------------------------------------------------------
    def yahoo_symbol(self, symbol: str) -> str:
        inst = self.cfg.instruments.get(symbol)
        if inst and inst.yahoo:
            return inst.yahoo
        return symbol.replace(".", "-") if not symbol.startswith("^") else symbol

    def _download(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame | None:
        try:
            import yfinance as yf
        except ImportError:
            self.warnings.append("yfinance is not installed: pip install yfinance")
            return None
        ysym = self.yahoo_symbol(symbol)
        for attempt in range(2):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = yf.download(ysym, start=start.strftime("%Y-%m-%d"), auto_adjust=False, actions=True,
                                     progress=False, threads=False)
                if df is None or df.empty:
                    raise ValueError("no rows returned")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                out = pd.DataFrame({
                    "close": df["Close"].astype(float),
                    "dividends": df["Dividends"].astype(float) if "Dividends" in df else 0.0,
                    "splits": df["Stock Splits"].astype(float) if "Stock Splits" in df else 0.0,
                })
                idx = pd.DatetimeIndex(pd.to_datetime(out.index))
                out.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
                out = out[~out.index.duplicated(keep="last")].dropna(subset=["close"]).sort_index()
                return out
            except Exception as exc:  # network, rate limit, unknown symbol
                err = f"{type(exc).__name__}: {str(exc)[:120]}"
                time.sleep(1 + attempt)
        self.warnings.append(f"Yahoo download failed for {symbol} ({ysym}): {err}")
        return None

    def _from_cache(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame | None:
        path = self.cache / f"{_safe(symbol)}.csv"
        meta_path = self.cache / f"{_safe(symbol)}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        cached = _read_local(path) if path.exists() else None
        fresh = (cached is not None and time.time() - path.stat().st_mtime < self.cfg.price_max_age_hours * 3600
                 and pd.Timestamp(meta.get("start", "2100-01-01")) <= start)
        if fresh or self.cfg.offline:
            if cached is None and self.cfg.offline:
                self.warnings.append(f"No cached prices for {symbol} (offline mode)")
            return cached
        dl = self._download(symbol, min(start, pd.Timestamp(meta.get("start", start))))
        if dl is None:
            if cached is not None:
                self.warnings.append(f"Using cached prices for {symbol} (last {cached.index[-1]:%Y-%m-%d})")
            return cached
        self.cache.mkdir(parents=True, exist_ok=True)
        dl.rename_axis("date").to_csv(path)
        meta_path.write_text(json.dumps({"start": str(min(start, pd.Timestamp(meta.get("start", start))).date()),
                                         "yahoo": self.yahoo_symbol(symbol)}))
        return dl

    def history(self, symbol: str, start) -> pd.DataFrame | None:
        """Daily frame with columns close, dividends, splits (or None when no market data exists)."""
        start = pd.Timestamp(start).normalize()
        if symbol in self._mem:
            df = self._mem[symbol]
        else:
            local = self.cfg.prices_dir / f"{_safe(symbol)}.csv"
            if local.exists():
                df, self.sources[symbol] = _read_local(local), "local file"
            else:
                df = self._from_cache(symbol, start - pd.Timedelta(days=10))
                self.sources[symbol] = "Yahoo Finance" if df is not None else "none"
            df = self._apply_marks(symbol, df)
            self._mem[symbol] = df
        return df

    def _apply_marks(self, symbol: str, df: pd.DataFrame | None) -> pd.DataFrame | None:
        m = self.marks[self.marks["symbol"] == symbol] if len(self.marks) else self.marks
        if m is None or m.empty:
            return df
        mk = pd.Series(m["price"].to_numpy(float), index=pd.DatetimeIndex(m["date"]).normalize())
        mk = mk[~mk.index.duplicated(keep="last")]
        if df is None:
            self.sources[symbol] = "marks.csv"
            return pd.DataFrame({"close": mk, "dividends": 0.0, "splits": 0.0}).sort_index()
        df = df.reindex(df.index.union(mk.index))
        df.loc[mk.index, "close"] = mk.to_numpy()
        df[["dividends", "splits"]] = df[["dividends", "splits"]].fillna(0.0)
        return df.sort_index()

    def closes(self, symbols, start) -> pd.DataFrame:
        cols = {}
        for s in symbols:
            h = self.history(s, start)
            if h is not None:
                cols[s] = h["close"]
        return pd.DataFrame(cols)

    # ---------------------------------------------------------------------------------------------
    def calendar(self, start, end=None) -> pd.DatetimeIndex:
        """Trading days: the benchmark's dates, else business days."""
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end or pd.Timestamp.today()).normalize()
        h = self.history(self.cfg.benchmark, start)
        if h is not None and len(h.loc[start:end]):
            idx = h.loc[start:end].index
            # include business days after the benchmark's last bar (e.g. today's fills before data arrives)
            extra = pd.bdate_range(idx[-1] + pd.Timedelta(days=1), end)
            return idx.append(extra) if len(extra) <= 3 else idx
        self.warnings.append(f"No benchmark data for {self.cfg.benchmark}: using Mon-Fri business days")
        return pd.bdate_range(start, end)

    def benchmark_returns(self, start) -> pd.Series:
        """Daily total return of the benchmark (price change + dividend on its ex-date)."""
        h = self.history(self.cfg.benchmark, start)
        if h is None:
            return pd.Series(dtype=float, name=self.cfg.benchmark)
        c = h["close"]
        r = (c + h["dividends"]) / c.shift(1) - 1
        return r.iloc[1:].rename(self.cfg.benchmark)

    def risk_free(self, index: pd.DatetimeIndex) -> pd.Series:
        """Daily risk-free return on ``index``: yield known at the previous close, accrued over calendar days."""
        rf = self.cfg.risk_free
        if isinstance(rf, float):
            return pd.Series((1 + rf) ** (1 / TD) - 1, index=index, name="rf")
        h = self.history(rf, index.min() - pd.Timedelta(days=30)) if len(index) else None
        if h is None or h.empty:
            self.warnings.append(f"No risk-free data ({rf}): Sharpe ratios use 0%")
            return pd.Series(0.0, index=index, name="rf")
        y = h["close"].clip(lower=0, upper=25) / 100
        y = y.reindex(y.index.union(index)).ffill().reindex(index)
        days = index.to_series().diff().dt.days.fillna(1).to_numpy()
        out = (1 + y.shift(1).bfill()) ** (days / 365.0) - 1
        return out.fillna(0.0).rename("rf")


def split_factor_after(splits: pd.Series, dates) -> np.ndarray:
    """Product of split ratios with ex-date strictly after each date: converts old units into today's."""
    s = splits[splits.fillna(0) > 0] if splits is not None else None
    dates = pd.DatetimeIndex(dates)
    if s is None or s.empty:
        return np.ones(len(dates))
    sd, ratio = s.index.values, s.to_numpy(float)
    cum_after = np.cumprod(ratio[::-1])[::-1]
    k = np.searchsorted(sd, dates.values, side="right")
    out = np.ones(len(dates))
    ok = k < len(sd)
    out[ok] = cum_after[k[ok]]
    return out
