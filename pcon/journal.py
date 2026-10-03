"""Trade and cash-flow journal: the record of what you actually did.

``trades.csv``
    date, strategy, symbol, side, quantity, price, fees, note
    * ``side``: BUY or SELL (SHORT = SELL, COVER = BUY). ``quantity`` is always positive.
    * ``price``: fill price per share / contract. ``fees``: total commission + fees for the fill.

``cashflows.csv``
    date, strategy, type, amount, note
    * ``deposit`` / ``withdrawal``: money in / out of a sleeve (sign is taken from the type).
    * ``transfer``: signed amount moved between sleeves (+ into this sleeve, - out of it). Use two rows.
    * ``dividend`` / ``interest``: income (keeps the sign you give; negative = withholding / debit interest).
    * ``fee``: an expense not tied to a fill (platform, data, margin interest).
    Deposits, withdrawals and transfers are *external* flows and are removed from returns (time-weighted).
    Dividends, interest and fees are part of performance.

``marks.csv`` (optional)
    date, symbol, price  -- manual closing prices that override / fill the market data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CASH_SLEEVE, PortfolioConfig

TRADE_COLUMNS = ["date", "strategy", "symbol", "side", "quantity", "price", "fees", "note"]
CASH_COLUMNS = ["date", "strategy", "type", "amount", "note"]
MARK_COLUMNS = ["date", "symbol", "price"]
SIDES = {"BUY": 1, "COVER": 1, "SELL": -1, "SHORT": -1}
FLOW_TYPES = {"deposit", "withdrawal", "transfer", "dividend", "interest", "fee"}
EXTERNAL_TYPES = {"deposit", "withdrawal", "transfer"}


class JournalError(ValueError):
    pass


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype={"note": str}, skipinitialspace=True, comment="#")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in columns if c not in df.columns and c not in ("note", "fees")]
    if missing:
        raise JournalError(f"{path.name}: missing column(s) {missing}; expected {columns}")
    for c in columns:
        if c not in df.columns:
            df[c] = "" if c == "note" else 0.0
    return df[columns]


def _dates(s: pd.Series, name: str) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    if d.isna().any():
        bad = s[d.isna()].tolist()[:5]
        raise JournalError(f"{name}: unreadable date(s) {bad} (use YYYY-MM-DD)")
    return d.dt.tz_localize(None) if getattr(d.dt, "tz", None) is not None else d


def _numbers(s: pd.Series, name: str, col: str) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    if v.isna().any():
        rows = list(np.flatnonzero(v.isna().to_numpy())[:5] + 2)
        raise JournalError(f"{name}: non-numeric {col} on line(s) {rows}")
    return v.astype(float)


def _check_strategy(s: pd.Series, cfg: PortfolioConfig, name: str) -> pd.Series:
    s = s.astype(str).str.strip()
    known = set(cfg.strategies) | {CASH_SLEEVE}
    unknown = sorted(set(s) - known)
    if unknown:
        raise JournalError(f"{name}: unknown strategy id(s) {unknown}; known: {sorted(known)} "
                           f"(add them to portfolio.yaml or use '{CASH_SLEEVE}' for unallocated money)")
    return s


def validate_trades(df: pd.DataFrame, cfg: PortfolioConfig, name: str = "trades.csv") -> pd.DataFrame:
    """Validated trades with a signed ``qty`` column (+ buy, - sell), sorted by date (stable)."""
    df = df.copy()
    if df.empty:
        return df.assign(qty=pd.Series(dtype=float))
    df["date"] = _dates(df["date"], name)
    df["strategy"] = _check_strategy(df["strategy"], cfg, name)
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["side"] = df["side"].astype(str).str.strip().str.upper()
    bad = sorted(set(df["side"]) - set(SIDES))
    if bad:
        raise JournalError(f"{name}: side must be one of {sorted(SIDES)}, got {bad}")
    if (df["symbol"].isin(["", "NAN", "NONE"])).any():
        raise JournalError(f"{name}: every fill needs a symbol")
    df["quantity"] = _numbers(df["quantity"], name, "quantity").abs()
    df["price"] = _numbers(df["price"], name, "price")
    if (df["price"] <= 0).any():
        raise JournalError(f"{name}: prices must be positive")
    df["fees"] = _numbers(df["fees"].replace("", 0).fillna(0), name, "fees").abs()
    df["note"] = df["note"].fillna("").astype(str)
    df["qty"] = df["quantity"] * df["side"].map(SIDES)
    df = df[df["quantity"] > 0]
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def load_trades(cfg: PortfolioConfig) -> pd.DataFrame:
    return validate_trades(_read(cfg.trades_path, TRADE_COLUMNS), cfg, cfg.trades_path.name)


def validate_cashflows(df: pd.DataFrame, cfg: PortfolioConfig, name: str = "cashflows.csv") -> pd.DataFrame:
    """Validated cash flows with a signed ``value`` column and an ``external`` flag."""
    df = df.copy()
    if df.empty:
        return df.assign(value=pd.Series(dtype=float), external=pd.Series(dtype=bool))
    df["date"] = _dates(df["date"], name)
    df["strategy"] = _check_strategy(df["strategy"], cfg, name)
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    bad = sorted(set(df["type"]) - FLOW_TYPES)
    if bad:
        raise JournalError(f"{name}: type must be one of {sorted(FLOW_TYPES)}, got {bad}")
    amt = _numbers(df["amount"], name, "amount")
    sign = df["type"].map({"deposit": 1, "withdrawal": -1, "fee": -1})
    df["value"] = np.where(sign.notna(), amt.abs() * sign.fillna(1), amt)
    df["external"] = df["type"].isin(EXTERNAL_TYPES)
    df["note"] = df["note"].fillna("").astype(str)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def load_cashflows(cfg: PortfolioConfig) -> pd.DataFrame:
    return validate_cashflows(_read(cfg.cashflows_path, CASH_COLUMNS), cfg, cfg.cashflows_path.name)


def unbalanced_transfers(cashflows: pd.DataFrame) -> pd.Series:
    """Dates whose transfer rows do not net to zero (a transfer needs a matching row in the other sleeve)."""
    t = cashflows[cashflows["type"] == "transfer"] if len(cashflows) else cashflows
    if t is None or t.empty:
        return pd.Series(dtype=float)
    net = t.groupby(t["date"].dt.normalize())["value"].sum()
    return net[net.abs() > 0.005]


def load_marks(cfg: PortfolioConfig) -> pd.DataFrame:
    df = _read(cfg.marks_path, MARK_COLUMNS)
    if df.empty:
        return df
    df["date"] = _dates(df["date"], cfg.marks_path.name)
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["price"] = _numbers(df["price"], cfg.marks_path.name, "price")
    return df


def _append(path: Path, columns: list[str], row: dict) -> None:
    new = not path.exists() or path.stat().st_size == 0
    line = pd.DataFrame([{c: row.get(c, "") for c in columns}])
    line.to_csv(path, mode="a", header=new, index=False)


def append_trade(cfg: PortfolioConfig, date, strategy: str, symbol: str, side: str, quantity: float,
                 price: float, fees: float = 0.0, note: str = "") -> dict:
    """Validate one fill and append it to ``trades.csv``."""
    side = side.strip().upper()
    if side not in SIDES:
        raise JournalError(f"side must be one of {sorted(SIDES)}")
    if strategy not in cfg.strategies and strategy != CASH_SLEEVE:
        raise JournalError(f"unknown strategy '{strategy}'")
    if not (float(quantity) > 0 and float(price) > 0):
        raise JournalError("quantity and price must be positive")
    row = dict(date=pd.Timestamp(date).strftime("%Y-%m-%d"), strategy=strategy, symbol=symbol.strip().upper(),
               side=side, quantity=float(quantity), price=float(price), fees=abs(float(fees or 0)), note=note)
    _append(cfg.trades_path, TRADE_COLUMNS, row)
    return row


def append_cashflow(cfg: PortfolioConfig, date, strategy: str, type_: str, amount: float, note: str = "") -> dict:
    type_ = type_.strip().lower()
    if type_ not in FLOW_TYPES:
        raise JournalError(f"type must be one of {sorted(FLOW_TYPES)}")
    if strategy not in cfg.strategies and strategy != CASH_SLEEVE:
        raise JournalError(f"unknown strategy '{strategy}'")
    row = dict(date=pd.Timestamp(date).strftime("%Y-%m-%d"), strategy=strategy, type=type_,
               amount=float(amount), note=note)
    _append(cfg.cashflows_path, CASH_COLUMNS, row)
    return row


def append_transfer(cfg: PortfolioConfig, date, from_strategy: str, to_strategy: str, amount: float,
                    note: str = "") -> list[dict]:
    """Move money between two sleeves: writes the two matching ``transfer`` rows."""
    amount = abs(float(amount))
    if from_strategy == to_strategy:
        raise JournalError("a transfer needs two different sleeves")
    if amount <= 0:
        raise JournalError("transfer amount must be positive")
    return [append_cashflow(cfg, date, from_strategy, "transfer", -amount, note or f"to {to_strategy}"),
            append_cashflow(cfg, date, to_strategy, "transfer", amount, note or f"from {from_strategy}")]


def _save(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    out = df[columns].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    if path.exists():                                   # keep the previous version next to it
        path.with_suffix(path.suffix + ".bak").write_bytes(path.read_bytes())
    out.to_csv(path, index=False)


def save_cashflows(cfg: PortfolioConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Validate an edited cash-flow table and overwrite ``cashflows.csv`` (previous file kept as .bak)."""
    df = df.dropna(how="all")
    for c in CASH_COLUMNS:
        if c not in df.columns:
            df[c] = "" if c == "note" else np.nan
    v = validate_cashflows(df[CASH_COLUMNS], cfg)
    _save(cfg.cashflows_path, v, CASH_COLUMNS)
    return v


def save_trades(cfg: PortfolioConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Validate an edited fills table and overwrite ``trades.csv`` (previous file kept as .bak)."""
    df = df.dropna(how="all")
    for c in TRADE_COLUMNS:
        if c not in df.columns:
            df[c] = "" if c == "note" else (0.0 if c == "fees" else np.nan)
    v = validate_trades(df[TRADE_COLUMNS], cfg)
    _save(cfg.trades_path, v, TRADE_COLUMNS)
    return v


def init_files(root: Path) -> None:
    """Create empty journal files with headers (never overwrites)."""
    for name, cols in (("trades.csv", TRADE_COLUMNS), ("cashflows.csv", CASH_COLUMNS), ("marks.csv", MARK_COLUMNS)):
        p = root / name
        if not p.exists():
            p.write_text(",".join(cols) + "\n")
