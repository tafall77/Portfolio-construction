"""Workspace configuration (``portfolio.yaml``).

A *workspace* is one folder holding everything about one book:

    portfolio.yaml      strategies, target weights, benchmark, settings (this module)
    trades.csv          every fill you executed, tagged with the strategy it belongs to
    cashflows.csv       deposits, withdrawals, transfers between sleeves, dividends, interest, fees
    marks.csv           optional manual prices (futures rolls, delisted names, OTC)
    backtests/*.csv     daily return series exported from each research notebook
    prices/*.csv        optional local price files (used instead of Yahoo, e.g. the demo)
    cache/              downloaded prices (safe to delete)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]
CASH_SLEEVE = "cash"          # reserved strategy id for money that is not allocated to any strategy


@dataclass
class StrategyConfig:
    id: str
    name: str
    short: str
    color: str
    target_weight: float = 0.0
    description: str = ""
    universe: str = ""
    backtest: Path | None = None            # CSV with date,return[,exposure]
    expectation_start: pd.Timestamp | None = None   # first date of the honest out-of-sample record
    active: bool = True                     # False = tracked but excluded from target allocation


@dataclass
class InstrumentConfig:
    symbol: str
    multiplier: float = 1.0
    type: str = "equity"                    # equity | future
    yahoo: str | None = None                # Yahoo ticker if it differs from the journal symbol


@dataclass
class AllocationConfig:
    start: pd.Timestamp | None = None       # analysis window start (None = latest strategy start)
    end: pd.Timestamp | None = None
    rebalance: str = "M"                    # D, W, M, Q, A or none
    min_weight: float = 0.0
    max_weight: float = 1.0
    walk_forward_lookback_years: float = 3.0
    walk_forward_step: str = "Q"
    bootstrap_samples: int = 300


@dataclass
class AlertConfig:
    drawdown_percentile: float = 0.95       # live drawdown worse than this share of backtest paths -> alert
    return_percentile: float = 0.05         # live return below this percentile of the expectation cone -> alert
    vol_ratio: float = 1.5                  # live vol / expected vol above this -> alert
    weight_drift: float = 0.05              # |actual - target| sleeve weight above this -> rebalance alert
    tracking_error: float = 0.05            # annualised live-vs-model tracking error above this -> alert
    correlation: float = 0.8                # trailing 63-day correlation between sleeves above this -> alert


@dataclass
class PortfolioConfig:
    root: Path
    name: str = "Systematic portfolio"
    currency: str = "USD"
    benchmark: str = "SPY"
    risk_free: str | float = "^IRX"         # Yahoo T-bill yield ticker, or a constant annual rate
    inception: pd.Timestamp | None = None   # None = first cash flow
    auto_dividends: bool = True             # credit dividends from Yahoo on ex-date (turn off if you log them)
    offline: bool = False                   # never download; use cache / local files only
    price_max_age_hours: float = 6.0
    demo: bool = False
    strategies: dict[str, StrategyConfig] = field(default_factory=dict)
    instruments: dict[str, InstrumentConfig] = field(default_factory=dict)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    # ---- paths -----------------------------------------------------------------------------------
    @property
    def trades_path(self) -> Path:
        return self.root / "trades.csv"

    @property
    def cashflows_path(self) -> Path:
        return self.root / "cashflows.csv"

    @property
    def marks_path(self) -> Path:
        return self.root / "marks.csv"

    @property
    def prices_dir(self) -> Path:
        return self.root / "prices"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    # ---- helpers ---------------------------------------------------------------------------------
    def instrument(self, symbol: str) -> InstrumentConfig:
        return self.instruments.get(symbol) or InstrumentConfig(symbol=symbol)

    def target_weights(self) -> pd.Series:
        w = pd.Series({k: s.target_weight for k, s in self.strategies.items() if s.active}, dtype=float)
        return w / w.sum() if w.sum() > 0 else w

    def label(self, sid: str) -> str:
        if sid in self.strategies:
            return self.strategies[sid].short
        return "Unallocated" if sid == CASH_SLEEVE else sid

    def color(self, sid: str) -> str:
        if sid in self.strategies:
            return self.strategies[sid].color
        return "#898781"


def _ts(v) -> pd.Timestamp | None:
    if v is None or v == "":
        return None
    return pd.Timestamp(v)


def load_config(path: str | Path) -> PortfolioConfig:
    """Read ``portfolio.yaml``. ``path`` may be the file or the workspace folder."""
    path = Path(path)
    if path.is_dir():
        path = path / "portfolio.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No portfolio.yaml at {path}. Run `python -m pcon init` to create a workspace.")
    raw = yaml.safe_load(path.read_text()) or {}
    root = path.parent.resolve()
    p = raw.get("portfolio", {}) or {}

    strategies = {}
    for k, (sid, s) in enumerate((raw.get("strategies") or {}).items()):
        if sid == CASH_SLEEVE:
            raise ValueError(f"'{CASH_SLEEVE}' is reserved for unallocated cash; pick another strategy id")
        s = s or {}
        bt = s.get("backtest")
        strategies[sid] = StrategyConfig(
            id=sid, name=s.get("name", sid), short=s.get("short", s.get("name", sid)),
            color=s.get("color", DEFAULT_COLORS[k % len(DEFAULT_COLORS)]),
            target_weight=float(s.get("target_weight", 0.0) or 0.0), description=s.get("description", ""),
            universe=s.get("universe", ""), backtest=(root / bt) if bt else None,
            expectation_start=_ts(s.get("expectation_start")), active=bool(s.get("active", True)))

    instruments = {}
    for sym, spec in (raw.get("instruments") or {}).items():
        spec = spec or {}
        instruments[sym] = InstrumentConfig(symbol=sym, multiplier=float(spec.get("multiplier", 1.0)),
                                            type=spec.get("type", "equity"), yahoo=spec.get("yahoo"))

    a = raw.get("allocation", {}) or {}
    alloc = AllocationConfig(
        start=_ts(a.get("start")), end=_ts(a.get("end")), rebalance=str(a.get("rebalance", "M")),
        min_weight=float(a.get("min_weight", 0.0)), max_weight=float(a.get("max_weight", 1.0)),
        walk_forward_lookback_years=float(a.get("walk_forward_lookback_years", 3.0)),
        walk_forward_step=str(a.get("walk_forward_step", "Q")),
        bootstrap_samples=int(a.get("bootstrap_samples", 300)))
    al = raw.get("alerts", {}) or {}
    alerts = AlertConfig(**{k: float(v) for k, v in al.items() if k in AlertConfig.__dataclass_fields__})

    rf = p.get("risk_free", "^IRX")
    return PortfolioConfig(
        root=root, name=p.get("name", "Systematic portfolio"), currency=p.get("currency", "USD"),
        benchmark=p.get("benchmark", "SPY"), risk_free=float(rf) if isinstance(rf, (int, float)) else str(rf),
        inception=_ts(p.get("inception")), auto_dividends=bool(p.get("auto_dividends", True)),
        offline=bool(p.get("offline", False)), price_max_age_hours=float(p.get("price_max_age_hours", 6.0)),
        demo=bool(p.get("demo", False)), strategies=strategies, instruments=instruments, allocation=alloc,
        alerts=alerts)
