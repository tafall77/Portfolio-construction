"""``Book``: one object that loads a workspace and exposes every analysis the dashboard shows."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from . import allocation as A
from . import expectations as E
from . import metrics as M
from .backtests import load_all
from .config import CASH_SLEEVE, load_config
from .journal import load_cashflows, load_marks, load_trades, unbalanced_transfers
from .ledger import build_ledger
from .prices import PriceStore


class Book:
    def __init__(self, workspace: str | Path, end=None):
        self.cfg = load_config(workspace)
        self.trades = load_trades(self.cfg)
        self.cashflows = load_cashflows(self.cfg)
        self.marks = load_marks(self.cfg)
        self.store = PriceStore(self.cfg, self.marks)
        self.ledger = build_ledger(self.cfg, self.trades, self.cashflows, self.store, end)
        self.backtests, self._bt_warnings = load_all(self.cfg)
        self._cmp: dict = {}

    # ---- reference series --------------------------------------------------------------------
    @cached_property
    def _all_dates(self) -> pd.DatetimeIndex:
        idx = self.ledger.dates
        for bt in self.backtests.values():
            idx = idx.union(bt.index)
        return idx

    @cached_property
    def rf(self) -> pd.Series:
        return self.store.risk_free(self._all_dates) if len(self._all_dates) else pd.Series(dtype=float)

    @cached_property
    def bench(self) -> pd.Series:
        if not len(self._all_dates):
            return pd.Series(dtype=float)
        return self.store.benchmark_returns(self._all_dates.min() - pd.Timedelta(days=10))

    @property
    def strategy_ids(self) -> list[str]:
        return list(self.cfg.strategies)

    def label(self, sid: str) -> str:
        return self.cfg.label(sid)

    # ---- live --------------------------------------------------------------------------------
    def live_returns(self) -> dict[str, pd.Series]:
        L = self.ledger
        if L.empty:
            return {}
        return {s: L.returns[s].dropna() for s in L.sleeves if L.returns[s].notna().any()}

    def portfolio_returns(self) -> pd.Series:
        return self.ledger.total["ret"].dropna() if not self.ledger.empty else pd.Series(dtype=float)

    def live_start(self, sid: str | None = None) -> pd.Timestamp | None:
        r = self.portfolio_returns() if sid is None else self.live_returns().get(sid)
        if r is not None and len(r):
            return r.index[0]
        return self.cfg.inception

    # ---- expectations ------------------------------------------------------------------------
    def backtest_returns(self, sid: str) -> pd.Series | None:
        bt = self.backtests.get(sid)
        return bt["ret"] if bt is not None else None

    def expected(self, sid: str) -> pd.Series | None:
        """The backtest window used as this strategy's expectation (honest OOS, before going live)."""
        r = self.backtest_returns(sid)
        if r is None:
            return None
        return E.expectation_window(r, self.cfg.strategies[sid].expectation_start, self.live_start(sid))

    def expected_portfolio(self) -> pd.Series | None:
        """Target-weighted mix of the strategies' expectation windows (common dates, configured rebalance)."""
        tgt = self.cfg.target_weights()
        series = {s: self.expected(s) for s in tgt.index if self.expected(s) is not None and tgt[s] > 0}
        if not series:
            return None
        R = A.align(series)
        if len(R) < 60:
            return None
        return A.portfolio_returns(R, tgt.reindex(R.columns) / tgt.reindex(R.columns).sum(),
                                   self.cfg.allocation.rebalance)

    def comparison(self, sid: str | None = None) -> E.Comparison | None:
        key = sid or "__portfolio__"
        if key not in self._cmp:
            live = self.portfolio_returns() if sid is None else self.live_returns().get(sid)
            exp = self.expected_portfolio() if sid is None else self.expected(sid)
            self._cmp[key] = (E.compare(live, exp, self.rf, self.rf)
                              if live is not None and exp is not None else None)
        return self._cmp[key]

    def model_on_live_dates(self, sid: str) -> pd.Series | None:
        bt = self.backtest_returns(sid)
        live = self.live_returns().get(sid)
        if bt is None or live is None or not len(live):
            return None
        m = bt.loc[live.index[0]:]
        return m if len(m) else None

    def tracking(self, sid: str) -> dict | None:
        m = self.model_on_live_dates(sid)
        return E.tracking(self.live_returns()[sid], m) if m is not None else None

    def health(self) -> list[dict]:
        return E.health_checks(self)

    def data_warnings(self) -> list[str]:
        out = list(dict.fromkeys(self._bt_warnings + self.ledger.warnings + self.store.warnings))
        for d, v in unbalanced_transfers(self.cashflows).items():
            out.append(f"Transfers on {d:%Y-%m-%d} net to {v:+,.2f} instead of 0: each transfer needs a matching "
                       "row in the other sleeve (use 'Move money between strategies' in Transactions).")
        for sid in self.cfg.strategies:
            bt, live = self.backtest_returns(sid), self.live_returns().get(sid)
            if bt is not None and live is not None and len(live) and bt.index[-1] < live.index[0]:
                out.append(f"{self.label(sid)}: backtest ends {bt.index[-1]:%Y-%m-%d}, before live trading began; "
                           "re-export it to enable live-vs-model tracking.")
        return out

    # ---- allocation --------------------------------------------------------------------------
    def strategy_returns(self, source: str = "backtest", strategies: list[str] | None = None) -> dict[str, pd.Series]:
        """Per-strategy daily returns for portfolio construction.

        ``backtest``: the whole exported backtest (the common window is then cut by ``allocation.start``).
        ``backtest+live``: the backtest up to the day before going live, then the live record.
        ``live``: the live record only.
        """
        out = {}
        for sid in strategies or self.strategy_ids:
            bt = self.backtest_returns(sid)
            live = self.live_returns().get(sid)
            if source == "live":
                if live is not None and len(live):
                    out[sid] = live
                continue
            if bt is None:
                continue
            if source == "backtest+live" and live is not None and len(live):
                bt = pd.concat([bt.loc[:live.index[0] - pd.Timedelta(days=1)], live])
            out[sid] = bt
        return out

    def alloc_matrix(self, source: str = "backtest", strategies: list[str] | None = None, start=None,
                     end=None) -> pd.DataFrame:
        a = self.cfg.allocation
        return A.align(self.strategy_returns(source, strategies), start or a.start, end or a.end)

    def rebalance_orders(self, target: pd.Series | None = None) -> pd.DataFrame:
        if self.ledger.empty:
            return pd.DataFrame()
        tgt = self.cfg.target_weights() if target is None else target
        nav = self.ledger.nav.iloc[-1].drop(labels=[CASH_SLEEVE], errors="ignore")
        unalloc = self.ledger.nav.iloc[-1].get(CASH_SLEEVE, 0.0)
        tgt = tgt.reindex(nav.index).fillna(0.0)
        df = A.rebalance_orders(nav, tgt, {s: self.label(s) for s in nav.index})
        if unalloc:
            # unallocated cash is deployed in proportion to the targets
            total = nav.sum() + unalloc
            df["Target $"] = tgt.to_numpy() * total
            df["Transfer $"] = df["Target $"] - df["Current $"]
            df["Current weight"] = df["Current $"] / total
        return df

    # ---- capital ---------------------------------------------------------------------------
    def capital_summary(self) -> pd.DataFrame:
        """Money in and out per sleeve and for the whole book, with money- and time-weighted returns."""
        L, cf = self.ledger, self.cashflows
        if L.empty:
            return pd.DataFrame()
        asof = L.dates[-1]
        rows = {}
        for s in L.sleeves + ["__total__"]:
            g = cf if s == "__total__" else cf[cf["strategy"] == s]
            v = lambda typ, sign=None: g.loc[(g["type"] == typ) & ((g["value"] > 0) if sign == "+" else
                                                                   (g["value"] < 0) if sign == "-" else True),
                                             "value"].sum() if len(g) else 0.0
            nav = L.total["nav"].iloc[-1] if s == "__total__" else L.nav[s].iloc[-1]
            ext = g[g["external"]] if len(g) else g
            if s == "__total__" and len(ext):          # transfers between sleeves cancel out for the book
                ext = ext[ext["type"] != "transfer"]
            net_in = ext["value"].sum() if len(ext) else 0.0
            inc = L.income.sum() if s == "__total__" else L.income[s]
            inc_total = float(inc.sum())
            cf_fees = v("fee")
            trade_fees = float((L.fees.sum() if s == "__total__" else L.fees[s]).sum())
            r = self.portfolio_returns() if s == "__total__" else self.live_returns().get(s, pd.Series(dtype=float))
            flows = ext.groupby(ext["date"].dt.normalize())["value"].sum() if len(ext) else pd.Series(dtype=float)
            mwr = M.xirr(list(flows.index) + [asof], list(-flows.to_numpy()) + [nav])
            yrs = ((asof - flows.index.min()).days / 365.0) if len(flows) else np.nan
            rows["Total" if s == "__total__" else self.label(s)] = {
                "Deposited": v("deposit"), "Withdrawn": abs(v("withdrawal")),
                "Transfers in": v("transfer", "+"), "Transfers out": abs(v("transfer", "-")),
                "Net invested": net_in, "NAV": nav, "P&L": nav - net_in,
                "Dividends & interest": inc_total - cf_fees, "Fees": trade_fees - cf_fees,
                "Money-weighted return": (1 + mwr) ** yrs - 1 if np.isfinite(mwr) and yrs > 0 else np.nan,
                "Money-weighted (ann.)": mwr,
                "Time-weighted return": (1 + r).prod() - 1 if len(r) else np.nan,
                "First flow": flows.index.min() if len(flows) else pd.NaT}
        return pd.DataFrame(rows).T

    def flows_by_period(self, freq: str = "M") -> pd.DataFrame:
        """Deposits, withdrawals and net new money per period for the whole book (transfers excluded)."""
        cf = self.cashflows
        if cf is None or cf.empty:
            return pd.DataFrame()
        ext = cf[cf["type"].isin(["deposit", "withdrawal"])]
        if ext.empty:
            return pd.DataFrame()
        key = ext["date"].dt.to_period(freq)
        df = pd.DataFrame({"Deposits": ext["value"].where(ext["type"] == "deposit", 0.0).groupby(key).sum(),
                           "Withdrawals": ext["value"].where(ext["type"] == "withdrawal", 0.0).groupby(key).sum()})
        full = pd.period_range(df.index.min(), max(df.index.max(), pd.Timestamp.today().to_period(freq)), freq=freq)
        df = df.reindex(full, fill_value=0.0)
        df["Net new money"] = df["Deposits"] + df["Withdrawals"]
        df["Cumulative net invested"] = df["Net new money"].cumsum()
        return df

    def net_invested(self, sid: str | None = None) -> pd.Series:
        """Cumulative external flows on the ledger's dates (book level: transfers cancel)."""
        L = self.ledger
        if L.empty:
            return pd.Series(dtype=float)
        f = L.flows.sum(axis=1) if sid is None else L.flows[sid]
        return f.cumsum()

    # ---- summaries ---------------------------------------------------------------------------
    def kpis(self) -> dict:
        L = self.ledger
        if L.empty:
            return {}
        t = L.total
        r = self.portfolio_returns()
        nav = t["nav"].iloc[-1]
        today = t.index[-1]
        def ret_since(d):
            x = r.loc[d:]
            return (1 + x).prod() - 1 if len(x) else np.nan
        flows = t["flows"].sum()
        return {"NAV": nav, "Net invested": flows, "P&L (ITD)": nav - flows,
                "Day P&L": t["pnl"].iloc[-1], "Day return": r.iloc[-1] if len(r) else np.nan,
                "MTD": ret_since(today.replace(day=1)), "YTD": ret_since(today.replace(month=1, day=1)),
                "ITD (TWR)": (1 + r).prod() - 1 if len(r) else np.nan, "As of": today,
                "Gross exposure": (t["long"].iloc[-1] - t["short"].iloc[-1]) / nav if nav else np.nan,
                "Net exposure": (t["long"].iloc[-1] + t["short"].iloc[-1]) / nav if nav else np.nan,
                "Cash": t["cash"].iloc[-1]}

    def live_summary(self) -> pd.DataFrame:
        series = {"Portfolio": self.portfolio_returns(), **{self.label(s): r for s, r in self.live_returns().items()}}
        series = {k: v for k, v in series.items() if len(v) > 1}
        if not series:
            return pd.DataFrame()
        return M.summary_frame(series, self.rf, self.bench)

    def trade_summary(self) -> pd.DataFrame:
        rt = self.ledger.round_trips
        if rt is None or rt.empty:
            return pd.DataFrame()
        cols = {"All": M.trade_summary(rt)}
        for sid, g in rt.groupby("strategy"):
            cols[self.label(sid)] = M.trade_summary(g)
        return pd.DataFrame(cols)

    def attribution(self, start=None, end=None) -> pd.DataFrame:
        """P&L by position over a window (price + dividends, fees shown separately per sleeve)."""
        pp = self.ledger.position_pnl
        if pp is None or pp.empty:
            return pd.DataFrame()
        s = pp.loc[start:end].sum()
        df = s.rename("P&L").reset_index()
        df["strategy"] = df["sleeve"].map(self.label)
        return df.sort_values("P&L")
