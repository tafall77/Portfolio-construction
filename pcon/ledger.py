"""Rebuild the book from the journal: positions, cash, NAV and returns for every strategy sleeve.

Each strategy is a *sleeve*: its own cash account and positions, funded with deposits / transfers.
That is what lets the dashboard give every strategy an honest stand-alone track record while the
total portfolio is simply the sum of the sleeves.

Accounting conventions
* Fills are booked on their trade date (snapped forward to the next trading day if needed).
* Quantities and prices are restated into today's share units with the split history, because
  market data is split-adjusted for the whole history.
* Futures (``type: future`` in portfolio.yaml) are booked at notional: buying debits
  ``qty x multiplier x price`` from cash and the position is worth ``qty x multiplier x close``. NAV is
  therefore identical to margin accounting (cash + realised + unrealised P&L), and exposure is notional.
* Dividends are credited on the ex-date to positions held at the previous close (``auto_dividends``).
* Returns are time-weighted: external flows (deposits, withdrawals, transfers) are assumed to arrive at
  the start of the day, ``r_t = NAV_t / (NAV_{t-1} + flow_t) - 1``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CASH_SLEEVE, PortfolioConfig
from .prices import PriceStore, split_factor_after

MIN_CAPITAL = 1e-6


@dataclass
class Ledger:
    dates: pd.DatetimeIndex
    sleeves: list[str]
    nav: pd.DataFrame                 # dates x sleeves
    cash: pd.DataFrame                # dates x sleeves (futures notional excluded, i.e. margin-style cash)
    flows: pd.DataFrame               # external flows (deposits/withdrawals/transfers)
    income: pd.DataFrame              # dividends + interest - non-trade fees
    fees: pd.DataFrame                # trading fees
    pnl: pd.DataFrame                 # NAV change excluding external flows
    returns: pd.DataFrame             # time-weighted daily returns (NaN before a sleeve is funded)
    long_exposure: pd.DataFrame
    short_exposure: pd.DataFrame
    n_positions: pd.DataFrame
    positions: pd.DataFrame           # dates x (sleeve, symbol) quantity in today's units
    values: pd.DataFrame              # dates x (sleeve, symbol) market value / notional
    position_pnl: pd.DataFrame        # dates x (sleeve, symbol) price P&L + dividends (fees excluded)
    prices: pd.DataFrame              # dates x symbols marks used
    total: pd.DataFrame               # nav, flows, pnl, ret, long, short for the whole book
    trades: pd.DataFrame              # journal fills with adjusted units and notional
    round_trips: pd.DataFrame
    open_positions: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return len(self.dates) == 0

    def weights(self) -> pd.DataFrame:
        """Sleeve weights in total NAV."""
        tot = self.nav.sum(axis=1)
        return self.nav.div(tot.where(tot.abs() > MIN_CAPITAL), axis=0)


def empty_ledger(warnings: list[str] | None = None) -> Ledger:
    e = pd.DataFrame()
    idx = pd.DatetimeIndex([])
    return Ledger(idx, [], e, e, e, e, e, e, e, e, e, e, e, e, e, e,
                  pd.DataFrame(columns=["nav", "flows", "pnl", "ret", "long", "short"]), e, e, e, warnings or [])


def _snap(calendar: pd.DatetimeIndex, dates) -> np.ndarray:
    pos = calendar.searchsorted(pd.DatetimeIndex(dates).normalize(), side="left")
    return np.minimum(pos, len(calendar) - 1)


def twr_returns(nav: pd.Series, flows: pd.Series) -> pd.Series:
    """Daily time-weighted returns with start-of-day external flows."""
    base = nav.shift(1).fillna(0.0) + flows
    r = nav / base.where(base > MIN_CAPITAL) - 1
    return r


def build_ledger(cfg: PortfolioConfig, trades: pd.DataFrame, cashflows: pd.DataFrame, store: PriceStore,
                 end=None) -> Ledger:
    warn: list[str] = []
    if trades.empty and cashflows.empty:
        return empty_ledger(["The journal is empty: add a deposit and your first fills."])
    first = min([d for d in (trades["date"].min() if len(trades) else None,
                             cashflows["date"].min() if len(cashflows) else None) if d is not None])
    if cfg.inception is not None:
        first = min(first, cfg.inception)
    end = pd.Timestamp(end or pd.Timestamp.today()).normalize()
    last_event = max([d for d in (trades["date"].max() if len(trades) else None,
                                  cashflows["date"].max() if len(cashflows) else None) if d is not None])
    if last_event.normalize() > end:
        warn.append(f"Journal has entries dated after {end:%Y-%m-%d}; they are booked on the last day shown.")
    cal = store.calendar(first, end)
    if len(cal) == 0:
        return empty_ledger(["No trading days in range."])
    if cal[0] > pd.Timestamp(first).normalize():
        cal = pd.DatetimeIndex([pd.Timestamp(first).normalize()]).append(cal)
    n = len(cal)

    sleeves = [s for s in cfg.strategies] + ([CASH_SLEEVE] if (
        (trades["strategy"] == CASH_SLEEVE).any() if len(trades) else False) or (
        (cashflows["strategy"] == CASH_SLEEVE).any() if len(cashflows) else False) else [])

    # ---- market data, split restatement --------------------------------------------------------
    tr = trades.copy()
    symbols = sorted(tr["symbol"].unique()) if len(tr) else []
    hist = {s: store.history(s, first - pd.Timedelta(days=10)) for s in symbols}
    mult = {s: cfg.instrument(s).multiplier for s in symbols}
    is_future = {s: cfg.instrument(s).type == "future" for s in symbols}
    if len(tr):
        factor = np.ones(len(tr))
        for s in symbols:
            h = hist[s]
            if h is not None and not is_future[s]:
                m = (tr["symbol"] == s).to_numpy()
                factor[m] = split_factor_after(h["splits"], tr.loc[m, "date"])
        tr["split_factor"] = factor
        tr["qty_adj"] = tr["qty"] * factor
        tr["price_adj"] = tr["price"] / factor
        tr["mult"] = tr["symbol"].map(mult)
        tr["notional"] = tr["qty_adj"] * tr["price_adj"] * tr["mult"]
        tr["i"] = _snap(cal, tr["date"])
        tr["book_date"] = cal[tr["i"].to_numpy()]

    P = pd.DataFrame(index=cal, columns=symbols, dtype=float)
    D = pd.DataFrame(0.0, index=cal, columns=symbols)
    for s in symbols:
        h = hist[s]
        if h is not None and len(h):
            c = h["close"]
            P[s] = c.reindex(c.index.union(cal)).ffill().reindex(cal)
            if cfg.auto_dividends and not is_future[s]:
                dv = h["dividends"][h["dividends"] > 0]
                dv = dv[(dv.index >= cal[0]) & (dv.index <= cal[-1])]
                if len(dv):
                    D[s] = pd.Series(dv.to_numpy(), index=cal[_snap(cal, dv.index)]).groupby(level=0).sum() \
                        .reindex(cal, fill_value=0.0)
        else:
            warn.append(f"No market prices for {s}: marked at your fill prices (add closes to marks.csv)")
        # fill gaps (before data starts / no data) with the latest fill price
        fills = tr[tr["symbol"] == s].groupby("i")["price_adj"].last()
        fill_px = pd.Series(fills.to_numpy(), index=cal[fills.index.to_numpy()])
        P[s] = P[s].combine_first(fill_px.reindex(cal)).ffill()

    # ---- positions and values per (sleeve, symbol) --------------------------------------------
    keys = sorted(set(zip(tr["strategy"], tr["symbol"]))) if len(tr) else []
    cols = pd.MultiIndex.from_tuples(keys, names=["sleeve", "symbol"]) if keys else \
        pd.MultiIndex.from_tuples([], names=["sleeve", "symbol"])
    Q = pd.DataFrame(0.0, index=cal, columns=cols)
    trade_val = pd.DataFrame(0.0, index=cal, columns=cols)       # + qty*price*mult bought that day
    for (sl, sym), g in tr.groupby(["strategy", "symbol"]) if len(tr) else []:
        q = g.groupby("i")["qty_adj"].sum()
        v = g.groupby("i")["notional"].sum()
        Q.iloc[q.index.to_numpy(), Q.columns.get_loc((sl, sym))] = q.to_numpy()
        trade_val.iloc[v.index.to_numpy(), trade_val.columns.get_loc((sl, sym))] = v.to_numpy()
    Q = Q.cumsum()
    Q = Q.where(Q.abs() > 1e-9, 0.0)
    sym_of = [k[1] for k in keys]
    px = P[sym_of].to_numpy() if keys else np.zeros((n, 0))
    m_arr = np.array([mult[s] for s in sym_of]) if keys else np.zeros(0)
    V = pd.DataFrame(Q.to_numpy() * px * m_arr, index=cal, columns=cols)
    div_income = pd.DataFrame(Q.shift(1).fillna(0.0).to_numpy() * (D[sym_of].to_numpy() if keys else px) * m_arr,
                              index=cal, columns=cols) if keys else pd.DataFrame(index=cal, columns=cols)
    pos_pnl = V.diff().fillna(V) - trade_val + div_income

    # ---- cash per sleeve -----------------------------------------------------------------------
    def per_sleeve(df_rows: pd.DataFrame, value_col: str) -> pd.DataFrame:
        out = pd.DataFrame(0.0, index=cal, columns=sleeves)
        if len(df_rows):
            i = _snap(cal, df_rows["date"])
            g = pd.DataFrame({"i": i, "s": df_rows["strategy"].to_numpy(), "v": df_rows[value_col].to_numpy()})
            g = g.groupby(["i", "s"])["v"].sum()
            for (ii, s), v in g.items():
                out.iat[ii, out.columns.get_loc(s)] += v
        return out

    ext = cashflows[cashflows["external"]] if len(cashflows) else cashflows
    inc = cashflows[~cashflows["external"]] if len(cashflows) else cashflows
    F = per_sleeve(ext, "value")
    INC = per_sleeve(inc, "value")
    fees = per_sleeve(tr.assign(fee_v=tr["fees"]) if len(tr) else tr, "fee_v") if len(tr) else \
        pd.DataFrame(0.0, index=cal, columns=sleeves)
    tcash = per_sleeve(tr.assign(tv=-tr["notional"]) if len(tr) else tr, "tv") if len(tr) else \
        pd.DataFrame(0.0, index=cal, columns=sleeves)
    DIV = (div_income.T.groupby(level="sleeve").sum().T.reindex(columns=sleeves, fill_value=0.0)
           if keys else pd.DataFrame(0.0, index=cal, columns=sleeves))
    cash_ledger = (F + INC + DIV + tcash - fees).cumsum()
    by_sleeve = lambda df: df.T.groupby(level="sleeve").sum().T.reindex(columns=sleeves, fill_value=0.0) \
        if keys else pd.DataFrame(0.0, index=cal, columns=sleeves)
    MV = by_sleeve(V)
    nav = cash_ledger + MV
    fut_cols = [k for k in keys if is_future[k[1]]]
    fut_notional = by_sleeve(V[fut_cols]) if fut_cols else pd.DataFrame(0.0, index=cal, columns=sleeves)
    cash = nav - (MV - fut_notional)

    longv = by_sleeve(V.clip(lower=0))
    shortv = by_sleeve(V.clip(upper=0))
    npos = by_sleeve((Q != 0).astype(float))
    pnl = nav.diff().fillna(nav) - F
    rets = pd.DataFrame({s: twr_returns(nav[s], F[s]) for s in sleeves}, index=cal)

    tot_nav, tot_flow = nav.sum(axis=1), F.sum(axis=1)
    total = pd.DataFrame({"nav": tot_nav, "flows": tot_flow, "pnl": pnl.sum(axis=1),
                          "ret": twr_returns(tot_nav, tot_flow), "long": longv.sum(axis=1),
                          "short": shortv.sum(axis=1), "cash": cash.sum(axis=1)})

    for s in sleeves:
        if ((nav[s] <= MIN_CAPITAL) & (MV[s].abs() > 0)).any():
            warn.append(f"{cfg.label(s)}: NAV is zero or negative while holding positions. "
                        "Did you record the deposit / transfer that funds this sleeve?")

    rt, op = round_trips(tr, cal, P, nav, mult)
    trades_out = tr.drop(columns=["i"], errors="ignore") if len(tr) else tr
    return Ledger(cal, sleeves, nav, cash, F, INC + DIV, fees, pnl, rets, longv, shortv, npos, Q, V, pos_pnl, P,
                  total, trades_out, rt, op, warn + store.warnings)


def round_trips(tr: pd.DataFrame, cal: pd.DatetimeIndex, P: pd.DataFrame, nav: pd.DataFrame,
                mult: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Position episodes (flat -> flat) per sleeve and symbol, plus the open ones.

    P&L is price P&L net of fees (dividends are in the sleeve NAV but not in trade P&L). Average cost:
    adds move the average, reductions realise P&L against it. A fill that flips the position is split.
    """
    closed, open_ = [], []
    if tr is None or tr.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (sl, sym), g in tr.groupby(["strategy", "symbol"], sort=True):
        m = mult[sym]
        ep = None
        pos = 0.0
        for row in g.itertuples(index=False):
            q, p, fee, i = row.qty_adj, row.price_adj, row.fees, row.i
            legs = [(q, fee)]
            if pos != 0 and np.sign(q) != np.sign(pos) and abs(q) > abs(pos) + 1e-12:
                legs = [(-pos, fee * abs(pos) / abs(q)), (q + pos, fee * (abs(q) - abs(pos)) / abs(q))]
            for lq, lf in legs:
                if abs(pos) < 1e-12:
                    ep = dict(strategy=sl, symbol=sym, direction="long" if lq > 0 else "short", entry_i=i,
                              entry_date=row.book_date, avg_cost=0.0, entry_qty=0.0, entry_value=0.0,
                              exit_qty=0.0, exit_value=0.0, realized=0.0, fees=0.0, fills=0, max_qty=0.0)
                    pos = 0.0
                d = 1.0 if ep["direction"] == "long" else -1.0
                ep["fees"] += lf
                ep["fills"] += 1
                if np.sign(lq) == d:                                   # adding
                    ep["avg_cost"] = (ep["avg_cost"] * abs(pos) + p * abs(lq)) / (abs(pos) + abs(lq))
                    ep["entry_qty"] += abs(lq)
                    ep["entry_value"] += abs(lq) * p * m
                else:                                                  # reducing
                    ep["realized"] += (p - ep["avg_cost"]) * abs(lq) * d * m
                    ep["exit_qty"] += abs(lq)
                    ep["exit_value"] += abs(lq) * p * m
                pos += lq
                ep["max_qty"] = max(ep["max_qty"], abs(pos))
                if abs(pos) < 1e-9:
                    pos = 0.0
                    pnl = ep["realized"] - ep["fees"]
                    closed.append(dict(
                        strategy=sl, symbol=sym, direction=ep["direction"], entry_date=ep["entry_date"],
                        exit_date=row.book_date, quantity=ep["max_qty"],
                        entry_price=ep["entry_value"] / (ep["entry_qty"] * m),
                        exit_price=ep["exit_value"] / (ep["exit_qty"] * m), pnl=pnl,
                        ret=pnl / ep["entry_value"] if ep["entry_value"] else np.nan, fees=ep["fees"],
                        days=(row.book_date - ep["entry_date"]).days, bars=int(i - ep["entry_i"]),
                        fills=ep["fills"]))
                    ep = None
        if ep is not None and abs(pos) > 0:
            last = P[sym].iloc[-1]
            d = 1.0 if ep["direction"] == "long" else -1.0
            unreal = (last - ep["avg_cost"]) * abs(pos) * d * m
            sleeve_nav = nav[sl].iloc[-1]
            open_.append(dict(
                strategy=sl, symbol=sym, direction=ep["direction"], quantity=pos, avg_cost=ep["avg_cost"],
                last=last, market_value=pos * last * m, unrealized=unreal,
                unrealized_pct=unreal / (ep["avg_cost"] * abs(pos) * m) if ep["avg_cost"] else np.nan,
                realized=ep["realized"] - ep["fees"], entry_date=ep["entry_date"],
                days=(cal[-1] - ep["entry_date"]).days, bars=int(len(cal) - 1 - ep["entry_i"]),
                weight_sleeve=pos * last * m / sleeve_nav if abs(sleeve_nav) > MIN_CAPITAL else np.nan))
    return pd.DataFrame(closed), pd.DataFrame(open_)
