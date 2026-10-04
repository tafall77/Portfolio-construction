"""Today's orders: each strategy's current signal, sized to the capital the allocation gives it.

The research notebooks decide *what* each strategy holds; their export cells write it to
``<workspace>/signals/<strategy>.json`` (see ``strategy_exports/``). This module turns those signals into
share quantities for your account:

    capital_s  = account NAV x invested share x allocation weight of strategy s
    orders_s   = what strategy s should hold with capital_s  -  what its sleeve holds now

Two signal kinds:

``weights``      target weight per symbol, rest in cash (regime filter: S&P / Nasdaq-100 / cash; rolling momentum:
                 S&P long or cash). Rebalanced to target only when the signal changes or the sleeve has drifted
                 more than ``band`` from it, as the backtest does (it does not trade between scheduled fills).
``stock_picks``  the SMA / Piotroski account: positions are sized once at entry and never rebalanced, so the
                 orders are the model's next-open exits and entries, plus resting take-profit limits.

Market orders of the same timing are netted across strategies (one SPY order instead of a buy and a sell).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PortfolioConfig

KINDS = ("weights", "stock_picks")
WEIGHT_BAND = 0.10          # 'weights' sleeves: trade when a weight is this far from target (or the signal flips)
ORDER_COLUMNS = ["strategy", "symbol", "side", "quantity", "order", "limit", "price", "value", "reason"]


@dataclass
class Plan:
    """Orders and context for one strategy."""
    strategy: str
    capital: float
    orders: pd.DataFrame
    targets: pd.DataFrame                      # what the sleeve should hold vs what it holds
    notes: list[str] = field(default_factory=list)
    status: str = ""                           # one-line summary for the overview table


# ---- signal files --------------------------------------------------------------------------------
def signals_dir(cfg: PortfolioConfig) -> Path:
    return cfg.root / "signals"


def load_signals(cfg: PortfolioConfig) -> tuple[dict[str, dict], list[str]]:
    """``{strategy: signal}`` for every ``signals/<strategy>.json`` of a configured strategy, plus warnings."""
    out, warn = {}, []
    d = signals_dir(cfg)
    for sid in cfg.strategies:
        p = d / f"{sid}.json"
        if not p.exists():
            continue
        try:
            sig = json.loads(p.read_text())
            if sig.get("kind") not in KINDS:
                raise ValueError(f"unknown kind {sig.get('kind')!r} (expected one of {KINDS})")
            sig["as_of"] = pd.Timestamp(sig["as_of"])
            if sig["kind"] == "weights":
                sig["weights"] = {str(k): float(v) for k, v in (sig.get("weights") or {}).items()}
                if sig.get("next_weights") is not None:
                    sig["next_weights"] = {str(k): float(v) for k, v in sig["next_weights"].items()}
                for w in (sig["weights"], sig.get("next_weights") or {}):
                    if sum(w.values()) > 1 + 1e-6 or min(w.values(), default=0) < 0:
                        raise ValueError("weights must be >= 0 and sum to at most 1 (the rest is cash)")
            out[sid] = sig
        except Exception as exc:  # a broken file must not take the dashboard down
            warn.append(f"signals/{sid}.json could not be read: {exc}")
    return out, warn


def staleness(sig: dict, today=None) -> int:
    """Trading days between the signal's close and the last completed trading day."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    last_close = today - pd.offsets.BDay(1)        # the last close before today's session
    return max(0, len(pd.bdate_range(sig["as_of"] + pd.Timedelta(days=1), last_close)))


def is_stale(sig: dict, today=None) -> bool:
    """Daily signals go stale after one trading day. A monthly signal that carries its next trade date stays
    valid until the month after that date (re-run the notebook once the new month's data is out)."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    if sig.get("next_trade"):
        nt = pd.Timestamp(sig["next_trade"])
        return today >= (nt + pd.offsets.MonthBegin(1))
    return staleness(sig, today) > 1


def current_weights(sig: dict, today=None) -> tuple[dict, str | None]:
    """The weights to hold today, and a note on a scheduled change (monthly signals)."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    w, nw, nt = sig["weights"], sig.get("next_weights"), sig.get("next_trade")
    if nt is None or nw is None:
        return w, None
    fmt = lambda x: ", ".join(f"{k} {v:.0%}" for k, v in x.items() if v > 0) or "all cash"
    if today >= pd.Timestamp(nt):
        return nw, f"Monthly trade date ({nt}): move to {fmt(nw)}."
    if nw != w:
        return w, f"Scheduled: on {nt} the model moves from {fmt(w)} to {fmt(nw)}. Until then, hold."
    return w, f"Next monthly trade date {nt}: no change planned ({fmt(nw)})."


# ---- helpers -------------------------------------------------------------------------------------
def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ORDER_COLUMNS)


def _price(sym: str, prices: dict, fallback=None) -> float:
    p = prices.get(sym)
    if p is None or not np.isfinite(p) or p <= 0:
        p = fallback
    return float(p) if p is not None and np.isfinite(p) and p > 0 else float("nan")


def _qty(value: float, price: float, mult: float, futures: bool) -> float:
    if not np.isfinite(price):
        return float("nan")
    q = value / (price * mult)
    return float(round(q) if futures else math.floor(q + 1e-9))


# ---- weights sleeves -----------------------------------------------------------------------------
def plan_weights(sid: str, sig: dict, capital: float, positions: dict[str, float], prices: dict[str, float],
                 cfg: PortfolioConfig, band: float = WEIGHT_BAND, today=None) -> Plan:
    weights, schedule_note = current_weights(sig, today)
    target = {k: float(v) for k, v in weights.items() if v > 0}
    held = {k: q for k, q in positions.items() if abs(q) > 1e-9}
    timing = sig.get("order", "At the close")
    rows, trows, notes = [], [], [schedule_note] if schedule_note else []
    syms = list(dict.fromkeys(list(target) + list(held)))
    cur_w, tgt_q = {}, {}
    for s in syms:
        inst = cfg.instrument(s)
        m, fut = inst.multiplier, inst.type == "future"
        px = _price(s, prices, (sig.get("prices") or {}).get(s))
        cur_w[s] = held.get(s, 0.0) * px * m / capital if capital > 0 and np.isfinite(px) else float("nan")
        tgt_q[s] = _qty(capital * target.get(s, 0.0), px, m, fut)
        trows.append({"symbol": s, "target weight": target.get(s, 0.0), "current weight": cur_w[s],
                      "target quantity": tgt_q[s], "current quantity": held.get(s, 0.0), "price": px})
    flipped = [s for s in syms if (target.get(s, 0) > 0) != (abs(held.get(s, 0)) > 1e-9)]
    drift = max((abs(cur_w[s] - target.get(s, 0.0)) for s in syms if np.isfinite(cur_w[s])), default=0.0)
    if capital <= 0:
        notes.append("No capital allocated to this strategy.")
    elif not flipped and drift <= band:
        notes.append(f"Holdings match the signal (largest drift {drift:.1%}, band ±{band:.0%}): no trade. "
                     "The model only trades when the signal changes.")
    else:
        why = (f"signal change: {', '.join(flipped)}" if flipped else f"drift {drift:.1%} beyond ±{band:.0%}")
        for s in syms:
            if not np.isfinite(tgt_q[s]):
                notes.append(f"No price for {s}: size it yourself at {target.get(s, 0):.0%} of {capital:,.0f}.")
                continue
            d = tgt_q[s] - held.get(s, 0.0)
            if abs(d) < 1e-9:
                continue
            px = _price(s, prices, (sig.get("prices") or {}).get(s))
            rows.append(dict(strategy=sid, symbol=s, side="BUY" if d > 0 else "SELL", quantity=abs(d), order=timing,
                             limit=np.nan, price=px, value=abs(d) * px * cfg.instrument(s).multiplier, reason=why))
    cash_w = 1 - sum(target.values())
    status = (", ".join(f"{s} {w:.0%}" for s, w in target.items()) or "all cash") + \
             (f", cash {cash_w:.0%}" if target and cash_w > 1e-6 else "")
    return Plan(sid, capital, _frame(rows), pd.DataFrame(trows), notes, status)


# ---- stock-picking sleeve (SMA / Piotroski) --------------------------------------------------------
def plan_stock_picks(sid: str, sig: dict, capital: float, positions: dict[str, float], prices: dict[str, float],
                     cfg: PortfolioConfig) -> Plan:
    rows, trows, notes = [], [], []
    held = {k: q for k, q in positions.items() if q > 1e-9}
    model = {h["symbol"]: h for h in sig.get("holdings", [])}
    last = lambda s, d=None: _price(s, prices, d)
    kept_value = 0.0
    for s, q in held.items():
        h = model.get(s)
        px = last(s, (h or {}).get("last_close"))
        if h is None:
            rows.append(dict(strategy=sid, symbol=s, side="SELL", quantity=q, order="Market on open", limit=np.nan,
                             price=px, value=q * px, reason="not held by the model (check before selling)"))
        elif h.get("exit"):
            rows.append(dict(strategy=sid, symbol=s, side="SELL", quantity=q, order="Market on open", limit=np.nan,
                             price=px, value=q * px, reason=h["exit"]))
        else:
            kept_value += q * px if np.isfinite(px) else 0.0
            if h.get("take_profit") and np.isfinite(h["take_profit"]):
                rows.append(dict(strategy=sid, symbol=s, side="SELL", quantity=q, order="Limit, day",
                                 limit=round(float(h["take_profit"]), 2), price=px, value=q * h["take_profit"],
                                 reason="take profit at the SMA200 (re-enter every day: the level moves)"))
            if h.get("stop") and np.isfinite(h["stop"]):
                rows.append(dict(strategy=sid, symbol=s, side="SELL", quantity=q, order="Stop, day",
                                 limit=round(float(h["stop"]), 2), price=px, value=q * h["stop"], reason="stop loss"))
        trows.append({"symbol": s, "model": "exit" if h and h.get("exit") else ("hold" if h else "not held"),
                      "your quantity": q, "price": px, "since": (h or {}).get("entry_date", "")})
    for s, h in model.items():
        if s not in held and not h.get("exit"):
            trows.append({"symbol": s, "model": "hold (you don't)", "your quantity": 0.0,
                          "price": last(s, h.get("last_close")), "since": h.get("entry_date", "")})
            notes.append(f"The model has held {s} since {h.get('entry_date', '?')} and you don't. It will not buy it "
                         "again unless it reappears as an entry: don't chase it.")
    if not sig.get("market_ok", True):
        notes.append("Market filter is risk-off (the benchmark is below its 200-day SMA): no new entries.")
    cash = max(capital - kept_value, 0.0)
    if kept_value > capital * 1.02 and capital > 0:
        notes.append(f"The positions you keep are worth {kept_value:,.0f}, more than this sleeve's capital "
                     f"({capital:,.0f}). The model never trims a position: the excess unwinds as positions exit, and "
                     "there are no new entries until the sleeve has cash.")
    budget = capital * float(sig.get("position_weight", 0) or 0)
    n_buy = 0
    for b in sig.get("buys", []):
        s = b["symbol"]
        if s in held:
            continue
        px = last(s, b.get("last_close"))
        dollars = min(capital * float(b.get("weight", sig.get("position_weight", 0))), cash)
        if budget and dollars < 0.02 * budget:
            notes.append("Not enough cash in the sleeve for further entries.")
            break
        q = _qty(dollars, px, 1.0, False)
        if not np.isfinite(q) or q < 1:
            notes.append(f"{s}: {dollars:,.0f} does not buy one share at {px:,.2f}.")
            continue
        cash -= q * px
        n_buy += 1
        tp = b.get("take_profit")
        rows.append(dict(strategy=sid, symbol=s, side="BUY", quantity=q, order="Market on open", limit=np.nan,
                         price=px, value=q * px,
                         reason=(f"new entry; skip it if it opens at or above {tp:,.2f} (its take-profit) and buy "
                                 "the first backup instead" if tp else "new entry")))
    free = int(sig.get("max_positions", 0) or 0) - sum(1 for s in held if s in model and not model[s].get("exit"))
    if sig.get("market_ok", True) and free > 0 and not sig.get("buys"):
        notes.append(f"{free} free slot(s), but no stock passes the entry filters at this close: no new entries.")
    if sig.get("backups"):
        notes.append("Backups, in order, if an entry opens above its take-profit: " +
                     ", ".join(f"{b['symbol']} (skip at {b['take_profit']:,.2f})" if b.get("take_profit")
                               else b["symbol"] for b in sig["backups"]) + ".")
    n_exit = sum(r["side"] == "SELL" and r["order"] == "Market on open" for r in rows)
    status = (f"{len(model)} model holdings · {n_exit} exits · {n_buy} entries"
              f"{'' if sig.get('market_ok', True) else ' · market filter off'}")
    return Plan(sid, capital, _frame(rows), pd.DataFrame(trows), notes, status)


def plan(sid: str, sig: dict, capital: float, positions: dict[str, float], prices: dict[str, float],
         cfg: PortfolioConfig, band: float = WEIGHT_BAND, today=None) -> Plan:
    if sig["kind"] == "weights":
        return plan_weights(sid, sig, capital, positions, prices, cfg, band, today)
    return plan_stock_picks(sid, sig, capital, positions, prices, cfg)


def net_orders(plans: list[Plan]) -> pd.DataFrame:
    """One line per symbol and order type: market orders of the same timing are netted across strategies;
    limit and stop orders stay one per position."""
    o = pd.concat([p.orders for p in plans if len(p.orders)], ignore_index=True) if plans else _frame([])
    if o.empty:
        return pd.DataFrame(columns=["symbol", "side", "quantity", "order", "limit", "price", "value", "strategies"])
    o = o.assign(signed=np.where(o["side"] == "BUY", 1, -1) * o["quantity"])
    mkt = o[o["limit"].isna()]
    rest = o[o["limit"].notna()]
    out = []
    for (sym, order), g in mkt.groupby(["symbol", "order"], sort=False):
        q = g["signed"].sum()
        if abs(q) < 1e-9:
            continue
        px = g["price"].iloc[0]
        out.append(dict(symbol=sym, side="BUY" if q > 0 else "SELL", quantity=abs(q), order=order, limit=np.nan,
                        price=px, value=abs(q) * px, strategies=", ".join(dict.fromkeys(g["strategy"]))))
    for r in rest.itertuples():
        out.append(dict(symbol=r.symbol, side=r.side, quantity=r.quantity, order=r.order, limit=r.limit,
                        price=r.price, value=r.value, strategies=r.strategy))
    df = pd.DataFrame(out)
    return df.sort_values(["order", "side", "symbol"], key=lambda c: c.map(
        {"At the close": 0, "Market on open": 1}).fillna(2) if c.name == "order" else c).reset_index(drop=True)
