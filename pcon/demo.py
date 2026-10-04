"""Synthetic demo workspace: lets you explore the dashboard before you have a live record.

Everything here is MADE UP: prices are simulated (seeded, reproducible), the three "backtests" are simple
stand-ins with roughly realistic risk profiles, and the live trades are produced by toy versions of the
rules on the simulated prices. Real tickers are used only so the screens look familiar. Nothing in this
folder says anything about the real strategies or markets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

START, LIVE_START, END = "2004-01-02", "2026-01-02", "2026-10-02"
SEED = 20261010

DEMO_CONFIG = {
    "portfolio": {"name": "DEMO book (synthetic data)", "currency": "USD", "benchmark": "SPY", "risk_free": "^IRX",
                  "auto_dividends": True, "offline": True, "demo": True},
    "strategies": {
        "sma_piotroski": {"name": "SMA-200 Pullback + Piotroski F-Score", "short": "SMA-PF", "color": "#2a78d6",
                          "target_weight": 0.40, "universe": "S&P 500 stocks, max 5 positions",
                          "description": "Buys quality (F-Score >= 7) stocks between SMA50 and SMA200, exits at SMA200.",
                          "backtest": "backtests/sma_piotroski.csv", "risk_budget": 0.40, "expectation_start": "2014-01-01"},
        "regime_filter": {"name": "Macro Regime Filter (unemployment x policy rate)", "short": "Regime",
                          "color": "#eb6834", "target_weight": 0.30, "universe": "SPY / QQQ / T-bills",
                          "description": "Expansion confidence sets NDX / SPX / cash exposure, monthly.",
                          "backtest": "backtests/regime_filter.csv", "risk_budget": 0.25, "expectation_start": "2005-01-01"},
        "rolling_momentum": {"name": "Rolling Time-Series Momentum", "short": "Momentum", "color": "#1baf7a",
                             "target_weight": 0.30, "universe": "SPY / T-bills",
                             "description": "Long the index when its trailing return is positive, else T-bills.",
                             "backtest": "backtests/rolling_momentum.csv", "risk_budget": 0.35, "expectation_start": "2010-01-01"},
    },
    "allocation": {"start": None, "rebalance": "M", "min_weight": 0.0, "max_weight": 0.7,
                   "walk_forward_lookback_years": 3, "walk_forward_step": "Q", "bootstrap_samples": 200},
    "alerts": {"drawdown_percentile": 0.95, "return_percentile": 0.05, "vol_ratio": 1.5, "weight_drift": 0.05,
               "tracking_error": 0.05, "correlation": 0.8},
}

STOCKS = {  # symbol: (sector, beta, start price, market-cap rank, dividend per quarter in % of price)
    "JNJ": ("Health Care", 0.6, 62.0, 3, 0.7), "PG": ("Consumer Staples", 0.5, 55.0, 4, 0.6),
    "HD": ("Consumer Discretionary", 1.0, 35.0, 5, 0.5), "CSCO": ("Information Technology", 1.1, 24.0, 6, 0.0),
    "XOM": ("Energy", 0.8, 41.0, 2, 0.8), "UNH": ("Health Care", 0.9, 30.0, 7, 0.3),
    "LIN": ("Materials", 0.9, 48.0, 8, 0.0), "CAT": ("Industrials", 1.2, 40.0, 9, 0.5),
    "MSFT": ("Information Technology", 1.1, 27.0, 1, 0.2), "KO": ("Consumer Staples", 0.55, 44.0, 10, 0.7),
}
SPLITS = {"CAT": ("2026-05-15", 2.0)}            # a 2-for-1 split inside the live window


def _irx(dates: pd.DatetimeIndex) -> pd.Series:
    knots = {"2004-01-01": 0.9, "2006-06-01": 4.9, "2007-08-01": 4.6, "2008-12-01": 0.05, "2015-11-01": 0.1,
             "2018-12-01": 2.4, "2019-12-01": 1.5, "2020-04-01": 0.05, "2022-02-01": 0.1, "2023-06-01": 5.2,
             "2024-08-01": 5.1, "2025-06-01": 4.3, "2026-10-31": 3.9}
    k = pd.Series(knots, dtype=float)
    k.index = pd.to_datetime(k.index)
    return k.reindex(k.index.union(dates)).interpolate("time").reindex(dates)


def _market(dates: pd.DatetimeIndex, rng) -> pd.Series:
    """S&P-like daily returns: GARCH-style vol clustering, scripted bear markets, ~9.5 % CAGR overall."""
    n = len(dates)
    stress = [("2007-10-10", "2009-03-09", -0.0014, 1.9), ("2011-07-22", "2011-10-03", -0.0016, 1.7),
              ("2015-08-10", "2016-02-11", -0.0005, 1.3), ("2018-09-20", "2018-12-24", -0.0019, 1.5),
              ("2020-02-19", "2020-03-23", -0.0100, 3.5), ("2022-01-03", "2022-10-12", -0.0009, 1.4),
              ("2025-02-19", "2025-04-08", -0.0030, 2.0), ("2026-03-02", "2026-04-17", -0.0012, 1.6)]
    mu, boost = np.zeros(n), np.ones(n)
    for a, b, drift, vb in stress:
        m = (dates >= a) & (dates <= b)
        mu[m], boost[m] = drift, vb
    r = np.empty(n)
    var = 0.008 ** 2
    for t in range(n):
        e = np.sqrt(var) * rng.standard_t(5) / np.sqrt(5 / 3)
        r[t] = e * boost[t]
        var = min(0.0000018 + 0.07 * e ** 2 + 0.90 * var, 0.00025)
    calm = mu == 0
    mu[calm] = (np.log(1.095) * n / 252 - np.log1p(r + mu).sum()) / calm.sum()
    return pd.Series(r + mu, index=dates)


def _write_prices(path: Path, close: pd.Series, dividends: pd.Series | None = None,
                  splits: pd.Series | None = None) -> None:
    df = pd.DataFrame({"date": close.index.strftime("%Y-%m-%d"), "close": close.round(4).to_numpy(),
                       "dividends": (dividends if dividends is not None else pd.Series(0.0, index=close.index))
                       .reindex(close.index).fillna(0).round(4).to_numpy(),
                       "splits": (splits if splits is not None else pd.Series(0.0, index=close.index))
                       .reindex(close.index).fillna(0).to_numpy()})
    df.to_csv(path, index=False)


class _Sim:
    """Tiny execution simulator: whole shares, 2 bp slippage, Alpaca-like tiny fees."""

    def __init__(self, sleeve: str, cash: float):
        self.sleeve, self.cash, self.pos, self.trades = sleeve, cash, {}, []

    def nav(self, px: dict) -> float:
        return self.cash + sum(q * px[s] for s, q in self.pos.items())

    def trade(self, date, sym, qty, px, rng):
        if sym in SPLITS and pd.Timestamp(date) < pd.Timestamp(SPLITS[sym][0]):
            lot = int(SPLITS[sym][1])                  # pre-split fills must be whole pre-split shares
            qty = int(qty / lot) * lot
        if qty == 0:
            return
        fill = px * (1 + np.sign(qty) * abs(rng.normal(0.0002, 0.0002)))
        fee = round(0.000003 * abs(qty) + (0.0000206 * abs(qty) * fill + min(0.000195 * abs(qty), 9.79)
                                           if qty < 0 else 0), 2)
        self.cash -= qty * fill + fee
        self.pos[sym] = self.pos.get(sym, 0) + qty
        if self.pos[sym] == 0:
            del self.pos[sym]
        self.trades.append(dict(date=date, strategy=self.sleeve, symbol=sym, side="BUY" if qty > 0 else "SELL",
                                quantity=abs(qty), price=round(fill, 2), fees=fee, note=""))

    def target(self, date, weights: dict, px: dict, rng, band: float = 0.02):
        nav = self.nav(px)
        for sym in sorted(set(weights) | set(self.pos), key=lambda s: weights.get(s, 0)):
            want = int(weights.get(sym, 0) * nav / px[sym])
            have = self.pos.get(sym, 0)
            if abs(want - have) * px[sym] > band * nav or (want == 0 and have != 0):
                self.trade(date, sym, want - have, px[sym], rng)


def build_demo(root: str | Path, force: bool = False) -> Path:
    root = Path(root)
    if (root / "portfolio.yaml").exists() and not force:
        return root
    rng = np.random.default_rng(SEED)
    (root / "prices").mkdir(parents=True, exist_ok=True)
    (root / "backtests").mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(START, END)
    irx = _irx(dates)
    cash_r = ((1 + irx.shift(1).bfill() / 100) ** (1 / 252) - 1)

    # ---- simulated market --------------------------------------------------------------------
    spx = _market(dates, rng)
    ndx = 0.0001 + 1.2 * spx + rng.normal(0, 0.006, len(dates))
    spy_px = 112 * (1 + spx).cumprod()
    qqq_px = 36 * (1 + ndx).cumprod()
    bil_px = 91.5 * (1 + cash_r).cumprod()
    _write_prices(root / "prices" / "SPY.csv", spy_px)
    _write_prices(root / "prices" / "QQQ.csv", qqq_px)
    _write_prices(root / "prices" / "BIL.csv", bil_px)
    _write_prices(root / "prices" / "_IRX.csv", irx)

    stock_px, stock_div, stock_split = {}, {}, {}
    for sym, (_, beta, p0, _, dq) in STOCKS.items():
        idio = rng.normal(0.00015, 0.012, len(dates))
        idio += 0.004 * np.sin(np.arange(len(dates)) / rng.uniform(40, 90)) / 30    # slow mean reversion swings
        r = beta * spx + idio
        px = p0 * (1 + r).cumprod()
        div = pd.Series(0.0, index=dates)
        if dq:
            exd = dates[(dates.month.isin([2, 5, 8, 11])) & (dates.day <= 7)]
            exd = pd.DatetimeIndex(pd.Series(exd).groupby([exd.year, exd.month]).first().to_numpy())
            div.loc[exd] = (px.loc[exd] * dq / 100).round(4)
        spl = pd.Series(0.0, index=dates)
        if sym in SPLITS:
            d, ratio = SPLITS[sym]
            sd = dates[dates.searchsorted(pd.Timestamp(d))]
            spl.loc[sd] = ratio
            px = px / ratio                                   # Yahoo-style: whole history in post-split units
        stock_px[sym], stock_div[sym], stock_split[sym] = px, div, spl
        _write_prices(root / "prices" / f"{sym}.csv", px, div, spl)

    # ---- backtest stand-ins (history before live) -------------------------------------------
    pre = dates < LIVE_START
    mom_pos = (spy_px / spy_px.shift(63) - 1 > 0).astype(float).shift(1).fillna(0)
    mom_r = mom_pos * spx + (1 - mom_pos) * cash_r - mom_pos.diff().abs().fillna(0) * 0.0005

    six_m = (1 + spx).rolling(126).apply(np.prod, raw=True) - 1          # macro data lags the market ~2 months
    conf = (0.8 + 1.5 * six_m.shift(42).fillna(0) + rng.normal(0, 0.08, len(dates))).clip(0, 1)
    month_end = dates.to_series().groupby(dates.to_period("M")).transform("max") == dates.to_series()
    conf_m = conf.where(month_end).ffill().shift(1).fillna(0.75)
    w_ndx = (conf_m >= 0.7).astype(float)
    w_spx = ((conf_m >= 0.4) & (conf_m < 0.7)).astype(float)
    reg_r = w_ndx * ndx + w_spx * spx + (1 - w_ndx - w_spx) * cash_r

    sma200 = spy_px.rolling(200).mean()
    risk_on = (spy_px > sma200).shift(1).fillna(True)
    expo = (0.55 + 0.35 * np.sin(np.arange(len(dates)) / 37.0) ** 2 + rng.normal(0, 0.03, len(dates))).clip(0, 1)
    expo = pd.Series(expo, index=dates).where(risk_on, np.nan).ffill() * np.where(risk_on, 1.0, 0.6)
    expo = expo.rolling(30, min_periods=1).mean()
    sma_r = expo * (0.65 * spx + 0.00028 + rng.normal(0, 0.0075, len(dates)))

    # ---- live period: toy rules on simulated prices -> journal --------------------------------
    live = dates[dates >= LIVE_START]
    caps = {"sma_piotroski": 120_000.0, "regime_filter": 90_000.0, "rolling_momentum": 90_000.0}
    sims = {k: _Sim(k, v) for k, v in caps.items()}
    flows = [dict(date=LIVE_START, strategy=k, type="deposit", amount=v, note="initial allocation")
             for k, v in caps.items()]
    all_px = {"SPY": spy_px, "QQQ": qqq_px, "BIL": bil_px, **stock_px}
    sma50 = {s: p.rolling(50).mean() for s, p in stock_px.items()}
    s200 = {s: p.rolling(200).mean() for s, p in stock_px.items()}
    held_since: dict = {}
    for i, d in enumerate(live):
        px = {s: float(p.loc[d]) for s, p in all_px.items()}
        prev = dates[dates.get_loc(d) - 1]
        for s, sim in sims.items():                      # dividends (the sims work in post-split units)
            for sym in list(sim.pos):
                if sym in stock_div and stock_div[sym].loc[d] > 0:
                    sim.cash += sim.pos[sym] * stock_div[sym].loc[d]
        if d == pd.Timestamp("2026-06-01"):
            flows.append(dict(date=d, strategy="regime_filter", type="deposit", amount=25_000.0, note="top-up"))
            sims["regime_filter"].cash += 25_000.0
        if d == pd.Timestamp("2026-08-03"):
            flows += [dict(date=d, strategy="sma_piotroski", type="transfer", amount=-10_000.0, note="rebalance"),
                      dict(date=d, strategy="rolling_momentum", type="transfer", amount=10_000.0, note="rebalance")]
            sims["sma_piotroski"].cash -= 10_000.0
            sims["rolling_momentum"].cash += 10_000.0
        # momentum: daily signal from yesterday's close
        m_on = spy_px.loc[prev] / spy_px.loc[:prev - pd.Timedelta(days=90)].iloc[-1] - 1 > 0   # 90-day window
        sims["rolling_momentum"].target(d, {"SPY": 0.99} if m_on else {"BIL": 0.99}, px, rng, band=0.03)
        # regime: monthly
        if i == 0 or d.month != live[i - 1].month:
            c = conf.loc[prev]
            w = {"QQQ": 0.99} if c >= 0.7 else ({"SPY": 0.99} if c >= 0.4 else {"BIL": 0.99})
            sims["regime_filter"].target(d, w, px, rng, band=0.03)
        # SMA-PF: exits then entries, decided on yesterday's close
        sim = sims["sma_piotroski"]
        for sym in list(sim.pos):
            if stock_px[sym].loc[prev] >= s200[sym].loc[prev] or (d - held_since[sym]).days > 365:
                sim.trade(d, sym, -sim.pos[sym], px[sym], rng)
        mkt_ok = spy_px.loc[prev] > sma200.loc[prev]
        if mkt_ok and len(sim.pos) < 5:
            sectors = {STOCKS[s][0] for s in sim.pos}
            cands = sorted((s for s in STOCKS if s not in sim.pos and sma50[s].loc[prev] < stock_px[s].loc[prev]
                            < s200[s].loc[prev]), key=lambda s: STOCKS[s][3])
            for s in cands:
                if len(sim.pos) >= 5:
                    break
                if STOCKS[s][0] in sectors:
                    continue
                q = int(min(sim.nav(px) / 5, sim.cash) / px[s])
                if q > 0:
                    sim.trade(d, s, q, px[s], rng)
                    held_since[s] = d
                    sectors.add(STOCKS[s][0])

    trades = pd.DataFrame([t for s in sims.values() for t in s.trades]).sort_values("date", kind="stable")
    # journal quantities/prices before a split are in the old (pre-split) units, as your broker shows them
    for sym, (d, ratio) in SPLITS.items():
        m = (trades["symbol"] == sym) & (pd.to_datetime(trades["date"]) < pd.Timestamp(d))
        trades.loc[m, "quantity"] = (trades.loc[m, "quantity"] / ratio).round(4)
        trades.loc[m, "price"] = (trades.loc[m, "price"] * ratio).round(2)
    trades["date"] = pd.to_datetime(trades["date"]).dt.strftime("%Y-%m-%d")
    trades.to_csv(root / "trades.csv", index=False)
    cf = pd.DataFrame(flows)
    cf["date"] = pd.to_datetime(cf["date"]).dt.strftime("%Y-%m-%d")
    cf.to_csv(root / "cashflows.csv", index=False)
    (root / "marks.csv").write_text("date,symbol,price\n")
    (root / "portfolio.yaml").write_text("# DEMO workspace: synthetic data, see pcon/demo.py\n"
                                         + yaml.safe_dump(DEMO_CONFIG, sort_keys=False))

    # ---- backtests: history + "model" on live dates (live sleeve returns + execution noise) ----
    from .book import Book
    book = Book(root)
    starts = {"sma_piotroski": "2011-01-03", "regime_filter": "2004-03-01", "rolling_momentum": "2005-01-03"}
    for sid, hist in (("sma_piotroski", sma_r), ("regime_filter", reg_r), ("rolling_momentum", mom_r)):
        lr = book.live_returns().get(sid, pd.Series(dtype=float))
        model_live = lr + rng.normal(0.00003, 0.0006, len(lr))
        r = pd.concat([hist[pre].loc[starts[sid]:], model_live])
        expo_s = {"sma_piotroski": expo, "regime_filter": w_ndx + w_spx, "rolling_momentum": mom_pos}[sid]
        pd.DataFrame({"date": r.index.strftime("%Y-%m-%d"), "return": r.round(8).to_numpy(),
                      "exposure": expo_s.reindex(r.index).round(4).to_numpy()}) \
            .to_csv(root / "backtests" / f"{sid}.csv", index=False)
        (root / "backtests" / f"{sid}.meta.json").write_text(json.dumps(DEMO_META[sid], indent=2))
    return root


# Synthetic selection records, in the format the export cells write (see strategy_exports/).
DEMO_META = {
    "sma_piotroski": {
        "strategy": "sma_piotroski", "source": "DEMO (synthetic)", "data_end": END, "passed_selection": True,
        "selected": "Walk-forward account; currently trading SMA50 | <=1/sector | 5 pos | equal",
        "selection_rule": "Each out-of-sample year trades the configuration with the most scorecard points over the "
                          "previous 3 years (180 configurations).",
        "final_tests": {"Look-ahead audit: truncated-data rebuild identical": True,
                        "PSR of the out-of-sample record > 0.95": True,
                        "Out-of-sample Sharpe >= SPY buy & hold": False,
                        "Deflated Sharpe of the best configuration > 0.95": True,
                        "PBO < 0.5 (CSCV over the configuration grid)": True,
                        "Walk-forward efficiency >= 0.5": True,
                        "Bootstrap 5th-percentile Sharpe > 0": True},
        "verdict": "DEMO: 6/7 final tests passed.", "oos_start": "2014-01-01"},
    "regime_filter": {
        "strategy": "regime_filter", "source": "DEMO (synthetic)", "data_end": END, "passed_selection": True,
        "selected": "Optimized Tiers (>= 70%: 100% Nasdaq-100 | 40-60%: 100% S&P 500 | < 40%: Cash)",
        "selection_rule": "Allocation rule with the highest Sharpe on the untouched test window (2005 -> today); "
                          "optimised tiers were fitted on 1986-2004 only.",
        "final_tests": {"Look-ahead audit: truncated-data rebuild matches": True,
                        "Engine causality: future returns do not change the past": True,
                        "Test window: Sharpe >= S&P 500 B&H": True,
                        "Test window: max drawdown shallower than S&P 500 B&H": True,
                        "Test window: Sharpe gain significant (bootstrap P(diff <= 0) < 0.05)": True,
                        "Placebo timing test: p < 0.10": True,
                        "Deflated Sharpe > 0.95 (design window, 540 trials)": True,
                        "PBO < 0.5 (CSCV over the tier grid)": True},
        "verdict": "DEMO: 8/8 final tests passed.", "oos_start": "2005-01-01"},
    "rolling_momentum": {
        "strategy": "rolling_momentum", "source": "DEMO (synthetic)", "data_end": END, "passed_selection": True,
        "selected": "90D lookback, long/flat S&P 500",
        "selection_rule": "Lookback that survived all four in-sample gates (2000-2020) with the highest in-sample "
                          "Sharpe; frozen for all out-of-sample and cross-market tests.",
        "final_tests": {"1. Selected lookback survived all in-sample gates": True,
                        "2a. OOS Sharpe >= S&P 500 buy & hold": True,
                        "2b. OOS Sharpe gain is significant (bootstrap p < 0.05)": True,
                        "3. OOS max drawdown shallower than buy & hold": True,
                        "4a. Low overfitting risk: PBO < 0.5 (candidate set)": True,
                        "4b. Survives deflation: DSR > 0.95 (N = 4)": True,
                        "5. Timing skill beyond exposure (circular-shift p < 0.05)": True,
                        "6. Transfers to NQ: OOS Sharpe >= Nasdaq-100 buy & hold": True,
                        "7. Beats B&H Sharpe at 2x cost and with a 1-day delay": True},
        "verdict": "DEMO: SUPPORTED.", "oos_start": "2021-01-01"},
}
