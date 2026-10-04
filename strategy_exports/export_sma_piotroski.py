# =================================================================================================
# Export for the Portfolio-construction dashboard  ->  backtests/sma_piotroski.csv (+ .meta.json)
# Paste as the LAST cell of SMA_Piotroski_Backtest.ipynb (repo SMA_Pfscore) and run it after "Run All"
# (it needs section 8's diagnostics and, if enabled, section 9's look-ahead audit).
#
# Which strategy is exported: the notebook's FINAL selection, never a hand-picked one.
#   * the walk-forward OUT-OF-SAMPLE account (WF): every year it trades the configuration that won the
#     scorecard on the preceding in-sample years, under the current rules (the notebook advises against
#     switching to an ablation / ranking variant after seeing its table);
#   * its final tests are saved next to the returns: look-ahead audit, PSR, out-of-sample Sharpe vs SPY,
#     Deflated Sharpe, PBO, walk-forward efficiency and the bootstrap Sharpe interval. The dashboard shows them.
#
# Live tracking: once you trade the strategy, set LIVE_START to your first live day and re-run the notebook
# with REFRESH_DATA = True. From LIVE_START on, the export switches to a model account opened that same
# day with the walk-forward parameters, so it holds what your account should hold (same entries, same
# exits) and the dashboard can measure your execution against it day by day.
# =================================================================================================
import json
import os
from pathlib import Path

EXPORT_DIR = Path(os.environ.get("PCON_BACKTESTS", "../Portfolio-construction/data/backtests"))
LIVE_START = None            # e.g. "2026-10-05" = first day you traded this strategy live

ret, expo = WF["returns"], WF["exposure"]
if LIVE_START:
    cut = pd.Timestamp(LIVE_START)
    fresh = run_backtest(FEAT, schedule, start=cut)          # model account opened the day you went live
    ret = pd.concat([ret.loc[:cut - pd.Timedelta(days=1)], fresh["returns"]])
    expo = pd.concat([expo.loc[:cut - pd.Timedelta(days=1)], fresh["exposure"]])

oos_sr, bench_sr = perf_stats(OOS_R).get("Sharpe", float("nan")), perf_stats(B_OOS).get("Sharpe", float("nan"))
audit_ok = bool(audit["PASS"].all()) if "audit" in globals() and RUN_LOOKAHEAD_AUDIT else None
tests = {
    "Look-ahead audit: truncated-data rebuild identical": audit_ok,
    "PSR of the out-of-sample record > 0.95": bool(psr(OOS_R) > 0.95),
    f"Out-of-sample Sharpe >= {BENCHMARK} buy & hold": bool(oos_sr >= bench_sr),
    "Deflated Sharpe of the best configuration > 0.95": bool(dsr > 0.95),
    "PBO < 0.5 (CSCV over the configuration grid)": bool(pbo < 0.5),
    "Walk-forward efficiency >= 0.5": bool(WFE >= 0.5) if pd.notna(WFE) else None,
    "Bootstrap 5th-percentile Sharpe > 0": bool(ci.loc["Sharpe", "5%"] > 0),
}
n_pass = sum(v is True for v in tests.values())
n_eval = sum(v is not None for v in tests.values())
meta = {
    "strategy": "sma_piotroski",
    "selected": f"Walk-forward account; currently trading {label(schedule[-1][1])}",
    "selection_rule": f"Each out-of-sample year trades the configuration with the most scorecard points over the "
                      f"previous {WF_IS_YEARS} years ({len(LABELS)} configurations); the out-of-sample account is "
                      "the strategy.",
    "passed_selection": True,
    "final_tests": tests,
    "verdict": (f"{n_pass}/{n_eval} final tests passed. Out-of-sample Sharpe {oos_sr:.2f} vs {bench_sr:.2f} for "
                f"{BENCHMARK}; PSR {psr(OOS_R):.0%}, DSR {dsr:.0%}, PBO {pbo:.0%}, walk-forward efficiency {WFE:.2f}."),
    "oos_start": f"{OOS_R.index[0]:%Y-%m-%d}",
    "source": "SMA_Piotroski_Backtest.ipynb",
    "live_start": LIVE_START,
}

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
out = pd.DataFrame({"return": ret, "exposure": expo}).dropna(subset=["return"]).rename_axis("date")
meta["data_end"] = f"{out.index[-1]:%Y-%m-%d}"
out.to_csv(EXPORT_DIR / "sma_piotroski.csv", date_format="%Y-%m-%d", float_format="%.8f")
(EXPORT_DIR / "sma_piotroski.meta.json").write_text(json.dumps(meta, indent=2))
print(f"Saved {len(out):,} days ({out.index[0]:%Y-%m-%d} -> {out.index[-1]:%Y-%m-%d}) to "
      f"{(EXPORT_DIR / 'sma_piotroski.csv').resolve()}")
print(meta["selected"])
print(meta["verdict"])

# ---- the account's next-open plan -> signals/sma_piotroski.json (read by the dashboard's Orders tab) ------
# Re-applies run_backtest's own rules at the last close: which holdings exit at the next open, which candidates
# fill the free slots (ranking, sector cap, market filter, sizing), and each holding's take-profit limit.
# Run with REFRESH_DATA = True so the last close is the latest one. Tickers are Yahoo symbols (BRK-B = BRK.B).
from collections import Counter

SIGNALS_DIR = Path(os.environ.get("PCON_SIGNALS", EXPORT_DIR.parent / "signals"))
acct = fresh if LIVE_START else WF               # the model account your sleeve follows
dts, i = FEAT["dates"], len(FEAT["dates"]) - 1
prm = schedule[0][1]
for d0, p0 in schedule:                          # parameters in force at the last close
    if d0 is None or pd.Timestamp(d0) <= dts[i]:
        prm = p0
maxp, cap, fmin = prm["max_positions"], prm["max_per_sector"], prm["f_min"]
tick, sec, sec_names = list(FEAT["tickers"]), FEAT["sector_code"], FEAT["sector_names"]
mkt_ok = bool(FEAT["market_ok"][i])
num_ = lambda x: float(x) if x is not None and np.isfinite(x) else None

holdings = []
for r in acct["open"].itertuples():
    j = tick.index(r.ticker)
    bars_held_at_open = i + 1 - dts.get_loc(r.entry_date)
    exit_ = ("market filter" if MARKET_FILTER == "liquidate" and not mkt_ok else
             "F-Score below min" if not FEAT["fscore"][i, j] >= F_EXIT else
             "time limit" if MAX_HOLD_DAYS and bars_held_at_open >= MAX_HOLD_DAYS else
             "take profit (SMA200)" if TP_MODE == "close" and FEAT["close"][i, j] >= FEAT["sma200"][i, j] else None)
    holdings.append(dict(symbol=r.ticker, sector=r.sector, entry_date=f"{r.entry_date:%Y-%m-%d}",
                         entry_price=num_(r.entry_px), last_close=num_(FEAT["close"][i, j]),
                         take_profit=num_(FEAT["sma200"][i, j]) if TP_MODE == "limit" else None,
                         stop=num_(r.entry_px * (1 - STOP_LOSS)) if STOP_LOSS else None, exit=exit_))
kept = [h for h in holdings if not h["exit"]]
slots = maxp - len(kept)
picks = []
if slots > 0 and (MARKET_FILTER is None or mkt_ok):
    with np.errstate(invalid="ignore"):
        ok = FEAT["base"][i] & FEAT["above"][prm["sma_fast"]][i] & (FEAT["fscore"][i] >= fmin)
        if ENTRY_TRIGGER == "upgrade":
            ok &= FEAT["fscore_prev"][i] < F_EXIT
    cand = np.flatnonzero(ok)
    C_, S2, MC_, FS_ = FEAT["close"][i], FEAT["sma200"][i], FEAT["mcap"][i], FEAT["fscore"][i]
    if RANK_BY == "upside":
        cand = cand[np.argsort(-(S2[cand] / C_[cand]), kind="stable")]
    elif RANK_BY == "fscore":
        cand = cand[np.lexsort((-MC_[cand], -FS_[cand]))]
    else:
        cand = cand[np.argsort(-MC_[cand], kind="stable")]
    used = Counter(sec[tick.index(h["symbol"])] for h in kept)
    kept_set = {h["symbol"] for h in kept}
    for j in cand:
        if tick[j] in kept_set or (cap and used[sec[j]] >= cap):
            continue
        w = 1.0
        if prm["sizing"] == "inverse_vol" and FEAT["vol"][i, j] > 0 and np.isfinite(FEAT["vol_ref"][i]):
            w = float(np.clip(FEAT["vol_ref"][i] / FEAT["vol"][i, j], *VOL_WEIGHT_BOUNDS))
        picks.append(dict(symbol=tick[j], sector=sec_names[sec[j]], weight=w / maxp, last_close=num_(C_[j]),
                          take_profit=num_(S2[j])))
        used[sec[j]] += 1
        if len(picks) >= slots + 3:              # the entries plus three backups
            break
signal = {
    "strategy": "sma_piotroski", "kind": "stock_picks", "selected": label(prm),
    "as_of": f"{dts[i]:%Y-%m-%d}", "generated": f"{pd.Timestamp.now():%Y-%m-%dT%H:%M:%S}", "source": meta["source"],
    "max_positions": int(maxp), "position_weight": 1.0 / maxp, "market_ok": mkt_ok,
    "holdings": holdings, "buys": picks[:max(slots, 0)], "backups": picks[max(slots, 0):],
    "execution": "Exits and entries at the next open (market-on-open). Skip an entry that opens at or above its "
                 "take-profit and take the next backup. Positions are sized once, at entry, and never rebalanced."
                 + (" Take-profit: a sell limit at each holding's SMA200, re-entered daily." if TP_MODE == "limit" else ""),
}
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
(SIGNALS_DIR / "sma_piotroski.json").write_text(json.dumps(signal, indent=2))
print(f"Next open: {sum(bool(h['exit']) for h in holdings)} exits, {len(signal['buys'])} entries "
      f"({', '.join(b['symbol'] for b in signal['buys']) or 'none'}), {len(kept)} holdings kept"
      f"{'' if mkt_ok else '; market filter is risk-off'} -> {(SIGNALS_DIR / 'sma_piotroski.json').resolve()}")
