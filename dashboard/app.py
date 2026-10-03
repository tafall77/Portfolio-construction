"""Portfolio dashboard: track record, expected vs actual, and portfolio construction across strategies.

Run:  streamlit run dashboard/app.py                      (your workspace in data/, or the demo)
      streamlit run dashboard/app.py -- --workspace PATH   (any workspace folder)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcon import allocation as A  # noqa: E402
from pcon import charts as C  # noqa: E402
from pcon import expectations as E  # noqa: E402
from pcon import metrics as M  # noqa: E402
from pcon.backtests import tests_summary  # noqa: E402
from pcon.book import Book  # noqa: E402
from pcon.config import CASH_SLEEVE  # noqa: E402
from pcon.demo import build_demo  # noqa: E402
from pcon.journal import (CASH_COLUMNS, FLOW_TYPES, SIDES, TRADE_COLUMNS, JournalError,  # noqa: E402
                          append_cashflow, append_trade, append_transfer, save_cashflows, save_trades)
from pcon.workspace import init_workspace  # noqa: E402

st.set_page_config(page_title="Portfolio construction", page_icon="📈", layout="wide")

LIVE_WS = ROOT / "data"
DEMO_WS = ROOT / "examples" / "demo"
LEVEL_ICON = {"red": "🔴", "amber": "🟠", "green": "🟢", "info": "ℹ️"}
STATUS_ICON = {"In line": "🟢 In line", "Watch": "🟠 Watch", "Below expectations": "🔴 Below expectations",
               "Above expectations": "🔵 Above expectations", "Lower than expected": "🔵 Lower than expected",
               "Higher than expected": "🟠 Higher than expected", "n/a": "–"}


# ------------------------------------------------------------------------------------------------
# workspace + caching
# ------------------------------------------------------------------------------------------------
def cli_workspace() -> str | None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--workspace", default=None)
    args, _ = p.parse_known_args(sys.argv[1:])
    return args.workspace


def signature(ws: Path) -> tuple:
    files = [f for f in ws.rglob("*") if f.is_file() and "cache" not in f.relative_to(ws).parts]
    return tuple(sorted((str(f.relative_to(ws)), f.stat().st_mtime_ns, f.stat().st_size) for f in files))


@st.cache_resource(show_spinner="Rebuilding the book from the journal ...", max_entries=4)
def load_book(ws: str, sig: tuple, refresh: int, day: str) -> Book:
    if refresh:
        for f in (Path(ws) / "cache" / "prices").glob("*.csv"):
            f.unlink()
    return Book(ws)


@st.cache_data(show_spinner="Testing every combination of strategies (walk-forward) ...", max_entries=16)
def cached_explore(R: pd.DataFrame, rf: pd.Series, methods: tuple, rebalance: str, hi: float, target: pd.Series,
                   wf_days: int, step: str) -> pd.DataFrame:
    return A.explore(R, rf, methods, rebalance, 0.0, hi, target, wf_days, step, n_resamples=20)


@st.cache_data(show_spinner="Optimising allocations ...", max_entries=16)
def cached_methods(R: pd.DataFrame, rf: pd.Series, hi: float, target: pd.Series, rebalance: str, wf_days: int,
                   step: str, n_resamples: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    W, rows, wf = {}, {}, {}
    for m in A.METHODS:
        w = A.optimize(R, m, rf, 0.0, hi, target, n_resamples)
        W[m] = w
        r = A.portfolio_returns(R, w, rebalance)
        s = A.stats_row(r, rf)
        if m in ("equal", "target"):
            r_wf = r.iloc[wf_days:]
        else:
            r_wf, _ = A.walk_forward(R, m, rf, wf_days, step, rebalance, 0.0, hi, target, 30)
        s_wf = A.stats_row(r_wf, rf)
        rows[m] = {**s, **{f"WF {k}": v for k, v in s_wf.items() if k in ("CAGR", "Sharpe", "Max drawdown",
                                                                             "Calmar")}}
        wf[m] = r_wf
    return pd.DataFrame(W).T, pd.DataFrame(rows).T, wf


@st.cache_data(show_spinner=False, max_entries=16)
def cached_marginal(R, rf, method, rebalance, hi, target):
    return A.marginal(R, rf, method, rebalance, 0.0, hi, target)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_sharpe_test(a, b, rf):
    return A.sharpe_diff_test(a, b, rf)


def theme() -> C.Theme:
    try:
        return C.Theme(dark=(st.context.theme.type == "dark"))
    except Exception:
        return C.Theme(False)


T = theme()


def chart(fig, key=None):
    """Render with the title as page text (Plotly titles collide with top legends inside Streamlit)."""
    title = fig.layout.title.text if fig.layout.title else None
    if title:
        st.markdown(f"<div style='font-weight:600;font-size:0.95rem;margin:0.2rem 0 -0.4rem 0'>{title}</div>",
                    unsafe_allow_html=True)
        fig.update_layout(title=None, margin_t=max(int(fig.layout.margin.t or 0) - 34, 4),
                          height=int(fig.layout.height) - 34)
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False, "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d"]})


def pct(v, d=1):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{d}%}"


def money(v, cur="$"):
    return "–" if v is None or not np.isfinite(v) else f"{'-' if v < 0 else ''}{cur}{abs(v):,.0f}"


def compact(v, cur="$"):
    if v is None or not np.isfinite(v):
        return "–"
    a = abs(v)
    body = f"{a / 1e6:.2f}M" if a >= 1e6 else f"{a / 1e3:.1f}K" if a >= 1e4 else f"{a:,.0f}"
    return f"{'-' if v < 0 else ''}{cur}{body}"


def num(v, d=2):
    return "–" if v is None or not np.isfinite(v) else f"{v:,.{d}f}"


def diverging_bg(v, lim: float = 0.4) -> str:
    """Cell background on the diverging pair (warm = loss, cool = gain), neutral at zero."""
    if v is None or not isinstance(v, (int, float, np.floating)) or not np.isfinite(v):
        return ""
    neg, _, pos = T["div"]
    a = min(abs(float(v)) / lim, 1.0) * 0.55
    return f"background-color: {C._rgba(pos if v >= 0 else neg, a)}"


def selection_panel(sid: str):
    """What was exported for this strategy and whether it passed the research notebook's final tests."""
    meta = book.backtest_meta.get(sid)
    if book.backtest_returns(sid) is None:
        return
    if meta is None:
        st.info("This backtest export carries no selection record. Re-run the updated export cell in "
                "`strategy_exports/` so the dashboard can confirm it is the configuration that passed the "
                "notebook's final tests.", icon="ℹ️")
        return
    n_pass, n_eval, failed = tests_summary(meta)
    icon = "✅" if not failed and meta.get("passed_selection", True) else "⚠️"
    st.markdown(f"{icon} **Exported configuration:** {meta.get('selected', '–')}  \n"
                f"**Selection rule:** {meta.get('selection_rule', '–')}  \n"
                f"**Final tests:** {n_pass}/{n_eval} passed · source `{meta.get('source', '?')}` · data to "
                f"{meta.get('data_end', '?')}")
    with st.expander("Final tests and verdict from the research notebook", expanded=bool(failed)):
        rows = [{"Test": k, "Result": {True: "✅ pass", False: "❌ fail", None: "– n/a"}[v]}
                for k, v in meta.get("final_tests", {}).items()]
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Test"))
        if meta.get("verdict"):
            st.markdown(f"**Verdict:** {meta['verdict']}")


def tests_label(sid: str) -> str:
    meta = book.backtest_meta.get(sid)
    if meta is None:
        return "–"
    n_pass, n_eval, failed = tests_summary(meta)
    return f"{'✅' if not failed else '⚠️'} {n_pass}/{n_eval}"


def metric_table(df: pd.DataFrame, rows: list[str] | None = None):
    if df is None or df.empty:
        st.info("Not enough data yet.")
        return
    d = df.reindex([r for r in (rows or df.index) if r in df.index])
    st.dataframe(M.format_frame(d), height=min(38 + 35 * len(d), 900))


# ------------------------------------------------------------------------------------------------
# sidebar
# ------------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Workspace")
    cli = cli_workspace()
    options = ([cli] if cli else []) + (["data (my book)"] if (LIVE_WS / "portfolio.yaml").exists() else []) \
        + ["examples/demo (synthetic)"]
    choice = st.selectbox("Book", options, help="A workspace is a folder with portfolio.yaml, trades.csv, "
                                                "cashflows.csv and backtests/. Create yours with `python -m pcon init`.")
    if choice.startswith("examples/demo"):
        ws = DEMO_WS
        if not (ws / "portfolio.yaml").exists():
            with st.spinner("Generating the synthetic demo workspace ..."):
                build_demo(ws)
    elif choice.startswith("data"):
        ws = LIVE_WS
    else:
        ws = Path(choice).expanduser().resolve()
    if not (LIVE_WS / "portfolio.yaml").exists() and not cli:
        st.caption("No personal workspace yet.")
        if st.button("Create my workspace in data/", width="stretch"):
            init_workspace(LIVE_WS)
            st.rerun()
    if "refresh" not in st.session_state:
        st.session_state.refresh = 0
    if st.button("Refresh market data", width="stretch", help="Re-download prices from Yahoo Finance"):
        st.session_state.refresh += 1
        st.cache_data.clear()
    st.divider()

try:
    book = load_book(str(ws), signature(ws), st.session_state.refresh, str(date.today()))
except Exception as exc:  # config / journal errors are the user's to fix: show them plainly
    st.error(f"Could not load the workspace at `{ws}`:\n\n{exc}")
    st.stop()

cfg, L = book.cfg, book.ledger
SIDS = book.strategy_ids
LABELS = {s: book.label(s) for s in SIDS} | {CASH_SLEEVE: "Unallocated"}
COLORS = {s: cfg.color(s) for s in SIDS} | {CASH_SLEEVE: "#898781"}
cur = "$" if cfg.currency == "USD" else cfg.currency + " "

with st.sidebar:
    st.header("Portfolio construction")
    source = st.radio("Return history", ["backtest", "backtest+live", "live"], index=0,
                      format_func={"backtest": "Backtests", "backtest+live": "Backtests, then live",
                                   "live": "Live only"}.get,
                      help="Which return series the allocation analysis uses. Backtests are the long, honest "
                           "out-of-sample exports; 'then live' swaps in your real record from the day you went live.")
    rebalance = st.selectbox("Rebalance sleeves", ["M", "Q", "A", "D"], index=["M", "Q", "A", "D"].index(
        cfg.allocation.rebalance if cfg.allocation.rebalance in ("M", "Q", "A", "D") else "M"),
        format_func={"M": "Monthly", "Q": "Quarterly", "A": "Annually", "D": "Daily (constant mix)"}.get)
    max_w = st.slider("Max weight per strategy", 0.2, 1.0, float(cfg.allocation.max_weight), 0.05)
    wf_years = st.slider("Walk-forward estimation window (years)", 1.0, 5.0,
                         float(cfg.allocation.walk_forward_lookback_years), 0.5)
    st.caption("Settings here only change the analysis. Your target weights live in portfolio.yaml.")

# ------------------------------------------------------------------------------------------------
# header
# ------------------------------------------------------------------------------------------------
asof = L.dates[-1] if not L.empty else None
st.title(cfg.name)
st.caption(" · ".join(x for x in [
    f"as of {asof:%a %d %b %Y}" if asof is not None else "no live record yet",
    f"{len(SIDS)} strategies", f"benchmark {cfg.benchmark}", f"workspace `{ws.relative_to(ROOT) if ws.is_relative_to(ROOT) else ws}`"]))
if cfg.demo:
    st.warning("**Demo workspace.** Prices, backtests and trades here are synthetic (generated by `pcon/demo.py`) "
               "and say nothing about the real strategies or markets. Switch to *data (my book)* once you have "
               "created your workspace.", icon="🧪")

health = book.health()
n_red = sum(h["level"] == "red" for h in health)
n_amb = sum(h["level"] == "amber" for h in health)
with st.expander(f"Health checks: {LEVEL_ICON['red']} {n_red} red · {LEVEL_ICON['amber']} {n_amb} amber · "
                 f"{sum(h['level'] == 'green' for h in health)} green", expanded=bool(n_red)):
    for h in health:
        st.markdown(f"{LEVEL_ICON[h['level']]} **{h['scope']}**: {h['msg']}")

tabs = st.tabs(["Overview", "Strategies", "Expected vs actual", "Portfolio construction", "Risk",
                "Positions & trades", "Transactions", "Journal & data"])


# ------------------------------------------------------------------------------------------------
# 1. overview
# ------------------------------------------------------------------------------------------------
def page_overview():
    if L.empty:
        st.info("No live record yet. Add a deposit for each strategy and your first fills in the **Journal & data** "
                "tab (or edit trades.csv / cashflows.csv). The construction and expectation views already work "
                "from the backtests.")
        return
    k = book.kpis()
    pr = book.portfolio_returns()
    live_sr = M.sharpe(pr, book.rf) if len(pr) > 20 else np.nan
    mdd = M.max_drawdown(pr)
    cdd = M.drawdown(pr).iloc[-1] if len(pr) else np.nan
    c = st.columns(6)
    c[0].metric("NAV", compact(k["NAV"], cur), f"{compact(k['Day P&L'], cur)} today", border=True,
                help=f"{money(k['NAV'], cur)} · net invested {money(k['Net invested'], cur)}")
    c[1].metric("P&L since start", compact(k["P&L (ITD)"], cur), pct(k["ITD (TWR)"]) + " TWR", border=True,
                help="Money P&L, and the time-weighted return (deposits and withdrawals removed).")
    c[2].metric("Year to date", pct(k["YTD"]), f"MTD {pct(k['MTD'])}", border=True)
    c[3].metric("Sharpe (live)", num(live_sr), border=True,
                help="Annualised, in excess of T-bills. Needs many months of data to mean much: see Expected vs actual.")
    c[4].metric("Drawdown", pct(cdd), f"max {pct(mdd)}", delta_color="off", border=True)
    c[5].metric("Gross exposure", pct(k["Gross exposure"], 0), f"cash {compact(k['Cash'], cur)}", delta_color="off",
                border=True)

    left, right = st.columns([2.1, 1])
    with left:
        view = st.radio("Equity curve", ["Return (time-weighted)", f"NAV ({cfg.currency})"], horizontal=True,
                        label_visibility="collapsed")
        if view.startswith("Return"):
            series = {"Portfolio": (pr, "portfolio")}
            series |= {LABELS[s]: (r, COLORS[s]) for s, r in book.live_returns().items() if s in LABELS}
            if len(book.bench):
                series[cfg.benchmark] = (book.bench.loc[pr.index[0]:pr.index[-1]], "benchmark")
            chart(C.growth(series, T, "Cumulative return, live", height=360), "ov_growth")
            chart(C.drawdowns({k_: v for k_, v in series.items()}, T, height=210), "ov_dd")
        else:
            chart(C.nav_chart(L.total["nav"], L.total["flows"], T, "NAV and net invested capital"), "ov_nav")
    with right:
        w_now = L.weights().iloc[-1]
        chart(C.weights_vs_target(w_now, cfg.target_weights(), COLORS, LABELS, T, "Allocation: actual vs target",
                                  height=60 + 46 * max(len(w_now), 1)), "ov_w")
        rb = book.rebalance_orders()
        if len(rb) and (rb["Transfer $"].abs() > cfg.alerts.weight_drift * rb["Current $"].sum()).any():
            st.caption("Drift beyond the band: see *Portfolio construction → Rebalance*.")
        st.markdown("**Strategy health**")
        for h in [h for h in health if h["scope"] in LABELS.values() or h["scope"] == "Portfolio"]:
            st.markdown(f"{LEVEL_ICON[h['level']]} **{h['scope']}**: {h['msg']}")

    st.subheader("Scoreboard")
    rows = []
    nav_now = L.nav.iloc[-1]
    tgt = cfg.target_weights()
    for s in L.sleeves:
        r = book.live_returns().get(s, pd.Series(dtype=float))
        cmp_ = book.comparison(s) if s in cfg.strategies else None
        tr = book.tracking(s) if s in cfg.strategies else None
        op = L.open_positions
        rows.append({
            "Strategy": LABELS.get(s, s), "NAV": nav_now[s], "Weight": nav_now[s] / nav_now.sum(),
            "Target": tgt.get(s, np.nan), "Return (live)": (1 + r).prod() - 1 if len(r) else np.nan,
            "Volatility": r.std() * np.sqrt(252) if len(r) > 2 else np.nan,
            "Sharpe": M.sharpe(r, book.rf) if len(r) > 20 else np.nan,
            "Max DD": M.max_drawdown(r) if len(r) else np.nan,
            "Open positions": int((op["strategy"] == s).sum()) if len(op) else 0,
            "Pctile vs expected": cmp_.table.loc["Return", "Percentile"] if cmp_ else np.nan,
            "Status": STATUS_ICON.get(cmp_.table.loc["Return", "Status"], "–") if cmp_ else "–",
            "vs model": tr["Implementation shortfall"] if tr else np.nan,
            "Final tests": tests_label(s) if s in cfg.strategies else "–"})
    df = pd.DataFrame(rows).set_index("Strategy")
    st.dataframe(df.style.format({"NAV": lambda v: money(v, cur), "Weight": "{:.1%}", "Target": "{:.0%}",
                                  "Return (live)": "{:.2%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}",
                                  "Max DD": "{:.1%}", "Pctile vs expected": "{:.0%}",
                                  "vs model": "{:+.2%}"}, na_rep="–"))
    st.caption("*Pctile vs expected*: where the live return sits among same-length backtest paths (50% = as "
               "expected). *vs model*: live minus the backtest on the same days (execution shortfall). *Final "
               "tests*: how many of the research notebook's final tests the exported configuration passed.")


# ------------------------------------------------------------------------------------------------
# 2. strategies
# ------------------------------------------------------------------------------------------------
def page_strategies():
    st.subheader("Live performance")
    ls = book.live_summary()
    if ls.empty:
        st.info("No live returns yet.")
    else:
        if len(book.bench) and len(book.portfolio_returns()):
            pr = book.portfolio_returns()
            ls[cfg.benchmark] = pd.Series(M.perf_summary(book.bench.loc[pr.index[0]:pr.index[-1]], book.rf))
        metric_table(ls, ["Start", "Years", "Total return", "CAGR", "Volatility", "Sharpe", "Sortino",
                          "Max drawdown", "Current drawdown", "Calmar", "Days since peak", "VaR 95% (1d)",
                          "CVaR 95% (1d)", "Worst day", "Best day", "% positive days", "Beta", "Alpha (ann.)",
                          "Correlation", "Up capture", "Down capture", "PSR (SR>0)"])

    sid = st.radio("Strategy", SIDS, format_func=lambda s: cfg.strategies[s].name, horizontal=True)
    s_cfg = cfg.strategies[sid]
    st.caption(" · ".join(x for x in [s_cfg.description, s_cfg.universe] if x))
    selection_panel(sid)
    live = book.live_returns().get(sid)
    model = book.model_on_live_dates(sid)
    if live is not None and len(live) > 1:
        a, b = st.columns(2)
        series = {"Live": (live, s_cfg.color)}
        if model is not None:
            series["Model (backtest on live dates)"] = (model, C.MODEL)
        if len(book.bench):
            series[cfg.benchmark] = (book.bench.loc[live.index[0]:live.index[-1]], "benchmark")
        with a:
            chart(C.growth(series, T, f"{s_cfg.short}: live vs model", height=320), f"st_g_{sid}")
        with b:
            chart(C.drawdowns({"Live": (live, s_cfg.color)} | ({"Model": (model, C.MODEL)} if model is not None
                                                               else {}), T, f"{s_cfg.short}: drawdown",
                              height=320), f"st_dd_{sid}")
        a, b = st.columns(2)
        with a:
            chart(C.monthly_heatmap(M.monthly_table(live), T, "Monthly returns (live)"), f"st_hm_{sid}")
        with b:
            ex = (L.long_exposure[sid] - L.short_exposure[sid]) / L.nav[sid].where(L.nav[sid] > 0)
            chart(C.lines(pd.DataFrame({"Gross exposure": ex.loc[live.index[0]:]}), {"Gross exposure": s_cfg.color},
                          T, "Exposure (share of sleeve NAV)", yfmt=".0%", height=240), f"st_ex_{sid}")
    else:
        st.info(f"No live record for {s_cfg.short} yet.")

    op = L.open_positions
    if len(op) and (op["strategy"] == sid).any():
        st.markdown("**Open positions**")
        o = op[op["strategy"] == sid].drop(columns=["strategy"]).set_index("symbol")
        st.dataframe(o.style.format({"quantity": "{:,.4g}", "avg_cost": "{:,.2f}", "last": "{:,.2f}",
                                     "market_value": "{:,.0f}", "unrealized": "{:+,.0f}", "unrealized_pct": "{:+.2%}",
                                     "realized": "{:+,.0f}", "weight_sleeve": "{:.1%}",
                                     "entry_date": "{:%Y-%m-%d}"}, na_rep="–"))
    rt = L.round_trips
    if len(rt) and (rt["strategy"] == sid).any():
        g = rt[rt["strategy"] == sid]
        a, b = st.columns([1, 2.2])
        with a:
            st.markdown("**Trade statistics (live)**")
            ts = pd.Series(M.trade_summary(g))
            st.dataframe(pd.DataFrame({"value": [M.fmt(k_, v) for k_, v in ts.items()]}, index=ts.index))
        with b:
            st.markdown("**Closed trades**")
            st.dataframe(g.drop(columns=["strategy"]).sort_values("exit_date", ascending=False).style.format(
                {"entry_date": "{:%Y-%m-%d}", "exit_date": "{:%Y-%m-%d}", "quantity": "{:,.4g}",
                 "entry_price": "{:,.2f}", "exit_price": "{:,.2f}", "pnl": "{:+,.0f}", "ret": "{:+.2%}",
                 "fees": "{:,.2f}"}), hide_index=True)

    bt = book.backtest_returns(sid)
    with st.expander(f"Backtest tearsheet: {s_cfg.short} (the expectation)", expanded=live is None):
        if bt is None:
            st.info("No backtest export for this strategy. See `strategy_exports/README.md`.")
        else:
            exp_start = book.expectation_start(sid)
            st.caption(f"Export covers {bt.index[0]:%Y-%m-%d} → {bt.index[-1]:%Y-%m-%d}. Expectation window starts "
                       f"{(exp_start or bt.index[0]):%Y-%m-%d} (set `expectation_start` to the honest "
                       "out-of-sample start).")
            e = book.expected(sid)
            ser = {s_cfg.short: (bt.loc[exp_start:] if exp_start is not None else bt, s_cfg.color)}
            if len(book.bench):
                b0 = ser[s_cfg.short][0].index
                ser[cfg.benchmark] = (book.bench.loc[b0[0]:b0[-1]], "benchmark")
            a, b = st.columns([1.6, 1])
            with a:
                chart(C.growth(ser, T, "Backtest growth of 1 (log)", log=True, height=320), f"bt_g_{sid}")
                chart(C.drawdowns(ser, T, height=200), f"bt_dd_{sid}")
            with b:
                metric_table(M.summary_frame({"Backtest": e, **({cfg.benchmark: book.bench.loc[e.index[0]:e.index[-1]]}
                                                                 if len(book.bench) and e is not None and len(e) else {})},
                                             book.rf, book.bench),
                             ["Start", "End", "Years", "CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown",
                              "Calmar", "Longest drawdown (days)", "Worst month", "% positive months", "Beta",
                              "Correlation", "PSR (SR>0)"])
            st.markdown("**Worst drawdowns (expectation window)**")
            ddt = M.drawdown_table(e, 5)
            if len(ddt):
                st.dataframe(ddt.style.format({"peak": "{:%Y-%m-%d}", "trough": "{:%Y-%m-%d}",
                                               "recovery": lambda v: "not yet" if pd.isna(v) else f"{v:%Y-%m-%d}",
                                               "depth": "{:.1%}"}), hide_index=True)
            chart(C.monthly_heatmap(M.monthly_table(e), T, "Monthly returns (backtest)"), f"bt_hm_{sid}")


# ------------------------------------------------------------------------------------------------
# 3. expected vs actual
# ------------------------------------------------------------------------------------------------
def page_expected():
    st.markdown("Each live metric is placed inside the distribution of the same metric over thousands of "
                "**same-length paths resampled from the backtest** (63-day blocks). Around the 50th percentile = "
                "exactly as expected; below the 5th = worse than 95 % of what the backtest normally produces.")
    scope = st.radio("Scope", ["__portfolio__"] + SIDS, horizontal=True,
                     format_func=lambda s: "Portfolio (target mix)" if s == "__portfolio__" else LABELS[s])
    sid = None if scope == "__portfolio__" else scope
    color = "portfolio" if sid is None else COLORS[sid]
    live = book.portfolio_returns() if sid is None else book.live_returns().get(sid)
    exp = book.expected_portfolio() if sid is None else book.expected(sid)
    if exp is None or len(exp) < 60:
        st.info("No usable backtest expectation for this scope yet: export the strategy's backtest into "
                "`backtests/` (see `strategy_exports/`).")
        return
    if live is None or len(live) < 2:
        st.info("No live record yet: showing what to expect over the next 12 months.")
        fwd = E.forward_cone(exp, pd.Timestamp.today(), 1.0)
        chart(C.cone(pd.DataFrame(), None, T, color, "Expected range, next 12 months", forward=fwd), "ex_fwd_only")
        fr = E.forward_risk(exp)
        st.dataframe(pd.Series({k_: pct(v) for k_, v in fr.items()}, name="next 12 months"))
        return
    cmp_ = book.comparison(sid)
    a, b = st.columns([1.7, 1])
    with a:
        show_fwd = st.toggle("Project the cone 12 months forward", value=True)
        cn = E.cone(exp, live.index)
        last_w = float((1 + live).prod())
        fwd = E.forward_cone(exp, live.index[-1], last_w) if show_fwd else None
        chart(C.cone(cn, live, T, color, "Live path inside the backtest's expectation cone", forward=fwd,
                     height=400), f"ex_cone_{scope}")
    with b:
        t = cmp_.table
        st.markdown(f"**After {cmp_.horizon} trading days**")
        for key, f in (("Return", "{:+.1%}"), ("Max drawdown", "{:.1%}"), ("Sharpe", "{:.2f}")):
            st.markdown(f"{STATUS_ICON[t.loc[key, 'Status']].split(' ')[0]} **{key}** {f.format(t.loc[key, 'Actual'])} "
                        f"· expected {f.format(t.loc[key, 'Expected'])} "
                        f"(90%: {f.format(t.loc[key, 'Low (5%)'])} to {f.format(t.loc[key, 'High (95%)'])}) "
                        f"· **{pct(t.loc[key, 'Percentile'], 0)}** percentile")
        stt = cmp_.sharpe_test
        if stt:
            st.markdown("**Is the live Sharpe consistent with the backtest?**")
            st.markdown(
                f"Live **{num(stt['Live Sharpe'])}** vs backtest **{num(stt['Backtest Sharpe'])}** "
                f"(standard error of the live estimate ±{num(stt['Std error (live)'])}). "
                f"z = {num(stt['z-score'])}, p = {num(stt['p-value (two-sided)'])}: "
                + ("**no evidence the edge has changed**." if stt["p-value (two-sided)"] > 0.05 else
                   "**the live Sharpe is statistically different from the backtest**.")
                + f" With the backtest's Sharpe you need about **{num(stt['Years needed to confirm SR>0 (95%)'], 1)} "
                  f"years** of live data to prove SR > 0 at 95 % confidence (you have {num(stt['Years live'], 2)}).")
    disp = pd.DataFrame({
        "Expected (median)": [M.fmt(k_ if k_ != "Return" else "Total return", v) for k_, v in t["Expected"].items()],
        "90% range": [f"{M.fmt(k_ if k_ != 'Return' else 'Total return', lo)} to "
                      f"{M.fmt(k_ if k_ != 'Return' else 'Total return', hi)}"
                      for k_, lo, hi in zip(t.index, t["Low (5%)"], t["High (95%)"])],
        "Actual": [M.fmt(k_ if k_ != "Return" else "Total return", v) for k_, v in t["Actual"].items()],
        "Percentile": [pct(v, 0) for v in t["Percentile"]],
        "Status": t["Status"].map(STATUS_ICON).to_numpy()}, index=t.index)
    st.dataframe(disp)
    c = st.columns(3)
    for col, key, fmt in zip(c, ["Return", "Max drawdown", "Sharpe"], [".0%", ".0%", ".1f"]):
        with col:
            chart(C.distribution(cmp_.distributions[key], cmp_.table.loc[key, "Actual"], T,
                                 f"{key}: backtest paths vs actual", fmt, color if color != "portfolio" else "#2a78d6"),
                  f"ex_dist_{scope}_{key}")

    a, b = st.columns(2)
    with a:
        st.markdown("**Backtest profile vs live (annualised)**")
        prof = pd.DataFrame({"Backtest (expected)": cmp_.expected_profile,
                             "Live (actual)": M.perf_summary(live, book.rf)})
        metric_table(prof, ["Years", "CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar",
                            "% positive days", "% positive months", "Worst day", "Skew", "Excess kurtosis"])
    with b:
        st.markdown("**What to expect over the next 12 months** (from the backtest)")
        fr = E.forward_risk(exp)
        nav_now = (L.total["nav"].iloc[-1] if sid is None else L.nav[sid].iloc[-1])
        st.dataframe(pd.DataFrame({"share": [pct(v) for v in fr.values()],
                                   cfg.currency: [money(v * nav_now, cur) if "P(" not in k_ else "–"
                                                  for k_, v in fr.items()]}, index=list(fr)))
    if sid is not None:
        st.subheader("Live vs model: execution quality")
        tr = book.tracking(sid)
        if tr is None:
            st.info("The backtest export does not cover the live period yet. Re-run the strategy notebook and its "
                    "export cell after going live: the dashboard then compares every live day with the model's day "
                    "and isolates slippage, missed or late signals and sizing differences.")
        else:
            a, b = st.columns([1.7, 1])
            with a:
                s = tr["series"]
                chart(C.lines(pd.DataFrame({"Live": s["live"] - 1, "Model": s["model"] - 1,
                                            "Shortfall (live vs model)": s["shortfall"]}),
                              {"Live": COLORS[sid], "Model": C.MODEL, "Shortfall (live vs model)": "ink"}, T,
                              "Cumulative return: live vs model on the same days", yfmt=".1%", height=320, zero=True),
                      f"ex_tr_{sid}")
            with b:
                st.dataframe(pd.Series({k_: (pct(v, 2) if k_ not in ("days", "start", "end", "Correlation",
                                                                      "Beta to model")
                                             else (num(v) if k_ in ("Correlation", "Beta to model") else
                                                   (f"{v:%Y-%m-%d}" if isinstance(v, pd.Timestamp) else str(v))))
                                        for k_, v in tr.items() if k_ != "series"}, name="value"))
                st.caption("Shortfall < 0: you earned less than the model on the same days. Correlation well below "
                           "0.95 usually means trades were missed, late, or sized differently.")


# ------------------------------------------------------------------------------------------------
# 4. portfolio construction
# ------------------------------------------------------------------------------------------------
def page_construction():
    R = book.alloc_matrix(source)
    if R.shape[1] < 2 or len(R) < 300:
        st.info("Portfolio construction needs at least two strategies with overlapping daily returns (about a year "
                "or more). Export the backtests into `backtests/` (see `strategy_exports/`)."
                + (f" Currently: {R.shape[1]} strategies, {len(R)} common days." if len(R) else ""))
        return
    rf = book.rf.reindex(R.index).fillna(0.0)
    target = cfg.target_weights().reindex(R.columns).fillna(0.0)
    wf_days = int(wf_years * 252)
    st.caption(f"Common window **{R.index[0]:%Y-%m-%d} → {R.index[-1]:%Y-%m-%d}** ({len(R) / 252:.1f} years, "
               f"{len(R):,} days) · strategies: {', '.join(LABELS[c] for c in R.columns)} · source: {source} · "
               f"{ {'M': 'monthly', 'Q': 'quarterly', 'A': 'annual', 'D': 'daily'}[rebalance]} rebalancing · "
               f"max weight {max_w:.0%}. Walk-forward = weights estimated on the trailing {wf_years:g} years only, "
               "held for the next quarter.")

    W, stats_m, wf = cached_methods(R, rf, max_w, target, rebalance, wf_days, "Q", cfg.allocation.bootstrap_samples)
    best_wf = stats_m["WF Sharpe"].astype(float).idxmax()
    rec = "resampled" if stats_m.loc["resampled", "WF Sharpe"] >= stats_m.loc[best_wf, "WF Sharpe"] - 0.05 else best_wf

    # ---- recommendation ----------------------------------------------------------------------
    st.subheader("Recommendation")
    r_t = A.portfolio_returns(R, W.loc["target"], rebalance)
    r_r = A.portfolio_returns(R, W.loc[rec], rebalance)
    test = cached_sharpe_test(r_r, r_t, rf)
    mg = cached_marginal(R, rf, "target", rebalance, max_w, target)
    drop = [LABELS[i] for i, v in mg["Verdict"].items() if str(v).startswith("Removal")]
    c = st.columns([1.2, 1])
    with c[0]:
        st.markdown(
            f"- **Allocation method:** *{A.METHODS[rec]}* "
            f"(walk-forward Sharpe {num(stats_m.loc[rec, 'WF Sharpe'])} vs {num(stats_m.loc['target', 'WF Sharpe'])} "
            f"for your targets; the best out-of-sample method was *{A.METHODS[best_wf]}*).\n"
            f"- **Weights:** " + ", ".join(f"{LABELS[k_]} **{v:.0%}**" for k_, v in W.loc[rec].items()) + "\n"
            f"- **vs your targets:** Sharpe {num(stats_m.loc[rec, 'Sharpe'])} vs {num(stats_m.loc['target', 'Sharpe'])}, "
            f"max drawdown {pct(stats_m.loc[rec, 'Max drawdown'])} vs {pct(stats_m.loc['target', 'Max drawdown'])}"
            + (f"; the improvement is {'statistically meaningful' if test.get('p_not_better', 1) < 0.1 else 'not statistically significant'} "
               f"(P(no improvement) = {pct(test.get('p_not_better'), 0)})." if test else ".") + "\n"
            + (f"- **Strategies the data would drop:** {', '.join(drop)} (see the add/remove table: removal is "
               "only worth acting on when it is significant)." if drop else
               "- **Every strategy earns its place** at your target weights (each one raises the portfolio Sharpe).")
        )
        st.caption("Optimised weights are estimates. Prefer the walk-forward columns and the resampled allocation "
                   "over in-sample maximum-Sharpe weights, which overfit.")
    with c[1]:
        yaml_snip = "\n".join(f"  {k_}:\n    target_weight: {v:.2f}" for k_, v in W.loc[rec].items())
        st.markdown("To adopt it, set in `portfolio.yaml`:")
        st.code("strategies:\n" + yaml_snip, language="yaml")

    # ---- which strategies --------------------------------------------------------------------
    st.subheader("1 · Which strategies belong in the book?")
    st.markdown("Adding a little of strategy *k* raises the portfolio Sharpe if **Sharpe(k) > correlation(k, rest) × "
                "Sharpe(rest)** (the *hurdle*). The table also shows the actual Sharpe of the portfolio with and "
                "without each strategy (at your target weights, renormalised) and a paired-bootstrap probability "
                "that including it does **not** help.")
    mgd = mg.copy()
    mgd.index = [LABELS[i] for i in mgd.index]
    st.dataframe(mgd.style.format({"Sharpe alone": "{:.2f}", "CAGR alone": "{:.1%}", "Max DD alone": "{:.1%}",
                                   "Corr to rest": "{:.2f}", "Hurdle Sharpe": "{:.2f}", "Sharpe without": "{:.2f}",
                                   "Sharpe with": "{:.2f}", "Δ Sharpe": "{:+.2f}", "Δ CAGR": "{:+.1%}",
                                   "Δ Max DD": "{:+.1%}", "Weight": "{:.0%}", "P(no improvement)": "{:.0%}"},
                                  na_rep="–"))
    st.caption("Δ columns = with minus without the strategy. Δ Max DD > 0 means a shallower drawdown with it. "
               "*P(no improvement)* below 10% = the gain is statistically meaningful; above 90% = removing it is.")
    methods = ("target", "equal", "inverse_vol", "risk_parity", "min_variance", "max_sharpe", "resampled")
    ex = cached_explore(R, rf, methods, rebalance, max_w, target, wf_days, "Q")
    obj = st.selectbox("Rank combinations by", ["WF Sharpe", "Sharpe", "WF CAGR", "CAGR", "Calmar",
                                                 "WF Max drawdown", "Max drawdown", "Sortino"], index=0)
    exd = ex.copy()
    exd["subset"] = exd["subset"].apply(lambda s: " + ".join(LABELS[x] for x in s.split(" + ")))
    exd["method"] = exd["method"].map(lambda m: "—" if m == "single" else A.METHODS.get(m, m))
    exd = exd.sort_values(obj, ascending=False).rename(columns=LABELS)
    best_by_subset = exd.groupby("subset", sort=False).head(1).set_index("subset")[obj]
    a, b = st.columns([1, 1.6])
    with a:
        chart(C.hbar(best_by_subset, T, f"Best {obj} per combination", fmt=".2f" if "Sharpe" in obj or obj in (
            "Calmar", "Sortino") else ".1%", colors=[T["ink2"]] * len(best_by_subset)), "pc_subsets")
    with b:
        pc = {c_: "{:.0%}" for c_ in LABELS.values()} | {k_: "{:.1%}" for k_ in ("CAGR", "Volatility", "Max drawdown",
                                                                                  "Worst month", "% positive months",
                                                                                  "WF CAGR", "WF Max drawdown")}
        pc |= {k_: "{:.2f}" for k_ in ("Sharpe", "Sortino", "Calmar", "WF Sharpe")}
        st.dataframe(exd.drop(columns=["n"]).style.format(pc, na_rep="–"), hide_index=True, height=420)

    # ---- how much each -------------------------------------------------------------------------
    st.subheader("2 · How much capital to each?")
    a, b = st.columns([1, 1.25])
    with a:
        Wd = W.rename(index=A.METHODS)
        chart(C.weights_by_method(Wd, COLORS, LABELS, T, "Weights by allocation method", height=340), "pc_wm")
    with b:
        sm = stats_m.rename(index=A.METHODS)[["CAGR", "Volatility", "Sharpe", "Max drawdown", "Calmar",
                                              "WF CAGR", "WF Sharpe", "WF Max drawdown"]]
        st.dataframe(sm.style.format({"CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}",
                                      "Max drawdown": "{:.1%}", "Calmar": "{:.2f}", "WF CAGR": "{:.1%}",
                                      "WF Sharpe": "{:.2f}", "WF Max drawdown": "{:.1%}"}, na_rep="–")
                     .highlight_max(subset=["WF Sharpe"], color=C._rgba("#2a78d6", 0.18)), height=320)
        st.caption("In-sample columns use weights fitted on the whole window (optimistic). WF columns are the honest "
                   "out-of-sample record of each method.")
    a, b = st.columns(2)
    with a:
        fr, cloud = A.efficient_frontier(R, rf, 0.0, max_w)
        mu, cov = A.estimate(R, rf)
        pts = {LABELS[c_]: (float(np.sqrt(cov[i, i])), float(mu[i]), COLORS[c_]) for i, c_ in enumerate(R.columns)}
        for m, col in (("target", "portfolio"), (rec, C.MODEL)):
            w = W.loc[m].to_numpy()
            pts[A.METHODS[m]] = (float(np.sqrt(w @ cov @ w)), float(w @ mu), col)
        chart(C.frontier(fr, cloud, pts, T, "Risk / return of every long-only mix"), "pc_front")
    with b:
        ser = {"Your targets": (r_t, "portfolio"), A.METHODS[rec]: (r_r, C.MODEL)}
        ser |= {LABELS[c_]: (R[c_], COLORS[c_]) for c_ in R.columns}
        chart(C.growth(ser, T, "Growth of 1 over the window (log)", log=True, height=400), "pc_growth")

    # ---- risk budget ---------------------------------------------------------------------------
    st.subheader("3 · Diversification and risk budget")
    a, b, c3 = st.columns(3)
    with a:
        chart(C.corr_heatmap(A.correlations(R, "M"), LABELS, T, "Correlation (monthly returns)"), "pc_corr_m")
    with b:
        sc = A.stress_correlation(R, book.bench, 0.1) if len(book.bench) else pd.DataFrame()
        if len(sc):
            chart(C.corr_heatmap(sc, LABELS, T, f"Correlation on {cfg.benchmark}'s worst 10% days"), "pc_corr_s")
        else:
            chart(C.corr_heatmap(A.correlations(R, "D"), LABELS, T, "Correlation (daily)"), "pc_corr_d")
    with c3:
        rc = pd.concat({"Your targets": A.risk_contributions(R, W.loc["target"])["% of risk"],
                        A.METHODS[rec]: A.risk_contributions(R, W.loc[rec])["% of risk"]}, axis=1).T
        chart(C.weights_by_method(rc, COLORS, LABELS, T, "Share of portfolio volatility", height=300), "pc_rc")
    rcor = A.rolling_correlations(R, 126)
    rcor.columns = [" / ".join(LABELS[x] for x in c_.split(" / ")) for c_ in rcor.columns]
    pair_colors = dict(zip(rcor.columns, ["#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948", "#52514e"]))
    chart(C.lines(rcor, pair_colors, T, "Rolling 6-month correlation between strategies", yfmt=".1f", height=260,
                  zero=True), "pc_rcorr")
    dc = A.drawdown_contributions(R, W.loc["target"], rebalance)
    if len(dc):
        st.caption(f"Worst drawdown of the target mix ({dc['peak'].iloc[0]:%Y-%m-%d} → {dc['trough'].iloc[0]:%Y-%m-%d}): "
                   + ", ".join(f"{LABELS[i]} {v:.0%}" for i, v in dc["% of drawdown"].items()) + " of the loss.")

    st.markdown("**Stress windows** (each strategy's full backtest where it covers the window)")
    full = {LABELS[s]: book.backtest_returns(s) for s in SIDS if book.backtest_returns(s) is not None}
    full["Target mix"] = r_t
    if len(book.bench):
        full[cfg.benchmark] = book.bench
    stt = A.stress_test(full).astype(float)
    st.dataframe(stt.style.format("{:.1%}", na_rep="–").map(diverging_bg, lim=0.4) if len(stt) else stt)

    # ---- rebalance -----------------------------------------------------------------------------
    st.subheader("4 · Rebalance")
    if L.empty:
        st.info("No live sleeves yet.")
    else:
        which = st.radio("Bring sleeves to", ["Target weights (portfolio.yaml)", f"Recommended ({A.METHODS[rec]})"],
                         horizontal=True)
        tw = cfg.target_weights() if which.startswith("Target") else W.loc[rec]
        orders = book.rebalance_orders(tw)
        a, b = st.columns([1.3, 1])
        with a:
            st.dataframe(orders.style.format({"Current $": "{:,.0f}", "Current weight": "{:.1%}",
                                              "Target weight": "{:.1%}", "Target $": "{:,.0f}",
                                              "Transfer $": "{:+,.0f}"}))
            st.caption("Move cash between sleeves with a `transfer` row per sleeve in cashflows.csv, then let each "
                       "strategy size its next trades from its new sleeve NAV.")
        with b:
            chart(C.hbar(orders["Transfer $"], T, "Transfers needed", height=60 + 40 * len(orders)), "pc_orders")


# ------------------------------------------------------------------------------------------------
# 5. risk
# ------------------------------------------------------------------------------------------------
def page_risk():
    if L.empty:
        st.info("No live positions yet.")
        return
    nav_tot = L.total["nav"]
    gross = (L.long_exposure - L.short_exposure).div(nav_tot.where(nav_tot > 0), axis=0)
    a, b = st.columns([1.3, 1])
    with a:
        chart(C.exposure(gross[[c_ for c_ in gross.columns if gross[c_].abs().sum() > 0]], COLORS, LABELS, T,
                         "Market exposure by strategy (share of total NAV)"), "rk_exp")
    with b:
        st.markdown("**Value at risk (1 day)**")
        nav_now = nav_tot.iloc[-1]
        pr = book.portfolio_returns()
        rows = {}
        if len(pr) >= 20:
            rows["Live record"] = {"VaR 95%": pr.quantile(0.05), "CVaR 95%": pr[pr <= pr.quantile(0.05)].mean(),
                                   "VaR 99%": pr.quantile(0.01)}
        ep = book.expected_portfolio()
        if ep is not None and len(ep) > 250:
            rows["Backtest of target mix"] = {"VaR 95%": ep.quantile(0.05),
                                              "CVaR 95%": ep[ep <= ep.quantile(0.05)].mean(),
                                              "VaR 99%": ep.quantile(0.01)}
        if rows:
            v = pd.DataFrame(rows).T
            st.dataframe(pd.concat({"% of NAV": v.map(lambda x: pct(x, 2)),
                                    cfg.currency: v.map(lambda x: money(x * nav_now, cur))}, axis=1))
            st.caption("Historical simulation. The backtest row uses years of data and is the more reliable one "
                       "until the live record is long.")
        if ep is not None:
            fr = E.forward_risk(ep)
            st.markdown(f"Next 12 months (backtest of the target mix): probability of a loss **{pct(fr['P(loss)'], 0)}**, "
                        f"median max drawdown **{pct(fr['Median max drawdown'])}**, 1-in-20 max drawdown "
                        f"**{pct(fr['5% worst max drawdown'])}** (≈ {money(fr['5% worst max drawdown'] * nav_now, cur)}).")

    st.markdown("**Holdings netted across strategies**")
    op = L.open_positions
    if len(op):
        net = op.groupby("symbol").agg(quantity=("quantity", "sum"), market_value=("market_value", "sum"),
                                       unrealized=("unrealized", "sum"),
                                       strategies=("strategy", lambda s: ", ".join(LABELS.get(x, x) for x in s)))
        net["% of NAV"] = net["market_value"] / nav_tot.iloc[-1]
        net = net.sort_values("market_value", ascending=False)
        a, b = st.columns([1.4, 1])
        with a:
            st.dataframe(net.style.format({"quantity": "{:,.4g}", "market_value": "{:,.0f}", "unrealized": "{:+,.0f}",
                                           "% of NAV": "{:.1%}"}))
        with b:
            chart(C.hbar(net["% of NAV"], T, "Position size (% of total NAV)", fmt=".1%",
                         colors=[T["ink2"]] * len(net)), "rk_conc")
    else:
        st.caption("No open positions.")

    lr = pd.DataFrame(book.live_returns())
    pr = book.portfolio_returns()
    if len(pr) >= 63:
        a, b = st.columns(2)
        with a:
            rv = pd.DataFrame({"Portfolio": M.rolling_vol(pr, 63)} | {LABELS.get(s, s): M.rolling_vol(r, 63)
                                                                       for s, r in book.live_returns().items()})
            chart(C.lines(rv, {"Portfolio": "portfolio"} | {LABELS[s]: COLORS[s] for s in SIDS}, T,
                          "Rolling 3-month volatility", yfmt=".0%"), "rk_vol")
        with b:
            if len(book.bench):
                rb = pd.DataFrame({"Portfolio": M.rolling_beta(pr, book.bench, 63)})
                chart(C.lines(rb, {"Portfolio": "portfolio"}, T, f"Rolling 3-month beta to {cfg.benchmark}",
                              yfmt=".2f", zero=True), "rk_beta")
    else:
        st.caption("Rolling volatility and beta appear after 3 months of live data.")
    if lr.shape[1] > 1 and len(lr.dropna()) >= 40:
        lc = lr.dropna().corr()
        a, _ = st.columns([1, 1.4])
        with a:
            chart(C.corr_heatmap(lc, LABELS, T, "Live correlation between sleeves (daily)"), "rk_lc")


# ------------------------------------------------------------------------------------------------
# 6. positions & trades
# ------------------------------------------------------------------------------------------------
def page_positions():
    if L.empty:
        st.info("No fills yet.")
        return
    op = L.open_positions
    st.subheader("Open positions")
    if len(op):
        o = op.copy()
        o["strategy"] = o["strategy"].map(LABELS)
        st.dataframe(o.style.format({"quantity": "{:,.4g}", "avg_cost": "{:,.2f}", "last": "{:,.2f}",
                                     "market_value": "{:,.0f}", "unrealized": "{:+,.0f}", "unrealized_pct": "{:+.2%}",
                                     "realized": "{:+,.0f}", "weight_sleeve": "{:.1%}", "entry_date": "{:%Y-%m-%d}"},
                                    na_rep="–"), hide_index=True)
    else:
        st.caption("Flat.")

    st.subheader("P&L attribution")
    end = L.dates[-1]
    period = st.radio("Period", ["Since inception", "Year to date", "Month to date", "Last 30 days"], horizontal=True)
    start = {"Since inception": None, "Year to date": end.replace(month=1, day=1),
             "Month to date": end.replace(day=1), "Last 30 days": end - pd.Timedelta(days=30)}[period]
    att = book.attribution(start, None)
    if len(att):
        att = att[att["P&L"].abs() > 0.5]
        s = pd.Series(att["P&L"].to_numpy(), index=[f"{r.symbol} ({r.strategy})" for r in att.itertuples()])
        fees = L.fees.loc[start:].sum() if start is not None else L.fees.sum()
        inc = L.income.loc[start:].sum() if start is not None else L.income.sum()
        a, b = st.columns([1.6, 1])
        with a:
            chart(C.hbar(s.sort_values(ascending=False), T, "P&L by position (price + dividends)"), "pt_att")
        with b:
            summ = pd.DataFrame({"Positions P&L": att.groupby("strategy")["P&L"].sum(),
                                 "Fees": -fees.rename(LABELS)})
            summ["Net"] = summ.sum(axis=1)
            st.dataframe(summ.style.format("{:+,.0f}", na_rep="–"))
            st.caption(f"Dividends and interest booked in the period: {money(inc.sum(), cur)} (part of positions P&L "
                       "when auto-credited; otherwise in the sleeve's cash).")

    rt = L.round_trips
    st.subheader("Closed trades")
    if len(rt):
        ts = book.trade_summary()
        st.dataframe(M.format_frame(ts))
        r2 = rt.copy()
        r2["strategy"] = r2["strategy"].map(LABELS)
        st.dataframe(r2.sort_values("exit_date", ascending=False).style.format(
            {"entry_date": "{:%Y-%m-%d}", "exit_date": "{:%Y-%m-%d}", "quantity": "{:,.4g}", "entry_price": "{:,.2f}",
             "exit_price": "{:,.2f}", "pnl": "{:+,.0f}", "ret": "{:+.2%}", "fees": "{:,.2f}"}), hide_index=True)
    else:
        st.caption("No round trips closed yet.")
    with st.expander("All fills (trades.csv, in your broker's units)"):
        st.dataframe(book.trades[["date", "strategy", "symbol", "side", "quantity", "price", "fees", "note"]]
                     .sort_values("date", ascending=False), hide_index=True)
    with st.expander("Cash flows (cashflows.csv)"):
        st.dataframe(book.cashflows, hide_index=True)


# ------------------------------------------------------------------------------------------------
# 7. journal & data
# ------------------------------------------------------------------------------------------------
FLOW_HELP = {"deposit": "money into a sleeve (positive amount)", "withdrawal": "money out (positive amount)",
             "transfer": "between sleeves: + into this one, - out of it", "dividend": "income (if not auto-credited)",
             "interest": "income on cash", "fee": "expense not tied to a fill"}


def _editor_key(name: str, path: Path) -> str:
    """A new widget after every save, so stale edits are never re-applied to the reloaded file."""
    return f"{name}_{path.stat().st_mtime_ns if path.exists() else 0}"


def page_transactions():
    cs = book.capital_summary()
    if cs.empty:
        st.info("No deposits yet. Log the money you put into each strategy below: every sleeve needs a deposit "
                "(or a transfer from the unallocated `cash` sleeve) before its first buy.")
    else:
        tot = cs.loc["Total"]
        c = st.columns(6)
        c[0].metric("Deposited", compact(tot["Deposited"], cur), border=True, help=money(tot["Deposited"], cur))
        c[1].metric("Withdrawn", compact(tot["Withdrawn"], cur), border=True, help=money(tot["Withdrawn"], cur))
        c[2].metric("Net invested", compact(tot["Net invested"], cur), border=True,
                    help="Deposits minus withdrawals. Transfers between strategies do not change it.")
        c[3].metric("NAV", compact(tot["NAV"], cur), border=True, help=money(tot["NAV"], cur))
        c[4].metric("P&L", compact(tot["P&L"], cur),
                    pct(tot["P&L"] / tot["Net invested"]) + " on capital" if tot["Net invested"] else None,
                    border=True, help="NAV minus net invested capital.")
        c[5].metric("Money-weighted", pct(tot["Money-weighted return"]),
                    f"TWR {pct(tot['Time-weighted return'])}", delta_color="off", border=True,
                    help="Money-weighted (XIRR) counts *when* you added or removed money: it is your personal "
                         f"return. Annualised: {pct(tot['Money-weighted (ann.)'])}. Time-weighted strips the timing "
                         "out: it is the strategy's return, comparable with the backtest.")

        a, b = st.columns([1.5, 1])
        with a:
            scope = st.selectbox("Capital curve", ["__total__"] + [s for s in L.sleeves],
                                 format_func=lambda s: "Whole book" if s == "__total__" else LABELS.get(s, s))
            nav = L.total["nav"] if scope == "__total__" else L.nav[scope]
            flows = L.total["flows"] if scope == "__total__" else L.flows[scope]
            chart(C.nav_chart(nav, flows, T, "NAV vs net invested capital (the gap is your P&L)"), "tx_nav")
        with b:
            freq = st.radio("Period", ["M", "Q", "Y"], horizontal=True,
                            format_func={"M": "Monthly", "Q": "Quarterly", "Y": "Yearly"}.get)
            fp = book.flows_by_period(freq)
            if len(fp):
                chart(C.flow_bars(fp, T, "Deposits and withdrawals (whole book)", height=300), "tx_bars")
            else:
                st.caption("No deposits or withdrawals yet.")

        st.subheader("Capital by strategy")
        disp = cs.copy()
        disp.insert(2, "Transfers (net)", disp["Transfers in"] - disp["Transfers out"])
        disp = disp.drop(columns=["First flow", "Transfers in", "Transfers out", "Money-weighted (ann.)"]).rename(
            columns={"Money-weighted return": "Money-weighted", "Time-weighted return": "Time-weighted"})
        money_cols = ["Deposited", "Withdrawn", "Transfers (net)", "Net invested", "NAV", "P&L",
                      "Dividends & interest", "Fees"]
        st.dataframe(disp.style.format({**{k: (lambda v: money(v, cur)) for k in money_cols},
                                        "Money-weighted": "{:+.2%}", "Time-weighted": "{:+.2%}"}, na_rep="–"))
        st.caption("Transfers move money between strategies: they change each sleeve's capital but not the book's. "
                   "Dividends, interest and fees are performance, not capital. *Money-weighted* counts when money "
                   "went in or out (your personal return); *time-weighted* does not (the strategy's return, "
                   "comparable with its backtest).")
        if len(fp):
            with st.expander("Flows per period"):
                st.dataframe(fp.rename(index=str).style.format(lambda v: money(v, cur)))

    st.subheader("Record a cash movement")
    a, b = st.columns(2)
    with a:
        with st.form("cash", clear_on_submit=True):
            st.markdown("**Deposit, withdrawal, dividend, interest or fee**")
            c1, c2 = st.columns(2)
            d = c1.date_input("Date", value=date.today(), key="cf_d")
            sid = c2.selectbox("Strategy sleeve", SIDS + [CASH_SLEEVE], format_func=lambda s: LABELS.get(s, s),
                               key="cf_s")
            c1, c2 = st.columns(2)
            typ = c1.selectbox("Type", ["deposit", "withdrawal", "dividend", "interest", "fee"],
                               format_func=lambda t_: f"{t_}: {FLOW_HELP[t_]}")
            amt = c2.number_input("Amount", min_value=0.0, step=100.0, format="%.2f")
            note = st.text_input("Note", key="cf_n", placeholder="bank reference, reason ...")
            if st.form_submit_button("Add", type="primary"):
                try:
                    append_cashflow(cfg, d, sid, typ, amt, note)
                    st.rerun()
                except JournalError as exc:
                    st.error(str(exc))
    with b:
        with st.form("transfer", clear_on_submit=True):
            st.markdown("**Move money between strategies**")
            c1, c2 = st.columns(2)
            d = c1.date_input("Date", value=date.today(), key="tr_d")
            amt = c2.number_input("Amount", min_value=0.0, step=100.0, format="%.2f", key="tr_a")
            c1, c2 = st.columns(2)
            src = c1.selectbox("From", SIDS + [CASH_SLEEVE], format_func=lambda s: LABELS.get(s, s), key="tr_f")
            dst = c2.selectbox("To", SIDS + [CASH_SLEEVE], index=min(1, len(SIDS)),
                               format_func=lambda s: LABELS.get(s, s), key="tr_t")
            note = st.text_input("Note", key="tr_n", placeholder="rebalance, ...")
            if st.form_submit_button("Transfer", type="primary"):
                try:
                    append_transfer(cfg, d, src, dst, amt, note)
                    st.rerun()
                except JournalError as exc:
                    st.error(str(exc))
        st.caption("Writes the two matching `transfer` rows. Use the `Unallocated` (cash) sleeve for money you "
                   "deposited into the account but have not given to a strategy yet.")

    st.subheader("All cash movements")
    st.caption("Edit cells, add rows at the bottom or select rows and delete them, then **Save**. Every row is "
               "validated before anything is written, and the previous file is kept as `cashflows.csv.bak`.")
    raw = book.cashflows[CASH_COLUMNS].copy() if len(book.cashflows) else pd.DataFrame(columns=CASH_COLUMNS)
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    edited = st.data_editor(
        raw, num_rows="dynamic", hide_index=True, key=_editor_key("cf_editor", cfg.cashflows_path),
        column_config={
            "date": st.column_config.DateColumn("date", required=True, format="YYYY-MM-DD"),
            "strategy": st.column_config.SelectboxColumn("strategy", options=SIDS + [CASH_SLEEVE], required=True),
            "type": st.column_config.SelectboxColumn("type", options=sorted(FLOW_TYPES), required=True),
            "amount": st.column_config.NumberColumn("amount", format="%.2f", required=True),
            "note": st.column_config.TextColumn("note")})
    if st.button("Save cash movements", type="primary"):
        try:
            save_cashflows(cfg, edited)
            st.success("Saved.")
            st.rerun()
        except JournalError as exc:
            st.error(f"Not saved: {exc}")


def page_journal():
    if cfg.demo:
        st.caption("Entries added here go to the demo workspace's CSV files.")
    a, b = st.columns([1, 1.2])
    with a:
        st.subheader("Log a fill")
        with st.form("trade", clear_on_submit=True):
            c1, c2 = st.columns(2)
            d = c1.date_input("Trade date", value=date.today())
            sid = c2.selectbox("Strategy", SIDS + [CASH_SLEEVE], format_func=lambda s: LABELS.get(s, s))
            c1, c2, c3 = st.columns([1.2, 1, 1])
            sym = c1.text_input("Symbol", placeholder="AAPL")
            side = c2.selectbox("Side", list(SIDES))
            qty = c3.number_input("Quantity", min_value=0.0, step=1.0, format="%.4f")
            c1, c2 = st.columns(2)
            px = c1.number_input("Fill price", min_value=0.0, step=0.01, format="%.4f")
            fee = c2.number_input("Fees (total)", min_value=0.0, step=0.01, format="%.2f")
            note = st.text_input("Note", placeholder="signal date, order id, reason ...")
            if st.form_submit_button("Add fill", type="primary"):
                try:
                    row = append_trade(cfg, d, sid, sym, side, qty, px, fee, note)
                    st.success(f"Added {row['side']} {row['quantity']:g} {row['symbol']} @ {row['price']:g} "
                               f"to {LABELS.get(sid, sid)}.")
                    st.rerun()
                except JournalError as exc:
                    st.error(str(exc))
        st.caption("Quantities and prices exactly as your broker shows them on the trade date (splits are "
                   "handled automatically). Deposits and transfers live in the **Transactions** tab.")
    with b:
        st.subheader("Data status")
        rows = []
        for s in SIDS:
            bt = book.backtest_returns(s)
            sc = cfg.strategies[s]
            meta = book.backtest_meta.get(s) or {}
            rows.append({"Strategy": LABELS[s], "Backtest file": sc.backtest.name if sc.backtest else "–",
                         "Loaded": "yes" if bt is not None else "no",
                         "Exported configuration": meta.get("selected", "no selection record"),
                         "Final tests": tests_label(s),
                         "From": bt.index[0] if bt is not None else pd.NaT,
                         "To": bt.index[-1] if bt is not None else pd.NaT,
                         "Expectation from": book.expectation_start(s),
                         "Covers live period": "yes" if book.model_on_live_dates(s) is not None else "no"})
        st.dataframe(pd.DataFrame(rows).set_index("Strategy").style.format(
            {"From": lambda v: "–" if pd.isna(v) else f"{v:%Y-%m-%d}",
             "To": lambda v: "–" if pd.isna(v) else f"{v:%Y-%m-%d}",
             "Expectation from": lambda v: "whole export" if v is None or pd.isna(v) else f"{v:%Y-%m-%d}"}))
        if book.store.sources:
            st.caption("Price sources: " + ", ".join(f"{k} ({v})" for k, v in sorted(book.store.sources.items())))
        for w in book.data_warnings():
            st.markdown(f"ℹ️ {w}")

    st.subheader("All fills")
    st.caption("Fix a typo, add or delete rows, then **Save**. Rows are validated first; the previous file is kept "
               "as `trades.csv.bak`.")
    raw = book.trades[TRADE_COLUMNS].copy() if len(book.trades) else pd.DataFrame(columns=TRADE_COLUMNS)
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    for c_ in ("quantity", "price", "fees"):
        raw[c_] = pd.to_numeric(raw[c_], errors="coerce")
    edited = st.data_editor(
        raw, num_rows="dynamic", hide_index=True, key=_editor_key("tr_editor", cfg.trades_path),
        column_config={
            "date": st.column_config.DateColumn("date", required=True, format="YYYY-MM-DD"),
            "strategy": st.column_config.SelectboxColumn("strategy", options=SIDS + [CASH_SLEEVE], required=True),
            "symbol": st.column_config.TextColumn("symbol", required=True),
            "side": st.column_config.SelectboxColumn("side", options=list(SIDES), required=True),
            "quantity": st.column_config.NumberColumn("quantity", min_value=0.0, format="%.4f", required=True),
            "price": st.column_config.NumberColumn("price", min_value=0.0, format="%.4f", required=True),
            "fees": st.column_config.NumberColumn("fees", min_value=0.0, format="%.2f"),
            "note": st.column_config.TextColumn("note")})
    if st.button("Save fills", type="primary"):
        try:
            save_trades(cfg, edited)
            st.success("Saved.")
            st.rerun()
        except JournalError as exc:
            st.error(f"Not saved: {exc}")
    st.caption(f"Files: `{cfg.trades_path}` · `{cfg.cashflows_path}` · `{cfg.marks_path}` · "
               f"`{cfg.root / 'backtests'}`. Editing them in Excel works too; the dashboard reloads on save.")


for tab, page in zip(tabs, [page_overview, page_strategies, page_expected, page_construction, page_risk,
                            page_positions, page_transactions, page_journal]):
    with tab:
        page()
