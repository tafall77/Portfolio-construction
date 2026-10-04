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
from pcon import orders as O  # noqa: E402
from pcon import runner as RUN  # noqa: E402
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
               "Higher than expected": "🟠 Higher than expected", "Too early": "⏳ Too early (needs ~6 months)",
               "n/a": "–"}


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
def cached_explore(R: pd.DataFrame, rf: pd.Series, methods: tuple, rebalance: str, lo: float, hi: float,
                   target: pd.Series, budgets: pd.Series | None, wf_days: int, step: str) -> pd.DataFrame:
    return A.explore(R, rf, methods, rebalance, lo, hi, target, wf_days, step, n_resamples=20, budgets=budgets)


@st.cache_data(show_spinner="Optimising allocations ...", max_entries=16)
def cached_methods(R: pd.DataFrame, rf: pd.Series, lo: float, hi: float, target: pd.Series,
                   budgets: pd.Series | None, rebalance: str, wf_days: int, step: str,
                   n_resamples: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return A.compare_methods(R, rf, lo, hi, target, budgets, rebalance, wf_days, step, n_resamples)


@st.cache_data(show_spinner="Bootstrapping weight uncertainty ...", max_entries=16)
def cached_uncertainty(R, rf, lo, hi, budgets):
    return A.weight_uncertainty(R, ("risk_parity", "min_variance", "max_sharpe"), rf, lo, hi, n=60,
                                budgets=budgets)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_marginal(R, rf, method, rebalance, lo, hi, target, budgets):
    return A.marginal(R, rf, method, rebalance, lo, hi, target, budgets=budgets)


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


def table(df: pd.DataFrame, fmt, style=None, **kwargs):
    """``st.dataframe(df.style.format(fmt))`` with missing values shown as '–'.

    Streamlit draws a null cell as a grey 'None' whatever the Styler's ``na_rep``, so a column with gaps is sent
    as text: '–' in the gaps, the other cells formatted by ``fmt`` and right-aligned. ``style(styler, df)`` gets the
    original frame, so colouring rules should read their values from ``df``.
    """
    fmt = fmt if isinstance(fmt, dict) else {c: fmt for c in df.columns}

    def render(f, v):
        return v if isinstance(v, str) else f.format(v) if isinstance(f, str) else f(v) if f else str(v)
    gaps = [c for c in df.columns if df[c].dtype.kind in "fO" and df[c].isna().any()]
    shown = df.copy()
    for c in gaps:
        shown[c] = ["–" if pd.isna(v) else render(fmt.get(c), v) for v in df[c]]
    sty = shown.style.format({c: (lambda v, f=f: render(f, v)) for c, f in fmt.items()
                              if c in shown.columns and c not in gaps})
    if style is not None:
        sty = style(sty, df)
    config = {c: st.column_config.Column(alignment="right") for c in gaps} | kwargs.pop("column_config", {})
    st.dataframe(sty, column_config=config, **kwargs)


def numeric_bg(df: pd.DataFrame, fn):
    """Styler.apply callback that colours each cell from the numeric value in ``df``."""
    return lambda _: df.map(lambda v: fn(v) if isinstance(v, (int, float, np.floating)) else "")


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

tabs = st.tabs(["Overview", "Orders", "Strategies", "Expected vs actual", "Portfolio construction", "Risk",
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
    table(df, {"NAV": lambda v: money(v, cur), "Weight": "{:.1%}", "Target": "{:.0%}", "Return (live)": "{:.2%}",
               "Volatility": "{:.1%}", "Sharpe": "{:.2f}", "Max DD": "{:.1%}", "Pctile vs expected": "{:.0%}",
               "vs model": "{:+.2%}"})
    st.caption("*Pctile vs expected*: where the live return sits among same-length backtest paths (50% = as "
               "expected). *vs model*: live minus the backtest on the same days (execution shortfall). *Final "
               "tests*: how many of the research notebook's final tests the exported configuration passed.")


# ------------------------------------------------------------------------------------------------
# 2. strategies
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# orders
# ------------------------------------------------------------------------------------------------
ORDER_HELP = {
    "At the close": "market-on-close (MOC) order, or as close to the close as you can, on the day the signal is "
                    "computed. The regime filter only trades on its monthly date; momentum trades the day it flips.",
    "Market on open": "market-on-open order at the next session's open (SMA / Piotroski exits and entries).",
    "Limit, day": "sell limit at the take-profit price, valid for the day; enter it again each morning (the "
                  "SMA200 moves).",
    "Stop, day": "sell stop at the stop-loss price, valid for the day.",
}


def _progress(sid: str) -> str:
    tail = RUN.log_tail(ws, sid, 60).splitlines()
    cell = next((l for l in reversed(tail) if l.startswith("@@ cell")), "")
    return cell.replace("@@ ", "") if cell else "starting"


@st.fragment(run_every=4)
def update_progress():
    stt = RUN.read_status(ws)
    if stt.get("running"):
        sid = stt["running"]
        name = LABELS.get(sid, sid)
        st.info(f"⏳ Updating **{name}** ({_progress(sid) if sid in RUN.SPECS else 'starting'}). Typical time: "
                f"{RUN.SPECS[sid].minutes if sid in RUN.SPECS else 'a few minutes'}. You can keep using the dashboard; "
                "this refreshes by itself.")
    elif st.session_state.get("update_seen_running"):
        st.session_state.update_seen_running = False
        st.rerun(scope="app")                                  # finished: reload the book with the new signals
    if stt.get("running"):
        st.session_state.update_seen_running = True


def signals_panel():
    """Run the strategies on fresh data (the notebooks in strategies/, unchanged) and show what each produced."""
    stt = RUN.read_status(ws)
    res = stt.get("results", {})
    settings = RUN.load_settings(ws)
    need_sec = "sma_piotroski" in SIDS and not RUN.valid_sec_contact(settings.get("sec_contact", ""))
    rows = []
    for s_ in SIDS:
        sig, r = book.signals.get(s_), res.get(s_, {})
        rows.append({"Strategy": LABELS[s_],
                     "Signal from close of": f"{sig['as_of']:%a %d %b}" if sig else "–",
                     "Up to date": ("no" if O.is_stale(sig) else "yes") if sig else "no signal",
                     "Last update": (("✅ " if r.get("ok") else "❌ ") + r.get("finished", "")[5:16]) if r else "never",
                     "Result": r.get("message", "").split(" -> ")[0][:90]})
    a, b = st.columns([3, 1])
    with a:
        st.dataframe(pd.DataFrame(rows).set_index("Strategy"), width="stretch")
    with b:
        running = bool(stt.get("running"))
        stale = any(r["Up to date"] != "yes" for r in rows)
        if st.button("Update signals now", type="primary" if stale else "secondary", width="stretch",
                     disabled=running or cfg.demo,
                     help="Runs the three strategies on fresh market data (Yahoo, FRED, SEC). Do it after the US "
                          "close, before placing orders."):
            RUN.start_update(ws)
            st.session_state.update_seen_running = True
            st.rerun()
        if cfg.demo:
            st.caption("Demo book: its signals are synthetic. Switch to *data (my book)* to run the strategies.")
        if stt.get("crashed"):
            st.caption("The last update stopped before finishing (window closed?). Run it again.")
    update_progress()
    for s_, r in res.items():
        if not r.get("ok") and s_ in SIDS:
            with st.expander(f"❌ {LABELS[s_]}: what went wrong"):
                st.code(RUN.log_tail(ws, s_, 25) or r.get("message", ""), language=None)
    with st.expander("Data sources" + (" · ⚠️ needs your SEC contact" if need_sec else ""), expanded=need_sec and not cfg.demo):
        st.markdown("The SMA / Piotroski strategy reads company filings from the SEC, which asks every user for a "
                    "contact in the request header. It is saved in your workspace (`settings.json`, never uploaded).")
        with st.form("sources"):
            contact = st.text_input("Your name and e-mail", value=settings.get("sec_contact", ""),
                                    placeholder="Jane Doe jane@example.org")
            fred = st.text_input("FRED API key (optional)", value=settings.get("fred_api_key", ""), type="password",
                                 help="Without one, the regime filter uses FRED's public CSV download.")
            if st.form_submit_button("Save"):
                if contact and not RUN.valid_sec_contact(contact):
                    st.error("Enter a name followed by an e-mail address, e.g. `Jane Doe jane@example.org`.")
                else:
                    RUN.save_settings(ws, sec_contact=contact.strip(), fred_api_key=fred.strip())
                    st.success("Saved.")


def fills_logger(plans: list):
    """Tick the orders you executed, adjust quantity / price, and write them to the journal in one click."""
    o = pd.concat([p_.orders for p_ in plans if len(p_.orders)], ignore_index=True) if plans else pd.DataFrame()
    if o.empty:
        return
    st.subheader("4 · Log what you executed")
    st.caption("Tick each order you filled and enter the actual fill price (and quantity, if different). Limit and "
               "stop orders: tick only if they filled. Netted orders are logged per strategy at the same price.")
    ed = pd.DataFrame({"Filled": False, "Strategy": o["strategy"].map(LABELS), "Symbol": o["symbol"], "Side": o["side"],
                       "Quantity": o["quantity"].astype(float),
                       "Fill price": o["limit"].fillna(o["price"]).round(2), "Fees": 0.0})
    edited = st.data_editor(ed, hide_index=True, width="stretch", key=f"fills_{len(book.trades)}",
                            disabled=["Strategy", "Symbol", "Side"],
                            column_config={"Quantity": st.column_config.NumberColumn(min_value=0.0, format="%.4g"),
                                           "Fill price": st.column_config.NumberColumn(min_value=0.0, format="%.4f"),
                                           "Fees": st.column_config.NumberColumn(min_value=0.0, format="%.2f")})
    c1, c2 = st.columns([1, 3])
    with c1:
        day = st.date_input("Fill date", value=date.today(), key="fill_date")
    with c2:
        st.write("")
        if st.button("Add ticked fills to the journal", type="primary", disabled=not edited["Filled"].any()):
            done, errors = 0, []
            for _, r_ in edited[edited["Filled"]].iterrows():
                try:
                    sid = next(k for k, v in LABELS.items() if v == r_["Strategy"])
                    append_trade(cfg, day, sid, r_["Symbol"], r_["Side"], r_["Quantity"], r_["Fill price"],
                                 r_["Fees"], note="logged from Orders")
                    done += 1
                except JournalError as exc:
                    errors.append(f"{r_.Symbol}: {exc}")
            if errors:
                st.error("Not logged: " + "; ".join(errors))
            if done:
                st.success(f"Logged {done} fill(s).")
                st.rerun()



def page_orders():
    st.markdown("What to trade next. Each strategy runs on fresh market data, its **latest signal** is sized to the "
                "capital the **allocation** gives it, and compared with what its sleeve holds today. Orders of the "
                "same timing are netted across strategies.")
    st.subheader("Signals")
    signals_panel()
    ctx = allocation_context()
    options = ([f"Best Sharpe ({A.METHODS[ctx['rec']]})"] if ctx else []) + ["Your target weights (portfolio.yaml)"]
    c1, c2 = st.columns([1.4, 1])
    with c1:
        how = st.radio("Split the account between strategies by", options, horizontal=True,
                       help="Best Sharpe is the Portfolio construction recommendation: the method with the best "
                            "walk-forward (out-of-sample) Sharpe. It needs the three backtest exports.")
    weights = (ctx["W"].loc[ctx["rec"]] if how.startswith("Best") else cfg.target_weights()).reindex(SIDS).fillna(0.0)
    nav_now = float(L.nav.iloc[-1].sum()) if not L.empty else 0.0
    with c2:
        account = st.number_input(f"Account value ({cfg.currency})", min_value=0.0, value=float(round(nav_now, 2)),
                                  step=1000.0, help="Defaults to the book's NAV from your journal. Change it to size "
                                                    "orders for money you are about to deposit.")
    invest = 1.0
    if cfg.allocation.target_vol and ctx:
        invest = min(invest_fraction(A.portfolio_returns(ctx["R"], weights.reindex(ctx["R"].columns).fillna(0.0),
                                                         rebalance), cfg.allocation.target_vol,
                                     cfg.allocation.vol_cap), 1.0)
        st.caption(f"Volatility targeting ({cfg.allocation.target_vol:.0%}): **{invest:.0%}** of the account goes to "
                   f"the strategies, {1 - invest:.0%} stays in T-bills.")
    if not ctx:
        st.info("No best-Sharpe allocation yet: it needs the backtest exports (`backtests/`). Using your target weights.")

    # ---- capital per strategy ----------------------------------------------------------------
    navs = L.nav.iloc[-1] if not L.empty else pd.Series(dtype=float)
    rows = []
    for s_ in SIDS:
        sig = book.signals.get(s_)
        rows.append({"Strategy": LABELS[s_], "Weight": weights.get(s_, 0.0) * invest,
                     "Capital": account * invest * weights.get(s_, 0.0), "Sleeve now": float(navs.get(s_, 0.0)),
                     "Signal as of": f"{sig['as_of']:%Y-%m-%d}" if sig else "missing"})
    capdf = pd.DataFrame(rows).set_index("Strategy")
    capdf["Transfer"] = capdf["Capital"] - capdf["Sleeve now"]
    st.subheader("1 · Capital per strategy")
    table(capdf[["Weight", "Capital", "Sleeve now", "Transfer", "Signal as of"]],
          {"Weight": "{:.0%}", "Capital": lambda v: money(v, cur), "Sleeve now": lambda v: money(v, cur),
           "Transfer": lambda v: money(v, cur)})
    moves = capdf["Transfer"][capdf["Transfer"].abs() > max(1.0, 0.01 * account)]
    if len(moves):
        a, b = st.columns([3, 1])
        with a:
            st.caption("Strategies are bookkeeping sleeves inside one brokerage account. Moving capital between them "
                       "is not a trade: it is recorded as transfers (through *Unallocated* cash), so each strategy's "
                       "record starts from the capital it trades with.")
        with b:
            if st.button("Record these transfers", width="stretch", disabled=cfg.demo or abs(account - nav_now) > 1.0,
                         help="Writes the Transfer column to the journal, dated today. Only when the account value "
                              "equals the book's NAV (otherwise record the deposit first)."):
                for s_, amt in moves.items():
                    sid = next(k for k, v in LABELS.items() if v == s_)
                    src, dst = (CASH_SLEEVE, sid) if amt > 0 else (sid, CASH_SLEEVE)
                    append_transfer(cfg, date.today(), src, dst, round(abs(amt), 2), note="allocation (Orders tab)")
                st.success("Transfers recorded.")
                st.rerun()

    plans = book.order_plans(weights, account, invest)
    missing = [s_ for s_ in SIDS if s_ not in book.signals]

    # ---- netted orders -----------------------------------------------------------------------
    st.subheader("2 · Orders to place")
    net = O.net_orders(plans)
    if not plans:
        st.info("No signals yet: click **Update signals now** above.")
    elif net.empty:
        st.success("No orders: every strategy already holds what its signal says.")
    else:
        stale = [LABELS[s_] for s_, g in book.signals.items() if O.is_stale(g)]
        if stale:
            st.warning(f"Signals for {', '.join(stale)} are out of date: click **Update signals now** before "
                       "trading.", icon="⚠️")
        shown = net.rename(columns={"symbol": "Symbol", "side": "Side", "quantity": "Quantity", "order": "Order",
                                    "limit": "Limit / stop", "price": "Last price", "value": "≈ Value",
                                    "strategies": "Strategies"})
        shown["Strategies"] = shown["Strategies"].map(lambda x: ", ".join(LABELS.get(v, v) for v in x.split(", ")))
        table(shown, {"Quantity": "{:,.0f}", "Limit / stop": "{:,.2f}", "Last price": "{:,.2f}",
                      "≈ Value": lambda v: money(v, cur)}, hide_index=True,
              style=lambda sty, d: sty.map(lambda v: f"color: {T['div'][2]}; font-weight: 600" if v == "BUY" else
                                          f"color: {T['div'][0]}; font-weight: 600" if v == "SELL" else "",
                                          subset=["Side"]))
        for kind in shown["Order"].unique():
            if kind in ORDER_HELP:
                st.caption(f"**{kind}**: {ORDER_HELP[kind]}")
        st.caption("After filling, log each fill in **Journal & data** under the strategy shown, so every sleeve keeps "
                   "its own record. Quantities are whole shares at the last close; check them against the live price.")

    # ---- per strategy ------------------------------------------------------------------------
    st.subheader("3 · By strategy")
    for p_ in plans:
        sig = book.signals[p_.strategy]
        age = O.staleness(sig)
        with st.expander(f"{LABELS[p_.strategy]} · {money(p_.capital, cur)} · {p_.status}", expanded=True):
            st.markdown(f"Signal from the close of **{sig['as_of']:%a %d %b %Y}**"
                        + (f" ({age} trading days old{', out of date' if O.is_stale(sig) else ''})" if age > 1 else "")
                        + (f" · {sig['selected']}" if sig.get("selected") else ""))
            if sig.get("execution"):
                st.caption(sig["execution"])
            a, b = st.columns([1, 1.2])
            with a:
                if len(p_.targets):
                    fmt = {c_: "{:.1%}" for c_ in ("target weight", "current weight")} | \
                          {c_: "{:,.0f}" for c_ in ("target quantity", "current quantity", "your quantity")} | \
                          {"price": "{:,.2f}"}
                    table(p_.targets, fmt, hide_index=True)
            with b:
                if len(p_.orders):
                    table(p_.orders.drop(columns=["strategy"]), {"quantity": "{:,.0f}", "limit": "{:,.2f}",
                                                                 "price": "{:,.2f}", "value": "{:,.0f}"},
                          hide_index=True)
                else:
                    st.caption("No orders for this strategy.")
            for n_ in p_.notes:
                st.markdown(f"- {n_}")
    for s_ in missing:
        st.info(f"**{LABELS[s_]}**: no signal yet. Click **Update signals now** above.")
    fills_logger(plans)



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
        table(o, {"quantity": "{:,.4g}", "avg_cost": "{:,.2f}", "last": "{:,.2f}", "market_value": "{:,.0f}",
                  "unrealized": "{:+,.0f}", "unrealized_pct": "{:+.2%}", "realized": "{:+,.0f}",
                  "weight_sleeve": "{:.1%}", "entry_date": "{:%Y-%m-%d}"})
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
        fwd = E.forward_cone(exp, pd.Timestamp.today(), 1.0, book.rf)
        chart(C.cone(pd.DataFrame(), None, T, color, "Expected range, next 12 months", forward=fwd), "ex_fwd_only")
        fr = E.forward_risk(exp, book.rf)
        st.dataframe(pd.Series({k_: pct(v) for k_, v in fr.items()}, name="next 12 months"))
        return
    cmp_ = book.comparison(sid)
    a, b = st.columns([1.7, 1])
    with a:
        show_fwd = st.toggle("Project the cone 12 months forward", value=True)
        cn = E.cone(exp, live.index, book.rf)
        last_w = float((1 + live).prod())
        fwd = E.forward_cone(exp, live.index[-1], last_w, book.rf) if show_fwd else None
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
                f"(standard error of the difference ±{num(stt['Std error (difference)'])}, counting both estimates). "
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
        fr = E.forward_risk(exp, book.rf)
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
def allocation_context() -> dict | None:
    """Return matrix, every method's weights and the recommended method (shared by Construction and Orders)."""
    R = book.alloc_matrix(source)
    if R.shape[1] < 2 or len(R) < 300:
        return None
    rf = book.rf.reindex(R.index).fillna(0.0)
    target = cfg.target_weights().reindex(R.columns).fillna(0.0)
    budgets = cfg.risk_budgets()
    budgets = budgets.reindex(R.columns).fillna(0.0) if budgets is not None else None
    lo = cfg.allocation.min_weight
    wf_days = int(wf_years * 252)
    W, stats_m, wf = cached_methods(R, rf, lo, max_w, target, budgets, rebalance, wf_days, "Q",
                                    cfg.allocation.bootstrap_samples)
    rec = A.recommend(stats_m)
    return dict(R=R, rf=rf, target=target, budgets=budgets, lo=lo, wf_days=wf_days, W=W, stats_m=stats_m, wf=wf,
                rec=rec)


def invest_fraction(r: pd.Series, tv: float | None, cap: float) -> float:
    """Share of the book in the strategies under volatility targeting (1 when it is off)."""
    if not tv or not len(r):
        return 1.0
    sig_now = float(np.sqrt((r ** 2).ewm(halflife=cfg.allocation.vol_halflife).mean().iloc[-1] * 252))
    return float(min(cap, tv / sig_now)) if sig_now > 0 else 1.0


def page_construction():
    ctx = allocation_context()
    if ctx is None:
        R = book.alloc_matrix(source)
        st.info("Portfolio construction needs at least two strategies with overlapping daily returns (about a year "
                "or more). Export the backtests into `backtests/` (see `strategy_exports/`)."
                + (f" Currently: {R.shape[1]} strategies, {len(R)} common days." if len(R) else ""))
        return
    R, rf, target, budgets, lo, wf_days = (ctx[k] for k in ("R", "rf", "target", "budgets", "lo", "wf_days"))
    W, stats_m, wf, rec = ctx["W"], ctx["stats_m"], ctx["wf"], ctx["rec"]
    best_wf = stats_m["WF Sharpe"].astype(float).idxmax()
    n = R.shape[1]
    lo_used, hi_used, relaxed = A.feasible_bounds(n, lo, max_w)
    if relaxed:
        st.warning(f"A max weight of {max_w:.0%} (min {lo:.0%}) cannot hold with {n} fully invested strategies: "
                   f"{lo_used:.0%}-{hi_used:.0%} is used instead.", icon="⚠️")
    st.caption(f"Common window **{R.index[0]:%Y-%m-%d} → {R.index[-1]:%Y-%m-%d}** ({len(R) / 252:.1f} years, "
               f"{len(R):,} days) · strategies: {', '.join(LABELS[c] for c in R.columns)} · source: {source} · "
               f"{ {'M': 'monthly', 'Q': 'quarterly', 'A': 'annual', 'D': 'daily'}[rebalance]} rebalancing · "
               f"weights {lo_used:.0%}-{hi_used:.0%}. Walk-forward = weights estimated on the trailing "
               f"{wf_years:g} years only, held for the next quarter.")

    r_t = A.portfolio_returns(R, W.loc["target"], rebalance)
    r_r = A.portfolio_returns(R, W.loc[rec], rebalance)

    # ---- recommendation ----------------------------------------------------------------------
    st.subheader("Recommendation")
    test = cached_sharpe_test(wf[rec], wf["target"], rf) if rec != "target" and len(wf[rec]) else {}
    mg = cached_marginal(R, rf, "target", rebalance, lo, max_w, target, budgets)
    drop = [LABELS[i] for i, v in mg["Verdict"].items() if str(v).startswith("Removal candidate")]
    dv_t = A.diversification(R, W.loc["target"], book.bench if len(book.bench) else None)
    c = st.columns([1.2, 1])
    with c[0]:
        sig = ""
        if test and np.isfinite(test.get("p_not_better", np.nan)):
            sig = (f"; out of sample the improvement is "
                   f"{'statistically meaningful' if test['p_not_better'] < 0.1 else 'not statistically significant'} "
                   f"(P(no improvement) = {pct(test['p_not_better'], 0)})")
        st.markdown(
            f"- **Allocation method:** *{A.METHODS[rec]}* "
            f"(walk-forward Sharpe {num(stats_m.loc[rec, 'WF Sharpe'])} vs {num(stats_m.loc['target', 'WF Sharpe'])} "
            f"for your targets; the best out-of-sample method was *{A.METHODS[best_wf]}*).\n"
            f"- **Weights:** " + ", ".join(f"{LABELS[k_]} **{v:.0%}**" for k_, v in W.loc[rec].items()) + "\n"
            f"- **Out of sample vs your targets:** Sharpe {num(stats_m.loc[rec, 'WF Sharpe'])} vs "
            f"{num(stats_m.loc['target', 'WF Sharpe'])}, max drawdown {pct(stats_m.loc[rec, 'WF Max drawdown'])} vs "
            f"{pct(stats_m.loc['target', 'WF Max drawdown'])}{sig}.\n"
            + (f"- **Strategies the data would drop:** {', '.join(drop)} (removal is only worth acting on when it "
               "is significant)." if drop else
               "- **No strategy is a removal candidate** at your target weights.") + "\n"
            + (f"- **Diversification:** the book holds about **{dv_t['Effective number of bets']:.1f} independent "
               f"bets**" + (f"; {dv_t['Share of variance from benchmark beta']:.0%} of its variance is "
                            f"{cfg.benchmark} beta" if "Share of variance from benchmark beta" in dv_t else "")
               + " (see section 3).")
        )
        st.caption("Optimised weights are estimates. The recommendation and its significance test use the "
                   "walk-forward (out-of-sample) record, never in-sample maximum-Sharpe weights, which overfit.")
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
    table(mgd, {"Sharpe alone": "{:.2f}", "CAGR alone": "{:.1%}", "Max DD alone": "{:.1%}", "Corr to rest": "{:.2f}",
                "Hurdle Sharpe": "{:.2f}", "Sharpe without": "{:.2f}", "Sharpe with": "{:.2f}", "Δ Sharpe": "{:+.2f}",
                "Δ CAGR": "{:+.1%}", "Δ Max DD": "{:+.1%}", "Weight": "{:.0%}", "P(no improvement)": "{:.0%}"})
    st.caption("Δ columns = with minus without the strategy. Δ Max DD > 0 means a shallower drawdown with it. "
               "*P(no improvement)* below 10% = the gain is statistically meaningful; above 90% = removing it is. "
               "A strategy that passes the hurdle but lowers Sharpe at its current weight is over-weighted, not "
               "useless: keep it at a smaller weight.")
    methods = tuple(m for m in A.METHODS if m != "risk_budget" or budgets is not None)
    ex = cached_explore(R, rf, methods, rebalance, lo, max_w, target, budgets, wf_days, "Q")
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
        pc |= {k_: "{:.2f}" for k_ in ("Sharpe", "Sortino", "Calmar", "WF Sharpe")} | {"Max weight used": "{:.0%}"}
        table(exd.drop(columns=["n"]), pc, hide_index=True, height=420)
    if (ex["Max weight used"] > hi_used + 1e-9).any():
        st.caption(f"*Max weight used*: for 2-strategy subsets a cap below 50% cannot hold, so 50% is applied.")

    # ---- how much each -------------------------------------------------------------------------
    st.subheader("2 · How much capital to each?")
    a, b = st.columns([1, 1.35])
    with a:
        Wd = W.rename(index=A.METHODS)
        chart(C.weights_by_method(Wd, COLORS, LABELS, T, "Weights by allocation method", height=380), "pc_wm")
    with b:
        sm = stats_m.rename(index=A.METHODS)[["CAGR", "Volatility", "Sharpe", "Max drawdown", "WF CAGR", "WF Sharpe",
                                              "WF Max drawdown", "Weight turnover / quarter"]]
        best = f"background-color: {C._rgba('#2a78d6', 0.18)}"
        table(sm, {"CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}",
                   "WF CAGR": "{:.1%}", "WF Sharpe": "{:.2f}", "WF Max drawdown": "{:.1%}",
                   "Weight turnover / quarter": "{:.1%}"}, height=390,
              style=lambda sty, d: sty.apply(lambda col: np.where(d["WF Sharpe"] == d["WF Sharpe"].max(), best, ""),
                                             subset=["WF Sharpe"]))
        st.caption("In-sample columns use weights fitted on the whole window (optimistic). WF columns are each "
                   "method's honest out-of-sample record; *turnover* is how much its weights move at each quarterly "
                   "re-estimate (high = the method is chasing noise).")
    unc = cached_uncertainty(R, rf, lo, max_w, budgets)
    unc["cell"] = [f"{r.p50:.0%}  ({r.p5:.0%}–{r.p95:.0%})" for r in unc.itertuples()]
    ut = unc.pivot(index="method", columns="strategy", values="cell").rename(index=A.METHODS, columns=LABELS)
    st.markdown("**How stable are the weights?** Median and 90% range of each method's weights re-optimised on "
                "60 resampled histories:")
    st.dataframe(ut)
    a, b = st.columns(2)
    with a:
        fr, cloud = A.efficient_frontier(R, rf, lo, max_w)
        mu, cov = A.estimate(R, rf)
        pts = {LABELS[c_]: (float(np.sqrt(cov[i, i])), float(mu[i]), COLORS[c_]) for i, c_ in enumerate(R.columns)}
        for m, col in (("target", "portfolio"), (rec, C.MODEL)):
            w = W.loc[m].to_numpy()
            pts[A.METHODS[m]] = (float(np.sqrt(w @ cov @ w)), float(w @ mu), col)
        chart(C.frontier(fr, cloud, pts, T, f"Risk / return of every mix with weights {lo_used:.0%}-{hi_used:.0%}",
                         frontier_name=f"Efficient frontier (max weight {hi_used:.0%})",
                         cloud_name="Random mixes within the bounds"), "pc_front")
    with b:
        ser = {"Your targets": (r_t, "portfolio"), A.METHODS[rec]: (r_r, C.MODEL)}
        ser |= {LABELS[c_]: (R[c_], COLORS[c_]) for c_ in R.columns}
        chart(C.growth(ser, T, "Growth of 1 over the window (log)", log=True, height=400), "pc_growth")

    # ---- independent bets --------------------------------------------------------------------
    st.subheader("3 · How many independent bets does the book hold?")
    dv_r = A.diversification(R, W.loc[rec], book.bench if len(book.bench) else None)
    keys = ["Effective number of bets", "Diversification ratio", "Effective risk contributors", "Beta to benchmark",
            "Share of variance from benchmark beta"]
    dtab = pd.DataFrame({"Your targets": {k_: dv_t.get(k_) for k_ in keys},
                         A.METHODS[rec]: {k_: dv_r.get(k_) for k_ in keys}}).rename(
        index={"Beta to benchmark": f"Beta to {cfg.benchmark}",
               "Share of variance from benchmark beta": f"Share of variance that is {cfg.benchmark} beta"})
    a, b = st.columns([1, 1])
    with a:
        shown_d = dtab.astype(float).apply(lambda row: row.map(
            lambda v: "–" if not np.isfinite(v) else f"{v:.0%}" if row.name == dtab.index[-1] else f"{v:.2f}"), axis=1)
        table(shown_d, {}, column_config={c_: st.column_config.Column(alignment="right") for c_ in shown_d.columns})
        enb = dv_t.get("Effective number of bets", np.nan)
        st.caption(f"With {n} strategies the maximum is {n} bets. *Effective number of bets* (Meucci) counts "
                   "independent sources of variance; *diversification ratio* = average strategy vol / portfolio "
                   "vol (1 = none). Risk parity spreads volatility evenly across the sleeves, but it cannot create "
                   "bets that are not there.")
        if np.isfinite(enb) and enb < 1.8 and n >= 3:
            st.info(f"Your {n} strategies behave like about **{enb:.1f} independent bets**"
                    + (f": {dv_t['Share of variance from benchmark beta']:.0%} of the book's variance is "
                       f"{cfg.benchmark} beta" if "Share of variance from benchmark beta" in dv_t else "")
                    + ". All of them are long US equity with different timing rules, so they lose together in "
                      "sharp sell-offs. The next strategy that would add real diversification is one with low "
                      "equity beta (other asset classes, market-neutral or short-biased).", icon="ℹ️")
    with b:
        if "Strategy betas" in dv_t:
            bt_tab = pd.DataFrame({f"Beta to {cfg.benchmark}": dv_t["Strategy betas"],
                                   f"R² vs {cfg.benchmark}": dv_t["Strategy R2 vs benchmark"]}).rename(index=LABELS)
            st.dataframe(bt_tab.style.format({bt_tab.columns[0]: "{:.2f}", bt_tab.columns[1]: "{:.0%}"}))
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
        rcs = {"Your targets": A.risk_contributions(R, W.loc["target"])["% of risk"],
               A.METHODS[rec]: A.risk_contributions(R, W.loc[rec])["% of risk"]}
        if budgets is not None:
            rcs["Your risk budgets (goal)"] = budgets / budgets.sum()
        chart(C.weights_by_method(pd.concat(rcs, axis=1).T, COLORS, LABELS, T, "Share of portfolio volatility",
                                  height=300), "pc_rc")
    rcor = A.rolling_correlations(R, 126)
    rcor.columns = [" / ".join(LABELS[x] for x in c_.split(" / ")) for c_ in rcor.columns]
    pair_colors = dict(zip(rcor.columns, ["#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948", "#52514e"]))
    chart(C.lines(rcor, pair_colors, T, "Rolling 6-month correlation between strategies", yfmt=".1f", height=260,
                  zero=True), "pc_rcorr")
    dc = A.drawdown_contributions(R, W.loc["target"], rebalance)
    if len(dc):
        st.caption(f"Worst drawdown of the target mix ({dc['peak'].iloc[0]:%Y-%m-%d} → {dc['trough'].iloc[0]:%Y-%m-%d}): "
                   + ", ".join(f"{LABELS[i]} {v:.0%}" for i, v in dc["% of drawdown"].items()) + " of the loss.")
    st.markdown("**Stress windows** (close of the first date to close of the last; each strategy's full backtest "
                "where it covers the window)")
    full = {LABELS[s]: book.backtest_returns(s) for s in SIDS if book.backtest_returns(s) is not None}
    full["Target mix"] = r_t
    if len(book.bench):
        full[cfg.benchmark] = book.bench
    stt = A.stress_test(full).astype(float)
    if len(stt):
        table(stt, "{:.1%}", style=lambda sty, d: sty.apply(numeric_bg(d, lambda v: diverging_bg(v, 0.4)), axis=None))

    # ---- portfolio risk level ----------------------------------------------------------------
    st.subheader("4 · Portfolio risk level: volatility targeting")
    st.markdown("Institutional books size the *whole* portfolio to a risk budget and use T-bills as the dial: "
                "invest `min(cap, target ÷ forecast vol)` of the book in the strategies each month, the rest in "
                "T-bills. The forecast is an EWMA volatility known at the previous close (no look-ahead).")
    a, b = st.columns([1, 2])
    with a:
        tv = st.slider("Target volatility (0 = off)", 0.0, 0.25, float(cfg.allocation.target_vol or 0.0), 0.01,
                       format="%.2f")
        cap = st.slider("Maximum exposure", 0.5, 1.5, float(cfg.allocation.vol_cap), 0.05,
                        help="1.0 = never borrow. Above 1 assumes borrowing at the T-bill rate.")
    invest_now = 1.0
    if tv > 0:
        base = {"Your targets": r_t, A.METHODS[rec]: r_r}
        rows, expo = {}, {}
        for name, r_ in base.items():
            scaled, e = A.vol_target(r_, tv, rf, cfg.allocation.vol_halflife, cap, "M")
            rows[name] = A.stats_row(r_, rf)
            rows[f"{name} + vol target"] = A.stats_row(scaled, rf)
            expo[name] = e
        vt = pd.DataFrame(rows).T[["CAGR", "Volatility", "Sharpe", "Max drawdown", "Calmar", "Worst month"]]
        with b:
            st.dataframe(vt.style.format({"CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}",
                                          "Max drawdown": "{:.1%}", "Calmar": "{:.2f}", "Worst month": "{:.1%}"}))
        e_t = expo["Your targets"]
        sig_now = float(np.sqrt((r_t ** 2).ewm(halflife=cfg.allocation.vol_halflife).mean().iloc[-1] * 252))
        invest_now = invest_fraction(r_t, tv, cap)
        chart(C.lines(pd.DataFrame({"Exposure of the target mix": e_t}), {"Exposure of the target mix": "portfolio"},
                      T, "Share of the book invested in the strategies", yfmt=".0%", height=220), "pc_vt")
        st.markdown(f"**Today:** forecast volatility of your target mix is **{sig_now:.1%}**, so a {tv:.0%} target "
                    f"means investing **{invest_now:.0%}** of the book in the strategies and "
                    f"**{max(0.0, 1 - invest_now):.0%}** in T-bills (the *Unallocated* sleeve). The rebalance below "
                    "includes it.")
    else:
        with b:
            st.caption("Off: the book is always fully invested in the strategies. Set a target (or `target_vol` in "
                       "portfolio.yaml) to see the effect on Sharpe and drawdown.")

    # ---- rebalance -----------------------------------------------------------------------------
    st.subheader("5 · Rebalance")
    if L.empty:
        st.info("No live sleeves yet.")
    else:
        which = st.radio("Bring sleeves to", ["Target weights (portfolio.yaml)", f"Recommended ({A.METHODS[rec]})"],
                         horizontal=True)
        tw = cfg.target_weights() if which.startswith("Target") else W.loc[rec]
        band = cfg.alerts.weight_drift
        orders = book.rebalance_orders(tw, band=band, invest=min(invest_now, 1.0))
        a, b = st.columns([1.3, 1])
        with a:
            table(orders, {"Current $": "{:,.0f}", "Current weight": "{:.1%}", "Target weight": "{:.1%}",
                           "Target $": "{:,.0f}", "Transfer $": "{:+,.0f}"})
            if (orders["Transfer $"].abs() < 0.5).all():
                st.success(f"Every sleeve is within ±{band:.0%} of its target: no rebalance needed.")
            st.caption(f"Tolerance-band rebalancing: nothing moves until a sleeve drifts more than ±{band:.0%} "
                       "(`alerts.weight_drift`), then everything goes back to target. Record the moves with *Move "
                       "money between strategies* in the Transactions tab.")
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
            fr = E.forward_risk(ep, book.rf)
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
        table(o, {"quantity": "{:,.4g}", "avg_cost": "{:,.2f}", "last": "{:,.2f}", "market_value": "{:,.0f}",
                  "unrealized": "{:+,.0f}", "unrealized_pct": "{:+.2%}", "realized": "{:+,.0f}",
                  "weight_sleeve": "{:.1%}", "entry_date": "{:%Y-%m-%d}"}, hide_index=True)
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
            table(summ, "{:+,.0f}")
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
        table(disp, {**{k: (lambda v: money(v, cur)) for k in money_cols},
                     "Money-weighted": "{:+.2%}", "Time-weighted": "{:+.2%}"})
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


for tab, page in zip(tabs, [page_overview, page_orders, page_strategies, page_expected, page_construction, page_risk,
                            page_positions, page_transactions, page_journal]):
    with tab:
        page()
