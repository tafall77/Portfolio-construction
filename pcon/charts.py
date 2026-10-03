"""Plotly figures for the dashboard (one visual system: thin marks, hairline grid, fixed series colors).

Series colours follow the entity: each strategy keeps the colour set in portfolio.yaml everywhere, the
portfolio is primary ink and the benchmark is muted grey.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
             zero="#c3c2b7", neutral="#f0efec", good="#0ca30c", warning="#fab219", serious="#ec835a",
             critical="#d03b3b", seq=["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"],
             div=("#e34948", "#f0efec", "#2a78d6"))
DARK = dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
            zero="#4a4a47", neutral="#383835", good="#0ca30c", warning="#fab219", serious="#ec835a",
            critical="#d03b3b", seq=["#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"],
            div=("#e66767", "#383835", "#3987e5"))
MODEL = "#4a3aa7"          # categorical slot 7: "the model / the recommendation", never a strategy
DARK_STEP = {"#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70", "#eda100": "#c98500",
             "#e87ba4": "#d55181", "#008300": "#008300", "#4a3aa7": "#9085e9", "#e34948": "#e66767"}


class Theme:
    def __init__(self, dark: bool = False):
        self.dark = dark
        self.c = DARK if dark else LIGHT

    def __getitem__(self, k):
        return self.c[k]

    def series(self, color: str) -> str:
        if color in ("portfolio", "ink"):
            return self.c["ink"]
        if color == "benchmark":
            return self.c["muted"]
        return DARK_STEP.get(color, color) if self.dark else color


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def layout(fig: go.Figure, t: Theme, title: str | None = None, height: int = 340, yfmt: str | None = None,
           xfmt: str | None = None, legend: bool = True, hover: str = "x unified") -> go.Figure:
    top = (34 if title else 6) + (28 if legend else 0)
    fig.update_layout(
        height=height + top, margin=dict(l=56, r=20, t=top, b=36),
        title=dict(text=title, x=0, xanchor="left", xref="container", y=1, yref="container", yanchor="top",
                   pad=dict(t=8, l=4), font=dict(size=14, color=t["ink"])) if title else None,
        font=dict(family=FONT, size=12, color=t["ink2"]),
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"], hovermode=hover,
        hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor=t["surface"], bordercolor=t["axis"]),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=t["ink2"]),
                    bgcolor="rgba(0,0,0,0)"),
    )
    ax = dict(showgrid=True, gridcolor=t["grid"], gridwidth=1, zeroline=False, linecolor=t["axis"], linewidth=1,
              tickfont=dict(color=t["ink2"]), title_font=dict(color=t["ink2"]), automargin=True)
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    if xfmt:
        fig.update_xaxes(tickformat=xfmt)
    return fig


def _line(x, y, name, color, width=2.0, dash=None, hoverfmt=".1%", showlegend=True, fill=None, fillcolor=None,
          opacity=1.0):
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=width, dash=dash),
                      hovertemplate=f"%{{y:{hoverfmt}}}", showlegend=showlegend, fill=fill, fillcolor=fillcolor,
                      opacity=opacity)


def growth(series: dict[str, tuple[pd.Series, str]], t: Theme, title: str | None = None, log: bool = False,
           height: int = 360, as_pct: bool = True) -> go.Figure:
    """Cumulative return of daily return series (``name -> (returns, colour)``), all starting at 0 %."""
    fig = go.Figure()
    for name, (r, color) in series.items():
        r = r.dropna()
        if r.empty:
            continue
        w = (1 + r).cumprod()
        w = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)]), w])
        y = w - 1 if as_pct and not log else w
        fig.add_trace(_line(y.index, y, name, t.series(color), 2.4 if color == "portfolio" else 1.8,
                            hoverfmt=".1%" if as_pct and not log else ".3f"))
        fig.add_trace(go.Scatter(x=[y.index[-1]], y=[y.iloc[-1]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=8, color=t.series(color), line=dict(width=2, color=t["surface"]))))
    layout(fig, t, title, height, yfmt=".0%" if as_pct and not log else None)
    if log:
        fig.update_yaxes(type="log", title="growth of 1")
    return fig


def nav_chart(nav: pd.Series, flows: pd.Series, t: Theme, title: str | None = None, height: int = 300) -> go.Figure:
    """NAV in currency with cumulative net invested capital for reference."""
    fig = go.Figure()
    invested = flows.cumsum()
    fig.add_trace(_line(invested.index, invested, "Net invested", t["muted"], 1.4, hoverfmt=",.0f"))
    fig.add_trace(_line(nav.index, nav, "NAV", t["ink"], 2.2, hoverfmt=",.0f", fill="tonexty",
                        fillcolor=_rgba("#2a78d6", 0.08)))
    return layout(fig, t, title, height, yfmt=",.0f")


def drawdowns(series: dict[str, tuple[pd.Series, str]], t: Theme, title: str | None = "Drawdown",
              height: int = 230) -> go.Figure:
    fig = go.Figure()
    for name, (r, color) in series.items():
        r = r.dropna()
        if r.empty:
            continue
        w = (1 + r).cumprod()
        dd = w / w.cummax().clip(lower=1.0) - 1
        c = t.series(color)
        fig.add_trace(_line(dd.index, dd, name, c, 2.0 if color == "portfolio" else 1.4,
                            fill="tozeroy" if color == "portfolio" else None,
                            fillcolor=_rgba(c if c.startswith("#") else "#0b0b0b", 0.08)))
    return layout(fig, t, title, height, yfmt=".0%")


def cone(cone_df: pd.DataFrame, live: pd.Series | None, t: Theme, color: str, title: str | None = None,
         forward: pd.DataFrame | None = None, height: int = 400, start_value: float = 1.0) -> go.Figure:
    """Expectation cone (5-95 % and 25-75 % bands, median) with the live path on top."""
    fig = go.Figure()
    c = t.series(color)
    base = c if c.startswith("#") else "#2a78d6"

    def bands(df, label_suffix, show):
        if df is None or df.empty:
            return
        x = df.index
        fig.add_trace(go.Scatter(x=x, y=df["p95"] / start_value - 1, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=df["p5"] / start_value - 1, line=dict(width=0), fill="tonexty",
                                 fillcolor=_rgba(base, 0.10), name="5-95% of backtest paths" + label_suffix,
                                 showlegend=show, hovertemplate="5th pct: %{y:.1%}"))
        fig.add_trace(go.Scatter(x=x, y=df["p75"] / start_value - 1, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=df["p25"] / start_value - 1, line=dict(width=0), fill="tonexty",
                                 fillcolor=_rgba(base, 0.18), name="25-75%" + label_suffix, showlegend=show,
                                 hovertemplate="25th pct: %{y:.1%}"))
        fig.add_trace(_line(x, df["p50"] / start_value - 1, "Median expectation" + label_suffix, t["muted"], 1.4,
                            showlegend=show))

    bands(cone_df, "", True)
    if forward is not None and not forward.empty:
        bands(forward, " (next 12m)", False)
    if live is not None and len(live):
        w = (1 + live.dropna()).cumprod()
        w = pd.concat([pd.Series([1.0], index=[w.index[0] - pd.Timedelta(days=1)]), w])
        fig.add_trace(_line(w.index, w - 1, "Actual (live)", t["ink"], 2.4))
        fig.add_trace(go.Scatter(x=[w.index[-1]], y=[w.iloc[-1] - 1], mode="markers", showlegend=False,
                                 hoverinfo="skip",
                                 marker=dict(size=9, color=t["ink"], line=dict(width=2, color=t["surface"]))))
    return layout(fig, t, title, height, yfmt=".0%")


def distribution(dist: np.ndarray, actual: float, t: Theme, title: str, fmt: str = ".0%", color: str = "#2a78d6",
                 height: int = 230) -> go.Figure:
    d = dist[np.isfinite(dist)]
    fig = go.Figure(go.Histogram(x=d, nbinsx=40, marker=dict(color=_rgba(t.series(color), 0.55), line=dict(
        color=t["surface"], width=1)), hovertemplate=f"%{{x:{fmt}}}<br>%{{y}} paths<extra></extra>",
                                 showlegend=False))
    if np.isfinite(actual):
        fig.add_vline(x=actual, line=dict(color=t["ink"], width=2))
        fig.add_annotation(x=actual, y=1, yref="paper", text=f"actual {actual:{fmt}}", showarrow=False,
                           yanchor="bottom", font=dict(color=t["ink"], size=11))
    layout(fig, t, title, height, xfmt=fmt, legend=False, hover="closest")
    fig.update_yaxes(title="paths")
    fig.update_layout(bargap=0.04)
    return fig


def weights_vs_target(actual: pd.Series, target: pd.Series, colors: dict, labels: dict, t: Theme,
                      title: str | None = None, height: int = 220) -> go.Figure:
    """Horizontal bars of actual weight with a target tick per sleeve."""
    idx = list(dict.fromkeys(list(target.index) + list(actual.index)))
    a = actual.reindex(idx).fillna(0.0)
    g = target.reindex(idx).fillna(0.0)
    names = [labels.get(i, i) for i in idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=a, orientation="h", name="Actual", width=0.45, showlegend=False,
                         marker=dict(color=[t.series(colors.get(i, "#898781")) for i in idx], cornerradius=4),
                         hovertemplate="%{y}: %{x:.1%}<extra>actual</extra>", text=[f"{v:.0%}" for v in a],
                         textposition="outside", textfont=dict(color=t["ink2"])))
    fig.add_trace(go.Scatter(y=names, x=g, mode="markers", name="Target weight (bars = actual)",
                             marker=dict(symbol="line-ns", size=22, line=dict(width=3, color=t["ink"])),
                             hovertemplate="%{y}: %{x:.1%}<extra>target</extra>"))
    layout(fig, t, title, height, xfmt=".0%", hover="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, max(0.05, float(max(a.max(), g.max())) * 1.25)])
    _left_margin(fig, names)
    return fig


def monthly_heatmap(table: pd.DataFrame, t: Theme, title: str | None = None, height: int | None = None) -> go.Figure:
    if table.empty:
        return go.Figure()
    z = table.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(z[:, :12])) if np.isfinite(z[:, :12]).any() else 0.05
    neg, mid, pos = t["div"]
    text = [[("" if not np.isfinite(v) else f"{v:.1%}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=list(table.columns), y=[str(i) for i in table.index], zmid=0, zmin=-lim,
                               zmax=lim, colorscale=[[0, neg], [0.5, mid], [1, pos]], text=text,
                               texttemplate="%{text}", textfont=dict(size=11, color=t["ink"]), showscale=False,
                               xgap=2, ygap=2, hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>"))
    layout(fig, t, title, height or 40 + 30 * len(table), legend=False, hover="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_layout(margin=dict(l=48, t=(34 if title else 0) + 26, b=8))
    return fig


def corr_heatmap(corr: pd.DataFrame, labels: dict, t: Theme, title: str | None = None, height: int = 300) -> go.Figure:
    names = [labels.get(c, c) for c in corr.columns]
    neg, mid, pos = t["div"]
    z = corr.to_numpy(float)
    fig = go.Figure(go.Heatmap(z=z, x=names, y=names, zmin=-1, zmax=1, colorscale=[[0, neg], [0.5, mid], [1, pos]],
                               text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}",
                               textfont=dict(size=13, color=t["ink"]), xgap=2, ygap=2, showscale=False,
                               hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
    layout(fig, t, title, height, legend=False, hover="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    _left_margin(fig, names)
    return fig


def lines(df: pd.DataFrame, colors: dict, t: Theme, title: str | None = None, yfmt: str = ".2f", height: int = 260,
          zero: bool = False, hoverfmt: str | None = None) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        s = df[col].dropna()
        fig.add_trace(_line(s.index, s, str(col), t.series(colors.get(col, "#898781")), 1.6,
                            hoverfmt=hoverfmt or yfmt))
    if zero:
        fig.add_hline(y=0, line=dict(color=t["zero"], width=1))
    return layout(fig, t, title, height, yfmt=yfmt)


def weights_by_method(W: pd.DataFrame, colors: dict, labels: dict, t: Theme, title: str | None = None,
                      height: int = 320) -> go.Figure:
    """Stacked horizontal bars: one row per allocation method, one segment per strategy (2px gaps)."""
    fig = go.Figure()
    for col in W.columns:
        fig.add_trace(go.Bar(y=list(W.index), x=W[col], orientation="h", name=labels.get(col, col),
                             marker=dict(color=t.series(colors.get(col, "#898781")),
                                         line=dict(color=t["surface"], width=2)),
                             text=[f"{v:.0%}" if v >= 0.08 else "" for v in W[col]], textposition="inside",
                             insidetextanchor="middle", textfont=dict(color="#ffffff", size=11),
                             hovertemplate="%{y}: %{x:.1%}<extra>" + labels.get(col, col) + "</extra>"))
    layout(fig, t, title, height, xfmt=".0%", hover="closest")
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, 1])
    _left_margin(fig, W.index)
    return fig


def frontier(front: pd.DataFrame, cloud: pd.DataFrame, points: dict[str, tuple[float, float, str]], t: Theme,
             title: str | None = None, height: int = 400) -> go.Figure:
    """Risk/return plane: random long-only mixes, the efficient frontier and labelled portfolios."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cloud["Volatility"], y=cloud["Excess return"], mode="markers", name="Random mixes",
                             marker=dict(size=4, color=_rgba("#898781", 0.25)), hoverinfo="skip"))
    if len(front):
        fig.add_trace(_line(front["Volatility"], front["Excess return"], "Efficient frontier", t["ink2"], 1.6,
                            hoverfmt=".1%"))
    for name, (vol, ret, color) in points.items():
        fig.add_trace(go.Scatter(x=[vol], y=[ret], mode="markers+text", name=name, text=[name],
                                 textposition="top center", textfont=dict(color=t["ink2"], size=11),
                                 marker=dict(size=11, color=t.series(color), line=dict(width=2, color=t["surface"])),
                                 hovertemplate=f"{name}<br>vol %{{x:.1%}}<br>excess return %{{y:.1%}}<extra></extra>"))
    layout(fig, t, title, height, yfmt=".0%", xfmt=".0%", hover="closest")
    fig.update_xaxes(title="annualised volatility")
    fig.update_yaxes(title="annualised excess return")
    return fig


def _left_margin(fig: go.Figure, labels) -> None:
    fig.update_layout(margin_l=int(16 + 7 * max([len(str(x)) for x in labels] or [4])))


def hbar(values: pd.Series, t: Theme, title: str | None = None, fmt: str = ",.0f", height: int | None = None,
         colors: list[str] | None = None) -> go.Figure:
    """Horizontal bar for signed values (P&L attribution, transfers): negative bars in the diverging warm pole."""
    v = values.astype(float)
    neg, _, pos = t["div"]
    cols = colors or [pos if x >= 0 else neg for x in v]
    fig = go.Figure(go.Bar(y=[str(i) for i in v.index], x=v, orientation="h", width=0.6,
                           marker=dict(color=cols, cornerradius=4),
                           text=[format(x, fmt) for x in v], textposition="outside",
                           textfont=dict(color=t["ink2"], size=11),
                           hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    layout(fig, t, title, height or max(160, 40 + 26 * len(v)), legend=False, hover="closest")
    fig.add_vline(x=0, line=dict(color=t["zero"], width=1))
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_xaxes(tickformat=fmt)
    _left_margin(fig, v.index)
    lo_, hi_ = float(min(v.min(), 0)), float(max(v.max(), 0))
    pad = (hi_ - lo_) * 0.18 or 1.0
    fig.update_xaxes(range=[lo_ - (pad if lo_ < 0 else 0), hi_ + (pad if hi_ > 0 else 0)])
    return fig


def exposure(gross: pd.DataFrame, colors: dict, labels: dict, t: Theme, title: str | None = None,
             height: int = 260) -> go.Figure:
    """Stacked area of each sleeve's market exposure as a share of total NAV."""
    fig = go.Figure()
    for col in gross.columns:
        c = t.series(colors.get(col, "#898781"))
        fig.add_trace(go.Scatter(x=gross.index, y=gross[col], name=labels.get(col, col), mode="lines",
                                 stackgroup="one", line=dict(width=0.8, color=c), fillcolor=_rgba(c, 0.55),
                                 hovertemplate="%{y:.1%}"))
    return layout(fig, t, title, height, yfmt=".0%")


def flow_bars(df: pd.DataFrame, t: Theme, title: str | None = None, height: int = 300) -> go.Figure:
    """Deposits up (cool pole) and withdrawals down (warm pole) per period, one shared zero baseline."""
    neg, _, pos = t["div"]
    fmt = {"M": "%b %y", "Q": None, "Y": "%Y", "A-DEC": "%Y", "Y-DEC": "%Y"}
    f = fmt.get(getattr(df.index, "freqstr", ""), None) if isinstance(df.index, pd.PeriodIndex) else None
    x = [i.strftime(f) if f else str(i) for i in df.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=df["Deposits"], name="Deposits", marker=dict(color=pos, cornerradius=4),
                         hovertemplate="%{x}: %{y:,.0f}<extra>deposits</extra>"))
    fig.add_trace(go.Bar(x=x, y=df["Withdrawals"], name="Withdrawals", marker=dict(color=neg, cornerradius=4),
                         hovertemplate="%{x}: %{y:,.0f}<extra>withdrawals</extra>"))
    layout(fig, t, title, height, yfmt=",.0f", hover="closest")
    fig.update_layout(barmode="relative", bargap=0.45)
    fig.add_hline(y=0, line=dict(color=t["zero"], width=1))
    fig.update_xaxes(type="category", showgrid=False)
    return fig
