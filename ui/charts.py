"""Графики Plotly. Один график — одна шкала значений; разные меры идут отдельными панелями."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT = "Roboto, system-ui, sans-serif"


def _layout(fig: go.Figure, t: dict, height: int = 360, title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor=t["surface_container"], plot_bgcolor=t["surface_container"],
        font=dict(family=FONT, size=13, color=t["on_surface"]),
        title=dict(text=title, font=dict(size=16), x=0.01, xanchor="left", y=0.97) if title else None,
        margin=dict(l=12, r=16, t=64 if title else 28, b=12), hovermode="x unified",
        hoverlabel=dict(bgcolor=t["surface_high"], font=dict(family=FONT, color=t["on_surface"]), bordercolor=t["outline_variant"]),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                                       font=dict(color=t["on_surface_variant"], size=12)),
        separators=", ", bargap=0.25)
    fig.update_xaxes(gridcolor=t["outline_variant"], griddash="dot", linecolor=t["outline_variant"], zeroline=False,
                     tickfont=dict(color=t["on_surface_variant"]), showspikes=True, spikecolor=t["outline"],
                     spikethickness=1, spikedash="dot", spikemode="across", tickformat="%d.%m.%y", hoverformat="%d.%m.%Y")
    fig.update_yaxes(gridcolor=t["outline_variant"], griddash="dot", linecolor=t["outline_variant"], zeroline=False,
                     tickfont=dict(color=t["on_surface_variant"]), title_font=dict(color=t["on_surface_variant"], size=12))
    return fig


def price_chart(ind: pd.DataFrame, levels: dict, show: list[str], t: dict, symbol: str) -> go.Figure:
    """Цена с индикаторами и три панели под ней: объём, RSI, MACD."""
    s = t["series"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.52, 0.14, 0.16, 0.18])
    x = ind.index
    if "Полосы Боллинджера" in show:
        fig.add_trace(go.Scatter(x=x, y=ind["bb_upper"], line=dict(color=t["outline"], width=1), name="Боллинджер, верх",
                                 hovertemplate="%{y:.2f}"), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=ind["bb_lower"], line=dict(color=t["outline"], width=1), name="Боллинджер, низ",
                                 fill="tonexty", fillcolor="rgba(138,137,133,0.12)", hovertemplate="%{y:.2f}"), 1, 1)
    fig.add_trace(go.Candlestick(x=x, open=ind["open"], high=ind["high"], low=ind["low"], close=ind["close"],
                                 name=f"{symbol}/USDT", increasing_line_color=t["good"], decreasing_line_color=t["critical"],
                                 increasing_fillcolor=t["good"], decreasing_fillcolor=t["critical"], line_width=1), 1, 1)
    lines = {"SMA 20": ("sma20", s[0], "solid"), "SMA 50": ("sma50", s[1], "solid"),
             "EMA 12": ("ema12", s[2], "dot"), "EMA 26": ("ema26", s[3], "dot")}
    for name, (col, color, dash) in lines.items():
        if name in show:
            fig.add_trace(go.Scatter(x=x, y=ind[col], name=name, line=dict(color=color, width=2, dash=dash),
                                     hovertemplate="%{y:.2f}"), 1, 1)
    if "Уровни" in show:
        for key, label in (("support", "Поддержка"), ("resistance", "Сопротивление")):
            lv = levels.get(key)
            if lv:
                fig.add_hline(y=lv["level"], line=dict(color=t["on_surface_variant"], width=1.5, dash="dash"), row=1, col=1,
                              annotation_text=f"{label} {lv['level']:.2f}", annotation_position="top left",
                              annotation_font=dict(color=t["on_surface_variant"], size=12))
    up = ind["close"] >= ind["open"]
    fig.add_trace(go.Bar(x=x, y=ind["volume"], name="Объём", marker_color=[t["good"] if u else t["critical"] for u in up],
                         marker_line_width=0, opacity=0.7, hovertemplate="%{y:,.0f}", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=ind["vol_sma20"], name="Объём, SMA 20", line=dict(color=t["on_surface_variant"], width=1.5),
                             hovertemplate="%{y:,.0f}", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=ind["rsi14"], name="RSI 14", line=dict(color=s[0], width=2), hovertemplate="%{y:.1f}",
                             showlegend=False), 3, 1)
    for level in (30, 70):
        fig.add_hline(y=level, line=dict(color=t["outline"], width=1, dash="dash"), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=t["critical"], opacity=0.08, line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor=t["good"], opacity=0.08, line_width=0, row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=ind["hist"], name="Гистограмма MACD", marker_line_width=0, opacity=0.7,
                         marker_color=[t["good"] if v >= 0 else t["critical"] for v in ind["hist"].fillna(0)],
                         hovertemplate="%{y:.3f}", showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=x, y=ind["macd"], name="MACD", line=dict(color=s[0], width=2), hovertemplate="%{y:.3f}",
                             showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=x, y=ind["signal"], name="Сигнальная", line=dict(color=s[1], width=2), hovertemplate="%{y:.3f}",
                             showlegend=False), 4, 1)
    _layout(fig, t, height=760)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Цена, USDT", row=1, col=1)
    fig.update_yaxes(title_text="Объём", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], tickvals=[30, 50, 70], row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    return fig


def line(series: pd.Series, t: dict, title: str, unit: str = "", color_idx: int = 0, height: int = 260,
         smooth: int | None = 7) -> go.Figure:
    """Один показатель: суточные значения тонкой линией и среднее за 7 дней основной."""
    fig = go.Figure()
    c = t["series"][color_idx]
    if smooth:
        fig.add_trace(go.Scatter(x=series.index, y=series, name="за сутки", line=dict(color=t["outline"], width=1),
                                 hovertemplate="%{y:,.4s}"))
        fig.add_trace(go.Scatter(x=series.index, y=series.rolling(smooth, min_periods=1).mean(), name=f"среднее за {smooth} дней",
                                 line=dict(color=c, width=2), hovertemplate="%{y:,.4s}"))
    else:
        fig.add_trace(go.Scatter(x=series.index, y=series, name=title, line=dict(color=c, width=2), hovertemplate="%{y:,.4s}"))
    _layout(fig, t, height=height, title=title, legend=bool(smooth))
    fig.update_yaxes(title_text=unit)
    return fig


def indexed(metric: pd.Series, price: pd.Series, t: dict, metric_name: str, symbol: str) -> go.Figure:
    """Показатель и цена, приведённые к 100 в начале периода: общая шкала вместо двух осей."""
    df = pd.concat([metric.rolling(7, min_periods=1).mean().rename("m"), price.rename("p")], axis=1, sort=True).dropna()
    fig = go.Figure()
    if len(df):
        base = df.iloc[0]
        fig.add_trace(go.Scatter(x=df.index, y=df["m"] / base["m"] * 100, name=f"{metric_name} (среднее за 7 дней)",
                                 line=dict(color=t["series"][0], width=2), hovertemplate="%{y:.1f}"))
        fig.add_trace(go.Scatter(x=df.index, y=df["p"] / base["p"] * 100, name=f"Цена {symbol}",
                                 line=dict(color=t["series"][1], width=2), hovertemplate="%{y:.1f}"))
        fig.add_hline(y=100, line=dict(color=t["outline"], width=1, dash="dash"))
    _layout(fig, t, height=340, title=f"{metric_name} и цена, начало периода = 100")
    return fig


def sentiment_daily(daily: pd.DataFrame, t: dict) -> go.Figure:
    """Сверху число публикаций по тональности, снизу индекс настроений."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.5])
    for col, name, color in (("positive", "Положительные", t["good"]), ("neutral", "Нейтральные", t["neutral"]),
                             ("negative", "Отрицательные", t["critical"])):
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color,
                             marker_line=dict(width=1, color=t["surface_container"]), hovertemplate="%{y:.0f}"), 1, 1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["index"], name="Индекс за день", mode="markers",
                             marker=dict(color=t["outline"], size=7), hovertemplate="%{y:+.2f}"), 2, 1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["index_7d"], name="Среднее за 7 дней",
                             line=dict(color=t["series"][0], width=2), hovertemplate="%{y:+.2f}"), 2, 1)
    fig.add_hline(y=0, line=dict(color=t["outline"], width=1), row=2, col=1)
    _layout(fig, t, height=460)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Публикаций", row=1, col=1)
    fig.update_yaxes(title_text="Индекс, от −1 до +1", range=[-1.05, 1.05], row=2, col=1)
    return fig


def module_bars(rows: list[dict], t: dict) -> go.Figure:
    """Оценки четырёх модулей на общей шкале от -100 до +100."""
    names = [r["name"] for r in rows][::-1]
    vals = [r["score"] if r["score"] is not None else 0 for r in rows][::-1]
    labels = [(f"{r['score']:+.0f}" if r["score"] is not None else "нет данных")
              + ("" if r["state"] == "учтён" else f" · {r['state']}") for r in rows][::-1]
    colors = [t["neutral"] if r["state"] != "учтён" else t["good"] if (r["score"] or 0) >= 0 else t["critical"] for r in rows][::-1]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors, marker_line_width=0, text=labels,
                           textposition="outside", textfont=dict(color=t["on_surface"], size=13), cliponaxis=False,
                           hovertemplate="%{y}: %{x:+.1f}<extra></extra>", width=0.5))
    fig.add_vline(x=0, line=dict(color=t["outline"], width=1))
    _layout(fig, t, height=250, legend=False)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[-125, 125], tickvals=[-100, -30, 0, 30, 100], ticktext=["−100", "−30", "0", "+30", "+100"],
                     tickangle=0, showspikes=False)
    fig.update_yaxes(showgrid=False)
    return fig


def compare_lines(a: pd.Series, b: pd.Series, name_a: str, name_b: str, t: dict) -> go.Figure:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1, sort=True).dropna()
    fig = go.Figure()
    if len(df):
        fig.add_trace(go.Scatter(x=df.index, y=df["a"] / df["a"].iloc[0] * 100, name=name_a,
                                 line=dict(color=t["series"][0], width=2), hovertemplate="%{y:.1f}"))
        fig.add_trace(go.Scatter(x=df.index, y=df["b"] / df["b"].iloc[0] * 100, name=name_b,
                                 line=dict(color=t["series"][1], width=2), hovertemplate="%{y:.1f}"))
        fig.add_hline(y=100, line=dict(color=t["outline"], width=1, dash="dash"))
    return _layout(fig, t, height=320, title="Цена, начало периода = 100")
