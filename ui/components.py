"""Компоненты интерфейса: карточки, чипы, шкалы. Весь внешний текст экранируется."""
from __future__ import annotations

import html
import math
import re

import streamlit as st

esc = lambda s: html.escape(str(s)) if s is not None else ""


def ru(text) -> str:
    """Экранирует текст объяснения и ставит десятичную запятую в числах (1.05 -> 1,05)."""
    return re.sub(r"(?<=\d)\.(?=\d)", ",", esc(text))


def fmt_num(x, digits: int = 2, signed: bool = False) -> str:
    """Число в русской записи: пробел между разрядами, запятая перед дробной частью."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    s = f"{x:+,.{digits}f}" if signed else f"{x:,.{digits}f}"
    return s.replace(",", " ").replace(".", ",").replace("-", "−")


def fmt_big(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    for limit, unit in ((1e12, "трлн"), (1e9, "млрд"), (1e6, "млн"), (1e3, "тыс.")):
        if abs(x) >= limit:
            return f"{fmt_num(x / limit, 2)} {unit}"
    return fmt_num(x, 2)


def fmt_pct(x, digits: int = 1, signed: bool = True) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt_num(x * 100, digits, signed) + "%"


def fmt_score(x) -> str:
    return "—" if x is None else fmt_num(x, 0, signed=True)


def tone(x) -> str:
    return "flat" if x is None or abs(x) < 1e-9 else ("up" if x > 0 else "down")


def arrow(x) -> str:
    return "" if x is None or abs(x) < 1e-9 else ("▲ " if x > 0 else "▼ ")


def appbar(title: str, sub: str = "") -> None:
    st.html(f'<div class="md-appbar"><h1>{esc(title)}</h1><span class="sub">{esc(sub)}</span></div>')


def section(title: str) -> None:
    st.html(f'<div class="md-section">{esc(title)}</div>')


def card(label: str, value: str, support: str = "", kind: str = "", small: bool = False, extra: str = "") -> None:
    """value и support могут содержать подготовленную разметку (без внешнего текста)."""
    st.html(f'<div class="md-card {kind}"><div class="md-label">{esc(label)}</div>'
            f'<div class="md-value {"small" if small else ""}">{value}</div>'
            f'<div class="md-support">{support}</div>{extra}</div>')


def text_card(title: str, body_html: str, kind: str = "") -> None:
    st.html(f'<div class="md-card {kind}"><div class="md-title">{esc(title)}</div><div class="md-body">{body_html}</div></div>')


def chip(text: str, kind: str = "") -> str:
    dot = '<span class="dot"></span>' if kind in ("good", "bad", "warn") else ""
    return f'<span class="md-chip {kind}">{dot}{esc(text)}</span>'


def banner(text: str, kind: str = "info", title: str | None = None) -> None:
    icon = {"warn": "Предупреждение", "bad": "Ошибка", "info": "Сведения", "good": "Готово"}[kind]
    st.html(f'<div class="md-banner {kind}"><div><b>{esc(title or icon)}.</b> {ru(text)}</div></div>')


def scale(score: float | None) -> str:
    """Шкала интегральной оценки от -100 до +100 с отметкой."""
    if score is None:
        return ""
    pos = (max(-100, min(100, score)) + 100) / 2
    return (f'<div class="md-scale"><div class="thumb" style="left:{pos:.1f}%"></div></div>'
            '<div class="md-scale-labels"><span>−100</span><span>−30</span><span>0</span><span>+30</span><span>+100</span></div>')


def progress(value: float | None) -> str:
    v = 0 if value is None else max(0, min(100, value))
    return f'<div class="md-progress"><div style="width:{v:.0f}%"></div></div>'


def badge(score, digits: int = 0) -> str:
    if score is None:
        return '<span class="md-badge">нет данных</span>'
    cls = "pos" if score > 0 else "neg" if score < 0 else ""
    return f'<span class="md-badge {cls}">{fmt_num(score, digits, signed=True)}</span>'


def table(headers: list[str], rows: list[list[str]], num_cols: tuple = ()) -> None:
    """Таблица из подготовленной разметки ячеек."""
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    st.html(f'<div class="md-card outlined" style="padding:4px 8px"><table class="md-table"><thead><tr>{th}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def bullet_list(items: list[str]) -> str:
    return '<ul class="md-list">' + "".join(f"<li>{ru(i)}</li>" for i in items) + "</ul>"
