"""AvaLens — пользовательский интерфейс (Streamlit, оформление Material 3).

Запуск:  streamlit run app.py
Тестовый режим без сети (синтетические данные):  AVALENS_TEST_MODE=1 streamlit run app.py
"""
from __future__ import annotations

import inspect
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import integral, sentiment, technical
from analysis.integral import NAMES
from core import pipeline
from report import report as report_builder
from storage.db import Database
from ui import charts, theme
from ui.components import (appbar, arrow, badge, banner, bullet_list, card, chip, esc, fmt_big, fmt_num, fmt_pct,
                           fmt_score, progress, ru, scale, section, table, text_card, tone)

ROOT = Path(__file__).resolve().parent
TEST_MODE = os.environ.get("AVALENS_TEST_MODE") == "1"
PAGES = ["Главная панель", "Технический анализ", "Фундаментальный анализ", "Ончейн-анализ",
         "Анализ настроений", "Комплексный анализ", "История и данные"]
PLOT_CFG = {"displayModeBar": False, "locale": "ru"}
LABELS_RU = {"positive": "положительная", "negative": "отрицательная", "neutral": "нейтральная"}

st.set_page_config(page_title="AvaLens — анализ Avalanche", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def get_db() -> Database:
    if TEST_MODE:
        from tests.fake_api import FakeAPI, install
        install(FakeAPI())
        return Database(ROOT / "avalens_test.db")
    return Database(ROOT / "avalens.db")


@st.cache_resource
def get_cfg() -> dict:
    return pipeline.load_config()


db, cfg = get_db(), get_cfg()

# ---------------- тема ----------------
st.session_state.setdefault("dark", False)
st.session_state.setdefault("palette", next(iter(theme.PALETTES)))
T = theme.tokens(st.session_state["palette"], st.session_state["dark"])
if theme.apply_streamlit_theme(T):
    st.rerun()
st.markdown(theme.css(T), unsafe_allow_html=True)


def plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)


def frame(df: pd.DataFrame, height: int | None = None, key: str | None = None) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **({"height": height} if height else {}), key=key)


# ---------------- боковая панель ----------------
yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
with st.sidebar:
    st.html('<div style="padding:4px 16px 12px"><div style="font-size:22px;line-height:28px;font-weight:500">AvaLens</div>'
            '<div class="md-support">Комплексный анализ криптовалют</div></div>')
    page = st.radio("Раздел", PAGES, label_visibility="collapsed", key="page")
    st.divider()
    symbols = list(cfg["assets"])
    symbol = st.selectbox("Криптовалюта", symbols, format_func=lambda s: f"{cfg['assets'][s]['name']} ({s})", key="symbol")
    preset = st.selectbox("Период", ["90 дней", "180 дней", "365 дней", "Свой период"], index=1, key="preset")
    if preset == "Свой период":
        picked = st.date_input("Даты", (yesterday - timedelta(days=179), yesterday), max_value=yesterday,
                               format="DD.MM.YYYY", key="dates")
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            p_start, p_end = picked
        else:
            p_start, p_end = yesterday - timedelta(days=179), yesterday
    else:
        p_end = yesterday
        p_start = p_end - timedelta(days=int(preset.split()[0]) - 1)
    do_refresh = st.button("Обновить данные", width="stretch", help="Загрузить данные из внешних источников")
    do_run = st.button("Рассчитать и сохранить", type="primary", width="stretch",
                       help="Выполнить расчёт за выбранный период и записать его в историю")
    st.divider()
    st.toggle("Тёмная тема", key="dark")
    st.selectbox("Цветовая схема", list(theme.PALETTES), key="palette")

period_start, period_end = p_start.strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d")

# ---------------- загрузка и расчёт ----------------
if do_refresh:
    with st.status("Загрузка данных из источников…", expanded=True) as status:
        rep = pipeline.refresh(db, cfg, progress=lambda msg: st.write(msg))
        st.session_state["refresh_report"] = rep
        st.session_state["refresh_time"] = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
        failed = [r for r in rep if not r["ok"] and r["note"] != "skip"]
        status.update(label="Данные загружены" if not failed else f"Загрузка завершена, источников с ошибкой: {len(failed)}",
                      state="complete" if not failed else "error", expanded=False)
    st.session_state.pop("result_key", None)

has_data = len(db.load_prices(symbol)) > 0
key = (symbol, period_start, period_end)
if has_data and (do_run or st.session_state.get("result_key") != key):
    with st.spinner("Расчёт показателей…"):
        st.session_state["result"] = pipeline.analyze(db, cfg, symbol, period_start, period_end, save=bool(do_run))
        st.session_state["result_key"] = key
    if do_run:
        st.toast(f"Расчёт № {st.session_state['result']['run_id']} сохранён в историю")
res = st.session_state.get("result") if has_data else None
asset_name = cfg["assets"][symbol]["name"]


def show_refresh_problems() -> None:
    rep = st.session_state.get("refresh_report")
    if not rep:
        return
    for r in rep:
        if not r["ok"] and r["note"] != "skip":
            banner(f"{r['source']} ({r['symbol']}): {r['error']}. Используются последние сохранённые данные; "
                   f"если их нет, зависящий от источника модуль отключается.", "bad", "Источник недоступен")
        elif r.get("backup"):
            banner(f"{r['source']} ({r['symbol']}): {r['backup']}.", "warn", "Резервный источник")


def test_banner() -> None:
    if TEST_MODE:
        banner("Источники заменены имитацией, все числа синтетические и не описывают реальный рынок.", "warn", "Тестовый режим")


def explain(text: str) -> None:
    st.html(f'<div class="md-card outlined"><div class="md-label">ПОЧЕМУ ТАКАЯ ОЦЕНКА</div>'
            f'<div class="md-body" style="margin-top:6px">{ru(text)}</div></div>')


def quality_line(m: dict) -> str:
    p = m["quality_parts"]
    return (f"Качество данных q = {fmt_num(m['quality'])}: полнота {fmt_num(p['completeness'])} × "
            f"актуальность {fmt_num(p['freshness'])} × надёжность {fmt_num(p['reliability'])}")


# ======================= страницы =======================
def page_dashboard() -> None:
    it = res["integral"]
    quote = pipeline.current_quote(db, symbol)
    ta_d = res["modules"]["TA"]["details"]
    updated = pd.Timestamp(quote["time"]).strftime("%d.%m.%Y %H:%M UTC") if quote else "нет"
    appbar(f"{asset_name} ({symbol})", f"Период {p_start:%d.%m.%Y} — {p_end:%d.%m.%Y} · данные обновлены: {updated}")
    test_banner()
    show_refresh_problems()

    c1, c2, c3 = st.columns([1, 1.25, 1])
    with c1:
        if quote:
            ch = quote.get("change_pct_24h")
            sup = (f'<span class="{tone(ch)}">{arrow(ch)}{fmt_num(ch, 2, signed=True)}% за 24 часа</span><br>'
                   f'{esc(quote["source"])}, {updated}') if ch is not None else f'{esc(quote["source"])}, {updated}'
            card("ТЕКУЩАЯ ЦЕНА", f"{fmt_num(quote['price'])} <span style='font-size:16px'>USDT</span>", sup)
        elif ta_d.get("close"):
            card("ЦЕНА ЗАКРЫТИЯ", f"{fmt_num(ta_d['close'])} <span style='font-size:16px'>USDT</span>",
                 f"на {esc(ta_d['date'])}; текущая цена не загружена")
        else:
            card("ТЕКУЩАЯ ЦЕНА", "—", "нет данных")
    with c2:
        if it["s_score"] is None:
            card("ИНТЕГРАЛЬНАЯ ОЦЕНКА S", "Недостаточно данных", "Оценка не выдаётся: " + esc("; ".join(it["weak_points"][:2])),
                 kind="primary", small=True)
        else:
            card("ИНТЕГРАЛЬНАЯ ОЦЕНКА S", fmt_num(it["s_score"], 1, signed=True),
                 esc(it["verdict"]) + (" · предварительно" if it["preliminary"] else ""), kind="primary", extra=scale(it["s_score"]))
    with c3:
        state = ("данных недостаточно" if it["confidence"] < integral.C_MIN else
                 "вывод предварительный" if it["confidence"] < integral.C_FULL else "данные достаточны для вывода")
        card("ДОСТОВЕРНОСТЬ (CONFIDENCE SCORE)", f"{fmt_num(it['confidence'], 0)} <span style='font-size:16px'>из 100</span>",
             esc(state), extra=progress(it["confidence"]))

    cols = st.columns(4)
    for col, r in zip(cols, it["modules"]):
        with col:
            mark = {"учтён": "good", "исключён": "warn", "недоступен": "bad"}[r["state"]]
            card({"TA": "TA · ТЕХНИЧЕСКИЙ", "FA": "FA · ФУНДАМЕНТАЛЬНЫЙ", "ON": "ON · ОНЧЕЙН", "SA": "SA · НАСТРОЕНИЯ"}[r["module"]], f'<span class="{tone(r["score"])}">{fmt_score(r["score"])}</span>',
                 chip(r["state"], mark) + f' q = {fmt_num(r["quality"])}', small=True)

    if it["conflicts"]:
        for c in it["conflicts"][:2]:
            banner(c["explanation"], "warn", f"Противоречие ({c['severity']})")
    for w in res["warnings"][:4]:
        banner(w, "warn")

    left, right = st.columns([1.6, 1])
    with left:
        section("Динамика цены")
        prices = res["frames"]["prices"]
        view = prices.loc[pd.Timestamp(period_start):] if len(prices) else prices
        peer = res["comparison"]["peer"] if res.get("comparison") else None
        if peer and len(view):
            from data import quality
            pp, _ = quality.clean_prices(db.load_prices(peer, end=period_end))
            plot(charts.compare_lines(view["close"], pp["close"].loc[pd.Timestamp(period_start):] if len(pp) else pp.get("close", pd.Series(dtype=float)),
                                      symbol, peer, T))
        elif len(view):
            plot(charts.line(view["close"], T, f"Цена {symbol}, USDT", smooth=None))
    with right:
        section("Изменение цены")
        cmp_rows = {r["title"]: r for r in (res["comparison"]["rows"] if res.get("comparison") else [])}
        peer = res["comparison"]["peer"] if res.get("comparison") else None
        rows = []
        for key_, title in (("return_1d", "За сутки"), ("return_7d", "За 7 дней"), ("return_30d", "За 30 дней"), ("return_90d", "За 90 дней")):
            v = ta_d.get(key_)
            pv = cmp_rows.get({"return_7d": "Доходность за 7 дней", "return_30d": "Доходность за 30 дней",
                               "return_90d": "Доходность за 90 дней"}.get(key_, ""), {}).get(peer) if peer else None
            rows.append([title, f'<span class="{tone(v)}">{arrow(v)}{fmt_pct(v)}</span>',
                         f'<span class="{tone(pv)}">{fmt_pct(pv)}</span>' if peer else ""])
        table(["Интервал", symbol, peer or ""], rows, num_cols=(1, 2))
        st.html(f'<div class="md-support">Цены закрытия на {esc(ta_d.get("date", "—"))}. '
                f'{esc(peer or "")} — актив сравнения.</div>')

    section("Состояние источников")
    status = db.source_status()
    if status.empty:
        st.html('<div class="md-support">Запросов к источникам ещё не было.</div>')
    else:
        chips = []
        for r in status.itertuples():
            has = isinstance(r.last_success, str) and bool(r.last_success)
            ok = has and r.last_success == r.last_attempt
            when = pd.Timestamp(r.last_success).strftime("%d.%m %H:%M") if has else "нет успешных ответов"
            chips.append(chip(f"{r.source} · {when}", "good" if ok else "bad"))
        st.html("".join(chips) + '<div class="md-support">Время последнего успешного ответа, UTC. '
                                 'Красная отметка — последний запрос завершился ошибкой.</div>')


def page_technical() -> None:
    m = res["modules"]["TA"]
    appbar("Технический анализ", f"{asset_name} ({symbol}) · дневные свечи · оценка на {period_end}")
    test_banner()
    if m["status"] != "ok":
        banner(m["explanation"], "bad", "Модуль недоступен")
        return
    d = m["details"]
    tab_a, tab_b, tab_c = st.tabs(["Анализ", "Как посчитано", "Исходные данные"])
    with tab_a:
        c = st.columns(5)
        with c[0]:
            card("ОЦЕНКА TA", f'<span class="{tone(m["score"])}">{fmt_score(m["score"])}</span>', "от −100 до +100", kind="primary")
        with c[1]:
            card("ТРЕНД", esc({"up": "Восходящий", "down": "Нисходящий", "sideways": "Боковой", "mixed": "Переходный"}[d["trend"]]),
                 f"наклон {fmt_num(d['slope_pct_day'], 2, signed=True)}% в день", small=True)
        with c[2]:
            card("RSI (14)", fmt_num(d["rsi"], 1), esc(d["rsi_zone"]), small=True)
        with c[3]:
            sup = d.get("support")
            card("ПОДДЕРЖКА", fmt_num(sup["level"]) if sup else "—", f"касаний: {sup['touches']}" if sup else "не найдена", small=True)
        with c[4]:
            rs = d.get("resistance")
            card("СОПРОТИВЛЕНИЕ", fmt_num(rs["level"]) if rs else "—", f"касаний: {rs['touches']}" if rs else "не найдено", small=True)
        explain(m["explanation"])
        show = st.multiselect("Показать на графике цены", ["SMA 20", "SMA 50", "EMA 12", "EMA 26", "Полосы Боллинджера", "Уровни"],
                              default=["SMA 20", "SMA 50", "Полосы Боллинджера", "Уровни"], key="ta_show")
        plot(charts.price_chart(res["frames"]["indicators"], d, show, T, symbol))
        section("Технические сигналы")
        table(["Сигнал", "Значение", "Вес", "Вклад в TA", "Объяснение"],
              [[esc(s["name"]), badge(s["value"], 2), fmt_num(s["weight"]), badge(s["contribution"], 1), ru(s["text"])]
               for s in m["signals"]], num_cols=(1, 2, 3))
        if m.get("risks"):
            text_card("Риски", bullet_list(m["risks"]))
    with tab_b:
        st.markdown(f"""
**Оценка TA** — взвешенная сумма пяти сигналов, каждый в диапазоне от −1 до +1, умноженная на 100.
Индикатор за день *t* использует только цены до дня *t* включительно.

| Индикатор | Формула | Что показывает |
| --- | --- | --- |
| SMA(n) | среднее арифметическое последних *n* цен закрытия | среднюю цену за период |
| EMA(n) | EMAₜ = α·Pₜ + (1 − α)·EMAₜ₋₁, α = 2 / (n + 1); первое значение — SMA первых *n* цен | то же, но свежие цены весят больше, поэтому EMA быстрее реагирует на разворот |
| RSI(14) | 100 − 100 / (1 + средний рост / среднее падение), сглаживание Уайлдера | скорость изменения цены: выше 70 — перекупленность, ниже 30 — перепроданность |
| MACD | EMA(12) − EMA(26); сигнальная линия — EMA(9) от MACD | импульс движения: MACD выше сигнальной — импульс растёт |
| Полосы Боллинджера | SMA(20) ± 2 стандартных отклонения | волатильность: ширина канала растёт вместе с разбросом цен |
| Объём | среднее за 5 дней к среднему за 20 дней | подтверждает движение цены, если объём выше обычного не менее чем на 10% |

**Тренд.** Три признака: цена относительно SMA50, SMA20 относительно SMA50, наклон линейной регрессии логарифма цены за 30 дней
(порог ±0,1% в день). Все три «вверх» — восходящий тренд, все три «вниз» — нисходящий, наклон в пределах порога — боковое движение.

**Уровни поддержки и сопротивления.** Локальные минимумы и максимумы за 180 дней (экстремум среди 5 дней с каждой стороны);
экстремумы ближе 3% друг к другу объединяются в один уровень, число касаний — его сила.
""")
        section("Значения индикаторов за последние 10 дней")
        ind = res["frames"]["indicators"].tail(10)[["close", "sma20", "sma50", "ema12", "ema26", "rsi14", "macd", "signal", "hist",
                                                    "bb_upper", "bb_lower", "volume"]].round(4)
        frame(ind.reset_index().rename(columns={"date": "дата"}))
        st.html(f'<div class="md-support">{esc(quality_line(m))}</div>')
    with tab_c:
        rp = res["price_report"]
        st.html("".join([chip(f"Источник: {m['quality_parts']['source']}", "tonal"), chip(f"Свечей получено: {rp['rows']}"),
                         chip(f"Пропущено дней: {rp['missing_days']}", "warn" if rp["missing_days"] else "good"),
                         chip(f"Аномалий: {len(rp['anomalies'])}", "warn" if rp["anomalies"] else "good"),
                         chip(f"Последняя свеча: {rp['last_date']}")]))
        st.markdown("**Свечи, полученные от источника**")
        raw = res["frames"]["raw_prices"].copy()
        raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
        frame(raw.sort_values("date", ascending=False), height=320)
        st.download_button("Скачать исходные свечи (CSV)", raw.to_csv(index=False).encode("utf-8"), f"{symbol}_prices_raw.csv", "text/csv")
        st.markdown("**После обработки** — `is_filled`: день заполнен ценой предыдущего дня; `is_anomaly`: значение исключено как аномальное")
        clean = res["frames"]["prices"].reset_index()
        clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
        frame(clean.sort_values("date", ascending=False), height=320)


def page_fundamental() -> None:
    m = res["modules"]["FA"]
    d = m["details"]
    appbar("Фундаментальный анализ", f"{asset_name} ({symbol}) · оценка на {period_end}")
    test_banner()
    tab_a, tab_b, tab_c = st.tabs(["Анализ", "Как посчитано", "Исходные данные"])
    with tab_a:
        if m["status"] != "ok":
            banner(m["explanation"], "bad", "Недостаточно данных для оценки")
        c = st.columns(5)
        with c[0]:
            card("ОЦЕНКА FA", f'<span class="{tone(m["score"])}">{fmt_score(m["score"])}</span>',
                 f"оценено направлений: {d['n_scored']} из 8", kind="primary")
        with c[1]:
            card("MARKET CAP", fmt_big(d["market_cap"]), "долл.", small=True)
        with c[2]:
            card("FDV", fmt_big(d["fdv"]), "долл., при полной эмиссии", small=True)
        with c[3]:
            card("В ОБРАЩЕНИИ", fmt_big(d["circulating"]), f"максимум {fmt_big(d['max_supply'])}" if d["max_supply"] else "максимума нет", small=True)
        with c[4]:
            card("TVL СЕТИ", fmt_big(d["tvl"]), f"протоколов: {d['protocols_count']}" if d["protocols_count"] is not None else "долл.", small=True)
        if m["status"] == "ok":
            explain(m["explanation"])
        section("Оценка по направлениям")
        table(["Направление", "Балл", "Экономическое следствие для токена"],
              [[esc(c_["title"]), badge(c_["score"]), ru(c_["consequence"])] for c_ in d["criteria"]], num_cols=(1,))
        if m.get("risks"):
            text_card("Риски", bullet_list(m["risks"]))
        cmp_ = res.get("comparison")
        if cmp_:
            section(f"Сравнение с {cmp_['peer']}")
            def cell(v, kind):
                return "—" if v is None else fmt_pct(v) if kind == "pct" else fmt_big(v) if kind == "usd" else fmt_num(v)
            table(["Показатель", symbol, cmp_["peer"]],
                  [[esc(r["title"]), cell(r.get(symbol), r["kind"]), cell(r.get(cmp_["peer"]), r["kind"])] for r in cmp_["rows"]],
                  num_cols=(1, 2))
            if cmp_.get("correlation_90d") is not None:
                st.html(f'<div class="md-support">Корреляция дневных доходностей за 90 дней: {fmt_num(cmp_["correlation_90d"])}.</div>')
    with tab_b:
        st.markdown("""
Каждое из восьми направлений получает балл от −2 до +2 по правилу из таблицы ниже.
Направление без данных помечается «нет данных» и в расчёт не входит.

**FA = 100 × сумма баллов / (2 × n)**, где *n* — число оценённых направлений. При *n* < 5 оценка не выдаётся.
""")
        for c_ in d["criteria"]:
            with st.expander(f"{c_['title']}: {'нет данных' if c_['score'] is None else format(c_['score'], '+d')}"):
                st.markdown(f"**Правило.** {c_['rule']}")
                st.markdown("**Факты, по которым поставлен балл**")
                st.json(c_["facts"], expanded=True)
        st.html(f'<div class="md-support">{esc(quality_line(m))}</div>')
    with tab_c:
        st.markdown("**Рыночные снимки (CoinGecko, DefiLlama)** — сохраняются при каждом обновлении данных")
        frame(db.fundamentals_history(symbol).sort_values("snapshot_at", ascending=False), height=260)
        st.markdown(f"**Справочные факты о проекте** — файл `fundamentals/{symbol.lower()}.yaml`, проверены {d.get('facts_verified_at') or '—'}")
        facts_path = ROOT / "fundamentals" / f"{symbol.lower()}.yaml"
        st.code(facts_path.read_text(encoding="utf-8") if facts_path.exists() else "файл не найден", language="yaml")


def page_onchain() -> None:
    m = res["modules"]["ON"]
    d = m["details"]
    appbar("Ончейн-анализ", f"{asset_name} ({symbol}) · суточные показатели сети · оценка на {period_end}")
    test_banner()
    series = res["frames"]["onchain"]
    rows = d.get("metrics", [])
    tab_a, tab_b, tab_c = st.tabs(["Анализ", "Как посчитано", "Исходные данные"])
    with tab_a:
        if m["status"] != "ok":
            banner(m["explanation"], "bad", "Ончейн-оценка не рассчитана")
        avail = [r for r in rows if r.get("available")]
        cols = st.columns(4)
        with cols[0]:
            card("ОЦЕНКА ON", f'<span class="{tone(m["score"])}">{fmt_score(m["score"])}</span>',
                 f"показателей: {d.get('n_available', 0)} из {d.get('n_expected', 0)}", kind="primary")
        for col, r in zip(cols[1:], avail[:3]):
            with col:
                card(r["title"].upper(), fmt_big(r["last"]),
                     f'{esc(r["unit"])}<br><span class="{tone(r["delta"])}">{arrow(r["delta"])}{fmt_pct(r["delta"])}</span> к среднему за 30 дней', small=True)
        if m["status"] == "ok":
            explain(m["explanation"])
        section("Показатели сети")
        trows = []
        for r in rows:
            if r.get("available"):
                corr = r["corr"]["returns"]
                trows.append([f'{esc(r["title"])}<div class="md-support">{esc(r["meaning"])}</div>',
                              f'{fmt_big(r["last"])}<div class="md-support">{esc(r["unit"])}</div>',
                              f'<span class="{tone(r["change_7d"])}">{fmt_pct(r["change_7d"])}</span>',
                              f'<span class="{tone(r["delta"])}">{fmt_pct(r["delta"])}</span>', badge(r["signal"], 2),
                              "—" if corr is None else fmt_num(corr, 2, signed=True)])
            else:
                trows.append([f'{esc(r["title"])}<div class="md-support">{esc(r["meaning"])}</div>',
                              chip(r.get("note", "нет данных"), "bad"), "—", "—", badge(None), "—"])
        table(["Показатель", "Последнее значение", "Неделя к неделе", "7 дней к 30 дням", "Сигнал", "Связь с ценой, r"],
              trows, num_cols=(1, 2, 3, 4, 5))
        if avail and series:
            section("Динамика и сопоставление с ценой")
            names = {r["metric"]: r["title"] for r in avail}
            pick = st.selectbox("Показатель", list(names), format_func=names.get, key="on_pick")
            s = series[pick].loc[pd.Timestamp(period_start):]
            unit = next(r["unit"] for r in avail if r["metric"] == pick)
            a, b = st.columns(2)
            with a:
                plot(charts.line(s, T, names[pick], unit, height=340))
            with b:
                price = res["frames"]["prices"]["close"].loc[pd.Timestamp(period_start):]
                plot(charts.indexed(s, price, T, names[pick], symbol))
            others = [k for k in names if k != pick]
            for i in range(0, len(others), 2):
                for col, k in zip(st.columns(2), others[i:i + 2]):
                    with col:
                        plot(charts.line(series[k].loc[pd.Timestamp(period_start):], T, names[k],
                                         next(r["unit"] for r in avail if r["metric"] == k), color_idx=0))
        if d.get("conclusions"):
            text_card("Выводы", bullet_list(d["conclusions"]))
        if m.get("risks"):
            text_card("Риски", bullet_list(m["risks"]))
    with tab_b:
        st.markdown("""
Для каждого показателя сравниваются два окна: **среднее за последние 7 дней** и **среднее за предыдущие 30 дней**.

**сигнал = изменение / 25%**, ограниченный диапазоном от −1 до +1. Изменение на +25% и более даёт +1, на −25% и менее — −1.

**ON = 100 × среднее арифметическое сигналов** по всем полученным показателям. Если получено меньше трёх показателей, оценка не выдаётся.

**TVL** пересчитывается из долларов в токены сети (TVL / цена): иначе рост цены сам по себе выглядел бы ростом активности.

**Связь с ценой** — коэффициент корреляции Пирсона между дневными изменениями показателя и дневными изменениями цены за выбранный период.
Значение по модулю от 0,3 считается заметной связью. Корреляция не доказывает причинность.
""")
        st.html(f'<div class="md-support">{esc(quality_line(m))}</div>')
    with tab_c:
        raw = res["frames"]["onchain_raw"].copy()
        if raw.empty:
            banner("Ончейн-данных по этому активу в базе нет.", "warn")
        else:
            raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
            summary = raw.groupby(["metric", "source"]).agg(значений=("value", "size"), первая_дата=("date", "min"),
                                                            последняя_дата=("date", "max")).reset_index()
            frame(summary)
            wide = raw.pivot(index="date", columns="metric", values="value").sort_index(ascending=False).reset_index()
            frame(wide, height=360)
            st.download_button("Скачать ончейн-данные (CSV)", raw.to_csv(index=False).encode("utf-8"), f"{symbol}_onchain.csv", "text/csv")


def page_sentiment() -> None:
    m = res["modules"]["SA"]
    d = m["details"]
    appbar("Анализ настроений", f"{asset_name} ({symbol}) · окно {d.get('window_days')} дней до {period_end}")
    test_banner()
    posts = res["frames"].get("posts")
    daily = res["frames"].get("sentiment_daily")
    tab_a, tab_b, tab_c = st.tabs(["Анализ", "Как посчитано", "Публикации и ручная проверка"])
    with tab_a:
        if m["status"] != "ok":
            banner(m["explanation"], "bad", "Оценка настроений не рассчитана")
        n = d.get("n_posts", 0)
        cnt = d.get("counts", {"positive": 0, "negative": 0, "neutral": 0})
        share = lambda k: cnt[k] / n if n else None
        c = st.columns(5)
        with c[0]:
            card("ОЦЕНКА SA", f'<span class="{tone(m["score"])}">{fmt_score(m["score"])}</span>', "от −100 до +100", kind="primary")
        with c[1]:
            card("ПУБЛИКАЦИЙ УЧТЕНО", fmt_num(n, 0), f"из {d.get('n_window_total', 0)} за окно; в базе всего {d.get('n_total_in_db', 0)}", small=True)
        with c[2]:
            card("ПОЛОЖИТЕЛЬНЫЕ", fmt_pct(share("positive"), 0, signed=False), f"{cnt['positive']} публикаций", small=True)
        with c[3]:
            card("ОТРИЦАТЕЛЬНЫЕ", fmt_pct(share("negative"), 0, signed=False), f"{cnt['negative']} публикаций", small=True)
        with c[4]:
            card("НЕЙТРАЛЬНЫЕ", fmt_pct(share("neutral"), 0, signed=False), f"{cnt['neutral']} публикаций", small=True)
        if m["status"] == "ok":
            explain(m["explanation"])
        fg = d.get("fear_greed")
        fl = d.get("flags", {})
        st.html(chip(f"Fear & Greed всего рынка: {fmt_num(fg, 0) if fg is not None else 'нет данных'}", "tonal")
                + "".join(chip(f"{k}: {v}") for k, v in fl.items() if v)
                + "".join(chip(f"{k}: {v}", "tonal") for k, v in (d.get("by_source") or {}).items()))
        if daily is not None and len(daily):
            section("Динамика настроений")
            plot(charts.sentiment_daily(daily, T))
        if d.get("events"):
            section("Наиболее значимые информационные события")
            table(["Дата", "Событие", "Публикаций", "Тональность"],
                  [[esc(e["date"]), (f'<a href="{esc(e["url"])}" target="_blank" rel="noopener">{esc(e["title"])}</a>'
                                     if str(e.get("url") or "").startswith("http") else esc(e["title"])),
                    fmt_num(e["publications"], 0), badge(e["score"], 2)] for e in d["events"]], num_cols=(2, 3))
        fgs = res["frames"].get("fear_greed")
        if fgs is not None and len(fgs):
            section("Индекс страха и жадности")
            plot(charts.line(fgs.loc[pd.Timestamp(period_start):], T, "Fear & Greed Index (весь рынок)", "от 0 до 100", color_idx=3, smooth=None))
            st.html('<div class="md-support">Индекс описывает крипторынок в целом, а не выбранный актив, поэтому входит в оценку с весом 20%.</div>')
        if m.get("risks"):
            text_card("Риски", bullet_list(m["risks"]))
    with tab_b:
        st.markdown(inspect.getdoc(sentiment).split("\n", 2)[2])
        st.markdown(f"""
**Вес публикации в индексе.** {inspect.getdoc(sentiment.weigh).split(chr(10), 2)[2]}

**Итог.** Индекс публикаций = сумма (вес × знак) / сумма весов, где знак +1, 0 или −1.
SA = 100 × (0,8 × индекс публикаций + 0,2 × (Fear & Greed − 50) / 50).

| Особенность текста | Что делает система |
| --- | --- |
| Сарказм и ирония | положительная оценка при маркерах сарказма (lol, /s, yeah right…) переводится в нейтральную с флагом |
| Неоднозначные формулировки | вопрос в заголовке или смесь сильных положительных и отрицательных слов — нейтральная с флагом |
| Реклама | маркеры giveaway, promo code, presale и подобные — публикация исключается из индекса |
| Боты | заголовок почти без слов, обилие тикеров и хэштегов, 5 и более постов автора за сутки — исключается |
| Дубли | одинаковые заголовки не попадают в базу повторно |
| Крупные новости | пересказы одного события делят вес: k публикаций получают по 1/√k |

Словарь содержит {len(sentiment.LEXICON)} слов и словосочетаний на английском языке.
""")
        st.html(f'<div class="md-support">{esc(quality_line(m))}</div>')
    with tab_c:
        if posts is None or posts.empty:
            banner("Публикаций за выбранное окно нет.", "warn")
            return
        mc = d.get("manual_check", {})
        acc = mc.get("accuracy")
        st.html(chip(f"Проверено вручную: {mc.get('checked', 0)}", "tonal")
                + chip(f"Доля от окна: {fmt_pct(mc.get('share_of_window'), 0, signed=False)} (цель — не менее 20%)",
                       "good" if (mc.get("share_of_window") or 0) >= 0.2 else "warn")
                + chip(f"Совпадение с системой: {fmt_pct(acc, 0, signed=False) if acc is not None else '—'}"))
        st.markdown("Выберите в столбце **Ручная оценка** тональность, которую считаете верной, и нажмите «Сохранить проверку». "
                    "Столбец «Слова» показывает, что именно дало оценку.")
        view = pd.DataFrame({
            "id": posts["id"].to_numpy(),
            "Дата": posts["published_at"].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
            "Источник": posts["source"].to_numpy(),
            "Заголовок": posts["title"].to_numpy(),
            "Тональность": posts["label"].map(LABELS_RU).to_numpy(),
            "Оценка": posts["score"].round(2).to_numpy(),
            "Вес": posts["weight"].round(2).to_numpy(),
            "Слова": posts["matched_terms"].apply(lambda x: ", ".join(x)).to_numpy(),
            "Флаги": posts["flags"].apply(lambda x: ", ".join(x)).to_numpy(),
            "Ручная оценка": posts["manual_label"].map(LABELS_RU).fillna("").to_numpy(),
            "Ссылка": posts["url"].to_numpy(),
        })
        edited = st.data_editor(
            view, width="stretch", hide_index=True, height=460, key=f"posts_editor_{symbol}_{period_end}",
            disabled=[c for c in view.columns if c != "Ручная оценка"],
            column_config={"id": None, "Ссылка": st.column_config.LinkColumn(display_text="открыть"),
                           "Ручная оценка": st.column_config.SelectboxColumn(options=["", *LABELS_RU.values()])})
        if st.button("Сохранить проверку", type="primary"):
            back = {v: k for k, v in LABELS_RU.items()}
            changed = edited[edited["Ручная оценка"].fillna("") != view["Ручная оценка"]]
            for _, r in changed.iterrows():
                db.set_manual_label(int(r["id"]), back.get(r["Ручная оценка"] or "", None))
            st.session_state.pop("result_key", None)
            st.rerun()
        st.download_button("Скачать публикации (CSV)", view.drop(columns=["id"]).to_csv(index=False).encode("utf-8"),
                           f"{symbol}_posts.csv", "text/csv")


def page_integral() -> None:
    it = res["integral"]
    appbar("Комплексный анализ", f"{asset_name} ({symbol}) · период {p_start:%d.%m.%Y} — {p_end:%d.%m.%Y}")
    test_banner()
    c1, c2 = st.columns([1.3, 1])
    with c1:
        if it["s_score"] is None:
            card("ИНТЕГРАЛЬНАЯ ОЦЕНКА S", "Недостаточно данных для комплексной оценки",
                 "Система не формирует вывод, пока данных мало или они ненадёжны", kind="primary", small=True)
        else:
            card("ИНТЕГРАЛЬНАЯ ОЦЕНКА S", fmt_num(it["s_score"], 1, signed=True),
                 esc(it["verdict"]) + (" · вывод предварительный" if it["preliminary"] else ""), kind="primary", extra=scale(it["s_score"]))
    with c2:
        card("ДОСТОВЕРНОСТЬ (CONFIDENCE SCORE)", f"{fmt_num(it['confidence'], 0)} <span style='font-size:16px'>из 100</span>",
             f"порог вывода {integral.C_MIN}, порог уверенного вывода {integral.C_FULL}; согласованность периодов k = {fmt_num(it['k_sync'], 1)}",
             extra=progress(it["confidence"]))

    text_card("Итоговое аналитическое заключение", ru(it["summary"]))

    section("Противоречия между сигналами")
    if it["conflicts"]:
        for c in it["conflicts"]:
            banner(c["explanation"], "bad" if c["severity"] == "сильное" else "warn",
                   f"{c['severity'].capitalize()} противоречие, разрыв {fmt_num(c['gap'], 0)} пунктов")
    elif it["s_score"] is None:
        st.html('<div class="md-support">Противоречия не проверялись: недостаточно доступных модулей.</div>')
    else:
        banner(f"Нет пары модулей, в которой одна оценка не ниже +{integral.CONFLICT_LEVEL}, а другая не выше −{integral.CONFLICT_LEVEL}.",
               "good", "Противоречий не найдено")

    left, right = st.columns([1.1, 1])
    with left:
        section("Результаты четырёх модулей")
        plot(charts.module_bars(it["modules"], T))
    with right:
        section("Вклад в оценку S")
        rows = []
        for r in it["modules"]:
            contrib = it["contributions"].get(r["module"])
            rows.append([esc(r["name"]), badge(r["score"]), fmt_num(r["weight"]),
                         badge(contrib, 1) if contrib is not None else chip(r["state"], "warn" if r["state"] == "исключён" else "bad")])
        table(["Модуль", "Оценка", "Вес", "Вклад"], rows, num_cols=(1, 2, 3))

    section("Из чего складывается достоверность")
    rows = []
    for r in it["modules"]:
        p = r["quality_parts"]
        rows.append([esc(r["name"]), fmt_num(p.get("completeness")), fmt_num(p.get("freshness")), fmt_num(p.get("reliability")),
                     fmt_num(r["quality"]), fmt_num(100 * r["weight"] * r["quality"] * it["k_sync"], 1),
                     esc("; ".join(v for v in (p.get("notes") or {}).values() if v))])
    table(["Модуль", "Полнота", "Актуальность", "Надёжность", "q", "Вклад в C", "Пояснение"], rows, num_cols=(1, 2, 3, 4, 5))
    if it["weak_points"]:
        text_card("Слабые места данных", bullet_list(it["weak_points"]))

    section("Выявленные риски")
    if res["risks"]:
        table(["Модуль", "Риск"], [[esc(NAMES[r["module"]]), ru(r["text"])] for r in res["risks"]])
    else:
        st.html('<div class="md-support">Риски не выявлены.</div>')
    if res["warnings"]:
        section("Замечания к данным")
        for w in res["warnings"]:
            banner(w, "warn")

    section("Отчёт")
    md = report_builder.build(res, pipeline.current_quote(db, symbol))
    a, b = st.columns([1, 3])
    with a:
        st.download_button("Скачать заключение", md.encode("utf-8"), f"AvaLens_{symbol}_{period_end}.md", "text/markdown",
                           width="stretch")
    with b:
        st.html(f'<div class="md-support">Контрольная сумма входных данных: <code>{esc(res["input_hash"])}</code>. '
                f'Одинаковая сумма у двух расчётов означает одинаковые данные и одинаковый результат. '
                f'{"Расчёт № " + str(res["run_id"]) + " сохранён в истории." if res.get("run_id") else "Этот расчёт не сохранён: нажмите «Рассчитать и сохранить»."}</div>')
    with st.expander("Показать текст заключения"):
        st.markdown(md)
    with st.expander("Как считается интегральная оценка"):
        st.markdown(f"""
**S = Σ wᵢ·Xᵢ / Σ wᵢ** по доступным модулям. Веса: TA {integral.WEIGHTS['TA']}, FA {integral.WEIGHTS['FA']}, ON {integral.WEIGHTS['ON']}, SA {integral.WEIGHTS['SA']}.

**Confidence Score C = 100 · k · Σ wᵢ·qᵢ**, где qᵢ = полнота × актуальность × надёжность, k — согласованность периодов (1 или 0,9).

- модуль с q ниже {integral.Q_MIN} исключается из расчёта S;
- при C ниже {integral.C_MIN} или менее чем {integral.MIN_MODULES} доступных модулях вывод не формируется;
- при C ниже {integral.C_FULL} вывод помечается как предварительный;
- противоречие: одна оценка не ниже +{integral.CONFLICT_LEVEL}, другая не выше −{integral.CONFLICT_LEVEL}; при разрыве от {integral.STRONG_GAP} пунктов оно сильное, и формулировка «преимущественно» смягчается до «умеренно».

| Значение S | Интерпретация |
| --- | --- |
| от +30 до +100 | {integral.SCALE[0][1]} |
| от +10 до +30 | {integral.SCALE[1][1]} |
| от −10 до +10 | {integral.SCALE[2][1]} |
| от −30 до −10 | {integral.SCALE[3][1]} |
| от −100 до −30 | {integral.SCALE[4][1]} |
""")


def page_history() -> None:
    appbar("История и данные", "Сохранённые расчёты, состояние источников, исходные ответы")
    test_banner()
    tab_a, tab_b, tab_c = st.tabs(["История расчётов", "Источники", "Исходные ответы источников"])
    with tab_a:
        runs = db.list_runs()
        if runs.empty:
            banner("Сохранённых расчётов пока нет. Нажмите «Рассчитать и сохранить» в боковой панели.", "info")
        else:
            show = runs.rename(columns={"run_id": "№", "symbol": "Актив", "period_start": "Начало", "period_end": "Конец",
                                        "created_at": "Создан (UTC)", "algo_version": "Версия", "s_score": "S",
                                        "confidence": "Confidence", "verdict": "Вывод", "input_hash": "Контрольная сумма"})
            frame(show, height=300)
            groups = runs.groupby(["symbol", "period_start", "period_end", "input_hash"]).agg(
                n=("run_id", "size"), s_min=("s_score", "min"), s_max=("s_score", "max")).reset_index()
            repeated = groups[groups["n"] > 1]
            if len(repeated):
                same = bool(((repeated["s_min"] == repeated["s_max"]) | (repeated["s_min"].isna() & repeated["s_max"].isna())).all())
                banner(f"Повторных расчётов на одних и тех же данных: {int(repeated['n'].sum())} в {len(repeated)} группах; "
                       f"оценки {'совпадают — результат воспроизводим' if same else 'различаются'}.",
                       "good" if same else "bad", "Воспроизводимость")
            run_id = st.selectbox("Открыть расчёт", runs["run_id"].tolist(),
                                  format_func=lambda i: f"№ {i} · " + " · ".join(str(x) for x in runs[runs.run_id == i][["symbol", "period_start", "period_end", "created_at"]].iloc[0]))
            run = db.load_run(int(run_id))
            if run:
                text_card(f"Расчёт № {run['run_id']}: {run['verdict']}", ru(run["summary"]))
                table(["Модуль", "Оценка", "Качество q", "Состояние", "Объяснение"],
                      [[esc(NAMES.get(k, k)), badge(v["score"]), fmt_num(v["quality"]), esc(v["status"]), ru(v["explanation"])]
                       for k, v in run["modules"].items()], num_cols=(1, 2))
                for c in run["conflicts"]:
                    banner(c["explanation"], "warn", f"Противоречие ({c['severity']})")
                with st.expander("Подробности расчёта (сохранённые данные модулей)"):
                    st.json({k: v["details"] for k, v in run["modules"].items()}, expanded=False)
    with tab_b:
        if st.session_state.get("refresh_report"):
            st.markdown(f"**Последнее обновление в этом сеансе:** {st.session_state.get('refresh_time')}")
            rep = pd.DataFrame([r for r in st.session_state["refresh_report"] if r["note"] != "skip"])
            rep["ok"] = rep["ok"].map({True: "успех", False: "ошибка"})
            frame(rep.rename(columns={"source": "Источник", "symbol": "Актив", "ok": "Результат", "error": "Ошибка",
                                      "note": "Примечание", "rows": "Записей", "requests": "Запросов", "backup": "Резерв"}))
        st.markdown("**Все запросы к источникам за время работы системы**")
        status = db.source_status()
        if status.empty:
            banner("Запросов к источникам ещё не было.", "info")
        else:
            frame(status.rename(columns={"source": "Источник", "last_attempt": "Последний запрос", "last_success": "Последний успех",
                                         "errors": "Ошибок", "requests": "Запросов", "last_error": "Ошибка последнего запроса"}))
    with tab_c:
        status = db.source_status()
        if status.empty:
            banner("Исходных ответов пока нет.", "info")
            return
        src = st.selectbox("Источник", ["все"] + status["source"].tolist(), key="raw_src")
        raws = db.raw_list(None if src == "все" else src, limit=100)
        frame(raws.rename(columns={"source": "Источник", "url": "Адрес запроса", "fetched_at": "Время (UTC)", "http_status": "HTTP",
                                   "ok": "Успех", "error": "Ошибка", "size": "Размер, симв."}), height=280)
        raw_id = st.selectbox("Показать ответ", raws["id"].tolist(), key="raw_id")
        payload = db.raw_payload(int(raw_id)) or "(пустой ответ)"
        st.code(payload[:6000] + ("\n… показаны первые 6000 символов" if len(payload) > 6000 else ""), language="json")


# ======================= маршрутизация =======================
if not has_data:
    appbar("AvaLens", "Комплексный анализ криптовалют")
    test_banner()
    show_refresh_problems()
    text_card("Данных пока нет",
              f"В базе нет цен по активу {esc(asset_name)}. Нажмите <b>«Обновить данные»</b> в боковой панели: система запросит "
              "Binance, CoinGecko, Avalanche Metrics API, DefiLlama, новостные ленты, Reddit и индекс Fear &amp; Greed, "
              "сохранит ответы в базу и выполнит расчёт. Загрузка занимает до минуты.", kind="primary")
    if page == "История и данные":
        page_history()
elif res is None:
    banner("Не удалось выполнить расчёт.", "bad")
else:
    {"Главная панель": page_dashboard, "Технический анализ": page_technical, "Фундаментальный анализ": page_fundamental,
     "Ончейн-анализ": page_onchain, "Анализ настроений": page_sentiment, "Комплексный анализ": page_integral,
     "История и данные": page_history}[page]()
