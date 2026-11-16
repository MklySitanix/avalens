"""Оркестратор: единственный модуль, который обращается и к источникам, и к хранилищу.

refresh() — загрузка данных из внешних источников в базу.
analyze() — расчёт по сохранённым данным за выбранный период (сеть не нужна).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from analysis import fundamental, integral, onchain, sentiment, technical
from data import quality
from data.sources import avalanche_metrics, binance, coingecko, defillama, fear_greed, news_rss, reddit
from storage.db import Database

ROOT = Path(__file__).resolve().parent.parent
ALGO_VERSION = "1.0.0"
MARKET = "MARKET"  # условный «актив» для общерыночных индексов (Fear & Greed)


def load_config() -> dict:
    with open(ROOT / "config" / "assets.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_facts(symbol: str) -> dict | None:
    path = ROOT / "fundamentals" / f"{symbol.lower()}.yaml"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ====================== загрузка ======================
def _store_fetches(db: Database, res) -> None:
    for f in res.fetches:
        db.save_raw(f.source, f.url, f.status, f.ok, f.error, f.text, f.fetched_at)


def _closed_days(df, today: str):
    """Отбрасывает значение за текущие незавершённые сутки: оно неполное и исказило бы динамику."""
    return df if df is None or len(df) == 0 else df[df["date"] < today]


def _status(res, symbol: str, rows: int = 0, backup: str | None = None) -> dict:
    return {"source": res.source, "symbol": symbol, "ok": res.ok, "error": res.error, "note": res.note,
            "rows": rows, "requests": len(res.fetches), "backup": backup}


def refresh(db: Database, cfg: dict, symbols: list[str] | None = None, progress=None) -> list[dict]:
    """Опрашивает все источники. Отказ одного источника не останавливает остальные."""
    symbols = symbols or list(cfg["assets"])
    db.upsert_assets(cfg["assets"])
    today = utc_now().strftime("%Y-%m-%d")
    report: list[dict] = []
    say = progress or (lambda msg: None)

    general_feeds = [f for f in cfg["news_feeds"] if "{query}" not in f["url"]]
    query_feeds = [f for f in cfg["news_feeds"] if "{query}" in f["url"]]
    say("Новостные ленты")
    general = news_rss.fetch({}, general_feeds, "")
    _store_fetches(db, general)
    general_posts = general.frames.get("posts", [])

    for symbol in symbols:
        asset = cfg["assets"][symbol]
        snapshot: dict = {}

        say(f"{symbol}: Binance")
        b = binance.fetch(asset)
        _store_fetches(db, b)
        n = 0
        if b.ok:
            prices = b.frames["prices"]
            prices = prices[prices["date"] < today]  # незакрытая свеча текущего дня не сохраняется
            n = db.upsert_prices(symbol, "binance", prices)
        report.append(_status(b, symbol, n))

        say(f"{symbol}: CoinGecko")
        c = coingecko.fetch(asset, with_chart=not b.ok)
        _store_fetches(db, c)
        n = 0
        if "prices" in c.frames:  # резервный ценовой ряд
            cp = c.frames["prices"]
            n = db.upsert_prices(symbol, "coingecko", cp[cp["date"] < today])
        ticker = b.frames.get("ticker") if b.ok else None
        if ticker:
            snapshot.update({"binance_price": ticker["price"], "change_pct_24h": ticker["change_pct_24h"]})
        if "fundamentals" in c.frames:
            snapshot.update(c.frames["fundamentals"])
            if ticker:  # два независимых значения цены, полученные в одну минуту
                div = quality.price_divergence(ticker["price"], snapshot.get("price"))
                snapshot.update({"price_diff_pct": div["diff_pct"], "price_diverged": div["diverged"]})
        report.append(_status(c, symbol, n, backup=None if b.ok else "резерв цены вместо Binance"))

        if asset.get("chain_id"):
            say(f"{symbol}: Avalanche Metrics API")
            m = avalanche_metrics.fetch(asset)
            _store_fetches(db, m)
            report.append(_status(m, symbol, db.upsert_metrics(symbol, _closed_days(m.frames.get("metrics"), today))))

        say(f"{symbol}: DefiLlama")
        d = defillama.fetch(asset)
        _store_fetches(db, d)
        report.append(_status(d, symbol, db.upsert_metrics(symbol, _closed_days(d.frames.get("metrics"), today))))
        if "protocols_count" in d.frames:
            snapshot["protocols_count"] = d.frames["protocols_count"]

        if snapshot:
            src = "+".join(s for s, ok in (("binance", "binance_price" in snapshot),
                                           ("coingecko", "market_cap" in snapshot),
                                           ("defillama", "protocols_count" in snapshot)) if ok)
            db.insert_fundamentals(symbol, snapshot, src)

        say(f"{symbol}: новости и Reddit")
        q = news_rss.fetch(asset, query_feeds, symbol)
        _store_fetches(db, q)
        r = reddit.fetch(asset)
        _store_fetches(db, r)
        raw_posts = general_posts + q.frames.get("posts", []) + r.frames.get("posts", [])
        posts, stats = sentiment.prepare_posts(raw_posts, asset, symbol)
        new = db.upsert_posts(symbol, posts)
        news_res = news_rss.SourceResult(
            "news", ok=general.ok or q.ok, error=None if (general.ok or q.ok) else (q.error or general.error),
            fetches=general.fetches + q.fetches,
            note="; ".join(x for x in (general.note, q.note) if x) or None)
        st = _status(news_res, symbol, new)
        st["note"] = "; ".join(x for x in (st["note"], f"получено {stats['received']}, по активу {stats['kept']}, "
                                           f"не по теме {stats['irrelevant']}, дублей {stats['duplicates']}, новых {new}") if x)
        report.append(st)
        report.append(_status(r, symbol, len(r.frames.get("posts", []))))

    say("Fear & Greed Index")
    fg = fear_greed.fetch()
    _store_fetches(db, fg)
    report.append(_status(fg, MARKET, db.upsert_metrics(MARKET, fg.frames.get("metrics"))))
    return report


# ====================== расчёт ======================
def _hours(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    return max((later - earlier).total_seconds() / 3600.0, 0.0)


def _metric_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {name: g.set_index("date")["value"] for name, g in df.groupby("metric")}


def _digest(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode())
    return h.hexdigest()[:16]


def _peer_symbol(cfg: dict, symbol: str) -> str | None:
    return "ETH" if symbol != "ETH" and "ETH" in cfg["assets"] else ("AVAX" if symbol == "ETH" and "AVAX" in cfg["assets"] else None)


def analyze(db: Database, cfg: dict, symbol: str, period_start: str, period_end: str,
            params: dict | None = None, save: bool = True) -> dict:
    """Полный расчёт по данным базы на конец периода. Одинаковые данные и параметры дают одинаковый результат.

    Точка отсчёта актуальности — конец периода (следующая полночь UTC), а не текущее время,
    поэтому повторный расчёт за тот же период воспроизводим.
    """
    params = {"sentiment_window_days": sentiment.WINDOW_DAYS, **(params or {})}
    asset = cfg["assets"][symbol]
    rel, fresh = cfg["reliability"], cfg["freshness_hours"]
    end_ts = pd.Timestamp(period_end)
    start_ts = pd.Timestamp(period_start)
    ref = end_ts + pd.Timedelta(days=1)
    ref_iso = ref.strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []
    modules: dict[str, dict] = {}

    # ---------- цены и технический анализ ----------
    raw_prices = db.load_prices(symbol, end=period_end)
    price_source = raw_prices["source"].iloc[0] if len(raw_prices) else None
    prices, prep = quality.clean_prices(raw_prices)
    if price_source:
        db.mark_price_flags(symbol, price_source, prep["anomalies"])
    ta = technical.analyze(prices, period_start)
    if prep["anomalies"]:
        warnings.append(f"Цены: исключены аномальные значения за {', '.join(prep['anomalies'][:5])}"
                        + (" и другие даты" if len(prep["anomalies"]) > 5 else "") + ".")
    if prep["filled_days"]:
        warnings.append(f"Цены: {prep['filled_days']} пропущенных дней заполнены ценой предыдущего дня.")
    if prep["cut_before"] and prep["cut_before"] != "all":
        warnings.append(f"Цены: разрыв длиннее {quality.MAX_FILL_DAYS} дней — используются данные с {prep['cut_before']}.")
    if price_source == "coingecko":
        warnings.append("Цены получены из резервного источника CoinGecko: Binance был недоступен.")
    w_start = min(start_ts, end_ts - pd.Timedelta(days=89))
    n_expected = (end_ts - w_start).days + 1
    in_window = prices.loc[w_start:end_ts] if len(prices) else prices
    ta_c = round(float((in_window["is_filled"] == 0).sum()) / n_expected, 4) if len(prices) else 0.0
    ta_age = _hours(ref, pd.Timestamp(prep["last_date"]) + pd.Timedelta(days=1)) if prep["last_date"] else None
    ta_f = quality.freshness(ta_age, fresh["market"])
    snapshot = db.load_fundamentals(symbol, as_of=(ref + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    snap_age = _hours(ref, pd.Timestamp(snapshot["snapshot_at"]).tz_localize(None)) if snapshot else None
    if snapshot and snap_age > 3 * fresh["fundamentals"]:
        snapshot = None  # снимок слишком старый для этой даты
    ta_r, r_note = rel.get(price_source or "", 0.5), None
    if snapshot and snapshot.get("price_diverged") and snap_age <= 24:
        ta_r, r_note = 0.7, f"цены Binance и CoinGecko расходятся на {snapshot.get('price_diff_pct')}%"
        warnings.append(f"Противоречивые данные: {r_note}.")
    if ta_age is not None and ta_age > fresh["market"]:
        warnings.append(f"Рыночные данные устарели: последняя свеча за {prep['last_date']}, "
                        f"отставание от конца периода {ta_age / 24:.0f} дн.")
    modules["TA"] = {**ta, "quality": round(ta_c * ta_f * ta_r, 4), "quality_parts": {
        "completeness": ta_c, "freshness": ta_f, "reliability": ta_r, "source": price_source,
        "last_date": prep["last_date"],
        "notes": {"completeness": f"{prep['missing_days']} пропущенных дней, {len(prep['anomalies'])} аномалий",
                  "freshness": f"последняя свеча {prep['last_date']}", "reliability": r_note}}}

    # ---------- ончейн ----------
    mdf = db.load_metrics(symbol, end=period_end)
    raw_series = _metric_series(mdf)
    sources = mdf.drop_duplicates("metric").set_index("metric")["source"].to_dict() if len(mdf) else {}
    clean, mreports = {}, {}
    for name, s in raw_series.items():
        clean[name], mreports[name] = quality.clean_metric(s)
        if mreports[name]["anomalies"]:
            warnings.append(f"Ончейн: в ряду {name} исключены аномалии за {', '.join(mreports[name]['anomalies'][:3])}.")
    expected = asset.get("onchain_metrics", [])
    # показатель, который источник перестал обновлять, в оценку не входит
    stale_notes = {}
    for name in list(clean):
        last = mreports[name]["last_date"]
        if name in expected and last and _hours(ref, pd.Timestamp(last) + pd.Timedelta(days=1)) > 3 * fresh["onchain"]:
            stale_notes[name] = f"данные устарели: последнее значение за {last}"
            warnings.append(f"Ончейн: показатель {name} не обновляется источником с {last} — исключён из оценки.")
    fresh_metrics = {k: v for k, v in clean.items() if k not in stale_notes}
    close = prices["close"] if len(prices) else pd.Series(dtype=float)
    on = onchain.analyze(fresh_metrics, close, expected, period_start, stale_notes) if len(close) else {
        "status": "unavailable", "score": None, "details": {"metrics": [], "n_available": 0, "n_expected": len(expected)},
        "risks": [], "series": {}, "explanation": "Нет ценового ряда для сопоставления."}
    avail = [r["metric"] for r in on["details"]["metrics"] if r.get("available")]
    cover, fr, rr, last_dates = [], [], [], []
    for name in avail:
        s = clean[name]
        tail = s.loc[end_ts - pd.Timedelta(days=36):end_ts]
        cover.append(float(tail.notna().sum()) / 37)
        last = pd.Timestamp(mreports[name]["last_date"])
        last_dates.append(last)
        fr.append(quality.freshness(_hours(ref, last + pd.Timedelta(days=1)), fresh["onchain"]))
        rr.append(rel.get(sources.get(name, ""), 0.5))
    on_c = round(len(avail) / max(len(expected), 1) * (sum(cover) / len(cover)), 4) if avail else 0.0
    on_f = round(sum(fr) / len(fr), 4) if fr else 0.0
    on_r = round(sum(rr) / len(rr), 4) if rr else 0.0
    missing_metrics = [m for m in expected if m not in avail and m not in stale_notes]
    if missing_metrics:
        warnings.append("Ончейн: нет данных по показателям " + ", ".join(missing_metrics) + ".")
    if fr and on_f < 1:
        warnings.append(f"Ончейн-данные устарели: последние значения за {min(last_dates).strftime('%Y-%m-%d')}.")
    modules["ON"] = {**on, "quality": round(on_c * on_f * on_r, 4), "quality_parts": {
        "completeness": on_c, "freshness": on_f, "reliability": on_r,
        "last_date": max(last_dates).strftime("%Y-%m-%d") if last_dates else None,
        "notes": {"completeness": f"получено {len(avail)} показателей из {len(expected)}",
                  "freshness": f"последние значения {min(last_dates).strftime('%Y-%m-%d')}" if last_dates else "нет данных",
                  "reliability": None}}}

    # ---------- фундаментальный ----------
    facts = load_facts(symbol)
    peer_symbol = _peer_symbol(cfg, symbol)
    peer = None
    if peer_symbol:
        psnap = db.load_fundamentals(peer_symbol, as_of=(ref + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        ptvl = _metric_series(db.load_metrics(peer_symbol, end=period_end)).get("tvl")
        if psnap and ptvl is not None and len(ptvl):
            peer = {"symbol": peer_symbol, "market_cap": psnap.get("market_cap"), "tvl": float(ptvl.iloc[-1])}
    fa = fundamental.analyze(snapshot, facts, clean, peer, as_of=period_end)
    warnings.extend(fa["details"].get("warnings", []))
    fa_c = round(fa["details"]["n_scored"] / 8, 4)
    fa_f = quality.freshness(snap_age, fresh["fundamentals"]) if snapshot else 0.5
    if not snapshot:
        warnings.append("Фундаментальный анализ: рыночного снимка (капитализация, FDV) на эту дату в базе нет — "
                        "оценены только направления, не зависящие от него.")
    modules["FA"] = {**fa, "quality": round(fa_c * fa_f * rel["coingecko"], 4), "quality_parts": {
        "completeness": fa_c, "freshness": fa_f, "reliability": rel["coingecko"],
        "last_date": snapshot["snapshot_at"] if snapshot else None,
        "notes": {"completeness": f"оценено {fa['details']['n_scored']} направлений из 8",
                  "freshness": f"снимок от {snapshot['snapshot_at']}" if snapshot else "снимка нет", "reliability": None}}}

    # ---------- настроения ----------
    posts = db.load_posts(symbol, end=ref_iso)
    fg = _metric_series(db.load_metrics(MARKET, end=period_end)).get("fear_greed")
    sa = sentiment.analyze(posts, fg, period_end, period_start, params["sentiment_window_days"])
    sd = sa["details"]
    sa_c = round(min(1.0, sd.get("n_posts", 0) / sentiment.TARGET_POSTS), 4)
    last_post = pd.Timestamp(sd["last_post"]).tz_localize(None) if sd.get("last_post") else None
    sa_f = quality.freshness(_hours(ref, last_post), fresh["posts"]) if last_post is not None else 0.0
    by_src = sd.get("by_source") or {}
    total_posts = sum(by_src.values())
    sa_r = round(sum(n * rel.get("reddit" if s == "reddit" else "news", 0.5) for s, n in by_src.items()) / total_posts, 4) if total_posts else 0.0
    if sd.get("fear_greed") is None:
        warnings.append("Индекс Fear & Greed на конец периода недоступен — настроения оценены только по публикациям.")
    if last_post is not None and sa_f < 1:
        warnings.append(f"Публикации устарели: последняя от {sd['last_post'][:10]}.")
    modules["SA"] = {**sa, "quality": round(sa_c * sa_f * sa_r, 4), "quality_parts": {
        "completeness": sa_c, "freshness": sa_f, "reliability": sa_r, "last_date": sd.get("last_post"),
        "notes": {"completeness": f"{sd.get('n_posts', 0)} публикаций при целевых {sentiment.TARGET_POSTS}",
                  "freshness": f"последняя публикация {sd['last_post'][:10]}" if sd.get("last_post") else "публикаций нет",
                  "reliability": "доля Reddit снижает надёжность" if by_src.get("reddit") else None}}}

    # ---------- согласованность периодов и итог ----------
    k_sync, sync_note = 1.0, None
    ta_last = pd.Timestamp(prep["last_date"]) if prep["last_date"] else None
    on_last = max(last_dates) if last_dates else None
    if ta_last is not None and on_last is not None and abs((ta_last - on_last).days) > 1:
        k_sync, sync_note = 0.9, (f"Периоды не согласованы: цены до {ta_last.strftime('%Y-%m-%d')}, "
                                  f"ончейн до {on_last.strftime('%Y-%m-%d')}")
    elif last_post is not None and (ref - last_post).days > 3:
        k_sync, sync_note = 0.9, f"Периоды не согласованы: последняя публикация {sd['last_post'][:10]}"
    total = integral.combine(modules, k_sync, sync_note)

    risks = []
    for name in ("TA", "FA", "ON", "SA"):
        risks.extend({"module": name, "text": r} for r in modules[name].get("risks", []))

    comparison = _compare(db, cfg, symbol, peer_symbol, period_end, modules) if peer_symbol else None

    input_hash = _digest(
        ALGO_VERSION, params, symbol, period_start, period_end,
        prices.round(8).to_csv() if len(prices) else "",
        {k: v.round(6).to_csv() for k, v in sorted(clean.items())},
        {k: snapshot.get(k) for k in ("snapshot_at", "market_cap", "fdv", "circulating", "protocols_count")} if snapshot else None,
        facts, sorted(posts["text_hash"].tolist()) if len(posts) else [],
        sorted((int(i), m) for i, m in zip(posts["id"], posts["manual_label"]) if m) if len(posts) else [],
        None if fg is None else fg.tail(5).to_csv())

    result = {
        "symbol": symbol, "name": asset["name"], "period_start": period_start, "period_end": period_end,
        "algo_version": ALGO_VERSION, "params": params, "input_hash": input_hash,
        "modules": {k: {kk: vv for kk, vv in m.items() if kk not in ("indicators", "series", "posts", "daily", "levels")}
                    for k, m in modules.items()},
        "integral": total, "risks": risks, "warnings": warnings, "comparison": comparison,
        "price_report": prep, "metric_reports": mreports,
    }
    if save:
        result["run_id"] = db.save_run(result)
    result["frames"] = {
        "prices": prices, "raw_prices": raw_prices, "indicators": modules["TA"].get("indicators"),
        "levels": modules["TA"].get("levels", []), "onchain": modules["ON"].get("series", {}),
        "onchain_raw": mdf, "posts": modules["SA"].get("posts"), "sentiment_daily": modules["SA"].get("daily"),
        "fear_greed": fg,
    }
    return result


def _compare(db: Database, cfg: dict, symbol: str, peer: str, period_end: str, modules: dict) -> dict:
    """Сравнение с Ethereum: доходность, волатильность, оценка относительно TVL, комиссии."""
    out = {"peer": peer, "rows": []}
    pp, _ = quality.clean_prices(db.load_prices(peer, end=period_end))
    mine, _ = quality.clean_prices(db.load_prices(symbol, end=period_end))

    def stats(p: pd.DataFrame) -> dict:
        if len(p) < 31:
            return {}
        c = p["close"]
        r = lambda n: float(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else None
        return {"ret7": r(7), "ret30": r(30), "ret90": r(90),
                "vol30": float(c.pct_change().tail(30).std() * (365 ** 0.5))}

    a, b = stats(mine), stats(pp)
    for key, title, kind in (("ret7", "Доходность за 7 дней", "pct"), ("ret30", "Доходность за 30 дней", "pct"),
                             ("ret90", "Доходность за 90 дней", "pct"), ("vol30", "Волатильность (30 дней, годовая)", "pct")):
        out["rows"].append({"title": title, "kind": kind, symbol: a.get(key), peer: b.get(key)})
    ref = (pd.Timestamp(period_end) + pd.Timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for sym in (symbol, peer):
        snap = db.load_fundamentals(sym, as_of=ref) or {}
        ms = _metric_series(db.load_metrics(sym, end=period_end))
        tvl = float(ms["tvl"].iloc[-1]) if "tvl" in ms and len(ms["tvl"]) else None
        fees = float(ms["fees_usd"].tail(30).sum()) if "fees_usd" in ms and len(ms["fees_usd"]) >= 25 else None
        mc = snap.get("market_cap")
        for title, kind, val in (("Капитализация, долл.", "usd", mc), ("TVL, долл.", "usd", tvl),
                                 ("Капитализация / TVL", "num", mc / tvl if mc and tvl else None),
                                 ("Комиссии за 30 дней, долл.", "usd", fees),
                                 ("FDV / капитализация", "num", snap["fdv"] / mc if mc and snap.get("fdv") else None)):
            row = next((r for r in out["rows"] if r["title"] == title), None)
            if row is None:
                row = {"title": title, "kind": kind}
                out["rows"].append(row)
            row[sym] = val
    if len(mine) > 30 and len(pp) > 30:
        j = pd.concat([mine["close"].pct_change().rename("a"), pp["close"].pct_change().rename("b")], axis=1, sort=True).dropna().tail(90)
        out["correlation_90d"] = round(float(j["a"].corr(j["b"])), 3) if len(j) >= 30 else None
    return out


def current_quote(db: Database, symbol: str) -> dict | None:
    """Текущая цена из последнего сохранённого снимка (тикер Binance, иначе CoinGecko)."""
    snap = db.load_fundamentals(symbol)
    if not snap:
        return None
    price = snap.get("binance_price") or snap.get("price")
    if price is None:
        return None
    return {"price": price, "change_pct_24h": snap.get("change_pct_24h"), "time": snap["snapshot_at"],
            "source": "Binance" if snap.get("binance_price") else "CoinGecko"}
