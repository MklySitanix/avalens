"""Модуль ончейн-анализа: динамика показателей сети и их связь с ценой.

Для каждого показателя: изменение среднего за 7 дней к среднему за предыдущие 30 дней,
ограниченное ±25% и приведённое к [-1; +1]. ON = 100 * среднее по показателям.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLIP = 0.25
MIN_METRICS = 3
MIN_DAYS = 37

TITLES = {
    "activeAddresses": ("Активные адреса", "адресов в сутки", "число уникальных адресов, совершавших транзакции: мера пользовательской базы"),
    "txCount": ("Транзакции", "транзакций в сутки", "число транзакций: мера использования сети"),
    "feesPaid": ("Комиссии сети", "AVAX в сутки", "сумма уплаченных комиссий: платёжеспособный спрос на блочное пространство"),
    "tvl": ("TVL", "в токенах сети", "средства, заблокированные в протоколах сети, в пересчёте на токен — чтобы рост цены не выглядел ростом активности"),
    "validatorCount": ("Валидаторы", "узлов", "число валидаторов Primary Network: децентрализация и безопасность"),
    "delegatorCount": ("Делегаторы", "адресов", "число делегаторов: вовлечённость держателей в стейкинг"),
    "fees_usd": ("Комиссии сети", "долл. в сутки", "сумма уплаченных комиссий: платёжеспособный спрос на блочное пространство"),
    "dex_volume_usd": ("Объём DEX", "долл. в сутки", "объём торгов на децентрализованных биржах сети"),
}


def metric_signal(s: pd.Series) -> dict | None:
    s = s.dropna()
    if len(s) < MIN_DAYS:
        return None
    last7, prev30 = float(s.tail(7).mean()), float(s.iloc[-37:-7].mean())
    if prev30 <= 0:
        return None
    delta = last7 / prev30 - 1
    prev7 = float(s.iloc[-14:-7].mean())
    return {"last": float(s.iloc[-1]), "mean7": last7, "mean30_prev": prev30, "delta": round(delta, 4),
            "change_7d": round(last7 / prev7 - 1, 4) if prev7 > 0 else None,
            "signal": round(float(np.clip(delta / CLIP, -1, 1)), 3)}


def correlation(metric: pd.Series, price: pd.Series, period_start: str | None) -> dict:
    """Связь с ценой за период: корреляция дневных изменений и корреляция уровней."""
    df = pd.concat([metric.rename("m"), price.rename("p")], axis=1, sort=True).dropna()
    if period_start:
        df = df.loc[pd.Timestamp(period_start):]
    if len(df) < 20:
        return {"returns": None, "levels": None, "n": len(df)}
    ch = df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    r1 = float(ch["m"].corr(ch["p"])) if len(ch) >= 20 and ch["m"].std() > 0 else None
    r2 = float(df["m"].corr(df["p"])) if df["m"].std() > 0 else None
    return {"returns": None if r1 is None or r1 != r1 else round(r1, 3),
            "levels": None if r2 is None or r2 != r2 else round(r2, 3), "n": len(df)}


def analyze(metrics: dict[str, pd.Series], price: pd.Series, expected: list[str],
            period_start: str | None = None, notes: dict[str, str] | None = None) -> dict:
    """metrics — очищенные суточные ряды; price — цена закрытия; expected — показатели из реестра активов."""
    rows, series = [], {}
    price_ret30 = float(price.iloc[-1] / price.iloc[-31] - 1) if len(price) > 31 else None
    for name in expected:
        s = metrics.get(name)
        title, unit, meaning = TITLES.get(name, (name, "", ""))
        if s is None or s.dropna().empty:
            rows.append({"metric": name, "title": title, "unit": unit, "meaning": meaning, "available": False,
                         **({"note": notes[name]} if notes and name in notes else {})})
            continue
        if name == "tvl":  # TVL в токенах сети: убираем влияние цены
            aligned = price.reindex(s.index).ffill()
            s = (s / aligned).dropna()
        sig = metric_signal(s)
        series[name] = s
        if sig is None:
            rows.append({"metric": name, "title": title, "unit": unit, "meaning": meaning, "available": False,
                         "note": f"ряд короче {MIN_DAYS} дней"})
            continue
        rows.append({"metric": name, "title": title, "unit": unit, "meaning": meaning, "available": True,
                     **sig, "corr": correlation(s, price, period_start)})
    ok = [r for r in rows if r["available"]]
    details = {"metrics": rows, "n_available": len(ok), "n_expected": len(expected),
               "price_return_30d": None if price_ret30 is None else round(price_ret30, 4)}
    if len(ok) < MIN_METRICS:
        return {"status": "unavailable", "score": None, "details": details, "risks": [], "series": series,
                "explanation": f"Получено {len(ok)} показателей из {len(expected)}, для вывода нужно не менее "
                               f"{MIN_METRICS}. Расчёт ончейн-оценки не выполняется."}
    score = round(100 * float(np.mean([r["signal"] for r in ok])), 2)
    growing = [r["title"] for r in ok if r["delta"] > 0.05]
    falling = [r["title"] for r in ok if r["delta"] < -0.05]
    risks = [f"{r['title']}: снижение на {abs(r['delta']):.0%} к среднему за 30 дней" for r in ok if r["delta"] <= -0.15]

    conclusions = []
    if growing:
        conclusions.append("Растут: " + ", ".join(growing) + ".")
    if falling:
        conclusions.append("Снижаются: " + ", ".join(falling) + ".")
    if not growing and not falling:
        conclusions.append("Показатели сети стабильны: изменения в пределах ±5%.")
    if price_ret30 is not None:
        activity = float(np.mean([r["delta"] for r in ok]))
        if price_ret30 > 0.05 and activity < -0.05:
            conclusions.append(f"Цена за 30 дней выросла на {price_ret30:.0%}, а активность сети снизилась — "
                               f"рост цены не подкреплён использованием сети.")
        elif price_ret30 < -0.05 and activity > 0.05:
            conclusions.append(f"Цена за 30 дней упала на {abs(price_ret30):.0%}, а активность сети выросла — "
                               f"сетью пользуются больше, чем это отражено в цене.")
        elif abs(price_ret30) > 0.05 and np.sign(price_ret30) == np.sign(activity) and abs(activity) > 0.05:
            conclusions.append(f"Цена ({price_ret30:+.0%} за 30 дней) и активность сети движутся в одну сторону.")
    strong = [r for r in ok if r["corr"]["returns"] is not None and abs(r["corr"]["returns"]) >= 0.3]
    if strong:
        conclusions.append("Заметная связь дневных изменений с ценой: " + ", ".join(
            f"{r['title']} (r = {r['corr']['returns']:+.2f})" for r in strong) + ".")
    details["conclusions"] = conclusions
    explanation = (f"ON = 100 × среднее({', '.join(f'{r['signal']:+.2f}' for r in ok)}) = {score:+.0f}. "
                   + " ".join(conclusions))
    return {"status": "ok", "score": score, "details": details, "risks": risks, "series": series,
            "explanation": explanation}
