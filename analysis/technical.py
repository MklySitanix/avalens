"""Модуль технического анализа. Все индикаторы реализованы здесь, без сторонних библиотек.

Индикатор в день t использует только цены до дня t включительно — будущие данные не участвуют.
Оценка TA = 100 * взвешенная сумма пяти сигналов в диапазоне [-1; +1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {"trend": 0.30, "macd": 0.25, "rsi": 0.20, "volume": 0.15, "bollinger": 0.10}
MIN_DAYS = 60


# ---------- индикаторы ----------
def sma(s: pd.Series, n: int) -> pd.Series:
    """Простая скользящая средняя: среднее арифметическое последних n цен."""
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    """Экспоненциальная средняя: EMA_t = a*P_t + (1-a)*EMA_{t-1}, a = 2/(n+1).

    Первое значение — SMA первых n цен. В отличие от SMA свежие цены весят больше.
    """
    values = s.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < n:
        return pd.Series(out, index=s.index)
    alpha = 2.0 / (n + 1)
    out[n - 1] = values[:n].mean()
    for i in range(n, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=s.index)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Индекс относительной силы по Уайлдеру.

    Средний рост и среднее падение за первые n изменений — простые средние, далее
    avg_t = (avg_{t-1}*(n-1) + x_t)/n. RSI = 100 - 100/(1 + avg_gain/avg_loss).
    """
    values = s.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) <= n:
        return pd.Series(out, index=s.index)
    delta = np.diff(values)
    gain, loss = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    avg_gain, avg_loss = gain[:n].mean(), loss[:n].mean()

    def value(g: float, l: float) -> float:
        if l == 0:
            return 100.0 if g > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + g / l)

    out[n] = value(avg_gain, avg_loss)
    for i in range(n, len(delta)):
        avg_gain = (avg_gain * (n - 1) + gain[i]) / n
        avg_loss = (avg_loss * (n - 1) + loss[i]) / n
        out[i + 1] = value(avg_gain, avg_loss)
    return pd.Series(out, index=s.index)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD = EMA(12) - EMA(26); сигнальная линия = EMA(9) от MACD; гистограмма = их разность."""
    line = ema(s, fast) - ema(s, slow)
    valid = line.dropna()
    sig = ema(valid, signal).reindex(s.index)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Полосы Боллинджера: SMA(n) ± k стандартных отклонений (по генеральной совокупности)."""
    mid = sma(s, n)
    std = s.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * std, mid - k * std
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_width": (upper - lower) / mid, "bb_pct": (s - lower) / (upper - lower)})


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    out = prices[["open", "high", "low", "close", "volume"]].copy()
    out["sma20"], out["sma50"] = sma(close, 20), sma(close, 50)
    out["ema12"], out["ema26"] = ema(close, 12), ema(close, 26)
    out["rsi14"] = rsi(close, 14)
    out = out.join(macd(close)).join(bollinger(close))
    vol = prices["volume"].where(prices.get("is_filled", 0) == 0)
    out["vol_sma20"] = vol.rolling(20, min_periods=10).mean()
    out["vol_sma5"] = vol.rolling(5, min_periods=3).mean()
    return out


# ---------- тренд и уровни ----------
def trend_slope(close: pd.Series, n: int = 30) -> float:
    """Наклон линейной регрессии логарифма цены за n дней, % в день."""
    tail = close.dropna().tail(n)
    if len(tail) < n:
        return float("nan")
    return float(np.polyfit(np.arange(len(tail)), np.log(tail.to_numpy()), 1)[0] * 100)


def support_resistance(prices: pd.DataFrame, lookback: int = 180, window: int = 5, tol: float = 0.03) -> dict:
    """Уровни по локальным экстремумам: минимум/максимум среди window дней с каждой стороны.

    Близкие экстремумы (в пределах tol) объединяются; число касаний — сила уровня.
    """
    d = prices.tail(lookback)
    lows, highs = d["low"].to_numpy(), d["high"].to_numpy()
    pivots: list[tuple[float, str]] = []
    for i in range(window, len(d) - window):
        if lows[i] == lows[i - window:i + window + 1].min():
            pivots.append((float(lows[i]), d.index[i].strftime("%Y-%m-%d")))
        if highs[i] == highs[i - window:i + window + 1].max():
            pivots.append((float(highs[i]), d.index[i].strftime("%Y-%m-%d")))
    pivots.sort()
    clusters: list[dict] = []
    for level, day in pivots:
        if clusters and level <= clusters[-1]["level"] * (1 + tol):
            c = clusters[-1]
            c["level"] = (c["level"] * c["touches"] + level) / (c["touches"] + 1)
            c["touches"] += 1
            c["last"] = max(c["last"], day)
        else:
            clusters.append({"level": level, "touches": 1, "last": day})
    price = float(d["close"].iloc[-1])
    below = [c for c in clusters if c["level"] < price]
    above = [c for c in clusters if c["level"] > price]
    rnd = lambda c: {"level": round(c["level"], 4), "touches": c["touches"], "last": c["last"]} if c else None
    return {"support": rnd(below[-1] if below else None), "resistance": rnd(above[0] if above else None),
            "levels": [rnd(c) for c in clusters]}


# ---------- оценка ----------
def _sign(x: float, eps: float = 0.0) -> int:
    return 0 if (x != x or abs(x) <= eps) else (1 if x > 0 else -1)


def analyze(prices: pd.DataFrame, period_start: str | None = None) -> dict:
    """prices — очищенные свечи до конца периода (индекс — даты). Оценка даётся на последний день."""
    usable = int((prices.get("is_filled", pd.Series(0, index=prices.index)) == 0).sum()) if len(prices) else 0
    if usable < MIN_DAYS:
        return {"status": "unavailable", "score": None, "signals": [], "details": {"usable_days": usable},
                "explanation": f"Недостаточно данных: {usable} дневных свечей, для индикаторов нужно не менее {MIN_DAYS}."}
    ind = compute_indicators(prices)
    last = ind.iloc[-1]
    close = float(last["close"])
    slope = trend_slope(ind["close"])

    # 1. тренд: положение цены и средних + наклон
    parts = [_sign(close - last["sma50"]), _sign(last["sma20"] - last["sma50"]), _sign(slope, 0.1)]
    s_trend = float(np.mean(parts))
    if parts == [1, 1, 1]:
        trend, trend_ru = "up", "восходящий тренд"
    elif parts == [-1, -1, -1]:
        trend, trend_ru = "down", "нисходящий тренд"
    elif abs(slope) <= 0.1:
        trend, trend_ru = "sideways", "боковое движение"
    else:
        trend, trend_ru = "mixed", "переходное состояние: признаки тренда расходятся"
    t_text = (f"Цена {close:.2f} {'выше' if parts[0] > 0 else 'ниже'} SMA50 ({last['sma50']:.2f}), "
              f"SMA20 {'выше' if parts[1] > 0 else 'ниже'} SMA50, наклон за 30 дней {slope:+.2f}% в день — {trend_ru}.")

    # 2. MACD: знак линии и её положение относительно сигнальной
    s_macd = 0.5 * _sign(last["macd"] - last["signal"]) + 0.5 * _sign(last["macd"])
    prev = ind.iloc[-2]
    cross = _sign(last["hist"]) - _sign(prev["hist"])
    m_text = (f"MACD {last['macd']:+.3f}, сигнальная {last['signal']:+.3f}: импульс "
              f"{'растёт' if last['hist'] > 0 else 'слабеет'}"
              + (", сегодня линия MACD пересекла сигнальную снизу вверх." if cross > 0 else
                 ", сегодня линия MACD пересекла сигнальную сверху вниз." if cross < 0 else "."))

    # 3. RSI: импульс в средней зоне, в крайних зонах сигнал обнуляется
    r = float(last["rsi14"])
    if r >= 70:
        s_rsi, zone = 0.0, "перекупленность"
        r_text = f"RSI {r:.1f} — зона перекупленности: импульс сильный, но высок риск отката, сигнал нейтральный."
    elif r <= 30:
        s_rsi, zone = 0.0, "перепроданность"
        r_text = f"RSI {r:.1f} — зона перепроданности: давление продавцов сильное, но возможен отскок, сигнал нейтральный."
    else:
        s_rsi, zone = float(np.clip((r - 50) / 20, -1, 1)), "нейтральная зона"
        r_text = f"RSI {r:.1f} — {'покупатели сильнее' if r > 55 else 'продавцы сильнее' if r < 45 else 'силы равны'}."

    # 4. объём: подтверждает ли он движение цены за 5 дней
    ret5 = close / float(ind["close"].iloc[-6]) - 1
    direction = _sign(ret5, 0.01)
    ratio = float(last["vol_sma5"] / last["vol_sma20"]) if last["vol_sma20"] else float("nan")
    if ratio != ratio or direction == 0:
        s_vol = 0.0
        v_text = f"Цена за 5 дней изменилась на {ret5:+.1%} — движения, которое нужно подтверждать объёмом, нет."
    elif ratio >= 1.1:
        s_vol = float(direction)
        v_text = (f"Цена за 5 дней {ret5:+.1%}, объём в {ratio:.2f} раза выше среднего за 20 дней — "
                  f"объём подтверждает {'рост' if direction > 0 else 'падение'}.")
    elif ratio <= 0.9:
        s_vol = -0.5 * direction
        v_text = (f"Цена за 5 дней {ret5:+.1%}, но объём составляет {ratio:.2f} от среднего — "
                  f"{'рост' if direction > 0 else 'падение'} объёмом не подтверждено.")
    else:
        s_vol = 0.0
        v_text = f"Цена за 5 дней {ret5:+.1%}, объём близок к среднему ({ratio:.2f}) — подтверждения нет."

    # 5. полосы Боллинджера: положение цены в канале
    pct_b = float(last["bb_pct"])
    s_bb = float(np.clip(2 * (pct_b - 0.5), -1, 1))
    width_now = float(last["bb_width"])
    width_med = float(ind["bb_width"].tail(90).median())
    b_text = (f"Цена на {pct_b:.0%} высоты канала Боллинджера; ширина канала {width_now:.1%} "
              f"({'выше' if width_now > width_med else 'ниже'} медианы за 90 дней {width_med:.1%}) — "
              f"волатильность {'повышена' if width_now > width_med else 'понижена'}.")

    signals = [
        {"key": "trend", "name": "Тренд (SMA20, SMA50, наклон)", "value": round(s_trend, 3), "text": t_text},
        {"key": "macd", "name": "MACD (12, 26, 9)", "value": round(s_macd, 3), "text": m_text},
        {"key": "rsi", "name": "RSI (14)", "value": round(s_rsi, 3), "text": r_text},
        {"key": "volume", "name": "Подтверждение объёмом", "value": round(s_vol, 3), "text": v_text},
        {"key": "bollinger", "name": "Полосы Боллинджера (20, 2)", "value": round(s_bb, 3), "text": b_text},
    ]
    for s in signals:
        s["weight"] = WEIGHTS[s["key"]]
        s["contribution"] = round(100 * s["weight"] * s["value"], 2)
    score = round(sum(s["contribution"] for s in signals), 2)

    levels = support_resistance(prices)
    risks = []
    if zone == "перекупленность":
        risks.append("RSI в зоне перекупленности — повышен риск коррекции")
    if zone == "перепроданность":
        risks.append("RSI в зоне перепроданности — рынок под давлением продавцов")
    if levels["support"] and close / levels["support"]["level"] - 1 < 0.03:
        risks.append(f"Цена у поддержки {levels['support']['level']:.2f}: её пробой усилит падение")
    if width_now > 1.5 * width_med:
        risks.append("Волатильность в полтора раза выше обычной")

    view = ind if period_start is None else ind.loc[pd.Timestamp(period_start):]
    ret = lambda n: (round(close / float(ind["close"].iloc[-1 - n]) - 1, 4) if len(ind) > n else None)
    details = {
        "date": ind.index[-1].strftime("%Y-%m-%d"), "close": close, "trend": trend, "trend_ru": trend_ru,
        "slope_pct_day": round(slope, 4), "rsi": round(r, 2), "rsi_zone": zone,
        "macd": round(float(last["macd"]), 5), "macd_signal": round(float(last["signal"]), 5),
        "macd_hist": round(float(last["hist"]), 5),
        "sma20": round(float(last["sma20"]), 4), "sma50": round(float(last["sma50"]), 4),
        "ema12": round(float(last["ema12"]), 4), "ema26": round(float(last["ema26"]), 4),
        "bb_upper": round(float(last["bb_upper"]), 4), "bb_lower": round(float(last["bb_lower"]), 4),
        "bb_pct": round(pct_b, 4), "bb_width": round(width_now, 4), "volume_ratio": round(ratio, 3) if ratio == ratio else None,
        "return_1d": ret(1), "return_7d": ret(7), "return_30d": ret(30), "return_90d": ret(90),
        "volatility_30d": round(float(ind["close"].pct_change().tail(30).std() * np.sqrt(365)), 4),
        "support": levels["support"], "resistance": levels["resistance"], "usable_days": usable,
    }
    explanation = (f"TA = {score:+.0f}: " + "; ".join(
        f"{s['name']} {s['value']:+.2f} × {s['weight']:.2f}" for s in signals) + ". " + t_text)
    return {"status": "ok", "score": score, "signals": signals, "details": details, "risks": risks,
            "explanation": explanation, "indicators": view, "levels": levels["levels"]}
