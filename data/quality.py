"""Контроль качества данных: пропуски, аномалии, устаревание, расхождение источников.

Каждая найденная проблема попадает в отчёт: интерфейс показывает её предупреждением,
а комплексная оценка снижает Confidence Score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_FILL_DAYS = 3        # пропуск до 3 дней подряд заполняется, длиннее — разрывает ряд
SPIKE_LIMIT = 0.5        # отклонение цены более чем на 50% от медианы соседних дней
METRIC_SPIKE = 10.0      # ончейн-значение в 10 раз выше или ниже медианы за 30 дней
DIVERGENCE_LIMIT = 0.02  # расхождение цен двух источников более 2%


def _longest_run(mask: pd.Series) -> int:
    best = cur = 0
    for v in mask.to_numpy():
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def clean_prices(df: pd.DataFrame, end: str | pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Очищает дневные свечи. Возвращает ряд с индексом по дням без разрывов и отчёт.

    Аномальная свеча (нулевая цена, high < low, одиночный выброс) исключается и
    обрабатывается как пропуск. Заполненные дни помечены is_filled и не учитываются
    в объёмах и в полноте данных.
    """
    report = {"rows": 0, "expected_days": 0, "missing_days": 0, "filled_days": 0, "anomalies": [],
              "longest_gap": 0, "cut_before": None, "last_date": None, "usable_days": 0, "completeness": 0.0}
    cols = ["open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols + ["is_filled", "is_anomaly"]), report
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    if end is not None:
        d = d[d["date"] <= pd.Timestamp(end)]
    d = d.drop_duplicates("date", keep="last").sort_values("date").set_index("date")[cols].astype(float)
    report["rows"] = len(d)
    if d.empty:
        return pd.DataFrame(columns=cols + ["is_filled", "is_anomaly"]), report

    # 1. аномалии: недопустимые значения и одиночные выбросы
    invalid = (d["close"] <= 0) | (d["high"] < d["low"]) | d["close"].isna() | (d["volume"] < 0)
    med = d["close"].where(~invalid).rolling(7, center=True, min_periods=3).median()
    spike = (d["close"] / med - 1).abs() > SPIKE_LIMIT
    anomaly = invalid | spike.fillna(False)
    report["anomalies"] = [x.strftime("%Y-%m-%d") for x in d.index[anomaly]]
    d.loc[anomaly, cols] = np.nan

    # 2. пропуски: полный календарь дней
    full = pd.date_range(d.index.min(), d.index.max(), freq="D")
    d = d.reindex(full)
    missing = d["close"].isna()
    report["expected_days"] = len(full)
    report["missing_days"] = int(missing.sum())
    report["longest_gap"] = _longest_run(missing)

    # 3. длинный разрыв: используем только данные после него
    run = missing.groupby((~missing).cumsum()).cumsum()
    long_gap_days = run[run > MAX_FILL_DAYS]
    if len(long_gap_days):
        after = d.index[d.index > long_gap_days.index.max()]
        first_valid = d.loc[after, "close"].first_valid_index() if len(after) else None
        if first_valid is None:
            report["cut_before"] = "all"
            return pd.DataFrame(columns=cols + ["is_filled", "is_anomaly"]), report
        report["cut_before"] = first_valid.strftime("%Y-%m-%d")
        d = d.loc[first_valid:]
        missing = d["close"].isna()

    # 4. короткие пропуски: цена прошлого дня, объём неизвестен
    d["is_filled"] = missing.astype(int)
    d["is_anomaly"] = pd.Series(anomaly, index=anomaly.index).reindex(d.index).fillna(False).astype(int)
    prev_close = d["close"].ffill()
    for c in ("open", "high", "low", "close"):
        d[c] = d[c].fillna(prev_close)
    d.index.name = "date"
    report["filled_days"] = int(d["is_filled"].sum())
    report["usable_days"] = int((d["is_filled"] == 0).sum())
    report["last_date"] = d.index.max().strftime("%Y-%m-%d")
    report["completeness"] = round(report["usable_days"] / max(len(d), 1), 4)
    return d, report


def clean_metric(s: pd.Series, end: str | pd.Timestamp | None = None) -> tuple[pd.Series, dict]:
    """Очищает суточный ряд показателя (индекс — даты)."""
    report = {"rows": 0, "missing_days": 0, "anomalies": [], "last_date": None, "completeness": 0.0}
    if s is None or len(s) == 0:
        return pd.Series(dtype=float), report
    s = s.copy().astype(float)
    s.index = pd.to_datetime(s.index)
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    report["rows"] = len(s)
    if s.empty:
        return s, report
    med = s.rolling(30, min_periods=5).median().shift(1)
    ratio = s / med.where(med > 0)
    anomaly = (s < 0) | (ratio > METRIC_SPIKE) | (ratio < 1 / METRIC_SPIKE)
    anomaly = anomaly.fillna(False)
    report["anomalies"] = [x.strftime("%Y-%m-%d") for x in s.index[anomaly]]
    s = s.where(~anomaly)
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full)
    report["missing_days"] = int(s.isna().sum())
    report["completeness"] = round(1 - report["missing_days"] / max(len(full), 1), 4)
    s = s.interpolate(limit=MAX_FILL_DAYS, limit_area="inside")
    report["last_date"] = s.last_valid_index().strftime("%Y-%m-%d") if s.last_valid_index() is not None else None
    s.index.name = "date"
    return s, report


def freshness(age_hours: float | None, threshold_hours: float) -> float:
    """1 в пределах порога актуальности, линейно падает до 0 при тройном пороге."""
    if age_hours is None:
        return 0.0
    if age_hours <= threshold_hours:
        return 1.0
    if age_hours >= 3 * threshold_hours:
        return 0.0
    return round(1 - (age_hours - threshold_hours) / (2 * threshold_hours), 4)


def price_divergence(price_a: float | None, price_b: float | None) -> dict:
    """Сравнивает цену двух независимых источников."""
    if not price_a or not price_b:
        return {"checked": False, "diverged": False, "diff_pct": None}
    diff = abs(price_a - price_b) / ((price_a + price_b) / 2)
    return {"checked": True, "diverged": bool(diff > DIVERGENCE_LIMIT), "diff_pct": round(diff * 100, 3)}
