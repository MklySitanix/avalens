"""Binance Spot API — основной источник цены: дневные свечи OHLCV и текущая цена."""
from __future__ import annotations

import pandas as pd

from .base import SourceResult, http_get, to_date

# Основной адрес и официальное зеркало рыночных данных
HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")
NAME = "binance"


def parse_klines(rows: list) -> pd.DataFrame:
    """Ответ /api/v3/klines: [openTime, open, high, low, close, volume, closeTime, ...]."""
    out = [{"date": to_date(r[0], "ms"), "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
            "close": float(r[4]), "volume": float(r[5]), "close_time": int(r[6])} for r in rows]
    return pd.DataFrame(out)


def fetch(asset: dict, limit: int = 1000) -> SourceResult:
    pair = asset.get("binance_pair")
    res = SourceResult(NAME, ok=False)
    if not pair:
        res.error = "для актива не задана торговая пара"
        return res
    for host in HOSTS:
        f = http_get(NAME, f"{host}/api/v3/klines", {"symbol": pair, "interval": "1d", "limit": limit})
        res.fetches.append(f)
        if not f.ok:
            res.error = f.error
            continue
        if not isinstance(f.data, list) or not f.data or len(f.data[0]) < 7:
            f.ok, f.error = False, "неожиданный формат свечей"
            res.error = f.error
            continue
        res.frames["prices"] = parse_klines(f.data)
        t = http_get(NAME, f"{host}/api/v3/ticker/24hr", {"symbol": pair})
        res.fetches.append(t)
        if t.ok and isinstance(t.data, dict) and "lastPrice" in t.data:
            res.frames["ticker"] = {
                "price": float(t.data["lastPrice"]),
                "change_pct_24h": float(t.data.get("priceChangePercent", "nan")),
                "time": to_date(t.data.get("closeTime", 0), "ms") if t.data.get("closeTime") else None,
                "fetched_at": t.fetched_at,
            }
        res.ok, res.error = True, None
        return res
    return res
