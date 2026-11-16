"""CoinGecko (доступ без ключа, около 5 запросов в минуту).

Даёт то, чего нет у биржи: капитализацию, FDV и предложение токена.
Его цена — второе независимое значение для проверки противоречивости данных,
а ценовой ряд — резерв на случай недоступности Binance.
"""
from __future__ import annotations

import pandas as pd

from .base import SourceResult, http_get, to_date

BASE = "https://api.coingecko.com/api/v3"
NAME = "coingecko"


def parse_coin(d: dict) -> dict:
    md = d.get("market_data") or {}
    usd = lambda key: (md.get(key) or {}).get("usd")
    return {
        "price": usd("current_price"),
        "market_cap": usd("market_cap"),
        "fdv": usd("fully_diluted_valuation"),
        "circulating": md.get("circulating_supply"),
        "total_supply": md.get("total_supply"),
        "max_supply": md.get("max_supply"),
        "volume_24h": usd("total_volume"),
        "market_cap_rank": d.get("market_cap_rank"),
        "last_updated": md.get("last_updated") or d.get("last_updated"),
    }


def parse_chart(d: dict) -> pd.DataFrame:
    """market_chart: prices [[ms, price]], total_volumes [[ms, volume]]. Свечей нет: O=H=L=C."""
    vol = {to_date(t, "ms"): v for t, v in d.get("total_volumes", [])}
    rows = {}
    for t, p in d.get("prices", []):
        day = to_date(t, "ms")
        rows[day] = {"date": day, "open": p, "high": p, "low": p, "close": p, "volume": vol.get(day, 0.0)}
    return pd.DataFrame(list(rows.values()))


def fetch(asset: dict, with_chart: bool = False, days: int = 365) -> SourceResult:
    cid = asset.get("coingecko_id")
    res = SourceResult(NAME, ok=False)
    if not cid:
        res.error = "для актива не задан идентификатор CoinGecko"
        return res
    f = http_get(NAME, f"{BASE}/coins/{cid}", {
        "localization": "false", "tickers": "false", "community_data": "false",
        "developer_data": "false", "sparkline": "false"})
    res.fetches.append(f)
    if f.ok and isinstance(f.data, dict) and f.data.get("market_data"):
        res.frames["fundamentals"] = parse_coin(f.data)
        res.ok = True
    else:
        if f.ok:
            f.ok, f.error = False, "в ответе нет блока market_data"
        res.error = f.error
    if with_chart:
        c = http_get(NAME, f"{BASE}/coins/{cid}/market_chart",
                     {"vs_currency": "usd", "days": days, "interval": "daily"})
        res.fetches.append(c)
        if c.ok and isinstance(c.data, dict) and c.data.get("prices"):
            res.frames["prices"] = parse_chart(c.data)
            res.ok = True
        elif not res.ok:
            res.error = c.error or "в ответе нет цен"
    return res
