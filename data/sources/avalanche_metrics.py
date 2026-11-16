"""Avalanche Metrics API — официальный источник ончейн-данных сети, авторизация не нужна.

Суточные показатели C-Chain (chainId 43114): активные адреса, число транзакций, комиссии.
Показатели стейкинга Primary Network: число валидаторов и делегаторов.
Ответ разбит на страницы: {"results": [{"value", "timestamp"}], "nextPageToken"}.
Формат ответов и единицы (комиссии — в AVAX) сверены с реальным API 16.11.2026; на эту дату
показатели стейкинга в источнике заканчивались 31.08.2026 — устаревший ряд система исключает из оценки.
"""
from __future__ import annotations

import time

import pandas as pd

from .base import SourceResult, http_get, to_date

BASE = "https://metrics.avax.network/v2"
NAME = "avalanche_metrics"
CHAIN_METRICS = ("activeAddresses", "txCount", "feesPaid")
STAKING_METRICS = ("validatorCount", "delegatorCount")
MAX_PAGES = 12


def _paged(url: str, params: dict, res: SourceResult) -> list[dict]:
    rows: list[dict] = []
    token = None
    for _ in range(MAX_PAGES):
        p = dict(params)
        if token:
            p["pageToken"] = token
        f = http_get(NAME, url, p)
        res.fetches.append(f)
        if not f.ok:
            return rows
        if not isinstance(f.data, dict) or not isinstance(f.data.get("results"), list):
            f.ok, f.error = False, "в ответе нет списка results"
            return rows
        rows.extend(f.data["results"])
        token = f.data.get("nextPageToken")
        if not token:
            break
    return rows


def parse_rows(metric: str, rows: list[dict]) -> pd.DataFrame:
    out = [{"metric": metric, "date": to_date(r["timestamp"]), "value": float(r["value"]), "source": NAME}
           for r in rows if r.get("value") is not None and r.get("timestamp") is not None]
    return pd.DataFrame(out).drop_duplicates(["metric", "date"]) if out else pd.DataFrame(
        columns=["metric", "date", "value", "source"])


def fetch(asset: dict, days: int = 400, now: float | None = None) -> SourceResult:
    res = SourceResult(NAME, ok=False)
    chain_id = asset.get("chain_id")
    if not chain_id:
        res.error = "источник не применим к этому активу"
        res.note = "skip"
        return res
    end = int(now or time.time())
    params = {"startTimestamp": end - days * 86400, "endTimestamp": end, "timeInterval": "day", "pageSize": 1000}
    frames, missing = [], []
    def collect(metric: str, url: str, query: dict) -> None:
        df = parse_rows(metric, _paged(url, query, res))
        if df.empty:
            missing.append(metric)
        else:
            frames.append(df)

    for metric in CHAIN_METRICS:
        collect(metric, f"{BASE}/chains/{chain_id}/metrics/{metric}", params)
    network = asset.get("metrics_network")
    if network:
        sp = {"startTimestamp": params["startTimestamp"], "endTimestamp": end, "pageSize": 1000}
        for metric in STAKING_METRICS:
            collect(metric, f"{BASE}/networks/{network}/metrics/{metric}", sp)
    if frames:
        res.frames["metrics"] = pd.concat(frames, ignore_index=True)
        res.ok = True
    if missing:
        res.note = "не получены показатели: " + ", ".join(missing)
        if not res.ok:
            res.error = next((f.error for f in res.fetches if not f.ok), "нет данных")
    return res
