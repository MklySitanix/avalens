"""DefiLlama — история TVL сети, комиссии и число протоколов. Бесплатно, без ключа."""
from __future__ import annotations

import pandas as pd

from .base import SourceResult, http_get, to_date

BASE = "https://api.llama.fi"
NAME = "defillama"


def parse_tvl(rows: list) -> pd.DataFrame:
    out = [{"metric": "tvl", "date": to_date(r["date"]), "value": float(r["tvl"]), "source": NAME}
           for r in rows if r.get("tvl") is not None]
    return pd.DataFrame(out)


def parse_chart(metric: str, d: dict) -> pd.DataFrame:
    """totalDataChart: [[unix, value], ...]."""
    out = [{"metric": metric, "date": to_date(t), "value": float(v), "source": NAME}
           for t, v in d.get("totalDataChart") or [] if v is not None]
    return pd.DataFrame(out)


def fetch(asset: dict, with_protocols: bool = True) -> SourceResult:
    res = SourceResult(NAME, ok=False)
    chain = asset.get("llama_chain")
    if not chain:
        res.error = "для актива не задана сеть DefiLlama"
        return res
    frames, missing = [], []

    f = http_get(NAME, f"{BASE}/v2/historicalChainTvl/{chain}")
    res.fetches.append(f)
    if f.ok and isinstance(f.data, list) and f.data and "tvl" in f.data[0]:
        frames.append(parse_tvl(f.data))
    else:
        if f.ok:
            f.ok, f.error = False, "неожиданный формат TVL"
        missing.append("tvl")

    slug = asset.get("llama_fees_slug")
    if slug:
        f = http_get(NAME, f"{BASE}/summary/fees/{slug}", {"dataType": "dailyFees"})
        res.fetches.append(f)
        if f.ok and isinstance(f.data, dict) and f.data.get("totalDataChart"):
            frames.append(parse_chart("fees_usd", f.data))
        else:
            if f.ok:
                f.ok, f.error = False, "в ответе нет ряда комиссий"
            missing.append("fees_usd")

    f = http_get(NAME, f"{BASE}/overview/dexs/{chain}",
                 {"excludeTotalDataChartBreakdown": "true", "excludeTotalDataChart": "false"})
    res.fetches.append(f)
    if f.ok and isinstance(f.data, dict) and f.data.get("totalDataChart"):
        frames.append(parse_chart("dex_volume_usd", f.data))
    else:
        if f.ok:
            f.ok, f.error = False, "в ответе нет ряда объёма DEX"
        missing.append("dex_volume_usd")

    if with_protocols:
        f = http_get(NAME, f"{BASE}/protocols", timeout=30)
        if f.ok and isinstance(f.data, list):
            n = sum(1 for p in f.data if chain in (p.get("chains") or []))
            res.frames["protocols_count"] = n
            f.text = f'{{"protocols_total": {len(f.data)}, "on_{chain}": {n}}}'  # полный список слишком велик
        else:
            if f.ok:
                f.ok, f.error = False, "неожиданный формат списка протоколов"
            missing.append("protocols_count")
        res.fetches.append(f)

    if frames:
        res.frames["metrics"] = pd.concat(frames, ignore_index=True)
        res.ok = True
    else:
        res.error = next((x.error for x in res.fetches if not x.ok), "нет данных")
    if missing:
        res.note = "не получены показатели: " + ", ".join(missing)
    return res
