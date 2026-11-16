"""Alternative.me Fear & Greed Index — настроение всего крипторынка (не AVAX!)."""
from __future__ import annotations

import pandas as pd

from .base import SourceResult, http_get, to_date

URL = "https://api.alternative.me/fng/"
NAME = "alternative_me"


def parse(d: dict) -> pd.DataFrame:
    out = [{"metric": "fear_greed", "date": to_date(int(r["timestamp"])), "value": float(r["value"]),
            "source": NAME} for r in d.get("data", []) if r.get("value") is not None]
    return pd.DataFrame(out)


def fetch(limit: int = 400) -> SourceResult:
    res = SourceResult(NAME, ok=False)
    f = http_get(NAME, URL, {"limit": limit, "format": "json"})
    res.fetches.append(f)
    if f.ok and isinstance(f.data, dict) and f.data.get("data"):
        res.frames["metrics"] = parse(f.data)
        res.ok = True
    else:
        if f.ok:
            f.ok, f.error = False, "в ответе нет списка data"
        res.error = f.error
    return res
