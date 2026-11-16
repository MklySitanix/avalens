"""Имитация внешних API для тестов и тестового режима интерфейса.

Отвечает на те же адреса и в тех же форматах, что и настоящие источники, но данные
синтетические и детерминированные (зависят только от seed и даты). Позволяет
воспроизвести ситуации из протокола тестирования: отказ источника, устаревшие данные,
пропуски, аномалии, отсутствие ончейн-данных, противоречие между модулями.
"""
from __future__ import annotations

import json
import zlib
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests

POSITIVE = [
    "Avalanche surges as institutional adoption grows", "AVAX rallies after major partnership announcement",
    "Avalanche network upgrade launches successfully", "AVAX gains as tokenized fund expands on Avalanche",
    "Avalanche records strong growth in daily transactions", "Bullish breakout: AVAX climbs above resistance",
    "Avalanche wins new integration with payments giant", "AVAX ETF approval boosts optimism",
    "Avalanche ecosystem funding raised for developers", "AVAX rebounds as inflows return",
]
NEGATIVE = [
    "AVAX price drops amid market selloff", "Avalanche DeFi protocol hacked, funds drained",
    "AVAX plunges as outflows accelerate", "Bearish pressure: AVAX falls below support",
    "Avalanche faces concerns over validator decline", "AVAX slides after token unlock fears",
    "Avalanche app struggles with outage", "AVAX tumbles as liquidations hit traders",
    "Avalanche activity declines, analysts warn of weakness", "AVAX crashes to new lows amid uncertainty",
]
NEUTRAL = [
    "Avalanche Foundation publishes quarterly report", "AVAX trading volume steady this week",
    "Avalanche developers discuss roadmap at summit", "What is Avalanche and how does AVAX staking work",
    "Avalanche community call scheduled for next week", "AVAX holds range as traders wait",
]
SPECIAL = [
    "Claim your AVAX airdrop giveaway now, limited offer", "Avalanche kills two skiers in mountain snow rescue",
    "AVAX to the moon lol, great job devs", "Will AVAX recover?", "Bitcoin steady as markets wait for data",
]


class FakeResponse:
    def __init__(self, status_code: int, payload, url: str):
        self.status_code = status_code
        self.text = payload if isinstance(payload, str) else json.dumps(payload)
        self.url = url


class FakeAPI:
    def __init__(self, seed: int = 7, today: str | None = None, days: int = 420, down: tuple = (),
                 stale_days: int = 0, price_gap: tuple | None = None, price_spike_days_ago: int | None = None,
                 price_drift: float = 0.0008, activity_drift: float = 0.0, tone: float = 0.1,
                 n_posts: int = 60, cg_price_factor: float = 1.0, timeout_hosts: tuple = (),
                 staking_stale_days: int = 0):
        self.seed, self.days, self.down, self.stale = seed, days, down, stale_days
        self.today = datetime.strptime(today, "%Y-%m-%d").replace(tzinfo=timezone.utc) if today else \
            datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self.price_gap, self.spike = price_gap, price_spike_days_ago
        self.price_drift, self.activity_drift, self.tone, self.n_posts = price_drift, activity_drift, tone, n_posts
        self.cg_factor, self.timeout_hosts = cg_price_factor, timeout_hosts
        self.staking_stale = staking_stale_days
        self.calls: list[str] = []

    # ---------- ряды ----------
    def _rng(self, key: str):
        return np.random.default_rng(self.seed * 1000 + zlib.crc32(key.encode()) % 100000)

    def _days(self):
        """Даты от старой к новой; последняя — сегодня (минус stale_days)."""
        last = self.today - timedelta(days=self.stale)
        return [last - timedelta(days=self.days - 1 - i) for i in range(self.days)]

    def _walk(self, key: str, start: float, drift: float, vol: float, late_drift: float = 0.0):
        rng = self._rng(key)
        steps = rng.normal(drift, vol, self.days)
        steps[-45:] += late_drift
        return start * np.exp(np.cumsum(steps))

    def _price(self, sym: str):
        base = {"AVAXUSDT": 22.0, "ETHUSDT": 2600.0}.get(sym, 10.0)
        return self._walk("price" + sym, base, 0.0, 0.032, self.price_drift * 10)

    def klines(self, sym: str):
        close, days, rng = self._price(sym), self._days(), self._rng("ohlc" + sym)
        rows = []
        for i, (d, c) in enumerate(zip(days, close)):
            if self.price_gap and self.price_gap[0] <= self.days - 1 - i < self.price_gap[0] + self.price_gap[1]:
                continue
            o = close[i - 1] if i else c
            if self.spike is not None and self.days - 1 - i == self.spike:
                c = c * 4
            hi, lo = max(o, c) * (1 + abs(rng.normal(0, 0.01))), min(o, c) * (1 - abs(rng.normal(0, 0.01)))
            ts = int(d.timestamp() * 1000)
            rows.append([ts, f"{o:.4f}", f"{hi:.4f}", f"{lo:.4f}", f"{c:.4f}", f"{rng.uniform(2e6, 6e6):.2f}",
                         ts + 86399999, "0", 0, "0", "0", "0"])
        return rows

    def metric(self, key: str, start: float, vol: float = 0.06):
        vals = self._walk("m" + key, start, 0.0, vol, self.activity_drift * 10)
        return list(zip(self._days(), vals))

    # ---------- публикации ----------
    def posts(self, sym_kw: str = "AVAX"):
        rng = self._rng("posts")
        p_pos = min(max(0.35 + self.tone * 0.5, 0.02), 0.9)
        p_neg = min(max(0.35 - self.tone * 0.5, 0.02), 0.9)
        out = []
        for i in range(self.n_posts):
            u = rng.random()
            pool = POSITIVE if u < p_pos else NEGATIVE if u < p_pos + p_neg else NEUTRAL
            title = f"{pool[int(rng.integers(len(pool)))]} ({i})"
            when = self.today - timedelta(days=self.stale, hours=float(rng.uniform(1, 24 * 20)))
            out.append((title, when))
        base = self.today - timedelta(days=self.stale)
        out += [(t, base - timedelta(hours=5 + i)) for i, t in enumerate(SPECIAL)]
        out += [("Avalanche surges as institutional adoption grows (0)", base - timedelta(hours=30))]  # дубль
        return out

    def rss(self, items):
        body = "".join(
            f"<item><title>{t}</title><link>https://example.org/news/{abs(hash(t)) % 10**8}</link>"
            f"<pubDate>{format_datetime(w)}</pubDate><description>{t}. Synthetic test article.</description></item>"
            for t, w in items)
        return f'<?xml version="1.0"?><rss version="2.0"><channel><title>fake</title>{body}</channel></rss>'

    # ---------- маршрутизация ----------
    def __call__(self, url, params=None, timeout=10, headers=None):
        self.calls.append(url)
        host, path = urlparse(url).netloc, urlparse(url).path
        params = dict(params or {})
        params.update({k: v[0] for k, v in parse_qs(urlparse(url).query).items()})
        if any(h in host for h in self.timeout_hosts):
            raise requests.Timeout()
        if any(h in host for h in self.down):
            return FakeResponse(503, "Service Unavailable", url)
        ok = lambda payload: FakeResponse(200, payload, url)

        if "binance" in host:
            sym = params.get("symbol", "AVAXUSDT")
            if path.endswith("/klines"):
                return ok(self.klines(sym))
            p = self._price(sym)
            return ok({"symbol": sym, "lastPrice": f"{p[-1]:.4f}", "priceChangePercent": f"{(p[-1] / p[-2] - 1) * 100:.3f}",
                       "closeTime": int(self.today.timestamp() * 1000)})
        if "coingecko" in host:
            cid = path.split("/coins/")[1].split("/")[0]
            sym = "AVAXUSDT" if cid == "avalanche-2" else "ETHUSDT"
            p = self._price(sym)
            if path.endswith("/market_chart"):
                ms = [int(d.timestamp() * 1000) for d in self._days()]
                return ok({"prices": [[t, float(v)] for t, v in zip(ms, p)],
                           "total_volumes": [[t, 3e8] for t in ms]})
            circ, mx = (432e6, 720e6) if sym == "AVAXUSDT" else (120.7e6, None)
            price = float(p[-1]) * self.cg_factor
            return ok({"id": cid, "market_cap_rank": 20, "market_data": {
                "current_price": {"usd": price}, "market_cap": {"usd": price * circ},
                "fully_diluted_valuation": {"usd": price * (mx or circ)}, "circulating_supply": circ,
                "total_supply": 460e6 if mx else circ, "max_supply": mx, "total_volume": {"usd": 3.1e8},
                "last_updated": self.today.strftime("%Y-%m-%dT%H:%M:%SZ")}})
        if "metrics.avax.network" in host:
            name = path.rsplit("/", 1)[1]
            start = {"activeAddresses": 45000, "txCount": 900000, "feesPaid": 1500,
                     "validatorCount": 1400, "delegatorCount": 52000}.get(name)
            if start is None:
                return FakeResponse(404, '{"message":"not found"}', url)
            series = self.metric(name, start, 0.004 if "Count" in name and name != "txCount" else 0.07)
            if self.staking_stale and name in ("validatorCount", "delegatorCount"):
                series = series[:-self.staking_stale]
            lo, hi = int(params.get("startTimestamp", 0)), int(params.get("endTimestamp", 2 ** 40))
            rows = [{"value": round(float(v), 2), "timestamp": int(d.timestamp())} for d, v in reversed(series)
                    if lo <= d.timestamp() <= hi]
            size, off = int(params.get("pageSize", 100)), int(params.get("pageToken", 0) or 0)
            page = {"results": rows[off:off + size]}
            if off + size < len(rows):
                page["nextPageToken"] = str(off + size)
            return ok(page)
        if "llama.fi" in host:
            chain = path.rsplit("/", 1)[1]
            scale = 1.0 if chain.lower().startswith("aval") else 55.0
            if "historicalChainTvl" in path:
                return ok([{"date": int(d.timestamp()), "tvl": float(v)}
                           for d, v in self.metric("tvl" + chain.lower(), 1.4e9 * scale, 0.02)])
            if "summary/fees" in path:
                return ok({"total24h": 1, "totalDataChart": [[int(d.timestamp()), float(v)]
                           for d, v in self.metric("fees" + chain.lower(), 60000 * scale, 0.08)]})
            if "overview/dexs" in path:
                return ok({"totalDataChart": [[int(d.timestamp()), float(v)]
                           for d, v in self.metric("dex" + chain.lower(), 2.5e8 * scale, 0.08)]})
            if path.endswith("/protocols"):
                return ok([{"name": f"p{i}", "chains": ["Avalanche", "Ethereum"] if i % 3 == 0 else ["Ethereum"]}
                           for i in range(1200)])
        if "alternative.me" in host:
            rng = self._rng("fng")
            vals = np.clip(50 + np.cumsum(rng.normal(0, 4, self.days)), 5, 95)
            return ok({"data": [{"value": str(int(v)), "value_classification": "x", "timestamp": str(int(d.timestamp()))}
                                for d, v in reversed(list(zip(self._days(), vals)))]})
        if "reddit.com" in host:
            if path.endswith(".json"):
                items = self.posts()[: self.n_posts // 3]
                return ok({"data": {"children": [{"data": {
                    "title": t.replace("Avalanche", "AVAX").replace(")", " r)"), "selftext": "", "author": f"user{i % 9}",
                    "created_utc": w.timestamp(), "permalink": f"/r/Avax/comments/{i}/", "stickied": False}}
                    for i, (t, w) in enumerate(items)]}})
            return ok(self.rss([]))
        if "news.google.com" in host:
            q = params.get("q", "")
            if "Ethereum" in q:
                return ok(self.rss([(t.replace("Avalanche", "Ethereum").replace("AVAX", "ETH"), w) for t, w in self.posts()]))
            return ok(self.rss(self.posts()))
        if any(h in host for h in ("cointelegraph", "coindesk", "decrypt", "cryptoslate")):
            return ok(self.rss([(f"{t} [{host}]", w) for t, w in self.posts()[:8]]))
        return FakeResponse(404, "not found", url)


def install(api: FakeAPI) -> None:
    """Подменяет сетевой слой адаптеров источников."""
    from data.sources import base
    base._get = api
    base._sleep = lambda seconds: None
