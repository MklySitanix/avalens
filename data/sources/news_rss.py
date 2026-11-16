"""RSS-ленты криптоСМИ и поиск Google News — новости для анализа настроений."""
from __future__ import annotations

import calendar
import html
import re
from datetime import datetime, timezone
from urllib.parse import quote_plus

import feedparser

from .base import SourceResult, http_get

NAME = "news"
_TAG = re.compile(r"<[^>]+>")


def clean_html(s: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", s or ""))).strip()


def parse_feed(feed_name: str, xml_text: str) -> list[dict]:
    parsed = feedparser.parse(xml_text)
    posts = []
    for e in parsed.entries:
        ts = e.get("published_parsed") or e.get("updated_parsed")
        if not ts or not e.get("title"):
            continue
        published = datetime.fromtimestamp(calendar.timegm(ts), tz=timezone.utc)
        posts.append({
            "source": f"news:{feed_name}",
            "url": e.get("link"),
            "published_at": published.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "author": e.get("author") or (e.get("source") or {}).get("title") or feed_name,
            "title": clean_html(e.get("title")),
            "text": clean_html(e.get("summary"))[:1000],
        })
    return posts


def fetch(asset: dict, feeds: list[dict], symbol: str) -> SourceResult:
    res = SourceResult(NAME, ok=False)
    posts, failed = [], []
    query = quote_plus(f'{asset.get("name", symbol)} {symbol} crypto')
    for feed in feeds:
        f = http_get(f"news:{feed['name']}", feed["url"].replace("{query}", query), kind="text")
        if f.ok:
            items = parse_feed(feed["name"], f.text)
            if items:
                posts.extend(items)
            else:
                f.ok, f.error = False, "лента не содержит записей"
        res.fetches.append(f)
        if not f.ok:
            failed.append(feed["name"])
    res.frames["posts"] = posts
    res.ok = bool(posts)
    if failed:
        res.note = "недоступны ленты: " + ", ".join(failed)
    if not res.ok:
        res.error = next((x.error for x in res.fetches if not x.ok), "нет записей")
    return res
