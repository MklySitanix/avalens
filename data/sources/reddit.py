"""Reddit — мнение держателей. Публичный JSON сообщества; без авторизации Reddit может отказать."""
from __future__ import annotations

from datetime import datetime, timezone

from .base import SourceResult, http_get
from .news_rss import parse_feed

NAME = "reddit"


def parse_listing(d: dict) -> list[dict]:
    posts = []
    for child in (d.get("data") or {}).get("children", []):
        p = child.get("data") or {}
        if not p.get("title") or not p.get("created_utc"):
            continue
        flags = []
        if p.get("stickied") or p.get("distinguished"):
            flags.append("служебный пост")
        posts.append({
            "source": "reddit",
            "url": "https://www.reddit.com" + p.get("permalink", ""),
            "published_at": datetime.fromtimestamp(p["created_utc"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "author": p.get("author"),
            "title": p["title"].strip(),
            "text": (p.get("selftext") or "")[:1000],
            "flags": flags,
        })
    return posts


def fetch(asset: dict) -> SourceResult:
    res = SourceResult(NAME, ok=False)
    sub = asset.get("subreddit")
    if not sub:
        res.error = "для актива не задано сообщество"
        res.note = "skip"
        return res
    f = http_get(NAME, f"https://www.reddit.com/r/{sub}/new.json", {"limit": 100, "raw_json": 1}, pauses=(1,))
    res.fetches.append(f)
    if f.ok and isinstance(f.data, dict) and (f.data.get("data") or {}).get("children"):
        res.frames["posts"] = parse_listing(f.data)
        res.ok = True
        return res
    if f.ok:
        f.ok, f.error = False, "неожиданный формат ответа"
    # запасной путь: RSS-лента того же сообщества
    r = http_get(NAME, f"https://www.reddit.com/r/{sub}/new/.rss", kind="text", pauses=(1,))
    res.fetches.append(r)
    if r.ok:
        posts = parse_feed("reddit", r.text)
        for p in posts:
            p["source"] = "reddit"
        if posts:
            res.frames["posts"] = posts
            res.ok = True
            return res
        r.ok, r.error = False, "лента не содержит записей"
    res.error = f.error or r.error
    return res
