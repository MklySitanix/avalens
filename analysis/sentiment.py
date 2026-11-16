"""Модуль анализа рыночных настроений. Тональность определяется словарным методом.

Как система определяет тональность текста:
1. Текст приводится к нижнему регистру и разбивается на слова.
2. Каждое слово и словосочетание ищется в словаре LEXICON с весом от -2 до +2.
3. Отрицание (not, no, never, n't ...) в пределах трёх слов перед найденным словом
   меняет знак веса и ослабляет его (x -0,7); усилитель (very, massive ...) умножает на 1,5.
4. Слова заголовка считаются с весом 2, слова текста — с весом 1.
5. Сумма весов сжимается в диапазон [-1; +1]: score = tanh(сумма / 3).
6. score >= +0,2 — положительная публикация, <= -0,2 — отрицательная, иначе нейтральная.
Словарный метод не понимает смысла, поэтому сомнительные случаи (сарказм, вопрос,
смешанные сигналы) помечаются флагом и переводятся в нейтральные, а реклама и
вероятные боты исключаются из индекса.
"""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np
import pandas as pd

POS_T, NEG_T = 0.2, -0.2
MIN_POSTS = 10
TARGET_POSTS = 30
WINDOW_DAYS = 14

LEXICON = {
    # сильные положительные
    "surge": 2, "surges": 2, "surged": 2, "soar": 2, "soars": 2, "soared": 2, "rally": 2, "rallies": 2,
    "rallied": 2, "breakout": 2, "bullish": 2, "record high": 2, "all-time high": 2, "skyrocket": 2,
    "skyrockets": 2, "approval": 2, "approved": 2, "approves": 2, "milestone": 2, "outperform": 2,
    "outperforms": 2, "outperformed": 2,
    # умеренные положительные
    "gain": 1, "gains": 1, "gained": 1, "rise": 1, "rises": 1, "rose": 1, "rising": 1, "jump": 1, "jumps": 1,
    "jumped": 1, "climb": 1, "climbs": 1, "climbed": 1, "growth": 1, "grows": 1, "growing": 1, "grew": 1,
    "partnership": 1, "partners": 1, "partnered": 1, "launch": 1, "launches": 1, "launched": 1,
    "adoption": 1, "adopts": 1, "adopted": 1, "upgrade": 1, "upgrades": 1, "upgraded": 1, "integration": 1,
    "integrates": 1, "integrated": 1, "expands": 1, "expansion": 1, "recover": 1, "recovers": 1,
    "recovery": 1, "rebound": 1, "rebounds": 1, "rebounded": 1, "strong": 1, "stronger": 1, "boost": 1,
    "boosts": 1, "boosted": 1, "wins": 1, "win": 1, "success": 1, "successful": 1, "inflows": 1, "inflow": 1,
    "accumulate": 1, "accumulation": 1, "accumulating": 1, "optimistic": 1, "optimism": 1, "upside": 1,
    "support": 1, "backed": 1, "invests": 1, "investment": 1, "raises": 1, "raised": 1, "funding": 1,
    "tokenized": 1, "tokenization": 1, "institutional": 1, "etf": 1, "buy": 1, "buying": 1, "profit": 1,
    "profits": 1, "improves": 1, "improved": 1, "faster": 1, "cheaper": 1, "record": 1, "positive": 1,
    "promising": 1, "leads": 1, "leading": 1, "tops": 1, "higher": 1, "up": 1, "great": 1, "love": 1,
    # сильные отрицательные
    "crash": -2, "crashes": -2, "crashed": -2, "plunge": -2, "plunges": -2, "plunged": -2, "collapse": -2,
    "collapses": -2, "collapsed": -2, "hack": -2, "hacked": -2, "exploit": -2, "exploited": -2, "scam": -2,
    "fraud": -2, "rug pull": -2, "rugpull": -2, "bearish": -2, "lawsuit": -2, "sued": -2, "bankrupt": -2,
    "bankruptcy": -2, "stolen": -2, "drained": -2, "outage": -2, "halted": -2, "dump": -2, "dumps": -2,
    "dumped": -2, "tumbles": -2, "tumble": -2, "tumbled": -2, "plummets": -2, "plummet": -2, "plummeted": -2,
    "liquidations": -2, "liquidated": -2, "capitulation": -2,
    # умеренные отрицательные
    "fall": -1, "falls": -1, "fell": -1, "falling": -1, "drop": -1, "drops": -1, "dropped": -1, "decline": -1,
    "declines": -1, "declined": -1, "slump": -1, "slumps": -1, "slumped": -1, "slide": -1, "slides": -1,
    "slid": -1, "sink": -1, "sinks": -1, "sank": -1, "loss": -1, "losses": -1, "loses": -1, "lost": -1,
    "weak": -1, "weaker": -1, "weakness": -1, "risk": -1, "risks": -1, "concern": -1, "concerns": -1,
    "warning": -1, "warns": -1, "fear": -1, "fears": -1, "selloff": -1, "sell-off": -1, "selling": -1,
    "sell": -1, "outflows": -1, "outflow": -1, "delay": -1, "delays": -1, "delayed": -1, "bug": -1,
    "vulnerability": -1, "investigation": -1, "probe": -1, "fine": -1, "fined": -1, "ban": -1, "banned": -1,
    "bans": -1, "struggles": -1, "struggle": -1, "struggling": -1, "downturn": -1, "lower": -1, "down": -1,
    "pressure": -1, "resistance": -1, "rejected": -1, "rejects": -1, "unlock": -1, "unlocks": -1,
    "inflation": -1, "dilution": -1, "worst": -1, "bad": -1, "negative": -1, "dead": -1, "failing": -1,
    "fails": -1, "failed": -1, "failure": -1, "uncertainty": -1, "volatile": -1, "underperform": -1,
    "underperforms": -1, "underperformed": -1, "low": -1, "lows": -1,
}
PHRASES = sorted((k for k in LEXICON if " " in k), key=len, reverse=True)
NEGATIONS = {"not", "no", "never", "without", "isn't", "isnt", "wasn't", "won't", "wont", "can't", "cant",
             "doesn't", "doesnt", "didn't", "didnt", "fails", "failed", "unable", "nothing", "hardly"}
INTENSIFIERS = {"very", "massive", "massively", "huge", "sharply", "strongly", "extremely", "major", "big"}
PROMO = ("giveaway", "airdrop claim", "claim your", "promo code", "sign up", "sponsored", "press release",
         "presale", "pre-sale", "casino", "betting", "free spins", "best crypto to buy", "100x", "1000x",
         "referral", "bonus code", "join our telegram", "limited offer", "guaranteed returns")
SARCASM = ("/s", "yeah right", "sure buddy", "sure, ", "totally", "lol", "lmao", "🤡", "🙄", "great job",
           "what could go wrong", "thanks a lot", "as usual", "oh great", "so much for", "to the moon they said")
CRYPTO_CONTEXT = ("crypto", "token", "blockchain", "defi", "price", "coin", "avax", "staking", "network",
                  "chain", "trading", "etf", "market", "subnet", "validator", "l1")
STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "as", "at", "by", "with",
        "from", "its", "it", "this", "that", "after", "amid", "over", "into", "be", "has", "have", "will",
        "what", "why", "how", "new", "says", "price", "crypto"}
_WORD = re.compile(r"[a-z0-9$'\-]+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("’", "'")).strip()


def text_hash(title: str) -> str:
    """Хэш нормализованного заголовка: одинаковые перепечатки дают один хэш и не дублируются в базе."""
    key = re.sub(r"[^a-z0-9 ]", "", normalize(title))
    key = re.sub(r" - [a-z0-9 .]+$", "", key)
    return hashlib.sha1(key.encode()).hexdigest()


def is_relevant(title: str, text: str, asset: dict, symbol: str) -> bool:
    """Фильтр по активу: ключевое слово целиком, а «avalanche» — только в криптоконтексте."""
    full = normalize(f"{title} {text}")
    hits = [k for k in asset.get("keywords", []) if re.search(rf"(?<![a-z0-9]){re.escape(k.lower())}(?![a-z0-9])", full)]
    if re.search(rf"\${symbol.lower()}(?![a-z0-9])", full):
        hits.append("$" + symbol.lower())
    if not hits:
        return False
    generic = {str(asset.get("name", "")).lower()}
    if set(hits) <= generic:  # совпало только название-омоним
        if any(w in full for w in asset.get("exclude_words", [])) and not any(c in full for c in CRYPTO_CONTEXT):
            return False
    return True


def _score_part(text: str) -> tuple[float, list[str], float, float]:
    t = normalize(text)
    matched, total, pos, neg = [], 0.0, 0.0, 0.0
    for ph in PHRASES:
        if ph in t:
            w = LEXICON[ph]
            total += w
            pos, neg = pos + max(w, 0), neg + min(w, 0)
            matched.append(f"{ph} ({w:+d})")
            t = t.replace(ph, " ")
    tokens = _WORD.findall(t)
    for i, tok in enumerate(tokens):
        w = LEXICON.get(tok.strip("'-$"))
        if w is None:
            continue
        before = tokens[max(0, i - 3):i]
        note = ""
        negated = False
        for b in reversed(before):  # отрицание действует только на ближайшее оценочное слово
            if b.strip("'-$") in LEXICON:
                break
            if b in NEGATIONS or b.endswith("n't"):
                negated = True
                break
        if negated:
            w, note = -0.7 * w, ", отрицание"
        elif before and before[-1] in INTENSIFIERS:
            w, note = 1.5 * w, ", усилитель"
        total += w
        pos, neg = pos + max(w, 0), neg + min(w, 0)
        matched.append(f"{tok} ({w:+.1f}{note})")
    return total, matched, pos, neg


def score_post(title: str, text: str = "") -> dict:
    """Тональность одной публикации с объяснением: какие слова дали оценку и какие флаги сработали."""
    t_sum, t_match, t_pos, t_neg = _score_part(title)
    b_sum, b_match, b_pos, b_neg = _score_part(text)
    raw = 2 * t_sum + b_sum
    pos, neg = 2 * t_pos + b_pos, 2 * t_neg + b_neg
    score = math.tanh(raw / 3)
    full = normalize(f"{title} {text}")
    flags = []
    if any(p in full for p in PROMO):
        flags.append("реклама")
    if score > 0 and any(m in full for m in SARCASM):
        flags.append("возможный сарказм")
    mixed = pos > 0 and neg < 0 and min(pos, -neg) / max(pos, -neg) >= 0.5
    question = normalize(title).endswith("?")
    if mixed or question:
        flags.append("неоднозначно")
    letters = re.sub(r"[^a-z]", "", normalize(title))
    if len(letters) < 12 or len(re.findall(r"[#$]\w+", title)) >= 4:
        flags.append("вероятный бот")
    label = "positive" if score >= POS_T else "negative" if score <= NEG_T else "neutral"
    if "возможный сарказм" in flags or question or ("неоднозначно" in flags and abs(score) < 0.6):
        label = "neutral"
    return {"score": round(score, 4), "label": label, "matched_terms": t_match + b_match, "flags": flags}


def prepare_posts(raw_posts: list[dict], asset: dict, symbol: str) -> tuple[list[dict], dict]:
    """Фильтрация по активу, удаление дублей и классификация — перед записью в базу."""
    out, seen = [], set()
    stats = {"received": len(raw_posts), "irrelevant": 0, "duplicates": 0, "kept": 0}
    for p in raw_posts:
        if not is_relevant(p["title"], p.get("text", ""), asset, symbol):
            stats["irrelevant"] += 1
            continue
        h = text_hash(p["title"])
        if h in seen:
            stats["duplicates"] += 1
            continue
        seen.add(h)
        s = score_post(p["title"], p.get("text", ""))
        s["flags"] = list(dict.fromkeys(p.get("flags", []) + s["flags"]))
        out.append({**p, "text_hash": h, **s})
    stats["kept"] = len(out)
    return out, stats


def _tokens(title: str) -> set[str]:
    return {w for w in _WORD.findall(normalize(title)) if len(w) > 2 and w not in STOP}


def cluster_events(df: pd.DataFrame, threshold: float = 0.5, max_days: int = 3) -> pd.Series:
    """Группирует публикации об одном событии: похожесть заголовков (Жаккар) >= 0,5 в пределах 3 дней."""
    events: list[dict] = []
    ids = []
    for row in df.sort_values("published_at").itertuples():
        tok, when = _tokens(row.title), pd.Timestamp(row.published_at)
        found = None
        for i in range(len(events) - 1, -1, -1):
            e = events[i]
            if (when - e["last"]).days > max_days:
                break
            union = len(tok | e["tok"])
            if union and len(tok & e["tok"]) / union >= threshold:
                found = i
                break
        if found is None:
            events.append({"tok": tok, "last": when})
            found = len(events) - 1
        else:
            events[found]["last"] = when
        ids.append((row.Index, found))
    return pd.Series(dict(ids)).reindex(df.index)


def weigh(df: pd.DataFrame) -> pd.DataFrame:
    """Вес публикации в индексе.

    Новости 1,0; Reddit 0,6. Реклама, вероятные боты и служебные посты — 0.
    Автор с 5 и более постами за сутки считается ботом. Пересказы одного события
    делят вес: каждая из k публикаций события получает 1/sqrt(k), поэтому одна
    громкая новость не подменяет собой весь информационный фон.
    """
    d = df.copy()
    d["published_at"] = pd.to_datetime(d["published_at"], utc=True)
    d["day"] = d["published_at"].dt.strftime("%Y-%m-%d")
    d["flags"] = d["flags"].apply(list)
    per_author = d[d["author"].notna()].groupby(["author", "day"])["title"].transform("size")
    spam = per_author[per_author >= 5].index
    is_reddit = d["source"].eq("reddit")
    for i in spam:
        if is_reddit.loc[i] and "вероятный бот" not in d.at[i, "flags"]:
            d.at[i, "flags"] = d.at[i, "flags"] + ["вероятный бот"]
    d["event"] = cluster_events(d)
    d["event_size"] = d.groupby("event")["title"].transform("size")
    base = np.where(is_reddit, 0.6, 1.0)
    excluded = d["flags"].apply(lambda f: any(x in f for x in ("реклама", "вероятный бот", "служебный пост")))
    d["weight"] = np.where(excluded, 0.0, base / np.sqrt(d["event_size"]))
    d["sign"] = d["label"].map({"positive": 1, "negative": -1, "neutral": 0}).fillna(0)
    return d


def _index(d: pd.DataFrame) -> float | None:
    w = d["weight"].sum()
    return None if w <= 0 else float((d["weight"] * d["sign"]).sum() / w)


def analyze(posts: pd.DataFrame, fear_greed: pd.Series | None, period_end: str,
            period_start: str | None = None, window_days: int = WINDOW_DAYS) -> dict:
    """posts — публикации из базы; оценка считается по окну window_days до конца периода."""
    end_ts = pd.Timestamp(period_end, tz="UTC") + pd.Timedelta(days=1)
    start_ts = end_ts - pd.Timedelta(days=window_days)
    fg_value = None
    if fear_greed is not None and fear_greed.dropna().size:
        fg = fear_greed.dropna()
        fg = fg[fg.index <= pd.Timestamp(period_end)]
        if len(fg) and (pd.Timestamp(period_end) - fg.index[-1]).days <= 3:
            fg_value = float(fg.iloc[-1])
    empty_details = {"n_posts": 0, "fear_greed": fg_value, "window_days": window_days}
    if posts is None or posts.empty:
        return {"status": "unavailable", "score": None, "details": empty_details, "risks": [],
                "explanation": "Публикаций по активу в базе нет — оценка настроений не рассчитывается."}
    # тональность пересчитывается из текста при каждом расчёте: результат не зависит от того, когда пост попал в базу
    rescored = posts.apply(lambda r: score_post(r["title"], r["text"] or ""), axis=1, result_type="expand")
    all_posts = posts.copy()
    all_posts["score"], all_posts["label"] = rescored["score"], rescored["label"]
    all_posts["matched_terms"] = rescored["matched_terms"]
    all_posts["flags"] = [list(dict.fromkeys([f for f in old if f == "служебный пост"] + new))
                          for old, new in zip(posts["flags"], rescored["flags"])]
    d = weigh(all_posts)
    d = d[d["published_at"] < end_ts]
    win = d[d["published_at"] >= start_ts]
    counted = win[win["weight"] > 0]
    n = len(counted)

    daily = (d[d["weight"] > 0].groupby("day").apply(
        lambda g: pd.Series({"index": _index(g), "n": len(g), "positive": int((g["sign"] > 0).sum()),
                             "negative": int((g["sign"] < 0).sum()), "neutral": int((g["sign"] == 0).sum())}),
        include_groups=False) if len(d[d["weight"] > 0]) else pd.DataFrame())
    if len(daily):
        daily.index = pd.to_datetime(daily.index)
        daily = daily.sort_index()
        daily["index_7d"] = daily["index"].rolling(7, min_periods=1).mean()
        if period_start:
            daily = daily.loc[pd.Timestamp(period_start):]

    counts = {"positive": int((counted["sign"] > 0).sum()), "negative": int((counted["sign"] < 0).sum()),
              "neutral": int((counted["sign"] == 0).sum())}
    flags = {f: int(win["flags"].apply(lambda x: f in x).sum())
             for f in ("реклама", "возможный сарказм", "неоднозначно", "вероятный бот", "служебный пост")}
    by_source = counted.groupby(counted["source"].str.split(":").str[0]).size().to_dict()
    manual = d[d["manual_label"].notna() & (d["manual_label"] != "")]
    manual_check = {"checked": len(manual), "share_of_window": round(len(manual) / max(len(win), 1), 3),
                    "accuracy": round(float((manual["manual_label"] == manual["label"]).mean()), 3) if len(manual) else None}

    events = []
    for _, g in counted.groupby("event"):
        lead = g.sort_values("published_at").iloc[0]
        events.append({"title": lead["title"], "url": lead["url"], "date": lead["day"], "publications": len(g),
                       "label": g["label"].mode().iloc[0], "score": round(float(g["score"].mean()), 3),
                       "impact": round(float(g["weight"].sum() * abs(g["score"].mean())), 3)})
    events = sorted(events, key=lambda e: (e["impact"], e["publications"]), reverse=True)[:8]

    details = {"n_posts": n, "n_window_total": len(win), "counts": counts, "flags": flags, "by_source": by_source,
               "fear_greed": fg_value, "window_days": window_days, "events": events, "manual_check": manual_check,
               "last_post": None if win.empty else win["published_at"].max().strftime("%Y-%m-%dT%H:%M:%SZ"),
               "n_total_in_db": len(d)}
    result_frames = {"posts": win.sort_values("published_at", ascending=False), "daily": daily}
    if n < MIN_POSTS:
        return {"status": "unavailable", "score": None, "details": details, "risks": [], **result_frames,
                "explanation": f"За {window_days} дней учтено {n} публикаций, для оценки нужно не менее {MIN_POSTS}."}
    idx = _index(counted)
    details["post_index"] = round(idx, 4)
    if fg_value is not None:
        score = 100 * (0.8 * idx + 0.2 * (fg_value - 50) / 50)
        formula = (f"SA = 100 × (0,8 × {idx:+.2f} + 0,2 × ({fg_value:.0f} − 50) / 50) = {score:+.0f}")
    else:
        score = 100 * idx
        formula = f"SA = 100 × {idx:+.2f} = {score:+.0f} (индекс Fear & Greed недоступен, оценка только по публикациям)"
    score = round(float(np.clip(score, -100, 100)), 2)
    share = lambda k: counts[k] / n
    risks = []
    if events and events[0]["publications"] / n >= 0.25:
        risks.append(f"Четверть публикаций и более — об одном событии: «{events[0]['title'][:80]}»")
    if n < TARGET_POSTS:
        risks.append(f"Мало публикаций ({n} при целевых {TARGET_POSTS}) — оценка настроений неустойчива")
    if share("negative") >= 0.4:
        risks.append(f"Отрицательных публикаций {share('negative'):.0%}")
    explanation = (f"{formula}. Учтено {n} публикаций за {window_days} дней: положительных {share('positive'):.0%}, "
                   f"отрицательных {share('negative'):.0%}, нейтральных {share('neutral'):.0%}. "
                   f"Индекс публикаций — взвешенная разность долей положительных и отрицательных."
                   + (f" Fear & Greed = {fg_value:.0f} отражает весь рынок, поэтому его вес 20%." if fg_value is not None else ""))
    return {"status": "ok", "score": score, "details": details, "risks": risks, "explanation": explanation,
            **result_frames}
