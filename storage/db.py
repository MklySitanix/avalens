"""Модуль хранения данных: один файл SQLite, 9 таблиц трёх слоёв.

Сырые ответы (raw_responses) -> обработанные ряды (prices, onchain_metrics,
fundamentals, posts) -> результаты расчётов (analysis_runs, module_results, conflicts).
Результаты только добавляются: повторный расчёт создаёт новую строку analysis_runs.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "avalens.db"
MAX_RAW_BYTES = 1_500_000  # крупные ответы хранятся усечёнными

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    symbol TEXT PRIMARY KEY, name TEXT, binance_pair TEXT, coingecko_id TEXT,
    llama_chain TEXT, chain_id TEXT, keywords TEXT);
CREATE TABLE IF NOT EXISTS raw_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, url TEXT, fetched_at TEXT,
    http_status INTEGER, ok INTEGER, error TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT, date TEXT, source TEXT, open REAL, high REAL, low REAL, close REAL,
    volume REAL, is_anomaly INTEGER DEFAULT 0, is_filled INTEGER DEFAULT 0,
    PRIMARY KEY (symbol, date, source));
CREATE TABLE IF NOT EXISTS onchain_metrics (
    symbol TEXT, metric TEXT, date TEXT, value REAL, source TEXT, is_anomaly INTEGER DEFAULT 0,
    PRIMARY KEY (symbol, metric, date));
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol TEXT, snapshot_at TEXT, market_cap REAL, fdv REAL, circulating REAL,
    total_supply REAL, max_supply REAL, price REAL, extra TEXT, source TEXT,
    PRIMARY KEY (symbol, snapshot_at));
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, source TEXT, url TEXT,
    published_at TEXT, author TEXT, title TEXT, text TEXT, text_hash TEXT,
    label TEXT, score REAL, matched_terms TEXT, flags TEXT, manual_label TEXT,
    UNIQUE (symbol, text_hash));
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, period_start TEXT, period_end TEXT,
    created_at TEXT, algo_version TEXT, params TEXT, s_score REAL, confidence REAL,
    verdict TEXT, input_hash TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS module_results (
    run_id INTEGER, module TEXT, score REAL, quality REAL, status TEXT, details TEXT,
    explanation TEXT, PRIMARY KEY (run_id, module));
CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, module_a TEXT, module_b TEXT,
    rule TEXT, severity TEXT, explanation TEXT);
CREATE INDEX IF NOT EXISTS idx_raw_source ON raw_responses (source, fetched_at);
CREATE INDEX IF NOT EXISTS idx_posts_time ON posts (symbol, published_at);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Database:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------- справочник ----------
    def upsert_assets(self, assets: dict) -> None:
        rows = [(s, a.get("name"), a.get("binance_pair"), a.get("coingecko_id"), a.get("llama_chain"),
                 a.get("chain_id"), json.dumps(a.get("keywords", []), ensure_ascii=False))
                for s, a in assets.items()]
        self.conn.executemany("INSERT OR REPLACE INTO assets VALUES (?,?,?,?,?,?,?)", rows)
        self.conn.commit()

    # ---------- сырые ответы ----------
    def save_raw(self, source: str, url: str, http_status: int | None, ok: bool,
                 error: str | None, payload: str | None, fetched_at: str | None = None) -> None:
        if payload is not None and len(payload) > MAX_RAW_BYTES:
            payload = payload[:MAX_RAW_BYTES] + f"\n...[усечено, всего {len(payload)} символов]"
        self.conn.execute(
            "INSERT INTO raw_responses (source, url, fetched_at, http_status, ok, error, payload) "
            "VALUES (?,?,?,?,?,?,?)",
            (source, url, fetched_at or now_iso(), http_status, int(ok), error, payload))
        self.conn.commit()

    def source_status(self) -> pd.DataFrame:
        """Последний запрос и последний успешный запрос по каждому источнику."""
        q = """
        SELECT source,
               MAX(fetched_at) AS last_attempt,
               MAX(CASE WHEN ok=1 THEN fetched_at END) AS last_success,
               SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) AS errors,
               COUNT(*) AS requests
        FROM raw_responses GROUP BY source ORDER BY source"""
        df = pd.read_sql_query(q, self.conn)
        last_err = pd.read_sql_query(
            "SELECT source, error FROM raw_responses r WHERE id IN "
            "(SELECT MAX(id) FROM raw_responses GROUP BY source)", self.conn)
        return df.merge(last_err.rename(columns={"error": "last_error"}), on="source", how="left")

    def raw_list(self, source: str | None = None, limit: int = 50) -> pd.DataFrame:
        q = ("SELECT id, source, url, fetched_at, http_status, ok, error, LENGTH(payload) AS size "
             "FROM raw_responses")
        args: tuple = ()
        if source:
            q += " WHERE source=?"
            args = (source,)
        q += " ORDER BY id DESC LIMIT ?"
        return pd.read_sql_query(q, self.conn, params=args + (limit,))

    def raw_payload(self, raw_id: int) -> str | None:
        row = self.conn.execute("SELECT payload FROM raw_responses WHERE id=?", (raw_id,)).fetchone()
        return row["payload"] if row else None

    # ---------- цены ----------
    def upsert_prices(self, symbol: str, source: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        rows = [(symbol, str(r.date)[:10], source, r.open, r.high, r.low, r.close, r.volume)
                for r in df.itertuples()]
        self.conn.executemany(
            "INSERT OR REPLACE INTO prices (symbol, date, source, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def load_prices(self, symbol: str, source: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Дневные свечи. Без указания источника берётся основной (Binance), иначе резервный."""
        if source is None:
            have = [r[0] for r in self.conn.execute(
                "SELECT source, COUNT(*) c FROM prices WHERE symbol=? GROUP BY source ORDER BY c DESC",
                (symbol,))]
            if not have:
                return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "source"])
            source = "binance" if "binance" in have else have[0]
        q = "SELECT date, open, high, low, close, volume, source FROM prices WHERE symbol=? AND source=?"
        args = [symbol, source]
        if end:
            q += " AND date<=?"
            args.append(end)
        df = pd.read_sql_query(q + " ORDER BY date", self.conn, params=args)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def mark_price_flags(self, symbol: str, source: str, anomalies: list[str]) -> None:
        self.conn.execute("UPDATE prices SET is_anomaly=0 WHERE symbol=? AND source=?", (symbol, source))
        self.conn.executemany(
            "UPDATE prices SET is_anomaly=1 WHERE symbol=? AND source=? AND date=?",
            [(symbol, source, d) for d in anomalies])
        self.conn.commit()

    # ---------- метрики (ончейн и рыночные индексы) ----------
    def upsert_metrics(self, symbol: str, df: pd.DataFrame) -> int:
        """df: metric, date, value, source."""
        if df is None or df.empty:
            return 0
        rows = [(symbol, r.metric, str(r.date)[:10], float(r.value), r.source) for r in df.itertuples()]
        self.conn.executemany(
            "INSERT OR REPLACE INTO onchain_metrics (symbol, metric, date, value, source) VALUES (?,?,?,?,?)",
            rows)
        self.conn.commit()
        return len(rows)

    def load_metrics(self, symbol: str, end: str | None = None) -> pd.DataFrame:
        q = "SELECT metric, date, value, source FROM onchain_metrics WHERE symbol=?"
        args = [symbol]
        if end:
            q += " AND date<=?"
            args.append(end)
        df = pd.read_sql_query(q + " ORDER BY metric, date", self.conn, params=args)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ---------- фундаментальные снимки ----------
    def insert_fundamentals(self, symbol: str, snap: dict, source: str, snapshot_at: str | None = None) -> None:
        known = {k: snap.get(k) for k in ("market_cap", "fdv", "circulating", "total_supply", "max_supply", "price")}
        extra = {k: v for k, v in snap.items() if k not in known}
        self.conn.execute(
            "INSERT OR REPLACE INTO fundamentals VALUES (?,?,?,?,?,?,?,?,?,?)",
            (symbol, snapshot_at or now_iso(), known["market_cap"], known["fdv"], known["circulating"],
             known["total_supply"], known["max_supply"], known["price"],
             json.dumps(extra, ensure_ascii=False), source))
        self.conn.commit()

    def load_fundamentals(self, symbol: str, as_of: str | None = None) -> dict | None:
        """Последний снимок не позже as_of (ISO-время)."""
        q = "SELECT * FROM fundamentals WHERE symbol=?"
        args = [symbol]
        if as_of:
            q += " AND snapshot_at<=?"
            args.append(as_of)
        row = self.conn.execute(q + " ORDER BY snapshot_at DESC LIMIT 1", args).fetchone()
        if not row:
            return None
        d = dict(row)
        d.update(json.loads(d.pop("extra") or "{}"))
        return d

    def fundamentals_history(self, symbol: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT snapshot_at, market_cap, fdv, circulating, total_supply, max_supply, price, source "
            "FROM fundamentals WHERE symbol=? ORDER BY snapshot_at", self.conn, params=(symbol,))

    # ---------- публикации ----------
    def upsert_posts(self, symbol: str, posts: list[dict]) -> int:
        """Добавляет новые публикации; дубли по text_hash пропускаются. Возвращает число новых."""
        n = 0
        for p in posts:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO posts (symbol, source, url, published_at, author, title, text, "
                "text_hash, label, score, matched_terms, flags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (symbol, p["source"], p.get("url"), p["published_at"], p.get("author"), p["title"],
                 p.get("text", ""), p["text_hash"], p.get("label"), p.get("score"),
                 json.dumps(p.get("matched_terms", []), ensure_ascii=False),
                 json.dumps(p.get("flags", []), ensure_ascii=False)))
            n += cur.rowcount
        self.conn.commit()
        return n

    def load_posts(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM posts WHERE symbol=?"
        args = [symbol]
        if start:
            q += " AND published_at>=?"
            args.append(start)
        if end:
            q += " AND published_at<?"
            args.append(end)
        df = pd.read_sql_query(q + " ORDER BY published_at", self.conn, params=args)
        for col in ("matched_terms", "flags"):
            df[col] = df[col].apply(lambda s: json.loads(s) if s else [])
        return df

    def update_post_labels(self, rows: list[tuple]) -> None:
        """rows: (label, score, matched_terms_json, flags_json, id) — пересчёт тональности."""
        self.conn.executemany("UPDATE posts SET label=?, score=?, matched_terms=?, flags=? WHERE id=?", rows)
        self.conn.commit()

    def set_manual_label(self, post_id: int, label: str | None) -> None:
        self.conn.execute("UPDATE posts SET manual_label=? WHERE id=?", (label, post_id))
        self.conn.commit()

    # ---------- результаты ----------
    def save_run(self, result: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO analysis_runs (symbol, period_start, period_end, created_at, algo_version, params, "
            "s_score, confidence, verdict, input_hash, summary) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (result["symbol"], result["period_start"], result["period_end"], now_iso(),
             result["algo_version"], json.dumps(result["params"], ensure_ascii=False),
             result["integral"]["s_score"], result["integral"]["confidence"],
             result["integral"]["verdict"], result["input_hash"], result["integral"]["summary"]))
        run_id = cur.lastrowid
        for name, m in result["modules"].items():
            self.conn.execute(
                "INSERT INTO module_results VALUES (?,?,?,?,?,?,?)",
                (run_id, name, m.get("score"), m.get("quality"), m.get("status"),
                 json.dumps(m.get("details", {}), ensure_ascii=False, default=str), m.get("explanation")))
        for c in result["integral"]["conflicts"]:
            self.conn.execute(
                "INSERT INTO conflicts (run_id, module_a, module_b, rule, severity, explanation) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, c["module_a"], c["module_b"], c["rule"], c["severity"], c["explanation"]))
        self.conn.commit()
        return run_id

    def list_runs(self, symbol: str | None = None, limit: int = 200) -> pd.DataFrame:
        q = ("SELECT run_id, symbol, period_start, period_end, created_at, algo_version, s_score, "
             "confidence, verdict, input_hash FROM analysis_runs")
        args: tuple = ()
        if symbol:
            q += " WHERE symbol=?"
            args = (symbol,)
        return pd.read_sql_query(q + " ORDER BY run_id DESC LIMIT ?", self.conn, params=args + (limit,))

    def load_run(self, run_id: int) -> dict | None:
        run = self.conn.execute("SELECT * FROM analysis_runs WHERE run_id=?", (run_id,)).fetchone()
        if not run:
            return None
        out = dict(run)
        out["params"] = json.loads(out["params"] or "{}")
        out["modules"] = {}
        for r in self.conn.execute("SELECT * FROM module_results WHERE run_id=?", (run_id,)):
            d = dict(r)
            d["details"] = json.loads(d["details"] or "{}")
            out["modules"][d["module"]] = d
        out["conflicts"] = [dict(r) for r in self.conn.execute(
            "SELECT * FROM conflicts WHERE run_id=?", (run_id,))]
        return out
