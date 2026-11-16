"""Общий сценарий запроса к внешнему источнику (схема из раздела 8 архитектуры).

GET по HTTPS с тайм-аутом 10 с -> при тайм-ауте, кодах 429 и 5xx до 3 повторов
с паузами 1, 2, 4 с -> проверка формата ответа. Каждый ответ и каждая ошибка
возвращаются вызывающему коду и записываются им в raw_responses.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

TIMEOUT = 10
PAUSES = (1, 2, 4)
USER_AGENT = "AvaLens/1.0 (educational research project)"

# Точки подмены для тестов: тест заменяет _get и _sleep, сеть не нужна.
_get = requests.get
_sleep = time.sleep


@dataclass
class Fetch:
    """Результат одного запроса."""
    source: str
    url: str
    ok: bool
    status: int | None = None
    error: str | None = None
    text: str | None = None
    data: Any = None
    attempts: int = 1
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))


@dataclass
class SourceResult:
    """Результат работы адаптера: статус, сырые ответы и разобранные таблицы."""
    source: str
    ok: bool
    error: str | None = None
    fetches: list[Fetch] = field(default_factory=list)
    frames: dict = field(default_factory=dict)
    note: str | None = None


def http_get(source: str, url: str, params: dict | None = None, kind: str = "json",
             timeout: int = TIMEOUT, pauses: tuple = PAUSES) -> Fetch:
    """Запрос с повторами. kind: 'json' — ответ должен разбираться как JSON, 'text' — непустой текст."""
    last = Fetch(source, url, ok=False, error="запрос не выполнялся")
    full_url = url
    for attempt in range(len(pauses) + 1):
        try:
            resp = _get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            full_url = getattr(resp, "url", url) or url
            status = resp.status_code
            if status == 200:
                text = resp.text
                if not text or not text.strip():
                    last = Fetch(source, full_url, False, status, "пустой ответ", attempts=attempt + 1)
                elif kind == "json":
                    try:
                        return Fetch(source, full_url, True, status, None, text, json.loads(text), attempt + 1)
                    except ValueError:
                        last = Fetch(source, full_url, False, status, "ответ не является JSON",
                                     text[:2000], attempts=attempt + 1)
                else:
                    return Fetch(source, full_url, True, status, None, text, text, attempt + 1)
            elif status == 429 or status >= 500:
                last = Fetch(source, full_url, False, status, f"HTTP {status}", attempts=attempt + 1)
            else:  # прочие 4xx повторять бессмысленно
                return Fetch(source, full_url, False, status, f"HTTP {status}",
                             (resp.text or "")[:2000], attempts=attempt + 1)
        except requests.Timeout:
            last = Fetch(source, full_url, False, None, f"тайм-аут {timeout} с", attempts=attempt + 1)
        except requests.RequestException as exc:
            last = Fetch(source, full_url, False, None, f"сеть недоступна: {type(exc).__name__}",
                         attempts=attempt + 1)
        if attempt < len(pauses):
            _sleep(pauses[attempt])
    return last


def to_date(ts: float, unit: str = "s") -> str:
    """Unix-время -> дата UTC 'YYYY-MM-DD'."""
    if unit == "ms":
        ts = ts / 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
