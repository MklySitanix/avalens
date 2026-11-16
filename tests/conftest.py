import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import pipeline  # noqa: E402
from storage.db import Database  # noqa: E402
from tests.fake_api import FakeAPI, install  # noqa: E402

TODAY = "2026-11-16"
START, END = "2026-05-20", "2026-11-15"


@pytest.fixture
def cfg():
    return pipeline.load_config()


@pytest.fixture
def make_env(tmp_path, cfg, monkeypatch):
    """Создаёт базу и имитацию API с заданным сценарием; возвращает (db, api, отчёт загрузки)."""
    from datetime import datetime, timezone
    fixed = datetime.strptime(TODAY, "%Y-%m-%d").replace(hour=12, tzinfo=timezone.utc)
    monkeypatch.setattr(pipeline, "utc_now", lambda: fixed)  # «сегодня» в тестах не зависит от даты запуска

    def _make(name="t", **scenario):
        api = FakeAPI(today=TODAY, **scenario)
        install(api)
        db = Database(tmp_path / f"{name}.db")
        report = pipeline.refresh(db, cfg)
        return db, api, report
    return _make
