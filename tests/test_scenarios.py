"""Восемь тестовых ситуаций из задания (этап V). Сеть не используется: источники заменены имитацией."""
import pandas as pd

from analysis import sentiment
from core import pipeline
from data import quality
from tests.conftest import END, START


def run(db, cfg, symbol="AVAX", save=False):
    return pipeline.analyze(db, cfg, symbol, START, END, save=save)


def test_1_correct_market_data(make_env, cfg):
    """Корректные рыночные данные: успешная загрузка и расчёт индикаторов."""
    db, _, report = make_env()
    assert all(r["ok"] for r in report)
    res = run(db, cfg)
    ta = res["modules"]["TA"]
    assert ta["status"] == "ok" and -100 <= ta["score"] <= 100
    assert 0 <= ta["details"]["rsi"] <= 100
    ind = res["frames"]["indicators"]
    assert ind[["sma20", "sma50", "rsi14", "macd", "signal", "bb_upper"]].iloc[-1].notna().all()
    assert all(m["status"] == "ok" for m in res["modules"].values())
    assert res["integral"]["s_score"] is not None and res["integral"]["confidence"] >= 70


def test_2_missing_period(make_env, cfg):
    """Пропуск данных: короткий заполняется и помечается, полнота и достоверность снижаются."""
    db, _, _ = make_env(price_gap=(20, 3))
    res = run(db, cfg)
    assert res["price_report"]["filled_days"] == 3
    assert res["frames"]["prices"]["is_filled"].sum() == 3
    assert res["modules"]["TA"]["quality_parts"]["completeness"] < 1
    assert any("пропущенных дней" in w for w in res["warnings"])
    assert res["modules"]["TA"]["status"] == "ok"


def test_2b_long_gap_cuts_series(make_env, cfg):
    """Длинный пропуск не заполняется: используются только данные после разрыва."""
    db, _, _ = make_env(price_gap=(100, 10))
    res = run(db, cfg)
    assert res["price_report"]["cut_before"] is not None
    assert any("разрыв" in w for w in res["warnings"])


def test_3_api_unavailable(make_env, cfg):
    """Недоступность API: сообщение об ошибке, остальные источники работают, цена берётся из резерва."""
    db, api, report = make_env(down=("binance",))
    failed = [r for r in report if not r["ok"]]
    assert {r["source"] for r in failed} == {"binance"}
    assert all("503" in r["error"] for r in failed)
    assert sum("binance" in u for u in api.calls) >= 8          # 4 попытки на каждый из адресов
    status = db.source_status().set_index("source")
    assert status.loc["binance", "errors"] > 0 and pd.isna(status.loc["binance", "last_success"])
    res = run(db, cfg)
    assert res["modules"]["TA"]["quality_parts"]["source"] == "coingecko"
    assert any("резервного источника" in w for w in res["warnings"])
    assert res["modules"]["TA"]["quality_parts"]["reliability"] == cfg["reliability"]["coingecko"]


def test_3b_timeout_and_all_sources_down(make_env, cfg):
    """Все источники недоступны: расчёт не падает, вывод не формируется."""
    db, _, report = make_env(timeout_hosts=("binance", "coingecko", "avax", "llama", "alternative", "reddit", "google",
                                            "cointelegraph", "coindesk", "decrypt", "cryptoslate"))
    assert not any(r["ok"] for r in report)
    assert any("тайм-аут" in (r["error"] or "") for r in report)
    assert len(db.load_prices("AVAX")) == 0


def test_4_stale_data(make_env, cfg):
    """Устаревшие данные: предупреждение пользователя и отказ от уверенного вывода."""
    db, _, _ = make_env(stale_days=6)
    res = run(db, cfg)
    assert any("устарели" in w for w in res["warnings"])
    assert res["modules"]["TA"]["quality_parts"]["freshness"] == 0
    assert res["integral"]["s_score"] is None
    assert res["integral"]["verdict"] == "Недостаточно данных для комплексной оценки"


def test_5_anomaly(make_env, cfg):
    """Аномальное значение: выброс найден, исключён из расчёта и показан пользователю."""
    db, _, _ = make_env(price_spike_days_ago=10)
    res = run(db, cfg)
    assert len(res["price_report"]["anomalies"]) == 1
    assert any("аномальные" in w for w in res["warnings"])
    prices = res["frames"]["prices"]
    assert prices["is_anomaly"].sum() == 1
    assert prices["close"].pct_change().abs().max() < 0.5       # выброс x4 не попал в ряд
    clean_db, _, _ = make_env(name="clean")
    assert abs(res["modules"]["TA"]["details"]["rsi"] - run(clean_db, cfg)["modules"]["TA"]["details"]["rsi"]) < 5


def test_6_no_onchain_data(make_env, cfg):
    """Нет ончейн-данных: некорректный расчёт исключён, оценка считается по трём модулям."""
    db, _, report = make_env(down=("metrics.avax.network", "llama.fi"))
    assert not next(r for r in report if r["source"] == "avalanche_metrics")["ok"]
    res = run(db, cfg)
    on = res["modules"]["ON"]
    assert on["status"] == "unavailable" and on["score"] is None
    row = next(r for r in res["integral"]["modules"] if r["module"] == "ON")
    assert row["state"] == "недоступен"
    assert "ON" not in res["integral"]["contributions"]
    assert res["integral"]["confidence"] <= 75


def test_7_conflict_between_modules(make_env, cfg):
    """Противоречие между модулями: цена растёт, активность сети падает — конфликт показан с объяснением."""
    db, _, _ = make_env(price_drift=0.002, activity_drift=-0.004)
    res = run(db, cfg)
    assert res["modules"]["TA"]["score"] >= 20 and res["modules"]["ON"]["score"] <= -20
    pairs = {(c["module_a"], c["module_b"]) for c in res["integral"]["conflicts"]}
    assert ("TA", "ON") in pairs
    conflict = next(c for c in res["integral"]["conflicts"] if (c["module_a"], c["module_b"]) == ("TA", "ON"))
    assert "не подтверждён использованием сети" in conflict["explanation"]
    assert res["integral"]["summary"].startswith("Сигналы противоречат друг другу")


def test_8_reproducibility(make_env, cfg):
    """Повторный расчёт за тот же период: тот же результат и та же контрольная сумма."""
    db, _, _ = make_env()
    a, b = run(db, cfg, save=True), run(db, cfg, save=True)
    assert a["input_hash"] == b["input_hash"]
    assert a["integral"]["s_score"] == b["integral"]["s_score"]
    assert a["integral"]["confidence"] == b["integral"]["confidence"]
    assert {k: m["score"] for k, m in a["modules"].items()} == {k: m["score"] for k, m in b["modules"].items()}
    runs = db.list_runs("AVAX")
    assert len(runs) == 2 and runs["input_hash"].nunique() == 1
    saved = db.load_run(int(runs["run_id"].iloc[0]))
    assert set(saved["modules"]) == {"TA", "FA", "ON", "SA"}


def test_price_divergence_lowers_reliability(make_env, cfg):
    """Противоречивые данные: цены двух источников расходятся более чем на 2%."""
    db, _, _ = make_env(cg_price_factor=1.05)
    res = run(db, cfg)
    assert res["modules"]["TA"]["quality_parts"]["reliability"] == 0.7
    assert any("Противоречивые данные" in w for w in res["warnings"])


def test_sentiment_rules():
    asset = {"name": "Avalanche", "keywords": ["avalanche", "avax"], "exclude_words": ["snow", "ski", "skiers"]}
    assert sentiment.score_post("Avalanche surges as adoption grows")["label"] == "positive"
    assert sentiment.score_post("AVAX crashes after exploit")["label"] == "negative"
    assert sentiment.score_post("Avalanche community call scheduled")["label"] == "neutral"
    assert sentiment.score_post("Avalanche is not dead")["score"] > 0                      # отрицание
    assert "возможный сарказм" in sentiment.score_post("AVAX to the moon lol, great job devs")["flags"]
    assert sentiment.score_post("Will AVAX recover?")["label"] == "neutral"                 # вопрос
    assert "реклама" in sentiment.score_post("Claim your AVAX airdrop giveaway now")["flags"]
    assert not sentiment.is_relevant("Avalanche kills two skiers in mountain snow", "", asset, "AVAX")
    assert sentiment.is_relevant("Avalanche network upgrade goes live", "", asset, "AVAX")
    posts, stats = sentiment.prepare_posts(
        [{"title": "AVAX rallies", "source": "news:x", "published_at": "2026-11-10T00:00:00Z"},
         {"title": "AVAX rallies!", "source": "news:y", "published_at": "2026-11-10T01:00:00Z"},
         {"title": "Bitcoin steady", "source": "news:x", "published_at": "2026-11-10T00:00:00Z"}], asset, "AVAX")
    assert stats == {"received": 3, "irrelevant": 1, "duplicates": 1, "kept": 1}


def test_big_news_does_not_dominate():
    """Девять пересказов одной новости весят как три публикации."""
    rows = [{"title": f"Avalanche surges as institutional adoption grows outlet{i}", "text": "", "source": "news:a",
             "published_at": "2026-11-10T00:00:00Z", "author": f"a{i}", "label": "positive", "score": 0.9,
             "flags": [], "manual_label": None} for i in range(9)]
    w = sentiment.weigh(pd.DataFrame(rows))
    assert w["event"].nunique() == 1 and abs(w["weight"].sum() - 3.0) < 1e-9


def test_freshness_function():
    assert quality.freshness(10, 24) == 1 and quality.freshness(72, 24) == 0
    assert quality.freshness(48, 24) == 0.5 and quality.freshness(None, 24) == 0


def test_stale_metric_excluded_from_score(make_env, cfg):
    """Показатель, который источник перестал обновлять, исключается из оценки, а не искажает её."""
    db, _, _ = make_env(staking_stale_days=77)
    res = run(db, cfg)
    rows = {r["metric"]: r for r in res["modules"]["ON"]["details"]["metrics"]}
    assert not rows["validatorCount"]["available"] and "устарели" in rows["validatorCount"]["note"]
    assert res["modules"]["ON"]["status"] == "ok" and res["modules"]["ON"]["details"]["n_available"] == 4
    assert res["modules"]["ON"]["quality_parts"]["completeness"] < 0.7
    assert any("не обновляется источником" in w for w in res["warnings"])


def test_unfinished_day_is_not_stored(make_env, cfg):
    """Значения за текущие незавершённые сутки в базу не попадают."""
    db, _, _ = make_env()
    assert db.load_prices("AVAX")["date"].max() < pd.Timestamp("2026-11-16")
    assert db.load_metrics("AVAX")["date"].max() < pd.Timestamp("2026-11-16")
