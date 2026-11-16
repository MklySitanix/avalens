"""Проверка правильности математических расчётов: значения посчитаны вручную (см. протокол тестирования)."""
import numpy as np
import pandas as pd
import pytest

from analysis import integral, technical as t

PRICES = pd.Series([10, 11, 10.5, 11.5, 12, 11], dtype=float)


def test_sma_manual():
    # (10+11+10.5)/3 = 10.5; (11+10.5+11.5)/3 = 11; (10.5+11.5+12)/3 = 11.3333; (11.5+12+11)/3 = 11.5
    got = t.sma(PRICES, 3).dropna().tolist()
    assert got == pytest.approx([10.5, 11.0, 11.3333, 11.5], abs=1e-4)


def test_ema_manual():
    # alpha = 2/(3+1) = 0.5; старт = SMA(3) = 10.5; далее 0.5*11.5+0.5*10.5 = 11.0; 11.5; 11.25
    got = t.ema(PRICES, 3).dropna().tolist()
    assert got == pytest.approx([10.5, 11.0, 11.5, 11.25], abs=1e-9)


def test_rsi_manual():
    # изменения: +1, -0.5, +1, +0.5, -1; период 3
    # шаг 1: рост (1+0+1)/3 = 0.6667, падение 0.5/3 = 0.1667, RS = 4 -> RSI = 80
    # шаг 2: рост (0.6667*2+0.5)/3 = 0.6111, падение (0.1667*2+0)/3 = 0.1111, RS = 5.5 -> RSI = 84.615
    # шаг 3: рост (0.6111*2+0)/3 = 0.4074, падение (0.1111*2+1)/3 = 0.4074, RS = 1 -> RSI = 50
    got = t.rsi(PRICES, 3).dropna().tolist()
    assert got == pytest.approx([80.0, 84.6154, 50.0], abs=1e-3)


def test_rsi_bounds_and_flat():
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert t.rsi(up).iloc[-1] == 100.0
    assert t.rsi(pd.Series([5.0] * 40)).iloc[-1] == 50.0


def test_bollinger_manual():
    # окно [10, 11, 10.5]: среднее 10.5, ст. отклонение sqrt((0.25+0.25+0)/3) = 0.408248
    bb = t.bollinger(PRICES, 3, 2).iloc[2]
    assert bb["bb_mid"] == pytest.approx(10.5)
    assert bb["bb_upper"] == pytest.approx(10.5 + 2 * 0.408248, abs=1e-5)
    assert bb["bb_lower"] == pytest.approx(10.5 - 2 * 0.408248, abs=1e-5)


def test_macd_is_difference_of_emas():
    rng = np.random.default_rng(0)
    s = pd.Series(50 + np.cumsum(rng.normal(0, 1, 120)))
    m = t.macd(s)
    assert (m["macd"] - (t.ema(s, 12) - t.ema(s, 26))).abs().max() < 1e-12
    assert (m["hist"] - (m["macd"] - m["signal"])).abs().max() < 1e-12


def test_indicators_do_not_use_future_data():
    rng = np.random.default_rng(1)
    s = pd.Series(50 + np.cumsum(rng.normal(0, 1, 150)))
    full, cut = t.rsi(s), t.rsi(s.iloc[:100])
    assert full.iloc[99] == pytest.approx(cut.iloc[99])
    assert t.macd(s)["signal"].iloc[99] == pytest.approx(t.macd(s.iloc[:100])["signal"].iloc[99])


def test_integral_formula_and_scale():
    mods = {k: {"score": v, "quality": 1.0, "status": "ok"} for k, v in {"TA": 40, "FA": 20, "ON": 0, "SA": -10}.items()}
    r = integral.combine(mods)
    # 0.30*40 + 0.25*20 + 0.25*0 + 0.20*(-10) = 15
    assert r["s_score"] == pytest.approx(15.0)
    assert r["confidence"] == pytest.approx(100.0)
    assert [integral.interpret(x)[0] for x in (30, 29.9, 10, 9.9, -9.9, -10, -29.9, -30)] == [0, 1, 1, 2, 2, 3, 3, 4]


def test_integral_weights_renormalised_when_module_missing():
    mods = {"TA": {"score": 40, "quality": 1, "status": "ok"}, "FA": {"score": 20, "quality": 1, "status": "ok"},
            "ON": {"score": None, "quality": 0, "status": "unavailable"}, "SA": {"score": -10, "quality": 1, "status": "ok"}}
    r = integral.combine(mods)
    # (0.30*40 + 0.25*20 + 0.20*(-10)) / 0.75 = 20
    assert r["s_score"] == pytest.approx(20.0)
    assert r["confidence"] == pytest.approx(75.0)
