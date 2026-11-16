"""Модуль фундаментального анализа: 8 направлений, каждое оценивается от -2 до +2.

Каждая оценка получена правилом из данных и сопровождается экономическим следствием
для токена. Направление без данных получает оценку None («недостаточно данных») и в
среднее не входит. FA = 100 * сумма баллов / (2 * n), n — число оценённых направлений;
при n < 5 модуль считается недоступным.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

MIN_CRITERIA = 5
FACTS_MAX_AGE_DAYS = 180


def _crit(key, title, score, facts, rule, consequence):
    return {"key": key, "title": title, "score": score, "facts": facts, "rule": rule, "consequence": consequence}


def _sum_tail(s: pd.Series | None, n: int) -> float | None:
    if s is None or s.dropna().shape[0] < n * 0.8:
        return None
    return float(s.dropna().tail(n).sum())


def _change(s: pd.Series | None, n: int) -> float | None:
    if s is None:
        return None
    s = s.dropna()
    if len(s) <= n or s.iloc[-1 - n] <= 0:
        return None
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1)


def analyze(snapshot: dict | None, facts: dict | None, metrics: dict[str, pd.Series],
            peer: dict | None = None, as_of: str | None = None) -> dict:
    """snapshot — рыночный снимок (CoinGecko + DefiLlama), facts — справочный YAML проекта,
    metrics — очищенные ряды (tvl, fees_usd, validatorCount), peer — показатели актива сравнения."""
    snapshot, facts, peer = snapshot or {}, facts or {}, peer or {}
    mc, fdv = snapshot.get("market_cap"), snapshot.get("fdv")
    circ, max_supply = snapshot.get("circulating"), snapshot.get("max_supply") or facts.get("max_supply")
    total = snapshot.get("total_supply")
    price = snapshot.get("price")
    tvl = metrics.get("tvl")
    fees = metrics.get("fees_usd")
    tvl_last = float(tvl.dropna().iloc[-1]) if tvl is not None and tvl.dropna().size else None
    criteria, risks, warnings = [], [], []

    # возраст справочных фактов
    if facts.get("verified_at") and as_of:
        age = (date.fromisoformat(as_of[:10]) - date.fromisoformat(str(facts["verified_at"]))).days
        if age > FACTS_MAX_AGE_DAYS:
            warnings.append(f"Справочные факты о проекте проверялись {age} дней назад — требуется проверка.")

    # 1. экономическая модель
    demand = facts.get("demand_sources") or []
    if demand:
        n = len(demand)
        score = 2 if n >= 3 else 1 if n == 2 else 0
        criteria.append(_crit(
            "model", "Экономическая модель", score,
            {"назначение": facts.get("token_purpose"), "источники спроса": [d["name"] for d in demand]},
            "3 и более независимых источника спроса: +2; два: +1; один: 0",
            f"У токена {n} независимых источника спроса: падение одного вида активности не обнуляет "
            f"потребность в токене." if n >= 2 else
            "Спрос на токен зависит от одного вида активности — его спад прямо бьёт по цене."))
    else:
        criteria.append(_crit("model", "Экономическая модель", None, {}, "нет справочных данных",
                              "Недостаточно данных для оценки."))

    # 2. токеномика: сколько эмиссии ещё впереди
    if circ and max_supply:
        share = circ / max_supply
        score = 2 if share >= 0.8 else 1 if share >= 0.6 else 0 if share >= 0.4 else -1
        criteria.append(_crit(
            "supply", "Токеномика", score,
            {"в обращении": circ, "общее предложение": total, "максимум": max_supply, "доля в обращении": round(share, 4)},
            "доля в обращении от максимума: от 80% +2; от 60% +1; от 40% 0; меньше −1",
            f"В обращении {share:.0%} максимальной эмиссии: будущий выпуск может увеличить предложение "
            f"ещё на {1 / share - 1:.0%}, это потолок размывания доли держателя."))
    elif circ and total:
        share = circ / total
        score = 1 if share >= 0.95 else 0
        criteria.append(_crit(
            "supply", "Токеномика", score,
            {"в обращении": circ, "общее предложение": total, "максимум": None},
            "максимума эмиссии нет: при доле в обращении от 95% выпущенного +1, иначе 0",
            "Жёсткого предела эмиссии нет: предложение ограничивает только текущая политика выпуска и сжигания, "
            "зато почти все выпущенные токены уже на рынке и давления разблокировок нет."))
    else:
        criteria.append(_crit("supply", "Токеномика", None, {}, "нет данных о предложении",
                              "Недостаточно данных для оценки."))

    # 3. рыночная оценка: FDV/MC и капитализация к TVL
    if mc and fdv:
        ratio = fdv / mc
        s1 = 1 if ratio <= 1.1 else 0 if ratio <= 1.5 else -1 if ratio <= 2.5 else -2
        facts3 = {"Market Cap": mc, "FDV": fdv, "FDV/MC": round(ratio, 3)}
        text = (f"FDV превышает капитализацию в {ratio:.2f} раза: "
                + ("почти вся оценка уже обеспечена токенами в обращении." if ratio <= 1.1 else
                   "часть оценки приходится на ещё не выпущенные токены, их выход будет давить на цену."))
        s2 = 0
        if tvl_last:
            mctvl = mc / tvl_last
            facts3["MC/TVL"] = round(mctvl, 2)
            if peer.get("market_cap") and peer.get("tvl"):
                peer_ratio = peer["market_cap"] / peer["tvl"]
                facts3[f"MC/TVL {peer.get('symbol', 'сравнение')}"] = round(peer_ratio, 2)
                s2 = 1 if mctvl < 0.8 * peer_ratio else -1 if mctvl > 1.25 * peer_ratio else 0
                text += (f" На доллар средств в протоколах сети приходится {mctvl:.1f} долл. капитализации "
                         f"против {peer_ratio:.1f} у {peer.get('symbol')}: сеть оценена "
                         f"{'дешевле' if s2 > 0 else 'дороже' if s2 < 0 else 'сопоставимо'} относительно своего TVL.")
        score = int(np.clip(s1 + s2, -2, 2))
        if ratio > 1.5:
            risks.append(f"FDV выше капитализации в {ratio:.2f} раза — риск размывания")
        criteria.append(_crit(
            "valuation", "Рыночная оценка", score, facts3,
            "FDV/MC: до 1,1 +1; до 1,5 0; до 2,5 −1; выше −2. MC/TVL ниже 0,8 от актива сравнения +1; выше 1,25 −1",
            text))
    else:
        criteria.append(_crit("valuation", "Рыночная оценка", None, {}, "нет рыночного снимка на эту дату",
                              "Недостаточно данных: капитализация и FDV на дату расчёта не сохранены."))

    # 4. распределение токенов
    dist = facts.get("distribution")
    if dist:
        insiders = dist.get("team_foundation", 0) + dist.get("investors", 0)
        score = 2 if insiders <= 10 else 1 if insiders <= 20 else 0 if insiders <= 35 else -1 if insiders <= 50 else -2
        if insiders > 35:
            risks.append(f"Команде и инвесторам выделено {insiders}% эмиссии")
        criteria.append(_crit(
            "distribution", "Распределение токенов", score,
            {"распределение, % эмиссии": dist, "команда, фонд и инвесторы": insiders,
             "источник": facts.get("distribution_source"),
             "концентрация по адресам": "нет данных в бесплатных источниках"},
            "доля команды, фонда и инвесторов: до 10% +2; до 20% +1; до 35% 0; до 50% −1; выше −2",
            f"Команде, фонду и ранним инвесторам выделено {insiders}% эмиссии: чем выше эта доля, тем больше "
            f"потенциальное давление продаж и влияние узкого круга держателей на сеть."))
    else:
        criteria.append(_crit("distribution", "Распределение токенов", None, {},
                              "нет проверенных данных о распределении",
                              "Недостаточно данных для оценки."))

    # 5. эмиссионная политика: инфляция за вычетом сжигания
    infl = facts.get("inflation_ref_pct")
    fees30 = _sum_tail(fees, 30)
    burn_pct = None
    if fees30 is not None and mc:
        share_burned = {"full": 1.0, "partial": 0.8, "none": 0.0}.get(facts.get("fee_burn"), 0.0)
        burn_pct = fees30 * 365 / 30 / mc * 100 * share_burned
    if infl is not None:
        net = infl - (burn_pct or 0.0)
        score = 2 if net <= 0 else 1 if net <= 2 else 0 if net <= 5 else -1 if net <= 8 else -2
        if net > 5:
            risks.append(f"Чистая инфляция около {net:.1f}% в год")
        criteria.append(_crit(
            "emission", "Эмиссионная политика", score,
            {"инфляция, % в год (справочно)": infl, "источник": facts.get("inflation_source"),
             "сжигание, % капитализации в год": None if burn_pct is None else round(burn_pct, 3),
             "чистая инфляция, %": round(net, 3), "сжигание": facts.get("fee_burn_note"),
             "разблокировки": facts.get("unlocks_note")},
            "чистая инфляция: до 0% +2; до 2% +1; до 5% 0; до 8% −1; выше −2",
            f"Выпуск наград увеличивает предложение на {infl:.1f}% в год, сжигание комиссий убирает "
            f"{0 if burn_pct is None else burn_pct:.2f}%: держатель, который не участвует в стейкинге, "
            f"теряет около {max(net, 0):.1f}% доли в год."))
    else:
        criteria.append(_crit("emission", "Эмиссионная политика", None,
                              {"сжигание": facts.get("fee_burn_note"), "разблокировки": facts.get("unlocks_note")},
                              "нет проверенного значения инфляции",
                              "Недостаточно данных: темп эмиссии на дату расчёта неизвестен."))

    # 6. технология: чек-лист из 4 свойств + децентрализация по числу валидаторов
    tech = (facts.get("technology") or {}).get("criteria") or {}
    if tech:
        passed = sum(1 for v in tech.values() if v.get("value"))
        score = {4: 2, 3: 1, 2: 0, 1: -1, 0: -2}.get(passed, 0)
        tfacts = {"консенсус": facts["technology"].get("consensus"),
                  "архитектура": facts["technology"].get("architecture"),
                  **{k: ("да" if v.get("value") else "нет") + f" — {v.get('note')}" for k, v in tech.items()}}
        val = metrics.get("validatorCount")
        if val is not None and val.dropna().size:
            tfacts["число валидаторов"] = int(val.dropna().iloc[-1])
        criteria.append(_crit(
            "technology", "Технология", score, tfacts,
            "выполнено свойств из 4 (быстрая финализация, PoS, совместимость с EVM, горизонтальное масштабирование): 4 → +2; 3 → +1; 2 → 0; 1 → −1; 0 → −2",
            f"Выполнено {passed} из 4 технологических свойств: они определяют, сколько приложений сеть способна "
            f"обслужить и насколько дёшево разработчикам переносить их с других сетей, то есть будущий спрос на газ."))
    else:
        criteria.append(_crit("technology", "Технология", None, {}, "нет справочных данных",
                              "Недостаточно данных для оценки."))

    # 7. экосистема: число протоколов и динамика TVL
    protocols = snapshot.get("protocols_count")
    tvl90 = _change(tvl, 90)
    if protocols is not None or tvl90 is not None:
        s1 = 0 if protocols is None else 1 if protocols >= 300 else 0 if protocols >= 100 else -1
        s2 = 0 if tvl90 is None else 1 if tvl90 > 0.10 else -1 if tvl90 < -0.10 else 0
        if tvl90 is not None and tvl90 < -0.10:
            risks.append(f"TVL сети снизился на {abs(tvl90):.0%} за 90 дней")
        criteria.append(_crit(
            "ecosystem", "Экосистема", int(np.clip(s1 + s2, -2, 2)),
            {"протоколов в сети": protocols, "TVL, долл.": tvl_last,
             "изменение TVL за 90 дней": None if tvl90 is None else round(tvl90, 4)},
            "протоколов от 300 +1; от 100 0; меньше −1. TVL за 90 дней: рост более 10% +1; падение более 10% −1",
            (f"В сети {protocols} протоколов; " if protocols is not None else "")
            + (f"TVL за 90 дней изменился на {tvl90:+.0%}. " if tvl90 is not None else "")
            + "Приложения создают транзакции и удерживают капитал в сети — это прямой источник спроса на токен как газ и залог."))
    else:
        criteria.append(_crit("ecosystem", "Экосистема", None, {}, "нет данных о протоколах и TVL",
                              "Недостаточно данных для оценки."))

    # 8. экономическая активность: комиссии относительно капитализации
    if fees30 is not None and mc:
        annual = fees30 * 365 / 30
        yield_pct = annual / mc * 100
        score = 2 if yield_pct >= 1 else 1 if yield_pct >= 0.3 else 0 if yield_pct >= 0.1 else -1 if yield_pct >= 0.03 else -2
        fee_trend = _change(fees.rolling(30).sum() if fees is not None else None, 30)
        if yield_pct < 0.1:
            risks.append("Комиссии сети малы относительно капитализации: оценка опирается на ожидания, а не на доход")
        criteria.append(_crit(
            "activity", "Экономическая активность", score,
            {"комиссии за 30 дней, долл.": round(fees30), "в годовом выражении, долл.": round(annual),
             "комиссии к капитализации, % в год": round(yield_pct, 4),
             "изменение 30-дневных комиссий": None if fee_trend is None else round(fee_trend, 4)},
            "годовые комиссии к капитализации: от 1% +2; от 0,3% +1; от 0,1% 0; от 0,03% −1; меньше −2",
            f"Пользователи платят сети около {format(round(annual), ',').replace(',', ' ')} долл. в год — это {yield_pct:.2f}% капитализации. "
            f"Комиссии — мера платёжеспособного спроса на блочное пространство; при сжигании они "
            f"превращаются в сокращение предложения токена."))
    else:
        criteria.append(_crit("activity", "Экономическая активность", None, {}, "нет ряда комиссий или капитализации",
                              "Недостаточно данных для оценки."))

    scored = [c for c in criteria if c["score"] is not None]
    n = len(scored)
    details = {"criteria": criteria, "n_scored": n, "market_cap": mc, "fdv": fdv, "circulating": circ,
               "total_supply": total, "max_supply": max_supply, "price": price, "tvl": tvl_last,
               "protocols_count": protocols, "snapshot_at": snapshot.get("snapshot_at"),
               "facts_verified_at": facts.get("verified_at"), "warnings": warnings}
    if n < MIN_CRITERIA:
        missing = ", ".join(c["title"] for c in criteria if c["score"] is None)
        return {"status": "unavailable", "score": None, "details": details, "risks": risks,
                "explanation": f"Оценено {n} направлений из 8, нужно не менее {MIN_CRITERIA}. Нет данных: {missing}."}
    total_points = sum(c["score"] for c in scored)
    score = round(100 * total_points / (2 * n), 2)
    explanation = (f"FA = 100 × ({total_points:+d}) / (2 × {n}) = {score:+.0f}. "
                   + "; ".join(f"{c['title']} {c['score']:+d}" for c in scored) + "."
                   + (f" Не оценено: {', '.join(c['title'] for c in criteria if c['score'] is None)}." if n < 8 else ""))
    return {"status": "ok", "score": score, "details": details, "risks": risks, "explanation": explanation}
