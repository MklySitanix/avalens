"""Модуль комплексной оценки (алгоритм из раздела 10 архитектуры).

S = сумма(w_i * X_i) / сумма(w_i) по доступным модулям; Confidence Score
C = 100 * k * сумма(w_i * q_i), q_i = полнота * актуальность * надёжность.
При C < 50 или менее чем трёх доступных модулях оценка S не выдаётся.
"""
from __future__ import annotations

from itertools import combinations

WEIGHTS = {"TA": 0.30, "FA": 0.25, "ON": 0.25, "SA": 0.20}
NAMES = {"TA": "Технический анализ", "FA": "Фундаментальный анализ", "ON": "Ончейн-анализ",
         "SA": "Анализ настроений"}
Q_MIN = 0.4
C_MIN, C_FULL = 50, 70
MIN_MODULES = 3
CONFLICT_LEVEL = 20
STRONG_GAP = 80

SCALE = [(30, "Преимущественно положительная совокупность сигналов"), (10, "Умеренно положительная оценка"),
         (-10, "Нейтральное состояние"), (-30, "Умеренно отрицательная оценка"),
         (-101, "Преимущественно отрицательная совокупность сигналов")]

# (положительный модуль, отрицательный модуль) -> объяснение для пользователя
RULES = {
    ("TA", "ON"): "Цена растёт, а активность сети падает: рост не подтверждён использованием сети и может быть спекулятивным.",
    ("ON", "TA"): "Активность сети растёт, а цена слабеет: рынок пока не учитывает рост использования сети.",
    ("TA", "FA"): "Тренд восходящий при слабых фундаментальных показателях: краткосрочный импульс расходится с долгосрочными рисками токена.",
    ("FA", "TA"): "Фундаментальные показатели сильные, а цена снижается: рынок оценивает актив хуже, чем следует из его экономики.",
    ("TA", "SA"): "Цена растёт при отрицательном информационном фоне: рост идёт вопреки новостям и может оказаться неустойчивым.",
    ("SA", "TA"): "Информационный фон положительный, а цена падает: позитивные новости не поддержаны покупками или уже учтены в цене.",
    ("FA", "ON"): "Токеномика сильная, а активность сети падает: спрос на блочное пространство снижается, сжигание комиссий ослабевает.",
    ("ON", "FA"): "Сеть используется активнее, но фундаментальные показатели токена слабые: рост активности не превращается в ценность для держателя.",
    ("FA", "SA"): "Фундаментальные показатели сильные при отрицательном фоне: новости могут переоценивать краткосрочные проблемы.",
    ("SA", "FA"): "Информационный фон положительный при фундаментальных рисках: фон может недооценивать риски проекта.",
    ("ON", "SA"): "Активность сети растёт при отрицательном фоне: обсуждение отстаёт от реального использования сети.",
    ("SA", "ON"): "Информационный фон положительный, а активность сети падает: обсуждение опережает реальное использование сети.",
}


def interpret(s: float) -> tuple[int, str]:
    """Возвращает (номер ступени шкалы, текст). Границы — из таблицы интерпретации задания."""
    level = 0 if s >= 30 else 1 if s >= 10 else 2 if s > -10 else 3 if s > -30 else 4
    return level, SCALE[level][1]


def find_conflicts(scores: dict[str, float]) -> list[dict]:
    out = []
    for a, b in combinations(scores, 2):
        hi, lo = (a, b) if scores[a] >= scores[b] else (b, a)
        if scores[hi] >= CONFLICT_LEVEL and scores[lo] <= -CONFLICT_LEVEL:
            gap = scores[hi] - scores[lo]
            out.append({"module_a": hi, "module_b": lo, "rule": f"{hi}>=+{CONFLICT_LEVEL} и {lo}<=-{CONFLICT_LEVEL}",
                        "gap": round(gap, 1), "severity": "сильное" if gap >= STRONG_GAP else "умеренное",
                        "explanation": f"{NAMES[hi]} {scores[hi]:+.0f} против {NAMES[lo]} {scores[lo]:+.0f}. "
                                       + RULES[(hi, lo)]})
    return sorted(out, key=lambda c: -c["gap"])


def combine(modules: dict[str, dict], k_sync: float = 1.0, sync_note: str | None = None) -> dict:
    """modules: {'TA': {'score', 'quality', 'status', 'quality_parts'}, ...}."""
    rows, available = [], {}
    for name, w in WEIGHTS.items():
        m = modules.get(name, {})
        q = float(m.get("quality") or 0.0)
        score = m.get("score")
        if m.get("status") != "ok" or score is None:
            state, reason = "недоступен", m.get("explanation") or "нет данных"
            q_eff = min(q, 0.0) if score is None else q
        elif q < Q_MIN:
            state, reason = "исключён", f"качество данных q = {q:.2f} ниже порога {Q_MIN}"
            q_eff = q
        else:
            state, reason, q_eff = "учтён", None, q
            available[name] = float(score)
        rows.append({"module": name, "name": NAMES[name], "weight": w, "score": score, "quality": round(q_eff, 3),
                     "state": state, "reason": reason, "quality_parts": m.get("quality_parts", {})})

    confidence = round(100 * k_sync * sum(r["weight"] * r["quality"] for r in rows), 1)
    weak = [f"{r['name']}: {r['reason']}" for r in rows if r["state"] != "учтён"]
    for r in rows:
        p = r["quality_parts"]
        if r["state"] == "учтён":
            for key, label in (("completeness", "полнота"), ("freshness", "актуальность"), ("reliability", "надёжность")):
                if p.get(key) is not None and p[key] < 0.8:
                    weak.append(f"{r['name']}: {label} {p[key]:.2f}" + (f" — {p['notes'][key]}" if p.get("notes", {}).get(key) else ""))
    if k_sync < 1 and sync_note:
        weak.append(sync_note)

    base = {"weights": WEIGHTS, "modules": rows, "confidence": confidence, "k_sync": k_sync,
            "weak_points": weak, "conflicts": [], "contributions": {}}
    if confidence < C_MIN or len(available) < MIN_MODULES:
        why = (f"доступно модулей: {len(available)} из 4, нужно не менее {MIN_MODULES}" if len(available) < MIN_MODULES
               else f"Confidence Score {confidence:.0f} ниже порога {C_MIN}")
        return {**base, "s_score": None, "verdict": "Недостаточно данных для комплексной оценки", "level": None,
                "preliminary": False, "softened": False,
                "summary": f"Недостаточно данных для комплексной оценки: {why}. " + " ".join(w + "." for w in weak)}

    w_sum = sum(WEIGHTS[m] for m in available)
    s = round(sum(WEIGHTS[m] * x for m, x in available.items()) / w_sum, 1)
    contributions = {m: round(WEIGHTS[m] * x / w_sum, 2) for m, x in available.items()}
    conflicts = find_conflicts(available)
    level, verdict = interpret(s)
    softened = False
    if any(c["severity"] == "сильное" for c in conflicts) and level in (0, 4):
        level, softened = (1 if level == 0 else 3), True
        verdict = SCALE[level][1] + " (смягчено из-за сильного противоречия между модулями)"
    preliminary = confidence < C_FULL

    formula = " + ".join(f"{WEIGHTS[m]:.2f} × ({x:+.0f})" for m, x in available.items())
    parts = [f"S = ({formula}) / {w_sum:.2f} = {s:+.1f}. {verdict}."]
    if len(available) < 4:
        parts.append("Веса пересчитаны на доступные модули.")
    if conflicts:
        parts.insert(0, "Сигналы противоречат друг другу: " + " ".join(c["explanation"] for c in conflicts[:2]))
    if preliminary:
        parts.append(f"Вывод предварительный: Confidence Score {confidence:.0f} ниже {C_FULL}.")
    parts.append(f"Достоверность {confidence:.0f} из 100.")
    return {**base, "s_score": s, "verdict": verdict, "level": level, "preliminary": preliminary,
            "softened": softened, "conflicts": conflicts, "contributions": contributions,
            "summary": " ".join(parts)}
