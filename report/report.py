"""Формирование аналитического заключения в Markdown по результату расчёта."""
from __future__ import annotations

from analysis.integral import NAMES


def _num(x, digits=2):
    return "—" if x is None else f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def build(result: dict, quote: dict | None = None) -> str:
    it = result["integral"]
    lines = [f"# AvaLens: аналитическое заключение по {result['name']} ({result['symbol']})", "",
             f"Период: {result['period_start']} — {result['period_end']}. Версия алгоритма {result['algo_version']}. "
             f"Контрольная сумма входных данных: `{result['input_hash']}`."
             + (f" Расчёт № {result['run_id']}." if result.get("run_id") else ""), ""]
    if quote:
        lines += [f"Цена на момент загрузки данных: {_num(quote['price'])} USDT ({quote['source']}, {quote['time']}).", ""]
    lines += ["## Итог", "", it["summary"], ""]
    lines += ["| Модуль | Оценка | Вес | Качество данных q | Состояние |", "| --- | --- | --- | --- | --- |"]
    for r in it["modules"]:
        score = "—" if r["score"] is None else f"{r['score']:+.0f}"
        lines.append(f"| {r['name']} | {score} | {r['weight']:.2f} | {r['quality']:.2f} | {r['state']} |")
    lines += ["", f"Интегральная оценка S: **{'—' if it['s_score'] is None else format(it['s_score'], '+.1f')}**. "
                  f"Confidence Score: **{it['confidence']:.0f} из 100**.", ""]
    if it["conflicts"]:
        lines += ["## Противоречия между модулями", ""]
        lines += [f"- **{c['severity'].capitalize()}** ({c['module_a']} и {c['module_b']}, разрыв {c['gap']:.0f}): {c['explanation']}"
                  for c in it["conflicts"]] + [""]
    else:
        lines += ["## Противоречия между модулями", "", "Противоречий между модулями не найдено.", ""]
    lines += ["## Результаты модулей", ""]
    for key in ("TA", "FA", "ON", "SA"):
        m = result["modules"][key]
        lines += [f"### {NAMES[key]}", "", m.get("explanation") or "", ""]
        if key == "TA" and m["status"] == "ok":
            lines += [f"- {s['name']}: {s['value']:+.2f}. {s['text']}" for s in m["signals"]] + [""]
        if key == "FA":
            for c in m["details"]["criteria"]:
                score = "нет данных" if c["score"] is None else f"{c['score']:+d}"
                lines.append(f"- **{c['title']}: {score}.** {c['consequence']}")
            lines.append("")
        if key == "ON":
            for r in m["details"].get("metrics", []):
                if r.get("available"):
                    lines.append(f"- {r['title']}: {r['delta']:+.1%} к среднему за 30 дней, сигнал {r['signal']:+.2f}.")
                else:
                    lines.append(f"- {r['title']}: нет данных.")
            lines.append("")
        if key == "SA" and m["details"].get("events"):
            lines += ["Наиболее значимые события:", ""]
            lines += [f"- {e['date']}: {e['title']} (публикаций: {e['publications']})" for e in m["details"]["events"][:5]] + [""]
    lines += ["## Выявленные риски", ""]
    lines += [f"- {NAMES[r['module']]}: {r['text']}." for r in result["risks"]] or ["Риски не выявлены."]
    lines += ["", "## Качество данных", ""]
    lines += [f"- {w}" for w in result["warnings"]] or ["Замечаний к данным нет."]
    if it["weak_points"]:
        lines += [""] + [f"- {w}" for w in it["weak_points"]]
    lines += ["", "---", "Заключение сформировано автоматически по правилам, описанным в документации системы. "
                          "Оно не является инвестиционной рекомендацией."]
    return "\n".join(lines)
