"""Оформление в стиле Material 3 (Material You): цветовые роли, формы, типографика.

Цветовая схема строится из выбранного исходного цвета (как динамический цвет Material You):
три набора ролей для светлого и тёмного режима. Режим переключается в интерфейсе.
"""
from __future__ import annotations

import streamlit as st

PALETTES = {
    "Лавина (красная)": {
        "light": dict(primary="#B02A2F", on_primary="#FFFFFF", primary_container="#FFDAD7", on_primary_container="#410005",
                      secondary_container="#F5DDDB", on_secondary_container="#2C1514", surface="#FFF8F7",
                      surface_low="#FEF1EF", surface_container="#F8EBE9", surface_high="#F2E5E3", on_surface="#231918",
                      on_surface_variant="#534342", outline="#857371", outline_variant="#D8C2BF"),
        "dark": dict(primary="#FFB3AE", on_primary="#68000F", primary_container="#8C1520", on_primary_container="#FFDAD7",
                     secondary_container="#5D3F3D", on_secondary_container="#FFDAD7", surface="#1A1110",
                     surface_low="#231918", surface_container="#271D1C", surface_high="#322826", on_surface="#F1DEDC",
                     on_surface_variant="#D8C2BF", outline="#A08C8A", outline_variant="#534342"),
    },
    "Базовая (фиолетовая)": {
        "light": dict(primary="#6750A4", on_primary="#FFFFFF", primary_container="#EADDFF", on_primary_container="#21005D",
                      secondary_container="#E8DEF8", on_secondary_container="#1D192B", surface="#FEF7FF",
                      surface_low="#F7F2FA", surface_container="#F3EDF7", surface_high="#ECE6F0", on_surface="#1D1B20",
                      on_surface_variant="#49454F", outline="#79747E", outline_variant="#CAC4D0"),
        "dark": dict(primary="#D0BCFF", on_primary="#381E72", primary_container="#4F378B", on_primary_container="#EADDFF",
                     secondary_container="#4A4458", on_secondary_container="#E8DEF8", surface="#141218",
                     surface_low="#1D1B20", surface_container="#211F26", surface_high="#2B2930", on_surface="#E6E0E9",
                     on_surface_variant="#CAC4D0", outline="#938F99", outline_variant="#49454F"),
    },
    "Океан (синяя)": {
        "light": dict(primary="#36618E", on_primary="#FFFFFF", primary_container="#D1E4FF", on_primary_container="#001D36",
                      secondary_container="#D7E3F8", on_secondary_container="#101C2B", surface="#F8F9FF",
                      surface_low="#F2F3FA", surface_container="#ECEEF4", surface_high="#E6E8EE", on_surface="#191C20",
                      on_surface_variant="#43474E", outline="#73777F", outline_variant="#C3C7CF"),
        "dark": dict(primary="#A0CAFD", on_primary="#003258", primary_container="#194975", on_primary_container="#D1E4FF",
                     secondary_container="#3B4858", on_secondary_container="#D7E3F8", surface="#111418",
                     surface_low="#191C20", surface_container="#1D2024", surface_high="#272A2F", on_surface="#E1E2E8",
                     on_surface_variant="#C3C7CF", outline="#8D9199", outline_variant="#43474E"),
    },
}

# Цвета данных не зависят от схемы интерфейса: фиксированный порядок рядов и статусные цвета
DATA = {
    "light": dict(series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
                  good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b",
                  good_text="#006300", critical_text="#a82020", warning_text="#7a5200", neutral="#8a8985"),
    "dark": dict(series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c"],
                 good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b",
                 good_text="#5fd65f", critical_text="#ff8f8f", warning_text="#fab219", neutral="#8a8985"),
}


def tokens(palette: str, dark: bool) -> dict:
    mode = "dark" if dark else "light"
    t = dict(PALETTES.get(palette, next(iter(PALETTES.values())))[mode])
    t.update(DATA[mode])
    t["mode"] = mode
    return t


def apply_streamlit_theme(t: dict) -> bool:
    """Передаёт цвета во встроенную тему Streamlit (нужно для таблиц и полей ввода). True — нужен перезапуск."""
    want = {"theme.base": t["mode"], "theme.primaryColor": t["primary"], "theme.backgroundColor": t["surface"],
            "theme.secondaryBackgroundColor": t["surface_container"], "theme.textColor": t["on_surface"]}
    changed = False
    for key, value in want.items():
        if st._config.get_option(key) != value:
            st._config.set_option(key, value)
            changed = True
    return changed


def css(t: dict) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap');
:root {{
  --md-primary:{t['primary']}; --md-on-primary:{t['on_primary']};
  --md-primary-container:{t['primary_container']}; --md-on-primary-container:{t['on_primary_container']};
  --md-secondary-container:{t['secondary_container']}; --md-on-secondary-container:{t['on_secondary_container']};
  --md-surface:{t['surface']}; --md-surface-low:{t['surface_low']}; --md-surface-container:{t['surface_container']};
  --md-surface-high:{t['surface_high']}; --md-on-surface:{t['on_surface']}; --md-on-surface-variant:{t['on_surface_variant']};
  --md-outline:{t['outline']}; --md-outline-variant:{t['outline_variant']};
  --st-good:{t['good']}; --st-warning:{t['warning']}; --st-critical:{t['critical']};
  --st-good-text:{t['good_text']}; --st-critical-text:{t['critical_text']}; --st-warning-text:{t['warning_text']};
}}
html, body, .stApp, [data-testid="stAppViewContainer"], .stMarkdown, button, input, textarea, select {{
  font-family: 'Roboto', 'Google Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}}
.stApp {{ background: var(--md-surface); color: var(--md-on-surface); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1320px; }}

/* навигационная панель (navigation drawer) */
[data-testid="stSidebar"] {{ background: var(--md-surface-low); border-right: none; }}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{ gap: 2px; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"] {{
  border-radius: 28px; padding: 10px 16px; margin: 0; width: 100% !important; box-sizing: border-box; display: flex; transition: background .15s;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"]:hover {{ background: color-mix(in srgb, var(--md-on-surface) 8%, transparent); }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
  background: var(--md-secondary-container); color: var(--md-on-secondary-container); font-weight: 500;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div > div:first-child:not([data-testid]) {{ display: none; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"] {{
  background: var(--md-secondary-container); color: var(--md-on-secondary-container); font-weight: 500;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"] p {{ font-weight: 500; }}
[data-testid="stSidebar"] [data-testid="stRadio"], [data-testid="stSidebar"] [data-testid="stRadio"] > div {{ width: 100%; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {{ font-size: 14px; letter-spacing: .1px; }}

/* кнопки: filled и tonal */
.stButton > button, .stDownloadButton > button {{
  border-radius: 20px; border: none; padding: 8px 24px; font-weight: 500; letter-spacing: .1px;
  background: var(--md-secondary-container); color: var(--md-on-secondary-container); box-shadow: none;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background: color-mix(in srgb, var(--md-secondary-container) 88%, var(--md-on-secondary-container));
  color: var(--md-on-secondary-container); border: none;
}}
.stButton > button[kind="primary"] {{ background: var(--md-primary); color: var(--md-on-primary); }}
.stButton > button[kind="primary"] p {{ color: var(--md-on-primary); }}
.stButton > button[kind="primary"]:hover {{
  background: color-mix(in srgb, var(--md-primary) 90%, var(--md-on-primary)); color: var(--md-on-primary);
}}
.stButton > button:focus-visible {{ outline: 2px solid var(--md-primary); outline-offset: 2px; }}

/* поля ввода, списки */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], .stDateInput > div > div {{
  border-radius: 12px !important;
}}
/* вкладки */
[data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--md-outline-variant); }}
[data-baseweb="tab"] {{ padding: 10px 16px; font-weight: 500; }}
[data-baseweb="tab-highlight"] {{ background: var(--md-primary); height: 3px; border-radius: 3px 3px 0 0; }}
/* раскрывающиеся блоки и контейнеры */
[data-testid="stExpander"] details {{ border-radius: 16px; border: 1px solid var(--md-outline-variant); background: var(--md-surface-low); }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 16px; }}
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
[data-testid="stPlotlyChart"] {{ border-radius: 16px; overflow: hidden; }}
hr {{ border-color: var(--md-outline-variant); }}

/* компоненты AvaLens */
.md-appbar {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin: 0 0 4px; }}
.md-appbar h1 {{ font-size: 28px; line-height: 36px; font-weight: 400; margin:0; padding:0; color: var(--md-on-surface); }}
.md-appbar .sub {{ font-size: 14px; color: var(--md-on-surface-variant); }}
.md-card {{ background: var(--md-surface-container); border-radius: 16px; padding: 16px 20px; height: 100%; margin-bottom: 12px; }}
.md-card.outlined {{ background: var(--md-surface); border: 1px solid var(--md-outline-variant); }}
.md-card.primary {{ background: var(--md-primary-container); color: var(--md-on-primary-container); }}
.md-card.primary .md-label, .md-card.primary .md-support {{ color: var(--md-on-primary-container); opacity:.8; }}
.md-label {{ font-size: 12px; line-height:16px; font-weight: 500; letter-spacing: .5px; color: var(--md-on-surface-variant); }}
.md-value {{ font-size: 32px; line-height: 40px; font-weight: 400; margin-top: 4px; font-variant-numeric: tabular-nums; }}
.md-value.small {{ font-size: 24px; line-height: 32px; }}
.md-support {{ font-size: 13px; line-height: 18px; color: var(--md-on-surface-variant); margin-top: 4px; }}
.md-title {{ font-size: 16px; line-height: 24px; font-weight: 500; letter-spacing:.15px; margin: 0 0 6px; }}
.md-body {{ font-size: 14px; line-height: 20px; letter-spacing:.25px; }}
.md-section {{ font-size: 22px; line-height: 28px; font-weight: 400; margin: 18px 0 10px; }}
.md-chip {{ display:inline-flex; align-items:center; gap:6px; height: 28px; padding: 0 12px; border-radius: 8px;
  border: 1px solid var(--md-outline-variant); font-size: 13px; font-weight: 500; margin: 2px 6px 2px 0; white-space: nowrap; }}
.md-chip.tonal {{ background: var(--md-secondary-container); color: var(--md-on-secondary-container); border-color: transparent; }}
.md-chip .dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--md-outline); }}
.md-chip.good .dot {{ background: var(--st-good); }} .md-chip.bad .dot {{ background: var(--st-critical); }}
.md-chip.warn .dot {{ background: var(--st-warning); }}
.up {{ color: var(--st-good-text); }} .down {{ color: var(--st-critical-text); }} .flat {{ color: var(--md-on-surface-variant); }}
.md-banner {{ display:flex; gap:12px; align-items:flex-start; border-radius: 12px; padding: 12px 16px; margin: 6px 0;
  background: var(--md-surface-high); font-size: 14px; line-height: 20px; }}
.md-banner.warn {{ border-left: 4px solid var(--st-warning); }} .md-banner.bad {{ border-left: 4px solid var(--st-critical); }}
.md-banner.info {{ border-left: 4px solid var(--md-primary); }} .md-banner.good {{ border-left: 4px solid var(--st-good); }}
.md-banner b {{ font-weight: 500; }}
.md-scale {{ position: relative; height: 12px; border-radius: 6px; margin: 22px 0 6px;
  background: linear-gradient(90deg, var(--st-critical) 0 35%, color-mix(in srgb, var(--st-critical) 45%, var(--md-surface-high)) 35% 45%,
    var(--md-outline-variant) 45% 55%, color-mix(in srgb, var(--st-good) 45%, var(--md-surface-high)) 55% 65%, var(--st-good) 65% 100%); }}
.md-scale .thumb {{ position:absolute; top:-6px; width: 6px; height: 24px; border-radius: 3px; background: var(--md-on-surface);
  outline: 3px solid var(--md-surface-container); transform: translateX(-50%); }}
.md-card.primary .md-scale .thumb {{ outline-color: var(--md-primary-container); background: var(--md-on-primary-container); }}
.md-scale-labels {{ display:flex; justify-content:space-between; font-size: 11px; color: var(--md-on-surface-variant); }}
.md-progress {{ height: 8px; border-radius: 4px; background: var(--md-surface-high); overflow:hidden; margin-top: 12px; }}
.md-progress > div {{ height: 100%; border-radius: 4px; background: var(--md-primary); }}
.md-table {{ width:100%; border-collapse: collapse; font-size: 14px; }}
.md-table th {{ text-align:left; font-size: 12px; font-weight: 500; letter-spacing:.5px; color: var(--md-on-surface-variant);
  padding: 8px 12px; border-bottom: 1px solid var(--md-outline-variant); }}
.md-table td {{ padding: 10px 12px; border-bottom: 1px solid var(--md-outline-variant); vertical-align: top; line-height: 20px; }}
.md-table tr:last-child td {{ border-bottom: none; }}
.md-table td.num, .md-table th.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.md-badge {{ display:inline-block; min-width: 34px; text-align:center; padding: 2px 8px; border-radius: 8px; font-weight: 500;
  font-variant-numeric: tabular-nums; background: var(--md-surface-high); }}
.md-badge.pos {{ background: color-mix(in srgb, var(--st-good) 22%, transparent); color: var(--st-good-text); }}
.md-badge.neg {{ background: color-mix(in srgb, var(--st-critical) 22%, transparent); color: var(--st-critical-text); }}
.md-list {{ margin: 0; padding-left: 18px; }} .md-list li {{ margin: 4px 0; font-size: 14px; line-height: 20px; }}
a {{ color: var(--md-primary); }}
</style>
"""
