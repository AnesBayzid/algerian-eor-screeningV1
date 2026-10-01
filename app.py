"""
Algerian EOR Screening Platform
==============================
Dual-Engine Streamlit App for:
  - Classical reservoir screening rules
  - Physics-informed ML ensemble using V5-style engineered features

The app loads trained model artifacts for inference. If artifacts are absent
but the private training workbook is available, it trains and caches the
models locally; heuristic placeholder output is reserved for training failure.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def _build_favicon() -> Image.Image:
    """Small palette-matched favicon (accent-green ring on transparent
    background) generated in-code, so it matches the app's actual color
    system instead of a generic emoji that doesn't track the palette.
    """
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    accent = (46, 139, 107, 255)  # #2E8B6B
    draw.ellipse((4, 4, size - 4, size - 4), outline=accent, width=7)
    draw.ellipse((24, 24, size - 24, size - 24), fill=accent)
    return img


st.set_page_config(
    page_title="Algerian EOR Screening Platform",
    page_icon=_build_favicon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EOR_CLASSES = [
    "Miscible Gas Injection",
    "Immiscible Gas Injection",
    "WAG Injection",
    "Chemical EOR",
    "Hybrid Gas-Chemical",
    "Secondary Waterflooding",
]

DEFAULT_INPUTS = {
    "Lithology": "Sandstone",
    "Permeability": 70.0,
    "Porosity": 9.3,
    "Oil_Viscosity": 0.30,
    "Temperature": 102.0,
    "API_Gravity": 43.0,
    "Depth_m": 3200.0,
    "Pressure_bar": 190.0,
}

LANGUAGES = {"English": "en", "Français": "fr", "العربية": "ar"}

TRANSLATIONS = {
    "en": {
        "title": "Algerian EOR Decision-Support Suite",
        "subtitle": "Dual-engine technical screening: classical rules + physics-informed ML",
        "language": "Language", "inputs": "Reservoir Inputs",
        "project": "Project", "horizon": "Producing Horizon",
        "section_project": "Project", "section_rock": "Rock & reservoir",
        "section_depth": "Depth & pressure", "section_fluid": "Fluid & thermal",
        "preset": "Geological Preset / Lithology Guide", "custom": "Custom field data",
        "manual": "Manual lithology and field data", "lithology": "Lithology",
        "permeability": "Permeability", "porosity": "Porosity", "viscosity": "Oil Viscosity",
        "temperature": "Temperature", "gravity": "Oil Gravity", "depth": "Depth", "pressure": "Pressure",
        "guide": "Algerian geological guide",
        "guide_text": "Use this general formation-family guide when selecting the dominant lithology for an Algerian reservoir.",
        "proxies": "Physical Proxies", "classical": "Engine 1: Classical Screening",
        "ml": "Engine 2: ML Ensemble", "consensus": "Consensus Dashboard",
        "tab1_short": "Engine 1", "tab2_short": "Engine 2", "tab3_short": "Consensus Dashboard",
        "ml_only_result": "Engine B: ML-Only Result",
        "combined_result": "Rule-Adjusted Score: Engine A + Engine B",
        "ml_mean": "Engine B Probability (%)",
        "model_comparison": "Model Probability Comparison",
        "model_detail": "Detailed probability view",
        "combined_score": "Rule-adjusted score",
        "engine1_title": "Classical Heuristic Screening Windows (Preliminary Rules Engine)",
        "engine1_disclaimer": "These are initial heuristic screening windows. Detailed screening requires local thermodynamics, mobility-ratio analysis, and modern chemical-formulation testing.",
        "screening_caption": "PASS = ideal range, MARGINAL = near boundary, FAIL = outside the screening range.",
        "method_details": "Parameter details", "violations": "Violations", "predicted": "Predicted Method",
        "fallback": "Fallback mode is active because trained model artifacts were not found.",
        "fallback_warning": "No trained model files were found. Results are heuristic placeholders, not validated model predictions.",
        "probability": "Probability", "eor_method": "EOR Method", "combined": "Rule-adjusted score",
        "engine2_caption": "Compare the three trained algorithms on the same field inputs. Select a model below to inspect its full probability distribution.",
        "worked_example": "Worked example: Project 1, Trias S1 sandstone. Use the left sidebar to enter a different reservoir.",
        "pressure_note": "Depth and pressure are displayed and normalized, but current classical rules do not yet apply formation-pressure or MMP correlations.",
    },
    "fr": {
        "title": "Suite algérienne d'aide à la décision EOR",
        "subtitle": "Criblage technique à deux moteurs : règles classiques + ML informé par la physique",
        "language": "Langue", "inputs": "Données du réservoir", "preset": "Préréglage géologique / guide de lithologie",
        "project": "Projet", "horizon": "Horizon producteur",
        "section_project": "Projet", "section_rock": "Roche et réservoir",
        "section_depth": "Profondeur et pression", "section_fluid": "Fluides et thermique",
        "custom": "Données personnalisées", "manual": "Lithologie et données saisies manuellement", "lithology": "Lithologie",
        "permeability": "Perméabilité", "porosity": "Porosité", "viscosity": "Viscosité de l'huile",
        "temperature": "Température", "gravity": "Gravité API", "depth": "Profondeur", "pressure": "Pression",
        "guide": "Guide géologique algérien",
        "guide_text": "Utilisez ce guide général des familles stratigraphiques pour sélectionner la lithologie dominante d'un réservoir algérien.",
        "proxies": "Indicateurs physiques", "classical": "Moteur 1 : criblage classique", "ml": "Moteur 2 : ensemble ML",
        "tab1_short": "Moteur 1", "tab2_short": "Moteur 2", "tab3_short": "Tableau de consensus",
        "consensus": "Tableau de consensus", "screening_caption": "PASS = plage idéale, MARGINAL = proche de la limite, FAIL = hors plage.",
        "ml_only_result": "Moteur B : résultat ML seul", "combined_result": "Score ajusté par les règles : moteurs A + B", "ml_mean": "Probabilité du moteur B (%)",
        "model_comparison": "Comparaison des probabilités des modèles", "model_detail": "Détail des probabilités", "combined_score": "Score ajusté par les règles",
        "engine1_title": "Fenêtres heuristiques classiques (moteur de règles préliminaires)",
        "engine1_disclaimer": "Ces fenêtres sont heuristiques et préliminaires. Le criblage détaillé nécessite la thermodynamique locale, les rapports de mobilité et des essais de formulations chimiques modernes.",
        "method_details": "Détails des paramètres", "violations": "Dépassements", "predicted": "Méthode prédite",
        "fallback": "Le mode secours est actif car les modèles entraînés sont absents.",
        "fallback_warning": "Aucun modèle entraîné trouvé. Les résultats sont heuristiques et non validés.",
        "probability": "Probabilité", "eor_method": "Méthode EOR", "combined": "Score ajusté par les règles",
        "engine2_caption": "Comparez les trois algorithmes sur les mêmes données. Sélectionnez un modèle pour examiner sa distribution complète des probabilités.",
        "worked_example": "Exemple : Projet 1, grès Trias S1. Utilisez la barre latérale pour saisir un autre réservoir.",
        "pressure_note": "La profondeur et la pression sont normalisées, mais les règles actuelles n'appliquent pas encore les corrélations de pression de formation ou de MMP.",
    },
    "ar": {
        "title": "منصة دعم قرار الاستخلاص المعزز للنفط في الجزائر",
        "subtitle": "فحص تقني بمحركين: قواعد كلاسيكية وتعلم آلي مدعوم بالفيزياء",
        "language": "اللغة", "inputs": "بيانات المكمن", "preset": "الإعداد الجيولوجي / دليل الصخور",
        "project": "المشروع", "horizon": "الأفق المنتج",
        "section_project": "المشروع", "section_rock": "الصخر والمكمن",
        "section_depth": "العمق والضغط", "section_fluid": "السوائل والحرارة",
        "custom": "بيانات مخصصة", "manual": "بيانات الصخور والمكمن يدوياً", "lithology": "الليثولوجيا",
        "permeability": "النفاذية", "porosity": "المسامية", "viscosity": "لزوجة النفط",
        "temperature": "درجة الحرارة", "gravity": "كثافة API", "depth": "العمق", "pressure": "الضغط",
        "guide": "الدليل الجيولوجي الجزائري",
        "guide_text": "استخدم هذا الدليل العام للعائلات التكوينية لاختيار الليثولوجيا السائدة في المكمن الجزائري.",
        "proxies": "المؤشرات الفيزيائية", "classical": "المحرك 1: الفحص الكلاسيكي", "ml": "المحرك 2: ensemble للتعلم الآلي",
        "tab1_short": "المحرك 1", "tab2_short": "المحرك 2", "tab3_short": "لوحة التوافق",
        "consensus": "لوحة التوافق", "screening_caption": "PASS = النطاق المثالي، MARGINAL = قريب من الحد، FAIL = خارج النطاق.",
        "ml_only_result": "المحرك B: نتيجة التعلم الآلي فقط", "combined_result": "النتيجة المعدلة بالقواعد: المحركان A وB", "ml_mean": "احتمال المحرك B (%)",
        "model_comparison": "مقارنة احتمالات النماذج", "model_detail": "تفصيل الاحتمالات", "combined_score": "النتيجة المعدلة بالقواعد",
        "engine1_title": "نوافذ الفحص الكلاسيكية الإرشادية (محرك القواعد الأولي)",
        "engine1_disclaimer": "هذه حدود فحص إرشادية أولية. يتطلب الفحص التفصيلي الديناميكا الحرارية المحلية ونسب الحركة واختبارات التركيبات الكيميائية الحديثة.",
        "method_details": "تفاصيل المعايير", "violations": "المخالفات", "predicted": "الطريقة المتوقعة",
        "fallback": "الوضع الاحتياطي فعال لأن ملفات النماذج غير موجودة.",
        "fallback_warning": "لم يتم العثور على نموذج مدرب. النتائج تقريبية وليست تنبؤات نموذج معتمد.",
        "probability": "الاحتمال", "eor_method": "طريقة EOR", "combined": "النتيجة المعدلة بالقواعد",
        "engine2_caption": "قارن الخوارزميات الثلاث على بيانات المكمن نفسها. اختر نموذجاً لعرض توزيع احتمالاته بالتفصيل.",
        "worked_example": "مثال تطبيقي: المشروع 1، الحجر الرملي Trias S1. استخدم الشريط الجانبي لإدخال مكمن آخر.",
        "pressure_note": "تم توحيد العمق والضغط، لكن القواعد الحالية لا تطبق بعد علاقات ضغط المكمن أو MMP.",
    },
}

GEOLOGICAL_LITHOLOGY_GUIDE = pd.DataFrame([
    {"Formation family": "Triassic", "General lithology": "Sandstone"},
    {"Formation family": "Devonian", "General lithology": "Sandstone"},
    {"Formation family": "Ordovician quartzite", "General lithology": "Quartzite sandstone"},
    {"Formation family": "Carboniferous", "General lithology": "Carbonate"},
])


def t(language: str, key: str) -> str:
    return TRANSLATIONS.get(language, TRANSLATIONS["en"]).get(key, key)


def apply_language_css(language: str) -> None:
    direction = "rtl" if language == "ar" else "ltr"
    st.markdown(
        f"""<style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        /* ---- Design tokens ------------------------------------------- */
        :root {{
            --eor-bg: #0B1220;
            --eor-panel: #131C2B;
            --eor-panel-raised: #182338;
            --eor-border: #26344A;
            --eor-text: #E8EDF4;
            --eor-text-muted: #8CA0B8;
            --eor-accent: #2E8B6B;
            --eor-accent-strong: #46B58A;
            --eor-pass: #37B679;
            --eor-marginal: #E8A33D;
            --eor-fail: #E5484D;
            --eor-radius: 8px;
            --eor-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace;
        }}

        html, body, [data-testid="stAppViewContainer"] {{ direction: {direction}; }}
        [data-testid="stAppViewContainer"] {{ font-family:'IBM Plex Sans',sans-serif; }}

        /* ---- Header / brand -------------------------------------------- */
        .eor-header {{
            display:flex; flex-direction:column; gap:2px;
            padding-bottom:14px; margin-bottom:18px;
            border-bottom:1px solid var(--eor-border);
        }}
        .eor-brand {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
        .dz-flag {{ width:44px; height:30px; position:relative; overflow:hidden; flex:0 0 44px; border:1px solid var(--eor-border); border-radius:2px; background:linear-gradient(90deg,#006233 0 50%,#fff 50%); }}
        .dz-flag-crescent {{ position:absolute; width:16px; height:16px; left:14px; top:6px; border-radius:50%; background:#d21034; }}
        .dz-flag-crescent:after {{ content:''; position:absolute; width:13px; height:13px; left:5px; top:-3px; border-radius:50%; background:#fff; }}
        .dz-flag-star {{ position:absolute; left:24px; top:7px; color:#d21034; font-size:11px; line-height:1; }}
        .eor-brand h1 {{ margin:0; font-size:1.5rem; font-weight:600; letter-spacing:-0.01em; line-height:1.25; }}
        .eor-subtitle {{ color:var(--eor-text-muted); font-size:0.92rem; margin:0; }}

        /* ---- Sidebar section labels ------------------------------------ */
        .eor-section-label {{
            display:flex; align-items:center; gap:6px;
            font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;
            color:var(--eor-text-muted); margin:18px 0 6px 0;
        }}
        .eor-section-label:first-of-type {{ margin-top:4px; }}

        /* ---- KPI / metric cards ------------------------------------------ */
        [data-testid="stMetric"] {{
            background:var(--eor-panel);
            border:1px solid var(--eor-border);
            border-radius:var(--eor-radius);
            padding:14px 16px 12px 16px;
        }}
        [data-testid="stMetricLabel"] {{ color:var(--eor-text-muted); font-size:0.8rem; }}
        [data-testid="stMetricValue"] {{ font-family:var(--eor-mono); font-size:1.5rem; }}

        /* ---- Status pills ------------------------------------------------ */
        .eor-status-row {{
            display:flex; align-items:center; justify-content:space-between; gap:12px;
            padding:10px 2px; border-bottom:1px solid var(--eor-border);
            flex-wrap:wrap;
        }}
        .eor-status-row:last-child {{ border-bottom:none; }}
        .eor-status-row-flush {{ border-bottom:none; padding:2px 2px 10px 2px; }}
        .eor-status-name {{ display:flex; align-items:center; gap:8px; font-weight:500; }}
        .eor-pill {{
            display:inline-flex; align-items:center; gap:4px;
            font-family:var(--eor-mono); font-size:0.72rem; font-weight:600;
            padding:3px 10px; border-radius:999px; white-space:nowrap;
        }}
        .eor-pill.pass {{ background:rgba(55,182,121,0.16); color:var(--eor-pass); }}
        .eor-pill.marginal {{ background:rgba(232,163,61,0.16); color:var(--eor-marginal); }}
        .eor-pill.fail {{ background:rgba(229,72,77,0.16); color:var(--eor-fail); }}

        /* ---- Combined-result bars (responsive, no fixed px widths) ------- */
        .eor-combined-row {{
            display:flex; align-items:center; gap:12px; flex-wrap:wrap;
            padding:8px 0;
        }}
        .eor-combined-name {{ flex:1 1 180px; min-width:140px; font-weight:500; }}
        .eor-combined-bar-wrap {{ flex:3 1 220px; min-width:140px; background:var(--eor-panel-raised); height:10px; border-radius:6px; overflow:hidden; }}
        .eor-combined-bar-fill {{ height:100%; border-radius:6px; }}
        .eor-combined-pct {{ flex:0 0 56px; text-align:right; font-family:var(--eor-mono); font-size:0.85rem; }}

        /* ---- Native alert boxes (st.info/warning/success/error) match the
           same card language (radius + hairline border) as the rest of the
           design system, instead of Streamlit's slightly different default
           rounding/padding sitting next to custom cards. Only cosmetic
           properties are touched; color/icon logic is left to Streamlit. */
        [data-testid="stAlert"] {{
            border-radius:var(--eor-radius) !important;
            border:1px solid var(--eor-border) !important;
        }}

        /* ---- Bordered containers (st.container(border=True)) reuse the
           same panel background/radius as KPI cards for one consistent
           "card" language across the app. */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius:var(--eor-radius) !important;
            border-color:var(--eor-border) !important;
        }}

        /* ---- Numeric / mono utility -------------------------------------- */
        .eor-mono {{ font-family:var(--eor-mono); }}
        .eor-caption-like {{ color:var(--eor-text-muted); font-size:0.85rem; margin:2px 0 10px 0; }}

        /* ---- Mobile tightening -------------------------------------------- */
        @media (max-width: 640px) {{
            .eor-brand h1 {{ font-size:1.2rem; }}
            .eor-subtitle {{ font-size:0.82rem; }}
            .eor-combined-name {{ flex-basis:100%; }}
            .eor-combined-pct {{ flex:0 0 auto; }}
            /* Technical unit-conversion note: useful on a wide desktop
               screen, just clutter on a phone - hidden below this width. */
            .eor-desktop-only {{ display:none !important; }}
        }}

        /* Persistent accent and focus treatment without a distracting pulse. */
        [data-testid="collapsedControl"] {{
            background:var(--eor-accent) !important;
            border-radius:8px !important;
            padding:6px !important;
            box-shadow:0 2px 8px rgba(0,0,0,0.35);
        }}
        [data-testid="collapsedControl"] svg {{ color:#0B1220 !important; fill:#0B1220 !important; }}

        </style>""",
        unsafe_allow_html=True,
    )


def plausibility_warnings(permeability: float, porosity_pct: float, field_units: Dict[str, float]) -> List[str]:
    """Flag inputs well outside typical conventional-reservoir engineering
    ranges. These are general plausibility bounds, not the exact min/max of
    the ML models' training data (which isn't available to check directly) -
    the point is to catch likely typos or unit-entry mistakes, not to make a
    precise in-distribution/out-of-distribution claim about the models.
    """
    checks = [
        (permeability, 0.1, 2000.0, "Permeability", "mD"),
        (porosity_pct, 1.0, 40.0, "Porosity", "%"),
        (field_units["API_Gravity"], 10.0, 55.0, "API Gravity", "°API"),
        (field_units["Oil_Viscosity"], 0.01, 50.0, "Oil Viscosity", "cP"),
        (field_units["Temperature"], 50.0, 356.0, "Temperature", "°F"),
        (field_units["Depth_ft"], 300.0, 20000.0, "Depth", "ft"),
        (field_units["Pressure_psi"], 70.0, 10000.0, "Pressure", "psi"),
    ]
    warnings_out = []
    for value, lo, hi, label, unit in checks:
        if value < lo or value > hi:
            warnings_out.append(f"{label} ({value:.2f} {unit}) is well outside the typical conventional-reservoir range ({lo:g}-{hi:g} {unit}) - double-check the value and its unit.")
    return warnings_out


def status_pill_html(status: str) -> str:
    """Return a small colored pill span for PASS / MARGINAL / FAIL.

    Streamlit's ":material/xxx:" icon shortcode is only converted inside
    Streamlit's own text elements (headers, tabs, st.info/warning/success,
    the "icon=" argument, etc). It is documented to NOT be converted when
    unsafe_allow_html=True is used, so it must not be used inside this raw
    HTML string - it would just show as literal text. Plain Unicode
    characters are used here instead, which render everywhere reliably.
    """
    key = {"PASS": "pass", "MARGINAL": "marginal", "FAIL": "fail"}.get(status, "pass")
    symbol = {"PASS": "✓", "MARGINAL": "⚠", "FAIL": "✕"}.get(status, "✓")
    return f"<span class='eor-pill {key}'>{symbol} {status}</span>"


def convert_to_field_units(depth: float, depth_unit: str, pressure: float, pressure_unit: str,
                           temperature: float, temperature_unit: str, viscosity: float,
                           viscosity_unit: str, gravity: float, gravity_unit: str) -> Dict[str, float]:
    """Normalize user inputs to ft, psi, Fahrenheit, cP, and API gravity."""
    depth_ft = depth if depth_unit == "ft" else depth * 3.280839895
    pressure_psi = pressure if pressure_unit == "psi" else pressure * {"bar": 14.5037738, "MPa": 145.037738}.get(pressure_unit, 1.0)
    temperature_f = temperature if temperature_unit == "°F" else temperature * 9 / 5 + 32
    viscosity_cp = float(viscosity)  # 1 mPa.s = 1 cP
    gravity_api = gravity if gravity_unit == "°API" else 141.5 / max(float(gravity), 1e-6) - 131.5
    return {
        "Depth_ft": float(depth_ft), "Pressure_psi": float(pressure_psi),
        "Temperature": float(temperature_f), "Oil_Viscosity": float(viscosity_cp),
        "API_Gravity": float(gravity_api),
    }


def standardize_eor_method_v3(raw_str: Any) -> str | None:
    """Map the workbook's indicator-column names to the six V5 target classes."""
    text = str(raw_str).lower().strip()
    if any(keyword in text for keyword in ["steam", "thermal", "combustion", "fire", "hot water"]):
        return None
    if "wag" in text:
        return "WAG Injection"
    if "miscible" in text and "immiscible" not in text:
        return "Miscible Gas Injection"
    if "immiscible" in text and "wag" not in text:
        return "Immiscible Gas Injection"
    if any(keyword in text for keyword in ["foam", "sag", "surfactant", "micellar", "alkaline"]):
        return "Hybrid Gas-Chemical"
    if any(keyword in text for keyword in ["polymer", "chemical", "asp", "meor", "soap"]):
        return "Chemical EOR"
    if any(keyword in text for keyword in ["gas", "co2", "hydrocarbon"]):
        return "Miscible Gas Injection"
    if "water" in text or "flooding" in text:
        return "Secondary Waterflooding"
    return None


def load_and_clean_data() -> tuple[pd.DataFrame, pd.Series]:
    """Read Screening_Original.xlsx and reproduce the V5 training target."""
    workbook = Path(__file__).resolve().parent / "Screening_Original.xlsx"
    if not workbook.exists():
        raise FileNotFoundError(f"Training workbook not found: {workbook}")

    data = pd.read_excel(workbook, sheet_name="MAIN")
    data.columns = data.columns.astype(str).str.strip()
    required = {
        "Lithology": "Producing Horizon Lithology.1",
        "Permeability": "Permeability (md)",
        "Porosity": "Reservoir Porosity (%)",
        "Oil_Viscosity": "Viscosity (CP)",
        "Temperature": "Temperature (F)",
        "API_Gravity": "Oil Gravity (API)",
    }
    missing = [column for column in required.values() if column not in data.columns]
    if missing:
        raise KeyError(f"Missing training columns: {missing}")

    frame = pd.DataFrame({key: data[column] for key, column in required.items()})
    numeric_fields = ["Permeability", "Porosity", "Oil_Viscosity", "Temperature", "API_Gravity"]
    for field in numeric_fields:
        frame[field] = pd.to_numeric(
            frame[field].astype(str).str.replace(r"[^0-9eE+\-.]", "", regex=True),
            errors="coerce",
        )
    frame["Porosity"] = frame["Porosity"].apply(normalize_percentage_to_fraction)

    source_columns = set(required.values()) | {"IOR/EOR Method", "IOR/EOR Method.1"}
    method_columns = [column for column in data.columns if column not in source_columns]

    def extract_method(row: pd.Series) -> str | None:
        for column in method_columns:
            value = row.get(column)
            if value == 1 or value is True or str(value).strip() == "1":
                return column
        return None

    raw_methods = data.apply(extract_method, axis=1)
    target = raw_methods.apply(standardize_eor_method_v3)
    valid = target.notna() & frame.notna().all(axis=1)
    return frame.loc[valid].reset_index(drop=True), target.loc[valid].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def normalize_percentage_to_fraction(value: float) -> float:
    """Convert percentage to decimal fraction if needed."""
    v = safe_float(value, 0.0)
    if v > 1.0 and v <= 100.0:
        return v / 100.0
    return v


def as_lower_list(values: Iterable[str]) -> List[str]:
    return [str(v).strip().lower() for v in values if str(v).strip()]


def find_model_file(root_dir: Path, candidates: List[str]) -> Path | None:
    """Find a model artifact by searching a prioritized list of names."""
    for candidate in candidates:
        path = root_dir / candidate
        if path.exists():
            return path
    return None


def ensure_feature_columns(data: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
    """Add missing features with safe defaults to keep the model pipeline stable."""
    out = data.copy()
    for col in expected_columns:
        if col not in out.columns:
            out[col] = 0.0
    return out[expected_columns]


def lithology_to_workbook_code(lithology: Any) -> int:
    """Convert the UI lithology label to the workbook's numeric lithology code."""
    text = str(lithology).lower()
    if "quartz" in text or "mixed clastic" in text:
        return 3
    if "sandstone" in text:
        return 1
    if "limestone" in text:
        return 8
    if "dolomite" in text or "carbonate" in text:
        return 2
    return 1


# ---------------------------------------------------------------------------
# 1. Physical proxy calculator
# ---------------------------------------------------------------------------

def calculate_physical_proxies(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute key domain-informed proxy features from reservoir properties."""
    out = frame.copy()

    perm = pd.to_numeric(out["Permeability"], errors="coerce").fillna(0.0)
    por = pd.to_numeric(out["Porosity"], errors="coerce").fillna(0.0)
    visc = pd.to_numeric(out["Oil_Viscosity"], errors="coerce").fillna(0.0)

    out["Mobility_Proxy"] = perm / (visc + 1e-6)

    # RQI = 0.0314 * sqrt(K / phi)
    phi_safe = por.clip(lower=1e-6)
    out["RQI"] = 0.0314 * np.sqrt(perm / phi_safe)

    # Phi_z = por / (1 - por)
    phi_z = phi_safe / (1.0 - phi_safe + 1e-6)
    out["FZI"] = out["RQI"] / (phi_z + 1e-6)

    return out


def add_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    """V5-compatible alias for the physical proxy feature builder."""
    return calculate_physical_proxies(frame)


# ---------------------------------------------------------------------------
# 2. Classical Table / screening engine
# ---------------------------------------------------------------------------

def evaluate_status(value: float, ideal_min: float, ideal_max: float, marginal_min: float, marginal_max: float) -> str:
    if ideal_min <= value <= ideal_max:
        return "PASS"
    if marginal_min <= value <= marginal_max:
        return "MARGINAL"
    return "FAIL"


def classical_screening(lithology: str, perm: float, por: float, visc: float, temp: float, api: float, depth_ft: float = 0.0, pressure_psi: float | None = None, mmp_psi: float | None = None) -> Dict[str, Dict[str, Any]]:
    """Apply preliminary heuristic rules, not a substitute for PVT or MMP studies."""
    results: Dict[str, Dict[str, Any]] = {}

    pressure_available = pressure_psi is not None and pressure_psi > 0
    mmp_available = mmp_psi is not None and mmp_psi > 0
    if pressure_available and mmp_available:
        pressure_value = float(pressure_psi)
        mmp_value = float(mmp_psi)
        pressure_status = "PASS" if pressure_value >= 1.1 * mmp_value else "MARGINAL" if pressure_value >= mmp_value else "FAIL"
        pressure_detail = f"P_res={pressure_value:.0f} psi; MMP={mmp_value:.0f} psi"
        pressure_note = "Pressure/MMP gate applied."
    else:
        pressure_status = "PASS" if depth_ft >= 4000 else "MARGINAL" if depth_ft >= 3000 else "FAIL"
        pressure_detail = f"Depth proxy={depth_ft:.0f} ft"
        pressure_note = "Depth is acting as a proxy because P_res or MMP was not provided."

    miscible_params = {
        "Pressure vs MMP": pressure_status,
        "Oil Viscosity": evaluate_status(visc, 0.0, 10.0, 0.0, 25.0),
        "API Gravity": evaluate_status(api, 22.0, 100.0, 20.0, 100.0),
        "Porosity": evaluate_status(por, 0.03, 1.0, 0.025, 1.0),
        "Pressure basis": pressure_detail,
    }
    miscible_status = "FAIL" if "FAIL" in miscible_params.values() else "MARGINAL" if "MARGINAL" in miscible_params.values() else "PASS"
    miscible_violations = [pressure_note] if not (pressure_available and mmp_available) else []
    if pressure_status == "FAIL" and pressure_available and mmp_available:
        miscible_violations.append("Reservoir pressure is below MMP.")
    if miscible_params["Oil Viscosity"] == "FAIL":
        miscible_violations.append("Oil viscosity exceeds 25 cP.")
    results["Miscible Gas Injection"] = {"status": miscible_status, "parameters": miscible_params, "violations": miscible_violations}

    immiscible_params = {
        "API Gravity": evaluate_status(api, 11.0, 35.0, 8.0, 40.0),
        "Oil Viscosity": evaluate_status(visc, 0.6, 592.0, 0.1, 1000.0),
        "Permeability": evaluate_status(perm, 30.0, 10000.0, 20.0, 10000.0),
        "Porosity": evaluate_status(por, 0.15, 1.0, 0.10, 1.0),
    }
    immiscible_fails = sum(1 for v in immiscible_params.values() if v == "FAIL")
    if immiscible_fails == 0:
        immiscible_status = "PASS"
    elif immiscible_fails <= 1:
        immiscible_status = "MARGINAL"
    else:
        immiscible_status = "FAIL"
    results["Immiscible Gas Injection"] = {
        "status": immiscible_status,
        "parameters": immiscible_params,
        "violations": [k for k, v in immiscible_params.items() if v == "FAIL"],
    }

    wag_params = {
        "Permeability": evaluate_status(perm, 10.0, 10000.0, 2.0, 10000.0),
        "Oil Viscosity": evaluate_status(visc, 0.0, 5.0, 0.0, 15.0),
        "API Gravity": evaluate_status(api, 20.0, 100.0, 15.0, 100.0),
        "Porosity": evaluate_status(por, 0.10, 1.0, 0.05, 1.0),
    }
    wag_fails = sum(1 for v in wag_params.values() if v == "FAIL")
    if wag_fails == 0:
        wag_status = "PASS"
    elif wag_fails <= 1:
        wag_status = "MARGINAL"
    else:
        wag_status = "FAIL"
    results["WAG Injection"] = {
        "status": wag_status,
        "parameters": wag_params,
        "violations": [k for k, v in wag_params.items() if v == "FAIL"],
    }

    chemical_params = {
        "Temperature": evaluate_status(temp, 0.0, 180.0, 0.0, 220.0),
        "Oil Viscosity": evaluate_status(visc, 0.0, 150.0, 0.0, 1000.0),
        "Permeability": evaluate_status(perm, 15.0, 10000.0, 5.0, 10000.0),
        "API Gravity": evaluate_status(api, 13.0, 42.5, 10.0, 45.0),
    }
    chemical_fails = sum(1 for v in chemical_params.values() if v == "FAIL")
    if chemical_fails == 0:
        chemical_status = "PASS"
    elif chemical_fails <= 1:
        chemical_status = "MARGINAL"
    else:
        chemical_status = "FAIL"
    results["Chemical EOR"] = {
        "status": chemical_status,
        "parameters": chemical_params,
        "violations": [k for k, v in chemical_params.items() if v == "FAIL"],
    }

    gas_status = "PASS" if miscible_status == "PASS" or wag_status == "PASS" else "MARGINAL" if miscible_status == "MARGINAL" or wag_status == "MARGINAL" else "FAIL"
    hybrid_status = "FAIL" if gas_status == "FAIL" or chemical_status == "FAIL" else "PASS" if gas_status == "PASS" and chemical_status == "PASS" else "MARGINAL"
    results["Hybrid Gas-Chemical"] = {
        "status": hybrid_status,
        "parameters": {
            "Gas displacement driver": gas_status,
            "Chemical/surfactant stabilizer": chemical_status,
        },
        "violations": [
            f"Gas displacement driver is {gas_status}." if gas_status != "PASS" else "",
            f"Chemical/surfactant stabilizer is {chemical_status}." if chemical_status != "PASS" else "",
        ],
    }
    results["Hybrid Gas-Chemical"]["violations"] = [item for item in results["Hybrid Gas-Chemical"]["violations"] if item]

    waterflood_params = {
        "API Gravity": evaluate_status(api, 15.0, 35.0, 10.0, 40.0),
        "Oil Viscosity": evaluate_status(visc, 0.0, 100.0, 0.0, 200.0),
        "Permeability": evaluate_status(perm, 10.0, 10000.0, 5.0, 10000.0),
        "Porosity": evaluate_status(por, 0.10, 1.0, 0.08, 1.0),
    }
    waterflood_fails = sum(1 for v in waterflood_params.values() if v == "FAIL")
    if waterflood_fails == 0:
        water_status = "PASS"
    elif waterflood_fails <= 1:
        water_status = "MARGINAL"
    else:
        water_status = "FAIL"
    results["Secondary Waterflooding"] = {
        "status": water_status,
        "parameters": waterflood_params,
        "violations": [k for k, v in waterflood_params.items() if v == "FAIL"],
    }

    return results


# ---------------------------------------------------------------------------
# 3. V5-style feature builder
# ---------------------------------------------------------------------------

def build_classical_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Pass_Miscible"] = (
        (out["API_Gravity"] >= 22)
        & (out["Oil_Viscosity"] <= 35)
        & (out["Permeability"] >= 1.5)
        & (out["Porosity"] >= 0.03)
    ).astype(int)

    out["Pass_Immiscible"] = (
        (out["API_Gravity"].between(11, 35))
        & (out["Oil_Viscosity"].between(0.6, 592))
        & (out["Permeability"] >= 30)
    ).astype(int)

    out["Pass_WAG"] = (
        (out["API_Gravity"].between(33, 39))
        & (out["Oil_Viscosity"].between(0.3, 0.9))
        & (out["Permeability"].between(130, 1000))
    ).astype(int)

    out["Pass_Chemical"] = (
        (out["API_Gravity"].between(13, 42.5))
        & (out["Oil_Viscosity"] <= 6500)
        & (out["Permeability"] >= 1.8)
    ).astype(int)

    out["Pass_Hybrid"] = (
        (out["Pass_Chemical"] == 1)
        & ((out["Pass_Miscible"] == 1) | (out["Pass_WAG"] == 1))
    ).astype(int)

    out["Pass_Waterflood"] = 1
    return out


def build_v5_feature_frame(user_input: Dict[str, Any]) -> pd.DataFrame:
    """Construct the same feature set expected by the V5 pipeline."""
    record = {
        "Lithology": lithology_to_workbook_code(user_input.get("Lithology", "Carbonate")),
        "Permeability": safe_float(user_input.get("Permeability"), 120.0),
        "Porosity": normalize_percentage_to_fraction(safe_float(user_input.get("Porosity"), 18.0)),
        "Oil_Viscosity": safe_float(user_input.get("Oil_Viscosity"), 8.0),
        "Temperature": safe_float(user_input.get("Temperature"), 180.0),
        "API_Gravity": safe_float(user_input.get("API_Gravity"), 32.0),
    }

    frame = pd.DataFrame([record])
    frame = calculate_physical_proxies(frame)
    frame = build_classical_flags(frame)
    return frame


# ---------------------------------------------------------------------------
# 4. Model loading (trained weights or fallback mode)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_bundle() -> Dict[str, Any]:
    """Load saved artifacts or train the real V5 ensemble from the Excel dataset."""
    root_dir = Path(__file__).resolve().parent

    artifact_paths = {
        "lightgbm": find_model_file(root_dir, ["lightgbm_v5.joblib", "lightgbm_v5.pkl", "lightgbm_v5.model"]),
        "xgboost": find_model_file(root_dir, ["xgboost_v5.joblib", "xgboost_v5.pkl", "xgboost_v5.model"]),
        "rf": find_model_file(root_dir, ["random_forest_v5.joblib", "random_forest_v5.pkl", "rf_v5.joblib", "rf_v5.pkl"]),
        "scaler": find_model_file(root_dir, ["v5_scaler.joblib", "v5_scaler.pkl", "scaler.joblib", "scaler.pkl"]),
        "lithology_encoder": find_model_file(root_dir, ["v5_lithology_encoder.joblib", "lithology_encoder.joblib"]),
        "target_map": find_model_file(root_dir, ["v5_target_mapping.json", "target_mapping.json"]),
        "feature_names": find_model_file(root_dir, ["v5_feature_names.json", "feature_names.json"]),
    }

    loaded = {"status": "training", "warning": ""}

    if joblib is None:
        loaded["warning"] = "joblib is not installed; training models directly from Excel."

    all_found = all(path is not None for path in artifact_paths.values())
    if all_found:
        try:
            models = {}
            if artifact_paths["lightgbm"] is not None:
                models["LightGBM V5"] = joblib.load(artifact_paths["lightgbm"])
            if artifact_paths["xgboost"] is not None:
                models["XGBoost"] = joblib.load(artifact_paths["xgboost"])
            if artifact_paths["rf"] is not None:
                models["Random Forest"] = joblib.load(artifact_paths["rf"])

            scaler = joblib.load(artifact_paths["scaler"])
            with open(artifact_paths["target_map"], "r", encoding="utf-8") as f:
                target_map = json.load(f)
            with open(artifact_paths["feature_names"], "r", encoding="utf-8") as f:
                feature_names = json.load(f)

            loaded = {
                "status": "loaded",
                "models": models,
                "scaler": scaler,
                "lith_encoder": joblib.load(artifact_paths["lithology_encoder"]),
                "target_map": {int(k): v for k, v in target_map.items()},
                "feature_names": feature_names,
                "warning": "",
            }
        except Exception as exc:  # pragma: no cover
            loaded["warning"] = f"Model artifact loading failed: {exc}. Falling back to heuristic mode."

    if loaded.get("status") == "loaded":
        return loaded

    # The workbook is the training data. Train real models when pre-exported
    # artifacts are absent; do not silently substitute heuristic probabilities.
    try:
        X_raw, y_raw = load_and_clean_data()
        valid_classes = y_raw.value_counts()
        valid_classes = valid_classes[valid_classes >= 2].index
        mask = y_raw.isin(valid_classes)
        X_raw, y_raw = X_raw.loc[mask].copy(), y_raw.loc[mask].copy()

        lith_encoder = LabelEncoder()
        lith_encoder.fit(X_raw["Lithology"].astype(str))
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y_raw.astype(str))

        combined = add_physics_features(build_classical_flags(X_raw))
        combined["Lithology"] = lith_encoder.transform(combined["Lithology"].astype(str))
        feature_names = combined.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            combined, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "LightGBM V5": lgb.LGBMClassifier(
                n_estimators=200, class_weight="balanced", random_state=42,
                max_depth=6, learning_rate=0.05, verbose=-1,
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=200, random_state=42, max_depth=8,
                learning_rate=0.1, eval_metric="mlogloss", n_jobs=-1,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, class_weight="balanced", max_depth=15,
                random_state=42, n_jobs=-1,
            ),
        }

        trained_models = {}
        metrics = {}
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            trained_models[model_name] = model
            metrics[model_name] = {
                "accuracy": accuracy_score(y_test, predictions) * 100,
                "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
            }

        return {
            "status": "trained_from_excel",
            "models": trained_models,
            "scaler": scaler,
            "lith_encoder": lith_encoder,
            "target_map": {idx: name for idx, name in enumerate(target_encoder.classes_)},
            "feature_names": feature_names,
            "metrics": metrics,
            "warning": "Real models were trained from Screening_Original.xlsx and cached for this session.",
        }
    except Exception as exc:
        return {
            "status": "fallback",
            "warning": f"Real model training failed: {exc}",
        }


# ---------------------------------------------------------------------------
# 5. Fallback probability engine
# ---------------------------------------------------------------------------

def heuristic_probability_by_rules(user_input: Dict[str, Any]) -> Dict[str, float]:
    """Generate a safe probability distribution when real model artifacts are missing."""
    perm = safe_float(user_input.get("Permeability"), 120.0)
    por = normalize_percentage_to_fraction(safe_float(user_input.get("Porosity"), 18.0))
    visc = safe_float(user_input.get("Oil_Viscosity"), 8.0)
    api = safe_float(user_input.get("API_Gravity"), 32.0)

    scores = {name: 0.10 for name in EOR_CLASSES}

    # Dominant tendency heuristics based on the V5 domain rules.
    if api >= 22 and visc <= 35 and perm >= 1.5 and por >= 0.03:
        scores["Miscible Gas Injection"] += 0.35
    if api >= 11 and api <= 35 and visc >= 0.6 and visc <= 592 and perm >= 30:
        scores["Immiscible Gas Injection"] += 0.30
    if 30 <= api <= 42 and 0.2 <= visc <= 1.2 and 100 <= perm <= 2000 and por >= 0.10:
        scores["WAG Injection"] += 0.20
    if api >= 13 and visc <= 6500 and perm >= 1.8:
        scores["Chemical EOR"] += 0.35
    if api >= 22 and visc <= 35 and perm >= 1.5 and por >= 0.03:
        scores["Hybrid Gas-Chemical"] += 0.10
    if visc <= 100 and perm >= 10 and por >= 0.10:
        scores["Secondary Waterflooding"] += 0.15

    # Bias toward hybrid/chemical for moderate-viscosity carbonate reservoirs.
    if "carbonate" in str(user_input.get("Lithology", "")).lower():
        scores["Hybrid Gas-Chemical"] += 0.20
        scores["Chemical EOR"] += 0.15

    # Normalize to probability-like values.
    total = sum(scores.values())
    if total <= 0:
        return {name: 1.0 / len(EOR_CLASSES) for name in EOR_CLASSES}
    return {name: max(0.0, value / total) for name, value in scores.items()}


# ---------------------------------------------------------------------------
# 6. Ensemble inference with loaded models or fallback
# ---------------------------------------------------------------------------

def predict_ml_system(user_input: Dict[str, Any]) -> Dict[str, Any]:
    """Return a model prediction dictionary for each available model or fallback engine."""
    model_bundle = load_model_bundle()

    if model_bundle.get("status") not in {"loaded", "trained_from_excel"}:
        return {
            "mode": "fallback",
            "warning": model_bundle.get("warning", "Fallback mode active."),
            "results": {
                "Heuristic Engine": {
                    "prediction": max(heuristic_probability_by_rules(user_input).items(), key=lambda x: x[1])[0],
                    "probabilities": heuristic_probability_by_rules(user_input),
                }
            },
        }

    # Real model path.
    feature_frame = build_v5_feature_frame(user_input)
    feature_names = model_bundle["feature_names"]

    lith_encoder = model_bundle.get("lith_encoder")
    if lith_encoder is not None and "Lithology" in feature_frame.columns:
        feature_frame["Lithology"] = lith_encoder.transform(feature_frame["Lithology"].astype(str))

    # Ensure required features exist.
    for feature in feature_names:
        if feature not in feature_frame.columns:
            feature_frame[feature] = 0.0

    feature_frame = feature_frame[feature_names]
    X_scaled = model_bundle["scaler"].transform(feature_frame)

    results = {}
    for model_name, model in model_bundle["models"].items():
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[0]
            else:
                raise AttributeError("Model does not expose predict_proba.")

            target_mapping = model_bundle["target_map"]
            probs = {target_mapping[int(idx)]: float(prob) for idx, prob in enumerate(proba)}
            predicted = max(probs.items(), key=lambda x: x[1])[0]
            results[model_name] = {
                "prediction": predicted,
                "probabilities": probs,
            }
        except Exception as exc:
            results[model_name] = {
                "prediction": "Unavailable",
                "probabilities": {cls: 0.0 for cls in EOR_CLASSES},
                "error": str(exc),
            }

    return {"mode": "trained" if model_bundle.get("status") == "trained_from_excel" else "loaded", "warning": model_bundle.get("warning", ""), "results": results}


# ---------------------------------------------------------------------------
# 7. Dashboard visuals
# ---------------------------------------------------------------------------

def plot_probability_chart(probabilities: Dict[str, float], title: str = "") -> go.Figure:
    # No in-chart title: the markdown header placed just above this chart
    # (e.g. the model name) already states it, so repeating it here would
    # just be visual noise. The "title" argument is kept for compatibility
    # with any other caller, but is intentionally not rendered.
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color="#2E8B6B")])
    fig.update_layout(
        xaxis_title="EOR Method",
        yaxis_title="Probability",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#E8EDF4"),
        height=400,
        margin=dict(l=20, r=20, t=10, b=80),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Main app layout
# ---------------------------------------------------------------------------

def render_kpi_cards(inputs: Dict[str, Any]) -> None:
    """Render key physical proxy metrics as KPI cards."""
    record = {
        "Permeability": safe_float(inputs.get("Permeability"), 120.0),
        "Porosity": normalize_percentage_to_fraction(safe_float(inputs.get("Porosity"), 18.0)),
        "Oil_Viscosity": safe_float(inputs.get("Oil_Viscosity"), 8.0),
    }
    mobility = record["Permeability"] / (record["Oil_Viscosity"] + 1e-6)
    rqi = 0.0314 * math.sqrt(record["Permeability"] / (record["Porosity"] + 1e-6))
    phi_z = record["Porosity"] / (1.0 - record["Porosity"] + 1e-6)
    fzi = rqi / (phi_z + 1e-6)

    kpi_cols = st.columns(3)
    kpi_cols[0].metric(
        ":material/speed: Mobility Proxy", f"{mobility:.3f}",
        help="Ratio describing how easily displacing fluid moves relative to the oil in place. Higher values generally favor miscible/gas methods; lower values favor waterflooding or chemical EOR.",
    )
    kpi_cols[1].metric(
        ":material/water_drop: RQI", f"{rqi:.3f}",
        help="Rock Quality Index: a permeability-porosity proxy for flow-path quality within the reservoir rock. Higher values indicate better-connected, higher-quality flow paths.",
    )
    kpi_cols[2].metric(
        ":material/grain: FZI", f"{fzi:.3f}",
        help="Flow Zone Indicator: groups rock into hydraulic units with similar pore-throat characteristics. Used alongside RQI to characterize reservoir quality and heterogeneity.",
    )


# ---------------------------------------------------------------------------
# 9. Streamlit application entry
# ---------------------------------------------------------------------------

def main() -> None:
    with st.sidebar:
        language_name = st.selectbox("Language / Langue / اللغة", list(LANGUAGES), index=0)
    language = LANGUAGES[language_name]
    apply_language_css(language)

    flag_html = (
        "<div class='dz-flag' role='img' aria-label='Algeria flag'>"
        "<span class='dz-flag-crescent'></span><span class='dz-flag-star'>★</span></div>"
    )
    st.markdown(
        f"""<div class='eor-header'>
            <div class='eor-brand'>{flag_html}<h1>{t(language, 'title')}</h1></div>
            <p class='eor-subtitle'>{t(language, 'subtitle')}</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.info(t(language, "worked_example"), icon=":material/edit_note:")

    with st.sidebar:
        st.header(f":material/tune: {t(language, 'inputs')}")
        with st.expander(t(language, "guide"), expanded=False, icon=":material/map:"):
            st.write(t(language, "guide_text"))
            st.dataframe(GEOLOGICAL_LITHOLOGY_GUIDE, hide_index=True, width="stretch")

        st.markdown(
            f"<div class='eor-section-label'>{t(language, 'section_project')}</div>",
            unsafe_allow_html=True,
        )
        project_cols = st.columns(2)
        project_name = project_cols[0].text_input(t(language, "project"), value="Project 1")
        producing_horizon = project_cols[1].text_input(t(language, "horizon"), value="Trias S1")

        st.markdown(
            f"<div class='eor-section-label'>{t(language, 'section_rock')}</div>",
            unsafe_allow_html=True,
        )
        lithology_options = ["Carbonate", "Carbonate / Dolomite", "Carbonate / Limestone", "Sandstone", "Sandstone / Quartzite", "Mixed Clastic", "Unknown"]
        lithology = st.selectbox(t(language, "lithology"), lithology_options, index=lithology_options.index("Sandstone"))

        permeability = st.number_input(f"{t(language, 'permeability')} (mD)", value=70.0, min_value=0.1, step=1.0)
        porosity_pct = st.number_input(f"{t(language, 'porosity')} (%)", value=9.3, min_value=0.1, max_value=60.0, step=0.1)

        st.markdown(
            f"<div class='eor-section-label'>{t(language, 'section_depth')}</div>",
            unsafe_allow_html=True,
        )
        depth_unit = st.selectbox(f"{t(language, 'depth')} unit", ["m", "ft"])
        depth = st.number_input(f"{t(language, 'depth')} ({depth_unit})", value=3200.0 if depth_unit == "m" else 10500.0, min_value=1.0, step=100.0)
        pressure_unit = st.selectbox(f"{t(language, 'pressure')} unit", ["bar", "psi", "MPa"])
        pressure = st.number_input(f"{t(language, 'pressure')} ({pressure_unit})", value=190.0 if pressure_unit == "bar" else (2756.0 if pressure_unit == "psi" else 19.0), min_value=0.1, step=10.0)
        mmp = st.number_input(f"MMP ({pressure_unit}, optional)", value=270.0 if pressure_unit == "bar" else (3916.0 if pressure_unit == "psi" else 27.0), min_value=0.0, step=5.0 if pressure_unit == "bar" else 50.0, help="Enter 0 when MMP is unavailable; depth will be used only as a proxy for miscible-gas screening.")
        mmp_psi = mmp if pressure_unit == "psi" else mmp * {"bar": 14.5037738, "MPa": 145.037738}.get(pressure_unit, 1.0)

        st.markdown(
            f"<div class='eor-section-label'>{t(language, 'section_fluid')}</div>",
            unsafe_allow_html=True,
        )
        temperature_unit = st.selectbox(f"{t(language, 'temperature')} unit", ["°C", "°F"])
        temperature = st.number_input(f"{t(language, 'temperature')} ({temperature_unit})", value=102.0 if temperature_unit == "°C" else 215.6, min_value=-50.0, step=1.0)
        viscosity_unit = st.selectbox(f"{t(language, 'viscosity')} unit", ["cP", "mPa·s"])
        viscosity = st.number_input(f"{t(language, 'viscosity')} ({viscosity_unit})", value=0.30, min_value=0.001, step=0.1)
        gravity_unit = st.selectbox(f"{t(language, 'gravity')} unit", ["°API", "Specific Gravity (SG)"])
        gravity = st.number_input(f"{t(language, 'gravity')} ({gravity_unit})", value=43.0 if gravity_unit == "°API" else 0.811, min_value=0.01, step=0.1)

    field_units = convert_to_field_units(depth, depth_unit, pressure, pressure_unit, temperature, temperature_unit, viscosity, viscosity_unit, gravity, gravity_unit)
    input_data = {
        "Lithology": lithology, "Permeability": permeability, "Porosity": porosity_pct,
        "Oil_Viscosity": field_units["Oil_Viscosity"], "Temperature": field_units["Temperature"],
        "API_Gravity": field_units["API_Gravity"], "Depth_ft": field_units["Depth_ft"], "Pressure_psi": field_units["Pressure_psi"],
    }

    for warning_text in plausibility_warnings(permeability, porosity_pct, field_units):
        st.warning(warning_text, icon=":material/warning:")

    st.subheader(f":material/analytics: {t(language, 'proxies')}")
    if project_name or producing_horizon:
        st.caption(f"<span class='eor-mono'>{project_name}, {producing_horizon} ({lithology})</span>", unsafe_allow_html=True)
    render_kpi_cards(input_data)
    st.markdown(
        f"<div class='eor-desktop-only eor-caption-like'>"
        f"{t(language, 'pressure_note')} "
        f"<span class='eor-mono'>Normalized: {field_units['Depth_ft']:.1f} ft | "
        f"{field_units['Pressure_psi']:.1f} psi | {field_units['Temperature']:.1f} °F | "
        f"{field_units['Oil_Viscosity']:.3f} cP | {field_units['API_Gravity']:.2f} °API</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    classical_results = classical_screening(lithology, permeability, normalize_percentage_to_fraction(porosity_pct), field_units["Oil_Viscosity"], field_units["Temperature"], field_units["API_Gravity"], field_units["Depth_ft"], field_units["Pressure_psi"], mmp_psi if mmp_psi > 0 else None)
    with st.spinner("Running screening models..."):
        model_result = predict_ml_system(input_data)
    tab1, tab2, tab3 = st.tabs([
        f":material/rule: {t(language, 'tab1_short')}",
        f":material/science: {t(language, 'tab2_short')}",
        f":material/insights: {t(language, 'tab3_short')}",
    ])

    with tab1:
        st.subheader(t(language, "engine1_title"))
        st.caption(t(language, "screening_caption"))
        st.info(t(language, "engine1_disclaimer"), icon=":material/info:")
        for method_name, result in classical_results.items():
            status = result["status"]
            with st.container(border=True):
                st.markdown(
                    f"<div class='eor-status-row eor-status-row-flush'>"
                    f"<span class='eor-status-name'>{method_name}</span>{status_pill_html(status)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if result["violations"]:
                    st.warning(f"{t(language, 'violations')}: {', '.join(result['violations'])}", icon=":material/warning:")
                with st.expander(f"{t(language, 'method_details')}", icon=":material/list_alt:"):
                    st.dataframe(pd.DataFrame([{"Parameter": k, "Status": v} for k, v in result["parameters"].items()]), hide_index=True, width="stretch")

    with tab2:
        st.subheader(t(language, "ml"))
        st.caption(t(language, "engine2_caption"))
        if model_result["mode"] == "trained":
            st.success("Real ensemble models trained from Screening_Original.xlsx are active.", icon=":material/check_circle:")
        elif model_result.get("warning"):
            st.warning(t(language, "fallback_warning"), icon=":material/warning:")

        display_order = sorted(
            model_result["results"].items(),
            key=lambda item: 0 if "random forest" in item[0].lower() else 1,
        )
        prediction_cols = st.columns(len(display_order))
        for col, (model_name, info) in zip(prediction_cols, display_order):
            col.metric(model_name, info["prediction"])

        model_names = [name for name, _ in display_order]
        probability_matrix = [
            [display_order_model[1]["probabilities"].get(class_name, 0.0) for class_name in EOR_CLASSES]
            for display_order_model in display_order
        ]
        probability_text = [[f"{value:.1%}" for value in row] for row in probability_matrix]
        comparison_fig = go.Figure(data=[go.Heatmap(
            z=probability_matrix,
            x=EOR_CLASSES,
            y=model_names,
            text=probability_text,
            texttemplate="%{text}",
            colorscale=[[0.0, "#182338"], [0.5, "#286B63"], [1.0, "#46B58A"]],
            zmin=0,
            zmax=1,
            colorbar=dict(title="Probability"),
            hovertemplate="Model: %{y}<br>EOR method: %{x}<br>Probability: %{z:.1%}<extra></extra>",
        )])
        comparison_fig.update_layout(
            title=None,
            xaxis_title=t(language, "eor_method"),
            yaxis_title="Model",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#E8EDF4"),
            height=300,
            margin=dict(l=20, r=20, t=10, b=110),
        )
        st.markdown(f"##### :material/compare_arrows: {t(language, 'model_comparison')}")
        st.plotly_chart(comparison_fig, width="stretch")

        selected_model = st.selectbox(
            t(language, "model_detail"),
            model_names,
            index=0,
            key="engine_b_detail_model",
        )
        st.plotly_chart(
            plot_probability_chart(model_result["results"][selected_model]["probabilities"], ""),
            width="stretch",
        )

    with tab3:
        st.subheader(t(language, "consensus"))
        st.caption(t(language, "screening_caption"))
        results = model_result["results"]

        # Engine B now feeds the consensus/combined result from Random
        # Forest alone (more stable than averaging across all models),
        # rather than the mean across LightGBM + XGBoost + Random Forest.
        rf_key = next((name for name in results if "random forest" in name.lower()), None)
        if rf_key is not None:
            consensus_scores = {method: float(results[rf_key]["probabilities"].get(method, 0.0)) for method in EOR_CLASSES}
            score_source = rf_key
        else:
            # Fallback/heuristic mode has no "Random Forest" entry; average
            # over whatever single engine is available instead of failing.
            consensus_scores = {method: float(np.mean([item["probabilities"].get(method, 0.0) for item in results.values()])) for method in EOR_CLASSES}
            score_source = "mean of available fallback outputs"

        # Combined result (Engine A rule status applied to Engine B
        # probabilities) is the headline number, so it's shown first.
        st.markdown(f"##### :material/stacked_bar_chart: {t(language, 'combined_result')}")
        st.caption(f"Engine B source: {score_source}. Engine A applies a heuristic multiplier to produce a screening score; this is not a calibrated success probability.")
        st.markdown(
            f"{status_pill_html('PASS')} &nbsp; {status_pill_html('MARGINAL')} &nbsp; {status_pill_html('FAIL')}",
            unsafe_allow_html=True,
        )
        combined_rank = []
        for method_name in EOR_CLASSES:
            status = classical_results[method_name]["status"]
            weight = {"PASS": 1.0, "MARGINAL": 0.7, "FAIL": 0.2}[status]
            combined_rank.append((method_name, consensus_scores[method_name] * weight, status))
        combined_rank.sort(key=lambda item: item[1], reverse=True)
        bar_color = {"PASS": "#37B679", "MARGINAL": "#E8A33D", "FAIL": "#E5484D"}
        # Bars are scaled relative to the top-ranked score, not to an
        # absolute 0-100% track. With an absolute scale, scores like
        # 16%/6%/2% all render as thin slivers that are hard to tell apart;
        # relative scaling makes the ranking readable at a glance, while the
        # printed percentage label still shows the real, un-rescaled number.
        top_score = combined_rank[0][1] if combined_rank and combined_rank[0][1] > 0 else 1.0
        with st.container(border=True):
            for method_name, score, status in combined_rank:
                color = bar_color[status]
                bar_width_pct = (score / top_score) * 100.0
                st.markdown(
                    f"<div class='eor-combined-row'>"
                    f"<span class='eor-combined-name'>{method_name}</span>"
                    f"{status_pill_html(status)}"
                    f"<div class='eor-combined-bar-wrap'><div class='eor-combined-bar-fill' "
                    f"style='width:{bar_width_pct:.1f}%;background:{color};'></div></div>"
                    f"<span class='eor-combined-pct eor-mono'>{score * 100:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        fig = go.Figure(data=[go.Bar(
            x=[x[0] for x in combined_rank], y=[x[1] for x in combined_rank],
            marker_color=[bar_color[x[2]] for x in combined_rank],
        )])
        fig.update_layout(
            xaxis_title=t(language, "eor_method"), yaxis_title=t(language, "combined_score"),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#E8EDF4"),
            height=400, margin=dict(l=20, r=20, t=10, b=100),
        )
        st.plotly_chart(fig, width="stretch")

        st.divider()

        # Engine B on its own, before Engine A's rule weighting, shown second
        # as supporting detail behind the headline combined result above.
        st.markdown(f"##### :material/query_stats: {t(language, 'ml_only_result')}")
        st.caption(f"Engine B only: probabilities from {score_source}; Engine A has not modified them.")
        ml_only_rank = sorted(consensus_scores.items(), key=lambda item: item[1], reverse=True)
        st.dataframe(
            pd.DataFrame([
                {"Rank": index, "EOR Method": method, t(language, "ml_mean"): f"{score:.1%}"}
                for index, (method, score) in enumerate(ml_only_rank, start=1)
            ]),
            hide_index=True,
            width="stretch",
        )
        ml_only_fig = go.Figure(data=[go.Bar(x=[item[0] for item in ml_only_rank], y=[item[1] for item in ml_only_rank], marker_color="#3E8FD0")])
        ml_only_fig.update_layout(
            xaxis_title=t(language, "eor_method"), yaxis_title=t(language, "probability"),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#E8EDF4"),
            height=360, margin=dict(l=20, r=20, t=10, b=100),
        )
        st.plotly_chart(ml_only_fig, width="stretch")


if __name__ == "__main__":
    main()

