"""
Algerian EOR Screening Platform
==============================
Dual-Engine Streamlit App for:
  - Classical reservoir screening rules
  - Physics-informed ML ensemble using V5-style engineered features

This version is intentionally robust to missing trained model artifacts.
If the model files are not present, the app does not crash and instead
falls back to a transparent, rule-based placeholder probability engine so the
UI remains usable while the user trains or saves the real models.
"""

from __future__ import annotations

import base64
import json
import math
import os
import warnings
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

st.set_page_config(
    page_title="Algerian EOR Screening Platform",
    page_icon="🛢️",
    layout="wide",
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
    "Lithology": "Carbonate",
    "Permeability": 120.0,
    "Porosity": 18.0,
    "Oil_Viscosity": 8.0,
    "Temperature": 82.22,
    "API_Gravity": 32.0,
    "Depth_m": 1828.8,
    "Pressure_bar": 172.37,
}

LANGUAGES = {"English": "en", "Français": "fr", "العربية": "ar"}

TRANSLATIONS = {
    "en": {
        "title": "Algerian EOR Decision-Support Suite",
        "subtitle": "Dual-engine technical screening: classical rules + physics-informed ML",
        "language": "Language", "inputs": "Reservoir Inputs",
        "preset": "Geological Preset / Lithology Guide", "custom": "Custom field data",
        "manual": "Manual lithology and field data", "lithology": "Lithology",
        "permeability": "Permeability", "porosity": "Porosity", "viscosity": "Oil Viscosity",
        "temperature": "Temperature", "gravity": "Oil Gravity", "depth": "Depth", "pressure": "Pressure",
        "guide": "Algerian geological guide",
        "guide_text": "Use this general formation-family guide when selecting the dominant lithology for an Algerian reservoir.",
        "proxies": "Physical Proxies", "classical": "Engine 1: Classical Screening",
        "ml": "Engine 2: ML Ensemble", "consensus": "Consensus Dashboard",
        "ml_only_result": "Engine B: ML-Only Result",
        "combined_result": "Combined Result: Engine A + Engine B",
        "ml_mean": "Mean ML Probability",
        "engine1_title": "Classical Heuristic Screening Windows (Preliminary Rules Engine)",
        "engine1_disclaimer": "These are initial heuristic screening windows. Detailed screening requires local thermodynamics, mobility-ratio analysis, and modern chemical-formulation testing.",
        "screening_caption": "PASS = ideal range, MARGINAL = near boundary, FAIL = outside the screening range.",
        "method_details": "Parameter details", "violations": "Violations", "predicted": "Predicted Method",
        "fallback": "Fallback mode is active because trained model artifacts were not found.",
        "fallback_warning": "No trained model files were found. Results are heuristic placeholders, not validated model predictions.",
        "probability": "Probability", "eor_method": "EOR Method", "combined": "Combined Technical Confidence",
        "pressure_note": "Depth and pressure are displayed and normalized, but current classical rules do not yet apply formation-pressure or MMP correlations.",
        "recommendation": "Screening Verdict", "confidence": "Combined technical confidence",
        "reservoir_properties": "Reservoir Properties", "advanced_inputs": "Depth, Pressure & MMP",
        "fluid_properties": "Fluid Properties", "optional": "optional", "normalized": "Normalized",
        "depth_unit": "Depth unit", "pressure_unit": "Pressure unit", "temperature_unit": "Temperature unit",
        "viscosity_unit": "Viscosity unit", "gravity_unit": "Gravity unit",
        "ml_engine_note": "These probabilities come from Engine B only; Engine A has not modified them.",
        "combined_note": "Engine A rule status is now applied to the standalone Engine B probabilities.",
        "screening_caption": "PASS = ideal range, MARGINAL = near boundary, FAIL = outside the screening range.",
        "method_count": "6 EOR methods screened", "dual_engine": "Dual engine",
        "engine1_tab": "Engine 1: Classical",
        "model_confidence": "Model probability distribution",
    },
    "fr": {
        "title": "Suite algérienne d'aide à la décision EOR",
        "subtitle": "Criblage technique à deux moteurs : règles classiques + ML informé par la physique",
        "language": "Langue", "inputs": "Données du réservoir", "preset": "Préréglage géologique / guide de lithologie",
        "custom": "Données personnalisées", "manual": "Lithologie et données saisies manuellement", "lithology": "Lithologie",
        "permeability": "Perméabilité", "porosity": "Porosité", "viscosity": "Viscosité de l'huile",
        "temperature": "Température", "gravity": "Gravité API", "depth": "Profondeur", "pressure": "Pression",
        "guide": "Guide géologique algérien",
        "guide_text": "Utilisez ce guide général des familles stratigraphiques pour sélectionner la lithologie dominante d'un réservoir algérien.",
        "proxies": "Indicateurs physiques", "classical": "Moteur 1 : criblage classique", "ml": "Moteur 2 : ensemble ML",
        "consensus": "Tableau de consensus", "screening_caption": "PASS = plage idéale, MARGINAL = proche de la limite, FAIL = hors plage.",
        "ml_only_result": "Moteur B : résultat ML seul", "combined_result": "Résultat combiné : moteur A + moteur B", "ml_mean": "Probabilité ML moyenne",
        "engine1_title": "Fenêtres heuristiques classiques (moteur de règles préliminaires)",
        "engine1_disclaimer": "Ces fenêtres sont heuristiques et préliminaires. Le criblage détaillé nécessite la thermodynamique locale, les rapports de mobilité et des essais de formulations chimiques modernes.",
        "method_details": "Détails des paramètres", "violations": "Dépassements", "predicted": "Méthode prédite",
        "fallback": "Le mode secours est actif car les modèles entraînés sont absents.",
        "fallback_warning": "Aucun modèle entraîné trouvé. Les résultats sont heuristiques et non validés.",
        "probability": "Probabilité", "eor_method": "Méthode EOR", "combined": "Confiance technique combinée",
        "pressure_note": "La profondeur et la pression sont normalisées, mais les règles actuelles n'appliquent pas encore les corrélations de pression de formation ou de MMP.",
        "recommendation": "Verdict de criblage", "confidence": "Confiance technique combinée",
        "reservoir_properties": "Propriétés du réservoir", "advanced_inputs": "Profondeur, pression et MMP",
        "fluid_properties": "Propriétés du fluide", "optional": "optionnel", "normalized": "Normalisé",
        "depth_unit": "Unité de profondeur", "pressure_unit": "Unité de pression",
        "temperature_unit": "Unité de température", "viscosity_unit": "Unité de viscosité",
        "gravity_unit": "Unité de gravité",
        "ml_engine_note": "Ces probabilités proviennent uniquement du moteur B ; le moteur A ne les a pas modifiées.",
        "combined_note": "Le statut des règles du moteur A est maintenant appliqué aux probabilités du moteur B seul.",
        "screening_caption": "PASS = plage idéale, MARGINAL = proche de la limite, FAIL = hors plage.",
        "method_count": "6 méthodes EOR évaluées", "dual_engine": "Double moteur",
        "engine1_tab": "Moteur 1 : Classique",
        "model_confidence": "Distribution de probabilité du modèle",
    },
    "ar": {
        "title": "منصة دعم قرار الاستخلاص المعزز للنفط في الجزائر",
        "subtitle": "فحص تقني بمحركين: قواعد كلاسيكية وتعلم آلي مدعوم بالفيزياء",
        "language": "اللغة", "inputs": "بيانات المكمن", "preset": "الإعداد الجيولوجي / دليل الصخور",
        "custom": "بيانات مخصصة", "manual": "بيانات الصخور والمكمن يدوياً", "lithology": "الليثولوجيا",
        "permeability": "النفاذية", "porosity": "المسامية", "viscosity": "لزوجة النفط",
        "temperature": "درجة الحرارة", "gravity": "كثافة API", "depth": "العمق", "pressure": "الضغط",
        "guide": "الدليل الجيولوجي الجزائري",
        "guide_text": "استخدم هذا الدليل العام للعائلات التكوينية لاختيار الليثولوجيا السائدة في المكمن الجزائري.",
        "proxies": "المؤشرات الفيزيائية", "classical": "المحرك 1: الفحص الكلاسيكي", "ml": "المحرك 2: ensemble للتعلم الآلي",
        "consensus": "لوحة التوافق", "screening_caption": "PASS = النطاق المثالي، MARGINAL = قريب من الحد، FAIL = خارج النطاق.",
        "ml_only_result": "المحرك B: نتيجة التعلم الآلي فقط", "combined_result": "النتيجة المجمعة: المحرك A + المحرك B", "ml_mean": "متوسط احتمال التعلم الآلي",
        "engine1_title": "نوافذ الفحص الكلاسيكية الإرشادية (محرك القواعد الأولي)",
        "engine1_disclaimer": "هذه حدود فحص إرشادية أولية. يتطلب الفحص التفصيلي الديناميكا الحرارية المحلية ونسب الحركة واختبارات التركيبات الكيميائية الحديثة.",
        "method_details": "تفاصيل المعايير", "violations": "المخالفات", "predicted": "الطريقة المتوقعة",
        "fallback": "الوضع الاحتياطي فعال لأن ملفات النماذج غير موجودة.",
        "fallback_warning": "لم يتم العثور على نموذج مدرب. النتائج تقريبية وليست تنبؤات نموذج معتمد.",
        "probability": "الاحتمال", "eor_method": "طريقة EOR", "combined": "الثقة التقنية المجمعة",
        "pressure_note": "تم توحيد العمق والضغط، لكن القواعد الحالية لا تطبق بعد علاقات ضغط المكمن أو MMP.",
        "recommendation": "حكم الفحص", "confidence": "الثقة التقنية المجمعة",
        "reservoir_properties": "خصائص المكمن", "advanced_inputs": "العمق والضغط و MMP",
        "fluid_properties": "خصائص السائل", "optional": "اختياري", "normalized": "القيم الموحّدة",
        "depth_unit": "وحدة العمق", "pressure_unit": "وحدة الضغط",
        "temperature_unit": "وحدة الحرارة", "viscosity_unit": "وحدة اللزوجة",
        "gravity_unit": "وحدة الكثافة",
        "ml_engine_note": "هذه الاحتمالات تأتي من المحرك B فقط؛ لم يعدّلها المحرك A.",
        "combined_note": "تم الآن تطبيق حالة قواعد المحرك A على احتمالات المحرك B وحده.",
        "screening_caption": "PASS = النطاق المثالي، MARGINAL = قريب من الحد، FAIL = خارج النطاق.",
        "method_count": "6 طرق EOR مقيّمة", "dual_engine": "محركان",
        "engine1_tab": "المحرك 1: الكلاسيكي",
        "model_confidence": "توزيع احتمال النموذج",
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


@st.cache_data
def load_flag_image_base64(filename: str = "algeria-flag.png") -> str | None:
    """Read the flag PNG from the repo root and return it as a base64 string.

    Returns None if the file isn't found, so the caller can fall back
    gracefully instead of breaking the header layout.
    """
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def apply_language_css(language: str) -> None:
    """Install the design system and the writing direction for the active language."""
    mount(language)


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

METHOD_ICONS: Dict[str, str] = {
    "Miscible Gas Injection": "drop",
    "Immiscible Gas Injection": "waves",
    "WAG Injection": "merge",
    "Chemical EOR": "flask",
    "Hybrid Gas-Chemical": "target",
    "Secondary Waterflooding": "droplet",
}


def plot_probability_chart(probabilities: Dict[str, float], highlight: Optional[str] = None) -> go.Figure:
    """Horizontal probability bars in the design-system palette."""
    return probability_figure(probabilities, highlight=highlight)


# ---------------------------------------------------------------------------
# 8. Main app layout
# ---------------------------------------------------------------------------

def physical_proxies(inputs: Dict[str, Any]) -> Dict[str, float]:
    """Compute the domain proxy metrics shown above the screening engines."""
    record = {
        "Permeability": safe_float(inputs.get("Permeability"), 120.0),
        "Porosity": normalize_percentage_to_fraction(safe_float(inputs.get("Porosity"), 18.0)),
        "Oil_Viscosity": safe_float(inputs.get("Oil_Viscosity"), 8.0),
    }
    mobility = record["Permeability"] / (record["Oil_Viscosity"] + 1e-6)
    rqi = 0.0314 * math.sqrt(record["Permeability"] / (record["Porosity"] + 1e-6))
    phi_z = record["Porosity"] / (1.0 - record["Porosity"] + 1e-6)
    fzi = rqi / (phi_z + 1e-6)
    return {"Mobility Proxy": mobility, "RQI": rqi, "FZI": fzi}


def render_kpi_cards(inputs: Dict[str, Any]) -> None:
    """Render the physical proxy metrics as responsive KPI cards."""
    proxies = physical_proxies(inputs)
    hints = {
        "Mobility Proxy": "k / µ",
        "RQI": "0.0314 · √(k/φ)",
        "FZI": "RQI / φz",
    }
    icons = {"Mobility Proxy": "waves", "RQI": "trend", "FZI": "grid"}
    render(
        kpi_grid(
            [
                {
                    "icon": icons[label],
                    "label": label,
                    "value": f"{value:.3f}",
                    "hint": hints[label],
                }
                for label, value in proxies.items()
            ]
        )
    )


def render_language_selector() -> str:
    with st.sidebar:
        selected = st.pills(
            "Language / Langue / اللغة",
            list(LANGUAGES),
            default=list(LANGUAGES)[0],
            selection_mode="single",
            label_visibility="collapsed",
            width="stretch",
        )
    return LANGUAGES[selected or list(LANGUAGES)[0]]


def render_input_panel(language: str) -> Dict[str, Any]:
    """Collect reservoir inputs in a mobile-first, progressively disclosed form."""
    with st.sidebar:
        render(section_header("sliders", t(language, "inputs")))

        with st.expander(t(language, "guide"), expanded=False):
            st.write(t(language, "guide_text"))
            st.dataframe(GEOLOGICAL_LITHOLOGY_GUIDE, hide_index=True, width="stretch")

        render(group_label("layers", t(language, "reservoir_properties")))
        lithology_options = [
            "Carbonate", "Carbonate / Dolomite", "Carbonate / Limestone",
            "Sandstone", "Sandstone / Quartzite", "Mixed Clastic", "Unknown",
        ]
        lithology = st.selectbox(t(language, "lithology"), lithology_options, index=0)
        permeability = st.number_input(
            f"{t(language, 'permeability')} (mD)", value=120.0, min_value=0.1, step=1.0
        )
        porosity_pct = st.number_input(
            f"{t(language, 'porosity')} (%)", value=18.0, min_value=0.1, max_value=60.0, step=0.1
        )

        render(group_label("droplet", t(language, "fluid_properties")))
        temperature_unit = st.selectbox(t(language, "temperature_unit"), ["°C", "°F"])
        temperature = st.number_input(
            f"{t(language, 'temperature')} ({temperature_unit})",
            value=82.22 if temperature_unit == "°C" else 180.0,
            min_value=-50.0,
            step=1.0,
        )
        viscosity_unit = st.selectbox(t(language, "viscosity_unit"), ["cP", "mPa·s"])
        viscosity = st.number_input(
            f"{t(language, 'viscosity')} ({viscosity_unit})",
            value=8.0, min_value=0.001, step=0.1,
        )
        gravity_unit = st.selectbox(t(language, "gravity_unit"), ["°API", "Specific Gravity (SG)"])
        gravity = st.number_input(
            f"{t(language, 'gravity')} ({gravity_unit})",
            value=32.0 if gravity_unit == "°API" else 0.865,
            min_value=0.01,
            step=0.1,
        )

        with st.expander(t(language, "advanced_inputs"), expanded=False):
            depth_unit = st.selectbox(t(language, "depth_unit"), ["m", "ft"])
            depth = st.number_input(
                f"{t(language, 'depth')} ({depth_unit})",
                value=1828.8 if depth_unit == "m" else 6000.0, min_value=1.0, step=100.0,
            )
            pressure_unit = st.selectbox(t(language, "pressure_unit"), ["bar", "psi", "MPa"])
            pressure = st.number_input(
                f"{t(language, 'pressure')} ({pressure_unit})",
                value=172.37 if pressure_unit == "bar" else (2500.0 if pressure_unit == "psi" else 17.24),
                min_value=0.1, step=10.0,
            )
            mmp = st.number_input(
                f"MMP ({pressure_unit}, {t(language, 'optional')})",
                value=0.0, min_value=0.0,
                step=5.0 if pressure_unit == "bar" else 50.0,
                help="Enter 0 when MMP is unavailable; depth will be used only as a proxy for miscible-gas screening.",
            )

    mmp_psi = mmp if pressure_unit == "psi" else mmp * {"bar": 14.5037738, "MPa": 145.037738}.get(pressure_unit, 1.0)
    field_units = convert_to_field_units(
        depth, depth_unit, pressure, pressure_unit, temperature, temperature_unit,
        viscosity, viscosity_unit, gravity, gravity_unit,
    )
    return {
        "Lithology": lithology,
        "Permeability": permeability,
        "Porosity": porosity_pct,
        "Oil_Viscosity": field_units["Oil_Viscosity"],
        "Temperature": field_units["Temperature"],
        "API_Gravity": field_units["API_Gravity"],
        "Depth_ft": field_units["Depth_ft"],
        "Pressure_psi": field_units["Pressure_psi"],
        "MMP_psi": mmp_psi,
    }


def render_header(language: str) -> None:
    render(
        hero_header(
            t(language, "title"),
            t(language, "subtitle"),
            load_flag_image_base64("algeria-flag.png"),
            [
                {"icon": "scales", "text": t(language, "classical")},
                {"icon": "cpu", "text": t(language, "ml")},
                {"icon": "filter", "text": t(language, "method_count")},
            ],
        )
    )


def render_engine_one(language: str, classical_results: Dict[str, Dict[str, Any]]) -> None:
    render(section_header("scales", t(language, "engine1_title")))
    st.caption(t(language, "screening_caption"))
    render(callout(t(language, "engine1_disclaimer"), "info", "shield"))
    for method_name, result in classical_results.items():
        render(
            status_card(
                method_name,
                result["status"],
                result["violations"],
                icon_name=METHOD_ICONS.get(method_name, "flask"),
                violations_label=t(language, "violations"),
            )
        )
        with st.expander(f"{t(language, 'method_details')}: {method_name}"):
            render(parameter_table(result["parameters"]))


def render_engine_two(language: str, model_result: Dict[str, Any]) -> None:
    render(section_header("cpu", t(language, "ml")))
    if model_result["mode"] == "trained":
        render(
            callout(
                "Real ensemble models trained from Screening_Original.xlsx are active.", "ok", "check"
            )
        )
    elif model_result.get("warning"):
        render(callout(t(language, "fallback_warning"), "warn", "alert"))

    for model_name, info in model_result["results"].items():
        render(model_card(model_name, info["prediction"], t(language, "predicted")))
        render(column_header(t(language, "eor_method"), t(language, "probability")))
        show_figure(
            plot_probability_chart(info["probabilities"], highlight=info["prediction"]),
            key=f"model-{model_name}",
        )


def consensus_scores(model_result: Dict[str, Any]) -> Dict[str, float]:
    """Mean probability of every method across the available ML models."""
    results = model_result["results"]
    return {
        method: float(
            np.mean([item["probabilities"].get(method, 0.0) for item in results.values()])
        )
        for method in EOR_CLASSES
    }


def combined_ranking(
    scores: Dict[str, float], classical_results: Dict[str, Dict[str, Any]]
) -> List[tuple]:
    """Engine A rule status weighted onto the standalone Engine B probabilities."""
    combined = []
    for method_name in EOR_CLASSES:
        status = classical_results[method_name]["status"]
        weight = {"PASS": 1.0, "MARGINAL": 0.7, "FAIL": 0.2}[status]
        combined.append((method_name, scores[method_name] * weight, status))
    combined.sort(key=lambda item: item[1], reverse=True)
    return combined


def render_consensus(
    language: str,
    combined: List[tuple],
    ml_only_rank: List[tuple],
) -> None:
    render(section_header("target", t(language, "consensus")))

    render(
        section_header(
            "cpu", t(language, "ml_only_result"), subtitle=t(language, "ml_engine_note"), sub=True
        )
    )
    render(column_header(t(language, "eor_method"), t(language, "ml_mean")))
    render(ranking_list([{"method": method, "value": score} for method, score in ml_only_rank]))

    render(
        section_header(
            "merge", t(language, "combined_result"), subtitle=t(language, "combined_note"), sub=True
        )
    )
    render(column_header(t(language, "eor_method"), t(language, "combined")))
    render(
        ranking_list(
            [{"method": method, "value": score} for method, score, _ in combined],
            show_status=True,
            statuses={method: status for method, _, status in combined},
        )
    )

    render(
        section_header(
            "chartArea", t(language, "combined"), subtitle=t(language, "screening_caption"), sub=True
        )
    )
    show_figure(
        plot_probability_chart(
            {method: score for method, score, _ in combined}, highlight=combined[0][0]
        ),
        key="combined-confidence",
    )


# ---------------------------------------------------------------------------
# 9. Streamlit application entry
# ---------------------------------------------------------------------------

def main() -> None:
    language = render_language_selector()
    apply_language_css(language)
    input_data = render_input_panel(language)

    classical_results = classical_screening(
        input_data["Lithology"],
        input_data["Permeability"],
        normalize_percentage_to_fraction(input_data["Porosity"]),
        input_data["Oil_Viscosity"],
        input_data["Temperature"],
        input_data["API_Gravity"],
        input_data["Depth_ft"],
        input_data["Pressure_psi"],
        input_data["MMP_psi"] if input_data["MMP_psi"] > 0 else None,
    )
    model_result = predict_ml_system(input_data)
    scores = consensus_scores(model_result)
    ml_only_rank = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    combined = combined_ranking(scores, classical_results)

    render_header(language)
    render(section_header("trend", t(language, "proxies")))
    render_kpi_cards(input_data)
    render(callout(t(language, "pressure_note"), "info", "shield"))
    st.caption(
        f"{t(language, 'normalized')}: {input_data['Depth_ft']:.1f} ft | "
        f"{input_data['Pressure_psi']:.1f} psi | {input_data['Temperature']:.1f} °F | "
        f"{input_data['Oil_Viscosity']:.3f} cP | {input_data['API_Gravity']:.2f} °API"
    )

    top_method, top_score, top_status = combined[0]
    render(
        verdict_card(
            top_method,
            top_score,
            top_status,
            t(language, "recommendation"),
            hint=t(language, "screening_caption"),
            icon_name="target",
        )
    )

    tab1, tab2, tab3 = st.tabs(
        [t(language, "engine1_tab"), t(language, "ml"), t(language, "consensus")]
    )
    with tab1:
        render_engine_one(language, classical_results)
    with tab2:
        render_engine_two(language, model_result)
    with tab3:
        render_consensus(language, combined, ml_only_rank)


# ---------------------------------------------------------------------------
# 10. Design system
# ---------------------------------------------------------------------------

FONT_STACK = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

TOKENS: Dict[str, str] = {
    "bg": "#F4F5F7",
    "surface": "#FFFFFF",
    "surface_2": "#FAFBFC",
    "ink": "#0E1116",
    "ink_2": "#39414D",
    "muted": "#697180",
    "faint": "#98A1AE",
    "line": "#E6E8EC",
    "line_strong": "#D4D9E0",
    "accent": "#12527E",
    "accent_soft": "#EAF1F7",
    "accent_line": "#C4D8E8",
    "track": "#EDEFF3",
    "pass": "#1B7F5A",
    "pass_soft": "#E7F4EF",
    "pass_line": "#BFE0D2",
    "marginal": "#A96B12",
    "marginal_soft": "#FBF0DC",
    "marginal_line": "#EBD5A9",
    "fail": "#B4453C",
    "fail_soft": "#FAEBE9",
    "fail_line": "#EFC7C3",
}

STATUS_TONE = {"PASS": "pass", "MARGINAL": "marginal", "FAIL": "fail"}
STATUS_ICON = {"PASS": "check", "MARGINAL": "alert", "FAIL": "close"}
TAB_ICONS = ["scales", "cpu", "target"]

ICONS: Dict[str, str] = {
    "layers": '<path d="M12 2.7 2.5 7.4 12 12.1l9.5-4.7z"/><path d="m2.5 12.2 9.5 4.7 9.5-4.7"/><path d="m2.5 16.9 9.5 4.7 9.5-4.7"/>',
    "droplet": '<path d="M12 3.2s6 6.1 6 10.1a6 6 0 0 1-12 0c0-4 6-10.1 6-10.1z"/>',
    "waves": '<path d="M2.5 6.5c.7.6 1.3 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.2 0 1.8.4 2.5 1"/><path d="M2.5 12c.7.6 1.3 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.2 0 1.8.4 2.5 1"/><path d="M2.5 17.5c.7.6 1.3 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.2 0 1.8.4 2.5 1"/>',
    "grid": '<rect x="3" y="3" width="7.5" height="7.5" rx="1.6"/><rect x="13.5" y="3" width="7.5" height="7.5" rx="1.6"/><rect x="3" y="13.5" width="7.5" height="7.5" rx="1.6"/><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.6"/>',
    "thermometer": '<path d="M14 14.8V4.5a2.5 2.5 0 1 0-5 0v10.3a4.5 4.5 0 1 0 5 0Z"/><path d="M11.5 8v6.6"/>',
    "flask": '<path d="M10 2.5v7.2L4.6 19.1A2 2 0 0 0 6.4 22h11.2a2 2 0 0 0 1.8-2.9L14 9.7V2.5"/><path d="M8.5 2.5h7"/><path d="M7.2 16.5h9.6"/>',
    "depth": '<path d="M12 3.2v13.6"/><path d="m6.5 11.5 5.5 5.5 5.5-5.5"/><path d="M4.5 21.2h15"/>',
    "gauge": '<path d="M3.4 19a10 10 0 1 1 17.2 0"/><path d="m12 14 4.2-4.2"/><circle cx="12" cy="14" r="1.1"/><circle cx="16.2" cy="9.8" r="1.1"/>',
    "scales": '<path d="m16 16 3-8 3 8a4.2 4.2 0 0 1-6 0Z"/><path d="m2 16 3-8 3 8a4.2 4.2 0 0 1-6 0Z"/><path d="M7 21.3h10"/><path d="M12 3.2v18.1"/><path d="M3 7.2h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2.4"/><rect x="9" y="9" width="6" height="6" rx="1.2"/><path d="M9.5 2v2M14.5 2v2M9.5 20v2M14.5 20v2M2 9.5h2M2 14.5h2M20 9.5h2M20 14.5h2"/>',
    "target": '<circle cx="12" cy="12" r="8.6"/><circle cx="12" cy="12" r="4.6"/><circle cx="12" cy="12" r="1"/>',
    "merge": '<circle cx="18" cy="18" r="2.6"/><circle cx="6" cy="6" r="2.6"/><path d="M6 21.2V9a9 9 0 0 0 9 9"/>',
    "check": '<circle cx="12" cy="12" r="8.6"/><path d="m8.4 12.2 2.4 2.4 4.8-5"/>',
    "alert": '<path d="M10.3 4 2.4 17.8A2 2 0 0 0 4.1 20.8h15.8a2 2 0 0 0 1.7-3L13.7 4a2 2 0 0 0-3.4 0Z"/><path d="M12 9.4v4.2"/><path d="M12 16.9h.01"/>',
    "close": '<circle cx="12" cy="12" r="8.6"/><path d="m15 9-6 6M9 9l6 6"/>',
    "info": '<circle cx="12" cy="12" r="8.6"/><path d="M12 11.2v5.2"/><path d="M12 7.9h.01"/>',
    "book": '<path d="M12 7.2v13.4"/><path d="M3.4 17.8a1 1 0 0 1-1-1V4.2a1 1 0 0 1 1-1h4.9a4 4 0 0 1 3.7 2.1 4 4 0 0 1 3.7-2.1h4.9a1 1 0 0 1 1 1v12.6a1 1 0 0 1-1 1h-5.4a3 3 0 0 0-2.6 1.2 3 3 0 0 0-2.6-1.2Z"/>',
    "globe": '<circle cx="12" cy="12" r="8.6"/><path d="M3.4 12h17.2"/><path d="M12 3.4a15 15 0 0 1 0 17.2 15 15 0 0 1 0-17.2Z"/>',
    "sliders": '<path d="M4 21v-6.5M4 10.5V3M12 21v-8.5M12 8.5V3M20 21v-4.5M20 12.5V3"/><path d="M1.5 14.5h5M9.5 8.5h5M17.5 16.5h5"/>',
    "percent": '<path d="M19 5 5 19"/><circle cx="7.5" cy="7.5" r="2.5"/><circle cx="16.5" cy="16.5" r="2.5"/>',
    "chart": '<path d="M18 20.5V11M12 20.5V3.5M6 20.5v-6"/>',
    "trend": '<path d="m22 7-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/>',
    "list": '<path d="M9 6.5h12M9 12h12M9 17.5h12"/><path d="M4.2 6.5h.01M4.2 12h.01M4.2 17.5h.01"/>',
    "file": '<path d="M14 3.2H7.4a2 2 0 0 0-2 2v13.6a2 2 0 0 0 2 2h9.2a2 2 0 0 0 2-2V8.2Z"/><path d="M14 3.2v5h4.6"/><path d="M8.8 13.4h6.4M8.8 16.6h4.4"/>',
    "table": '<rect x="3" y="4" width="18" height="16" rx="2.4"/><path d="M3 9.6h18M3 15h18M9.2 9.6V20"/>',
    "filter": '<path d="M3.5 5.2h17l-6.6 7.8v5.6l-3.8 2v-7.6Z"/>',
    "checkCircle": '<circle cx="12" cy="12" r="8.6"/><path d="m8.4 12.2 2.4 2.4 4.8-5"/>',
    "refresh": '<path d="M20.5 11.5a8.5 8.5 0 1 0-1.6 5.6"/><path d="M20.5 5.5v6h-6"/>',
    "sparkle": '<path d="M12 3.2 13.6 9 19.4 10.6 13.6 12.2 12 18 10.4 12.2 4.6 10.6 10.4 9Z"/><path d="M18.5 3.5v3M20 5h-3"/>',
    "shield": '<path d="M12 21.2s7-3 7-8.6V6.1L12 3.2 5 6.1v6.5c0 5.6 7 8.6 7 8.6Z"/><path d="M12 9.2v4"/><path d="M12 15.6h.01"/>',
    "chartArea": '<path d="M3.5 3.5v17h17"/><path d="m7 15.5 3.5-4 3 2.5 4.5-6.5"/>',
    "drop": '<path d="M12 3.4s5.6 5.8 5.6 9.5a5.6 5.6 0 0 1-11.2 0c0-3.7 5.6-9.5 5.6-9.5Z"/>',
}

PLOT_COLORS = {
    "accent": TOKENS["accent"],
    "neutral": "#C7CDD6",
    "track": TOKENS["line"],
    "muted": TOKENS["muted"],
    "ink": TOKENS["ink_2"],
    "font": FONT_STACK,
}


def icon(name: str, size: int = 18, stroke: float = 1.7, cls: str = "") -> str:
    """Return an outlined 24x24 SVG glyph string for the given icon name."""
    body = ICONS.get(name, ICONS["info"])
    class_attr = f' class="{cls}"' if cls else ""
    return (
        f'<svg{class_attr} viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


def icon_data_uri(name: str, stroke: float = 1.8) -> str:
    """Encode an icon as a data URI so it can be used as a CSS mask."""
    raw = icon(name, size=24, stroke=stroke).replace('"', "'")
    return "data:image/svg+xml," + quote(raw, safe="/:=!'()*-._ ~")


def _vars() -> str:
    return "".join(f"--eor-{key.replace('_', '-')}:{value};" for key, value in TOKENS.items())


BASE_CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{__VARS__--eor-font:__FONT__;--eor-radius:12px;--eor-radius-lg:16px;
--eor-shadow-1:0 1px 2px rgba(14,17,22,.04),0 10px 26px -18px rgba(14,17,22,.28);
--eor-accent-ring:rgba(18,82,126,.13)}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body,.stApp,[data-testid="stAppViewContainer"]{background:var(--eor-bg);color:var(--eor-ink);
font-family:var(--eor-font);font-size:15px;line-height:1.55;letter-spacing:-.005em;
-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
::selection{background:var(--eor-accent-soft);color:var(--eor-ink)}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:#D6DAE1;border-radius:9px}
::-webkit-scrollbar-track{background:transparent}

#MainMenu,footer,[data-testid="stStatusWidget"],[data-testid="stDecoration"],
[data-testid="stToolbar"]{visibility:hidden}
button[data-testid="stBaseButton-header"]:has(div[data-testid="stToolbarActionButtonIcon"]){display:none!important}

/* ---------- layout rhythm ---------- */
[data-testid="stMain"] .block-container,.block-container{padding:1rem 1rem 5.5rem;max-width:1140px}
[data-testid="stVerticalBlock"],[data-testid="stVerticalBlockBorderWrapper"]>div{gap:.7rem}
[data-testid="stHorizontalBlock"]{gap:.7rem}
h1,h2,h3,h4{margin:0;letter-spacing:-.02em}
.stMarkdown p{margin-block:0 .35rem}
.stCaption,.stMarkdown p[data-testid="stCaptionContainer"] p{color:var(--eor-muted);font-size:.8rem;line-height:1.5}
.stMarkdown p[data-testid="stCaptionContainer"]{margin-block:0}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{background:var(--eor-surface);border-inline-end:1px solid var(--eor-line)}
section[data-testid="stSidebar"] .block-container{padding-block:1.1rem 3rem}
section[data-testid="stSidebarCollapseButton"] button,
section[data-testid="stSidebarCollapsedControl"] button{border-radius:8px}

/* ---------- native widgets ---------- */
.stApp label,.stApp summary{color:var(--eor-ink-2);font-size:12.5px;font-weight:550;letter-spacing:.005em}
div[data-baseweb="input"],div[data-baseweb="select"]>div:first-child{
background:var(--eor-surface-2)!important;border:1px solid var(--eor-line)!important;
border-radius:10px!important;box-shadow:none!important;transition:border-color .15s,box-shadow .15s}
div[data-baseweb="input"]:focus-within,div[data-baseweb="select"]>div:focus-within{
border-color:var(--eor-accent)!important;box-shadow:0 0 0 3px var(--eor-accent-ring)!important}
div[data-baseweb="input"] input,div[data-baseweb="select"]>div:first-child{
background:transparent!important;border:0!important;box-shadow:none!important;color:var(--eor-ink);
font-family:var(--eor-font);font-size:15px;font-weight:550;padding-block:.5rem}
div[data-baseweb="input"] input:focus,div[data-baseweb="select"]>div:first-child:focus{outline:none}
div[data-baseweb="input"] input::placeholder{color:var(--eor-faint);font-weight:450}
div[data-testid="stSelectbox"] svg{color:var(--eor-faint)}
div[data-baseweb="popover"] li{font-size:14px;border-radius:8px;padding:.45rem .6rem}
div[data-baseweb="popover"] li:hover{background:var(--eor-accent-soft)}
div[data-testid^="stNumberInputStep"]{border:0!important;background:transparent!important;
color:var(--eor-faint)!important;border-radius:7px;width:26px;height:26px;margin-inline:2px}
div[data-testid^="stNumberInputStep"]:hover{color:var(--eor-accent)!important;background:var(--eor-accent-soft)}
div[data-testid="stWidgetLabel"] label{padding-block-end:.25rem}

div[data-testid="stExpander"]{border:1px solid var(--eor-line);border-radius:var(--eor-radius);
background:var(--eor-surface);box-shadow:none;overflow:hidden}
div[data-testid="stExpander"] summary{list-style:none;padding:.6rem .85rem}
div[data-testid="stExpander"] summary::-webkit-details-marker{display:none}
div[data-testid="stExpander"] summary:hover{background:var(--eor-surface-2)}
div[data-testid="stExpander"] summary svg{color:var(--eor-faint);transition:transform .2s ease}
div[data-testid="stExpander"][open] summary svg{transform:rotate(90deg)}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"]>div{padding-block:.2rem .4rem}
.stElementContainer:has(.eor-card)+.stElementContainer div[data-testid="stExpander"]{
margin-block-start:-7px;border-start-start-radius:0;border-start-end-radius:0;border-block-start-color:transparent}

div[data-testid="stTabs"]{gap:.9rem}
div[data-testid="stTabs"] [role="tablist"]{gap:3px;background:var(--eor-surface-2);
border:1px solid var(--eor-line);border-radius:13px;padding:4px;overflow-x:auto;scrollbar-width:none}
div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar{display:none}
button[data-baseweb="tab"]{display:inline-flex;align-items:center;gap:.45rem;height:auto;
padding:.5rem .8rem;border-radius:9px;background:transparent;color:var(--eor-muted);
font-family:var(--eor-font);font-size:13.5px;font-weight:550;white-space:nowrap;transition:.15s}
button[data-baseweb="tab"] p{font-size:13.5px!important;line-height:1.35!important;margin:0!important}
button[data-baseweb="tab"]:hover{color:var(--eor-ink-2);background:#F0F2F5}
button[data-baseweb="tab"][aria-selected="true"]{background:var(--eor-surface);color:var(--eor-ink);
box-shadow:var(--eor-shadow-1)}
button[data-baseweb="tab"]::before{content:"";display:none;width:15px;height:15px;flex:none;
background:currentColor;-webkit-mask-size:contain;mask-size:contain;
-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-position:center;mask-position:center}

.stButton>button,.stDownloadButton>button,.stLinkButton>button{
border-radius:10px;border:1px solid var(--eor-line);background:var(--eor-surface);
color:var(--eor-ink);font-family:var(--eor-font);font-size:13.5px;font-weight:600;box-shadow:none}
.stButton>button:hover{border-color:var(--eor-accent-line);color:var(--eor-accent);background:var(--eor-accent-soft)}
.stButton>button[kind="primary"]{background:var(--eor-accent);border-color:var(--eor-accent);color:#fff}
.stButton>button[kind="primary"]:hover{background:#0E4468;color:#fff}
.stButton>button:focus-visible,.stApp a:focus-visible,button:focus-visible,summary:focus-visible,
div[data-baseweb="input"] input:focus-visible{outline:2px solid var(--eor-accent);outline-offset:2px}

div[data-testid="stDataFrame"]{border:1px solid var(--eor-line);border-radius:var(--eor-radius);overflow:hidden}
div[data-testid="stDataFrame"] *{font-family:var(--eor-font)!important;font-size:13px!important}
div[data-testid="stDataFrameResizeHandle"]{display:none}
div[data-testid="stAlert"]{border-radius:var(--eor-radius);border:1px solid var(--eor-line)}

.js-plotly-plot .plotly .modebar,.plotly .modebar{display:none!important}
[data-testid="stPlotlyChart"]{border-radius:var(--eor-radius)}

/* ---------- composed components ---------- */
.eor-hero{position:relative;overflow:hidden;border:1px solid var(--eor-line);border-radius:var(--eor-radius-lg);
background:linear-gradient(180deg,var(--eor-surface) 0%,var(--eor-surface-2) 100%);
padding:1.1rem;box-shadow:var(--eor-shadow-1)}
.eor-hero-art{position:absolute;inset-block:0;inset-inline-end:0;width:min(56%,440px);height:100%;
color:var(--eor-accent);opacity:.11;pointer-events:none}
.eor-hero-inner{position:relative}
.eor-hero-top{display:flex;align-items:flex-start;gap:.8rem}
.eor-flag{width:36px;height:25px;flex:none;border-radius:4px;border:1px solid var(--eor-line-strong);
box-shadow:0 1px 2px rgba(14,17,22,.08);display:block;margin-block-start:.15rem}
.eor-hero h1{font-size:1.22rem;line-height:1.25;font-weight:650;letter-spacing:-.025em}
.eor-hero-sub{margin:.25rem 0 0;color:var(--eor-muted);font-size:.85rem;line-height:1.5;max-width:58ch}
.eor-chips{display:flex;flex-wrap:wrap;gap:.35rem;margin-block-start:.85rem}
.eor-chip{display:inline-flex;align-items:center;gap:.35rem;border:1px solid var(--eor-line);
background:var(--eor-surface);color:var(--eor-ink-2);border-radius:999px;padding:.22rem .55rem;
font-size:.715rem;font-weight:550;letter-spacing:.01em;white-space:nowrap}
.eor-chip svg{color:var(--eor-accent);flex:none}

.eor-section{display:flex;align-items:center;gap:.65rem;margin:1.35rem 0 .6rem}
.eor-section-icon{width:32px;height:32px;flex:none;border-radius:9px;display:grid;place-items:center;
background:var(--eor-accent-soft);color:var(--eor-accent);border:1px solid var(--eor-accent-line)}
.eor-section-title{font-size:1rem;font-weight:650;letter-spacing:-.015em;line-height:1.3}
.eor-section-sub{margin:.05rem 0 0;font-size:.78rem;color:var(--eor-muted);line-height:1.45}
.eor-rule{flex:1;height:1px;background:var(--eor-line);min-width:12px}
.eor-section--sub{margin:1.15rem 0 .5rem}
.eor-section--sub .eor-section-icon{width:28px;height:28px;border-radius:8px}
.eor-section--sub .eor-section-icon svg{width:15px;height:15px}
.eor-section--sub .eor-section-title{font-size:.92rem}
.eor-section--sub .eor-rule{display:none}
.eor-group{display:flex;align-items:center;gap:.4rem;margin:.55rem 0 .1rem;color:var(--eor-muted);
font-size:.675rem;font-weight:650;text-transform:uppercase;letter-spacing:.09em}
.eor-group svg{color:var(--eor-faint);flex:none}

.eor-kpis{display:grid;gap:.55rem;grid-template-columns:1fr}
.eor-kpi{display:flex;align-items:center;gap:.75rem;border:1px solid var(--eor-line);border-radius:var(--eor-radius);
background:var(--eor-surface);padding:.7rem .85rem;box-shadow:var(--eor-shadow-1)}
.eor-kpi-icon{width:34px;height:34px;flex:none;border-radius:10px;display:grid;place-items:center;
background:var(--eor-accent-soft);color:var(--eor-accent)}
.eor-kpi-body{min-width:0;flex:1}
.eor-kpi-label{font-size:.665rem;text-transform:uppercase;letter-spacing:.09em;color:var(--eor-muted);font-weight:650}
.eor-kpi-value{font-size:1.12rem;font-weight:650;letter-spacing:-.02em;line-height:1.3;
font-variant-numeric:tabular-nums}
.eor-kpi-hint{font-size:.7rem;color:var(--eor-faint);line-height:1.4}

.eor-card{border:1px solid var(--eor-line);border-inline-start:3px solid var(--eor-line-strong);
border-radius:var(--eor-radius);background:var(--eor-surface);padding:.8rem .9rem;box-shadow:var(--eor-shadow-1)}
.eor-card--pass{border-inline-start-color:var(--eor-pass)}
.eor-card--marginal{border-inline-start-color:var(--eor-marginal)}
.eor-card--fail{border-inline-start-color:var(--eor-fail)}
.eor-card--neutral{border-inline-start-color:var(--eor-accent)}
.eor-card-head{display:flex;align-items:center;gap:.55rem;flex-wrap:wrap}
.eor-card-icon{color:var(--eor-faint);flex:none;display:grid;place-items:center}
.eor-card-head h3{font-size:.93rem;font-weight:600;line-height:1.35;flex:1 1 auto;min-width:0}
.eor-badge{display:inline-flex;align-items:center;gap:.28rem;border-radius:999px;padding:.16rem .5rem;
font-size:.655rem;font-weight:700;letter-spacing:.055em;text-transform:uppercase;border:1px solid;white-space:nowrap}
.eor-badge--pass{color:var(--eor-pass);background:var(--eor-pass-soft);border-color:var(--eor-pass-line)}
.eor-badge--marginal{color:var(--eor-marginal);background:var(--eor-marginal-soft);border-color:var(--eor-marginal-line)}
.eor-badge--fail{color:var(--eor-fail);background:var(--eor-fail-soft);border-color:var(--eor-fail-line)}
.eor-badge--neutral{color:var(--eor-ink-2);background:var(--eor-surface-2);border-color:var(--eor-line-strong)}
.eor-notes{margin:.55rem 0 0;padding:0;list-style:none;display:grid;gap:.3rem}
.eor-notes li{display:flex;align-items:flex-start;gap:.4rem;font-size:.795rem;color:var(--eor-ink-2);line-height:1.45}
.eor-notes li svg{flex:none;margin-block-start:.15rem}
.eor-note--warn{color:var(--eor-marginal)}
.eor-note--warn svg{color:var(--eor-marginal)}
.eor-note--muted{color:var(--eor-muted)}
.eor-note--muted svg{color:var(--eor-faint)}

.eor-verdict{display:flex;align-items:center;gap:.85rem;border:1px solid var(--eor-line);
border-radius:var(--eor-radius);background:linear-gradient(180deg,var(--eor-surface) 0%,var(--eor-surface-2) 100%);
padding:.9rem 1rem;box-shadow:var(--eor-shadow-1)}
.eor-ring{width:54px;height:54px;flex:none}
.eor-ring circle{fill:none;stroke-width:3.4;stroke-linecap:round}
.eor-ring .ring-track{stroke:var(--eor-track)}
.eor-ring .ring-val{stroke:var(--eor-accent);transition:stroke-dashoffset .5s ease}
.eor-ring .ring-text{fill:var(--eor-ink);font-size:13px;font-weight:700;text-anchor:middle;
font-family:var(--eor-font)}
.eor-verdict-body{min-width:0;flex:1}
.eor-verdict-label{font-size:.665rem;text-transform:uppercase;letter-spacing:.09em;color:var(--eor-muted);font-weight:650}
.eor-verdict-value{font-size:1.12rem;font-weight:650;letter-spacing:-.02em;line-height:1.3}
.eor-verdict-meta{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem;margin-block-start:.3rem}
.eor-verdict-hint{margin:.3rem 0 0;font-size:.775rem;color:var(--eor-muted);line-height:1.5}

.eor-list{list-style:none;margin:.1rem 0 0;padding:0;border:1px solid var(--eor-line);
border-radius:var(--eor-radius);background:var(--eor-surface);overflow:hidden;box-shadow:var(--eor-shadow-1)}
.eor-list-head{display:flex;justify-content:space-between;align-items:center;gap:.5rem;
padding:0 .15rem .4rem;margin-block-end:.4rem;border-block-end:1px solid var(--eor-line);
font-size:.665rem;text-transform:uppercase;letter-spacing:.085em;font-weight:650;color:var(--eor-muted)}
.eor-list>li+li{border-block-start:1px solid var(--eor-line)}
.eor-list>li{padding:.7rem .85rem}
.eor-list>li.eor-lead{background:var(--eor-surface-2)}
.eor-row-head{display:flex;align-items:center;gap:.55rem}
.eor-rank{display:inline-grid;place-items:center;min-width:20px;height:20px;padding-inline:.3rem;border-radius:6px;
background:var(--eor-track);color:var(--eor-muted);font-size:.68rem;font-weight:700;flex:none;
font-variant-numeric:tabular-nums}
.eor-lead .eor-rank{background:var(--eor-accent);color:#fff}
.eor-name{flex:1 1 auto;min-width:0;font-size:.875rem;font-weight:600;line-height:1.4;letter-spacing:-.01em}
.eor-score{font-size:.875rem;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.01em;flex:none}
.eor-track{height:6px;border-radius:999px;background:var(--eor-track);overflow:hidden;margin-block-start:.5rem}
.eor-fill{height:100%;border-radius:999px;background:var(--eor-accent)}
.eor-fill--pass{background:var(--eor-pass)}
.eor-fill--marginal{background:var(--eor-marginal)}
.eor-fill--fail{background:var(--eor-fail)}
.eor-fill--neutral{background:var(--eor-neutral,#C7CDD6)}
.eor-row-meta{display:flex;align-items:center;gap:.4rem;margin-block-start:.45rem}

.eor-params{list-style:none;margin:0;padding:0;border:1px solid var(--eor-line);
border-radius:var(--eor-radius);overflow:hidden;background:var(--eor-surface)}
.eor-params>li{display:flex;align-items:center;justify-content:space-between;gap:.75rem;
padding:.55rem .8rem;font-size:.83rem}
.eor-params>li+li{border-block-start:1px solid var(--eor-line)}
.eor-params>li:nth-child(odd){background:var(--eor-surface-2)}
.eor-param-name{color:var(--eor-ink-2);min-width:0}
.eor-param-note{color:var(--eor-muted);font-size:.76rem;text-align:end;font-variant-numeric:tabular-nums}
.eor-dot{width:8px;height:8px;border-radius:50%;flex:none;display:inline-block}
.eor-dot--pass{background:var(--eor-pass)}
.eor-dot--marginal{background:var(--eor-marginal)}
.eor-dot--fail{background:var(--eor-fail)}

.eor-callout{display:flex;gap:.6rem;border:1px solid var(--eor-line);border-radius:var(--eor-radius);
background:var(--eor-surface-2);padding:.75rem .85rem;font-size:.83rem;line-height:1.55;color:var(--eor-ink-2)}
.eor-callout svg{flex:none;margin-block-start:.1rem}
.eor-callout--info{background:var(--eor-accent-soft);border-color:var(--eor-accent-line);color:#12384F}
.eor-callout--info svg{color:var(--eor-accent)}
.eor-callout--warn{background:var(--eor-marginal-soft);border-color:var(--eor-marginal-line);color:#6E4708}
.eor-callout--warn svg{color:var(--eor-marginal)}
.eor-callout--ok{background:var(--eor-pass-soft);border-color:var(--eor-pass-line);color:#125741}
.eor-callout--ok svg{color:var(--eor-pass)}

@media (min-width:560px){
[data-testid="stMain"] .block-container,.block-container{padding:1.35rem 1.5rem 6rem}
.eor-hero{padding:1.4rem 1.5rem}
.eor-flag{width:44px;height:30px}
.eor-hero h1{font-size:1.5rem}
.eor-hero-sub{font-size:.9rem}
.eor-ring{width:60px;height:60px}
.eor-ring .ring-text{font-size:14px}
}
@media (min-width:720px){
.eor-kpis{grid-template-columns:repeat(3,1fr)}
.eor-kpi{padding:.8rem 1rem}
}
@media (min-width:1200px){
[data-testid="stMain"] .block-container,.block-container{padding-block:1.6rem}
}
@media (prefers-reduced-motion:reduce){
*{animation-duration:.001ms!important;transition-duration:.001ms!important}
}
"""

COMPONENT_CSS = """
@media (max-width:639px){
button[data-baseweb="tab"]{padding:.5rem .65rem;font-size:13px}
.eor-verdict{align-items:flex-start}
}
"""


def _build_css() -> str:
    css = BASE_CSS.replace("__VARS__", _vars()).replace("__FONT__", FONT_STACK)
    for index, name in enumerate(TAB_ICONS, start=1):
        uri = icon_data_uri(name)
        css += (
            f'\ndiv[data-testid="stTabs"] [role="tablist"] > button:nth-of-type({index})::before'
            f"{{display:block;-webkit-mask-image:url('{uri}');mask-image:url('{uri}')}}"
        )
    return css + COMPONENT_CSS


def mount(language: str) -> None:
    """Install the design system stylesheet and the writing direction."""
    direction = "rtl" if language == "ar" else "ltr"
    st.html(
        f'<style>html,body,[data-testid="stAppViewContainer"]{{direction:{direction};}}'
        f"{_build_css()}</style>"
    )


# ---------------------------------------------------------------------------
# HTML components
# ---------------------------------------------------------------------------

def hero_header(
    title: str,
    subtitle: str,
    flag_base64: Optional[str],
    chips: Sequence[Dict[str, str]],
) -> str:
    flag = (
        f'<img class="eor-flag" src="data:image/png;base64,{flag_base64}" alt="Algeria flag" />'
        if flag_base64
        else f'<span class="eor-chip">{icon("globe", 15)} 🇩🇿</span>'
    )
    chip_html = "".join(
        f'<span class="eor-chip">{icon(chip.get("icon", "sparkle"), 14)}'
        f'{escape(chip.get("text", ""))}</span>'
        for chip in chips
    )
    return f"""<header class="eor-hero">
  <svg class="eor-hero-art" viewBox="0 0 440 190" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
    <g fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round">
      <path d="M-30 128C40 92 96 168 168 132s128-64 200-26 96 46 120 24"/>
      <path d="M-30 150C40 114 96 190 168 154s128-64 200-26 96 46 120 24"/>
      <path d="M-30 106C40 70 96 146 168 110s128-64 200-26 96 46 120 24"/>
      <path d="M-30 84C40 48 96 124 168 88s128-64 200-26 96 46 120 24" stroke-dasharray="3 6"/>
      <path d="M-30 62C40 26 96 102 168 66s128-64 200-26 96 46 120 24" stroke-dasharray="3 6"/>
      <path d="M96 -10v210M232 -10v210M344 -10v210" stroke-dasharray="2 7" opacity=".7"/>
    </g>
    <g fill="none" stroke="currentColor" stroke-width="1.1">
      <circle cx="96" cy="132" r="4"/><circle cx="232" cy="110" r="4"/><circle cx="344" cy="152" r="4"/>
    </g>
  </svg>
  <div class="eor-hero-inner">
    <div class="eor-hero-top">{flag}
      <div><h1>{escape(title)}</h1><p class="eor-hero-sub">{escape(subtitle)}</p></div>
    </div>
    <div class="eor-chips">{chip_html}</div>
  </div>
</header>"""


def section_header(icon_name: str, title: str, subtitle: str = "", sub: bool = False) -> str:
    sub = f'<p class="eor-section-sub">{escape(subtitle)}</p>' if subtitle else ""
    modifier = " eor-section--sub" if sub else ""
    return f"""<div class="eor-section{modifier}">
  <span class="eor-section-icon">{icon(icon_name, 17)}</span>
  <div><div class="eor-section-title">{escape(title)}</div>{sub}</div>
  <span class="eor-rule"></span>
</div>"""


def group_label(icon_name: str, text: str) -> str:
    """A compact uppercase label used to group form fields."""
    return f'<div class="eor-group">{icon(icon_name, 14)}<span>{escape(text)}</span></div>'


def kpi_grid(items: Sequence[Dict[str, str]]) -> str:
    cards = []
    for item in items:
        hint = f'<div class="eor-kpi-hint">{escape(item["hint"])}</div>' if item.get("hint") else ""
        cards.append(
            f"""<div class="eor-kpi">
  <span class="eor-kpi-icon">{icon(item.get("icon", "chart"), 18)}</span>
  <div class="eor-kpi-body">
    <div class="eor-kpi-label">{escape(item["label"])}</div>
    <div class="eor-kpi-value">{escape(item["value"])}</div>
    {hint}
  </div>
</div>"""
        )
    return f'<div class="eor-kpis">{"".join(cards)}</div>'


def _ring(pct: float, tone: str = "accent") -> str:
    radius = 19.0
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1.0 - max(0.0, min(1.0, pct)))
    stroke = f"var(--eor-{tone})" if tone != "accent" else "var(--eor-accent)"
    return f"""<svg class="eor-ring" viewBox="0 0 44 44" aria-hidden="true">
  <circle class="ring-track" cx="22" cy="22" r="{radius}"/>
  <circle class="ring-val" cx="22" cy="22" r="{radius}" stroke="{stroke}"
    stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}"
    transform="rotate(-90 22 22)"/>
  <text class="ring-text" x="22" y="26">{int(round(pct * 100))}</text>
</svg>"""


def verdict_card(
    method: str,
    confidence: float,
    status: str,
    label: str,
    hint: str = "",
    icon_name: str = "target",
) -> str:
    tone = STATUS_TONE.get(status, "neutral")
    meta = f'<span class="eor-badge eor-badge--{tone}">{icon(STATUS_ICON.get(status, "info"), 12)}{escape(status)}</span>'
    hint_html = f'<p class="eor-verdict-hint">{escape(hint)}</p>' if hint else ""
    return f"""<div class="eor-verdict">
  {_ring(confidence, tone)}
  <div class="eor-verdict-body">
    <div class="eor-verdict-label">{escape(label)}</div>
    <div class="eor-verdict-value">{escape(method)}</div>
    <div class="eor-verdict-meta">{icon(icon_name, 14, cls="eor-card-icon")}{meta}</div>
    {hint_html}
  </div>
</div>"""


def status_card(
    method: str,
    status: str,
    violations: Sequence[str] = (),
    icon_name: str = "flask",
    violations_label: str = "",
) -> str:
    """A method card carrying its screening status badge and any violations."""
    tone = STATUS_TONE.get(status, "neutral")
    notes = ""
    if violations:
        label = f"{violations_label}: " if violations_label else ""
        notes = (
            '<ul class="eor-notes">'
            f'<li class="eor-note--warn">{icon("alert", 14)}'
            f"<span>{escape(label + ', '.join(violations))}</span></li></ul>"
        )
    return f"""<div class="eor-card eor-card--{tone}">
  <div class="eor-card-head">
    <span class="eor-card-icon">{icon(icon_name, 17)}</span>
    <h3>{escape(method)}</h3>
    <span class="eor-badge eor-badge--{tone}">{icon(STATUS_ICON.get(status, "info"), 12)}{escape(status)}</span>
  </div>
  {notes}
</div>"""


def model_card(model_name: str, prediction: str, predicted_label: str, icon_name: str = "cpu") -> str:
    return f"""<div class="eor-card eor-card--neutral">
  <div class="eor-card-head">
    <span class="eor-card-icon">{icon(icon_name, 17)}</span>
    <h3>{escape(model_name)}</h3>
  </div>
  <div class="eor-verdict-meta" style="margin-block:.55rem 0">
    <span class="eor-badge eor-badge--neutral">{escape(predicted_label)}</span>
    <span style="font-size:.93rem;font-weight:600;letter-spacing:-.01em">{escape(prediction)}</span>
  </div>
</div>"""


def column_header(left: str, right: str) -> str:
    """A hairline micro-header that labels the two columns of a ranking list."""
    return (
        f'<div class="eor-list-head"><span>{escape(left)}</span>'
        f"<span>{escape(right)}</span></div>"
    )


def ranking_list(
    rows: Sequence[Dict[str, Any]],
    show_status: bool = False,
    statuses: Optional[Dict[str, str]] = None,
) -> str:
    statuses = statuses or {}
    top = max((row["value"] for row in rows), default=0.0) or 1.0
    items = []
    for index, row in enumerate(rows, start=1):
        method = row["method"]
        value = float(row["value"])
        ratio = max(2.0, value / top * 100.0)
        status = statuses.get(method)
        tone = STATUS_TONE.get(status or "", "neutral")
        fill = f"eor-fill--{tone}" if show_status else "eor-fill"
        badge = (
            f'<span class="eor-badge eor-badge--{tone}">'
            f"{icon(STATUS_ICON.get(status or '', 'info'), 12)}{escape(status or '')}</span>"
            if show_status
            else ""
        )
        lead = " eor-lead" if index == 1 else ""
        items.append(
            f"""<li class="{lead.strip()}">
  <div class="eor-row-head">
    <span class="eor-rank">{index}</span>
    <span class="eor-name">{escape(method)}</span>
    {badge}
    <span class="eor-score">{value * 100:.1f}%</span>
  </div>
  <div class="eor-track"><span class="eor-fill {fill}" style="width:{ratio:.1f}%"></span></div>
</li>"""
        )
    return f'<ul class="eor-list">{"".join(items)}</ul>'


def parameter_table(parameters: Dict[str, str]) -> str:
    rows = []
    for name, value in parameters.items():
        tone = STATUS_TONE.get(value)
        if tone:
            cell = (
                f'<span class="eor-badge eor-badge--{tone}">'
                f"{icon(STATUS_ICON.get(value, 'info'), 11)}{escape(value)}</span>"
            )
        else:
            cell = f'<span class="eor-param-note">{escape(value)}</span>'
        rows.append(
            f'<li><span class="eor-param-name">{escape(name)}</span>{cell}</li>'
        )
    return f'<ul class="eor-params">{"".join(rows)}</ul>'


def callout(text: str, tone: str = "info", icon_name: Optional[str] = None) -> str:
    default_icon = {"info": "info", "warn": "alert", "ok": "check"}[tone]
    return (
        f'<div class="eor-callout eor-callout--{tone}">'
        f'{icon(icon_name or default_icon, 17)}<span>{escape(text)}</span></div>'
    )


def note_list(items: Sequence[str], tone: str = "muted", icon_name: str = "info") -> str:
    entries = "".join(
        f'<li class="eor-note--{tone}">{icon(icon_name, 14)}<span>{escape(item)}</span></li>'
        for item in items
    )
    return f'<ul class="eor-notes">{entries}</ul>'


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def probability_figure(
    probabilities: Dict[str, float],
    highlight: Optional[str] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """Horizontal, mobile-friendly probability bars in the design-system palette."""
    items = sorted(probabilities.items(), key=lambda item: item[1])
    labels = [item[0] for item in items]
    values = [float(item[1]) for item in items]
    colors = [
        PLOT_COLORS["accent"] if highlight and label == highlight else PLOT_COLORS["neutral"]
        for label in labels
    ]
    upper = max(values) if values else 1.0
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{value * 100:.1f}%" for value in values],
            textposition="outside",
            textfont=dict(size=12, color=PLOT_COLORS["muted"], family=PLOT_COLORS["font"]),
            hovertemplate="%{y} · %{x:.1%}<extra></extra>",
            cliponaxis=False,
            width=0.55,
        )
    )
    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOT_COLORS["font"], color=PLOT_COLORS["ink"], size=12.5),
        showlegend=False,
        bargap=0.55,
        barcornerradius=6,
        height=height or (58 + 40 * max(len(labels), 1)),
        margin=dict(l=4, r=52, t=6, b=6),
        xaxis=dict(
            range=[0, max(upper * 1.22, 0.12)],
            tickformat=".0%",
            tickfont=dict(size=11, color=PLOT_COLORS["muted"]),
            gridcolor=PLOT_COLORS["track"],
            griddash="solid",
            gridwidth=1,
            zeroline=False,
            showline=False,
            ticks="",
            fixedrange=True,
        ),
        yaxis=dict(
            tickfont=dict(size=12.5, color=PLOT_COLORS["ink"]),
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
            automargin=True,
            fixedrange=True,
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=TOKENS["line"],
            font=dict(size=12, color=PLOT_COLORS["ink"], family=PLOT_COLORS["font"]),
        ),
    )
    return fig


def show_figure(fig: go.Figure, key: Optional[str] = None) -> None:
    st.plotly_chart(
        fig,
        key=key,
        width="stretch",
        config={"displayModeBar": False, "displaylogo": False, "responsive": True},
    )


def render(html: str) -> None:
    st.html(html)


if __name__ == "__main__":
    main()

