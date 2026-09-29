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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

import ui_kit as ui

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
    ui.mount(language)


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
    return ui.probability_figure(probabilities, highlight=highlight)


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
    ui.render(
        ui.kpi_grid(
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
        ui.render(ui.section_header("sliders", t(language, "inputs")))

        with st.expander(t(language, "guide"), expanded=False):
            st.write(t(language, "guide_text"))
            st.dataframe(GEOLOGICAL_LITHOLOGY_GUIDE, hide_index=True, width="stretch")

        ui.render(ui.group_label("layers", t(language, "reservoir_properties")))
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

        ui.render(ui.group_label("droplet", t(language, "fluid_properties")))
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
    ui.render(
        ui.hero_header(
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
    ui.render(ui.section_header("scales", t(language, "engine1_title")))
    st.caption(t(language, "screening_caption"))
    ui.render(ui.callout(t(language, "engine1_disclaimer"), "info", "shield"))
    for method_name, result in classical_results.items():
        ui.render(
            ui.status_card(
                method_name,
                result["status"],
                result["violations"],
                icon_name=METHOD_ICONS.get(method_name, "flask"),
                violations_label=t(language, "violations"),
            )
        )
        with st.expander(f"{t(language, 'method_details')}: {method_name}"):
            ui.render(ui.parameter_table(result["parameters"]))


def render_engine_two(language: str, model_result: Dict[str, Any]) -> None:
    ui.render(ui.section_header("cpu", t(language, "ml")))
    if model_result["mode"] == "trained":
        ui.render(
            ui.callout(
                "Real ensemble models trained from Screening_Original.xlsx are active.", "ok", "check"
            )
        )
    elif model_result.get("warning"):
        ui.render(ui.callout(t(language, "fallback_warning"), "warn", "alert"))

    for model_name, info in model_result["results"].items():
        ui.render(ui.model_card(model_name, info["prediction"], t(language, "predicted")))
        ui.render(ui.column_header(t(language, "eor_method"), t(language, "probability")))
        ui.show_figure(
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
    ui.render(ui.section_header("target", t(language, "consensus")))

    ui.render(
        ui.section_header(
            "cpu", t(language, "ml_only_result"), subtitle=t(language, "ml_engine_note"), sub=True
        )
    )
    ui.render(ui.column_header(t(language, "eor_method"), t(language, "ml_mean")))
    ui.render(ui.ranking_list([{"method": method, "value": score} for method, score in ml_only_rank]))

    ui.render(
        ui.section_header(
            "merge", t(language, "combined_result"), subtitle=t(language, "combined_note"), sub=True
        )
    )
    ui.render(ui.column_header(t(language, "eor_method"), t(language, "combined")))
    ui.render(
        ui.ranking_list(
            [{"method": method, "value": score} for method, score, _ in combined],
            show_status=True,
            statuses={method: status for method, _, status in combined},
        )
    )

    ui.render(
        ui.section_header(
            "chartArea", t(language, "combined"), subtitle=t(language, "screening_caption"), sub=True
        )
    )
    ui.show_figure(
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
    ui.render(ui.section_header("trend", t(language, "proxies")))
    render_kpi_cards(input_data)
    ui.render(ui.callout(t(language, "pressure_note"), "info", "shield"))
    st.caption(
        f"{t(language, 'normalized')}: {input_data['Depth_ft']:.1f} ft | "
        f"{input_data['Pressure_psi']:.1f} psi | {input_data['Temperature']:.1f} °F | "
        f"{input_data['Oil_Viscosity']:.3f} cP | {input_data['API_Gravity']:.2f} °API"
    )

    top_method, top_score, top_status = combined[0]
    ui.render(
        ui.verdict_card(
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


if __name__ == "__main__":
    main()

