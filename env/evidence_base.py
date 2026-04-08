"""
Evidence-based clinical rules for ClinicalTriageEnv.
All rules are deterministic lookup tables — no ML, no randomness.

Rule structure:
  key   → human-readable rule name
  value → dict with:
    "trigger"      : callable(vitals, labs, history, symptoms) → bool
    "triage_level" : int (ESI 1-5, lower = more urgent; None = not level-specific)
    "disposition"  : str | None  (expected disposition if triggered)
    "safe_interventions"       : list[str]
    "contraindicated_interventions": list[str]
    "description"  : str
"""
from __future__ import annotations

from typing import Any, Callable

# ---------------------------------------------------------------------------
# Helper accessors
# ---------------------------------------------------------------------------

def _v(vitals: dict, key: str, default: float = 0.0) -> float:
    return float(vitals.get(key, default))

def _l(labs: dict | None, key: str, default: float = 0.0) -> float:
    if labs is None:
        return default
    return float(labs.get(key, default))

def _has(lst: list[str], *terms: str) -> bool:
    joined = " ".join(lst).lower()
    return any(t.lower() in joined for t in terms)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

CLINICAL_RULES: dict[str, dict[str, Any]] = {

    # ── SpO2 Rules ──────────────────────────────────────────────────────────
    "spo2_critical": {
        "trigger": lambda v, l, h, s: _v(v, "spo2") < 90,
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["supplemental_oxygen", "intubation_prep", "iv_access"],
        "contraindicated_interventions": [],
        "description": "SpO2 < 90% → immediate airway management, ESI-1",
    },
    "spo2_moderate_hypoxia": {
        "trigger": lambda v, l, h, s: 90 <= _v(v, "spo2") < 94,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["supplemental_oxygen", "nebulizer_treatment", "iv_access"],
        "contraindicated_interventions": [],
        "description": "SpO2 90-93% → supplemental O2, ESI-2",
    },
    "spo2_borderline": {
        "trigger": lambda v, l, h, s: 94 <= _v(v, "spo2") < 96,
        "triage_level": 3,
        "disposition": "admit_ward",
        "safe_interventions": ["supplemental_oxygen", "pulse_oximetry_monitoring"],
        "contraindicated_interventions": [],
        "description": "SpO2 94-95% → monitoring, ESI-3",
    },

    # ── Blood Pressure Rules ─────────────────────────────────────────────────
    "bp_hypertensive_emergency": {
        "trigger": lambda v, l, h, s: _v(v, "sbp") >= 180,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["iv_labetalol", "iv_nicardipine", "ecg_monitoring"],
        "contraindicated_interventions": ["sublingual_nifedipine"],
        "description": "SBP ≥ 180 → hypertensive emergency, avoid sublingual nifedipine (reflex tachycardia)",
    },
    "bp_septic_shock": {
        "trigger": lambda v, l, h, s: _v(v, "sbp") < 90,
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["iv_fluid_bolus", "vasopressors", "blood_cultures", "broad_spectrum_antibiotics"],
        "contraindicated_interventions": [],
        "description": "SBP < 90 → shock; aggressive resuscitation, ESI-1",
    },
    "bp_hypotension_mild": {
        "trigger": lambda v, l, h, s: 90 <= _v(v, "sbp") < 100,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["iv_fluid_bolus", "iv_access", "ecg_monitoring"],
        "contraindicated_interventions": [],
        "description": "SBP 90-99 → mild hypotension, ESI-2",
    },

    # ── Heart Rate Rules ────────────────────────────────────────────────────
    "hr_critical_tachycardia": {
        "trigger": lambda v, l, h, s: _v(v, "hr") > 150,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["ecg_monitoring", "iv_adenosine", "cardioversion_prep"],
        "contraindicated_interventions": [],
        "description": "HR > 150 → SVT/VT workup, ESI-2",
    },
    "hr_bradycardia_severe": {
        "trigger": lambda v, l, h, s: _v(v, "hr") < 40,
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["iv_atropine", "transcutaneous_pacing", "ecg_monitoring"],
        "contraindicated_interventions": [],
        "description": "HR < 40 → severe bradycardia, pacemaker prep, ESI-1",
    },
    "hr_bradycardia_moderate": {
        "trigger": lambda v, l, h, s: 40 <= _v(v, "hr") < 50,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["ecg_monitoring", "iv_atropine"],
        "contraindicated_interventions": ["beta_blockers"],
        "description": "HR 40-49 → moderate bradycardia, avoid beta-blockers",
    },
    "hr_tachycardia_moderate": {
        "trigger": lambda v, l, h, s: 100 < _v(v, "hr") <= 150,
        "triage_level": 3,
        "disposition": "admit_ward",
        "safe_interventions": ["iv_fluids", "ecg_monitoring", "fever_workup"],
        "contraindicated_interventions": [],
        "description": "HR 101-150 → moderate tachycardia, evaluate cause",
    },

    # ── GCS / Neurological Rules ────────────────────────────────────────────
    "gcs_unresponsive": {
        "trigger": lambda v, l, h, s: _v(v, "gcs") <= 8,
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["airway_management", "intubation_prep", "head_ct", "iv_access"],
        "contraindicated_interventions": ["oral_medications"],
        "description": "GCS ≤ 8 → loss of airway protection, immediate intubation, ESI-1",
    },
    "gcs_moderate": {
        "trigger": lambda v, l, h, s: 9 <= _v(v, "gcs") <= 12,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["head_ct", "neuro_monitoring", "iv_access"],
        "contraindicated_interventions": ["sedatives_without_airway_control"],
        "description": "GCS 9-12 → moderate altered consciousness, ESI-2",
    },

    # ── Respiratory Rate Rules ──────────────────────────────────────────────
    "rr_critical": {
        "trigger": lambda v, l, h, s: _v(v, "rr") > 30 or _v(v, "rr") < 8,
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["supplemental_oxygen", "bvm_ventilation", "intubation_prep"],
        "contraindicated_interventions": [],
        "description": "RR > 30 or < 8 → respiratory emergency, ESI-1",
    },
    "rr_elevated": {
        "trigger": lambda v, l, h, s: 24 < _v(v, "rr") <= 30,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["supplemental_oxygen", "nebulizer_treatment", "chest_xray"],
        "contraindicated_interventions": [],
        "description": "RR 25-30 → tachypnea workup, ESI-2",
    },

    # ── Temperature Rules ───────────────────────────────────────────────────
    "temp_hyperthermia_critical": {
        "trigger": lambda v, l, h, s: _v(v, "temp_c") >= 40.0,
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["cooling_measures", "iv_fluids", "blood_cultures", "broad_spectrum_antibiotics"],
        "contraindicated_interventions": [],
        "description": "Temp ≥ 40°C → heat stroke / sepsis emergency",
    },
    "temp_fever_high": {
        "trigger": lambda v, l, h, s: 38.5 <= _v(v, "temp_c") < 40.0,
        "triage_level": 3,
        "disposition": "admit_ward",
        "safe_interventions": ["acetaminophen", "blood_cultures", "iv_fluids"],
        "contraindicated_interventions": [],
        "description": "Temp 38.5-39.9°C → significant fever workup",
    },
    "temp_hypothermia": {
        "trigger": lambda v, l, h, s: _v(v, "temp_c") < 35.0,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["active_rewarming", "iv_fluids_warmed", "ecg_monitoring"],
        "contraindicated_interventions": [],
        "description": "Temp < 35°C → hypothermia, active rewarming",
    },

    # ── Lab-based Rules ─────────────────────────────────────────────────────
    "troponin_elevated": {
        "trigger": lambda v, l, h, s: _l(l, "troponin") >= 0.04,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["aspirin", "heparin", "serial_ecg", "cardiology_consult"],
        "contraindicated_interventions": ["nsaids"],
        "description": "Troponin ≥ 0.04 → ACS, NSAIDs contraindicated",
    },
    "lactate_critical": {
        "trigger": lambda v, l, h, s: _l(l, "lactate") >= 4.0,
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["iv_fluid_bolus", "vasopressors", "blood_cultures", "broad_spectrum_antibiotics"],
        "contraindicated_interventions": [],
        "description": "Lactate ≥ 4 → septic shock, ESI-1",
    },
    "lactate_elevated": {
        "trigger": lambda v, l, h, s: 2.0 <= _l(l, "lactate") < 4.0,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["iv_fluid_bolus", "sepsis_workup"],
        "contraindicated_interventions": [],
        "description": "Lactate 2-3.9 → hypoperfusion, sepsis protocol",
    },
    "creatinine_aki": {
        "trigger": lambda v, l, h, s: _l(l, "creatinine") >= 2.0,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["iv_fluids", "nephrology_consult", "hold_nephrotoxins"],
        "contraindicated_interventions": ["nsaids", "aminoglycosides", "iv_contrast_standard_dose"],
        "description": "Creatinine ≥ 2.0 → AKI; NSAIDs and nephrotoxins contraindicated",
    },
    "glucose_hypoglycemia": {
        "trigger": lambda v, l, h, s: _l(l, "glucose", 100) < 60,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["dextrose_iv", "glucagon_im"],
        "contraindicated_interventions": ["insulin"],
        "description": "Glucose < 60 → hypoglycemia; insulin contraindicated",
    },
    "glucose_hyperglycemia_critical": {
        "trigger": lambda v, l, h, s: _l(l, "glucose", 100) > 400,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["insulin_drip", "iv_fluids", "potassium_replacement"],
        "contraindicated_interventions": [],
        "description": "Glucose > 400 → DKA/HHS workup",
    },
    "inr_coagulopathy": {
        "trigger": lambda v, l, h, s: _l(l, "inr") > 3.0,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["vitamin_k", "ffp", "hematology_consult"],
        "contraindicated_interventions": ["aspirin", "nsaids", "heparin"],
        "description": "INR > 3.0 → coagulopathy; antiplatelets and anticoagulants contraindicated",
    },

    # ── Symptom / History Combination Rules ────────────────────────────────
    "chest_pain_acs": {
        "trigger": lambda v, l, h, s: _has(s, "chest pain") and _has(h, "hypertension", "diabetes", "smoking", "cad"),
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["ecg_12lead", "aspirin", "serial_troponin", "cardiology_consult"],
        "contraindicated_interventions": ["nsaids"],
        "description": "Chest pain + cardiac risk factors → ACS workup",
    },
    "stroke_signs": {
        "trigger": lambda v, l, h, s: _has(s, "facial droop", "arm weakness", "slurred speech", "sudden headache"),
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["head_ct", "neuro_consult", "stroke_protocol", "tpa_assessment"],
        "contraindicated_interventions": ["aspirin_before_ct"],
        "description": "FAST signs → stroke protocol, ESI-1, no aspirin before CT",
    },
    "anaphylaxis": {
        "trigger": lambda v, l, h, s: _has(s, "anaphylaxis", "angioedema", "urticaria", "wheezing") and (
            _v(v, "sbp") < 90 or _v(v, "hr") > 120
        ),
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["epinephrine_im", "iv_fluids", "diphenhydramine", "corticosteroids"],
        "contraindicated_interventions": ["beta_blockers"],
        "description": "Anaphylaxis → IM epinephrine immediately; beta-blockers contraindicated",
    },
    "renal_failure_nsaid": {
        "trigger": lambda v, l, h, s: _has(h, "renal_failure", "ckd", "aki") or _l(l, "creatinine") >= 1.5,
        "triage_level": None,
        "disposition": None,
        "safe_interventions": ["acetaminophen", "iv_fluids", "dose_adjusted_medications"],
        "contraindicated_interventions": ["nsaids", "aminoglycosides", "iv_contrast_standard_dose"],
        "description": "CKD/AKI → NSAIDs strictly contraindicated",
    },
    "respiratory_arrest_risk": {
        "trigger": lambda v, l, h, s: _has(s, "respiratory arrest", "apnea", "bluish lips"),
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["bvm_ventilation", "intubation_prep", "iv_access"],
        "contraindicated_interventions": ["sedatives_without_airway_control"],
        "description": "Respiratory arrest signs → immediate airway protection",
    },
    "sepsis_criteria": {
        "trigger": lambda v, l, h, s: (
            _v(v, "temp_c") > 38.3 or _v(v, "temp_c") < 36.0
        ) and _v(v, "hr") > 90 and _v(v, "rr") > 20,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["blood_cultures", "broad_spectrum_antibiotics", "iv_fluid_bolus", "lactate_level"],
        "contraindicated_interventions": [],
        "description": "SIRS criteria met → sepsis bundle",
    },
    "gib_active": {
        "trigger": lambda v, l, h, s: _has(s, "hematemesis", "melena", "hematochezia", "bloody stool") and _has(h, "nsaids", "warfarin", "aspirin"),
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["iv_ppi", "type_and_screen", "gi_consult"],
        "contraindicated_interventions": ["nsaids", "aspirin", "anticoagulants"],
        "description": "Active GIB on anticoagulants → hold anticoagulants, PPI, GI consult",
    },
    "pe_suspected": {
        "trigger": lambda v, l, h, s: _has(s, "pleuritic chest pain", "hemoptysis", "leg swelling") and _v(v, "hr") > 100,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["ctpa", "dvt_ultrasound", "heparin", "o2_supplementation"],
        "contraindicated_interventions": [],
        "description": "Suspected PE → CTPA, anticoagulation",
    },
    "head_trauma_severe": {
        "trigger": lambda v, l, h, s: _has(s, "loss of consciousness", "amnesia") and _v(v, "gcs") < 15,
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["head_ct", "spine_precautions", "neuro_monitoring"],
        "contraindicated_interventions": ["anticoagulants_without_imaging"],
        "description": "Head trauma with altered GCS → CT head first",
    },
    "pediatric_fever_infant": {
        "trigger": lambda v, l, h, s: _v(v, "temp_c") > 38.0 and _has(h, "infant", "neonate", "age<3months"),
        "triage_level": 2,
        "disposition": "admit_ward",
        "safe_interventions": ["lumbar_puncture", "blood_cultures", "iv_antibiotics", "urine_culture"],
        "contraindicated_interventions": ["aspirin"],
        "description": "Fever in infant <3 months → full sepsis workup; aspirin contraindicated (Reye's)",
    },
    "cardiac_arrest": {
        "trigger": lambda v, l, h, s: _v(v, "hr") < 20 and _v(v, "gcs") <= 3,
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["cpr", "defibrillation", "epinephrine_iv", "airway_management"],
        "contraindicated_interventions": [],
        "description": "Cardiac arrest → immediate CPR/ACLS",
    },
    "warfarin_bleeding_reversal": {
        "trigger": lambda v, l, h, s: _has(h, "warfarin") and (
            _l(l, "inr") > 5.0 or _has(s, "major bleeding", "intracranial bleed")
        ),
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["vitamin_k_iv", "4f_pcc", "hematology_consult"],
        "contraindicated_interventions": ["nsaids", "aspirin"],
        "description": "Warfarin + supratherapeutic INR or major bleed → 4F-PCC reversal",
    },
    "hyperkalemia_ecg_changes": {
        "trigger": lambda v, l, h, s: _l(l, "potassium") >= 6.5,
        "triage_level": 1,
        "disposition": "escalate_icu",
        "safe_interventions": ["calcium_gluconate", "insulin_dextrose", "kayexalate", "ecg_monitoring"],
        "contraindicated_interventions": ["potassium_supplementation"],
        "description": "Severe hyperkalemia → cardiac stabilization with calcium, then shift",
    },
}


def get_triggered_rules(
    vitals: dict,
    labs: dict | None,
    history: list[str],
    symptoms: list[str],
) -> list[str]:
    """Return list of rule names that fire for this patient state."""
    triggered = []
    for name, rule in CLINICAL_RULES.items():
        try:
            if rule["trigger"](vitals, labs, history, symptoms):
                triggered.append(name)
        except Exception:
            pass
    return triggered


def is_contraindicated(
    intervention: str,
    vitals: dict,
    labs: dict | None,
    history: list[str],
    symptoms: list[str],
) -> bool:
    """Return True if the intervention is contraindicated for this patient."""
    intervention_lower = intervention.lower().replace(" ", "_").replace("-", "_")
    for rule in CLINICAL_RULES.values():
        try:
            if rule["trigger"](vitals, labs, history, symptoms):
                if any(
                    ci.lower() in intervention_lower or intervention_lower in ci.lower()
                    for ci in rule["contraindicated_interventions"]
                ):
                    return True
        except Exception:
            pass
    return False


def is_safe_intervention(
    intervention: str,
    vitals: dict,
    labs: dict | None,
    history: list[str],
    symptoms: list[str],
) -> bool:
    """Return True if the intervention is explicitly recommended for this patient."""
    intervention_lower = intervention.lower().replace(" ", "_").replace("-", "_")
    for rule in CLINICAL_RULES.values():
        try:
            if rule["trigger"](vitals, labs, history, symptoms):
                if any(
                    si.lower() in intervention_lower or intervention_lower in si.lower()
                    for si in rule["safe_interventions"]
                ):
                    return True
        except Exception:
            pass
    return False


def expected_triage_level(
    vitals: dict,
    labs: dict | None,
    history: list[str],
    symptoms: list[str],
) -> int:
    """Compute expected ESI triage level (1=most urgent) from rule triggers."""
    triggered = get_triggered_rules(vitals, labs, history, symptoms)
    min_level = 5
    for name in triggered:
        lvl = CLINICAL_RULES[name].get("triage_level")
        if lvl is not None and lvl < min_level:
            min_level = lvl
    return min_level


def expected_disposition(
    vitals: dict,
    labs: dict | None,
    history: list[str],
    symptoms: list[str],
) -> str:
    """Return expected disposition based on highest-priority triggered rules."""
    triggered = get_triggered_rules(vitals, labs, history, symptoms)
    priority_order = ["escalate_icu", "admit_ward", "discharge"]
    for prio in priority_order:
        for name in triggered:
            if CLINICAL_RULES[name].get("disposition") == prio:
                return prio
    # Default safe discharge only if ESI 4-5
    lvl = expected_triage_level(vitals, labs, history, symptoms)
    return "discharge" if lvl >= 4 else "admit_ward"
