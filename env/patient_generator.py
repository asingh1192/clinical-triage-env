"""
Seeded synthetic patient generator for ClinicalTriageEnv.
All data is SYNTHETIC — no real PHI.

Difficulty tiers:
  easy   → single textbook abnormal vital
  medium → 2-3 interacting conditions
  hard   → complex multi-system with contradictory signals
"""
from __future__ import annotations

import random
from typing import Optional

from .models import PatientObservation


# ---------------------------------------------------------------------------
# Patient template library
# ---------------------------------------------------------------------------

_EASY_TEMPLATES = [
    # textbook single-vital abnormality
    {
        "chief_complaint": "Mild chest discomfort",
        "base_vitals": {"hr": 85, "sbp": 140, "dbp": 90, "spo2": 98, "rr": 16, "temp_c": 37.0, "gcs": 15},
        "delta": {"sbp": 40},          # SBP → 180 hypertensive
        "symptoms": ["chest discomfort"],
        "history": ["hypertension"],
        "labs": None,
        "expected_esi": 3,
        "expected_disposition": "admit_ward",
    },
    {
        "chief_complaint": "Shortness of breath at rest",
        "base_vitals": {"hr": 100, "sbp": 120, "dbp": 78, "spo2": 88, "rr": 22, "temp_c": 37.1, "gcs": 15},
        "delta": {},
        "symptoms": ["shortness of breath", "dyspnea"],
        "history": ["asthma"],
        "labs": None,
        "expected_esi": 1,
        "expected_disposition": "escalate_icu",
    },
    {
        "chief_complaint": "Slow heart rate found on routine check",
        "base_vitals": {"hr": 35, "sbp": 110, "dbp": 72, "spo2": 97, "rr": 14, "temp_c": 36.8, "gcs": 15},
        "delta": {},
        "symptoms": ["lightheadedness"],
        "history": [],
        "labs": None,
        "expected_esi": 1,
        "expected_disposition": "escalate_icu",
    },
    {
        "chief_complaint": "High fever and runny nose",
        "base_vitals": {"hr": 98, "sbp": 118, "dbp": 76, "spo2": 97, "rr": 18, "temp_c": 39.2, "gcs": 15},
        "delta": {},
        "symptoms": ["fever", "rhinorrhea", "myalgia"],
        "history": [],
        "labs": None,
        "expected_esi": 3,
        "expected_disposition": "admit_ward",
    },
    {
        "chief_complaint": "Ankle sprain after walking",
        "base_vitals": {"hr": 78, "sbp": 122, "dbp": 80, "spo2": 99, "rr": 14, "temp_c": 37.0, "gcs": 15},
        "delta": {},
        "symptoms": ["ankle pain", "swelling"],
        "history": [],
        "labs": None,
        "expected_esi": 5,
        "expected_disposition": "discharge",
    },
]

_MEDIUM_TEMPLATES = [
    # 2-3 interacting conditions
    {
        "chief_complaint": "Chest pain and sweating in a diabetic patient",
        "base_vitals": {"hr": 112, "sbp": 95, "dbp": 62, "spo2": 94, "rr": 20, "temp_c": 37.3, "gcs": 15},
        "delta": {},
        "symptoms": ["chest pain", "diaphoresis", "nausea"],
        "history": ["type2_diabetes", "hypertension", "smoking"],
        "labs": {"troponin": 2.1, "glucose": 280, "creatinine": 1.4},
        "expected_esi": 1,
        "expected_disposition": "escalate_icu",
    },
    {
        "chief_complaint": "Confusion and shortness of breath in elderly COPD patient",
        "base_vitals": {"hr": 108, "sbp": 100, "dbp": 65, "spo2": 91, "rr": 26, "temp_c": 38.7, "gcs": 12},
        "delta": {},
        "symptoms": ["confusion", "shortness of breath", "cough", "fever"],
        "history": ["copd", "hypertension"],
        "labs": {"creatinine": 1.8, "lactate": 3.5},
        "expected_esi": 1,
        "expected_disposition": "escalate_icu",
    },
    {
        "chief_complaint": "Black tarry stools on warfarin therapy",
        "base_vitals": {"hr": 118, "sbp": 98, "dbp": 60, "spo2": 97, "rr": 18, "temp_c": 36.9, "gcs": 15},
        "delta": {},
        "symptoms": ["melena", "dizziness", "weakness"],
        "history": ["atrial_fibrillation", "warfarin", "nsaids"],
        "labs": {"inr": 6.2, "hemoglobin": 7.4},
        "expected_esi": 2,
        "expected_disposition": "admit_ward",
    },
    {
        "chief_complaint": "Fever and productive cough with renal failure",
        "base_vitals": {"hr": 102, "sbp": 108, "dbp": 68, "spo2": 93, "rr": 22, "temp_c": 39.1, "gcs": 14},
        "delta": {},
        "symptoms": ["fever", "cough", "sputum", "confusion"],
        "history": ["ckd", "hypertension"],
        "labs": {"creatinine": 3.2, "lactate": 2.8},
        "expected_esi": 2,
        "expected_disposition": "admit_ward",
    },
    {
        "chief_complaint": "Severe headache and elevated blood pressure",
        "base_vitals": {"hr": 90, "sbp": 195, "dbp": 118, "spo2": 97, "rr": 16, "temp_c": 37.0, "gcs": 14},
        "delta": {},
        "symptoms": ["severe headache", "visual changes", "nausea"],
        "history": ["hypertension", "chronic_kidney_disease"],
        "labs": {"creatinine": 2.5},
        "expected_esi": 1,
        "expected_disposition": "escalate_icu",
    },
]

_HARD_TEMPLATES = [
    # complex multi-system with contradictory signals
    {
        "chief_complaint": "Septic shock with AKI and coagulopathy on warfarin",
        "base_vitals": {"hr": 128, "sbp": 82, "dbp": 50, "spo2": 92, "rr": 28, "temp_c": 39.5, "gcs": 11},
        "delta": {},
        "symptoms": ["fever", "hypotension", "confusion", "oliguria"],
        "history": ["warfarin", "renal_failure", "diabetes", "atrial_fibrillation"],
        "labs": {"creatinine": 4.1, "lactate": 5.2, "inr": 4.8, "glucose": 380},
        "expected_esi": 1,
        "expected_disposition": "escalate_icu",
    },
    {
        "chief_complaint": "Stroke-like symptoms with active GI bleed on anticoagulation",
        "base_vitals": {"hr": 132, "sbp": 88, "dbp": 55, "spo2": 95, "rr": 20, "temp_c": 37.5, "gcs": 10},
        "delta": {},
        "symptoms": ["facial droop", "arm weakness", "slurred speech", "melena"],
        "history": ["warfarin", "atrial_fibrillation", "hypertension", "nsaids"],
        "labs": {"inr": 7.1, "hemoglobin": 6.8, "creatinine": 1.9},
        "expected_esi": 1,
        "expected_disposition": "escalate_icu",
    },
    {
        "chief_complaint": "DKA with pneumonia and suspected PE in a COPD patient",
        "base_vitals": {"hr": 138, "sbp": 92, "dbp": 58, "spo2": 89, "rr": 32, "temp_c": 38.9, "gcs": 13},
        "delta": {},
        "symptoms": ["shortness of breath", "pleuritic chest pain", "hemoptysis", "confusion", "vomiting"],
        "history": ["type1_diabetes", "copd", "smoking"],
        "labs": {"glucose": 520, "creatinine": 2.1, "lactate": 3.8, "potassium": 6.9},
        "expected_esi": 1,
        "expected_disposition": "escalate_icu",
    },
    {
        "chief_complaint": "Post-cardiac arrest ROSC with hemodynamic instability",
        "base_vitals": {"hr": 42, "sbp": 85, "dbp": 50, "spo2": 91, "rr": 8, "temp_c": 34.5, "gcs": 6},
        "delta": {},
        "symptoms": ["loss of consciousness", "cardiac arrest", "amnesia"],
        "history": ["cad", "hypertension", "chronic_kidney_disease"],
        "labs": {"troponin": 18.5, "lactate": 6.5, "creatinine": 2.8, "potassium": 6.2},
        "expected_esi": 1,
        "expected_disposition": "escalate_icu",
    },
    {
        "chief_complaint": "Multi-trauma with head injury, pneumothorax and hypotension",
        "base_vitals": {"hr": 145, "sbp": 78, "dbp": 45, "spo2": 88, "rr": 35, "temp_c": 35.8, "gcs": 8},
        "delta": {},
        "symptoms": ["loss of consciousness", "chest pain", "leg deformity", "respiratory distress"],
        "history": ["warfarin"],
        "labs": {"inr": 3.5, "hemoglobin": 6.2, "lactate": 7.1},
        "expected_esi": 1,
        "expected_disposition": "escalate_icu",
    },
]


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class PatientGenerator:
    """Seeded synthetic patient generator."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = random.Random(seed)
        self._counter = 0

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"PT-{self._counter:04d}"

    def _build_patient(self, template: dict, difficulty: str) -> PatientObservation:
        vitals = dict(template["base_vitals"])
        for key, delta in template.get("delta", {}).items():
            vitals[key] = vitals[key] + delta
        # Add small jitter so identical templates differ slightly across episodes
        jitter_keys = ["hr", "sbp", "rr", "temp_c"]
        jitter_ranges = {"hr": 5, "sbp": 8, "rr": 2, "temp_c": 0.3}
        for k in jitter_keys:
            if k in vitals:
                jitter = self._rng.uniform(-jitter_ranges[k], jitter_ranges[k])
                vitals[k] = round(vitals[k] + jitter, 1)

        return PatientObservation(
            patient_id=self._next_id(),
            chief_complaint=template["chief_complaint"],
            vitals=vitals,
            symptoms=list(template["symptoms"]),
            history=list(template["history"]),
            labs=dict(template["labs"]) if template.get("labs") else None,
            time_in_queue=self._rng.randint(0, 30),
            current_triage_level=None,
        )

    def generate_easy(self) -> PatientObservation:
        template = self._rng.choice(_EASY_TEMPLATES)
        return self._build_patient(template, "easy")

    def generate_medium(self) -> PatientObservation:
        template = self._rng.choice(_MEDIUM_TEMPLATES)
        return self._build_patient(template, "medium")

    def generate_hard(self) -> PatientObservation:
        template = self._rng.choice(_HARD_TEMPLATES)
        return self._build_patient(template, "hard")

    def generate_queue(
        self,
        n: int,
        difficulty: str = "mixed",
        seed_override: Optional[int] = None,
    ) -> list[PatientObservation]:
        """Generate a queue of n patients."""
        if seed_override is not None:
            rng_save = self._rng
            self._rng = random.Random(seed_override)

        generators = {
            "easy": self.generate_easy,
            "medium": self.generate_medium,
            "hard": self.generate_hard,
        }

        patients = []
        for i in range(n):
            if difficulty == "mixed":
                tier = self._rng.choice(["easy", "medium", "hard"])
            else:
                tier = difficulty
            patients.append(generators[tier]())

        if seed_override is not None:
            self._rng = rng_save  # type: ignore[assignment]

        return patients

    def get_template_by_difficulty(self, difficulty: str) -> dict:
        """Get a raw template for grader ground-truth comparison."""
        if difficulty == "easy":
            return self._rng.choice(_EASY_TEMPLATES)
        elif difficulty == "medium":
            return self._rng.choice(_MEDIUM_TEMPLATES)
        else:
            return self._rng.choice(_HARD_TEMPLATES)


# Global convenience generator (seeded externally)
_default_generator: Optional[PatientGenerator] = None


def get_generator(seed: Optional[int] = None) -> PatientGenerator:
    global _default_generator
    if _default_generator is None or seed is not None:
        _default_generator = PatientGenerator(seed=seed)
    return _default_generator
