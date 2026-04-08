"""
Pydantic v2 models for ClinicalTriageEnv.
All patient data is SYNTHETIC — no real PHI.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class PatientObservation(BaseModel):
    """Observation returned after each env step."""

    patient_id: str = Field(..., description="Unique synthetic patient identifier")
    chief_complaint: str = Field(..., description="Primary reason for ED visit")
    vitals: dict[str, float] = Field(
        ...,
        description=(
            "Key vital signs: hr (bpm), sbp (mmHg), dbp (mmHg), "
            "spo2 (%), rr (breaths/min), temp_c (°C), gcs (3-15)"
        ),
    )
    symptoms: list[str] = Field(default_factory=list, description="Active symptom list")
    history: list[str] = Field(
        default_factory=list, description="Past medical / surgical history"
    )
    labs: Optional[dict[str, float]] = Field(
        default=None,
        description="Available lab results (e.g. creatinine, lactate, troponin)",
    )
    time_in_queue: int = Field(
        default=0, description="Minutes patient has been waiting (synthetic)"
    )
    current_triage_level: Optional[int] = Field(
        default=None,
        description="ESI triage level assigned so far (1-5)",
    )

    model_config = {"json_schema_extra": {"example": {
        "patient_id": "PT-001",
        "chief_complaint": "Chest pain with diaphoresis",
        "vitals": {
            "hr": 110, "sbp": 88, "dbp": 58,
            "spo2": 93, "rr": 22, "temp_c": 37.2, "gcs": 15
        },
        "symptoms": ["chest pain", "diaphoresis", "nausea"],
        "history": ["hypertension", "type2_diabetes"],
        "labs": {"troponin": 2.4, "lactate": 3.1},
        "time_in_queue": 5,
        "current_triage_level": None,
    }}}


class TriageAction(BaseModel):
    """Action submitted by the agent."""

    action_type: Literal[
        "assign_triage_level",
        "order_intervention",
        "set_disposition",
        "request_labs",
        "reassess",
        "discharge",
    ] = Field(..., description="Type of clinical action")
    patient_id: str = Field(..., description="Target patient")
    value: Optional[str] = Field(
        default=None,
        description=(
            "Action payload. "
            "assign_triage_level: '1'-'5'; "
            "order_intervention: intervention name; "
            "set_disposition: 'discharge'|'admit_ward'|'escalate_icu'; "
            "request_labs: lab panel name; "
            "reassess/discharge: optional note"
        ),
    )

    model_config = {"json_schema_extra": {"example": {
        "action_type": "assign_triage_level",
        "patient_id": "PT-001",
        "value": "1",
    }}}


class TriageReward(BaseModel):
    """Structured reward signal."""

    score: float = Field(
        ..., ge=-1.0, le=1.0, description="Step reward, clipped to [-1, 1]"
    )
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward contributions (may be negative)",
    )
