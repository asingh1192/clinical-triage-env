"""
ClinicalTriageEnv — core OpenEnv-compatible environment.

Implements the Gymnasium-style API:
    reset(seed?)  → PatientObservation
    step(action)  → (PatientObservation, float, bool, dict)
    state()       → dict

Dense reward shaping (per spec):
    +0.3  correct ESI triage level
    +0.2  appropriate evidence-based intervention
    +0.3  correct disposition
    +0.1  time efficiency bonus (high-acuity resolved fast)
    -0.2  contraindicated intervention ordered
    -0.3  critical patient incorrectly discharged
    -0.1  wasted step on already-resolved patient
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from .evidence_base import (
    expected_disposition,
    expected_triage_level,
    is_contraindicated,
    is_safe_intervention,
)
from .models import PatientObservation, TriageAction, TriageReward
from .patient_generator import PatientGenerator


class ClinicalTriageEnv:
    """Full OpenEnv-compatible clinical triage environment."""

    # ── Constants ─────────────────────────────────────────────────────────
    MAX_STEPS: int = 50
    HIGH_ACUITY_LEVELS = {1, 2}
    FAST_RESOLUTION_THRESHOLD = 5  # steps

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._generator = PatientGenerator(seed=seed)
        self._episode_id: str = ""
        self._step_count: int = 0
        self._start_time: float = 0.0
        self._done: bool = True
        self._current_patient: Optional[PatientObservation] = None
        self._action_history: list[dict] = []
        self._assigned_triage_level: Optional[int] = None
        self._assigned_disposition: Optional[str] = None
        self._ordered_interventions: list[str] = []
        self._episode_rewards: list[float] = []
        self._safety_violations: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> PatientObservation:
        """Start a new episode with a fresh patient."""
        if seed is not None:
            self._seed = seed
            self._generator = PatientGenerator(seed=seed)

        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._start_time = time.time()
        self._done = False
        self._action_history = []
        self._assigned_triage_level = None
        self._assigned_disposition = None
        self._ordered_interventions = []
        self._episode_rewards = []
        self._safety_violations = []

        # Pick difficulty randomly based on seed for variety
        import random
        rng = random.Random(seed if seed is not None else self._seed)
        difficulty = rng.choice(["easy", "medium", "hard"])

        if difficulty == "easy":
            self._current_patient = self._generator.generate_easy()
        elif difficulty == "medium":
            self._current_patient = self._generator.generate_medium()
        else:
            self._current_patient = self._generator.generate_hard()

        return self._current_patient

    def step(self, action: TriageAction) -> tuple[PatientObservation, float, bool, dict]:
        """Execute a clinical action and return (obs, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")
        if self._current_patient is None:
            raise RuntimeError("No active patient. Call reset() first.")

        self._step_count += 1

        reward, breakdown = self._compute_reward(action)
        self._episode_rewards.append(reward)

        # Clamp to [-1, 1] range — negatives are intentional penalty signals
        clipped = max(-1.0, min(1.0, reward))

        # Update state based on action
        self._apply_action(action)

        # Episode ends on: disposition set, max steps exceeded, or discharge
        done = self._check_done(action)
        self._done = done

        # Observation after action (same patient, updated triage level)
        obs = PatientObservation(
            **self._current_patient.model_dump()
            | {"current_triage_level": self._assigned_triage_level}
        )

        info = {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "reward_breakdown": breakdown,
            "assigned_triage_level": self._assigned_triage_level,
            "assigned_disposition": self._assigned_disposition,
            "safety_violations": list(self._safety_violations),
            "expected_triage_level": expected_triage_level(
                self._current_patient.vitals,
                self._current_patient.labs,
                self._current_patient.history,
                self._current_patient.symptoms,
            ),
            "expected_disposition": expected_disposition(
                self._current_patient.vitals,
                self._current_patient.labs,
                self._current_patient.history,
                self._current_patient.symptoms,
            ),
        }

        return obs, clipped, done, info

    def state(self) -> dict[str, Any]:
        """Return full current environment state."""
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "done": self._done,
            "current_patient": (
                self._current_patient.model_dump()
                if self._current_patient
                else None
            ),
            "assigned_triage_level": self._assigned_triage_level,
            "assigned_disposition": self._assigned_disposition,
            "ordered_interventions": list(self._ordered_interventions),
            "action_history": list(self._action_history),
            "episode_rewards": list(self._episode_rewards),
            "safety_violations": list(self._safety_violations),
            "elapsed_seconds": round(time.time() - self._start_time, 2) if self._start_time else 0,
        }

    # ── Private Helpers ───────────────────────────────────────────────────

    def _compute_reward(self, action: TriageAction) -> tuple[float, dict]:
        """Dense reward computation."""
        assert self._current_patient is not None
        p = self._current_patient
        breakdown: dict[str, float] = {}

        # Wasted step penalty (already done)
        if self._done:
            breakdown["wasted_step"] = -0.1
            return -0.1, breakdown

        # ── Triage level assignment ──────────────────────────────────────
        if action.action_type == "assign_triage_level" and action.value:
            try:
                assigned = int(action.value)
            except ValueError:
                assigned = 5
            expected = expected_triage_level(p.vitals, p.labs, p.history, p.symptoms)
            if assigned == expected:
                breakdown["triage_exact"] = 0.3
            elif abs(assigned - expected) == 1:
                breakdown["triage_close"] = 0.15
            else:
                breakdown["triage_wrong"] = 0.0

        # ── Intervention ordering ────────────────────────────────────────
        elif action.action_type == "order_intervention" and action.value:
            iv = action.value
            if is_contraindicated(iv, p.vitals, p.labs, p.history, p.symptoms):
                breakdown["contraindicated_intervention"] = -0.2
                self._safety_violations.append(f"contraindicated:{iv}")
            elif is_safe_intervention(iv, p.vitals, p.labs, p.history, p.symptoms):
                breakdown["evidence_based_intervention"] = 0.2
            else:
                breakdown["neutral_intervention"] = 0.05

        # ── Disposition ──────────────────────────────────────────────────
        elif action.action_type == "set_disposition" and action.value:
            disp = action.value
            expected_disp = expected_disposition(p.vitals, p.labs, p.history, p.symptoms)
            expected_lvl = expected_triage_level(p.vitals, p.labs, p.history, p.symptoms)

            if disp == expected_disp:
                breakdown["correct_disposition"] = 0.3
            else:
                breakdown["wrong_disposition"] = 0.0

            # Critical safety: discharging a high-acuity patient
            if disp == "discharge" and expected_lvl in self.HIGH_ACUITY_LEVELS:
                breakdown["critical_discharge_penalty"] = -0.3
                self._safety_violations.append(f"critical_discharge:ESI{expected_lvl}")

            # Time efficiency bonus: high acuity resolved quickly
            if expected_lvl in self.HIGH_ACUITY_LEVELS and self._step_count <= self.FAST_RESOLUTION_THRESHOLD:
                breakdown["time_efficiency"] = 0.1

        # ── Discharge ────────────────────────────────────────────────────
        elif action.action_type == "discharge":
            expected_lvl = expected_triage_level(p.vitals, p.labs, p.history, p.symptoms)
            if expected_lvl in self.HIGH_ACUITY_LEVELS:
                breakdown["unsafe_discharge_penalty"] = -0.3
                self._safety_violations.append(f"unsafe_discharge:ESI{expected_lvl}")
            else:
                breakdown["safe_discharge"] = 0.2

        # ── Labs / Reassess (small positive) ────────────────────────────
        elif action.action_type in ("request_labs", "reassess"):
            breakdown["diagnostic_step"] = 0.05

        total = sum(breakdown.values())
        return total, breakdown

    def _apply_action(self, action: TriageAction) -> None:
        """Mutate env state based on action."""
        record = action.model_dump()
        record["step"] = self._step_count
        self._action_history.append(record)

        if action.action_type == "assign_triage_level" and action.value:
            try:
                self._assigned_triage_level = int(action.value)
            except ValueError:
                pass

        elif action.action_type == "order_intervention" and action.value:
            self._ordered_interventions.append(action.value)

        elif action.action_type == "set_disposition" and action.value:
            self._assigned_disposition = action.value

        elif action.action_type == "discharge":
            self._assigned_disposition = "discharge"

    def _check_done(self, action: TriageAction) -> bool:
        """Episode terminates when disposition is set or max steps reached."""
        if self._step_count >= self.MAX_STEPS:
            return True
        if action.action_type in ("set_disposition", "discharge"):
            return True
        if self._assigned_disposition is not None:
            return True
        return False
