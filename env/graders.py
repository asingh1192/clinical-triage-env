"""
Deterministic graders for ClinicalTriageEnv tasks 1, 2, and 3.
All return float in [0.0, 1.0].
"""
from __future__ import annotations

import math
from typing import Any

from .evidence_base import (
    expected_disposition,
    expected_triage_level,
    is_contraindicated,
    is_safe_intervention,
)
from .models import PatientObservation
from .patient_generator import PatientGenerator

_SCORE_MIN = 0.001
_SCORE_MAX = 0.999

def _clamp(score: float) -> float:
    return round(max(_SCORE_MIN, min(_SCORE_MAX, float(score))), 4)


# ---------------------------------------------------------------------------
# Task 1 — Easy: Single patient, textbook ESI level matching
# ---------------------------------------------------------------------------

def grade_task1(
    patient: PatientObservation,
    assigned_triage_level: int,
    *,
    seed: int = 42,
) -> float:
    """
    Grade a single-patient triage assignment.
      assigned_triage_level == expected → 1.0
      |diff| == 1                       → 0.5
      else                              → 0.0
    """
    expected = expected_triage_level(
        patient.vitals, patient.labs, patient.history, patient.symptoms
    )
    diff = abs(assigned_triage_level - expected)
    if diff == 0:
        return _clamp(0.95)
    elif diff == 1:
        return _clamp(0.5)
    else:
        return _clamp(0.001)


# ---------------------------------------------------------------------------
# Kendall-tau helpers
# ---------------------------------------------------------------------------

def _kendall_tau(rank_a: list[int], rank_b: list[int]) -> float:
    """Compute Kendall-tau correlation in [0, 1] (normalized from [-1, 1])."""
    n = len(rank_a)
    if n < 2:
        return 0.95
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            delta_a = rank_a[i] - rank_a[j]
            delta_b = rank_b[i] - rank_b[j]
            product = delta_a * delta_b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.95
    tau = (concordant - discordant) / (n * (n - 1) / 2)
    score = (tau + 1) / 2  # map [-1, 1] → [0, 1]
    if score == 1.0:
        return 0.95
    if score == 0.0:
        return 0.001
    return score


# ---------------------------------------------------------------------------
# Task 2 — Medium: 5-patient queue triage ordering + intervention
# ---------------------------------------------------------------------------

def grade_task2(
    patients: list[PatientObservation],
    assigned_levels: list[int],
    agent_interventions: list[str | None],
    *,
    seed: int = 42,
) -> float:
    """
    Grade a 5-patient queue:
      Kendall-tau rank correlation (triage urgency ordering) × intervention score.
    Returns float in [0, 1].
    """
    if len(patients) != 5:
        raise ValueError(f"Task 2 requires exactly 5 patients, got {len(patients)}")
    if len(assigned_levels) != 5 or len(agent_interventions) != 5:
        raise ValueError("assigned_levels and agent_interventions must each have 5 elements")

    # ── Rank ordering score ──────────────────────────────────────────────
    expected_levels = [
        expected_triage_level(p.vitals, p.labs, p.history, p.symptoms)
        for p in patients
    ]

    # Convert ESI levels to urgency ranks (lower ESI = higher urgency → lower rank)
    def _to_rank(levels: list[int]) -> list[int]:
        # rank 1 = most urgent
        indexed = sorted(enumerate(levels), key=lambda x: x[1])
        ranks = [0] * len(levels)
        for rank, (i, _) in enumerate(indexed, start=1):
            ranks[i] = rank
        return ranks

    expected_ranks = _to_rank(expected_levels)
    assigned_ranks = _to_rank(assigned_levels)
    tau_score = _kendall_tau(expected_ranks, assigned_ranks)

    # ── Intervention score ───────────────────────────────────────────────
    iv_scores = []
    for p, iv in zip(patients, agent_interventions):
        if iv is None or iv.strip() == "":
            iv_scores.append(0.001)
        elif is_contraindicated(iv, p.vitals, p.labs, p.history, p.symptoms):
            iv_scores.append(0.05)
        elif is_safe_intervention(iv, p.vitals, p.labs, p.history, p.symptoms):
            iv_scores.append(0.95)
        else:
            iv_scores.append(0.3)  # neutral intervention

    intervention_score = sum(iv_scores) / len(iv_scores)

    final = tau_score * intervention_score
    return _clamp(final)


# ---------------------------------------------------------------------------
# Task 3 — Hard: 8-patient mixed queue, full workflow ≤ 20 steps
# ---------------------------------------------------------------------------

def grade_task3(
    patients: list[PatientObservation],
    assigned_levels: list[int],
    assigned_dispositions: list[str],
    agent_interventions: list[list[str]],
    total_steps: int,
    safety_violations: list[str],
    *,
    seed: int = 42,
) -> float:
    """
    Grade an 8-patient full workflow:
      - Triage accuracy    40% (exact ESI level match rate)
      - Disposition        40% (correct disposition rate)
      - Efficiency bonus   20% (≤20 steps)
      - Safety penalty     −0.15 per violation
    Returns float in [0, 1].
    """
    if len(patients) != 8:
        raise ValueError(f"Task 3 requires exactly 8 patients, got {len(patients)}")

    # ── Triage accuracy ──────────────────────────────────────────────────
    triage_scores = []
    for p, al in zip(patients, assigned_levels):
        expected = expected_triage_level(p.vitals, p.labs, p.history, p.symptoms)
        if al == expected:
            triage_scores.append(1.0)
        elif abs(al - expected) == 1:
            triage_scores.append(0.5)
        else:
            triage_scores.append(0.0)
    triage_accuracy = sum(triage_scores) / len(triage_scores)

    # ── Disposition correctness ──────────────────────────────────────────
    disp_scores = []
    for p, disp in zip(patients, assigned_dispositions):
        expected = expected_disposition(p.vitals, p.labs, p.history, p.symptoms)
        disp_scores.append(1.0 if disp == expected else 0.0)
    disp_accuracy = sum(disp_scores) / len(disp_scores)

    # ── Efficiency bonus ─────────────────────────────────────────────────
    efficiency = 1.0 if total_steps <= 20 else max(0.0, 1.0 - (total_steps - 20) * 0.05)

    # ── Composite score ──────────────────────────────────────────────────
    composite = (
        0.40 * triage_accuracy
        + 0.40 * disp_accuracy
        + 0.20 * efficiency
    )

    # ── Safety penalties ─────────────────────────────────────────────────
    penalty = len(safety_violations) * 0.15
    composite = max(0.001, composite - penalty)

    return _clamp(composite)


# ---------------------------------------------------------------------------
# Convenience: generate graded task episodes from seed
# ---------------------------------------------------------------------------

def build_task1_episode(seed: int = 42) -> dict[str, Any]:
    """Return a pre-built Task 1 episode dict for the agent to solve."""
    gen = PatientGenerator(seed=seed)
    patient = gen.generate_easy()
    expected = expected_triage_level(
        patient.vitals, patient.labs, patient.history, patient.symptoms
    )
    return {
        "patient": patient,
        "expected_triage_level": expected,
        "expected_disposition": expected_disposition(
            patient.vitals, patient.labs, patient.history, patient.symptoms
        ),
    }


def build_task2_episode(seed: int = 42) -> dict[str, Any]:
    """Return a pre-built Task 2 episode dict for the agent to solve."""
    gen = PatientGenerator(seed=seed)
    patients = [gen.generate_medium() for _ in range(5)]
    expected_levels = [
        expected_triage_level(p.vitals, p.labs, p.history, p.symptoms)
        for p in patients
    ]
    expected_dispositions = [
        expected_disposition(p.vitals, p.labs, p.history, p.symptoms)
        for p in patients
    ]
    return {
        "patients": patients,
        "expected_triage_levels": expected_levels,
        "expected_dispositions": expected_dispositions,
    }


def build_task3_episode(seed: int = 42) -> dict[str, Any]:
    """Return a pre-built Task 3 episode dict for the agent to solve."""
    gen = PatientGenerator(seed=seed)
    patients = [
        gen.generate_easy(),
        gen.generate_easy(),
        gen.generate_medium(),
        gen.generate_medium(),
        gen.generate_medium(),
        gen.generate_hard(),
        gen.generate_hard(),
        gen.generate_hard(),
    ]
    expected_levels = [
        expected_triage_level(p.vitals, p.labs, p.history, p.symptoms)
        for p in patients
    ]
    expected_dispositions = [
        expected_disposition(p.vitals, p.labs, p.history, p.symptoms)
        for p in patients
    ]
    return {
        "patients": patients,
        "expected_triage_levels": expected_levels,
        "expected_dispositions": expected_dispositions,
    }
