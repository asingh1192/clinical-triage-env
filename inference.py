# -*- coding: utf-8 -*-
"""
inference.py — LLM agent runner for ClinicalTriageEnv.

Reads from environment variables:
  API_BASE_URL  — OpenAI-compatible API base URL
  MODEL_NAME    — model identifier
  HF_TOKEN      — Hugging Face / API token (used as api_key)

Runs the agent against all 3 tasks sequentially and prints scores.
Uses ONLY the OpenAI Python client for LLM calls.

Target runtime: < 10 minutes total.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from typing import Any, Optional

import requests

# Fix Windows console encoding so print() works with any charset
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
OPENAI_API_BASE: str = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", HF_TOKEN or "dummy")

ENV_TIMEOUT: int = 30  # seconds per HTTP request
MAX_RETRY: int = 3

# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})


def _http_post(path: str, payload: dict) -> dict:
    url = f"{API_BASE_URL}{path}"
    for attempt in range(MAX_RETRY):
        try:
            r = _session.post(url, json=payload, timeout=ENV_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == MAX_RETRY - 1:
                print(f"[WARN] POST {path} failed after {MAX_RETRY} attempts: {exc}")
                return {}
            time.sleep(1)
    return {}


def _http_get(path: str) -> dict:
    url = f"{API_BASE_URL}{path}"
    for attempt in range(MAX_RETRY):
        try:
            r = _session.get(url, timeout=ENV_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == MAX_RETRY - 1:
                print(f"[WARN] GET {path} failed after {MAX_RETRY} attempts: {exc}")
                return {}
            time.sleep(1)
    return {}


def env_reset(seed: Optional[int] = None) -> dict:
    payload = {} if seed is None else {"seed": seed}
    return _http_post("/reset", payload)


def env_step(action_type: str, patient_id: str, value: Optional[str] = None) -> dict:
    payload: dict = {"action_type": action_type, "patient_id": patient_id}
    if value is not None:
        payload["value"] = value
    return _http_post("/step", payload)


def env_state() -> dict:
    return _http_get("/state")

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def get_llm_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=OPENAI_API_KEY or "dummy",
            base_url=OPENAI_API_BASE,
        )
    return _client


def llm_call(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    """Call LLM with fallback to deterministic response on failure."""
    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}. Using rule-based fallback.")
        return ""

# ---------------------------------------------------------------------------
# Rule-based fallback (used when LLM fails or returns malformed output)
# ---------------------------------------------------------------------------

def _rule_based_triage_level(obs: dict) -> int:
    """Deterministic fallback ESI assignment from vitals."""
    v = obs.get("vitals", {})
    spo2  = v.get("spo2",  99)
    sbp   = v.get("sbp",  120)
    hr    = v.get("hr",    80)
    gcs   = v.get("gcs",   15)
    rr    = v.get("rr",    16)
    temp  = v.get("temp_c", 37.0)
    labs  = obs.get("labs") or {}
    lactate   = labs.get("lactate", 0)
    troponin  = labs.get("troponin", 0)
    creatinine = labs.get("creatinine", 0)

    # ESI-1 criteria
    if (gcs <= 8 or spo2 < 90 or sbp < 90 or hr < 40 or rr > 30 or rr < 8
            or lactate >= 4.0 or temp >= 40.0):
        return 1
    # ESI-2 criteria
    if (spo2 < 94 or sbp < 100 or hr > 120 or hr < 50 or gcs < 13
            or lactate >= 2.0 or troponin >= 0.04 or creatinine >= 2.0
            or temp < 35.0 or sbp >= 180):
        return 2
    # ESI-3
    if (100 < hr <= 120 or 24 < rr <= 30 or 38.5 <= temp < 40.0 or spo2 < 96):
        return 3
    # ESI-4/5
    if len(obs.get("symptoms", [])) <= 1 and not obs.get("labs"):
        return 5
    return 4


def _rule_based_disposition(obs: dict) -> str:
    level = _rule_based_triage_level(obs)
    if level == 1:
        return "escalate_icu"
    elif level <= 3:
        return "admit_ward"
    else:
        return "discharge"


def _rule_based_intervention(obs: dict) -> str:
    v = obs.get("vitals", {})
    labs = obs.get("labs") or {}
    hist = " ".join(obs.get("history", [])).lower()
    symp = " ".join(obs.get("symptoms", [])).lower()
    spo2 = v.get("spo2", 99)
    sbp  = v.get("sbp", 120)
    if spo2 < 94:
        return "supplemental_oxygen"
    if sbp < 90:
        return "iv_fluid_bolus"
    if labs.get("troponin", 0) >= 0.04:
        return "aspirin"
    if labs.get("creatinine", 0) >= 2.0:
        return "iv_fluids"
    if "chest pain" in symp:
        return "ecg_12lead"
    if "fever" in symp:
        return "blood_cultures"
    return "iv_access"

# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior emergency medicine physician performing triage.
Given a patient observation JSON, you MUST respond with ONLY a valid JSON object.
Never include explanation text outside the JSON.

ESI Triage Levels:
  1 = Immediate (life-threatening)
  2 = Emergent (high risk, severe pain)
  3 = Urgent (multiple resources needed)
  4 = Less urgent (1 resource)
  5 = Non-urgent (no resources)

Dispositions: discharge | admit_ward | escalate_icu

Always base decisions on clinical evidence. Flag contraindicated medications."""

def _parse_llm_json(text: str, fallback: dict) -> dict:
    """Extract JSON from LLM output with fallback."""
    text = text.strip()
    # Try to find JSON block
    for start_char in ["{", "["]:
        idx = text.find(start_char)
        if idx >= 0:
            try:
                return json.loads(text[idx:])
            except Exception:
                pass
    # Try full parse
    try:
        return json.loads(text)
    except Exception:
        return fallback

# ---------------------------------------------------------------------------
# Task 1 — Easy: Single patient triage
# ---------------------------------------------------------------------------

def run_task1(seed: int = 42) -> float:
    print("\n" + "="*60)
    print("TASK 1 — Easy: Single Patient Triage")
    print("="*60)

    obs = env_reset(seed=seed)
    if not obs:
        print("[ERROR] Could not reset environment for Task 1")
        return 0.0

    pid = obs.get("patient_id", "PT-0001")
    print(f"Patient: {pid} | Complaint: {obs.get('chief_complaint','N/A')}")
    print(f"Vitals: {obs.get('vitals', {})}")

    # Ask LLM
    user_msg = (
        f"Patient observation:\n{json.dumps(obs, indent=2)}\n\n"
        "Respond ONLY with JSON: "
        '{"triage_level": <1-5>, "intervention": "<name>", "disposition": "<discharge|admit_ward|escalate_icu>", "reasoning": "<brief>"}'
    )
    llm_raw = llm_call(SYSTEM_PROMPT, user_msg)
    decision = _parse_llm_json(llm_raw, {})

    triage_level = decision.get("triage_level")
    if not isinstance(triage_level, int) or not (1 <= triage_level <= 5):
        triage_level = _rule_based_triage_level(obs)
        print(f"[fallback] LLM gave invalid triage_level → using rule-based: {triage_level}")

    intervention = decision.get("intervention") or _rule_based_intervention(obs)
    disposition  = decision.get("disposition")  or _rule_based_disposition(obs)

    print(f"LLM Decision → ESI: {triage_level}, Intervention: {intervention}, Disposition: {disposition}")

    # Execute actions against the env
    env_step("assign_triage_level", pid, str(triage_level))
    env_step("order_intervention", pid, intervention)
    result = env_step("set_disposition", pid, disposition)

    reward = result.get("reward", 0.0) if result else 0.0

    # Also compute grader score
    from env.graders import grade_task1
    from env.models import PatientObservation
    try:
        patient_obj = PatientObservation(**obs)
        score = grade_task1(patient_obj, triage_level, seed=seed)
    except Exception as exc:
        print(f"[WARN] Grader failed: {exc}, using env reward")
        score = reward

    print(f"Task 1 Score: {score:.4f}")
    return score

# ---------------------------------------------------------------------------
# Task 2 — Medium: 5-patient queue
# ---------------------------------------------------------------------------

def run_task2(seed: int = 42) -> float:
    print("\n" + "="*60)
    print("TASK 2 — Medium: 5-Patient Queue Triage + Interventions")
    print("="*60)

    from env.graders import build_task2_episode, grade_task2
    from env.models import PatientObservation

    # Build episode locally for grading (deterministic)
    try:
        episode = build_task2_episode(seed=seed)
        patients_data = episode["patients"]
    except Exception as exc:
        print(f"[ERROR] Could not build task2 episode: {exc}")
        return 0.0

    assigned_levels: list[int] = []
    agent_interventions: list[str | None] = []

    for i, patient in enumerate(patients_data):
        obs_dict = patient.model_dump()
        pid = obs_dict["patient_id"]
        print(f"  Patient {i+1}: {pid} | {obs_dict['chief_complaint'][:50]}")

        user_msg = (
            f"Patient {i+1}/5:\n{json.dumps(obs_dict, indent=2)}\n\n"
            "Respond ONLY with JSON: "
            '{"triage_level": <1-5>, "intervention": "<name>", "reasoning": "<brief>"}'
        )
        llm_raw = llm_call(SYSTEM_PROMPT, user_msg, max_tokens=256)
        decision = _parse_llm_json(llm_raw, {})

        triage_level = decision.get("triage_level")
        if not isinstance(triage_level, int) or not (1 <= triage_level <= 5):
            triage_level = _rule_based_triage_level(obs_dict)

        intervention = decision.get("intervention") or _rule_based_intervention(obs_dict)

        print(f"    → ESI {triage_level} | Intervention: {intervention}")
        assigned_levels.append(triage_level)
        agent_interventions.append(intervention)

        # Execute in env (reset for each patient)
        env_reset(seed=seed + i + 1)
        env_step("assign_triage_level", pid, str(triage_level))
        env_step("order_intervention", pid, intervention)

    score = grade_task2(
        patients_data,
        assigned_levels,
        agent_interventions,
        seed=seed,
    )
    print(f"Task 2 Score: {score:.4f}")
    return score

# ---------------------------------------------------------------------------
# Task 3 — Hard: 8-patient full workflow ≤ 20 steps
# ---------------------------------------------------------------------------

def run_task3(seed: int = 42) -> float:
    print("\n" + "="*60)
    print("TASK 3 — Hard: 8-Patient Full Workflow (≤20 steps)")
    print("="*60)

    from env.graders import build_task3_episode, grade_task3
    from env.models import PatientObservation

    try:
        episode = build_task3_episode(seed=seed)
        patients_data = episode["patients"]
    except Exception as exc:
        print(f"[ERROR] Could not build task3 episode: {exc}")
        return 0.0

    assigned_levels: list[int] = []
    assigned_dispositions: list[str] = []
    agent_interventions: list[list[str]] = []
    safety_violations: list[str] = []
    total_steps = 0

    for i, patient in enumerate(patients_data):
        obs_dict = patient.model_dump()
        pid = obs_dict["patient_id"]
        print(f"  Patient {i+1}: {pid} | {obs_dict['chief_complaint'][:50]}")

        user_msg = (
            f"Patient {i+1}/8 (full workflow required):\n{json.dumps(obs_dict, indent=2)}\n\n"
            "Respond ONLY with JSON: "
            '{"triage_level": <1-5>, '
            '"interventions": ["<iv1>", "<iv2>"], '
            '"disposition": "<discharge|admit_ward|escalate_icu>", '
            '"reasoning": "<brief>"}'
        )
        llm_raw = llm_call(SYSTEM_PROMPT, user_msg, max_tokens=400)
        decision = _parse_llm_json(llm_raw, {})

        triage_level = decision.get("triage_level")
        if not isinstance(triage_level, int) or not (1 <= triage_level <= 5):
            triage_level = _rule_based_triage_level(obs_dict)

        interventions = decision.get("interventions") or [_rule_based_intervention(obs_dict)]
        if not isinstance(interventions, list):
            interventions = [_rule_based_intervention(obs_dict)]
        interventions = [str(iv) for iv in interventions[:3]]  # max 3

        disposition = decision.get("disposition") or _rule_based_disposition(obs_dict)
        if disposition not in ("discharge", "admit_ward", "escalate_icu"):
            disposition = _rule_based_disposition(obs_dict)

        print(f"    → ESI {triage_level} | {interventions} | {disposition}")

        # Execute in env
        env_reset(seed=seed + i + 100)
        env_step("assign_triage_level", pid, str(triage_level))
        total_steps += 2

        for iv in interventions:
            result = env_step("order_intervention", pid, iv)
            total_steps += 1
            info = (result or {}).get("info", {})
            safety_violations.extend(info.get("safety_violations", []))

        env_step("set_disposition", pid, disposition)
        total_steps += 1

        assigned_levels.append(triage_level)
        assigned_dispositions.append(disposition)
        agent_interventions.append(interventions)

    print(f"  Total steps used: {total_steps}/20")

    score = grade_task3(
        patients_data,
        assigned_levels,
        assigned_dispositions,
        agent_interventions,
        total_steps,
        safety_violations,
        seed=seed,
    )
    print(f"Task 3 Score: {score:.4f}")
    return score

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("============================================================")
    print("         ClinicalTriageEnv -- LLM Agent Inference          ")
    print("============================================================")
    print(f"API Base URL : {API_BASE_URL}")
    print(f"Model        : {MODEL_NAME}")
    print(f"OpenAI Base  : {OPENAI_API_BASE}")

    # Verify server is reachable
    health = _http_get("/health")
    if health.get("status") != "ok":
        print(f"[WARNING] Health check response: {health}")
        print("Server may not be running. Attempting to continue with local graders.")

    start_time = time.time()

    scores: dict[str, float] = {}

    # Run all 3 tasks
    try:
        scores["task_1_easy"] = run_task1(seed=42)
    except Exception as exc:
        print(f"[ERROR] Task 1 failed: {exc}")
        scores["task_1_easy"] = 0.0

    try:
        scores["task_2_medium"] = run_task2(seed=42)
    except Exception as exc:
        print(f"[ERROR] Task 2 failed: {exc}")
        scores["task_2_medium"] = 0.0

    try:
        scores["task_3_hard"] = run_task3(seed=42)
    except Exception as exc:
        print(f"[ERROR] Task 3 failed: {exc}")
        scores["task_3_hard"] = 0.0

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    for task, score in scores.items():
        bar = "#" * int(score * 20)
        print(f"  {task:<20} {score:.4f}  |{bar:<20}|")
    mean_score = sum(scores.values()) / len(scores)
    print(f"  {'Mean':<20} {mean_score:.4f}")
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Validate all scores are in [0.0, 1.0]
    all_valid = all(0.0 <= s <= 1.0 for s in scores.values())
    if all_valid:
        print("\n[OK] All scores valid (0.0 - 1.0) - submission ready!")
    else:
        print("\n[FAIL] Some scores out of range - check graders!")
        sys.exit(1)


if __name__ == "__main__":
    main()
