# ClinicalTriageEnv

> **Meta × Hugging Face Reinforcement Learning Hackathon** — OpenEnv submission

[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-blue)](https://pypi.org/project/openenv-core/)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

**ClinicalTriageEnv** simulates a hospital emergency department. An RL agent or LLM agent must:

1. **Assess** synthetic patients with realistic presentations
2. **Assign** ESI (Emergency Severity Index) triage levels 1–5
3. **Order** evidence-based interventions while avoiding contraindicated medications
4. **Determine** appropriate dispositions (discharge / admit to ward / escalate to ICU)

All patient data is **100% synthetic** — no real PHI anywhere.

---

## Real-World Motivation

Emergency department overcrowding and triage errors cost lives. This environment provides:
- A safe, fully reproducible testbed for clinical decision-making agents
- 30+ deterministic clinical rules covering sepsis, ACS, stroke, coagulopathy, and more
- Dense reward shaping that penalises contraindicated orders and unsafe discharges
- Three progressive difficulty tiers matching real clinical complexity

---

## Observation Space

Each episode returns a `PatientObservation` JSON:

| Field | Type | Description |
|---|---|---|
| `patient_id` | `string` | Unique synthetic ID (e.g. `PT-0001`) |
| `chief_complaint` | `string` | Primary reason for ED visit |
| `vitals.hr` | `float` bpm | Heart rate (20–200) |
| `vitals.sbp` | `float` mmHg | Systolic blood pressure (50–250) |
| `vitals.dbp` | `float` mmHg | Diastolic blood pressure |
| `vitals.spo2` | `float` % | Oxygen saturation (70–100) |
| `vitals.rr` | `float` br/min | Respiratory rate (4–50) |
| `vitals.temp_c` | `float` °C | Temperature (32–42) |
| `vitals.gcs` | `float` | Glasgow Coma Scale (3–15) |
| `symptoms` | `list[str]` | Active symptoms |
| `history` | `list[str]` | Past medical/surgical history |
| `labs` | `dict \| null` | troponin, lactate, creatinine, glucose, INR, K+, Hgb |
| `time_in_queue` | `int` min | Minutes waiting (synthetic) |
| `current_triage_level` | `int \| null` | ESI level if already assigned |

---

## Action Space

Submit `TriageAction` JSON to `POST /step`:

| `action_type` | `value` | Effect |
|---|---|---|
| `assign_triage_level` | `"1"` – `"5"` | Set ESI triage level |
| `order_intervention` | intervention name | Order a clinical intervention |
| `set_disposition` | `"discharge"` \| `"admit_ward"` \| `"escalate_icu"` | Final patient routing |
| `request_labs` | lab panel name | Request diagnostic labs |
| `reassess` | optional note | Re-evaluate patient |
| `discharge` | optional note | Shorthand for safe discharge |

---

## Tasks

| Task | Difficulty | Patients | Steps | Grader | Baseline | Target |
|---|---|---|---|---|---|---|
| `task_1_easy` | Easy | 1 | 5 | Exact ESI match | 0.5 | 1.0 |
| `task_2_medium` | Medium | 5 | 20 | Kendall-tau × intervention score | 0.3 | 0.8 |
| `task_3_hard` | Hard | 8 | 20 | Triage 40% + Disposition 40% + Efficiency 20% | 0.2 | 0.7 |

### Reward Structure

```
+0.30  Correct ESI triage level
+0.20  Evidence-based intervention ordered
+0.30  Correct disposition (discharge/admit_ward/escalate_icu)
+0.10  Time efficiency bonus (high-acuity resolved in ≤5 steps)
−0.20  Contraindicated intervention ordered
−0.30  Critical patient (ESI 1-2) incorrectly discharged
−0.10  Wasted step on already-resolved patient
```

---

## Setup

### Local (Python)

```bash
# Clone / enter directory
cd clinical-triage-env

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run the agent
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
python inference.py
```

### Docker

```bash
# Build
docker build -t clinical-triage .

# Run
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -e MODEL_NAME=gpt-4o-mini \
  clinical-triage

# Verify health
curl http://localhost:7860/health
# → {"status":"ok"}
```

---

## API Reference

### `POST /reset`

Start a new episode.

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42}'
```

Response: `PatientObservation` JSON

### `POST /step`

Execute a triage action.

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"assign_triage_level","patient_id":"PT-0001","value":"2"}'
```

Response:
```json
{
  "observation": {...},
  "reward": 0.3,
  "done": false,
  "info": {
    "expected_triage_level": 2,
    "expected_disposition": "admit_ward",
    "safety_violations": []
  }
}
```

### `GET /state`

Full environment state including action history and reward breakdown.

### `GET /health`

```json
{"status": "ok"}
```

---

## Example Usage

```python
import requests

BASE = "http://localhost:7860"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"seed": 42}).json()
pid = obs["patient_id"]
print(f"Patient: {obs['chief_complaint']}")
print(f"SpO2: {obs['vitals']['spo2']}%  HR: {obs['vitals']['hr']} bpm")

# Assign triage level
result = requests.post(f"{BASE}/step", json={
    "action_type": "assign_triage_level",
    "patient_id": pid,
    "value": "2"
}).json()
print(f"Reward: {result['reward']}")

# Order intervention
result = requests.post(f"{BASE}/step", json={
    "action_type": "order_intervention",
    "patient_id": pid,
    "value": "supplemental_oxygen"
}).json()

# Set disposition
result = requests.post(f"{BASE}/step", json={
    "action_type": "set_disposition",
    "patient_id": pid,
    "value": "escalate_icu"
}).json()
print(f"Episode done: {result['done']}")
```

---

## Project Structure

```
clinical-triage-env/
├── openenv.yaml          ← Environment manifest
├── Dockerfile            ← HF Spaces deployment
├── inference.py          ← LLM agent runner
├── README.md
├── requirements.txt
├── env/
│   ├── __init__.py
│   ├── clinical_env.py   ← Core OpenEnv interface
│   ├── models.py         ← Pydantic v2 models
│   ├── patient_generator.py  ← Seeded synthetic generator
│   ├── graders.py        ← 3 deterministic graders
│   └── evidence_base.py  ← 30+ clinical rules
├── server/
│   └── app.py            ← FastAPI server
└── tests/
    └── test_env.py
```

---

## Clinical Rules (sample)

| Rule | Trigger | ESI | Contraindicated |
|---|---|---|---|
| `spo2_critical` | SpO2 < 90% | 1 | — |
| `bp_septic_shock` | SBP < 90 mmHg | 1 | — |
| `troponin_elevated` | Troponin ≥ 0.04 | 2 | NSAIDs |
| `creatinine_aki` | Creatinine ≥ 2.0 | 2 | NSAIDs, aminoglycosides |
| `gcs_unresponsive` | GCS ≤ 8 | 1 | Oral medications |
| `anaphylaxis` | Urticaria + SBP<90 | 1 | Beta-blockers |
| `renal_failure_nsaid` | CKD/AKI hx | — | NSAIDs |
| `warfarin_bleeding_reversal` | INR>5 + major bleed | 1 | NSAIDs, aspirin |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

> ⚠️ **Disclaimer**: All patients are synthetic. This environment is for research and education only. Do not use for real clinical decisions.
