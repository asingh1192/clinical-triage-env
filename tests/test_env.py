"""
Tests for ClinicalTriageEnv.
Run with: python -m pytest tests/test_env.py -v
"""
from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from env.clinical_env import ClinicalTriageEnv
from env.models import PatientObservation, TriageAction, TriageReward
from env.evidence_base import (
    expected_triage_level,
    expected_disposition,
    is_contraindicated,
    is_safe_intervention,
    get_triggered_rules,
    CLINICAL_RULES,
)
from env.patient_generator import PatientGenerator
from env.graders import (
    grade_task1,
    grade_task2,
    grade_task3,
    build_task1_episode,
    build_task2_episode,
    build_task3_episode,
)


# ── Evidence base tests ─────────────────────────────────────────────────────

class TestEvidenceBase:
    def test_spo2_critical_triggers(self):
        vitals = {"spo2": 88, "hr": 100, "sbp": 110, "gcs": 15, "rr": 18, "temp_c": 37.0}
        rules = get_triggered_rules(vitals, None, [], [])
        assert "spo2_critical" in rules

    def test_spo2_normal_no_trigger(self):
        vitals = {"spo2": 99, "hr": 75, "sbp": 120, "gcs": 15, "rr": 16, "temp_c": 37.0}
        rules = get_triggered_rules(vitals, None, [], [])
        assert "spo2_critical" not in rules

    def test_septic_shock_esi1(self):
        vitals = {"spo2": 95, "hr": 120, "sbp": 82, "gcs": 13, "rr": 22, "temp_c": 39.0}
        level = expected_triage_level(vitals, None, [], [])
        assert level == 1

    def test_hypertensive_crisis_esi2(self):
        vitals = {"spo2": 97, "hr": 88, "sbp": 185, "dbp": 110, "gcs": 15, "rr": 16, "temp_c": 37.0}
        level = expected_triage_level(vitals, None, [], [])
        assert level == 2

    def test_nsaid_contraindicated_aki(self):
        vitals = {"spo2": 97, "hr": 80, "sbp": 130, "gcs": 15, "rr": 16, "temp_c": 37.0}
        labs = {"creatinine": 2.5}
        assert is_contraindicated("nsaids", vitals, labs, [], [])

    def test_nsaid_not_contraindicated_healthy(self):
        vitals = {"spo2": 99, "hr": 72, "sbp": 118, "gcs": 15, "rr": 14, "temp_c": 37.0}
        assert not is_contraindicated("nsaids", vitals, None, [], [])

    def test_oxygen_safe_hypoxia(self):
        vitals = {"spo2": 92, "hr": 100, "sbp": 115, "gcs": 15, "rr": 22, "temp_c": 37.0}
        assert is_safe_intervention("supplemental_oxygen", vitals, None, [], [])

    def test_troponin_esi2(self):
        vitals = {"spo2": 97, "hr": 90, "sbp": 110, "gcs": 15, "rr": 18, "temp_c": 37.0}
        labs = {"troponin": 1.2}
        level = expected_triage_level(vitals, labs, [], [])
        assert level <= 2

    def test_lactate_critical_esi1(self):
        vitals = {"spo2": 95, "hr": 105, "sbp": 95, "gcs": 14, "rr": 22, "temp_c": 38.5}
        labs = {"lactate": 5.0}
        level = expected_triage_level(vitals, labs, [], [])
        assert level == 1

    def test_gcs_8_esi1(self):
        vitals = {"spo2": 97, "hr": 88, "sbp": 115, "gcs": 7, "rr": 16, "temp_c": 37.0}
        level = expected_triage_level(vitals, None, [], [])
        assert level == 1

    def test_rule_count(self):
        """Ensure we have at least 30 rules."""
        assert len(CLINICAL_RULES) >= 30

    def test_disposition_critical(self):
        vitals = {"spo2": 87, "hr": 130, "sbp": 85, "gcs": 10, "rr": 28, "temp_c": 39.5}
        labs = {"lactate": 5.5}
        disp = expected_disposition(vitals, labs, [], [])
        assert disp == "escalate_icu"

    def test_disposition_discharge(self):
        vitals = {"spo2": 99, "hr": 75, "sbp": 125, "gcs": 15, "rr": 14, "temp_c": 37.0}
        disp = expected_disposition(vitals, None, [], ["ankle pain"])
        assert disp == "discharge"


# ── Patient generator tests ─────────────────────────────────────────────────

class TestPatientGenerator:
    def test_easy_patient_structure(self):
        gen = PatientGenerator(seed=0)
        p = gen.generate_easy()
        assert isinstance(p, PatientObservation)
        assert p.patient_id.startswith("PT-")
        assert "hr" in p.vitals
        assert "spo2" in p.vitals

    def test_medium_patient_has_history(self):
        gen = PatientGenerator(seed=0)
        p = gen.generate_medium()
        assert len(p.history) > 0 or len(p.symptoms) > 0

    def test_hard_patient_has_labs(self):
        gen = PatientGenerator(seed=0)
        p = gen.generate_hard()
        assert p.labs is not None

    def test_seed_reproducibility(self):
        gen1 = PatientGenerator(seed=123)
        gen2 = PatientGenerator(seed=123)
        p1 = gen1.generate_easy()
        p2 = gen2.generate_easy()
        assert p1.model_dump() == p2.model_dump()

    def test_different_seeds_differ(self):
        gen1 = PatientGenerator(seed=1)
        gen2 = PatientGenerator(seed=2)
        p1 = gen1.generate_hard()
        p2 = gen2.generate_hard()
        # Patient IDs differ
        assert p1.chief_complaint != p2.chief_complaint or p1.vitals != p2.vitals

    def test_queue_generation(self):
        gen = PatientGenerator(seed=42)
        queue = gen.generate_queue(5, difficulty="mixed")
        assert len(queue) == 5
        for p in queue:
            assert isinstance(p, PatientObservation)


# ── Core environment tests ──────────────────────────────────────────────────

class TestClinicalEnv:
    def setup_method(self):
        self.env = ClinicalTriageEnv(seed=42)

    def test_reset_returns_observation(self):
        obs = self.env.reset(seed=42)
        assert isinstance(obs, PatientObservation)
        assert obs.patient_id is not None

    def test_step_triage_level(self):
        obs = self.env.reset(seed=42)
        action = TriageAction(
            action_type="assign_triage_level",
            patient_id=obs.patient_id,
            value="2",
        )
        next_obs, reward, done, info = self.env.step(action)
        assert isinstance(next_obs, PatientObservation)
        assert 0.0 <= reward <= 1.0
        assert isinstance(done, bool)
        assert "expected_triage_level" in info

    def test_step_intervention(self):
        obs = self.env.reset(seed=1)
        action = TriageAction(
            action_type="order_intervention",
            patient_id=obs.patient_id,
            value="supplemental_oxygen",
        )
        _, reward, done, info = self.env.step(action)
        assert reward >= 0.0  # should be non-negative for neutral/safe
        assert not done

    def test_contraindicated_gives_negative(self):
        # Create patient with AKI (creatinine >= 2.0)
        gen = PatientGenerator(seed=10)
        # Repeatedly generate until we get one with high creatinine
        for seed in range(100):
            env = ClinicalTriageEnv(seed=seed)
            obs = env.reset(seed=seed)
            labs = obs.labs or {}
            hist = " ".join(obs.history).lower()
            if labs.get("creatinine", 0) >= 2.0 or "ckd" in hist or "renal_failure" in hist:
                action = TriageAction(
                    action_type="order_intervention",
                    patient_id=obs.patient_id,
                    value="nsaids",
                )
                _, reward, _, info = env.step(action)
                assert reward < 0.0, f"Expected negative reward for contraindicated NSAID in AKI"
                assert len(info["safety_violations"]) > 0
                return
        pytest.skip("No AKI patient generated in first 100 seeds")

    def test_set_disposition_ends_episode(self):
        obs = self.env.reset(seed=42)
        # Assign level first
        self.env.step(TriageAction(
            action_type="assign_triage_level",
            patient_id=obs.patient_id,
            value="3",
        ))
        _, _, done, _ = self.env.step(TriageAction(
            action_type="set_disposition",
            patient_id=obs.patient_id,
            value="admit_ward",
        ))
        assert done is True

    def test_state_returns_dict(self):
        self.env.reset(seed=42)
        state = self.env.state()
        assert isinstance(state, dict)
        assert "episode_id" in state
        assert "step_count" in state
        assert "current_patient" in state

    def test_step_after_done_raises(self):
        obs = self.env.reset(seed=42)
        self.env.step(TriageAction(
            action_type="set_disposition",
            patient_id=obs.patient_id,
            value="discharge",
        ))
        with pytest.raises(RuntimeError):
            self.env.step(TriageAction(
                action_type="reassess",
                patient_id=obs.patient_id,
            ))

    def test_reset_clears_state(self):
        obs1 = self.env.reset(seed=1)
        self.env.step(TriageAction(
            action_type="assign_triage_level",
            patient_id=obs1.patient_id,
            value="2",
        ))
        obs2 = self.env.reset(seed=2)
        state = self.env.state()
        assert state["step_count"] == 0
        assert state["assigned_triage_level"] is None

    def test_critical_discharge_penalty(self):
        """Discharging an ESI-1 patient should give large negative reward."""
        # Find an ESI-1 patient
        for seed in range(50):
            env = ClinicalTriageEnv(seed=seed)
            obs = env.reset(seed=seed)
            lvl = expected_triage_level(
                obs.vitals, obs.labs, obs.history, obs.symptoms
            )
            if lvl == 1:
                _, reward, _, info = env.step(TriageAction(
                    action_type="set_disposition",
                    patient_id=obs.patient_id,
                    value="discharge",
                ))
                assert reward <= 0.0
                assert any("critical_discharge" in v or "unsafe_discharge" in v
                           for v in info["safety_violations"])
                return
        pytest.skip("No ESI-1 patient in first 50 seeds")


# ── Grader tests ────────────────────────────────────────────────────────────

class TestGraders:
    def test_grade_task1_exact(self):
        episode = build_task1_episode(seed=42)
        patient = episode["patient"]
        expected = episode["expected_triage_level"]
        score = grade_task1(patient, expected, seed=42)
        assert score == 1.0

    def test_grade_task1_off_by_one(self):
        episode = build_task1_episode(seed=42)
        patient = episode["patient"]
        expected = episode["expected_triage_level"]
        off = expected + 1 if expected < 5 else expected - 1
        score = grade_task1(patient, off, seed=42)
        assert score == 0.5

    def test_grade_task1_wrong(self):
        episode = build_task1_episode(seed=42)
        patient = episode["patient"]
        expected = episode["expected_triage_level"]
        wrong = 5 if expected < 3 else 1
        if abs(wrong - expected) > 1:
            score = grade_task1(patient, wrong, seed=42)
            assert score == 0.0

    def test_grade_task2_perfect(self):
        episode = build_task2_episode(seed=42)
        patients = episode["patients"]
        expected_levels = episode["expected_triage_levels"]
        # Perfect interventions from evidence base
        from env.evidence_base import CLINICAL_RULES
        interventions = []
        for p in patients:
            from env.evidence_base import get_triggered_rules
            triggered = get_triggered_rules(p.vitals, p.labs, p.history, p.symptoms)
            iv = None
            if triggered:
                rule = CLINICAL_RULES[triggered[0]]
                safe = rule.get("safe_interventions", [])
                if safe:
                    iv = safe[0]
            interventions.append(iv or "iv_access")

        score = grade_task2(patients, expected_levels, interventions, seed=42)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # should be decent with perfect triage

    def test_grade_task2_range(self):
        episode = build_task2_episode(seed=42)
        patients = episode["patients"]
        score = grade_task2(
            patients,
            [3, 3, 3, 3, 3],  # mediocre triage
            [None, None, None, None, None],
            seed=42,
        )
        assert 0.0 <= score <= 1.0

    def test_grade_task2_wrong_count_raises(self):
        episode = build_task2_episode(seed=42)
        with pytest.raises(ValueError):
            grade_task2(episode["patients"], [1, 2], [None, None], seed=42)

    def test_grade_task3_perfect(self):
        episode = build_task3_episode(seed=42)
        patients = episode["patients"]
        expected_levels = episode["expected_triage_levels"]
        expected_dispositions = episode["expected_dispositions"]
        ivs = [["iv_access"] for _ in patients]
        score = grade_task3(
            patients,
            expected_levels,
            expected_dispositions,
            ivs,
            16,  # under 20 steps
            [],  # no violations
            seed=42,
        )
        assert score >= 0.8  # triage + disp both perfect

    def test_grade_task3_safety_penalty(self):
        episode = build_task3_episode(seed=42)
        patients = episode["patients"]
        expected_levels = episode["expected_triage_levels"]
        expected_dispositions = episode["expected_dispositions"]
        ivs = [["nsaids"] for _ in patients]
        score_no_violation = grade_task3(
            patients, expected_levels, expected_dispositions, ivs, 16, [], seed=42
        )
        score_with_violation = grade_task3(
            patients, expected_levels, expected_dispositions, ivs, 16,
            ["contraindicated:nsaids", "unsafe_discharge:ESI1"],
            seed=42,
        )
        assert score_with_violation < score_no_violation

    def test_grade_task3_efficiency_penalty(self):
        episode = build_task3_episode(seed=42)
        patients = episode["patients"]
        expected_levels = episode["expected_triage_levels"]
        expected_dispositions = episode["expected_dispositions"]
        ivs = [["iv_access"] for _ in patients]
        score_efficient = grade_task3(
            patients, expected_levels, expected_dispositions, ivs, 18, [], seed=42
        )
        score_inefficient = grade_task3(
            patients, expected_levels, expected_dispositions, ivs, 40, [], seed=42
        )
        assert score_efficient >= score_inefficient

    def test_grade_task3_deterministic(self):
        episode = build_task3_episode(seed=99)
        patients = episode["patients"]
        expected_levels = episode["expected_triage_levels"]
        expected_dispositions = episode["expected_dispositions"]
        ivs = [["iv_access"] for _ in patients]
        s1 = grade_task3(patients, expected_levels, expected_dispositions, ivs, 16, [], seed=99)
        s2 = grade_task3(patients, expected_levels, expected_dispositions, ivs, 16, [], seed=99)
        assert s1 == s2

    def test_all_scores_in_range(self):
        for seed in range(5):
            episode1 = build_task1_episode(seed=seed)
            s1 = grade_task1(episode1["patient"], episode1["expected_triage_level"], seed=seed)
            assert 0.0 <= s1 <= 1.0

            episode2 = build_task2_episode(seed=seed)
            s2 = grade_task2(
                episode2["patients"],
                episode2["expected_triage_levels"],
                [None] * 5,
                seed=seed,
            )
            assert 0.0 <= s2 <= 1.0

            episode3 = build_task3_episode(seed=seed)
            s3 = grade_task3(
                episode3["patients"],
                episode3["expected_triage_levels"],
                episode3["expected_dispositions"],
                [["iv_access"]] * 8,
                16,
                [],
                seed=seed,
            )
            assert 0.0 <= s3 <= 1.0


# ── Model schema tests ──────────────────────────────────────────────────────

class TestModels:
    def test_patient_observation_schema(self):
        p = PatientObservation(
            patient_id="PT-0001",
            chief_complaint="Chest pain",
            vitals={"hr": 110, "sbp": 88, "dbp": 58, "spo2": 93, "rr": 22, "temp_c": 37.2, "gcs": 15},
            symptoms=["chest pain", "diaphoresis"],
            history=["hypertension"],
        )
        assert p.patient_id == "PT-0001"
        assert p.labs is None
        assert p.current_triage_level is None

    def test_triage_action_schema(self):
        a = TriageAction(
            action_type="assign_triage_level",
            patient_id="PT-0001",
            value="2",
        )
        assert a.action_type == "assign_triage_level"

    def test_triage_reward_clipped(self):
        import pytest
        with pytest.raises(Exception):
            TriageReward(score=1.5, breakdown={})  # should fail validation

    def test_triage_reward_valid(self):
        r = TriageReward(score=0.75, breakdown={"triage": 0.3, "intervention": 0.2, "disposition": 0.25})
        assert r.score == 0.75

    def test_model_json_roundtrip(self):
        p = PatientObservation(
            patient_id="PT-0002",
            chief_complaint="Shortness of breath",
            vitals={"hr": 100, "sbp": 120, "dbp": 78, "spo2": 92, "rr": 24, "temp_c": 38.1, "gcs": 15},
            symptoms=["dyspnea"],
            history=["asthma"],
            labs={"creatinine": 1.2},
        )
        json_str = p.model_dump_json()
        p2 = PatientObservation.model_validate_json(json_str)
        assert p2.patient_id == p.patient_id
        assert p2.labs == p.labs
