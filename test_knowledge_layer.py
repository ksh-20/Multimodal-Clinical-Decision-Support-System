"""
test_knowledge_layer.py — Comprehensive test suite for the knowledge layer.

Tests are organised so they run from fastest/no-deps to slowest/GPU-dependent:
  1. Schema serialization & contract tests
  2. Config sanity checks
  3. KG loading and query tests
  4. Reasoner offline mode tests
  5. Diabetes adapter unit test (requires diabetes_model.joblib)
  6. Image adapter unit tests (require PyTorch + model .pth files + test images)
  7. Full integration pipeline test (requires all models)

Run all tests:
  python test_knowledge_layer.py

Run specific group (via pattern matching):
  python test_knowledge_layer.py TestSchema
  python test_knowledge_layer.py TestKG
  python test_knowledge_layer.py TestReasoner
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))


class TestSchema(unittest.TestCase):
    def setUp(self):
        from schema import Finding, FindingType, PatientContext, PipelineInput, PipelineResult, Severity
        self.Finding = Finding
        self.FindingType = FindingType
        self.PatientContext = PatientContext
        self.PipelineInput = PipelineInput
        self.PipelineResult = PipelineResult
        self.Severity = Severity

    def test_finding_to_dict(self):
        f = self.Finding(
            type=self.FindingType.DIABETES_RISK,
            label="Diabetic",
            confidence=0.78,
            severity=self.Severity.HIGH,
            raw_probabilities={"Non-Diabetic": 0.22, "Diabetic": 0.78},
            alert=True,
            notes="Test note",
        )
        d = f.to_dict()
        self.assertEqual(d["type"], "diabetes_risk")
        self.assertEqual(d["severity"], "high")
        self.assertTrue(d["alert"])
        self.assertAlmostEqual(d["confidence"], 0.78)

    def test_finding_summary_line(self):
        f = self.Finding(
            type=self.FindingType.DIABETIC_RETINOPATHY,
            label="Moderate",
            confidence=0.65,
            severity=self.Severity.MODERATE,
            raw_probabilities={},
            alert=False,
        )
        line = f.summary_line()
        self.assertIn("DIABETIC_RETINOPATHY", line)
        self.assertIn("Moderate", line)
        self.assertNotIn("ALERT", line)

    def test_alert_finding_summary(self):
        f = self.Finding(
            type=self.FindingType.SKIN_LESION,
            label="Melanoma",
            confidence=0.82,
            severity=self.Severity.CRITICAL,
            raw_probabilities={},
            alert=True,
        )
        self.assertIn("ALERT", f.summary_line())

    def test_patient_context_serialization(self):
        from schema import PatientContext
        p = PatientContext(
            age=55,
            sex="female",
            known_conditions=["hypertension"],
            medications=["metformin"],
        )
        d = p.to_dict()
        self.assertEqual(d["age"], 55)
        self.assertIn("hypertension", d["known_conditions"])

    def test_pipeline_input_from_json(self):
        data = {
            "patient": {
                "age": 45,
                "sex": "male",
                "known_conditions": [],
                "medications": [],
            },
            "tabular_data": {"Glucose": 148, "BMI": 33.6, "Age": 45, "Pregnancies": 2,
                             "BloodPressure": 72, "SkinThickness": 0, "Insulin": 0, "DiabetesPedigreeFunction": 0.5},
            "eye_image_path": None,
            "skin_image_path": None,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(data, fh)
            tmp = fh.name
        try:
            pi = self.PipelineInput.from_json(tmp)
            self.assertEqual(pi.patient.age, 45)
            self.assertEqual(pi.tabular_data["Glucose"], 148)
        finally:
            os.unlink(tmp)

    def test_pipeline_result_to_json(self):
        from schema import Finding, FindingType, PipelineResult, Severity
        result = PipelineResult(
            findings=[],
            knowledge_graph_context={},
            reasoning="Test reasoning.",
            clinical_summary="Test summary.",
            alerts=[],
            recommendations=["Do something."],
            confidence_level="moderate",
            processing_time_seconds=0.5,
            timestamp="2026-01-01T00:00:00Z",
        )
        j = result.to_json()
        parsed = json.loads(j)
        self.assertIn("recommendations", parsed)


class TestConfig(unittest.TestCase):
    def test_config_imports(self):
        import config
        self.assertTrue(config.BASE_DIR.exists())
        self.assertIsInstance(config.DIABETES_ALERT_THRESHOLD, float)
        self.assertIsInstance(config.GEMINI_MODEL, str)
        self.assertIn("gemini", config.GEMINI_MODEL.lower())

    def test_path_types(self):
        from pathlib import Path
        import config
        for attr in ["MODELS_DIR", "DATA_DIR", "SAMPLES_DIR", "OUTPUTS_DIR", "KG_JSON_PATH"]:
            self.assertIsInstance(getattr(config, attr), Path, f"{attr} should be Path")


class TestKG(unittest.TestCase):
    def setUp(self):
        from medical_kg import MedicalKG
        self.kg = MedicalKG()

    def test_kg_loads(self):
        self.assertGreater(self.kg.G.number_of_nodes(), 20)
        self.assertGreater(self.kg.G.number_of_edges(), 30)

    def test_dr_context_grade0(self):
        ctx = self.kg.dr_context(0)
        self.assertEqual(ctx["grade"], 0)
        self.assertFalse(ctx["referrable"])
        self.assertEqual(ctx["monitoring_interval_months"], 12)

    def test_dr_context_grade4(self):
        ctx = self.kg.dr_context(4)
        self.assertTrue(ctx["referrable"])
        self.assertEqual(ctx["monitoring_interval_months"], 1)
        self.assertTrue(len(ctx["recommendations"]) > 0)

    def test_skin_context_melanoma(self):
        ctx = self.kg.skin_context("Melanoma")
        self.assertEqual(ctx["malignant_potential"], "malignant")
        self.assertTrue(len(ctx["recommendations"]) > 0)

    def test_skin_context_benign(self):
        ctx = self.kg.skin_context("Atopic Dermatitis")
        self.assertEqual(ctx["malignant_potential"], "benign")

    def test_diabetes_context_positive(self):
        ctx = self.kg.diabetes_context(True, glucose=180)
        self.assertTrue(ctx["positive"])
        self.assertGreater(len(ctx["recommendations"]), 0)
        self.assertGreater(len(ctx["complications_to_watch"]), 0)

    def test_diabetes_context_negative(self):
        ctx = self.kg.diabetes_context(False)
        self.assertFalse(ctx["positive"])
        self.assertEqual(ctx["recommendations"], [])

    def test_applicable_rules_melanoma(self):
        rules = self.kg.applicable_rules(["skin_lesion"], skin_label="Melanoma")
        self.assertTrue(any("melanoma" in r["id"] for r in rules))

    def test_applicable_rules_grade4(self):
        rules = self.kg.applicable_rules(["diabetic_retinopathy"], dr_grade=4)
        self.assertTrue(any("proliferative" in r["id"] for r in rules))

    def test_context_for_findings_full(self):
        ctx = self.kg.context_for_findings(
            finding_types=["diabetes_risk", "diabetic_retinopathy", "skin_lesion"],
            skin_label="Melanoma",
            dr_grade=3,
            diabetes_positive=True,
            glucose=180.0,
            known_conditions=["hypertension"],
        )
        self.assertIn("dr", ctx)
        self.assertIn("skin", ctx)
        self.assertIn("diabetes", ctx)
        self.assertIn("hypertension_note", ctx)


class TestReasoner(unittest.TestCase):
    def _make_findings(self):
        from schema import Finding, FindingType, Severity
        return [
            Finding(
                type=FindingType.DIABETES_RISK,
                label="Diabetic",
                confidence=0.75,
                severity=Severity.HIGH,
                raw_probabilities={"Non-Diabetic": 0.25, "Diabetic": 0.75},
                alert=True,
                metadata={"glucose": 160.0},
            ),
            Finding(
                type=FindingType.DIABETIC_RETINOPATHY,
                label="Moderate",
                confidence=0.62,
                severity=Severity.MODERATE,
                raw_probabilities={},
                alert=False,
                metadata={"grade": 2, "referrable": True},
            ),
        ]

    def test_offline_reasoner_returns_required_keys(self):
        from reasoner import _offline_reason
        from schema import PatientContext
        patient = PatientContext(known_conditions=["hypertension"])
        findings = self._make_findings()
        kg_ctx = {"dr": {"recommendations": ["Ophthalmology referral"]}, "applicable_rules": []}
        result = _offline_reason(patient, findings, kg_ctx)
        for key in ("reasoning", "clinical_summary", "alerts", "recommendations", "confidence_level"):
            self.assertIn(key, result)

    def test_offline_alerts_generated(self):
        from reasoner import _offline_reason
        from schema import PatientContext
        patient = PatientContext()
        findings = self._make_findings()
        result = _offline_reason(patient, findings, {"applicable_rules": []})
        self.assertTrue(len(result["alerts"]) > 0)

    def test_offline_empty_findings(self):
        from reasoner import _offline_reason
        from schema import PatientContext
        patient = PatientContext()
        result = _offline_reason(patient, [], {})
        self.assertEqual(result["confidence_level"], "low")
        self.assertEqual(result["alerts"], [])

    def test_reason_falls_back_offline_no_key(self):
        with patch("reasoner.GEMINI_API_KEY", ""):
            from reasoner import reason
            from schema import PatientContext
            patient = PatientContext()
            result = reason(patient, [], {})
            self.assertIn("reasoning", result)

    def test_reason_offline_no_key(self):
        with patch("reasoner.GEMINI_API_KEY", ""):
            from reasoner import reason
            from schema import PatientContext
            patient = PatientContext()
            result = reason(patient, self._make_findings(), {"applicable_rules": []})
            self.assertIn("recommendations", result)


class TestDiabetesAdapter(unittest.TestCase):
    def setUp(self):
        from config import DIABETES_MODEL_PATH
        if not DIABETES_MODEL_PATH.exists():
            self.skipTest(f"diabetes_model.joblib not found at {DIABETES_MODEL_PATH}. Run pima_fixed_training.py first.")

    def test_adapter_loads(self):
        from adapters import DiabetesAdapter
        adapter = DiabetesAdapter()
        self.assertTrue(adapter._load())
        self.assertIsNotNone(adapter._artifact)

    def test_positive_prediction(self):
        from adapters import DiabetesAdapter
        from schema import FindingType
        adapter = DiabetesAdapter()
        high_risk = {
            "Pregnancies": 6, "Glucose": 180, "BloodPressure": 82,
            "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.8, "Age": 55,
        }
        finding = adapter.predict(high_risk)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.type, FindingType.DIABETES_RISK)
        self.assertGreater(finding.confidence, 0.0)
        self.assertIn(finding.label, ["Diabetic", "Non-Diabetic"])

    def test_low_risk_prediction(self):
        from adapters import DiabetesAdapter
        adapter = DiabetesAdapter()
        low_risk = {
            "Pregnancies": 1, "Glucose": 89, "BloodPressure": 66,
            "SkinThickness": 23, "Insulin": 94, "BMI": 22.0,
            "DiabetesPedigreeFunction": 0.17, "Age": 21,
        }
        finding = adapter.predict(low_risk)
        self.assertIsNotNone(finding)
        self.assertLess(finding.confidence, 0.6)

    def test_missing_insulin_imputed(self):
        from adapters import DiabetesAdapter
        adapter = DiabetesAdapter()
        row_zero_insulin = {
            "Pregnancies": 3, "Glucose": 148, "BloodPressure": 72,
            "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627, "Age": 50,
        }
        finding = adapter.predict(row_zero_insulin)
        self.assertIsNotNone(finding)

    def test_artifact_has_required_keys(self):
        from adapters import DiabetesAdapter
        adapter = DiabetesAdapter()
        adapter._load()
        for key in ("scaler", "champion_model", "feature_list", "physio_cols", "train_medians"):
            self.assertIn(key, adapter._artifact)


class TestImageAdapters(unittest.TestCase):
    def _check_torch(self):
        try:
            import torch
            import torchvision
        except ImportError:
            self.skipTest("PyTorch not installed.")

    def _make_dummy_image(self, size=(224, 224), mode="RGB") -> str:
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            self.skipTest("Pillow not installed.")
        img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8), mode)
        fh = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(fh.name)
        return fh.name

    def test_dr_adapter_missing_model(self):
        self._check_torch()
        from adapters import DiabeticRetinopathyAdapter
        adapter = DiabeticRetinopathyAdapter()
        from config import DR_MODEL_PATH
        if not DR_MODEL_PATH.exists():
            self.assertFalse(adapter._load())
        else:
            self.assertTrue(adapter._load())

    def test_dr_adapter_missing_image(self):
        from adapters import DiabeticRetinopathyAdapter
        adapter = DiabeticRetinopathyAdapter()
        result = adapter.predict("/nonexistent/path/eye.png")
        self.assertIsNone(result)

    def test_skin_adapter_missing_image(self):
        from adapters import SkinDiseaseAdapter
        adapter = SkinDiseaseAdapter()
        result = adapter.predict("/nonexistent/path/lesion.jpg")
        self.assertIsNone(result)

    def test_dr_inference_with_dummy_image(self):
        self._check_torch()
        from config import DR_MODEL_PATH
        if not DR_MODEL_PATH.exists():
            self.skipTest("DR model not found.")
        dummy = self._make_dummy_image((224, 224))
        try:
            from adapters import DiabeticRetinopathyAdapter
            from schema import FindingType
            adapter = DiabeticRetinopathyAdapter()
            finding = adapter.predict(dummy)
            self.assertIsNotNone(finding)
            self.assertEqual(finding.type, FindingType.DIABETIC_RETINOPATHY)
            self.assertIn(finding.label, ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"])
        finally:
            os.unlink(dummy)

    def test_skin_inference_with_dummy_image(self):
        self._check_torch()
        from config import SKIN_MODEL_PATH
        if not SKIN_MODEL_PATH.exists():
            self.skipTest("Skin model not found.")
        dummy = self._make_dummy_image((224, 224))
        try:
            from adapters import SkinDiseaseAdapter
            from schema import FindingType
            adapter = SkinDiseaseAdapter()
            finding = adapter.predict(dummy)
            self.assertIsNotNone(finding)
            self.assertEqual(finding.type, FindingType.SKIN_LESION)
        finally:
            os.unlink(dummy)


class TestFullPipeline(unittest.TestCase):
    def test_pipeline_tabular_only_offline(self):
        with patch("reasoner.GEMINI_API_KEY", ""):
            from run_pipeline import run_pipeline
            from schema import PatientContext, PipelineInput

            from config import DIABETES_MODEL_PATH
            if not DIABETES_MODEL_PATH.exists():
                self.skipTest("diabetes_model.joblib not found.")

            patient = PatientContext(
                age=52,
                sex="female",
                known_conditions=["hypertension"],
                medications=["metformin"],
            )
            pi = PipelineInput(
                patient=patient,
                tabular_data={
                    "Pregnancies": 4, "Glucose": 155, "BloodPressure": 78,
                    "SkinThickness": 0, "Insulin": 0, "BMI": 31.0,
                    "DiabetesPedigreeFunction": 0.65, "Age": 52,
                },
            )
            result = run_pipeline(pi, run_images=False)

            self.assertTrue(len(result.findings) >= 1)
            self.assertIsInstance(result.reasoning, str)
            self.assertIsInstance(result.recommendations, list)
            self.assertGreater(result.processing_time_seconds, 0)
            j = result.to_json()
            parsed = json.loads(j)
            self.assertIn("findings", parsed)

    def test_pipeline_no_inputs(self):
        with patch("reasoner.GEMINI_API_KEY", ""):
            from run_pipeline import run_pipeline
            from schema import PatientContext, PipelineInput
            patient = PatientContext()
            pi = PipelineInput(patient=patient)
            result = run_pipeline(pi, run_images=True)
            self.assertEqual(len(result.findings), 0)
            self.assertIsInstance(result.clinical_summary, str)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if len(sys.argv) > 1:
        pattern = sys.argv[1]
        for cls in [TestSchema, TestConfig, TestKG, TestReasoner, TestDiabetesAdapter, TestImageAdapters, TestFullPipeline]:
            if pattern.lower() in cls.__name__.lower():
                suite.addTests(loader.loadTestsFromTestCase(cls))
    else:
        for cls in [TestSchema, TestConfig, TestKG, TestReasoner, TestDiabetesAdapter, TestImageAdapters, TestFullPipeline]:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
