"""
adapters.py — Model inference adapters.
Each adapter loads its model once and converts raw output to a Finding.
Image models require PyTorch + torchvision (requirements-images.txt).
The diabetes adapter requires only scikit-learn / joblib (requirements.txt).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    DIABETES_MODEL_PATH,
    DIABETES_PHYSIO_COLS,
    DR_CLASS_NAMES,
    DR_IMG_SIZE,
    DR_MODEL_PATH,
    DR_REFERABLE_GRADE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SKIN_CLASS_NAMES,
    SKIN_IMG_SIZE,
    SKIN_MALIGNANT_CLASSES,
    SKIN_MALIGNANT_THRESHOLD,
    SKIN_MODEL_PATH,
    DIABETES_ALERT_THRESHOLD,
)
from schema import Finding, FindingType, Severity

log = logging.getLogger(__name__)


def _try_import_torch():
    try:
        import torch
        import torchvision.models as tvm
        from torchvision import transforms
        from PIL import Image
        return torch, tvm, transforms, Image
    except ImportError:
        return None, None, None, None


def _severity_from_proba(proba: float, thresholds: Tuple[float, float, float, float]) -> Severity:
    low, mod, high, crit = thresholds
    if proba < low:
        return Severity.NORMAL
    if proba < mod:
        return Severity.LOW
    if proba < high:
        return Severity.MODERATE
    if proba < crit:
        return Severity.HIGH
    return Severity.CRITICAL


def _build_efficientnet_b0(num_classes: int, torch_mod, tvm_mod):
    import torch.nn as nn
    weights = tvm_mod.EfficientNet_B0_Weights.DEFAULT
    model = tvm_mod.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(256, num_classes),
    )
    return model


def _load_image(path: str, size: Tuple[int, int], transforms_mod, pil_mod) -> Any:
    eval_transforms = transforms_mod.Compose([
        transforms_mod.Resize(size),
        transforms_mod.ToTensor(),
        transforms_mod.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = pil_mod.open(path).convert("RGB")
    return eval_transforms(img).unsqueeze(0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _extract_state_dict(ckpt: Any) -> Any:
    """Safely extracts state dict whether saved as a full dict or model_state_dict."""
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


class DiabetesAdapter:
    """
    Loads the joblib pipeline saved by pima_fixed_training.py (or the notebook).
    Accepts a dict of raw PIMA feature values and returns a Finding.
    """

    def __init__(self) -> None:
        self._artifact: Optional[Dict[str, Any]] = None

    def _load(self) -> bool:
        if self._artifact is not None:
            return True
        try:
            import joblib
            model_path = DIABETES_MODEL_PATH
            if not model_path.exists():
                fallback = model_path.parent / "diabetes_pipeline.pkl"
                if fallback.exists():
                    model_path = fallback
            self._artifact = joblib.load(model_path)
            log.info("Diabetes model loaded from %s: %s", model_path.name, self._artifact.get("champion_name", "unknown"))
            return True
        except Exception as exc:
            log.warning("Diabetes model not available: %s", exc)
            return False

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        d = X.copy()
        d["HOMA_IR"] = (d["Glucose"] * d["Insulin"]) / 405.0
        d["BMI_Cat"] = pd.cut(
            d["BMI"], bins=[0, 18.5, 24.9, 29.9, 34.9, 100], labels=[0, 1, 2, 3, 4]
        ).astype(float)
        d["Glucose_Cat"] = pd.cut(
            d["Glucose"], bins=[0, 99, 139, 199, 300], labels=[0, 1, 2, 3]
        ).astype(float)
        d["Age_Cat"] = pd.cut(
            d["Age"], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]
        ).astype(float)
        d["Metabolic_Score"] = (
            (d["Glucose"] >= 140).astype(int)
            + (d["BMI"] >= 30).astype(int)
            + (d["BloodPressure"] >= 80).astype(int)
            + (d["Age"] >= 35).astype(int)
        )
        d["Insulin_Glucose_Ratio"] = d["Insulin"] / (d["Glucose"] + 1e-5)
        d["Glucose_BMI"] = d["Glucose"] * d["BMI"]
        d["Age_BMI"] = d["Age"] * d["BMI"]
        d["DPF_Glucose"] = d["DiabetesPedigreeFunction"] * d["Glucose"]
        d["Preg_Age_Ratio"] = d["Pregnancies"] / (d["Age"] + 1.0)
        d["Insulin_Log"] = np.log1p(d["Insulin"])
        d["Glucose_Log"] = np.log1p(d["Glucose"])
        return d

    def predict(self, raw_features: Dict[str, float]) -> Optional[Finding]:
        if not self._load():
            return None

        artifact = self._artifact
        train_medians: Dict[str, float] = artifact["train_medians"]
        physio_cols: List[str] = artifact["physio_cols"]
        feature_list: List[str] = artifact["feature_list"]
        scaler = artifact["scaler"]
        model = artifact["champion_model"]

        row = pd.DataFrame([raw_features])
        row[physio_cols] = row[physio_cols].replace(0, np.nan)
        for col in physio_cols:
            if col in train_medians:
                row[col] = row[col].fillna(train_medians[col])

        row_eng = self._engineer_features(row)
        row_eng = row_eng.reindex(columns=feature_list, fill_value=0.0)
        row_scaled = scaler.transform(row_eng)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row_scaled)[0, 1]
        else:
            proba = float(model.predict(row_scaled)[0])

        proba = float(proba)
        label = "Diabetic" if proba >= DIABETES_ALERT_THRESHOLD else "Non-Diabetic"
        alert = bool(proba >= DIABETES_ALERT_THRESHOLD)
        severity = _severity_from_proba(proba, (0.20, 0.40, 0.60, 0.80))

        glucose_val = raw_features.get("Glucose", 0.0)
        notes = (
            f"Diabetes risk probability: {proba:.1%}. "
            f"Input glucose: {glucose_val:.0f} mg/dL. "
            "Note: result is a screening estimate; confirm with fasting glucose/HbA1c."
        )

        return Finding(
            type=FindingType.DIABETES_RISK,
            label=label,
            confidence=proba,
            severity=severity,
            raw_probabilities={"Non-Diabetic": 1.0 - proba, "Diabetic": proba},
            alert=alert,
            notes=notes,
            metadata={"model": artifact.get("champion_name", "unknown"), "glucose": glucose_val},
        )


class DiabeticRetinopathyAdapter:
    """
    Loads best_dr_model.pth (EfficientNet-B0, 5-class DR grading).
    Accepts a fundus image path and returns a Finding.
    """

    NUM_CLASSES = 5
    CLASS_NAMES = DR_CLASS_NAMES

    def __init__(self) -> None:
        self._model = None
        self._device = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        torch, tvm, _, _ = _try_import_torch()
        if torch is None:
            log.warning("PyTorch not available; DR model cannot run.")
            return False
        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = _build_efficientnet_b0(self.NUM_CLASSES, torch, tvm)
            ckpt = torch.load(DR_MODEL_PATH, map_location=self._device, weights_only=False)
            model.load_state_dict(_extract_state_dict(ckpt))
            model.to(self._device)
            model.eval()
            self._model = model
            val_qwk = ckpt.get("val_qwk", 0) if isinstance(ckpt, dict) else 0
            epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
            log.info("DR model loaded (best val QWK=%.4f, epoch=%s)", val_qwk, epoch)
            return True
        except Exception as exc:
            log.warning("DR model not available: %s", exc)
            return False

    def predict(self, image_path: str) -> Optional[Finding]:
        if not os.path.exists(image_path):
            log.error("DR image not found: %s", image_path)
            return None
        if not self._load():
            return None

        torch, _, transforms_mod, pil_mod = _try_import_torch()
        import torch.nn as nn

        tensor = _load_image(image_path, DR_IMG_SIZE, transforms_mod, pil_mod).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = nn.Softmax(dim=1)(logits).cpu().numpy()[0]

        grade = int(np.argmax(probs))
        confidence = float(probs[grade])
        label = self.CLASS_NAMES[grade]
        referrable = grade >= DR_REFERABLE_GRADE
        alert = grade >= 3

        sev_map = {0: Severity.NORMAL, 1: Severity.LOW, 2: Severity.MODERATE, 3: Severity.HIGH, 4: Severity.CRITICAL}
        severity = sev_map[grade]

        notes = (
            f"DR Grade {grade} ({label}). "
            f"Referrable: {'YES' if referrable else 'no'}. "
            "Softmax scores are ranking estimates, not calibrated probabilities. "
            "Clinical grading by ophthalmologist required."
        )

        return Finding(
            type=FindingType.DIABETIC_RETINOPATHY,
            label=label,
            confidence=confidence,
            severity=severity,
            raw_probabilities={self.CLASS_NAMES[i]: float(probs[i]) for i in range(self.NUM_CLASSES)},
            alert=alert,
            notes=notes,
            metadata={"grade": grade, "referrable": referrable},
        )


class SkinDiseaseAdapter:
    """
    Loads best_skin_model.pth (EfficientNet-B0, 9-class skin disease classification).
    Accepts a lesion image path and returns a Finding.
    """

    NUM_CLASSES = 9
    CLASS_NAMES = SKIN_CLASS_NAMES

    def __init__(self) -> None:
        self._model = None
        self._device = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        torch, tvm, _, _ = _try_import_torch()
        if torch is None:
            log.warning("PyTorch not available; Skin model cannot run.")
            return False
        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = _build_efficientnet_b0(self.NUM_CLASSES, torch, tvm)
            ckpt = torch.load(SKIN_MODEL_PATH, map_location=self._device, weights_only=False)
            model.load_state_dict(_extract_state_dict(ckpt))
            model.to(self._device)
            model.eval()
            self._model = model
            epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
            log.info("Skin model loaded (epoch=%s)", epoch)
            return True
        except Exception as exc:
            log.warning("Skin model not available: %s", exc)
            return False

    def predict(self, image_path: str) -> Optional[Finding]:
        if not os.path.exists(image_path):
            log.error("Skin image not found: %s", image_path)
            return None
        if not self._load():
            return None

        torch, _, transforms_mod, pil_mod = _try_import_torch()
        import torch.nn as nn

        tensor = _load_image(image_path, SKIN_IMG_SIZE, transforms_mod, pil_mod).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = nn.Softmax(dim=1)(logits).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        label = self.CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        is_malignant = label in SKIN_MALIGNANT_CLASSES
        malignant_prob = sum(float(probs[i]) for i, n in enumerate(self.CLASS_NAMES) if n in SKIN_MALIGNANT_CLASSES)
        alert = is_malignant and confidence >= SKIN_MALIGNANT_THRESHOLD

        if is_malignant:
            severity = Severity.CRITICAL if label == "Melanoma" else Severity.HIGH
        elif label == "Actinic keratosis":
            severity = Severity.MODERATE
        else:
            severity = Severity.LOW if label != "Normal" else Severity.NORMAL

        notes = (
            f"Predicted: {label} (conf={confidence:.1%}). "
            f"Combined malignant/premalignant probability: {malignant_prob:.1%}. "
            "PPV in this output assumes a balanced test set; real-world PPV is lower at population prevalence. "
            "Dermatologist review required for any lesion of concern."
        )

        return Finding(
            type=FindingType.SKIN_LESION,
            label=label,
            confidence=confidence,
            severity=severity,
            raw_probabilities={self.CLASS_NAMES[i]: float(probs[i]) for i in range(self.NUM_CLASSES)},
            alert=alert,
            notes=notes,
            metadata={"is_malignant": is_malignant, "malignant_prob": malignant_prob},
        )


_diabetes_adapter: Optional[DiabetesAdapter] = None
_dr_adapter: Optional[DiabeticRetinopathyAdapter] = None
_skin_adapter: Optional[SkinDiseaseAdapter] = None


def get_diabetes_adapter() -> DiabetesAdapter:
    global _diabetes_adapter
    if _diabetes_adapter is None:
        _diabetes_adapter = DiabetesAdapter()
    return _diabetes_adapter


def get_dr_adapter() -> DiabeticRetinopathyAdapter:
    global _dr_adapter
    if _dr_adapter is None:
        _dr_adapter = DiabeticRetinopathyAdapter()
    return _dr_adapter


def get_skin_adapter() -> SkinDiseaseAdapter:
    global _skin_adapter
    if _skin_adapter is None:
        _skin_adapter = SkinDiseaseAdapter()
    return _skin_adapter

