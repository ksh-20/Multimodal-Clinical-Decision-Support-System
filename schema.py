"""
schema.py — Data contracts for the knowledge layer.
All inter-module data flows through these dataclasses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class FindingType(str, Enum):
    DIABETES_RISK = "diabetes_risk"
    DIABETIC_RETINOPATHY = "diabetic_retinopathy"
    SKIN_LESION = "skin_lesion"


class Severity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Finding:
    type: FindingType
    label: str
    confidence: float
    severity: Severity
    raw_probabilities: Dict[str, float]
    alert: bool
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["severity"] = self.severity.value
        return d

    def summary_line(self) -> str:
        alert_tag = " ⚠ ALERT" if self.alert else ""
        return (
            f"[{self.type.value.upper()}]{alert_tag} "
            f"{self.label} (conf={self.confidence:.1%}, severity={self.severity.value})"
        )


@dataclass
class PatientContext:
    age: Optional[int] = None
    sex: Optional[str] = None
    clinical_notes: Optional[str] = None
    known_conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineInput:
    patient: PatientContext
    tabular_data: Optional[Dict[str, float]] = None
    eye_image_path: Optional[str] = None
    skin_image_path: Optional[str] = None

    @classmethod
    def from_json(cls, path: str) -> "PipelineInput":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        patient_raw = raw.get("patient", {})
        patient = PatientContext(
            age=patient_raw.get("age"),
            sex=patient_raw.get("sex"),
            clinical_notes=patient_raw.get("clinical_notes"),
            known_conditions=patient_raw.get("known_conditions", []),
            medications=patient_raw.get("medications", []),
        )
        return cls(
            patient=patient,
            tabular_data=raw.get("tabular_data"),
            eye_image_path=raw.get("eye_image_path"),
            skin_image_path=raw.get("skin_image_path"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient": self.patient.to_dict(),
            "tabular_data": self.tabular_data,
            "eye_image_path": self.eye_image_path,
            "skin_image_path": self.skin_image_path,
        }


@dataclass
class PipelineResult:
    findings: List[Finding]
    knowledge_graph_context: Dict[str, Any]
    reasoning: str
    clinical_summary: str
    alerts: List[str]
    recommendations: List[str]
    confidence_level: str
    processing_time_seconds: float
    timestamp: str
    model_versions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["findings"] = [f.to_dict() for f in self.findings]
        return d

    def to_json(self, indent: int = 2) -> str:
        def _json_default(o):
            if hasattr(o, "item"):
                return o.item()
            if isinstance(o, (np.bool_, bool)):
                return bool(o)
            if isinstance(o, (np.floating, float)):
                return float(o)
            if isinstance(o, (np.integer, int)):
                return int(o)
            return str(o)
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=_json_default)

    def print_report(self) -> None:
        import sys
        out = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False)
        border = "=" * 70
        out.write(border + "\n")
        out.write("  AURA-MED MULTIMODAL CLINICAL REPORT\n")
        out.write(f"  Generated: {self.timestamp}\n")
        out.write(border + "\n")

        out.write("\n[FINDINGS]\n" + "-" * 40 + "\n")
        if self.findings:
            for f in self.findings:
                out.write(f"  {f.summary_line()}\n")
                if f.notes:
                    out.write(f"    -> {f.notes}\n")
        else:
            out.write("  No findings (no models ran or no inputs provided).\n")

        if self.alerts:
            out.write("\n[!] ALERTS\n" + "-" * 40 + "\n")
            for a in self.alerts:
                out.write(f"  * {a}\n")

        out.write("\n[RECOMMENDATIONS]\n" + "-" * 40 + "\n")
        for r in self.recommendations:
            out.write(f"  * {r}\n")

        out.write("\n[REASONING]\n" + "-" * 40 + "\n")
        out.write(self.reasoning + "\n")

        out.write("\n[CLINICAL SUMMARY]\n" + "-" * 40 + "\n")
        out.write(self.clinical_summary + "\n")

        out.write("\n" + border + "\n")
        out.write(f"  Confidence level: {self.confidence_level}\n")
        out.write(f"  Processing time:  {self.processing_time_seconds:.2f}s\n")
        out.write(border + "\n\n")
        out.flush()
