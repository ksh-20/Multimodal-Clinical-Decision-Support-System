"""
run_pipeline.py — Main entry point for the knowledge layer pipeline.

Usage examples:
  # From JSON file:
  python run_pipeline.py --input samples/patient_example.json

  # Inline arguments:
  python run_pipeline.py \\
      --patient-id P001 --age 55 --sex female \\
      --tabular samples/patient_example.json \\
      --eye samples/eye.png \\
      --skin samples/lesion.jpg

  # Save output:
  python run_pipeline.py --input samples/patient_example.json --save

  # Skip image models (tabular only):
  python run_pipeline.py --input samples/patient_example.json --no-images
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from adapters import (
    get_diabetes_adapter,
    get_dr_adapter,
    get_skin_adapter,
)
from config import LOG_LEVEL, OUTPUTS_DIR
from medical_kg import get_kg
from reasoner import reason
from schema import Finding, PatientContext, PipelineInput, PipelineResult

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

VERSION = "1.0.0"


def run_pipeline(
    pipeline_input: PipelineInput,
    run_images: bool = True,
) -> PipelineResult:
    t_start = time.perf_counter()
    patient = pipeline_input.patient
    findings: List[Finding] = []

    log.info("=== Pipeline start ===")

    if pipeline_input.tabular_data:
        log.info("Running diabetes adapter...")
        dm_finding = get_diabetes_adapter().predict(pipeline_input.tabular_data)
        if dm_finding:
            findings.append(dm_finding)
            log.info("  → %s", dm_finding.summary_line())
        else:
            log.warning("  → Diabetes model unavailable or returned no result.")

    if run_images:
        if pipeline_input.eye_image_path:
            log.info("Running DR adapter...")
            dr_finding = get_dr_adapter().predict(pipeline_input.eye_image_path)
            if dr_finding:
                findings.append(dr_finding)
                log.info("  → %s", dr_finding.summary_line())
            else:
                log.warning("  → DR model unavailable or image not found.")

        if pipeline_input.skin_image_path:
            log.info("Running skin adapter...")
            skin_finding = get_skin_adapter().predict(pipeline_input.skin_image_path)
            if skin_finding:
                findings.append(skin_finding)
                log.info("  → %s", skin_finding.summary_line())
            else:
                log.warning("  → Skin model unavailable or image not found.")



    finding_types = [f.type.value for f in findings]
    skin_label = next(
        (f.label for f in findings if f.type.value == "skin_lesion"), None
    )
    dr_grade = next(
        (f.metadata.get("grade") for f in findings if f.type.value == "diabetic_retinopathy"), None
    )
    diabetes_positive = next(
        (f.alert for f in findings if f.type.value == "diabetes_risk"), None
    )
    glucose_val = next(
        (f.metadata.get("glucose") for f in findings if f.type.value == "diabetes_risk"), None
    )

    log.info("Querying knowledge graph...")
    kg = get_kg()
    kg_context = kg.context_for_findings(
        finding_types=finding_types,
        skin_label=skin_label,
        dr_grade=dr_grade,
        diabetes_positive=diabetes_positive,
        glucose=glucose_val,
        known_conditions=patient.known_conditions,
    )

    log.info("Running reasoning step...")
    reasoning_result = reason(patient, findings, kg_context)

    elapsed = time.perf_counter() - t_start
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    result = PipelineResult(
        findings=findings,
        knowledge_graph_context=kg_context,
        reasoning=reasoning_result.get("reasoning", ""),
        clinical_summary=reasoning_result.get("clinical_summary", ""),
        alerts=reasoning_result.get("alerts", []),
        recommendations=reasoning_result.get("recommendations", []),
        confidence_level=reasoning_result.get("confidence_level", "low"),
        processing_time_seconds=round(elapsed, 3),
        timestamp=timestamp,
        model_versions={"pipeline": VERSION},
    )

    log.info("Pipeline complete in %.2fs — %d finding(s), %d alert(s).", elapsed, len(findings), len(result.alerts))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge Layer — multimodal clinical AI pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=str, help="Path to patient JSON input file.")
    parser.add_argument("--age", type=int)
    parser.add_argument("--sex", type=str)
    parser.add_argument("--conditions", nargs="*", default=[], help="Known conditions (space-separated).")
    parser.add_argument("--medications", nargs="*", default=[], help="Current medications (space-separated).")
    parser.add_argument("--notes", type=str, default=None, help="Clinical notes string.")
    parser.add_argument("--tabular", type=str, help="Path to JSON file containing PIMA feature dict.")
    parser.add_argument("--eye", type=str, help="Path to fundus image for DR grading.")
    parser.add_argument("--skin", type=str, help="Path to skin lesion image.")
    parser.add_argument("--no-images", action="store_true", help="Skip all image models.")
    parser.add_argument("--save", action="store_true", help="Save result JSON to outputs/.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress formatted report; only print JSON.")
    return parser.parse_args()


def _build_input_from_args(args: argparse.Namespace) -> PipelineInput:
    if args.input:
        return PipelineInput.from_json(args.input)

    patient = PatientContext(
        age=args.age,
        sex=args.sex,
        clinical_notes=args.notes,
        known_conditions=args.conditions or [],
        medications=args.medications or [],
    )

    tabular_data: Optional[Dict[str, Any]] = None
    if args.tabular:
        with open(args.tabular, "r", encoding="utf-8") as fh:
            tabular_data = json.load(fh)
        if "tabular_data" in tabular_data:
            tabular_data = tabular_data["tabular_data"]

    return PipelineInput(
        patient=patient,
        tabular_data=tabular_data,
        eye_image_path=args.eye,
        skin_image_path=args.skin,
    )


def main() -> None:
    args = _parse_args()

    if not args.input and not any([args.tabular, args.eye, args.skin]):
        print("[WARNING] No input provided. Running demo with sample patient...")
        sample_path = Path(__file__).parent / "samples" / "patient_example.json"
        if sample_path.exists():
            pipeline_input = PipelineInput.from_json(str(sample_path))
        else:
            print("[ERROR] No sample file found either. Pass --input or --tabular.")
            sys.exit(1)
    else:
        pipeline_input = _build_input_from_args(args)

    result = run_pipeline(pipeline_input, run_images=not args.no_images)

    if not args.quiet:
        result.print_report()

    if args.save or args.quiet:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUTS_DIR / f"result_{ts}.json"
        out_path.write_text(result.to_json(), encoding="utf-8")
        print(f"\nResult saved to: {out_path}")

    if args.quiet:
        print(result.to_json())


if __name__ == "__main__":
    main()
