"""
reasoner.py — LLM reasoning step (Google Gemini) + offline rule-based fallback.

Architecture:
  ML Models (Vision / Tabular Classifiers)
      ↓ Predictions & Probabilities
  Knowledge Graph Query Engine
      ↓ Structured Evidence Subgraph & Clinical Rules
  LLM / Offline Clinical Reasoner
      ↓ Interpretable Evidence Synthesis & Guideline Guidance
  Structured Decision-Support Summary

Role of the LLM:
  The LLM does NOT diagnose diseases directly. It acts as an interpretable
  evidence-synthesis agent that translates ML model outputs and symbolic
  Knowledge Graph associations into conservative, guideline-compliant clinical text.

Non-Causal Association Principle:
  Retrieved associations from the Knowledge Graph (e.g., Diabetes <-> Retinopathy)
  represent documented clinical co-occurrences and shared systemic risk factors,
  NOT direct causal determinism. The reasoner strictly avoids conflating correlation
  with single-cause causality.
"""
from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Dict, List, Optional

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)
from schema import Finding, FindingType, PatientContext

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = textwrap.dedent("""
You are a clinical decision-support and evidence-synthesis assistant integrated into a multimodal AI pipeline.
You receive structured findings from validated ML sub-models and an evidence subgraph from a curated Knowledge Graph.
Your task is to synthesize these into a coherent, conservative, guideline-grounded clinical explanation.

Architectural Role:
- You do NOT diagnose patients directly. Diagnosis is reserved for licensed healthcare providers.
- You explain ML model predictions in the context of clinical guidelines and Knowledge Graph associations.

Rules you MUST follow:
1. Never assert definitive diagnoses. Use conservative framing: "the model suggests", "screening finding indicates", "warrants further evaluation".
2. Always recommend confirmatory laboratory or specialist testing before any clinical action.
3. For skin lesion findings flagged as malignant/premalignant, always recommend urgent dermatologist biopsy review.
4. For DR grade >= 2, always recommend ophthalmology referral within established screening intervals.
5. Non-Causal Association Principle: You MUST NOT rewrite retrieved Knowledge Graph associations or correlations as direct causal relationships. Frame co-occurring findings as correlated clinical risk factors, comorbid manifestations, or shared systemic pathways (e.g., microvascular changes associated with chronic hyperglycemia).
6. Be concise, objective, and clear. The summary must be easily digestible by a general practitioner (GP).
7. Respond ONLY with a JSON object with exactly these keys:
   {
     "reasoning": "<string: 2-5 sentences — evidence trail and non-causal clinical synthesis>",
     "clinical_summary": "<string: 1-3 sentences in plain language>",
     "alerts": ["<string>", ...],
     "recommendations": ["<string>", ...],
     "confidence_level": "<one of: high | moderate | low>"
   }
""").strip()


def _build_user_prompt(
    patient: PatientContext,
    findings: List[Finding],
    kg_context: Dict[str, Any],
) -> str:
    pat_str = json.dumps(patient.to_dict(), indent=2)
    find_str = json.dumps([f.to_dict() for f in findings], indent=2)
    kg_str = json.dumps(kg_context, indent=2, default=str)
    return (
        "PATIENT CONTEXT:\n"
        + pat_str
        + "\n\n"
        + "ML MODEL FINDINGS (Primary Model Outputs):\n"
        + find_str
        + "\n\n"
        + "KNOWLEDGE GRAPH EVIDENCE SUBGRAPH (Correlated Guidelines & Associations):\n"
        + kg_str
        + "\n\n"
        + "Synthesise the above into the required JSON response following the Non-Causal Association Principle."
    )


def _validate(parsed: Dict[str, Any]) -> bool:
    return {"reasoning", "clinical_summary", "alerts", "recommendations", "confidence_level"}.issubset(parsed.keys())


def _call_gemini(prompt: str) -> Optional[Dict[str, Any]]:
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=_SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
                response_mime_type="application/json",
            ),
        )
        response = model.generate_content(prompt)
        text = response.text.strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        parsed = json.loads(text)
        if _validate(parsed):
            log.info("Gemini reasoning complete (model=%s).", GEMINI_MODEL)
            return parsed

        log.warning("Gemini response missing required fields; falling back to offline.")

    except Exception as exc:
        log.warning("Gemini API error (%s); falling back to offline reasoning.", exc)

    return None


def _offline_reason(
    patient: PatientContext,
    findings: List[Finding],
    kg_context: Dict[str, Any],
) -> Dict[str, Any]:
    alerts: List[str] = []
    recommendations: List[str] = []
    reasoning_parts: List[str] = []
    confidence_level = "moderate"

    for f in findings:
        if f.alert:
            alerts.append(f.summary_line())

    dm = next((f for f in findings if f.type == FindingType.DIABETES_RISK), None)
    dr = next((f for f in findings if f.type == FindingType.DIABETIC_RETINOPATHY), None)
    sk = next((f for f in findings if f.type == FindingType.SKIN_LESION), None)

    if dm:
        if dm.alert:
            reasoning_parts.append(
                f"Diabetes screening model suggests elevated risk (probability {dm.confidence:.1%}). "
                "Confirmatory fasting plasma glucose and HbA1c testing are required."
            )
            recommendations.extend([
                "Order fasting plasma glucose and HbA1c to confirm diabetes diagnosis.",
                "Assess BMI and initiate lifestyle modification counselling if overweight.",
                "Screen for comorbid hypertension and dyslipidaemia.",
            ])
            if dm.metadata.get("glucose", 0) >= 200:
                alerts.append("Glucose input >= 200 mg/dL: highly suggestive of diabetes; urgent confirmatory testing.")
            recommendations.extend(kg_context.get("diabetes", {}).get("recommendations", []))
        else:
            reasoning_parts.append(
                f"Diabetes screening result is low-risk (probability {dm.confidence:.1%})."
            )
            recommendations.append("Continue routine annual diabetes screening as per guidelines.")

    if dr:
        grade = dr.metadata.get("grade", 0)
        referrable = dr.metadata.get("referrable", False)
        reasoning_parts.append(
            f"Retinal fundus analysis: {dr.label} (Grade {grade}, conf={dr.confidence:.1%}). "
            f"Referrable status: {'Yes (Urgent Evaluation)' if referrable else 'No (Routine Monitoring)'}."
        )
        recommendations.extend(kg_context.get("dr", {}).get("recommendations", []))
        if grade == 4:
            alerts.append("Grade 4 Proliferative DR: sight-threatening; urgent ophthalmology referral required.")
        elif grade == 3:
            alerts.append("Grade 3 Severe NPDR: high risk of progression; urgent ophthalmology referral.")
        elif grade >= 2:
            recommendations.insert(0, "Ophthalmology referral within 6 months for Moderate NPDR.")

    if sk:
        malignant_pot = kg_context.get("skin", {}).get("malignant_potential", "unknown")
        reasoning_parts.append(
            f"Skin lesion model predicts {sk.label} "
            f"(conf={sk.confidence:.1%}, risk classification: {malignant_pot}). "
            "Note: dermatoscopy test set had ~20 samples/class; estimates carry clinical uncertainty."
        )
        recommendations.extend(kg_context.get("skin", {}).get("recommendations", []))
        if sk.alert:
            alerts.append(
                f"High-risk skin lesion predicted ({sk.label}): "
                "urgent dermatology biopsy and histopathology recommended."
            )

    # Multi-finding non-causal correlation synthesis
    active_count = sum(1 for f in [dm, dr, sk] if f and f.alert)
    if active_count >= 2:
        reasoning_parts.append(
            "Clinical Synthesis Note: Multiple comorbid screening alerts detected across organ systems. "
            "These findings represent correlated clinical manifestations and shared microvascular/systemic risk factors, "
            "not direct single-cause causality. Multidisciplinary referral is recommended."
        )

    if kg_context.get("hypertension_note"):
        recommendations.append(kg_context["hypertension_note"])

    for rule in kg_context.get("applicable_rules", []):
        recommendations.append(f"[Rule: {rule['id']}] {rule['action']}")

    if not findings:
        reasoning_parts = ["No model findings available. Ensure inputs (tabular data, images) were supplied."]
        recommendations = ["Supply tabular data and/or images to generate findings."]
        confidence_level = "low"

    reasoning = " ".join(reasoning_parts) if reasoning_parts else "No findings to reason over."
    alert_str = f" Active alerts: {len(alerts)}." if alerts else " No urgent alerts."
    clinical_summary = (
        f"Multimodal clinical assessment: {len(findings)} model finding(s) processed.{alert_str} "
        "Please review recommendations and arrange confirmatory investigations as indicated."
    )

    unique_recs = list(dict.fromkeys(r for r in recommendations if r))
    return {
        "reasoning": reasoning,
        "clinical_summary": clinical_summary,
        "alerts": alerts,
        "recommendations": unique_recs,
        "confidence_level": confidence_level,
    }


def reason(
    patient: PatientContext,
    findings: List[Finding],
    kg_context: Dict[str, Any],
) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        log.info("GEMINI_API_KEY not set — using offline reasoning.")
        return _offline_reason(patient, findings, kg_context)

    result = _call_gemini(_build_user_prompt(patient, findings, kg_context))

    if result is None:
        log.info("Falling back to offline reasoning.")
        result = _offline_reason(patient, findings, kg_context)

    return result
