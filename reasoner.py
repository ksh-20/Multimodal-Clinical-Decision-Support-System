"""
reasoner.py — LLM reasoning step (Google Gemini) + offline rule-based fallback.

Provider: Google Gemini only (google-generativeai SDK).
Model:    gemini-2.5-flash-lite  (configurable via GEMINI_MODEL in .env)

Behaviour:
  - If GEMINI_API_KEY is set  → calls Gemini API, validates response.
  - If GEMINI_API_KEY missing → automatically falls back to offline rule-based reasoning.
  - Any API error             → automatically falls back to offline reasoning.

The offline fallback uses the knowledge graph context + clinical rules to
produce the same output structure without any API call.
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
You are a clinical decision-support assistant integrated into a multimodal AI pipeline.
You receive structured findings from three validated ML models and a knowledge graph context.
Your task is to synthesise these into a coherent, conservative clinical summary.

Rules you MUST follow:
1. Never diagnose. Use language like "the model suggests", "screening finding indicates", "warrants further evaluation".
2. Always recommend confirmatory testing before any clinical action.
3. For skin lesion findings flagged as malignant/premalignant, always recommend dermatologist review.
4. For DR grade >= 2, always recommend ophthalmology referral.
5. Be concise. The summary must be readable by a non-specialist GP.
7. Respond ONLY with a JSON object with exactly these keys:
   {
     "reasoning": "<string: 2-5 sentences — evidence trail>",
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
    return (
        f"PATIENT CONTEXT:\n{json.dumps(patient.to_dict(), indent=2)}\n\n"
        f"ML MODEL FINDINGS:\n{json.dumps([f.to_dict() for f in findings], indent=2)}\n\n"
        f"KNOWLEDGE GRAPH CONTEXT:\n{json.dumps(kg_context, indent=2, default=str)}\n\n"
        "Synthesise the above into the required JSON response."
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
                "Confirmatory fasting glucose and HbA1c are required."
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
            f"Fundus analysis: {dr.label} (Grade {grade}, conf={dr.confidence:.1%}). "
            f"Referrable: {'Yes' if referrable else 'No'}."
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
            f"(conf={sk.confidence:.1%}, malignant potential: {malignant_pot}). "
            "Note: test set was small (~15 samples/class); estimates carry wide uncertainty."
        )
        recommendations.extend(kg_context.get("skin", {}).get("recommendations", []))
        if sk.alert:
            alerts.append(
                f"Malignant/premalignant skin lesion predicted ({sk.label}): "
                "urgent dermatology biopsy recommended."
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
