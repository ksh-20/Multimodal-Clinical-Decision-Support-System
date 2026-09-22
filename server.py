"""
server.py — FastAPI backend for the Multimodal Clinical Knowledge Layer.
Provides REST endpoints for:
  - Multi-modal inference (/api/analyze, /api/analyze-json)
  - Medical Knowledge Graph visualization (/api/kg)
  - Interactive LLM Follow-up Doctor Chatbot (/api/chat)
  - System health and sample data (/api/health, /api/sample)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    BASE_DIR,
    DIABETES_MODEL_PATH,
    DR_MODEL_PATH,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    KG_JSON_PATH,
    LOG_LEVEL,
    SAMPLES_DIR,
    SKIN_MODEL_PATH,
)
from medical_kg import get_kg
from run_pipeline import run_pipeline
from schema import PatientContext, PipelineInput

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("server")

app = FastAPI(
    title="Multimodal Clinical Knowledge Layer",
    version="1.0.0",
    description="Multimodal clinical decision support system combining ML models, Knowledge Graph, and LLM reasoning.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and samples directories
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_file))


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "models": {
            "diabetes": {
                "available": DIABETES_MODEL_PATH.exists() or (DIABETES_MODEL_PATH.parent / "diabetes_pipeline.pkl").exists(),
                "path": str(DIABETES_MODEL_PATH.name),
            },
            "diabetic_retinopathy": {
                "available": DR_MODEL_PATH.exists(),
                "path": str(DR_MODEL_PATH.name),
            },
            "skin_disease": {
                "available": SKIN_MODEL_PATH.exists(),
                "path": str(SKIN_MODEL_PATH.name),
            },
        },
        "llm": {
            "provider": "Clinical AI Engine",
            "model": "AI Reasoning Assistant",
            "configured": bool(GEMINI_API_KEY),
        },
        "knowledge_graph": {
            "available": KG_JSON_PATH.exists(),
            "nodes": get_kg().G.number_of_nodes(),
            "edges": get_kg().G.number_of_edges(),
            "rules": len(get_kg().clinical_rules),
        },
    }


@app.get("/api/sample")
def get_sample_patient() -> Dict[str, Any]:
    sample_json_path = SAMPLES_DIR / "patient_example.json"
    if not sample_json_path.exists():
        raise HTTPException(status_code=404, detail="Sample patient file not found")
    with open(sample_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Remove image paths from sample data so user must upload their own images
    clean_data = {
        "patient": data.get("patient", {}),
        "tabular_data": data.get("tabular_data", {}),
    }
    return {
        "patient_data": clean_data,
    }


@app.get("/api/json-template")
def get_json_template() -> Dict[str, Any]:
    """Returns the expected JSON structure and descriptions for patient data upload."""
    template_data = {
        "patient": {
            "age": 55,
            "sex": "female",
            "clinical_notes": "Patient presents with fatigue, polyuria, and increased thirst over past 3 months. Family history of type 2 diabetes.",
            "known_conditions": ["hypertension"],
            "medications": ["metformin", "amlodipine"]
        },
        "tabular_data": {
            "Pregnancies": 3,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 55
        }
    }
    field_descriptions = {
        "patient.age": "Patient age in years (integer)",
        "patient.sex": "Biological sex ('female', 'male', or 'other')",
        "patient.clinical_notes": "Free-text description of clinical symptoms and patient history",
        "patient.known_conditions": "List of current clinical conditions (e.g. ['hypertension'])",
        "patient.medications": "List of current medications (e.g. ['metformin'])",
        "tabular_data.Pregnancies": "Number of pregnancies (integer, 0 if male or nulliparous)",
        "tabular_data.Glucose": "Plasma glucose concentration in mg/dL (critical biomarker)",
        "tabular_data.BloodPressure": "Diastolic blood pressure in mm Hg",
        "tabular_data.SkinThickness": "Triceps skin fold thickness in mm (enter 0 if unavailable)",
        "tabular_data.Insulin": "2-Hour serum insulin in mu U/ml (enter 0 if unavailable)",
        "tabular_data.BMI": "Body mass index (weight in kg / (height in m)^2)",
        "tabular_data.DiabetesPedigreeFunction": "Diabetes pedigree genetic risk function score",
        "tabular_data.Age": "Patient age in years"
    }
    return {
        "template": template_data,
        "field_descriptions": field_descriptions,
    }


@app.get("/api/download-template")
def download_json_template():
    """Serves downloadable patient_template.json directly."""
    from fastapi.responses import Response
    template_data = {
        "patient": {
            "age": 55,
            "sex": "female",
            "clinical_notes": "Patient presents with fatigue, polyuria, and increased thirst over past 3 months. Family history of type 2 diabetes.",
            "known_conditions": ["hypertension"],
            "medications": ["metformin", "amlodipine"]
        },
        "tabular_data": {
            "Pregnancies": 3,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 55
        }
    }
    return Response(
        content=json.dumps(template_data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=patient_template.json"}
    )


@app.get("/api/kg")
def get_knowledge_graph() -> Dict[str, Any]:
    """Returns the entire Knowledge Graph structure for frontend graph visualization."""
    kg = get_kg()
    with open(KG_JSON_PATH, "r", encoding="utf-8") as f:
        raw_kg = json.load(f)

    # Format nodes with display colors and categories for vis/canvas
    category_colors = {
        "disease": "#ef4444",       # red
        "finding": "#3b82f6",       # blue
        "risk_factor": "#f59e0b",   # amber
        "complication": "#ec4899",  # pink
        "drug": "#10b981",          # emerald green
        "recommendation": "#8b5cf6",# purple
        "monitoring": "#06b6d4",    # cyan
    }

    nodes = []
    for n in raw_kg.get("nodes", []):
        ntype = n.get("type", "finding")
        nodes.append({
            "id": n["id"],
            "name": n.get("name", n["id"]),
            "type": ntype,
            "color": category_colors.get(ntype, "#64748b"),
            "attrs": {k: v for k, v in n.items() if k not in ("id", "name", "type")},
        })

    edges = []
    for e in raw_kg.get("edges", []):
        edges.append({
            "source": e["source"],
            "target": e["target"],
            "relation": e.get("relation", "relates_to"),
            "note": e.get("note", ""),
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "rules": raw_kg.get("clinical_rules", []),
    }



@app.get("/api/metrics")
def get_system_metrics() -> Dict[str, Any]:
    """Returns combined system metrics from KG structure and individual model training metrics."""
    kg = get_kg()
    kg_metrics = {}

    try:
        total_nodes = kg.G.number_of_nodes()
        total_edges = kg.G.number_of_edges()
        total_rules = len(kg.clinical_rules)

        with open(KG_JSON_PATH, "r", encoding="utf-8") as f:
            raw_kg = json.load(f)

        node_type_counts = {}
        for n in raw_kg.get("nodes", []):
            ntype = n.get("type", "other")
            node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1

        node_type_map = {n["id"]: n.get("type", "other") for n in raw_kg.get("nodes", [])}
        cross_modal_edges = 0
        for e in raw_kg.get("edges", []):
            src_type = node_type_map.get(e.get("source", ""), "")
            tgt_type = node_type_map.get(e.get("target", ""), "")
            if src_type and tgt_type and src_type != tgt_type:
                cross_modal_edges += 1

        all_edges = raw_kg.get("edges", [])
        connected_ids = set()
        for e in all_edges:
            connected_ids.add(e.get("source", ""))
            connected_ids.add(e.get("target", ""))
        nodes_with_edges = sum(1 for n in raw_kg.get("nodes", []) if n["id"] in connected_ids)
        kg_coverage = round(nodes_with_edges / max(total_nodes, 1) * 100, 1)

        kg_metrics = {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_rules": total_rules,
            "node_type_distribution": node_type_counts,
            "cross_modal_edges": cross_modal_edges,
            "kg_coverage_pct": kg_coverage,
            "nodes_with_connections": nodes_with_edges,
        }
    except Exception as e:
        log.warning("KG metrics computation error: %s", e)
        kg_metrics = {"error": str(e)}

    model_metrics_path = BASE_DIR / "model_metrics.json"
    model_metrics = {}
    dm, dr, sk = {}, {}, {}
    if model_metrics_path.exists():
        try:
            with open(model_metrics_path, "r", encoding="utf-8") as f:
                raw_model_metrics = json.load(f)
            model_metrics = {k: v for k, v in raw_model_metrics.items() if not k.startswith("_")}
            dm = model_metrics.get("diabetes_model", {})
            dr = model_metrics.get("diabetic_retinopathy_model", {})
            sk = model_metrics.get("skin_disease_model", {})
        except Exception as e:
            log.warning("Model metrics load error: %s", e)
            model_metrics = {"error": str(e)}

    auc_values, f1_values, acc_values = [], [], []
    try:
        if dm.get("metrics", {}).get("auc_roc"):
            auc_values.append(dm["metrics"]["auc_roc"])
        if dr.get("metrics", {}).get("auc_roc_multiclass"):
            auc_values.append(dr["metrics"]["auc_roc_multiclass"])
        if sk.get("metrics", {}).get("auc_roc_multiclass"):
            auc_values.append(sk["metrics"]["auc_roc_multiclass"])
        if dm.get("metrics", {}).get("f1_score"):
            f1_values.append(dm["metrics"]["f1_score"])
        if dr.get("metrics", {}).get("f1_macro"):
            f1_values.append(dr["metrics"]["f1_macro"])
        if sk.get("metrics", {}).get("f1_macro"):
            f1_values.append(sk["metrics"]["f1_macro"])
        if dm.get("metrics", {}).get("accuracy"):
            acc_values.append(dm["metrics"]["accuracy"])
        if dr.get("metrics", {}).get("accuracy"):
            acc_values.append(dr["metrics"]["accuracy"])
        if sk.get("metrics", {}).get("accuracy"):
            acc_values.append(sk["metrics"]["accuracy"])
    except Exception:
        pass

    combined_auc = round(sum(auc_values) / len(auc_values), 4) if auc_values else None
    combined_f1 = round(sum(f1_values) / len(f1_values), 4) if f1_values else None
    combined_accuracy = round(sum(acc_values) / len(acc_values), 4) if acc_values else None

    return {
        "knowledge_graph": kg_metrics,
        "model_metrics": model_metrics,
        "combined": {
            "system_auc_roc": combined_auc,
            "system_f1_macro": combined_f1,
            "system_accuracy": combined_accuracy,
            "models_evaluated": len(auc_values),
            "pipeline_version": "1.0.0",
            "description": (
                "Combined metrics are macro-averaged across all evaluated sub-models. "
                "AUC-ROC is threshold-independent and primary for clinical class-imbalanced datasets. "
                "Update model_metrics.json after retraining notebooks to reflect new values."
            ),
        },
    }


def _compute_active_subgraph(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes which nodes and edges in the KG are active given the current findings."""
    active_node_ids = set()
    active_edge_ids = set()

    for f in findings:
        ftype = f.get("type")
        label = f.get("label", "")
        meta = f.get("metadata", {})

        if ftype == "diabetes_risk":
            active_node_ids.add("hyperglycemia")
            active_node_ids.add("insulin_resistance")
            if f.get("alert") or label == "Diabetic":
                active_node_ids.add("diabetes_t2")
                active_node_ids.add("rec_glucose_control")
                active_node_ids.add("rec_bp_control")
                active_node_ids.add("rec_foot_exam")
                active_node_ids.add("rec_renal_screen")
                active_node_ids.add("metformin")

        elif ftype == "diabetic_retinopathy":
            grade = meta.get("grade", 0)
            dr_node = f"dr_grade{grade}"
            active_node_ids.add(dr_node)
            active_node_ids.add("diabetes_t2")
            if grade >= 2:
                active_node_ids.add("rec_fundus_urgent")
                active_node_ids.add("anti_vegf")
                active_node_ids.add("laser_photocoag")
            else:
                active_node_ids.add("rec_fundus_annual")

        elif ftype == "skin_lesion":
            skin_node_map = {
                "Actinic keratosis": "actinic_keratosis",
                "Atopic Dermatitis": "atopic_dermatitis",
                "Benign keratosis": "benign_keratosis",
                "Chickenpox": "chickenpox",
                "Epidermolysis Bullosa": "epidermolysis_bullosa",
                "Melanoma": "melanoma",
                "Normal": "normal_skin",
                "Squamous cell carcinoma": "squamous_cell_carcinoma",
                "Tinea Ringworm Candidiasis": "tinea_candidiasis",
            }
            node_id = skin_node_map.get(label)
            if node_id:
                active_node_ids.add(node_id)
                if meta.get("is_malignant"):
                    active_node_ids.add("rec_biopsy")
                else:
                    active_node_ids.add("rec_dermatology_review")

    # Find edges connecting active nodes
    kg = get_kg()
    active_edges = []
    for u, v, data in kg.G.edges(data=True):
        if u in active_node_ids and v in active_node_ids:
            active_edges.append({
                "source": u,
                "target": v,
                "relation": data.get("relation", ""),
                "note": data.get("note", ""),
            })

    return {
        "active_nodes": list(active_node_ids),
        "active_edges": active_edges,
    }


@app.post("/api/analyze")
async def analyze_patient(
    patient_json: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    sex: Optional[str] = Form(None),
    clinical_notes: Optional[str] = Form(None),
    known_conditions: Optional[str] = Form(None),
    medications: Optional[str] = Form(None),
    glucose: Optional[float] = Form(None),
    blood_pressure: Optional[float] = Form(None),
    skin_thickness: Optional[float] = Form(None),
    insulin: Optional[float] = Form(None),
    bmi: Optional[float] = Form(None),
    dpf: Optional[float] = Form(None),
    pregnancies: Optional[int] = Form(None),
    eye_image: Optional[UploadFile] = File(None),
    skin_image: Optional[UploadFile] = File(None),
    run_images: Optional[bool] = Form(True),
) -> Dict[str, Any]:
    temp_dir = Path(tempfile.mkdtemp(prefix="kl_upload_"))
    try:
        # 1. Parse patient context and tabular data
        if patient_json:
            try:
                data = json.loads(patient_json)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid patient_json: {e}")
            p_dict = data.get("patient", {})
            patient = PatientContext(
                age=p_dict.get("age", 50),
                sex=p_dict.get("sex", "unknown"),
                clinical_notes=p_dict.get("clinical_notes", ""),
                known_conditions=p_dict.get("known_conditions", []),
                medications=p_dict.get("medications", []),
            )
            tabular_data = data.get("tabular_data")
            eye_path = None
            skin_path = None
        else:
            kc = [c.strip() for c in known_conditions.split(",")] if known_conditions else []
            meds = [m.strip() for m in medications.split(",")] if medications else []
            patient = PatientContext(
                age=age or 50,
                sex=sex or "unknown",
                clinical_notes=clinical_notes or "",
                known_conditions=kc,
                medications=meds,
            )
            tabular_data = {
                "Glucose": glucose if glucose is not None else 120.0,
                "BloodPressure": blood_pressure if blood_pressure is not None else 75.0,
                "SkinThickness": skin_thickness if skin_thickness is not None else 25.0,
                "Insulin": insulin if insulin is not None else 0.0,
                "BMI": bmi if bmi is not None else 28.0,
                "DiabetesPedigreeFunction": dpf if dpf is not None else 0.5,
                "Pregnancies": pregnancies if pregnancies is not None else 0,
                "Age": age or 50,
            }
            eye_path = None
            skin_path = None

        # 2. Process uploaded images (strictly user uploaded)
        if eye_image and eye_image.filename:
            eye_dst = temp_dir / eye_image.filename
            with open(eye_dst, "wb") as f:
                shutil.copyfileobj(eye_image.file, f)
            eye_path = str(eye_dst)

        if skin_image and skin_image.filename:
            skin_dst = temp_dir / skin_image.filename
            with open(skin_dst, "wb") as f:
                shutil.copyfileobj(skin_image.file, f)
            skin_path = str(skin_dst)

        pi = PipelineInput(
            patient=patient,
            tabular_data=tabular_data,
            eye_image_path=eye_path,
            skin_image_path=skin_path,
        )

        # 3. Execute Pipeline
        result = run_pipeline(pi, run_images=bool(run_images))
        result_dict = result.to_dict()

        # 4. Attach active KG nodes and edges for visualization
        active_kg = _compute_active_subgraph(result_dict["findings"])
        result_dict["active_kg"] = active_kg

        return result_dict

    finally:
        # Schedule cleanup of temp uploaded files after request handling
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


class ChatRequest(BaseModel):
    message: str
    report: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = None


@app.post("/api/chat")
def chat_with_assistant(req: ChatRequest) -> Dict[str, Any]:
    """
    Conversational follow-up assistant with access to the patient diagnosis report,
    ML predictions, Knowledge Graph clinical pathways, and previous chat turns.
    """
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    report = req.report or {}
    history = req.history or []

    # If AI API is configured, generate intelligent response with full context
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)

            system_instruction = (
                "You are Dr. AI, an empathetic, highly knowledgeable clinical decision-support assistant. "
                "You are discussing the Multimodal AI Clinical Report for this patient. "
                "You have full visibility into the ML findings, Diabetic Retinopathy screening, "
                "Skin Lesion diagnosis, Diabetes risk score, Knowledge Graph guidelines, and Recommendations. "
                "RULES:\n"
                "1. Explain medical terms clearly and empathetically without medical jargon barrier.\n"
                "2. Cite the specific findings (e.g. 'The retinal scan indicated severe NPDR grade 3 with 91% confidence...').\n"
                "3. Explain WHY certain actions (like urgency of biopsy or ophthalmologist visit) are recommended.\n"
                "4. Remind the user that you provide clinical decision support and advice must be reviewed by their attending physician.\n"
                "5. Keep responses concise, structured, and easy to read (use short paragraphs or bullet points)."
            )

            prompt_context = (
                f"CLINICAL REPORT CONTEXT:\n{json.dumps(report, indent=2, default=str)}\n\n"
                f"CONVERSATION HISTORY:\n"
            )
            for turn in history[-6:]:  # last 6 turns for context
                role = "User" if turn.get("role") == "user" else "Assistant"
                prompt_context += f"{role}: {turn.get('content', '')}\n"

            prompt_context += f"\nUser Question: {user_msg}\nAssistant Reply:"

            model = genai.GenerativeModel(
                GEMINI_MODEL,
                system_instruction=system_instruction,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=1000,
                ),
            )
            response = model.generate_content(prompt_context)
            reply = response.text.strip()
            return {
                "reply": reply,
                "timestamp": time.strftime("%H:%M:%S"),
            }
        except Exception as e:
            log.warning("AI chat error: %s. Using rule-based fallback response.", e)

    # Intelligent offline fallback if API key not available or rate limited
    findings_summary = []
    for f in report.get("findings", []):
        findings_summary.append(f"{f.get('type')}: {f.get('label')} ({f.get('confidence', 0)*100:.1f}%)")

    fallback_reply = (
        f"Thank you for your question regarding the clinical findings. "
        f"Based on the clinical report findings ({', '.join(findings_summary) if findings_summary else 'recorded findings'}): \n\n"
        f"• The multimodal pipeline identified actionable risk indicators requiring specialist validation.\n"
        f"• Prioritized Recommendations: {'; '.join(report.get('recommendations', ['Consult primary physician for full evaluation'])[:3])}.\n"
        f"• Decision Rationale: {report.get('clinical_summary', 'Detailed screening performed.')}\n\n"
        f"Please consult with your licensed healthcare provider to interpret these screening results."
    )
    return {
        "reply": fallback_reply,
        "timestamp": time.strftime("%H:%M:%S"),
    }
