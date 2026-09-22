"""
config.py — Centralised configuration for the knowledge layer.
All paths, thresholds, and LLM settings live here.
LLM: Google Gemini only (google-generativeai SDK).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

BASE_DIR = Path(__file__).parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = BASE_DIR / "samples"
OUTPUTS_DIR = BASE_DIR / "outputs"
KG_JSON_PATH = BASE_DIR / "kg_relationships.json"

DR_MODEL_PATH = MODELS_DIR / "best_dr_model.pth"
SKIN_MODEL_PATH = MODELS_DIR / "best_skin_model.pth"
DIABETES_MODEL_PATH = MODELS_DIR / "diabetes_model.joblib"
PIMA_CSV_PATH = DATA_DIR / "pima-indians-diabetes.csv"

DR_IMG_SIZE = (224, 224)
SKIN_IMG_SIZE = (224, 224)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DR_CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
SKIN_CLASS_NAMES = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Chickenpox",
    "Epidermolysis Bullosa",
    "Melanoma",
    "Normal",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
]

SKIN_MALIGNANT_CLASSES = {"Actinic keratosis", "Melanoma", "Squamous cell carcinoma"}

DIABETES_PHYSIO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

DIABETES_ALERT_THRESHOLD = float(os.getenv("DIABETES_ALERT_THRESHOLD", "0.50"))
DR_REFERABLE_GRADE = int(os.getenv("DR_REFERABLE_GRADE", "2"))
SKIN_MALIGNANT_THRESHOLD = float(os.getenv("SKIN_MALIGNANT_THRESHOLD", "0.40"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
