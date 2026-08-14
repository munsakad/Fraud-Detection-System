import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = (BASE_DIR / os.getenv("ML_ARTIFACTS_DIR", "../ml")).resolve()
DATA_DIR = ML_DIR / "data"
MODELS_DIR = ML_DIR / "models"
REPORTS_DIR = ML_DIR / "reports"

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]

SIMULATOR_INTERVAL_SECONDS = float(os.getenv("SIMULATOR_INTERVAL_SECONDS", "1.5"))
LIVE_BUFFER_SIZE = int(os.getenv("LIVE_BUFFER_SIZE", "500"))

RISK_THRESHOLDS = {"medium": 0.2, "high": 0.5, "critical": 0.8}
