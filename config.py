import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# المسار الرئيسي للمشروع (بيشتغل على أي جهاز أوتوماتيك)
BASE_DIR = Path(__file__).resolve().parent

# مسارات الداتا والـ Artifacts
DATA_DIR = BASE_DIR / "data"
TELCO_DATA_PATH = DATA_DIR / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
POLICIES_DIR = DATA_DIR / "policies"
SHARED_SCHEMA_PATH = DATA_DIR / "shared_schema.json"

MODEL_PATH = BASE_DIR / "ml_model" / "model.joblib"
CHROMA_STORE_DIR = BASE_DIR / "vector_db" / "chroma_store"
LOGS_DIR = BASE_DIR / "logs"

# إعدادات الـ LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# إعدادات Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")