import os

from dotenv import load_dotenv

from . import ROOT

load_dotenv(os.path.join(ROOT, ".env"))
WEIGHTS = os.environ.get("WEIGHTS_DIR", os.path.join(ROOT, "models", "weights"))
DB_PATH = os.environ.get("DB_PATH", os.path.join(ROOT, "data", "app.db"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
