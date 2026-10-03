from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
FAQ_FILE = DATA_DIR / "sai_faqs.csv"


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        value = str(st.secrets.get(name, default)).strip()
    except Exception:
        value = default
    return value


GEMINI_API_KEY = _secret("GEMINI_API_KEY")
# Free-only project: use Gemini Flash-Lite by default.
# GEMINI_MODEL is intentionally fixed here so the app does not accidentally
# switch to a paid model through Streamlit Secrets.
GEMINI_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = _secret(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def validate_settings() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it under Streamlit Cloud → Settings → Secrets."
        )
