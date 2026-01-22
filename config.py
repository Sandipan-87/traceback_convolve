"""
TraceBack Configuration
=======================
"""

import streamlit as st
import os
from pathlib import Path

# =============================================================================
# QDRANT CLOUD CONFIGURATION (SECURE LOAD)
# =============================================================================

def load_secrets():
    """
    Securely load secrets from Streamlit secrets.toml or Environment Variables.
    """
    try:
        
        url = st.secrets["qdrant"]["url"]
        key = st.secrets["qdrant"]["api_key"]
        return url, key
    except (FileNotFoundError, KeyError):
        
        url = os.getenv("QDRANT_URL")
        key = os.getenv("QDRANT_API_KEY")
        
        return url, key

QDRANT_URL, QDRANT_API_KEY = load_secrets()

# =============================================================================
# COLLECTION SETTINGS
# =============================================================================

COLLECTION_NAME = "traceback_persons"
VECTOR_DIMENSION = 512

# =============================================================================
# PROCESSING SETTINGS
# =============================================================================

BATCH_SIZE = 100
FRAME_SKIP = 15  # Process every 15th frame
MIN_PERSON_HEIGHT = 80
MIN_PERSON_WIDTH = 40

# =============================================================================
# SEARCH SETTINGS
# =============================================================================

DEFAULT_TOP_K = 100
MIN_CONFIDENCE_THRESHOLD = 0.40
RED_ALERT_THRESHOLD = 0.85
DEDUP_TIME_WINDOW_SECONDS = 5  # Group matches within 5 seconds

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / "frames"
REPORTS_DIR = BASE_DIR / "reports"
UPLOADS_DIR = BASE_DIR / "uploads"


for dir_path in [FRAMES_DIR, REPORTS_DIR, UPLOADS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
