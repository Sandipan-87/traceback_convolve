"""
TraceBack Configuration
=======================
"""

from pathlib import Path

# =============================================================================
# QDRANT CLOUD CONFIGURATION
# =============================================================================

QDRANT_URL = "https://29df6888-9d47-4967-8d24-f165f0900abc.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.cIaS1ZDIxjymihM1aW59DAJr_BgFpbBSn4W52wFKNvs"

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

# Create directories
for dir_path in [FRAMES_DIR, REPORTS_DIR, UPLOADS_DIR]:
    dir_path.mkdir(exist_ok=True)