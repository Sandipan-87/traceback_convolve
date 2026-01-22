"""
TraceBack Command Center
========================
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

# Path setup
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# ============================================================================
# PAGE CONFIG (MUST BE FIRST!)
# ============================================================================

st.set_page_config(
    page_title="TraceBack Command Center",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS STYLES
# ============================================================================

st.markdown("""
<style>
/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Dark theme */
.stApp {
    background: linear-gradient(135deg, #0a0e17 0%, #1a1a2e 100%);
}

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600&display=swap');

h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    color: #00d4ff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a2332, #0d1321) !important;
    color: #00d4ff !important;
    border: 2px solid #00d4ff !important;
    border-radius: 0 !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
}

.stButton > button:hover {
    background: #00d4ff !important;
    color: #0a0e17 !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.5) !important;
}

/* Alerts */
.red-alert {
    background: linear-gradient(135deg, rgba(255, 58, 58, 0.2), rgba(139, 0, 0, 0.3));
    border: 2px solid #ff3a3a;
    padding: 20px;
    margin: 10px 0;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(255, 58, 58, 0.3); }
    50% { box-shadow: 0 0 40px rgba(255, 58, 58, 0.6); }
}

.green-alert {
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 100, 50, 0.2));
    border: 2px solid #00ff88;
    padding: 20px;
    margin: 10px 0;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: #00d4ff !important;
}

/* Inputs */
.stTextInput > div > div > input {
    background: #1a2332 !important;
    color: #e0e6ed !important;
    border: 1px solid #2a3a50 !important;
    border-radius: 0 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321, #0a0e17) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a2332 !important;
    border: 2px dashed #2a3a50 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHED RESOURCES
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_backend():
    """Load backend components once."""
    try:
        import config
        from src.database import QdrantManager
        from src.processor import EmbeddingModel
        
        return {
            "config": config,
            "qdrant": QdrantManager(),
            "model": EmbeddingModel(),
            "status": "ready",
            "error": None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@st.cache_data(ttl=30)
def get_db_stats(_qdrant):
    """Get database stats."""
    try:
        return _qdrant.get_stats()
    except:
        return {"total_vectors": 0, "status": "offline"}

# ============================================================================
# SESSION STATE
# ============================================================================

def init_state():
    """Initialize session state."""
    defaults = {
        "case_id": f"TB-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "upload_id": None,  # NEW: Session isolation ID
        "results": None,
        "summary": None,
        "child_info": {},
        "video_processed": False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_upload_id() -> str:
    """Generate unique upload session ID."""
    return str(uuid.uuid4())

def safe_image(path: str):
    """Load image safely."""
    try:
        if path and os.path.exists(path):
            return Image.open(path)
    except:
        pass
    return Image.new('RGB', (200, 250), color='#1a2332')

def show_alert(level: str, title: str, message: str):
    """Display styled alert."""
    color = "#ff3a3a" if level == "red" else "#00ff88"
    st.markdown(f"""
    <div class="{level}-alert">
        <h3 style="color: {color}; font-family: 'Orbitron', monospace; margin: 0;">{title}</h3>
        <p style="color: #e0e6ed; margin: 10px 0 0 0;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load backend
    backend = load_backend()
    
    if backend["status"] == "error":
        st.error(f"❌ System Error: {backend['error']}")
        st.stop()
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0d1321, #1a2332, #0d1321);
                border: 1px solid #2a3a50; border-top: 3px solid #00d4ff;
                padding: 20px 30px; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem;">🔍 TRACEBACK COMMAND CENTER</h1>
        <p style="color: #8892a0; margin: 5px 0 0 0; font-family: 'Rajdhani', sans-serif;
                  letter-spacing: 2px; font-size: 0.9rem;">
            MISSING CHILD RECOVERY SYSTEM • SESSION ISOLATED
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.markdown("##  System Control")
        st.markdown("---")
        
        # Database Status
        st.markdown("###  Database")
        if backend["qdrant"]:
            stats = get_db_stats(backend["qdrant"])
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Vectors", stats.get("total_vectors", 0))
            with col2:
                st.markdown("**🟢 Online**")
        
        st.markdown("---")
        
        # Session Info
        st.markdown("###  Current Session")
        st.code(st.session_state.case_id)
        
        if st.session_state.upload_id:
            st.caption(f"Upload ID: {st.session_state.upload_id[:8]}...")
            st.success("✅ Video processed")
        else:
            st.warning(" No video processed yet")
        
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.case_id = f"TB-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            st.session_state.upload_id = None
            st.session_state.results = None
            st.session_state.summary = None
            st.session_state.child_info = {}
            st.session_state.video_processed = False
        
        st.markdown("---")
        
        # VIDEO UPLOAD SECTION
        st.markdown("###  Upload CCTV")
        
        video_file = st.file_uploader(
            "Select Video",
            type=['mp4', 'avi', 'mov'],
            key="video_upload"
        )
        
        if video_file:
            if st.button("⚡ Process Video", use_container_width=True):
                import config
                from src.processor import CCTVProcessor
                
                # Generate NEW upload_id for this session
                new_upload_id = generate_upload_id()
                
                # Save video
                video_path = config.UPLOADS_DIR / video_file.name
                with open(video_path, 'wb') as f:
                    f.write(video_file.read())
                
                # Process
                processor = CCTVProcessor()
                progress = st.progress(0)
                status_text = st.empty()
                
                def update(p, m):
                    progress.progress(p)
                    status_text.text(m)
                
                result = processor.process_video(
                    str(video_path),
                    upload_id=new_upload_id,  # Session isolation!
                    progress_callback=update
                )
                
                if result["status"] == "completed":
                    st.session_state.upload_id = new_upload_id
                    st.session_state.video_processed = True
                    st.success(f"✅ Processed {result['vectors_uploaded']} frames!")
                    get_db_stats.clear()
                else:
                    st.error("Processing failed!")
        
        st.markdown("---")
        
        # Alternative: Load Dataset
        with st.expander("📂 Load Image Dataset"):
            dataset_path = st.text_input(
                "Folder Path",
                value="./data/market1501/bounding_box_train"
            )
            num_samples = st.slider("Samples", 50, 500, 200)
            
            if st.button("Load Dataset", use_container_width=True):
                if Path(dataset_path).exists():
                    from src.processor import CCTVProcessor
                    
                    new_upload_id = generate_upload_id()
                    processor = CCTVProcessor()
                    progress = st.progress(0)
                    
                    result = processor.process_images_folder(
                        dataset_path,
                        upload_id=new_upload_id,
                        max_images=num_samples,
                        progress_callback=lambda p, m: progress.progress(p)
                    )
                    
                    if result["status"] == "completed":
                        st.session_state.upload_id = new_upload_id
                        st.session_state.video_processed = True
                        st.success(f" Loaded {result['uploaded']} images!")
                        get_db_stats.clear()
                else:
                    st.error("Path not found!")
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
   # --- MAIN CONTENT ---
    col_left, col_right = st.columns([1, 2])
    
    # LEFT: SEARCH PANEL
    with col_left:
        st.markdown("###  Target Acquisition")
        
        if not st.session_state.upload_id:
            st.warning(" Upload & Process CCTV Video First")
            st.stop()
            
        # TABS FOR SEARCH MODE
        search_tab1, search_tab2 = st.tabs([" Image Search", " Text Search"])
        
        # --- TAB 1: IMAGE SEARCH ---
        with search_tab1:
            with st.form("image_search_form"):
                uploaded_photo = st.file_uploader("Reference Photo", type=['jpg', 'png'])
                if uploaded_photo:
                    st.image(uploaded_photo, caption="Reference", use_column_width=True)
                
                name_img = st.text_input("Name", key="name_img")
                age_img = st.text_input("Age", key="age_img")
                conf_img = st.slider("Min Confidence", 30, 95, 60, key="conf_img")
                
                btn_img_search = st.form_submit_button(" Search by Image", use_container_width=True)

            if btn_img_search and uploaded_photo:
                import config
                from src.search import SearchEngine
                
                ref_path = config.UPLOADS_DIR / f"ref_{st.session_state.case_id}.jpg"
                img = Image.open(uploaded_photo)
                img.save(ref_path)
                
                st.session_state.child_info = {"name": name_img, "age": age_img, "photo_path": str(ref_path)}
                
                with st.spinner("Analyzing vectors..."):
                    engine = SearchEngine()
                    results = engine.search_by_image(
                        img, 
                        upload_id=st.session_state.upload_id,
                        min_confidence=conf_img/100
                    )
                    st.session_state.results = results
                    st.session_state.summary = engine.get_summary(results)

        # --- TAB 2: TEXT SEARCH ---
        with search_tab2:
            st.markdown("Search by description (e.g., 'Boy in red t-shirt', 'Girl with backpack')")
            with st.form("text_search_form"):
                text_query = st.text_input("Description")
                conf_txt = st.slider("Min Confidence", 20, 90, 30, key="conf_txt") # Text usually needs lower conf
                
                btn_txt_search = st.form_submit_button(" Search by Text", use_container_width=True)
            
            if btn_txt_search and text_query:
                from src.search import SearchEngine
                
                st.session_state.child_info = {"name": "Unknown", "age": "Unknown", "photo_path": None}
                
                with st.spinner("Analyzing text embeddings..."):
                    engine = SearchEngine()
                    results = engine.search_by_text(
                        text_query, 
                        upload_id=st.session_state.upload_id,
                        min_confidence=conf_txt/100
                    )
                    st.session_state.results = results
                    st.session_state.summary = engine.get_summary(results)

    # RIGHT: RESULTS PANEL
    # RIGHT: RESULTS PANEL
    with col_right:
        # CHECK IF A SEARCH HAS BEEN PERFORMED (Even if 0 matches found)
        if st.session_state.results is not None:
            results = st.session_state.results
            summary = st.session_state.summary
            
            # --- CASE 1: MATCHES FOUND ---
            if results:
                # Alert Banner
                has_red = summary.get("has_red_alert", False)
                color = "#ff3a3a" if has_red else "#00ff88"
                title = "HIGH CONFIDENCE MATCH" if has_red else "MATCHES FOUND"
                
                st.markdown(f"""
                <div style="border: 2px solid {color}; padding: 15px; background: rgba(0,0,0,0.3); margin-bottom: 20px;">
                    <h2 style="color: {color} !important; margin:0;">{title}</h2>
                    <p style="margin:5px 0 0 0;">Found {len(results)} matches in session</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Results Grid
                cols = st.columns(3)
                for i, match in enumerate(results[:12]):
                    with cols[i % 3]:
                        st.image(safe_image(match.get('frame_path')), use_column_width=True)
                        
                        # Display Video Time
                        timestamp = match.get('timestamp', '00:00:00')
                        conf = match.get('confidence_pct', 0)
                        
                        st.markdown(f"""
                        <div style="text-align: center; background: #1a2332; padding: 5px; margin-bottom: 15px;">
                            <div style="color: #00d4ff; font-weight: bold; font-size: 1.1rem;">{timestamp}</div>
                            <div style="color: #8892a0; font-size: 0.9rem;">Confidence: {conf}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                # PDF Report
                st.markdown("---")
                if st.button(" Generate Report", use_container_width=True):
                    from src.report import ReportGenerator
                    gen = ReportGenerator()
                    path = gen.generate(
                        case_id=st.session_state.case_id,
                        child_info=st.session_state.child_info,
                        matches=results,
                        summary=summary
                    )
                    with open(path, "rb") as f:
                        st.download_button(" Download PDF", f, file_name="Report.pdf", mime="application/pdf")
            
            # --- CASE 2: NO MATCHES FOUND ---
            else:
                st.markdown("""
                <div style="text-align: center; padding: 40px; border: 2px dashed #ff3a3a; background: rgba(255, 58, 58, 0.1);">
                    <div style="font-size: 3rem; margin-bottom: 10px;">❌</div>
                    <h3 style="color: #ff3a3a;">NO MATCHES FOUND</h3>
                    <p style="color: #e0e6ed;">Try lowering the confidence threshold or using a different description.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # --- CASE 3: NO SEARCH PERFORMED YET ---
        else:
            st.markdown("""
            <div style="text-align: center; padding: 50px; opacity: 0.5;">
                <div style="font-size: 3rem;">🔍</div>
                <h3>Awaiting Search</h3>
                <p>Use the panel on the left to search by Image or Text</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
