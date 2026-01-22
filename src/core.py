"""
TraceBack Core Engine
=====================
Handles: Video Processing, Vector Search, Analytics, Timeline Generation
Optimized for stability and speed during live demos.
"""

import os
import cv2
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import hashlib
import time

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, 
    Filter, FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    COLLECTION_NAME = "traceback_persons"
    VECTOR_DIM = 512
    BATCH_SIZE = 50  # Prevents Qdrant timeout
    FRAME_SKIP = 15  # Process every 15th frame (2 FPS from 30 FPS video)
    MIN_PERSON_SIZE = (50, 100)  # Minimum detection size
    CONFIDENCE_THRESHOLD = 0.65
    RED_ALERT_THRESHOLD = 0.85
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    FRAMES_DIR = BASE_DIR / "frames"
    REPORTS_DIR = BASE_DIR / "reports"
    ASSETS_DIR = BASE_DIR / "assets"
    UPLOADS_DIR = BASE_DIR / "uploads"
    
    @classmethod
    def ensure_dirs(cls):
        for d in [cls.FRAMES_DIR, cls.REPORTS_DIR, cls.UPLOADS_DIR]:
            d.mkdir(exist_ok=True)

Config.ensure_dirs()

# ============================================================================
# VECTOR ENGINE (Singleton Pattern for Demo Stability)
# ============================================================================

class VectorEngine:
    _instance = None
    _model = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize model and Qdrant - called only once"""
        print(" Initializing TraceBack Vector Engine...")
        
        # Load CLIP model (best for person re-identification)
        self._model = SentenceTransformer('clip-ViT-B-32')
        
        # Initialize Qdrant (in-memory for demo - no external DB needed!)
        self._client = QdrantClient(":memory:")
        
        # Create collection if not exists
        try:
            self._client.get_collection(Config.COLLECTION_NAME)
        except:
            self._client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=Config.VECTOR_DIM,
                    distance=Distance.COSINE
                )
            )
        print(" Vector Engine Ready!")
    
    @property
    def model(self):
        return self._model
    
    @property
    def client(self):
        return self._client
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image to vector using CLIP"""
        # Resize for consistent encoding
        image = image.convert('RGB').resize((224, 224))
        vector = self._model.encode(image)
        return vector / np.linalg.norm(vector)  # Normalize
    
    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Batch encode for efficiency"""
        processed = [img.convert('RGB').resize((224, 224)) for img in images]
        vectors = self._model.encode(processed)
        # Normalize each vector
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

# ============================================================================
# CCTV PROCESSOR (Robust Batch Processing)
# ============================================================================

class CCTVProcessor:
    def __init__(self):
        self.engine = VectorEngine()
        self.camera_locations = self._load_camera_locations()
    
    def _load_camera_locations(self) -> Dict:
        """Load simulated camera locations"""
        loc_file = Config.ASSETS_DIR / "camera_locations.json"
        if loc_file.exists():
            with open(loc_file) as f:
                return json.load(f)
        return {}
    
    def _generate_frame_id(self, video_path: str, frame_num: int) -> str:
        """Generate unique ID for frame"""
        base = f"{video_path}_{frame_num}"
        return hashlib.md5(base.encode()).hexdigest()[:16]
    
    def _extract_person_regions(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple]]:
        """
        Simple person detection using edge detection + contours
        (No heavy ML model needed - works for demo!)
        """
        persons = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = frame.shape[:2]
        min_w, min_h = Config.MIN_PERSON_SIZE
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (person-like: tall rectangles)
            aspect = ch / max(cw, 1)
            if 1.5 < aspect < 4.0 and cw > min_w and ch > min_h:
                # Extract region with padding
                pad = 10
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(w, x + cw + pad), min(h, y + ch + pad)
                
                person_img = frame[y1:y2, x1:x2]
                persons.append((person_img, (x1, y1, x2, y2)))
        
        # Fallback: if no persons detected, use center crop
        if not persons:
            cx, cy = w // 2, h // 2
            crop_w, crop_h = w // 3, h // 2
            x1, y1 = cx - crop_w // 2, cy - crop_h // 2
            x2, y2 = cx + crop_w // 2, cy + crop_h // 2
            persons.append((frame[y1:y2, x1:x2], (x1, y1, x2, y2)))
        
        return persons[:5]  # Max 5 detections per frame
    
    def process_video(
        self, 
        video_path: str, 
        camera_id: str = "CAM_GATE_A",
        progress_callback=None
    ) -> Dict:
        """
        Process CCTV video with robust batch uploading
        Returns processing statistics
        """
        stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "persons_detected": 0,
            "vectors_uploaded": 0,
            "processing_time": 0,
            "errors": []
        }
        
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            stats["errors"].append("Failed to open video file")
            return stats
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stats["total_frames"] = total_frames
        
        # Batch accumulator
        batch_points = []
        frame_num = 0
        
        # Simulated timestamp (start from "now")
        base_time = datetime.now()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Skip frames for efficiency
            if frame_num % Config.FRAME_SKIP != 0:
                continue
            
            stats["processed_frames"] += 1
            
            # Calculate simulated timestamp
            elapsed_seconds = frame_num / fps
            timestamp = base_time + timedelta(seconds=elapsed_seconds)
            
            try:
                # Extract person regions
                persons = self._extract_person_regions(frame)
                
                for idx, (person_img, bbox) in enumerate(persons):
                    stats["persons_detected"] += 1
                    
                    # Convert to PIL
                    pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                    
                    # Encode to vector
                    vector = self.engine.encode_image(pil_img)
                    
                    # Save frame thumbnail
                    point_id = self._generate_frame_id(video_path, frame_num * 100 + idx)
                    frame_path = Config.FRAMES_DIR / f"{point_id}.jpg"
                    pil_img.save(frame_path, quality=85)
                    
                    # Create point
                    point = PointStruct(
                        id=abs(hash(point_id)) % (10**12),  # Numeric ID
                        vector=vector.tolist(),
                        payload={
                            "frame_num": frame_num,
                            "timestamp": timestamp.isoformat(),
                            "camera_id": camera_id,
                            "camera_name": self.camera_locations.get(camera_id, {}).get("name", camera_id),
                            "bbox": list(bbox),
                            "frame_path": str(frame_path),
                            "video_source": os.path.basename(video_path),
                            "zone": self.camera_locations.get(camera_id, {}).get("zone", "Unknown"),
                            "lat": self.camera_locations.get(camera_id, {}).get("lat", 0),
                            "lon": self.camera_locations.get(camera_id, {}).get("lon", 0)
                        }
                    )
                    batch_points.append(point)
                    
                    # Upload batch when full (PREVENTS TIMEOUT!)
                    if len(batch_points) >= Config.BATCH_SIZE:
                        self._upload_batch(batch_points)
                        stats["vectors_uploaded"] += len(batch_points)
                        batch_points = []
                
                # Progress callback
                if progress_callback:
                    progress = frame_num / total_frames
                    progress_callback(progress, f"Processing frame {frame_num}/{total_frames}")
                    
            except Exception as e:
                stats["errors"].append(f"Frame {frame_num}: {str(e)}")
                continue
        
        cap.release()
        
        # Upload remaining batch
        if batch_points:
            self._upload_batch(batch_points)
            stats["vectors_uploaded"] += len(batch_points)
        
        stats["processing_time"] = round(time.time() - start_time, 2)
        return stats
    
    def _upload_batch(self, points: List[PointStruct], max_retries: int = 3):
        """Upload batch with retry logic for stability"""
        for attempt in range(max_retries):
            try:
                self.engine.client.upsert(
                    collection_name=Config.COLLECTION_NAME,
                    points=points,
                    wait=True
                )
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief pause before retry
                else:
                    raise e
        return False

# ============================================================================
# SEARCH ENGINE
# ============================================================================

class SearchEngine:
    def __init__(self):
        self.engine = VectorEngine()
        self.camera_locations = self._load_camera_locations()
    
    def _load_camera_locations(self) -> Dict:
        loc_file = Config.ASSETS_DIR / "camera_locations.json"
        if loc_file.exists():
            with open(loc_file) as f:
                return json.load(f)
        return {}
    
    def search_person(
        self, 
        query_image: Image.Image, 
        top_k: int = 10,
        min_score: float = 0.5
    ) -> List[Dict]:
        """
        Search for matching persons in CCTV database
        Returns ranked matches with all metadata
        """
        # Encode query image
        query_vector = self.engine.encode_image(query_image)
        
        # Search Qdrant
        results = self.engine.client.search(
            collection_name=Config.COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=top_k,
            score_threshold=min_score
        )
        
        # Format results
        matches = []
        for r in results:
            match = {
                "id": r.id,
                "score": round(r.score, 4),
                "confidence_pct": round(r.score * 100, 1),
                "is_red_alert": r.score >= Config.RED_ALERT_THRESHOLD,
                "timestamp": r.payload.get("timestamp"),
                "camera_id": r.payload.get("camera_id"),
                "camera_name": r.payload.get("camera_name"),
                "zone": r.payload.get("zone"),
                "frame_path": r.payload.get("frame_path"),
                "video_source": r.payload.get("video_source"),
                "lat": r.payload.get("lat", 0),
                "lon": r.payload.get("lon", 0),
                "bbox": r.payload.get("bbox")
            }
            matches.append(match)
        
        return matches
    
    def get_collection_stats(self) -> Dict:
        """Get database statistics"""
        try:
            info = self.engine.client.get_collection(Config.COLLECTION_NAME)
            return {
                "total_vectors": info.points_count,
                "status": "🟢 Online",
                "collection": Config.COLLECTION_NAME
            }
        except:
            return {
                "total_vectors": 0,
                "status": "🔴 Offline",
                "collection": Config.COLLECTION_NAME
            }

# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    @staticmethod
    def generate_timeline(matches: List[Dict]) -> List[Dict]:
        """
        Generate investigation timeline from matches
        Sorted chronologically with movement analysis
        """
        if not matches:
            return []
        
        # Sort by timestamp
        sorted_matches = sorted(
            matches, 
            key=lambda x: x.get("timestamp", "")
        )
        
        timeline = []
        prev_time = None
        prev_location = None
        
        for i, match in enumerate(sorted_matches):
            ts = match.get("timestamp")
            if ts:
                current_time = datetime.fromisoformat(ts)
            else:
                current_time = datetime.now()
            
            event = {
                "sequence": i + 1,
                "timestamp": current_time.strftime("%H:%M:%S"),
                "date": current_time.strftime("%Y-%m-%d"),
                "location": match.get("camera_name", "Unknown"),
                "zone": match.get("zone", "Unknown"),
                "confidence": match.get("confidence_pct", 0),
                "is_red_alert": match.get("is_red_alert", False),
                "frame_path": match.get("frame_path"),
                "lat": match.get("lat", 0),
                "lon": match.get("lon", 0),
                "time_gap": None,
                "movement": None
            }
            
            # Calculate time gap from previous sighting
            if prev_time:
                gap = (current_time - prev_time).total_seconds()
                if gap < 60:
                    event["time_gap"] = f"{int(gap)} seconds"
                elif gap < 3600:
                    event["time_gap"] = f"{int(gap/60)} minutes"
                else:
                    event["time_gap"] = f"{round(gap/3600, 1)} hours"
            
            # Detect movement
            current_location = match.get("camera_id")
            if prev_location and current_location != prev_location:
                event["movement"] = f"Moved from {prev_location} → {current_location}"
            
            timeline.append(event)
            prev_time = current_time
            prev_location = current_location
        
        return timeline
    
    @staticmethod
    def get_summary(matches: List[Dict], timeline: List[Dict]) -> Dict:
        """Generate case summary statistics"""
        if not matches:
            return {
                "status": "NO_MATCHES",
                "highest_confidence": 0,
                "total_sightings": 0,
                "cameras_detected": 0,
                "first_seen": None,
                "last_seen": None,
                "total_duration": None,
                "alert_level": "GREEN"
            }
        
        confidences = [m.get("confidence_pct", 0) for m in matches]
        cameras = list(set(m.get("camera_id") for m in matches if m.get("camera_id")))
        
        # Determine alert level
        max_conf = max(confidences)
        if max_conf >= 85:
            alert_level = "RED"
        elif max_conf >= 70:
            alert_level = "ORANGE"
        elif max_conf >= 50:
            alert_level = "YELLOW"
        else:
            alert_level = "GREEN"
        
        first_event = timeline[0] if timeline else {}
        last_event = timeline[-1] if timeline else {}
        
        return {
            "status": "MATCH_FOUND",
            "highest_confidence": max_conf,
            "average_confidence": round(sum(confidences) / len(confidences), 1),
            "total_sightings": len(matches),
            "cameras_detected": len(cameras),
            "camera_list": cameras,
            "first_seen": f"{first_event.get('date', '')} {first_event.get('timestamp', '')}",
            "first_location": first_event.get("location", "Unknown"),
            "last_seen": f"{last_event.get('date', '')} {last_event.get('timestamp', '')}",
            "last_location": last_event.get("location", "Unknown"),
            "alert_level": alert_level,
            "red_alert_count": sum(1 for m in matches if m.get("is_red_alert"))
        }