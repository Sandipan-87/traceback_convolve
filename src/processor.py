"""
CCTV Video Processor
====================
"""

import os
import cv2
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Tuple, Callable, Optional
from pathlib import Path
from ultralytics import YOLO

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import PointStruct

import sys
sys.path.append('..')
import config
from .database import get_qdrant


class EmbeddingModel:
    """Singleton for CLIP model."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        print("Loading CLIP model...")
        self._model = SentenceTransformer('clip-ViT-B-32')
        print("Model loaded!")
    
    def encode(self, image: Image.Image) -> np.ndarray:
        """Encode single image to vector."""
        image = image.convert('RGB').resize((224, 224))
        vector = self._model.encode(image)
        return vector / np.linalg.norm(vector)

    # --- THIS WAS MISSING ---
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text description to vector."""
        vector = self._model.encode(text)
        return vector / np.linalg.norm(vector)
    # ------------------------
    
    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Encode multiple images efficiently."""
        processed = [img.convert('RGB').resize((224, 224)) for img in images]
        vectors = self._model.encode(processed)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


class PersonDetector:
    """AI-Powered Person Detection using YOLOv8."""
    
    def __init__(self):
        # Load the "Nano" model (smallest & fastest for CPU)
        print("Loading YOLOv8-Nano model...")
        self.model = YOLO("yolov8n.pt") 
        # It will auto-download 'yolov8n.pt' (6MB) on first run
        
    def detect(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Detect ONLY humans in the frame."""
        persons = []
        h, w = frame.shape[:2]
        
        # Run inference (classes=0 means 'person' only)
        results = self.model(frame, classes=[0], verbose=False, conf=0.4)
        
        for result in results:
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Check size constraints
                cw, ch = x2 - x1, y2 - y1
                if (config.MIN_PERSON_WIDTH < cw and config.MIN_PERSON_HEIGHT < ch):
                    
                    person_crop = frame[y1:y2, x1:x2]
                    persons.append((person_crop, (x1, y1, x2, y2)))
                    
        return persons[:5] # Limit to 5 detections to keep speed high


class CCTVProcessor:
    """
    Main CCTV processing pipeline.
    """
    
    def __init__(self):
        self.qdrant = get_qdrant()
        self.model = EmbeddingModel()
        self.detector = PersonDetector()
    
    def _generate_point_id(self, upload_id: str, frame_num: int, detection_idx: int) -> int:
        """Generate unique numeric ID for Qdrant point."""
        unique_str = f"{upload_id}_{frame_num}_{detection_idx}_{time.time()}"
        hash_hex = hashlib.md5(unique_str.encode()).hexdigest()
        return int(hash_hex[:15], 16)
    
    def process_video(
        self,
        video_path: str,
        upload_id: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Process video using EXACT internal video timestamps."""
        video_filename = os.path.basename(video_path)
        
        stats = {
            "video_filename": video_filename,
            "upload_id": upload_id,
            "total_frames": 0,
            "processed_frames": 0,
            "persons_detected": 0,
            "vectors_uploaded": 0,
            "processing_time_sec": 0,
            "errors": [],
            "status": "processing"
        }
        
        start_proc_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            stats["status"] = "error"
            stats["errors"].append("Failed to open video file")
            return stats
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stats["total_frames"] = total_frames
        
        points_batch = []
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- EXACT TIMESTAMP FIX ---
            # Get current position in milliseconds directly from the file
            current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            seconds_elapsed = current_ms / 1000.0
            
            frame_num += 1
            
            # Skip frames
            if frame_num % config.FRAME_SKIP != 0:
                continue
            
            stats["processed_frames"] += 1
            timestamp_formatted = self._format_timestamp(seconds_elapsed)
            # ---------------------------
            
            try:
                detections = self.detector.detect(frame)
                
                for det_idx, (person_crop, bbox) in enumerate(detections):
                    stats["persons_detected"] += 1
                    
                    pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                    vector = self.model.encode(pil_image)
                    
                    point_id = self._generate_point_id(upload_id, frame_num, det_idx)
                    frame_path = config.FRAMES_DIR / f"{point_id}.jpg"
                    pil_image.save(frame_path, quality=85)
                    
                    point = PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
                            "upload_id": upload_id,
                            "video_filename": video_filename,
                            "frame_num": frame_num,
                            "timestamp": timestamp_formatted, 
                            "timestamp_seconds": seconds_elapsed,
                            "frame_path": str(frame_path),
                            "bbox": list(bbox),
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    points_batch.append(point)
                    
                    if len(points_batch) >= config.BATCH_SIZE:
                        self.qdrant.upload_batch(points_batch)
                        stats["vectors_uploaded"] += len(points_batch)
                        points_batch = []
                
                if progress_callback:
                    progress_callback(
                        frame_num / total_frames, 
                        f"Processing {timestamp_formatted} | Found: {stats['persons_detected']}"
                    )
                    
            except Exception as e:
                stats["errors"].append(f"Frame {frame_num}: {str(e)}")
        
        cap.release()
        
        if points_batch:
            self.qdrant.upload_batch(points_batch)
            stats["vectors_uploaded"] += len(points_batch)
        
        stats["processing_time_sec"] = round(time.time() - start_proc_time, 2)
        stats["status"] = "completed"
        return stats
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def process_images_folder(
        self,
        folder_path: str,
        upload_id: str,
        max_images: int = 500,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Process a folder of images."""
        stats = {
            "upload_id": upload_id,
            "total_images": 0,
            "processed": 0,
            "uploaded": 0,
            "errors": [],
            "status": "processing"
        }
        
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = [
            f for f in Path(folder_path).iterdir()
            if f.suffix.lower() in image_extensions
        ][:max_images]
        
        stats["total_images"] = len(image_files)
        
        if not image_files:
            stats["status"] = "error"
            stats["errors"].append("No images found in folder")
            return stats
        
        points_batch = []
        
        for idx, img_path in enumerate(image_files):
            try:
                pil_image = Image.open(img_path).convert('RGB')
                vector = self.model.encode(pil_image)
                
                timestamp_seconds = idx * 2
                timestamp_formatted = self._format_timestamp(timestamp_seconds)
                
                point_id = self._generate_point_id(upload_id, idx, 0)
                frame_path = config.FRAMES_DIR / f"{point_id}.jpg"
                pil_image.save(frame_path, quality=85)
                
                point = PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "upload_id": upload_id,
                        "video_filename": "dataset_images",
                        "frame_num": idx,
                        "timestamp": timestamp_formatted,
                        "timestamp_seconds": timestamp_seconds,
                        "frame_path": str(frame_path),
                        "original_file": img_path.name,
                        "processed_at": datetime.now().isoformat()
                    }
                )
                points_batch.append(point)
                stats["processed"] += 1
                
                # Batch upload
                if len(points_batch) >= config.BATCH_SIZE:
                    self.qdrant.upload_batch(points_batch)
                    stats["uploaded"] += len(points_batch)
                    points_batch = []
                
                if progress_callback:
                    progress_callback(
                        (idx + 1) / len(image_files),
                        f"Processing {idx + 1}/{len(image_files)}"
                    )
                    
            except Exception as e:
                stats["errors"].append(f"{img_path.name}: {str(e)}")
        
        # Final batch
        if points_batch:
            self.qdrant.upload_batch(points_batch)
            stats["uploaded"] += len(points_batch)
        
        stats["status"] = "completed"
        return stats