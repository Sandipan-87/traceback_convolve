"""
TraceBack Search Engine
=======================
"""

from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
from PIL import Image

import sys
sys.path.append('..')
import config
from .database import get_qdrant
from .processor import EmbeddingModel


class SearchEngine:
    
    def __init__(self):
        self.qdrant = get_qdrant()
        self.model = EmbeddingModel()
    
    def search_by_image(
        self,
        query_image: Image.Image,
        upload_id: str,
        top_k: int = 100,
        min_confidence: float = 0.4,
        dedup_window_seconds: float = 5.0
    ) -> List[Dict]:
        """Search using an image."""
        query_vector = self.model.encode(query_image)
        return self._execute_search(query_vector, upload_id, top_k, min_confidence, dedup_window_seconds)

    def search_by_text(
        self,
        query_text: str,
        upload_id: str,
        top_k: int = 100,
        min_confidence: float = 0.3, # Text search usually needs lower threshold
        dedup_window_seconds: float = 5.0
    ) -> List[Dict]:
        """Search using a text description."""
        query_vector = self.model.encode_text(query_text)
        return self._execute_search(query_vector, upload_id, top_k, min_confidence, dedup_window_seconds)

    def _execute_search(self, query_vector, upload_id, top_k, min_confidence, dedup_window):
        """Internal helper to run search and dedup."""
        raw_matches = self.qdrant.search_with_filter(
            query_vector=query_vector.tolist(),
            upload_id=upload_id,
            top_k=top_k,
            min_score=min_confidence
        )
        
        if not raw_matches:
            return []
        
        return self._deduplicate_by_time(raw_matches, window_seconds=dedup_window)
    
    def _deduplicate_by_time(self, matches: List[Dict], window_seconds: float = 5.0) -> List[Dict]:
        """Group matches by time window."""
        if not matches:
            return []
        
        sorted_matches = sorted(
            matches, 
            key=lambda x: x.get("timestamp_seconds", 0)
        )
        
        deduped = []
        current_window_start = None
        current_window_best = None
        
        for match in sorted_matches:
            ts = match.get("timestamp_seconds", 0)
            
            if current_window_start is None:
                current_window_start = ts
                current_window_best = match
            elif ts - current_window_start <= window_seconds:
                if match.get("confidence_pct", 0) > current_window_best.get("confidence_pct", 0):
                    current_window_best = match
            else:
                deduped.append(current_window_best)
                current_window_start = ts
                current_window_best = match
        
        if current_window_best is not None:
            deduped.append(current_window_best)
        
        deduped.sort(key=lambda x: x.get("confidence_pct", 0), reverse=True)
        return deduped
    
    def get_summary(self, matches: List[Dict]) -> Dict:
        """Generate simple summary statistics."""
        if not matches:
            return {"status": "NO_MATCHES", "total_matches": 0, "highest_confidence": 0, "has_red_alert": False}
        
        confidences = [m.get("confidence_pct", 0) for m in matches]
        max_conf = max(confidences)
        
        return {
            "status": "MATCHES_FOUND",
            "total_matches": len(matches),
            "highest_confidence": max_conf,
            "average_confidence": round(sum(confidences) / len(confidences), 1),
            "has_red_alert": max_conf >= 85,
            "red_alert_count": sum(1 for c in confidences if c >= 85)
        }