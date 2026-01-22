"""
Qdrant Cloud Database Manager
=============================
Handles all vector database operations.
"""

import time
from typing import List, Dict, Optional
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    UpdateStatus
)

import sys
sys.path.append('..')
import config


class QdrantManager:
    """Singleton manager for Qdrant Cloud operations."""
    
    _instance = None
    _client = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not QdrantManager._initialized:
            self._connect()
            self._ensure_collection()
            QdrantManager._initialized = True
    
    def _connect(self) -> None:
        """Establish connection to Qdrant Cloud."""
        print(" Connecting to Qdrant Cloud...")
        
        try:
            self._client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
                timeout=60
            )
            
            collections = self._client.get_collections()
            print(f" Connected! Found {len(collections.collections)} collections")
            
        except Exception as e:
            print(f" Connection failed: {e}")
            raise ConnectionError(f"Failed to connect to Qdrant Cloud: {e}")
    
    def _ensure_collection(self) -> None:
        """Create collection AND payload index if they don't exist."""
        # 1. Create Collection
        try:
            self._client.get_collection(config.COLLECTION_NAME)
            print(f"Collection '{config.COLLECTION_NAME}' exists")
        except:
            print(f" Creating collection '{config.COLLECTION_NAME}'...")
            self._client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=config.VECTOR_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            print(f" Collection created!")

        # 2. CRITICAL: Create Index for 'upload_id' (This fixes the 400 Error)
        # Without this, Qdrant refuses to filter by session ID
        try:
            self._client.create_payload_index(
                collection_name=config.COLLECTION_NAME,
                field_name="upload_id",
                field_schema="keyword"
            )
            print(" Index for 'upload_id' created/verified!")
        except Exception as e:
            # It might fail if it already exists, which is fine
            print(f" Index check: {e}")
    
    @property
    def client(self) -> QdrantClient:
        return self._client
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        try:
            info = self._client.get_collection(config.COLLECTION_NAME)
            return {
                "total_vectors": info.points_count,
                "status": "🟢 Online"
            }
        except Exception as e:
            return {
                "total_vectors": 0,
                "status": f"🔴 Error: {e}"
            }
    
    def upload_batch(self, points: List[PointStruct], max_retries: int = 3) -> bool:
        """Upload a batch of points with retry logic."""
        for attempt in range(max_retries):
            try:
                result = self._client.upsert(
                    collection_name=config.COLLECTION_NAME,
                    points=points,
                    wait=True
                )
                
                if result.status == UpdateStatus.COMPLETED:
                    return True
                    
            except Exception as e:
                print(f" Upload attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                else:
                    raise
        
        return False
    
    def search_with_filter(
        self,
        query_vector: List[float],
        upload_id: str,
        top_k: int = 100,
        min_score: float = 0.4
    ) -> List[Dict]:
        """
        Search for similar vectors FILTERED by upload_id.
        This ensures session isolation.
        """
        # Build filter for session isolation
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="upload_id",
                    match=MatchValue(value=upload_id)
                )
            ]
        )
        
        # Execute search
        results = self._client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k,
            score_threshold=min_score,
            with_payload=True
        )
        
        # Format results
        matches = []
        for r in results:
            match = {
                "id": r.id,
                "score": round(r.score, 4),
                "confidence_pct": round(r.score * 100, 1),
                "is_red_alert": r.score >= config.RED_ALERT_THRESHOLD,
                "timestamp": r.payload.get("timestamp"),
                "timestamp_seconds": r.payload.get("timestamp_seconds", 0),
                "frame_path": r.payload.get("frame_path"),
                "video_filename": r.payload.get("video_filename"),
                "upload_id": r.payload.get("upload_id"),
                "frame_num": r.payload.get("frame_num")
            }
            matches.append(match)
        
        return matches
    
    def delete_by_upload_id(self, upload_id: str) -> bool:
        """Delete all vectors for a specific upload session."""
        try:
            self._client.delete(
                collection_name=config.COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="upload_id",
                                match=MatchValue(value=upload_id)
                            )
                        ]
                    )
                )
            )
            return True
        except Exception as e:
            print(f"Delete error: {e}")
            return False


def get_qdrant() -> QdrantManager:
    """Get the QdrantManager singleton instance."""
    return QdrantManager()