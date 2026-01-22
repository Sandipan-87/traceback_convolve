"""
TraceBack Analytics
===================
"""

from typing import List, Dict


class AnalyticsEngine:
    """Basic analytics without complex predictions."""
    
    @staticmethod
    def get_summary(matches: List[Dict]) -> Dict:
        """Generate simple summary statistics."""
        if not matches:
            return {
                "status": "NO_MATCHES",
                "total_matches": 0,
                "highest_confidence": 0,
                "has_red_alert": False
            }
        
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