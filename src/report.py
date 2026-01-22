"""
TraceBack Police Report Generator
==================================
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fpdf import FPDF
from PIL import Image

import sys
sys.path.append('..')
import config


class ReportGenerator:
    """
    Generates simplified PDF police reports.
    Contains: Header + Visual Evidence Grid
    """
    
    def __init__(self):
        self.reports_dir = config.REPORTS_DIR
    
    def generate(
        self,
        case_id: str,
        child_info: Dict,
        matches: List[Dict],
        summary: Dict,
        officer_name: str = "TraceBack System"
    ) -> str:
        """
        Generate simplified PDF report.
        
        Returns:
            Path to generated PDF file
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # ===== PAGE 1: HEADER + INFO =====
        pdf.add_page()
        self._add_header(pdf, case_id, child_info, summary)
        
        # ===== EVIDENCE SECTION =====
        if matches:
            self._add_evidence_grid(pdf, matches)
        
        # ===== FOOTER =====
        self._add_footer(pdf, officer_name)
        
        # Save
        filename = f"TraceBack_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = self.reports_dir / filename
        pdf.output(str(output_path))
        
        return str(output_path)
    
    def _add_header(self, pdf: FPDF, case_id: str, child_info: Dict, summary: Dict):
        """Add report header with case info."""
        
        # Alert banner
        has_alert = summary.get("has_red_alert", False)
        if has_alert:
            pdf.set_fill_color(220, 53, 69)  # Red
            alert_text = "HIGH-CONFIDENCE MATCH DETECTED"
        else:
            pdf.set_fill_color(40, 167, 69)  # Green
            alert_text = "SEARCH COMPLETE"
        
        pdf.rect(0, 0, 210, 35, 'F')
        
        # Title
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(255, 255, 255)
        pdf.set_y(8)
        pdf.cell(0, 10, "TRACEBACK - MISSING CHILD REPORT", ln=True, align="C")
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, alert_text, ln=True, align="C")
        
        # Reset
        pdf.set_text_color(0, 0, 0)
        pdf.set_y(45)
        
        # Case Info Box
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, f"  CASE ID: {case_id}", ln=True, fill=True)
        pdf.ln(5)
        
        # Two columns: Photo + Details
        start_y = pdf.get_y()
        
        # Photo (left)
        photo_path = child_info.get("photo_path")
        if photo_path and os.path.exists(photo_path):
            try:
                pdf.image(photo_path, x=15, y=start_y, w=45, h=55)
            except:
                pass
        
        # Details (right)
        pdf.set_xy(70, start_y)
        
        details = [
            ("Name", child_info.get("name", "Unknown")),
            ("Age", child_info.get("age", "Unknown")),
            ("Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Total Matches", summary.get("total_matches", 0)),
            ("Highest Confidence", f"{summary.get('highest_confidence', 0)}%"),
        ]
        
        for label, value in details:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(40, 6, f"{label}:")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, str(value), ln=True)
            pdf.set_x(70)
        
        pdf.set_y(start_y + 65)
    
    def _add_evidence_grid(self, pdf: FPDF, matches: List[Dict]):
        """Add visual evidence grid."""
        
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_fill_color(52, 58, 64)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "  VISUAL EVIDENCE", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        
        # Grid settings
        img_width = 55
        img_height = 55
        cols = 3
        x_start = 15
        x_gap = 63
        
        x_pos = x_start
        y_pos = pdf.get_y()
        count = 0
        
        for match in matches[:12]:  # Max 12 images
            frame_path = match.get("frame_path")
            
            # Check if we need new row
            if count > 0 and count % cols == 0:
                y_pos += img_height + 25
                x_pos = x_start
                
                # Check if we need new page
                if y_pos > 250:
                    pdf.add_page()
                    y_pos = 20
            
            if frame_path and os.path.exists(frame_path):
                try:
                    # Add image
                    pdf.image(frame_path, x=x_pos, y=y_pos, w=img_width, h=img_height)
                    
                    # Add caption
                    conf = match.get("confidence_pct", 0)
                    timestamp = match.get("timestamp", "N/A")
                    
                    # Confidence badge
                    pdf.set_xy(x_pos, y_pos + img_height + 2)
                    
                    if conf >= 85:
                        pdf.set_fill_color(220, 53, 69)
                        pdf.set_text_color(255, 255, 255)
                    elif conf >= 70:
                        pdf.set_fill_color(255, 193, 7)
                        pdf.set_text_color(0, 0, 0)
                    else:
                        pdf.set_fill_color(23, 162, 184)
                        pdf.set_text_color(255, 255, 255)
                    
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.cell(img_width, 6, f" {conf}% ", align="C", fill=True)
                    
                    # Timestamp
                    pdf.set_text_color(100, 100, 100)
                    pdf.set_xy(x_pos, y_pos + img_height + 9)
                    pdf.set_font("Helvetica", "", 8)
                    pdf.cell(img_width, 5, f"Time: {timestamp}", align="C")
                    
                    pdf.set_text_color(0, 0, 0)
                    
                    x_pos += x_gap
                    count += 1
                    
                except Exception as e:
                    print(f"Error adding image: {e}")
    
    def _add_footer(self, pdf: FPDF, officer_name: str):
        """Add footer to all pages."""
        total_pages = pdf.page_no()
        
        for page_num in range(1, total_pages + 1):
            pdf.page = page_num
            pdf.set_y(-20)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 4, f"Generated by: {officer_name}", align="C", ln=True)
            pdf.cell(0, 4, f"Page {page_num}/{total_pages} | CONFIDENTIAL - LAW ENFORCEMENT USE ONLY", align="C")