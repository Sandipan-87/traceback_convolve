# TraceBack: Privacy-First AI Forensics for Missing Child Recovery
### Submission for Convolve 4.0 (Pan-IIT Hackathon)

> **The Problem:** In child abduction cases, the "Golden Hour" is critical. Manual review of CCTV footage is slow, error-prone, and privacy-invasive.
>
> **The Solution:** TraceBack uses Multimodal AI (Text/Image Search) and **Qdrant Vector Search** to scan hours of footage in minutes. It utilizes **Session Isolation** to ensure strict data privacy for every investigation.

---

## Key Features
* **Multimodal Search:** Find subjects using a description ("Child in red shirt") or a reference photo.
* **Privacy-First Architecture:** Uses Qdrant Payload Filtering (`upload_id`) to isolate every investigation session.
* **Forensic Precision:** Extracts millisecond-accurate timestamps (`CAP_PROP_POS_MSEC`) for legal admissibility.
* **Temporal Analytics:** "Sighting Event" de-duplication reduces hundreds of frames into a clean timeline.
* **Automated Reporting:** Generates PDF forensic reports with evidence logs.

---

## Tech Stack
* **Frontend:** Streamlit (`app.py`)
* **Core Logic:** Python (`src/core.py`, `src/processor.py`)
* **AI Engine:** YOLOv8 (Person Detection) + CLIP (Embeddings)
* **Vector Database:** Qdrant Cloud
* **Reporting:** PDF Generation (`src/report.py`)

---

## Installation & Setup Guide

### Prerequisites
* Python 3.11.9 (Using latest python version gives heavy dependency error)
* Git
* A Qdrant Cloud API Key

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/TraceBack.git](https://github.com/YOUR_USERNAME/TraceBack.git)
cd TraceBack
