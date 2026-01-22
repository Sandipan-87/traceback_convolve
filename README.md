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

# Installation & Setup Guide

### Prerequisites
* Python 3.11.9 (Recommended to avoid dependency error)
* Git
* A Qdrant Cloud API Key

## 1. Clone the Repository
```bash
git clone https://github.com/Sandipan-87/traceback_convolve.git
cd traceback_convolve
```

## 2. Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.
Windows:

```bash

python -m venv venv
.\venv\Scripts\activate
```
Mac/Linux:

```bash

python3 -m venv venvsource venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Troubleshooting:** If you encounter torch/torchaudio version conflicts, run:

```bash
pip install --upgrade torch torchvision torchaudio ultralytics
```


## 4. Initialize System Folders

The application requires specific directories to store processed data. Run this command to create them:

**Windows (PowerShell):**

```powershell
md frames, reports, uploads, data
```

**Mac/Linux:**

```bash
mkdir frames reports uploads data
```
## 5. Configure API Secrets:
Create a file named .streamlit/secrets.toml in the root directory and add your Qdrant credentials:

```Ini, TOML
[qdrant]
url = "https://your-cluster-url.qdrant.tech"
api_key = "your-secret-api-key"
```
## 6. (Optional) Batch Data Setup

To use the "Batch Processing" feature (indexing a folder of images):

- Place your raw images into the `data/` folder.
- The application will automatically detect these images for batch indexing.


## Usage Guide

### 1. Run the App:

```bash
streamlit run app.py
```


### 2. Upload Video:

Drag and drop CCTV footage (MP4/AVI) into the sidebar.

### 3. Search:

- **Text Mode:** Type "Person with blue backpack".
- **Image Mode:** Upload a photo of the missing person.


### 4. Export:

Download the Forensic PDF Report from the "Results" tab.

