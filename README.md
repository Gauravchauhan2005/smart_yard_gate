# AI-Based Smart Yard Gate Automation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Computer Vision](https://img.shields.io/badge/YOLO-Ultralytics%20v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![OCR](https://img.shields.io/badge/OCR-PaddleOCR-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![Database](https://img.shields.io/badge/Database-MySQL%20%7C%20SQLAlchemy-blue.svg)](https://www.mysql.com/)
[![Architecture](https://img.shields.io/badge/Design-Clean%20%26%20Modular-brightgreen.svg)]()

> A production-grade Yard Management System (YMS) gate automation solution simulating real-world logistics facility ingress/egress. Powered by Ultralytics YOLO vehicle detection, OpenCV image preprocessing, PaddleOCR text recognition, Flask REST APIs, and an interactive operational dashboard.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Directory Structure](#project-directory-structure)
- [Installation](#installation)
- [Database Setup](#database-setup)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Model Setup](#model-setup)
- [Dataset](#dataset)
- [Model Evaluation](#model-evaluation)
- [Edge AI & Deployment Considerations](#edge-ai--deployment-considerations)
- [Development Roadmap](#development-roadmap)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Limitations](#limitations)
- [Author](#author)

---

## Project Overview

In high-throughput logistics terminals, distribution centers, and freight yards, managing truck check-in and check-out is traditionally slow, labor-intensive, and prone to human transcription errors. 

**AI-Based Smart Yard Gate Automation System** is an end-to-end, production-oriented gate automation platform designed to:
- Detect trucks, trailers, and license plates as vehicles approach security gates.
- Preprocess optical captures using custom OpenCV image enhancement algorithms.
- Read license plate numbers and trailer IDs using state-of-the-art OCR (PaddleOCR with robust fallbacks).
- Correlate vehicle data against gate schedules and yard slot capacity in real time.
- Present gate personnel and terminal dispatchers with a real-time dashboard tracking gate throughput, confidence scores, and yard inventory.

---

## Problem Statement

Traditional yard gate operations face key operational bottlenecks:
1. **Manual Ingress Delays**: Gate clerks manually typing vehicle numbers introduce 2-5 minutes of dwell time per truck, creating roadway congestion.
2. **Data Inaccuracy**: Erroneous plate entry results in misplaced trailers, lost inventory, and billing disputes.
3. **Lack of Ingress Visibility**: Yard dispatchers lack instantaneous, auditable visual evidence linked to vehicle check-in timestamps.
4. **Safety & Audit Gaps**: Unverified vehicles entering logistics yards introduce security and liability risks.

---

## Objectives

- **Automated Gate Processing**: Automatically capture, detect, and extract vehicle credentials from gate camera streams.
- **Robust Multi-Stage Vision Pipeline**: Decouple vehicle detection (YOLO) from plate localization and multi-stage image preprocessing (OpenCV) for reliable OCR ingestion.
- **High-Accuracy OCR**: Leverage PaddleOCR with confidence scoring, character cleaning, and fallback support.
- **Real-Time Yard Operations**: Provide instant check-in/check-out lifecycle tracking with gate assignment and yard location management.
- **Auditable Metrics**: Surface detection confidence, OCR confidence, automated vs. manual review ratios, and evaluation benchmarks.
- **Modular Production Design**: Adhere strictly to clean architecture, the Flask Application Factory pattern, robust error handling, test coverage, and environment configuration.

---

## Features

- **Automated Vehicle Ingestion**: Accepts gate camera stills or video streams for automated processing.
- **Computer Vision Pipeline**:
  - Vehicle & trailer detection via Ultralytics YOLO.
  - Identification & license plate region extraction.
  - Image preprocessing: Resizing, grayscale, bilateral denoising, adaptive thresholding, perspective normalization, and contrast enhancement.
  - OCR extraction with character normalization and confidence filtering.
- **Yard Management Dashboard**:
  - Live KPI cards: Vehicles Today, Inside Yard, Automated Entries, Average OCR/Detection Confidence.
  - Real-time gate activity feed with audit timestamps and status flags.
- **Vehicle Directory & History**: Comprehensive vehicle profiles with check-in/out timestamps, gate IDs, and detection crops.
- **Yard Inventory Map**: Slot allocation status and search/filter by gate, vehicle type, and time.
- **Analytics & Telemetry**: Visual charts measuring throughput by hour, detection confidence distribution, and gate performance.
- **RESTful API**: Standardized JSON endpoints with validation, structured error handlers, and HTTP status codes.

---

## Architecture

```
Gate Camera / Upload (Image/Video)
               │
               ▼
   [ Flask REST API / Web UI ]
               │
               ▼
     [ Detection Service ]
               │
               ├──> 1. Ultralytics YOLO (Vehicle Detection & Localization)
               │
               ├──> 2. License Plate Localization
               │
               ├──> 3. OpenCV Preprocessing Pipeline
               │       (Grayscale -> Denoise -> Adaptive Threshold -> Sharpen)
               │
               └──> 4. PaddleOCR Engine (Text Extraction & Confidence Scoring)
               │
               ▼
     [ Vehicle Service ]
               │
               ▼
    [ SQLAlchemy ORM ] ───> [ MySQL / SQLite Database ]
               │
               ▼
[ YMS Operational Dashboard & Real-Time Telemetry ]
```

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend Framework** | Python 3.10+, Flask 3.x, Flask-SQLAlchemy, REST API |
| **Computer Vision** | OpenCV (cv2), Ultralytics YOLOv8, PyTorch |
| **OCR Engine** | PaddleOCR (with flexible OCR fallback) |
| **Data & Metrics** | NumPy, Pandas |
| **Database** | MySQL (Production), SQLAlchemy ORM, SQLite (Dev/Test fallback) |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Chart.js |
| **Testing & Quality** | PyTest, PyTest-Cov, Flake8 |
| **Configuration** | Python-Dotenv, Modular Config Classes |

---

## Project Directory Structure

```
smart-yard-gate/
│
├── app.py                      # Flask Application Factory & Server Entry Point
├── config.py                   # Environment Configurations (Dev, Test, Prod)
├── requirements.txt            # Python Dependencies
├── .env.example                # Template for Environment Variables
├── .gitignore                  # Git Ignore Rules
├── README.md                   # Project Documentation
│
├── models/
│   └── yolo/                   # YOLO Model Weights (.pt, .onnx)
│
├── data/
│   ├── raw/                    # Inbound test images / camera captures
│   ├── processed/              # Preprocessed crops & test outputs
│   └── labels/                 # Annotation ground-truth labels
│
├── detection/
│   ├── __init__.py
│   ├── detector.py             # Reusable YOLO Vehicle Detector Class
│   ├── plate_detector.py       # Plate Detection & Localization Module
│   ├── preprocessing.py        # OpenCV Image Preprocessing Suite
│   └── ocr.py                  # PaddleOCR Text Recognition Engine
│
├── database/
│   ├── __init__.py
│   ├── db.py                   # SQLAlchemy Database Instance
│   └── models.py               # ORM Models (Vehicle, Gate, YardLocation, Detection)
│
├── routes/
│   ├── __init__.py             # Blueprint Exports
│   ├── main.py                 # Core UI Navigation Routes
│   ├── gate.py                 # Gate Ingress/Egress UI Routes
│   └── api.py                  # REST API Endpoints (/api/health, /api/detect, etc.)
│
├── services/
│   ├── __init__.py
│   ├── detection_service.py    # Vision Pipeline Orchestrator
│   ├── ocr_service.py          # OCR & Post-Processing Orchestrator
│   └── vehicle_service.py      # Business Logic (Check-in, Check-out, Yard status)
│
├── templates/
│   ├── base.html               # Shared HTML Skeleton
│   ├── index.html              # Main YMS Dashboard
│   ├── gate.html               # Gate Camera & Detection Interface
│   ├── vehicles.html           # Vehicle Ingress Log
│   ├── vehicle_detail.html     # Single Vehicle Audit View
│   └── analytics.html          # Performance & Throughput Analytics
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom Dashboard Styling
│   ├── js/
│   │   └── app.js              # Client-Side Interactions & Fetch API
│   └── uploads/                # Captured and Annotated Image Storage
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py             # API Endpoint & Health Check Tests
│   ├── test_detection.py       # Vision & Preprocessing Unit Tests
│   └── test_database.py        # ORM Models & Transaction Tests
│
└── scripts/
    ├── init_db.py              # Database Schema Initialization & Seeding
    └── evaluate_model.py       # Precision, Recall, mAP & OCR Evaluation
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/smart-yard-gate.git
cd smart-yard-gate
```

### 2. Create and Activate a Python Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Database Setup

1. Copy the sample environment file:
   ```bash
   cp .env.example .env
   ```
2. Configure your MySQL credentials in `.env`:
   ```ini
   DB_USER=root
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=3306
   DB_NAME=smart_yard_db
   USE_SQLITE=False
   ```
   *(Note: For instant local testing without running a MySQL server, set `USE_SQLITE=True`)*.

3. Run the database initialization script (Scheduled for Phase 3):
   ```bash
   python scripts/init_db.py
   ```

---

## Running the Application

### Development Server
```bash
python app.py
```
Or via the Flask CLI:
```bash
flask run --host=0.0.0.0 --port=5000
```
The application will be accessible at: `http://localhost:5000`

---

## API Documentation

| Method | Endpoint | Description | Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status & environment info | **Active** (Phase 1) |
| `POST` | `/api/detect` | Run YOLO detection on uploaded frame | Scheduled (Phase 5/10) |
| `POST` | `/api/ocr` | Run OCR pipeline on cropped plate | Scheduled (Phase 8/10) |
| `POST` | `/api/gate/check-in` | Automated vehicle check-in | Scheduled (Phase 9/10) |
| `POST` | `/api/gate/check-out` | Record vehicle egress & release slot | Scheduled (Phase 9/10) |
| `GET` | `/api/vehicles` | List all tracked vehicles with filters | Scheduled (Phase 10) |
| `GET` | `/api/vehicles/<id>` | Retrieve specific vehicle details & audit | Scheduled (Phase 10) |
| `GET` | `/api/gate/activity` | Recent gate ingress/egress transactions | Scheduled (Phase 10) |
| `GET` | `/api/yard` | Current yard occupancy & slot inventory | Scheduled (Phase 10/11) |
| `GET` | `/api/analytics` | Telemetry & model performance metrics | Scheduled (Phase 10/12) |

### Health Check Example
**Request:**
```bash
curl -X GET http://localhost:5000/api/health
```
**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "AI-Based Smart Yard Gate Automation System",
  "version": "1.0.0",
  "phase": "Phase 1 - Infrastructure & Flask Application Setup",
  "environment": "development",
  "debug": true,
  "timestamp": "2026-09-29T01:30:00.000000+00:00"
}
```

---

## Model Setup

- **Vehicle Detector**: Ultralytics YOLOv8 nano (`yolov8n.pt`) is downloaded automatically on first run to `models/yolo/`. Custom trained models targeting specific truck/trailer classes can be dropped into `models/yolo/` and referenced in `.env`.
- **License Plate Detector**: Uses high-resolution localization module with customizable threshold `DETECTION_CONF_THRESHOLD`.
- **OCR Engine**: PaddleOCR English/multilingual recognition model initialized dynamically in `detection/ocr.py`.

---

## Dataset

For custom model training and evaluation:
- Raw gate camera captures: `data/raw/`
- Processed, cropped, and annotated data: `data/processed/`
- YOLO format labels: `data/labels/`

Standard yard dataset classes:
1. `0: truck_cab`
2. `1: trailer`
3. `2: license_plate`
4. `3: container_number`

---

## Model Evaluation

Model evaluation metrics are calculated using ground truth test splits:
```bash
python scripts/evaluate_model.py
```
- **Detection Metrics**: IoU, Precision, Recall, mAP@50, mAP@50:95.
- **OCR Metrics**: Character Error Rate (CER), Exact Match Ratio, Mean Confidence.
*(All reported metrics are computed directly against verified evaluation sets — no fabricated benchmark numbers).*

---

## Edge AI & Deployment Considerations

For low-latency edge deployment at physical gates (e.g. NVIDIA Jetson Orin Nano / AGX Orin / Industrial Edge PCs):
- **Optimization Pipeline**:
  `PyTorch Model (.pt) ➔ ONNX Export ➔ TensorRT Engine (.engine) ➔ Edge Inference`
- **Target Ingress Latency**: < 250ms end-to-end (Detection + Preprocessing + OCR).
- **Target Throughput**: 15–30 FPS on Jetson Orin with FP16/INT8 precision.

---

## Development Roadmap

- [x] **Phase 1**: Project structure + Virtual environment setup + Flask application + Health check API
- [ ] **Phase 2**: Professional YMS Dashboard UI & Gate navigation
- [ ] **Phase 3**: Database schema, MySQL integration & SQLAlchemy ORM models
- [ ] **Phase 4**: Secure image & video upload pipeline with validation
- [ ] **Phase 5**: Ultralytics YOLO vehicle detection integration
- [ ] **Phase 6**: License plate detection & localization pipeline
- [ ] **Phase 7**: OpenCV image preprocessing pipeline
- [ ] **Phase 8**: PaddleOCR text extraction & character normalization
- [ ] **Phase 9**: Vehicle automated check-in / check-out workflow
- [ ] **Phase 10**: Full REST API implementation & error handling
- [ ] **Phase 11**: Real-time Yard inventory & slot allocation
- [ ] **Phase 12**: Operational analytics & Chart.js visualizations
- [ ] **Phase 13**: Model evaluation benchmarking script
- [ ] **Phase 14**: PyTest unit and integration test suite
- [ ] **Phase 15**: Production deployment documentation & GitHub polish
- [ ] **Phase 16**: Resume-ready technical project overview

---

## Screenshots

*(Screenshots will be captured and added following Phase 2 Dashboard & Gate UI completion).*

---

## Future Improvements

- Automated container ISO 6346 code recognition.
- RFID and UHF tag reader hardware integration alongside camera streams.
- WebSocket live streaming for sub-second gate camera feeds.
- Automated barrier arm relay control via MQTT / GPIO.

---

## Limitations

- Extreme weather (heavy rain, snow, lens glare) may reduce OCR confidence and require manual review.
- Custom license plate detection requires region-specific annotated datasets for non-standard formats.

---

## Author

Developed as a production-grade Computer Vision and Yard Management automation portfolio project.
