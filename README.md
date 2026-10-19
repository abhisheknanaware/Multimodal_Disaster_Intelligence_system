# 🌊 PRALAY SENSE — Multimodal Disaster Intelligence System

> AI-powered **flood & landslide risk prediction** for **Assam**, fusing Sentinel-1 SAR, Sentinel-2 optical, terrain (DEM) and rainfall data — with real-time alerts, expert review and citizen reporting.

Built for **Smart India Hackathon (SIH)**.

![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-Express-339933?logo=node.js&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?logo=mongodb&logoColor=white)
![Python](https://img.shields.io/badge/Python-FastAPI-009688?logo=fastapi&logoColor=white)
![CatBoost](https://img.shields.io/badge/ML-CatBoost-FFCC00)

---

## ✨ Features

- 🛰️ **Multimodal inputs** — Sentinel-1 SAR change, Sentinel-2 NDVI/NDWI, DEM, slope, aspect, TRI, TPI, rainfall
- 🤖 **ML risk engine** — CatBoost models for landslide & flood probability with per-feature explanations
- 🗺️ **Interactive risk map** — susceptibility tiles served live over Leaflet
- 🚨 **Alerts** — Email and WhatsApp (Twilio) notifications with a cron-based risk monitor
- 🧑‍🔬 **Expert Review** workflow and AI risk explanations
- 📣 **Citizen Portal & field reports** — community reporting with photo uploads
- 🛣️ **Impact assessment** — road and hazard impact analysis

---

## 🏗️ Architecture

```
┌────────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  Frontend (React)  │ ──▶ │  Backend (Node/TS)   │ ──▶ │  ML Service (FastAPI)  │
│  Vite · Leaflet    │     │  Express · Mongoose  │     │  CatBoost · rasterio   │
│  :5173             │     │  :5000               │     │  :8000                 │
└────────────────────┘     └──────────┬───────────┘     └───────────┬────────────┘
                                      │                             │
                                 ┌────▼─────┐              ┌────────▼─────────┐
                                 │ MongoDB  │              │ Rasters + Models │
                                 └──────────┘              └──────────────────┘
```

| Layer | Stack |
|---|---|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, Leaflet, Recharts |
| Backend | Node.js, Express 5, Mongoose, node-cron, Nodemailer, Twilio |
| ML Service | Python, FastAPI, CatBoost, rasterio, rio-tiler, pandas |
| Database | MongoDB |

---

## 📁 Project Structure

```
Sih/
├── Multimodal_Disaster_Intelligence_system/
│   ├── frontend/            # React dashboard (pages, map, alerts, citizen portal)
│   └── backend/             # Express API: risk, impact, expert, community, alerts
├── python-ml-service/       # FastAPI inference + raster tile server
├── training_scripts/        # Sentinel-2 preprocessing & model training
├── sentinel/sentinel/       # Landslide pipeline, experiments, model results
├── models/                  # Trained models (v2, v3)
├── data/                    # Feature datasets (CSV)
├── Processing/              # Terrain / landslide inventory (GPKG, SHP)
├── rainfall/                # IMD rainfall data
├── docs/                    # Sentinel-2 inventory & training reports
└── start_all_servers.bat    # Starts all 3 services on Windows
```

---

## ⚠️ Files Not in This Repository

Some files are **not stored on GitHub** because of size or security. You need them to run the full system.

### 1. Raster data (~19 GB) — download separately

GitHub limits files to 100 MB, so the geospatial rasters are hosted externally.

📥 **Download:** _<add Google Drive link here>_

Place them at these exact paths:

| File | Size |
|---|---|
| `sentinel/sentinel/taking_feature/maps/landslide_susceptibility_probability.tif` | 2.6 GB |
| `Processing/assam_slope_corrected.tif` | 1.6 GB |
| `Processing/assam_glo30_dem.tif` | 1.45 GB |
| `Processing/assam_aspect.tif` | 1.45 GB |
| `Processing/assam_TRI.tif` | 1.45 GB |
| `Processing/assam_TPI.tif` | 1.45 GB |
| `Processing/S1_SAR_Change.tif` | 470 MB |
| `sentel/sentinel/output/2026/S1A_IW_GRDH_..._TC.dim` + `.data/` folder | ~5 GB |
| `data/sentinel2/2025/processed/*.tif` | ~3 GB |

### 2. Trained models

The ML service loads v5 models from `models/v5/` (override with `MODEL_DIR`):

- `landslide_v5.cbm` — terrain susceptibility classifier; live risk = susceptibility × rainfall trigger (IMD heavy-rain classes)
- `flood_v5.cbm` — live Open-Meteo rainfall + soil moisture + terrain/floodplain features, monotonic in rainfall
- `imd_monsoon_climatology.npz` — IMD 2020–2025 Jun–Sep rainfall climatology used as a feature
- `metrics_v5.json` — spatial / temporal cross-validation metrics and decision thresholds

Rebuild with `python training_scripts/build_v5_datasets.py` then `python training_scripts/train_v5_models.py`. Features for training and serving come from the same module, `python-ml-service/feature_builder.py`.

| Model (spatial CV) | ROC-AUC | Macro-F1 | Precision | Recall |
|---|---|---|---|---|
| Landslide v4 (old) | 0.712 | 0.633 | 0.449 | 0.501 |
| **Landslide v5** | **0.845** | **0.729** | **0.557** | **0.703** |
| Flood, old features without S1 | 0.891 | 0.807 | 0.733 | 0.871 |
| **Flood v5** | **0.888** | **0.813** | **0.726** | **0.911** |

Without them, risk predictions return `0`.

### 3. Backend `.env`

Create `Multimodal_Disaster_Intelligence_system/backend/.env`. **Never commit this file.**

```env
PORT=5000
MONGODB_URI=
PYTHON_ML_URL=http://localhost:8000

# Email alerts
EMAIL_HOST=
EMAIL_PORT=
EMAIL_USER=
EMAIL_PASSWORD=
EMAIL_FROM=
TEST_EMAIL_TO=

# WhatsApp alerts (Twilio)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=
TEST_WHATSAPP_TO=

# Image uploads (Cloudinary)
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=

# AI explanations, routing & traffic
OPENAI_API_KEY=
ORS_API_KEY=
TRAFFIC_API_KEY=
```

---

## 🚀 Getting Started (Windows)

### Prerequisites

- Node.js 18+
- Python 3.9+
- MongoDB (local or Atlas)

### 1. Clone

```bash
git clone https://github.com/abhisheknanaware/Multimodal_Disaster_Intelligence_system.git
cd Multimodal_Disaster_Intelligence_system
```

Then add the raster data, models and `.env` listed above.

### 2. ML Service — port 8000

```bash
cd python-ml-service
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

> Set `PROJECT_ROOT` if the repo is not the parent folder of `python-ml-service`.

### 3. Backend — port 5000

```bash
cd Multimodal_Disaster_Intelligence_system/backend
npm install
npm run dev
```

### 4. Frontend — port 5173

```bash
cd Multimodal_Disaster_Intelligence_system/frontend
npm install
npm run dev
```

Or start all three at once:

```bash
start_all_servers.bat
```

---

## 🔌 API Overview

**ML Service (`:8000`)**

| Endpoint | Description |
|---|---|
| `GET /health` | Service health check |
| `GET /metadata` | Model and raster info |
| `GET /risk` | Landslide risk at a location |
| `GET /flood` | Flood risk at a location |
| `GET /sar` | Sentinel-1 SAR values at a location |
| `GET /tiles/baseline/{z}/{x}/{y}.png` | Landslide susceptibility map tiles |
| `GET /tiles/flood/{z}/{x}/{y}.png` | Flood map tiles |

**Backend (`:5000`)**

| Route | Description |
|---|---|
| `/api/risk` | Risk events and AI explanations |
| `/api/impact` | Road and hazard impact assessment |
| `/api/expert` | Expert review workflow |
| `/api/community` | Citizen field reports |
| `/api/alerts` | Alert management |
| `/api/recipients` | Alert recipients |

---

## 📊 Data Sources

- **Sentinel-1** — ESA Sentinel-1 GRD SAR
- **Sentinel-2** — ESA optical imagery (NDVI / NDWI)
- **DEM / Terrain** — Copernicus GLO-30
- **Rainfall** — IMD gridded rainfall, Open-Meteo API
- **Historical events** — ASDMA, NRSC landslide inventory
- **Road network** — OpenStreetMap

---

## 🛠️ Troubleshooting

- **Missing raster / model errors** — check the files in [Files Not in This Repository](#️-files-not-in-this-repository) are at the right paths, or set `PROJECT_ROOT`.
- **MongoDB connection fails** — whitelist your IP on Atlas, or make sure local MongoDB runs on port `27017`.
- **rasterio / GDAL install fails on Windows** — install with `conda install -c conda-forge rasterio`, or use prebuilt wheels.

---

## 👤 Author

**Abhishek Nanaware** — [@abhisheknanaware](https://github.com/abhisheknanaware)
