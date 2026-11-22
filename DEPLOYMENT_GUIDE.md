# AgriForecast Deployment Guide

This guide outlines four deployment options for the **AgriForecast: Agricultural Disease Time Series Forecasting & Surveillance Suite**, ranging from instant free cloud hosting to containerized production deployment and viva live tunneling.

---

## Deployment Options At a Glance

| Deployment Method | Cost | Setup Time | Best Suited For | Key Benefit |
|:---|:---:|:---:|:---|:---|
| **1. Streamlit Community Cloud** | Free | ~3 mins | Academic Project Submission & Viva | Official Streamlit host, auto-syncs with GitHub |
| **2. Hugging Face Spaces** | Free | ~4 mins | Machine Learning & Deep Learning Demo | 16 GB RAM free, handles PyTorch smoothly |
| **3. Docker Container (Render / Cloud Run)** | Free / Low | ~5 mins | Production & Isolated Containerization | Zero dependency discrepancies |
| **4. Quick Tunneling (Ngrok / Cloudflare)** | Free | 30 secs | Immediate In-Person Viva Demonstration | Shares localhost directly to a public URL |

---

## Method 1: Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is the fastest, free, and standard platform for deploying Streamlit applications directly from GitHub.

### Step 1: Push Repository to GitHub
Ensure your repository is pushed to your GitHub account:
```bash
git add .
git commit -m "Prepare AgriForecast for Phase 3 deployment"
git push origin main
```

### Step 2: Connect to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and click **Sign In with GitHub**.
2. Click **Create app** or **New app**.

### Step 3: Configure Deployment Settings
* **Repository:** Select `your-username/Disease_time_series` (or your repo name).
* **Branch:** `main` (or `master`).
* **Main file path:** `app.py`.
* **App URL:** (Optional) Customize your subdomain, e.g., `agriforecast-bennett.streamlit.app`.

### Step 4: Advanced Settings (Python Version)
Under **Advanced settings**:
* Select **Python 3.11** or **Python 3.12**.
* Click **Save** and then click **Deploy!**

Your live URL will be active in ~2–3 minutes at:  
`https://<your-app-name>.streamlit.app`

---

## Method 2: Hugging Face Spaces (Free ML Cloud)

Hugging Face Spaces provides 2 vCPU and 16 GB of memory for free, ideal for running PyTorch sequence models (LSTM, GRU, Temporal Transformer) alongside tree ensembles.

### Steps:
1. Create a free account on [huggingface.co](https://huggingface.co/).
2. Click **Spaces** > **Create new Space**.
3. Fill in:
   * **Space name:** `agri-disease-timeseries`
   * **License:** `mit` or `apache-2.0`
   * **Select the Space SDK:** Choose **Streamlit**
   * **Space hardware:** Free (2 vCPU · 16 GB RAM)
4. Clone the space repository locally or push your existing repository as a new remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/agri-disease-timeseries
   git push hf main
   ```
5. Hugging Face automatically reads `requirements.txt` and launches `app.py`.

---

## Method 3: Containerized Deployment via Docker

A production-ready [Dockerfile](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/Dockerfile) is included in the project root.

### Local Docker Build & Run:
```bash
# 1. Build the Docker image
docker build -t agriforecast:latest .

# 2. Run the container on port 8501
docker run -d -p 8501:8501 --name agriforecast_app agriforecast:latest

# 3. Verify in browser:
# http://localhost:8501
```

### Deploying the Docker Container to Cloud Platforms:
* **Render (Free Web Service):**
  1. Create a new **Web Service** on [render.com](https://render.com/).
  2. Connect your GitHub repository.
  3. Select **Docker** environment.
  4. Port: `8501`. Click **Deploy**.
* **Google Cloud Run (Serverless Container):**
  ```bash
  gcloud builds submit --tag gcr.io/<PROJECT_ID>/agriforecast
  gcloud run deploy agriforecast --image gcr.io/<PROJECT_ID>/agriforecast --port 8501 --allow-unauthenticated
  ```

---

## Method 4: Instant Viva Demonstration (Public URL in 30 Seconds)

If you are presenting your project to an evaluator, professor, or external examiner in person, and you want them to open the app immediately on their phone or laptop without deploying to a cloud server:

### Option A: Via Local Network (Same Wi-Fi)
1. Run the app binding to all interfaces:
   ```bash
   ./.venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```
2. Streamlit will output your **Network URL**:
   ```
   Network URL: http://192.168.x.x:8501
   ```
3. Anyone connected to the same Wi-Fi (e.g. Bennett University campus Wi-Fi or mobile hotspot) can open that IP in their mobile browser!

### Option B: Via Ngrok (Instant Public HTTPS URL)
1. In one terminal, run the app:
   ```bash
   ./.venv/bin/streamlit run app.py
   ```
2. In a second terminal, tunnel port 8501:
   ```bash
   ngrok http 8501
   ```
3. Ngrok gives you an instant public HTTPS link (e.g. `https://xxxx.ngrok-free.app`) that works from anywhere in the world.

---

## Pre-Deployment Verification Checklist

Before deploying, ensure:
- [x] Application passes all automated test cases: `pytest tests/` (113/113 passed).
- [x] Pretrained model artifacts are present in `models/` and `artifacts/`.
- [x] Dataset is present in `data/disease_timeseries_master.csv`.
- [x] Port 8501 is exposed in `.streamlit/config.toml` or `Dockerfile`.
- [x] Reports are accessible in `reports/`.
