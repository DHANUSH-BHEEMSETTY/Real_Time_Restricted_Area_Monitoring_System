# 🚀 **Real-Time Restricted Area Monitoring System with YOLO**

This project integrates **FastAPI**, **Streamlit**, and **Supabase (PostgreSQL)** to create a **real-time monitoring dashboard** for object detection logs. 

- **Streamlit** (Local) runs your webcam, detects objects using YOLO, and writes data directly to a cloud database.
- **Supabase** (Cloud Database) stores detection logs in real-time.
- **FastAPI + Render** (Cloud Dashboard) reads the live data and serves an interactive web dashboard using WebSockets.

---

## 🌐 **Live Websites & URLs**

The dashboard is fully deployed to the cloud and is accessible here:

- 📊 **Main Dashboard:** [https://real-time-restricted-area-monitoring.onrender.com/](https://real-time-restricted-area-monitoring.onrender.com/)
- 📋 **Data Table:** [https://real-time-restricted-area-monitoring.onrender.com/data](https://real-time-restricted-area-monitoring.onrender.com/data)
- ❤️ **API Health Check:** [https://real-time-restricted-area-monitoring.onrender.com/health](https://real-time-restricted-area-monitoring.onrender.com/health)

---

## 🛠️ **Installation & Setup (Local WebCam App)**

To run the camera detection on your local machine and stream data to the live dashboard, follow these steps:

### 1️⃣ **Install Dependencies**

Ensure you have Python 3.11+ installed. Install the local required dependencies:

```powershell
pip install -r requirements-local.txt
```

*(Note: The cloud dashboard uses `requirements.txt` which is optimized for deployment).*

### 2️⃣ **Configure Environment Variables**

Create a `.env` file in the root directory (you can copy `.env.example`). You need your Supabase credentials:

```ini
SUPABASE_URL=https://aingwekdutgirzjdzljg.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

### 3️⃣ **Launch the Streamlit Webcam App**

Start the Streamlit frontend. Make sure your environment variables are loaded:

```powershell
# Windows PowerShell
$env:SUPABASE_URL="https://aingwekdutgirzjdzljg.supabase.co"
$env:SUPABASE_KEY="your-supabase-anon-key"
python -m streamlit run streamlit_run.py
```

- Your webcam feed will open at **[http://localhost:8501](http://localhost:8501)**.
- Any violations detected will instantly be pushed to Supabase, and your live cloud dashboard will update automatically!

---

## 🔍 **How to Check Logs & Data**

There are three ways to view the detection logs:

1. **Live Cloud Dashboard (Recommended):**
   Visit the [Data Page](https://real-time-restricted-area-monitoring.onrender.com/data) to view the real-time updating table of all detections.
   
2. **Supabase Database:**
   Log into your Supabase project (`https://aingwekdutgirzjdzljg.supabase.co`) and view the `detections` table using the Table Editor.

3. **Local Fallback (CSV):**
   If Streamlit loses connection to the internet or Supabase is not configured, it will fall back to saving logs locally in `data/detection_log.csv`.

---

## 🏗️ **Architecture Details**

```text
[Your PC / Localhost]                 [Supabase Cloud]                 [Render Cloud]
Streamlit Webcam App  ───INSERT───▶  detections table  ◀───SELECT───  FastAPI Dashboard
 (Runs YOLO model)                    (PostgreSQL DB)                  (Public Website)
```

- **Data Handling**: Uses Supabase Python clients for ultra-low latency reads/writes.
- **Real-Time Data**: FastAPI uses WebSockets to push new rows from Supabase to connected web clients every second.
- **Alerts**: Triggers local `alert.mp3` sounds when objects enter the defined restricted zone.

---

