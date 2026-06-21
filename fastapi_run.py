from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel
import pandas as pd
import asyncio
import os
from collections import Counter
from supabase import create_client, Client

app = FastAPI(title="Restricted Area Monitoring Dashboard")

# Allow requests from the local Streamlit app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ── Supabase client ───────────────────────────────────────────────────────────
SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Table name in Supabase (create this in your Supabase SQL editor — see README)
TABLE = "detections"


# ── Helpers ───────────────────────────────────────────────────────────────────

def rows_to_df(rows: list) -> pd.DataFrame:
    """Convert a list of Supabase row dicts to a canonical DataFrame."""
    if not rows:
        return pd.DataFrame(
            columns=["id", "Timestamp", "Class", "Confidence", "Restricted Area Violation"]
        )
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "timestamp":  "Timestamp",
        "class_name": "Class",
        "confidence": "Confidence",
        "violation":  "Restricted Area Violation",
    })
    return df


# ── Pydantic model for POST /detect ──────────────────────────────────────────

class Detection(BaseModel):
    timestamp:  str
    class_name: str
    confidence: float
    violation:  str   # "Yes" | "No"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Render health-check endpoint."""
    return {"status": "ok"}


@app.post("/detect")
async def post_detection(det: Detection):
    """Receive a detection from the local Streamlit app and insert into Supabase."""
    supabase.table(TABLE).insert({
        "timestamp":  det.timestamp,
        "class_name": det.class_name,
        "confidence": round(det.confidence, 6),
        "violation":  det.violation,
    }).execute()
    return {"status": "saved"}


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/data", response_class=HTMLResponse)
async def get_data_page(request: Request):
    return templates.TemplateResponse("data.html", {"request": request})


# ── WebSocket: live dashboard ─────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_id = 0

    try:
        while True:
            # Fetch only rows newer than what we've already sent
            new_resp = (
                supabase.table(TABLE)
                .select("*")
                .gt("id", last_id)
                .order("id")
                .execute()
            )
            new_rows = new_resp.data or []

            if new_rows:
                last_id = new_rows[-1]["id"]

                # Full dataset for summary stats
                all_resp = supabase.table(TABLE).select("*").order("id").execute()
                all_rows = all_resp.data or []

                df      = rows_to_df(all_rows)
                new_df  = rows_to_df(new_rows)

                total_detections = len(df)
                total_violations = (df["Restricted Area Violation"] == "Yes").sum() if len(df) else 0
                most_frequent    = (
                    Counter(df["Class"]).most_common(1)[0][0] if len(df) else "N/A"
                )
                top_5_violations = (
                    df[df["Restricted Area Violation"] == "Yes"]
                    .tail(5)
                    .to_dict(orient="records")
                    if len(df) else []
                )

                payload = {
                    "timestamp":                 new_df["Timestamp"].tolist(),
                    "class":                     new_df["Class"].tolist(),
                    "confidence":                new_df["Confidence"]
                                                     .apply(lambda x: round(float(x) * 100, 2))
                                                     .tolist(),
                    "restricted_area_violation": new_df["Restricted Area Violation"].tolist(),
                    "summary": {
                        "total_detections":    int(total_detections),
                        "total_violations":    int(total_violations),
                        "most_frequent_class": most_frequent,
                        "top_5_violations":    top_5_violations,
                    },
                }
                await websocket.send_json(payload)

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket /ws] Error: {e}")


# ── WebSocket: full data table ────────────────────────────────────────────────

@app.websocket("/ws/data")
async def websocket_data_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            resp = (
                supabase.table(TABLE)
                .select("*")
                .order("timestamp", desc=True)
                .execute()
            )
            rows = resp.data or []
            df   = rows_to_df(rows)
            await websocket.send_json({"data": df.to_dict(orient="records")})
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket /ws/data] Error: {e}")
