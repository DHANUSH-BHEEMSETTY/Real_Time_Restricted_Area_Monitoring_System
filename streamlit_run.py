import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import threading
import random
from datetime import datetime
import os
import time
import requests  # For posting to the cloud API if Supabase creds aren't set

# Guard pygame so it doesn't crash on headless/audio-less systems
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ── Supabase / cloud config ───────────────────────────────────────────────────
# Priority 1: Direct Supabase write (fastest — skips the FastAPI middle layer)
# Priority 2: POST to CLOUD_API_URL/detect (if only the API URL is set)
# Priority 3: Local CSV fallback (no env vars set)
SUPABASE_URL  = os.environ.get("SUPABASE_URL",  "").rstrip("/")
SUPABASE_KEY  = os.environ.get("SUPABASE_KEY",  "")
CLOUD_API_URL = os.environ.get("CLOUD_API_URL", "").rstrip("/")
LOCAL_CSV     = "data/detection_log.csv"
TABLE         = "detections"

# Initialise Supabase client if credentials are present
_supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"[Supabase] Client init failed: {e}")


def save_detection_supabase(class_name: str, confidence: float, violation: bool):
    """Write directly to Supabase (fastest path — no FastAPI middle layer)."""
    try:
        _supabase_client.table(TABLE).insert({
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "class_name": class_name,
            "confidence": round(confidence, 6),
            "violation":  "Yes" if violation else "No",
        }).execute()
    except Exception as e:
        print(f"[Supabase INSERT] Failed: {e} — falling back to local CSV")
        save_detection_local(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), class_name, confidence, violation
        )


def save_detection_api(class_name: str, confidence: float, violation: bool):
    """POST a detection to the cloud FastAPI /detect endpoint."""
    payload = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "class_name": class_name,
        "confidence": confidence,
        "violation":  "Yes" if violation else "No",
    }
    try:
        resp = requests.post(f"{CLOUD_API_URL}/detect", json=payload, timeout=3)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Cloud POST] Failed: {e} — falling back to local CSV")
        save_detection_local(payload["timestamp"], class_name, confidence, violation)


def save_detection_local(timestamp: str, class_name: str, confidence: float, violation: bool):
    """Write a detection row to the local CSV (fallback / local-only mode)."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(LOCAL_CSV):
        pd.DataFrame(
            columns=["Timestamp", "Class", "Confidence", "Restricted Area Violation"]
        ).to_csv(LOCAL_CSV, index=False)
    row = pd.DataFrame([{
        "Timestamp":               timestamp,
        "Class":                   class_name,
        "Confidence":              confidence,
        "Restricted Area Violation": "Yes" if violation else "No",
    }])
    row.to_csv(LOCAL_CSV, mode="a", header=False, index=False)


# ── Session-state helper ──────────────────────────────────────────────────────

def get_app_state():
    """Return (or initialise) the persistent app state stored in session_state."""
    if "app" not in st.session_state:
        st.session_state.app = ObjectMonitoringApp()
    return st.session_state.app


# ── Main application class ────────────────────────────────────────────────────

class ObjectMonitoringApp:
    def __init__(self):
        self.models        = {}
        self.current_model = None
        self.cap           = None
        self.class_colors  = {}
        self.restricted_area = None
        self.object_entry_times = {}

        # Alert system
        self.alert_active = False
        self.alert_thread = None
        self.alert_classes = []

        # Initialise pygame safely
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
            except Exception as e:
                print(f"[Alert] pygame mixer init failed: {e}")

    def load_models(self, model_paths):
        """Load multiple YOLO models (only paths that exist on disk)."""
        for model_name, path in model_paths.items():
            if os.path.exists(path):
                self.models[model_name] = YOLO(path)
            else:
                print(f"[Warning] Model not found: {path}")
        if self.models:
            self.current_model = self.models[next(iter(self.models))]

    def generate_class_colors(self, model):
        return {
            model.names[cid]: tuple(random.randint(0, 255) for _ in range(3))
            for cid in model.names
        }

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            st.error("Error: Unable to access the webcam.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True

    def stop_webcam(self):
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            self.cap = None
            self.stop_alert()

    # ── Alert sound ───────────────────────────────────────────────────────────

    def play_alert_sound(self, sound_path):
        if not PYGAME_AVAILABLE:
            return
        try:
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play(-1)
            while self.alert_active:
                time.sleep(0.5)
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"[Alert] Sound error: {e}")

    def start_alert(self, sound_path):
        if not self.alert_active and PYGAME_AVAILABLE:
            self.alert_active = True
            self.alert_thread = threading.Thread(
                target=self.play_alert_sound, args=(sound_path,), daemon=True
            )
            self.alert_thread.start()

    def stop_alert(self):
        if self.alert_active:
            self.alert_active = False

    # ── ROI ───────────────────────────────────────────────────────────────────

    def draw_roi(self, frame):
        """Draw the elliptical restricted zone on the frame."""
        if self.current_model == self.models.get("Intrusion (YOLOv8n)"):
            h, w, _ = frame.shape
            center = (w // 2, h // 2)
            axes   = (w // 4, h // 8)
            self.restricted_area = (center, axes)
            cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 0, 255), 2)
        return frame

    def is_in_restricted_area(self, box):
        """
        True if box centre is inside the restricted ellipse.
        Uses the correct ellipse containment formula: (dx/a)^2 + (dy/b)^2 <= 1
        """
        if self.restricted_area:
            (cx, cy), (a, b) = self.restricted_area
            x1, y1, x2, y2  = box
            dx = (x1 + x2) / 2 - cx
            dy = (y1 + y2) / 2 - cy
            return (dx / a) ** 2 + (dy / b) ** 2 <= 1.0
        return False

    # ── Detection & logging ───────────────────────────────────────────────────

    def save_detection(self, class_name: str, confidence: float, violation: bool):
        """
        Save a detection event — priority order:
        1. Direct Supabase insert (if SUPABASE_URL + SUPABASE_KEY are set)
        2. POST to CLOUD_API_URL/detect (if only the API URL is set)
        3. Local CSV fallback
        """
        if _supabase_client:
            save_detection_supabase(class_name, confidence, violation)
        elif CLOUD_API_URL:
            save_detection_api(class_name, confidence, violation)
        else:
            save_detection_local(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                class_name, confidence, violation
            )

    def update_frame(self, model, confidence_threshold, selected_classes, alert_classes):
        """Process one webcam frame: detect, annotate, alert, log."""
        if not self.cap:
            return None, []

        ret, frame = self.cap.read()
        if not ret:
            return None, []

        results  = model(frame, conf=confidence_threshold, iou=0.3)
        detected = []
        annotated = frame.copy()
        any_violation = False

        for result in results[0].boxes:
            cid        = int(result.cls)
            class_name = model.names[cid]

            if class_name not in selected_classes:
                continue

            detected.append(class_name)
            color       = self.class_colors.get(class_name, (0, 255, 0))
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf        = float(result.conf[0])
            inside      = self.is_in_restricted_area([x1, y1, x2, y2])

            if inside:
                any_violation = True
                cv2.putText(annotated, "Object in Restricted Area!",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated, f"{class_name} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Throttle: log every 2 s per class
            if class_name not in self.object_entry_times:
                self.object_entry_times[class_name] = time.time()
            if time.time() - self.object_entry_times[class_name] > 2:
                self.save_detection(class_name, conf, inside)
                self.object_entry_times[class_name] = time.time()

        if any_violation and any(c in alert_classes for c in detected):
            self.start_alert("alert.mp3")
        else:
            self.stop_alert()

        return self.draw_roi(annotated), detected

    # ── Streamlit UI ──────────────────────────────────────────────────────────

    def run(self):
        st.set_page_config(page_title="Real-Time Object Monitoring System", layout="wide")
        st.markdown(
            "<h2 style='text-align:center'>🔍 Real-Time Intrusion Detection & Restricted Area Monitoring</h2>",
            unsafe_allow_html=True,
        )

        # Show connection mode banner
        if _supabase_client:
            st.success(f"🗄️ Supabase mode — writing directly to Supabase ({SUPABASE_URL})")
        elif CLOUD_API_URL:
            st.success(f"☁️ API mode — sending detections to **{CLOUD_API_URL}**")
        else:
            st.warning("💾 Local mode — set `SUPABASE_URL` + `SUPABASE_KEY` env vars to write to Supabase")

        st.sidebar.title("🔧 Settings")

        # All three model files
        model_paths = {
            "Intrusion (YOLOv8n)": "model/yolov8n.pt",
            "PPE Detection":       "model/ppe8n.pt",
            "Custom (best)":       "model/best.pt",
        }

        # Load models once per session
        if not self.models:
            with st.spinner("Loading models…"):
                self.load_models(model_paths)

        if not self.models:
            st.error("No model files found in model/. Please add a .pt file.")
            return

        selected_model = st.sidebar.selectbox("Select Model", options=list(self.models.keys()))

        if self.current_model != self.models.get(selected_model):
            self.current_model = self.models[selected_model]
            self.class_colors  = self.generate_class_colors(self.current_model)

        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
        available_classes    = list(self.current_model.names.values())
        selected_classes     = st.sidebar.multiselect("Objects to Detect", available_classes, default=[])
        self.alert_classes   = st.sidebar.multiselect("Objects for Alert Sound", available_classes, default=[])

        if st.sidebar.button("▶️ Start Webcam"):
            if self.start_webcam():
                st.success("Webcam started successfully!")

        if st.sidebar.button("⏹️ Stop Webcam"):
            self.stop_webcam()
            st.success("Webcam stopped.")

        if self.cap:
            frame_placeholder = st.empty()
            while self.cap.isOpened():
                frame, _ = self.update_frame(
                    self.current_model, confidence_threshold,
                    selected_classes, self.alert_classes,
                )
                if frame is not None:
                    frame_placeholder.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB"
                    )


if __name__ == "__main__":
    app = get_app_state()
    app.run()