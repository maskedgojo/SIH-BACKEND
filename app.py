# app.py
import os
import warnings
import pandas as pd
from flask import Flask, jsonify, request
from urllib.parse import urlparse, parse_qs
from sklearn.exceptions import InconsistentVersionWarning
from flask_cors import CORS   # ✅ NEW: CORS import

# ---------------- Suppress sklearn warnings ----------------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ---------------- Internal Imports ----------------
from src.pipeline import DetectionPipeline
from src.services.metrics import (
    arrival_rate_estimator,
    emergency_vehicle_flag,
    signal_allocation,
    format_csv_row,
    TrafficPipeline
)

from config import FLASK_HOST, FLASK_PORT

# ---------------- Flask App ----------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})   # ✅ Enable CORS for dev (later restrict to frontend origin)

# use pipeline instead of Predictor
MODEL_PATH = "models/yolov8n.pt"
pipeline = DetectionPipeline(model_path=MODEL_PATH, output_dir="runs", device="cpu")

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------- Helper Functions ----------------
def get_cycle_rows(limit: int = 50) -> pd.DataFrame:
    files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith("_data.csv")]
    if not files:
        return pd.DataFrame()
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(OUTPUT_FOLDER, f)))
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, latest_file))
    return df.tail(limit)


def get_latest_cycle_rows() -> pd.DataFrame:
    files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith("_data.csv")]
    if not files:
        return pd.DataFrame()
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(OUTPUT_FOLDER, f)))
    return pd.read_csv(os.path.join(OUTPUT_FOLDER, latest_file))


def log_legacy(df: pd.DataFrame, video_id: str):
    path = os.path.join(OUTPUT_FOLDER, f"{video_id}_legacy.csv")
    df.to_csv(path, index=False)


def log_full(df: pd.DataFrame, video_id: str):
    path = os.path.join(OUTPUT_FOLDER, f"{video_id}_full.csv")
    df.to_csv(path, index=False)


def log_snapshot(df: pd.DataFrame):
    path = os.path.join(OUTPUT_FOLDER, "latest_snapshot.csv")
    df.to_csv(path, index=False)


def response(success: bool, message: str = "", data: dict | None = None, status: int = 200):
    """Standard API response format"""
    return jsonify({
        "success": success,
        "message": message,
        "data": data or {}
    }), status


def extract_video_id(url: str) -> str:
    """Extract YouTube video_id from full URL"""
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    query = parse_qs(parsed.query)
    return query.get("v", ["unknown"])[0]


# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    return response(True, "🚦 Smart Traffic Backend is running")


# --- Traffic Data APIs ---
@app.route("/traffic/data", methods=["GET"])
def traffic_data():
    df = get_latest_cycle_rows()
    if df.empty:
        return response(False, "No traffic data available", status=404)
    return response(True, "Latest traffic data", df.iloc[-1].to_dict())


@app.route("/traffic/history", methods=["GET"])
def traffic_history():
    limit = int(request.args.get("limit", 50))
    df = get_cycle_rows(limit=limit)
    return response(True, f"Last {limit} cycles", df.to_dict(orient="records"))


@app.route("/signal/status", methods=["GET"])
def signal_status():
    signal_id = request.args.get("signal_id")
    if not signal_id:
        return response(False, "Missing signal_id", status=400)

    df = get_latest_cycle_rows()
    if df.empty:
        return response(False, f"No data for signal {signal_id}", status=404)

    df_signal = df[df["signal_id"] == signal_id]
    if df_signal.empty:
        return response(False, f"Signal {signal_id} not found", status=404)

    latest_row = df_signal.iloc[-1]
    return response(True, "Signal status", {
        "signal_id": signal_id,
        "status": "GREEN",  # TODO: replace with allocator output later
        "last_change": latest_row.get("timestamp", "")
    })


# --- Allocator API ---
@app.route("/allocator/simulate", methods=["POST"])
def allocator_simulate():
    payload = request.get_json(force=True)
    if not payload:
        return response(False, "Missing JSON body", status=400)

    evu_raw = payload.get("evu_raw", {})
    wait_time = payload.get("wait_time", {})
    emergency_flags = payload.get("emergency", {})

    try:
        tp = TrafficPipeline()
        _, _, allocation = tp.allocator_step(evu_raw, wait_time, emergency_flags)

        return response(True, "Allocator simulation successful", {
            "input": payload,
            "output": allocation
        })
    except Exception as e:
        return response(False, f"Allocator simulation failed: {e}", status=500)


# --- Model APIs ---
@app.route("/model/info", methods=["GET"])
def model_info():
    return response(True, "Model information", {
        "model_path": MODEL_PATH,
        "mode": "detection-pipeline"
    })


@app.route("/model/reload", methods=["POST"])
def model_reload():
    global pipeline
    pipeline = DetectionPipeline(model_path=MODEL_PATH, output_dir="runs", device="cpu")
    return response(True, "Model reloaded", {"model_path": MODEL_PATH})


# --- Prediction APIs ---
@app.route("/predict/row", methods=["POST"])
def predict_row():
    row = request.get_json(force=True)
    if not isinstance(row, dict):
        return response(False, "Expected JSON object", status=400)

    row_out = row.copy()
    row_out["prediction"] = 1  # TODO: hook real prediction from pipeline
    return response(True, "Row prediction", row_out)


@app.route("/predict/cycle/<int:cycle>", methods=["GET"])
def predict_cycle(cycle: int):
    df = get_cycle_rows(limit=1)
    predictions = [dict(r) for _, r in df.iterrows()]
    return response(True, f"Predictions for cycle {cycle}", {"rows": predictions})


# --- YouTube Upload & Process ---
@app.route("/upload/youtube", methods=["POST"])
def upload_youtube():
    data = request.get_json()
    if not data or "url" not in data:
        return response(False, "No YouTube URL provided", status=400)

    url = data["url"]
    video_id = extract_video_id(url)

    try:
        csv_path = pipeline.process_youtube_url(url)

        df = pd.read_csv(csv_path)
        out_csv = os.path.join(OUTPUT_FOLDER, f"{video_id}_data.csv")
        out_xls = os.path.join(OUTPUT_FOLDER, f"{video_id}_data.xlsx")
        df.to_csv(out_csv, index=False)
        df.to_excel(out_xls, index=False)

        log_legacy(df, video_id)
        log_full(df, video_id)
        log_snapshot(df)

        return response(True, "Video processed successfully", {
            "video_id": video_id,
            "csv_path": out_csv,
            "xls_path": out_xls,
            "rows": df.to_dict(orient="records")
        })

    except Exception as e:
        return response(False, f"Failed to process video: {str(e)}", status=500)


# --- CSV Download API ---
@app.route("/csv/<video_id>", methods=["GET"])
def csv_data(video_id):
    csv_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_data.csv")
    if not os.path.exists(csv_path):
        return response(False, "CSV not found. Upload/process video first", status=404)

    df = pd.read_csv(csv_path)
    return response(True, f"CSV data for {video_id}", {"rows": df.to_dict(orient="records")})


# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
