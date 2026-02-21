# config.py
import os

# ---------------- Flask Config ----------------
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")  # default: 0.0.0.0
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))  # default: 5000

# ---------------- Model Config ----------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")

# ---------------- Other Options ----------------
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
LOG_DIR = os.getenv("LOG_DIR", "logs")

# Create directories if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
