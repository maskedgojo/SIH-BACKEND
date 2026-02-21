# tests/test_pipeline.py
import os
import pandas as pd
from src.pipeline import DetectionPipeline

def test_pipeline_youtube_minimal():
    # NOTE: This test downloads a short sample YouTube video. Replace sample_url with a short clip you trust.
    pipeline = DetectionPipeline(model_path="models/yolov8n.pt", output_dir="runs", device="cpu", sample_fps=1)
    sample_url = "https://www.youtube.com/watch?v=z8EwY3Slsws"  # replace if blocked or too long
    csv_path = pipeline.process_youtube_url(sample_url)
    assert os.path.exists(csv_path), "CSV not created"
    df = pd.read_csv(csv_path)
    expected_cols = [
        "timestamp","signal_id","cycle","approach","C","EVU_raw","EVU_smooth","queue_len",
        "arrival_rate","emergency","D","W","W_norm","Dprime","G_allocated",
        "order_1","order_2","order_3","order_4","t_start","t_end","intergreen_L",
        "T_base","G_min","G_max","alpha_up","alpha_down","beta","W_th","delta","T_alloc"
    ]
    for c in expected_cols:
        assert c in df.columns, f"Missing column {c}"
    assert not df.empty, "DataFrame is empty"
