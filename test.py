# test.py
import sys
from src.pipeline import DetectionPipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <YOUTUBE_URL>")
        return
    url = sys.argv[1]
    pipeline = DetectionPipeline(model_path="models/yolov8n.pt", output_dir="runs", device="cpu", sample_fps=1)
    out = pipeline.process_youtube_url(url)
    print("CSV written to:", out)

if __name__ == "__main__":
    main()
