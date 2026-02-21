import pandas as pd

def detections_to_csv(detections, path):
    df = pd.DataFrame(detections)
    df.to_csv(path, index=False)
