import pandas as pd
from src.utils import CLASS_WEIGHTS

def evaluate_counts(csv_path):
    df = pd.read_csv(csv_path)
    summary = df.groupby("class").size().to_dict()
    weighted = sum(summary.get(c, 0) * CLASS_WEIGHTS.get(c, 1.0) for c in CLASS_WEIGHTS)
    return summary, weighted
