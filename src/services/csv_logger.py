# src/services/csv_logger.py
import pandas as pd
from typing import List, Dict

def save_csv_rows(rows: List[Dict], path: str) -> str:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path
