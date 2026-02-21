import numpy as np
from src.utils import CLASS_WEIGHTS

class LineCounter:
    def __init__(self, pt1, pt2):
        self.pt1, self.pt2 = pt1, pt2
        self.counts = {}

    def step(self, tracks, id2name, frame_idx):
        for t in tracks:
            cname = t["cls_name"]
            self.counts[cname] = self.counts.get(cname, 0) + 1
        return self.counts

def compute_evu(counts, weights=CLASS_WEIGHTS):
    return sum(counts.get(c, 0) * weights.get(c, 1.0) for c in weights)
