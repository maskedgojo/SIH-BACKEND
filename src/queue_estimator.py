import numpy as np

class QueueEstimator:
    def __init__(self, roi, px_to_meter=0.05):
        self.roi = roi
        self.px_to_meter = px_to_meter

    def estimate(self, tracks, id2name, stationary_thresh=6):
        x1, y1, x2, y2 = self.roi
        q_len = 0
        for t in tracks:
            bx1, by1, bx2, by2 = map(int, t["box"])
            if x1 <= bx1 <= x2 and y1 <= by1 <= y2:
                q_len += (by2 - by1) * self.px_to_meter
        return q_len
