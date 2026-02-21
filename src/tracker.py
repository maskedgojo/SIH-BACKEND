import numpy as np

class ByteTrackLite:
    def __init__(self, iou_thresh=0.3, min_conf=0.15, max_age=30):
        self.iou_thresh = iou_thresh
        self.min_conf = min_conf
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1

    def step(self, boxes, confs, cls_ids):
        ids = []
        for box, conf in zip(boxes, confs):
            if conf < self.min_conf: 
                ids.append(-1)
                continue
            tid = self.next_id
            self.tracks[tid] = {"box": box, "age": 0}
            ids.append(tid)
            self.next_id += 1
        return boxes, ids, cls_ids
