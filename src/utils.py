import cv2
import numpy as np

# Class weights for EVU calculation
CLASS_WEIGHTS = {
    "car": 1.0,
    "bike": 0.5,
    "bus": 2.0,
    "truck": 1.5
}

def draw_boxes(frame, boxes, ids=None, cls_names=None, scores=None):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = ""
        if cls_names: label += cls_names[i]
        if scores: label += f" {scores[i]:.2f}"
        if ids: label += f" id:{ids[i]}"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

def draw_count_line(frame, pt1, pt2):
    return cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

def draw_roi(frame, roi):
    x1, y1, x2, y2 = roi
    return cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

def overlay_text(frame, text, pos=(30, 30)):
    return cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
