# src/pipeline.py
import os
import cv2
import time
import tempfile
import shutil
from typing import List, Dict
from datetime import datetime
import pandas as pd
import yt_dlp

from src.detector import YOLOv8Detector
from src.tracker import ByteTrackLite
from src.counter import LineCounter, compute_evu
from src.queue_estimator import QueueEstimator
from src.services.metrics import (
    compute_evu_from_counts,
    asymmetric_ema,
    arrival_rate_estimator,
    compute_cycle_time_webster,
    compute_gmins_from_queue,
    compute_gmax_from_cycle,
    TrafficAllocator,
    format_csv_row_full,
    DEFAULT_INTERGREEN
)


class DetectionPipeline:
    def __init__(self,
                 model_path: str = "models/yolov8n.pt",
                 output_dir: str = "runs",
                 device: str = "cpu",
                 sample_fps: int = 1,
                 conf: float = 0.25):
        self.detector = YOLOv8Detector(model_path=model_path, device=device)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.sample_fps = max(1, int(sample_fps))
        self.conf = conf
        self.device = device

        # approaches, counters, queue estimators
        self.approaches = ["N", "E", "S", "W"]
        H, W = 720, 1280
        # counting line coordinates — keep as before or adjust per scene
        self.count_lines = {
            "N": ((640, 200), (1000, 200)),
            "E": ((1080, 360), (1080, 680)),
            "S": ((640, 620), (1000, 620)),
            "W": ((200, 360), (200, 680)),
        }
        self.rois = {
            "N": (400, 100, 1200, 300),
            "E": (900, 200, 1200, 700),
            "S": (400, 500, 1200, 700),
            "W": (100, 200, 400, 700),
        }
        self.line_counters = {a: LineCounter(*self.count_lines[a]) for a in self.approaches}
        self.queue_estimators = {a: QueueEstimator(self.rois[a]) for a in self.approaches}
        self.tracker = ByteTrackLite()

        # allocator instance (holds state EVU_prev, W_cum)
        self.allocator = TrafficAllocator(approaches=self.approaches)

    # -------- helpers
    def download_youtube_video(self, url: str) -> str:
        tmpdir = tempfile.mkdtemp(prefix="ytdl_")
        outtmpl = os.path.join(tmpdir, "video.%(ext)s")
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            fname = ydl.prepare_filename(info)
        # pick last file if not mp4
        if not fname.lower().endswith(".mp4"):
            files = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir)], key=os.path.getmtime)
            if files:
                fname = files[-1]
        return fname

    def _bin_by_approach(self, tracks_list: List[Dict], frame_shape) -> Dict[str, List[Dict]]:
        """
        Assign tracks to approaches using their centroid relative to frame center.
        """
        h, w = frame_shape[:2]
        bins = {a: [] for a in self.approaches}
        for tr in tracks_list:
            bx = tr["box"]
            cx = (float(bx[0]) + float(bx[2])) / 2.0
            cy = (float(bx[1]) + float(bx[3])) / 2.0
            vert = "N" if cy < (h / 2) else "S"
            # refine via horizontal split
            if vert == "N" or vert == "S":
                # if very far left or right classify as E/W
                if cx < (w * 0.25):
                    binname = "W"
                elif cx > (w * 0.75):
                    binname = "E"
                else:
                    binname = vert
            else:
                binname = vert
            bins[binname].append(tr)
        return bins

    def process_youtube_url(self, url: str, signal_id: str = "SIGNAL_001", max_frames: int = None) -> str:
        """
        Download youtube video, process frames end-to-end, compute metrics and write CSV.
        Returns csv_path (string).
        """
        video_path = self.download_youtube_video(url)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / self.sample_fps)))

        frame_idx = 0
        extracted = 0
        prev_total_count = 0
        prev_time = None
        cycle = 0
        rows = []

        # per-cycle accumulators (optional)
        # main state is in allocator (EVU_prev, W_cum)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            # ---- detection ----
            detections = self.detector.detect(frame, conf=self.conf)  # returns list of dicts with xmin,ymin,xmax,ymax,class
            # filter & convert detection to lists for tracker
            filt_boxes, filt_confs, filt_cls = [], [], []
            for d in detections:
                # assume detector returned 'class' as label name or numeric, mapping handled in YOLOv8Detector
                bbox = [d["xmin"], d["ymin"], d["xmax"], d["ymax"]]
                filt_boxes.append(bbox)
                filt_confs.append(d.get("confidence", 0.5))
                # map label name to small int index for tracker (we can use 0 placeholder)
                filt_cls.append(0)

            # ---- tracking ----
            t_boxes, t_ids, t_cls = self.tracker.step(filt_boxes, filt_confs, filt_cls)
            tracks_list = []
            for idx, (tb, tid) in enumerate(zip(t_boxes, t_ids)):
                # attempt to preserve class name if available
                cname = detections[idx].get("class", "car") if idx < len(detections) else "car"
                cidx = 0
                tracks_list.append({"id": tid, "box": tb, "cls_id": cidx, "cls_name": cname})

            # ---- counting + queue estimation ----
            per_app_counts = {a: {} for a in self.approaches}
            per_app_queue = {a: 0.0 for a in self.approaches}
            for a in self.approaches:
                counts = self.line_counters[a].step(tracks_list, {0: "car"}, frame_idx)
                # counts is dict class->count
                per_app_counts[a] = counts.copy()
                q = self.queue_estimators[a].estimate(tracks_list, {0: "car"})
                per_app_queue[a] = float(q)

            total_detections = sum(sum(list(v.values())) for v in per_app_counts.values())
            now_time = datetime.utcnow()
            if prev_time is None:
                dt = 1.0 / max(1.0, self.sample_fps)
            else:
                dt = max(1e-3, (now_time - prev_time).total_seconds())
            prev_time = now_time

            arrival_rate = arrival_rate_estimator(prev_total_count, total_detections, dt)  # vehicles per minute
            prev_total_count = total_detections

            # EVU raw per approach from counts
            EVU_raw = compute_evu_from_counts(per_app_counts)

            # call allocator step: need arrival rates in vph -> convert from per-minute to per-hour
            arrival_rates_vph = {a: arrival_rate * 60.0 * (per_app_counts[a].get(next(iter(per_app_counts[a]), ""), 0) / max(1, total_detections)) if total_detections>0 else 0.0 for a in self.approaches}
            # the above is a rough split of aggregated arrival rate into flows per approach; more advanced splitting possible

            # emergency detection
            emergency_flags = {a: any(["ambulance" in (t.get("cls_name","").lower()) for t in tracks_list]) for a in self.approaches}

            # dynamic cycle calculation (Webster) uses arrival_rates_vph
            cycle_time_dynamic = compute_cycle_time_webster(arrival_rates_vph)

            # run allocator
            EVU_smooth, Dprime, G_alloc = self.allocator.step(EVU_raw, per_app_queue, arrival_rates_vph, emergency_flags, cycle_time_dynamic=cycle_time_dynamic)

            # compute intermediates
            D = {a: (EVU_smooth[a] / max(1e-6, self.allocator.C_capacity.get(a, 10.0))) for a in self.approaches}
            W = per_app_queue.copy()
            Wmax = max(1e-6, max(W.values()))
            W_norm = {a: W[a] / Wmax for a in self.approaches}
            Dprime_full = {a: D[a] + self.allocator.beta * W_norm[a] for a in self.approaches}

            # order by Dprime then EVU_smooth
            order = sorted(self.approaches, key=lambda a: (Dprime_full[a], EVU_smooth[a]), reverse=True)

            # schedule times using intergreen
            t = 0.0
            t_start = {}
            t_end = {}
            for a in order:
                t_start[a] = t
                t_end[a] = t + float(G_alloc.get(a, 0.0))
                t = t_end[a] + DEFAULT_INTERGREEN

            # build rows for CSV (one per approach)
            params_snapshot = {
                "T_base": self.allocator.T_base,
                "G_min": min(self.allocator.G_min_default, min(compute_gmins_from_queue(per_app_queue).values())),
                "G_max": max(self.allocator.G_max_default, max(compute_gmax_from_cycle(cycle_time_dynamic).values())),
                "alpha_up": self.allocator.alpha_up,
                "alpha_down": self.allocator.alpha_down,
                "beta": self.allocator.beta,
                "W_th": self.allocator.W_th,
                "delta": self.allocator.delta,
                "T_alloc": self.allocator.T_alloc,
                "intergreen_L": DEFAULT_INTERGREEN
            }

            for a in self.approaches:
                row = format_csv_row_full(
                    signal_id=signal_id,
                    cycle=cycle,
                    approach=a,
                    EVU_raw=EVU_raw.get(a, 0.0),
                    EVU_smooth=EVU_smooth.get(a, 0.0),
                    queue_len=per_app_queue.get(a, 0.0),
                    arrival_rate=arrival_rates_vph.get(a, 0.0),
                    emergency=int(emergency_flags.get(a, False)),
                    D=D.get(a, 0.0),
                    W=W.get(a, 0.0),
                    W_norm=W_norm.get(a, 0.0),
                    Dprime=Dprime_full.get(a, 0.0),
                    G_allocated=G_alloc.get(a, 0.0),
                    order=order,
                    t_start=t_start.get(a, 0.0),
                    t_end=t_end.get(a, 0.0),
                    params_snapshot=params_snapshot
                )
                rows.append(row)

            cycle += 1
            frame_idx += 1
            extracted += 1
            if max_frames and extracted >= max_frames:
                break

        cap.release()

        # ensure at least one row
        if not rows:
            rows = [format_csv_row_full("SIGNAL_001", 0, "N", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ["N","E","S","W"], 0.0, 0.0, {})]

        # Save CSV
        df = pd.DataFrame(rows)
        ts = int(time.time())
        out_path = os.path.join(self.output_dir, f"signal_log_{ts}.csv")
        df.to_csv(out_path, index=False)

        # cleanup tmp youtube dir if present
        try:
            tmpdir = os.path.dirname(video_path)
            if tmpdir and tmpdir.startswith(tempfile.gettempdir()):
                shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

        return out_path
