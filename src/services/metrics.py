# src/services/metrics.py
"""
Full traffic metrics and allocator implementation + small wrappers expected by app.py.

Exports expected by app.py:
 - arrival_rate_estimator
 - emergency_vehicle_flag
 - signal_allocation
 - format_csv_row
 - TrafficAllocator (class)
 - format_csv_row_full
 - compute_evu_from_counts, etc.
"""

from typing import Dict, List, Tuple, Optional
import math
import time
from datetime import datetime
import numpy as np

# Basic config defaults (tune per-site)
DEFAULT_SAT_FLOW = 1800.0  # vehicles/hour per lane (saturation flow)
DEFAULT_HEADWAY = 2.0      # seconds per vehicle (approx)
DEFAULT_LOST_PER_PHASE = 4.0  # s amber+clear per phase
DEFAULT_INTERGREEN = 2.0
DEFAULT_CYCLE_BASE = 90.0  # baseline cycle length to anchor allocations if needed

# EVU weights
CLASS_WEIGHTS = {
    "car": 1.0, "bike": 0.5, "motorbike": 0.5, "bicycle": 0.5,
    "bus": 3.0, "truck": 3.0, "person": 0.2, "ambulance": 3.0, "police": 3.0
}


# ------------------------
# Low-level helpers
# ------------------------
def compute_evu_from_counts(counts: Dict[str, Dict[str, int]], weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    counts: {approach: {class_name: count}}
    returns EVU per approach
    """
    w = weights or CLASS_WEIGHTS
    evu = {}
    for a, cls_map in counts.items():
        s = 0.0
        for k, v in cls_map.items():
            s += w.get(k, 1.0) * v
        evu[a] = float(s)
    return evu


def asymmetric_ema(evu_raw: Dict[str, float], evu_prev: Dict[str, float], alpha_up: float, alpha_down: float) -> Dict[str, float]:
    """
    Apply asymmetric exponential smoothing per approach.
    """
    out = {}
    for a, val in evu_raw.items():
        prev = evu_prev.get(a, 0.0)
        if val > prev:
            out[a] = alpha_up * val + (1 - alpha_up) * prev
        else:
            out[a] = alpha_down * val + (1 - alpha_down) * prev
    return out


def arrival_rate_estimator(prev_count: int, curr_count: int, dt_seconds: float) -> float:
    """
    Returns arrival rate in vehicles per minute (aggregated).
    prev_count/curr_count are total vehicle counts sampled at two times.
    This is an alias/wrapper used by app.py (keeps previous semantics).
    """
    if dt_seconds <= 0:
        return 0.0
    diff = max(0, curr_count - prev_count)
    # vehicles per second -> per minute
    return (diff / dt_seconds) * 60.0


# ------------------------
# Webster cycle time computation (dynamic)
# ------------------------
def compute_cycle_time_webster(flows: Dict[str, float], sat_flow: float = DEFAULT_SAT_FLOW, lost_time_per_phase: float = DEFAULT_LOST_PER_PHASE) -> float:
    """
    Compute cycle time using Webster's formula:
      C0 = (1.5L + 5) / (1 - Y)
    where L = lost time per cycle (sum of intergreen per phase),
    Y = sum(flow_i / s_i) (flow ratio).
    flows: vehicles per hour per approach (q_i)
    s_i: saturation flow per approach (we pass same sat_flow by default)
    """
    s = sat_flow
    Y = sum((flows.get(a, 0.0) / max(1e-6, s)) for a in flows.keys())
    L = lost_time_per_phase * len(flows)
    if Y >= 0.95:
        return float(max(60.0, DEFAULT_CYCLE_BASE))
    C0 = (1.5 * L + 5.0) / (1.0 - Y)
    return float(max(30.0, min(180.0, C0)))


# ------------------------
# G_min and G_max calculators
# ------------------------
def compute_gmins_from_queue(queue_lengths_meters: Dict[str, float], avg_vehicle_length_m: float = 4.5, headway: float = DEFAULT_HEADWAY, sat_flow: float = DEFAULT_SAT_FLOW) -> Dict[str, float]:
    """
    Estimate minimal green needed to discharge the queued vehicles.
    queue_lengths_meters: approximate queue length in meters per approach
    """
    gmins = {}
    for a, q_m in queue_lengths_meters.items():
        nveh = q_m / max(0.1, avg_vehicle_length_m)  # estimated vehicles queued
        g = nveh * headway  # seconds needed to discharge
        gmins[a] = float(max(3.0, g))  # at least 3s
    return gmins


def compute_gmax_from_cycle(cycle_time: float, reserve: float = 5.0) -> Dict[str, float]:
    """
    Compute a per-approach G_max as a fraction of cycle (ensures not all green to a single phase).
    """
    gmax = {}
    max_allowed = max(10.0, 0.7 * cycle_time - reserve)
    for a in ["N", "E", "S", "W"]:
        gmax[a] = float(max_allowed)
    return gmax


# ------------------------
# Allocation algorithm (full)
# ------------------------
class TrafficAllocator:
    """
    Implements allocator with:
    - asymmetric EMA smoothing of EVU
    - D = EVU_smooth / C_i (C_i = capacity baseline)
    - W_norm (from queue lengths) and Dprime = D + beta * W_norm
    - proportional allocation of cycle time to Dprime
    - clamp to [G_min, G_max], renormalize iteratively
    - starvation correction (delta, T_alloc)
    - emergency preemption
    """

    def __init__(self,
                 approaches: List[str] = ["N", "E", "S", "W"],
                 C_capacity: Dict[str, float] = None,
                 alpha_up: float = 0.8,
                 alpha_down: float = 0.25,
                 beta: float = 0.6,
                 T_base: float = DEFAULT_CYCLE_BASE,
                 G_min_default: float = 5.0,
                 G_max_default: float = 60.0,
                 W_th: float = 45.0,
                 delta: float = 0.15,
                 T_alloc: float = 10.0):
        self.approaches = approaches
        self.C_capacity = C_capacity or {a: 10.0 for a in approaches}
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.beta = beta
        self.T_base = float(T_base)
        self.G_min_default = float(G_min_default)
        self.G_max_default = float(G_max_default)
        self.W_th = float(W_th)
        self.delta = delta
        self.T_alloc = float(T_alloc)

        # state
        self.EVU_prev = {a: 0.0 for a in self.approaches}
        self.W_cum = {a: 0.0 for a in self.approaches}

    def step(self,
             EVU_raw: Dict[str, float],
             queue_len: Dict[str, float],
             arrival_rates_vph: Dict[str, float],
             emergency_flags: Dict[str, bool] = None,
             cycle_time_dynamic: Optional[float] = None) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        EVU_raw: raw EVU per approach
        queue_len: queue length measure per approach (meters or proxy)
        arrival_rates_vph: arrival flow per approach (vehicles per hour)
        emergency_flags: per approach boolean
        cycle_time_dynamic: optional dynamic cycle length (if None, uses T_base)
        Returns: EVU_smooth, Dprime, G_allocated
        """
        emergency_flags = emergency_flags or {a: False for a in self.approaches}
        C_cycle = float(cycle_time_dynamic if cycle_time_dynamic is not None else self.T_base)

        # 1) Smooth EVU
        EVU_smooth = {}
        for a in self.approaches:
            prev = self.EVU_prev.get(a, 0.0)
            cur = EVU_raw.get(a, 0.0)
            if cur > prev:
                val = self.alpha_up * cur + (1 - self.alpha_up) * prev
            else:
                val = self.alpha_down * cur + (1 - self.alpha_down) * prev
            EVU_smooth[a] = float(val)
            self.EVU_prev[a] = float(val)

        # 2) Demand D = EVU_smooth / capacity_Ci
        D = {a: (EVU_smooth[a] / max(1e-6, self.C_capacity.get(a, 10.0))) for a in self.approaches}

        # 3) Wait proxy W (use queue lengths directly) and normalize
        W = {a: float(queue_len.get(a, 0.0)) for a in self.approaches}
        Wmax = max(1e-6, max(W.values()))
        W_norm = {a: W[a] / Wmax for a in self.approaches}

        # 4) Dprime
        Dprime = {a: D[a] + self.beta * W_norm[a] for a in self.approaches}

        # 5) Emergency handling
        if any(emergency_flags.get(a, False) for a in self.approaches):
            G = {a: (C_cycle if emergency_flags.get(a, False) else 0.0) for a in self.approaches}
            for a in self.approaches:
                if emergency_flags.get(a, False):
                    self.W_cum[a] = 0.0
                else:
                    self.W_cum[a] += C_cycle
            return EVU_smooth, Dprime, G

        # 6) Dynamic cycle
        if arrival_rates_vph and any(arrival_rates_vph.values()):
            Ct = compute_cycle_time_webster(arrival_rates_vph, sat_flow=DEFAULT_SAT_FLOW, lost_time_per_phase=DEFAULT_LOST_PER_PHASE)
            C_cycle = float(Ct)

        # 7) Initial proportional allocation by Dprime
        sumD = sum(Dprime.values()) + 1e-6
        G = {}
        for a in self.approaches:
            g = (Dprime[a] / sumD) * C_cycle
            G[a] = float(g)

        # 8) Compute G_min per approach from queue
        gmins = compute_gmins_from_queue(queue_len)
        for a in self.approaches:
            if gmins.get(a, 0.0) < self.G_min_default:
                gmins[a] = self.G_min_default

        # 9) G_max from cycle
        gmaxs = compute_gmax_from_cycle(C_cycle)

        # 10) Clamp to [G_min, G_max]
        for a in self.approaches:
            G[a] = float(max(gmins.get(a, self.G_min_default), min(gmaxs.get(a, self.G_max_default), G[a])))

        # 11) Renormalize to preserve total cycle time (accounting intergreen)
        def renorm(Gmap):
            total = sum(Gmap.values()) + 1e-6
            scale = (C_cycle - DEFAULT_INTERGREEN * len(Gmap)) / total
            newG = {}
            for k, v in Gmap.items():
                new_v = float(max(gmins.get(k, self.G_min_default), min(gmaxs.get(k, self.G_max_default), v * scale)))
                newG[k] = new_v
            return newG

        G = renorm(G)

        # 12) Starvation correction
        long_waiters = [a for a in self.approaches if self.W_cum.get(a, 0.0) > self.W_th]
        if long_waiters:
            sum_over = sum([(self.W_cum[a] - self.W_th) for a in long_waiters]) + 1e-6
            for a in long_waiters:
                add = self.delta * ((self.W_cum[a] - self.W_th) / sum_over) * self.T_alloc
                G[a] += add
            for _ in range(4):
                G = renorm(G)

        # 13) Update cumulative waits
        winners = [a for a in self.approaches if G[a] > min(gmins.get(a, self.G_min_default), 0.99)]
        for a in self.approaches:
            if a in winners:
                self.W_cum[a] = 0.0
            else:
                self.W_cum[a] += G.get(a, 0.0)

        G = {a: float(G[a]) for a in self.approaches}
        return EVU_smooth, Dprime, G


# ------------------------
# CSV Row formatter producing expected columns
# ------------------------
def format_csv_row_full(signal_id: str,
                        cycle: int,
                        approach: str,
                        EVU_raw: float,
                        EVU_smooth: float,
                        queue_len: float,
                        arrival_rate: float,
                        emergency: int,
                        D: float,
                        W: float,
                        W_norm: float,
                        Dprime: float,
                        G_allocated: float,
                        order: List[str],
                        t_start: float,
                        t_end: float,
                        params_snapshot: Dict = None) -> Dict:
    """
    Return dictionary matching expected CSV columns.
    params_snapshot: dictionary of allocator params to include for audit.
    """
    params_snapshot = params_snapshot or {}
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "signal_id": signal_id,
        "cycle": int(cycle),
        "approach": approach,
        "C": float(params_snapshot.get("T_base", DEFAULT_CYCLE_BASE)),
        "EVU_raw": float(EVU_raw),
        "EVU_smooth": float(EVU_smooth),
        "queue_len": float(queue_len),
        "arrival_rate": float(arrival_rate),
        "emergency": int(emergency),
        "D": float(D),
        "W": float(W),
        "W_norm": float(W_norm),
        "Dprime": float(Dprime),
        "G_allocated": float(G_allocated),
        "order_1": order[0] if len(order) > 0 else "",
        "order_2": order[1] if len(order) > 1 else "",
        "order_3": order[2] if len(order) > 2 else "",
        "order_4": order[3] if len(order) > 3 else "",
        "t_start": float(t_start),
        "t_end": float(t_end),
        "intergreen_L": float(params_snapshot.get("intergreen_L", DEFAULT_INTERGREEN)),
        # snapshot of allocator params
        "T_base": float(params_snapshot.get("T_base", DEFAULT_CYCLE_BASE)),
        "G_min": float(params_snapshot.get("G_min", 5.0)),
        "G_max": float(params_snapshot.get("G_max", 60.0)),
        "alpha_up": float(params_snapshot.get("alpha_up", 0.8)),
        "alpha_down": float(params_snapshot.get("alpha_down", 0.25)),
        "beta": float(params_snapshot.get("beta", 0.6)),
        "W_th": float(params_snapshot.get("W_th", 45.0)),
        "delta": float(params_snapshot.get("delta", 0.15)),
        "T_alloc": float(params_snapshot.get("T_alloc", 10.0))
    }
    return row


# ------------------------
# Small wrappers expected by app.py
# ------------------------
def emergency_vehicle_flag(detections: List[Dict]) -> bool:
    """
    detections: list of dicts that may have 'class' or 'name' keys.
    Returns True if ambulance/police/firetruck-like label found.
    """
    if not detections:
        return False
    emergency_names = {"ambulance", "police", "firetruck", "emergency", "siren"}
    for d in detections:
        # tolerant lookup
        name = None
        if isinstance(d, dict):
            name = d.get("class") or d.get("name") or d.get("label") or d.get("cls")
        else:
            name = str(d)
        if name is None:
            continue
        if isinstance(name, (int, float)):
            name = str(int(name))
        n = str(name).lower()
        if any(en in n for en in emergency_names):
            return True
    return False


def signal_allocation(queue_lengths: Dict[str, float], cycle_time: float = 60.0, G_min: float = 5.0, G_max: float = 60.0) -> Dict[str, float]:
    """
    Simple wrapper used by older pipeline/app code.
    Given queue_lengths (meters or counts) -> return per-approach green times using TrafficAllocator.
    """
    approaches = ["N", "E", "S", "W"]
    # Convert queue_lengths into EVU proxy (simple proportional)
    EVU_raw = {a: float(queue_lengths.get(a, 0.0)) for a in approaches}
    # arrival rates unknown here -> pass zeros
    arrival_rates = {a: 0.0 for a in approaches}

    allocator = TrafficAllocator(approaches=approaches, T_base=cycle_time, G_min_default=G_min, G_max_default=G_max)
    EVU_smooth, Dprime, G = allocator.step(EVU_raw, queue_lengths, arrival_rates, emergency_flags=None, cycle_time_dynamic=cycle_time)
    return G


def format_csv_row(*args, **kwargs):
    """
    Backwards-compatible alias used by app.py — forwards to format_csv_row_full.
    Keep signature flexible: expects (signal_id, cycle, approach, ...) per usage.
    """
    return format_csv_row_full(*args, **kwargs)


# ------------------------
# Lightweight TrafficPipeline convenience wrapper
# ------------------------
class TrafficPipeline:
    """
    Minimal wrapper around allocator + formatting to provide a run-like API.
    It DOES NOT perform detection. It expects caller to provide per-cycle counts/queues
    or you can extend it to call a detector externally.
    """

    def __init__(self, allocator: Optional[TrafficAllocator] = None):
        self.allocator = allocator or TrafficAllocator()

    def allocate_from_counts(self, counts_per_approach: Dict[str, Dict[str, int]], queue_lengths: Dict[str, float], arrival_rates_vph: Dict[str, float], cycle_time: Optional[float] = None):
        """
        counts_per_approach: {'N': {'car': 3, 'bus':1}, ...}
        queue_lengths: meters per approach
        arrival_rates_vph: vehicles per hour per approach (optional)
        Returns DataFrame-like list of rows (one row per approach)
        """
        EVU_raw = compute_evu_from_counts(counts_per_approach)
        ct = float(cycle_time) if cycle_time is not None else self.allocator.T_base
        EVU_smooth, Dprime, G = self.allocator.step(EVU_raw, queue_lengths, arrival_rates_vph or {a: 0.0 for a in EVU_raw.keys()}, cycle_time_dynamic=ct)
        # build rows
        rows = []
        approaches = list(EVU_raw.keys())
        # compute intermediates
        D = {a: (EVU_smooth[a] / max(1e-6, self.allocator.C_capacity.get(a, 10.0))) for a in approaches}
        W = {a: queue_lengths.get(a, 0.0) for a in approaches}
        Wmax = max(1e-6, max(W.values()))
        W_norm = {a: W[a] / Wmax for a in approaches}
        Dprime_full = {a: D[a] + self.allocator.beta * W_norm[a] for a in approaches}
        order = sorted(approaches, key=lambda a: (Dprime_full[a], EVU_smooth[a]), reverse=True)
        t = 0.0
        t_start = {}; t_end = {}
        for a in order:
            t_start[a] = t
            t_end[a] = t + float(G.get(a, 0.0))
            t = t_end[a] + DEFAULT_INTERGREEN
        params = {
            "T_base": self.allocator.T_base,
            "G_min": self.allocator.G_min_default,
            "G_max": self.allocator.G_max_default,
            "alpha_up": self.allocator.alpha_up,
            "alpha_down": self.allocator.alpha_down,
            "beta": self.allocator.beta,
            "W_th": self.allocator.W_th,
            "delta": self.allocator.delta,
            "T_alloc": self.allocator.T_alloc,
            "intergreen_L": DEFAULT_INTERGREEN
        }
        for a in approaches:
            row = format_csv_row_full(
                signal_id="SIGNAL_001",
                cycle=0,
                approach=a,
                EVU_raw=EVU_raw.get(a, 0.0),
                EVU_smooth=EVU_smooth.get(a, 0.0),
                queue_len=W.get(a, 0.0),
                arrival_rate=arrival_rates_vph.get(a, 0.0) if arrival_rates_vph else 0.0,
                emergency=0,
                D=D.get(a, 0.0),
                W=W.get(a, 0.0),
                W_norm=W_norm.get(a, 0.0),
                Dprime=Dprime_full.get(a, 0.0),
                G_allocated=G.get(a, 0.0),
                order=order,
                t_start=t_start.get(a, 0.0),
                t_end=t_end.get(a, 0.0),
                params_snapshot=params
            )
            rows.append(row)
        return rows


# Export list
__all__ = [
    "compute_evu_from_counts",
    "asymmetric_ema",
    "arrival_rate_estimator",
    "compute_cycle_time_webster",
    "compute_gmins_from_queue",
    "compute_gmax_from_cycle",
    "TrafficAllocator",
    "format_csv_row_full",
    "emergency_vehicle_flag",
    "signal_allocation",
    "format_csv_row",
    "TrafficPipeline",
]
