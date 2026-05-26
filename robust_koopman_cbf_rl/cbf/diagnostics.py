"""Logs per-step filter diagnostics and computes summary statistics."""
from __future__ import annotations
import numpy as np


class DiagnosticsBuffer:
    def __init__(self):
        self.records: list[dict] = []

    def log(self, **kwargs) -> None:
        self.records.append(dict(kwargs))

    def reset(self) -> None:
        self.records.clear()

    def summary(self) -> dict:
        if not self.records:
            return {}
        keys = list(self.records[0].keys())
        arr = {k: np.asarray([r[k] for r in self.records], dtype=np.float64) for k in keys}
        out = {}
        if "intervention" in arr:
            out["intervention_rate"] = float(arr["intervention"].mean())
        if "slack" in arr:
            out["slack_rate"] = float((arr["slack"] > 1e-6).mean())
            out["slack_mean"] = float(arr["slack"].mean())
        if "cbf_gap" in arr:
            out["cbf_gap_min"] = float(arr["cbf_gap"].min())
            out["cbf_gap_mean"] = float(arr["cbf_gap"].mean())
        if "correction_norm" in arr:
            out["correction_norm_mean"] = float(arr["correction_norm"].mean())
        if "h_value" in arr:
            out["h_min"] = float(arr["h_value"].min())
        return out
