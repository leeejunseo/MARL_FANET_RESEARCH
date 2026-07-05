import csv
from typing import Dict, Optional

import numpy as np


def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


class CsvLinkTraceProvider:
    """Provides per-step link matrices from a CSV trace.

    Expected columns:
    - step
    - i
    - j
    - connected
    - delay_ms
    - delivery
    - hop
    """

    def __init__(self, csv_path: str, num_drones: int, strict: bool = False):
        self.csv_path = csv_path
        self.num_drones = int(num_drones)
        self.strict = bool(strict)
        self._steps: Dict[int, Dict[str, np.ndarray]] = {}
        self._load()

    def _make_slot(self):
        connected = np.zeros((self.num_drones, self.num_drones), dtype=bool)
        np.fill_diagonal(connected, True)

        delay_ms = np.full((self.num_drones, self.num_drones), 300.0, dtype=float)
        np.fill_diagonal(delay_ms, 0.0)

        delivery = np.zeros((self.num_drones, self.num_drones), dtype=float)
        np.fill_diagonal(delivery, 1.0)

        hop = np.full((self.num_drones, self.num_drones), np.inf, dtype=float)
        np.fill_diagonal(hop, 0.0)

        return {
            "connected": connected,
            "delay_ms": delay_ms,
            "delivery": delivery,
            "hop": hop,
        }

    def _load(self):
        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row["step"])
                i = int(row["i"])
                j = int(row["j"])

                if i < 0 or i >= self.num_drones or j < 0 or j >= self.num_drones:
                    if self.strict:
                        raise ValueError(f"Invalid node index in trace: i={i}, j={j}")
                    continue

                slot = self._steps.setdefault(step, self._make_slot())
                slot["connected"][i, j] = _as_bool(row.get("connected", "0"))

                if row.get("delay_ms") not in (None, ""):
                    slot["delay_ms"][i, j] = float(row["delay_ms"])

                if row.get("delivery") not in (None, ""):
                    slot["delivery"][i, j] = float(row["delivery"])

                if row.get("hop") not in (None, ""):
                    hop_val = float(row["hop"])
                    slot["hop"][i, j] = hop_val if hop_val >= 0.0 else np.inf

        for step, slot in self._steps.items():
            c = slot["connected"]
            c_sym = np.logical_or(c, c.T)
            np.fill_diagonal(c_sym, True)
            slot["connected"] = c_sym

            for key in ["delay_ms", "delivery", "hop"]:
                m = slot[key]
                m_sym = np.where(np.isfinite(m), m, m.T)
                m_sym = np.where(np.isfinite(m_sym), m_sym, m)
                m_sym = np.where(np.isfinite(m_sym), m_sym, m.T)
                slot[key] = m_sym

    def has_step(self, step_idx: int) -> bool:
        return step_idx in self._steps

    def get_step_data(self, step_idx: int, positions=None, malicious_ids=None) -> Optional[Dict[str, np.ndarray]]:
        return self._steps.get(step_idx)
