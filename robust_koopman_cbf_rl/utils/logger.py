"""Lightweight CSV + JSON logger."""
from __future__ import annotations
from pathlib import Path
import csv
import json


class CSVLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields = None
        self._fh = None
        self._writer = None

    def log(self, row: dict):
        import warnings
        if self._writer is None:
            self._fields = list(row.keys())
            self._fh = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._writer.writeheader()
        else:
            new_keys = set(row.keys()) - set(self._fields)
            if new_keys:
                warnings.warn(f"CSVLogger: new keys {new_keys} after header written; they will be dropped.")
        self._writer.writerow({k: row.get(k, "") for k in self._fields})
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def dump_json(path, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
