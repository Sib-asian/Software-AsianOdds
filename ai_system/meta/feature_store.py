"""
Lightweight JSONL-based feature store that keeps track of predictions, context
features and (eventuali) risultati reali per alimentare il layer meta-adattivo.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class FeatureStore:
    """
    Persistenza append-only su file .jsonl per non richiedere database esterni.
    """

    def __init__(self, filepath: Path, max_entries: int = 50000):
        self.filepath = Path(filepath)
        self.max_entries = max_entries
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            self.filepath.touch()

    # --------------------------------------------------------------------- #
    # Persistenza
    # --------------------------------------------------------------------- #

    def record_prediction(
        self,
        *,
        match_id: str,
        match: Dict[str, Any],
        context_features: Dict[str, float],
        predictions: Dict[str, float],
        weights: Dict[str, float],
        probability: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = {
            "match_id": match_id,
            "timestamp": time.time(),
            "match": match,
            "context_features": context_features,
            "predictions": predictions,
            "weights": weights,
            "probability": probability,
            "metadata": metadata or {},
            "actual_outcome": None,
        }
        self._append_entry(entry)
        return entry

    def record_outcome(
        self,
        match_id: str,
        actual_outcome: float,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        entries = self._load_entries()
        updated_entry = None
        for entry in entries:
            if entry.get("match_id") != match_id:
                continue
            entry["actual_outcome"] = float(actual_outcome)
            entry["metadata"] = entry.get("metadata") or {}
            if extra_metadata:
                entry["metadata"]["outcome_metadata"] = extra_metadata
            entry["metadata"]["updated_at"] = time.time()
            updated_entry = entry
            break

        if updated_entry:
            self._write_entries(entries)
        return updated_entry

    # --------------------------------------------------------------------- #
    # Lettura
    # --------------------------------------------------------------------- #

    def load_recent_entries(self, limit: int = 2500) -> List[Dict[str, Any]]:
        entries = self._load_entries()
        if limit <= 0:
            return entries
        return entries[-limit:]

    def load_entry(self, match_id: str) -> Optional[Dict[str, Any]]:
        for entry in reversed(self._load_entries()):
            if entry.get("match_id") == match_id:
                return entry
        return None

    def iter_entries(self, limit: Optional[int] = None):
        """
        Itera sulle entry presenti (opzionale limit per ultime N).
        """
        entries = self._load_entries()
        if limit is not None and limit > 0:
            entries = entries[-limit:]
        for entry in entries:
            yield entry

    def count_entries(self) -> int:
        """Numero approssimativo di entry salvate (conta le righe valide)."""
        count = 0
        with self.filepath.open("r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    count += 1
        return count

    def aggregate(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Calcola statistiche aggregate sul meta store (facoltativo limit).
        """
        entries = list(self.iter_entries(limit))
        if not entries:
            return {}

        total = len(entries)
        with_outcome = sum(1 for e in entries if e.get("actual_outcome") is not None)
        avg_prob = sum(e.get("probability", 0.0) for e in entries) / total

        diffs = [
            (e.get("probability", 0.0) - e.get("actual_outcome", 0.0)) ** 2
            for e in entries
            if e.get("actual_outcome") is not None
        ]
        prob_rmse = (sum(diffs) / len(diffs)) ** 0.5 if diffs else None

        def collect(field_name: str):
            buckets: Dict[str, List[float]] = {}
            for entry in entries:
                data = entry.get(field_name) or {}
                for key, value in data.items():
                    buckets.setdefault(key, []).append(float(value))
            return buckets

        def summarize(buckets: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
            summary = {}
            for key, values in buckets.items():
                if not values:
                    continue
                summary[key] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
            return summary

        return {
            "total_entries": total,
            "entries_with_outcome": with_outcome,
            "outcome_ratio": with_outcome / total,
            "avg_probability": avg_prob,
            "probability_rmse": prob_rmse,
            "weights": summarize(collect("weights")),
            "predictions": summarize(collect("predictions")),
            "context_features": summarize(collect("context_features")),
        }

    def aggregate_windows(self, windows: Optional[List[Optional[int]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calcola aggregate per più finestre (es. ultimi 50, 200, tutti).
        """
        windows = windows or [50, 200, None]
        result: Dict[str, Dict[str, Any]] = {}
        for window in windows:
            key = f"last_{window}" if window else "all"
            result[key] = self.aggregate(limit=window)
        return result

    # --------------------------------------------------------------------- #
    # Helpers privati
    # --------------------------------------------------------------------- #

    def _append_entry(self, entry: Dict[str, Any]):
        with self.filepath.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._trim_if_needed()

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        with self.filepath.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _write_entries(self, entries: List[Dict[str, Any]]):
        with self.filepath.open("w", encoding="utf-8") as fp:
            for entry in entries[-self.max_entries :]:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _trim_if_needed(self):
        if self.max_entries <= 0:
            return
        entries = self._load_entries()
        if len(entries) <= self.max_entries:
            return
        self._write_entries(entries)


__all__ = ["FeatureStore"]
