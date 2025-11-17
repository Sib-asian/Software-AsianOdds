"""
Performance tracker che aggiorna le reliability dei modelli sulla base degli
outcome salvati nel feature store.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from .feature_store import FeatureStore
from .model_registry import ModelRegistry


class PerformanceTracker:
    def __init__(
        self,
        registry: ModelRegistry,
        feature_store: FeatureStore,
        decay: float = 0.9,
        bootstrap_window: int = 500,
    ):
        self.registry = registry
        self.feature_store = feature_store
        self.decay = decay
        self.bootstrap_window = bootstrap_window
        self._bootstrap_from_history()

    # ------------------------------------------------------------------ #

    def _bootstrap_from_history(self):
        if self.bootstrap_window <= 0:
            return
        entries = self.feature_store.load_recent_entries(self.bootstrap_window)
        for entry in entries:
            actual = entry.get("actual_outcome")
            if actual is None:
                continue
            self.update_from_entry(entry, float(actual), bootstrap=True)

    def update_from_entry(
        self,
        entry: Dict[str, Any],
        actual_outcome: float,
        bootstrap: bool = False,
    ):
        predictions = entry.get("predictions") or {}
        if not predictions:
            return
        for model_name, prediction in predictions.items():
            try:
                prediction = float(prediction)
            except (TypeError, ValueError):
                continue
            error = abs(actual_outcome - prediction)
            score = 1.0 - min(error, 1.0)
            decay = 0.5 if bootstrap else self.decay
            self.registry.update_reliability(model_name, score, decay=decay)

    def update_from_match(
        self,
        match_id: str,
        actual_outcome: float,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        entry = self.feature_store.record_outcome(match_id, actual_outcome, extra_metadata)
        if entry:
            self.update_from_entry(entry, float(actual_outcome))
            return True
        return False


__all__ = ["PerformanceTracker"]
