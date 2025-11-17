"""
Adaptive orchestrator che coordina:
- estrazione feature di contesto
- meta optimizer per i pesi
- feature store
- performance tracker
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from .context_features import build_context_features
from .feature_store import FeatureStore
from .model_registry import ModelRegistry
from .performance_tracker import PerformanceTracker
from .meta_optimizer import MetaOptimizer

logger = logging.getLogger(__name__)


def build_match_id(match: Dict[str, Any]) -> str:
    home = (match.get("home") or "home").lower().strip()
    away = (match.get("away") or "away").lower().strip()
    date = (match.get("date") or match.get("kickoff") or match.get("start_time") or "na").split("T")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", f"{date}-{home}-{away}")
    return slug.strip("-")


class AdaptiveOrchestrator:
    def __init__(
        self,
        *,
        meta_learner,
        config: Optional[Dict[str, Any]] = None,
        feature_store: Optional[FeatureStore] = None,
        registry: Optional[ModelRegistry] = None,
        tracker: Optional[PerformanceTracker] = None,
    ):
        self.config = config or {}
        self.meta_learner = meta_learner
        store_path = Path(self.config.get("store_path") or (Path(__file__).resolve().parent.parent / "data" / "meta_feature_store.jsonl"))
        max_entries = int(self.config.get("max_entries", 50000))
        self.registry = registry or ModelRegistry()
        self.feature_store = feature_store or FeatureStore(store_path, max_entries=max_entries)
        reliability_decay = float(self.config.get("reliability_decay", 0.9))
        bootstrap_window = int(self.config.get("bootstrap_window", 1000))
        self.performance_tracker = tracker or PerformanceTracker(
            registry=self.registry,
            feature_store=self.feature_store,
            decay=reliability_decay,
            bootstrap_window=bootstrap_window,
        )
        exploration_rate = float(self.config.get("exploration_rate", 0.08))
        self.meta_optimizer = MetaOptimizer(
            meta_learner=self.meta_learner,
            registry=self.registry,
            exploration_rate=exploration_rate,
        )

    # ------------------------------------------------------------------ #

    def register_model(
        self,
        name: str,
        *,
        model_type: str,
        tags: Optional[list] = None,
        priority: int = 1,
        supports_live: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.registry.ensure(
            name=name,
            model_type=model_type,
            tags=tags,
            priority=priority,
            supports_live=supports_live,
            metadata=metadata,
        )

    def blend_predictions(
        self,
        match: Dict[str, Any],
        predictions: Dict[str, float],
        api_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        match_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not predictions:
            raise ValueError("No predictions provided to AdaptiveOrchestrator")

        match_id = match_id or build_match_id(match)
        context_features = build_context_features(match, predictions, api_context)
        weights, diagnostics, meta_confidence = self.meta_optimizer.optimize(
            match_data=match,
            predictions=predictions,
            api_context=api_context or {},
            context_features=context_features,
        )

        probability = float(
            np.clip(
                sum(predictions[name] * weights.get(name, 0.0) for name in predictions),
                0.01,
                0.99,
            )
        )
        uncertainty = float(np.std(list(predictions.values()))) if len(predictions) > 1 else 0.0
        reliability_snapshot = self.registry.snapshot_reliability()

        storage_metadata = {
            "context": {
                "data_quality": context_features.get("data_availability"),
                "model_agreement": context_features.get("model_agreement"),
            },
            "diagnostics": diagnostics,
            "extra": metadata or {},
        }
        self.feature_store.record_prediction(
            match_id=match_id,
            match=match,
            context_features=context_features,
            predictions=predictions,
            weights=weights,
            probability=probability,
            metadata=storage_metadata,
        )

        return {
            "match_id": match_id,
            "probability": probability,
            "weights": weights,
            "uncertainty": uncertainty,
            "meta_confidence": meta_confidence,
            "context_features": context_features,
            "reliability": reliability_snapshot,
            "diagnostics": diagnostics,
            "store_path": str(self.feature_store.filepath),
        }

    def register_outcome(
        self,
        match_id: str,
        actual_outcome: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        updated = self.performance_tracker.update_from_match(match_id, actual_outcome, metadata)
        if not updated:
            logger.debug("Outcome registration skipped - match_id %s not found", match_id)
        return updated


__all__ = ["AdaptiveOrchestrator", "build_match_id"]
