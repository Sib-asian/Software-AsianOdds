"""
Meta optimizer che utilizza il MetaLearner di base e applica correzioni
contestuali (reliability, exploration) per ottenere i pesi finali.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np

from ..models.meta_learner import MetaLearner
from .model_registry import ModelRegistry


class MetaOptimizer:
    def __init__(
        self,
        meta_learner: MetaLearner,
        registry: ModelRegistry,
        exploration_rate: float = 0.08,
        min_weight: float = 0.02,
    ):
        self.meta_learner = meta_learner
        self.registry = registry
        self.exploration_rate = exploration_rate
        self.min_weight = min_weight

    def optimize(
        self,
        match_data: Dict[str, Any],
        predictions: Dict[str, float],
        api_context: Dict[str, Any],
        context_features: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, Any], float]:
        """
        Ritorna (pesi_finali, diagnostics, meta_confidence)
        """
        base_weights = self.meta_learner.calculate_weights(predictions, match_data, api_context)
        refined_weights, adjustments = self._refine_weights(base_weights, context_features)
        meta_confidence = self._estimate_confidence(context_features, refined_weights)
        diagnostics = {
            "base": base_weights,
            "refined": refined_weights,
            "adjustments": adjustments,
        }
        return refined_weights, diagnostics, meta_confidence

    # ------------------------------------------------------------------ #

    def _refine_weights(
        self,
        base_weights: Dict[str, float],
        context_features: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        data_quality = context_features.get("data_availability", 0.5)
        adjustments: Dict[str, Dict[str, float]] = {}
        adjusted = {}
        total = 0.0

        for model, weight in base_weights.items():
            reliability = self.registry.get_reliability(model)
            reliability_factor = 0.5 + reliability * 0.5
            context_factor = 0.75 + data_quality * 0.25
            new_weight = max(weight * reliability_factor * context_factor, self.min_weight)
            adjusted[model] = new_weight
            total += new_weight
            adjustments[model] = {
                "reliability": reliability,
                "reliability_factor": reliability_factor,
                "context_factor": context_factor,
            }

        # Normalizzazione e piccola esplorazione
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        if self.exploration_rate > 0 and len(adjusted) > 0:
            bonus = self.exploration_rate / len(adjusted)
            adjusted = {
                k: ((1 - self.exploration_rate) * v + bonus) for k, v in adjusted.items()
            }

        # Re-normalize to guard numerical drift
        normalization = sum(adjusted.values())
        if normalization > 0:
            adjusted = {k: v / normalization for k, v in adjusted.items()}

        return adjusted, adjustments

    def _estimate_confidence(
        self,
        context_features: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        agreement = context_features.get("model_agreement", 0.5)
        data_quality = context_features.get("data_availability", 0.5)
        reliability_avg = np.mean(list(weights.values()))  # weights already weighed by reliability
        confidence = 40.0
        confidence += agreement * 25
        confidence += data_quality * 20
        confidence += reliability_avg * 15
        return float(np.clip(confidence, 0, 100))


__all__ = ["MetaOptimizer"]
