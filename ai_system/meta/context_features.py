"""
Context feature engineering utilities used by the meta-adaptive layer.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import numpy as np

CONTEXT_FEATURE_NAMES: List[str] = [
    "league_quality",
    "data_availability",
    "historical_matches",
    "model_agreement",
    "time_to_kickoff",
    "h2h_relevance",
    "injuries_impact",
    "form_reliability",
    "is_top_league",
    "season_progress",
]


def encode_league_quality(league: Optional[str]) -> float:
    """Map league names to a normalized quality score."""
    if not league:
        return 0.75
    league = league.lower()
    scores = {
        "serie a": 0.90,
        "premier league": 0.95,
        "la liga": 0.92,
        "bundesliga": 0.90,
        "ligue 1": 0.85,
        "champions league": 1.00,
        "serie b": 0.70,
        "championship": 0.75,
    }
    for key, value in scores.items():
        if key in league:
            return value
    return 0.75


def build_context_features(
    match_data: Dict[str, Any],
    predictions: Optional[Dict[str, float]] = None,
    api_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Produce the feature dictionary used by the meta-learner and adaptive layer.
    """
    features: Dict[str, float] = {}

    league_quality = encode_league_quality(match_data.get("league"))
    features["league_quality"] = league_quality
    features["is_top_league"] = 1.0 if league_quality >= 0.85 else 0.0

    if api_context:
        metadata = api_context.get("metadata", {})
        features["data_availability"] = float(metadata.get("data_quality", 0.5))
        match_extra = api_context.get("match_data", {})
        h2h_info = match_extra.get("h2h", {})
        features["historical_matches"] = min(float(h2h_info.get("total", 5)) / 20.0, 1.0)
        features["h2h_relevance"] = min(float(h2h_info.get("total", 0)) / 10.0, 1.0)

        home_injuries = len(api_context.get("home_context", {}).get("data", {}).get("injuries", []) or [])
        away_injuries = len(api_context.get("away_context", {}).get("data", {}).get("injuries", []) or [])
        features["injuries_impact"] = min((home_injuries + away_injuries) / 10.0, 1.0)

        features["form_reliability"] = 0.8
    else:
        features["data_availability"] = 0.3
        features["historical_matches"] = 0.2
        features["h2h_relevance"] = 0.3
        features["injuries_impact"] = 0.0
        features["form_reliability"] = 0.3

    if predictions:
        pred_values = list(predictions.values())
        if len(pred_values) > 1:
            features["model_agreement"] = max(0.0, 1.0 - min(float(np.std(pred_values)), 1.0))
        else:
            features["model_agreement"] = 0.5
    else:
        features["model_agreement"] = 0.5

    hours_to_kickoff = match_data.get("hours_to_kickoff", 24)
    try:
        hours_to_kickoff = float(hours_to_kickoff)
    except (TypeError, ValueError):
        hours_to_kickoff = 24.0
    features["time_to_kickoff"] = min(hours_to_kickoff / 48.0, 1.0)

    features["season_progress"] = float(match_data.get("season_progress", 0.5))

    return features


def context_dict_to_array(
    context_features: Dict[str, float],
    feature_order: Optional[List[str]] = None,
) -> np.ndarray:
    """Convert a context feature dict into a numpy array with stable ordering."""
    feature_order = feature_order or CONTEXT_FEATURE_NAMES
    return np.array([float(context_features.get(name, 0.0)) for name in feature_order], dtype=np.float32)


__all__ = [
    "CONTEXT_FEATURE_NAMES",
    "build_context_features",
    "context_dict_to_array",
    "encode_league_quality",
]
