"""
Meta layer health evaluation utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .feature_store import FeatureStore
from .model_registry import ModelRegistry


def _build_alert(code: str, level: str, message: str, action: Optional[str] = None) -> Dict[str, Any]:
    alert = {
        "code": code,
        "level": level,
        "message": message,
    }
    if action:
        alert["action"] = action
    return alert


def evaluate_meta_health(
    store: FeatureStore,
    registry: ModelRegistry,
    *,
    limit: int = 500,
    exploration_rate: Optional[float] = None,
) -> Dict[str, Any]:
    summary = store.aggregate(limit=limit)
    window_summaries = store.aggregate_windows()
    reliability = registry.snapshot_reliability()
    history = registry.snapshot_history(limit=min(50, limit))
    alerts: List[Dict[str, Any]] = []

    total = summary.get("total_entries", 0) if summary else 0
    outcome_ratio = summary.get("outcome_ratio", 0.0) if summary else 0.0
    rmse = summary.get("probability_rmse") if summary else None

    if total == 0:
        alerts.append(_build_alert("meta_store_empty", "critical", "Il meta feature store è vuoto.", "Esegui analisi per popolarlo."))
    elif total < 25:
        alerts.append(_build_alert("few_meta_entries", "warning", f"Solo {total} entry presenti nel meta store.", "Aumenta il numero di analisi registrate."))

    if total > 0 and outcome_ratio < 0.1:
        alerts.append(_build_alert("low_outcome_feedback", "warning", f"Solo il {outcome_ratio:.0%} delle entry ha outcome registrati.", "Automatizza la registrazione dei risultati reali."))

    if rmse is not None and rmse > 0.25:
        alerts.append(_build_alert("high_probability_rmse", "warning", f"RMSE delle probabilità elevato ({rmse:.2f}).", "Verifica calibrazione e qualità dei dati."))

    if summary:
        weights = summary.get("weights") or {}
        for model, stats in weights.items():
            if stats["max"] < 0.2:
                alerts.append(_build_alert(f"low_weight_{model}", "info", f"Il modello {model} raramente contribuisce (peso max {stats['max']:.2f}).", "Valuta se riaddestrare o rimuovere il modello."))

    health = {
        "store_path": str(store.filepath),
        "entries": total,
        "exploration_rate": exploration_rate,
        "summary": summary,
        "reliability": reliability,
        "reliability_history": history,
        "window_summaries": window_summaries,
        "alerts": alerts,
    }
    return health


__all__ = ["evaluate_meta_health"]
