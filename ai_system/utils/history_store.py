"""
History store utilities to persist analysis rows on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _serialize_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    match = result.get("match", {})
    summary = result.get("summary", {})
    final_decision = result.get("final_decision", {})
    metadata = result.get("metadata", {})
    timing = result.get("timing", {})

    row = {
        "timestamp": metadata.get("timestamp"),
        "analysis_time": metadata.get("analysis_time_seconds"),
        "home": match.get("home"),
        "away": match.get("away"),
        "league": match.get("league"),
        "decision": final_decision.get("action"),
        "stake": final_decision.get("stake"),
        "timing": final_decision.get("timing"),
        "priority": final_decision.get("priority"),
        "market_regime": summary.get("market_regime"),
        "probability_calibrated": summary.get("probability_calibrated"),
        "probability_bayesian": summary.get("probability"),
        "confidence": summary.get("confidence"),
        "value_score": summary.get("value_score"),
        "expected_value": summary.get("expected_value"),
        "odds": summary.get("odds"),
        "movement_pattern": timing.get("movement_pattern"),
        "sharp_money": timing.get("sharp_money_detected"),
        "movement_trend": None,
    }

    movement = timing.get("movement")
    if isinstance(movement, dict):
        row["movement_trend"] = movement.get("trend")

    api_context = result.get("api_context", {})
    row["data_quality"] = api_context.get("metadata", {}).get("data_quality")

    # Store a compact JSON payload for debugging
    row["raw_summary"] = json.dumps(summary)

    return row


def append_analysis_to_history(result: Dict[str, Any], history_path: Path):
    """
    Appende l'analisi corrente su CSV persistente (creandolo se manca).
    """
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    row = _serialize_analysis(result)
    df = pd.DataFrame([row])
    header = not history_path.exists()
    df.to_csv(history_path, mode="a", index=False, header=header)
