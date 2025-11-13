"""
Odds Anomaly Detector
=====================

Identifies unusual movements in odds history, highlighting potential sharp
money, news leakage, or suspicious volatility that warrants human review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OddsMovement:
    time: str
    odds: float
    price_delta_pct: float
    volume: Optional[float] = None


class OddsAnomalyDetector:
    """Rule-based odds anomaly detection tailored for free tier data."""

    def __init__(self, config) -> None:
        self.config = config

    def detect(
        self,
        odds_history: Optional[List[Dict[str, Any]]],
        timing_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not getattr(self.config, "anomaly_detection_enabled", True):
            return {"status": "disabled", "anomalies": []}

        odds_history = odds_history or []
        if len(odds_history) < self.config.anomaly_min_points:
            return {"status": "insufficient_data", "anomalies": []}

        movements = self._prepare_movements(odds_history)
        if not movements:
            return {"status": "insufficient_data", "anomalies": []}

        anomalies = []
        for movement in movements:
            if abs(movement.price_delta_pct) >= self.config.anomaly_pct_threshold:
                anomalies.append(self._format_price_anomaly(movement))
            if movement.volume and movement.volume >= self.config.anomaly_volume_threshold:
                anomalies.append(self._format_volume_anomaly(movement))

        volatility_score = self._compute_volatility_score(movements)
        status = "volatile" if volatility_score >= self.config.anomaly_volatility_alert else "stable"

        result = {
            "status": status,
            "anomalies": anomalies,
            "volatility_score": round(volatility_score, 3),
            "sample_size": len(movements),
        }

        if timing_context:
            result["timing_alignment"] = timing_context.get("timing_recommendation")

        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _prepare_movements(self, odds_history: List[Dict[str, Any]]) -> List[OddsMovement]:
        cleaned = []
        for entry in odds_history:
            odds = entry.get("odds")
            time = entry.get("time")
            if odds is None or odds <= 0 or not time:
                continue
            cleaned.append(
                {
                    "time": time,
                    "odds": float(odds),
                    "volume": entry.get("volume"),
                }
            )

        if len(cleaned) < 2:
            return []

        movements: List[OddsMovement] = []
        for prev, current in zip(cleaned, cleaned[1:]):
            delta_pct = (current["odds"] - prev["odds"]) / prev["odds"]
            movements.append(
                OddsMovement(
                    time=current["time"],
                    odds=current["odds"],
                    price_delta_pct=delta_pct,
                    volume=current.get("volume"),
                )
            )
        return movements

    def _format_price_anomaly(self, movement: OddsMovement) -> Dict[str, Any]:
        direction = "sharp drop" if movement.price_delta_pct < 0 else "sharp rise"
        return {
            "type": "price",
            "time": movement.time,
            "severity": min(1.0, abs(movement.price_delta_pct) / (self.config.anomaly_pct_threshold * 2)),
            "description": f"{direction} di {movement.price_delta_pct:+.1%} (quota {movement.odds:.2f})",
        }

    def _format_volume_anomaly(self, movement: OddsMovement) -> Dict[str, Any]:
        return {
            "type": "volume",
            "time": movement.time,
            "severity": min(1.0, movement.volume / (self.config.anomaly_volume_threshold * 2)),
            "description": f"Volume spike x{movement.volume:.1f} rispetto media",
        }

    def _compute_volatility_score(self, movements: List[OddsMovement]) -> float:
        deltas = [mv.price_delta_pct for mv in movements]
        if not deltas:
            return 0.0

        volatility = pstdev(deltas) if len(deltas) > 1 else abs(deltas[0])
        mean_delta = mean(deltas)

        score = min(1.0, abs(volatility) / max(self.config.anomaly_pct_threshold, 1e-6))
        if abs(mean_delta) >= self.config.anomaly_pct_threshold:
            score = min(1.0, score + 0.2)

        return score

