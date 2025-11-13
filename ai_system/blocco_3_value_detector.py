"""
BLOCCO 3: Value Detector
=========================

Rileva true value bets vs traps usando XGBoost + odds movement analysis.

Funzionalità:
- ML classification per TRUE_VALUE vs TRAP vs UNCERTAIN
- Odds movement pattern recognition
- Sharp money detection
- Expected Value calculation con adjustments
- Similar bets historical ROI analysis

Input: prob calibrata + confidence + odds + history
Output: value score + EV + classification + reasoning
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import joblib
from datetime import datetime

# XGBoost
try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("⚠️ XGBoost not available. Install with: pip install xgboost")

from .config import AIConfig

logger = logging.getLogger(__name__)


class ValueDetector:
    """
    Detector di value bets usando XGBoost.

    Distingue tra:
    - TRUE_VALUE: Vero value bet (alta probabilità profitto)
    - TRAP: Falso value bet (bookmaker trap)
    - UNCERTAIN: Incerto (evitare)
    """

    def __init__(self, config: Optional[AIConfig] = None):
        """Inizializza Value Detector"""
        if not XGBOOST_AVAILABLE:
            logger.warning("⚠️ XGBoost not available, using rule-based detection")

        self.config = config or AIConfig()
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

        logger.info("✅ Value Detector initialized")

    def extract_features(
        self,
        calibrated_result: Dict,
        confidence_result: Dict,
        odds_data: Dict,
        historical_context: Dict
    ) -> np.ndarray:
        """Estrae features per value detection"""
        features = []

        # 1. Probability features
        prob_calibrated = calibrated_result.get("prob_calibrated", 0.5)
        prob_raw = calibrated_result.get("prob_raw", 0.5)
        features.append(prob_calibrated)
        features.append(prob_raw)
        features.append(abs(prob_calibrated - prob_raw))

        # 2. Confidence features
        confidence_score = confidence_result.get("confidence_score", 50.0) / 100.0
        features.append(confidence_score)

        # 3. Odds features
        odds_current = odds_data.get("odds_current", 2.0)
        implied_prob = 1.0 / odds_current if odds_current > 1.0 else 0.5
        features.append(odds_current)
        features.append(implied_prob)

        # Expected Value (basic)
        ev_basic = (prob_calibrated * odds_current) - 1.0
        features.append(ev_basic)

        # Probability vs Odds gap
        prob_odds_gap = prob_calibrated - implied_prob
        features.append(prob_odds_gap)

        # 4. Odds movement features
        odds_history = odds_data.get("odds_history") or []
        if len(odds_history) >= 3:
            odds_values = [o["odds"] for o in odds_history]
            odds_first = odds_values[0]
            odds_last = odds_values[-1]

            # Movement direction and magnitude
            odds_movement = (odds_last - odds_first) / odds_first
            features.append(odds_movement)

            # Volatility
            odds_std = np.std(odds_values)
            features.append(odds_std)

            # Trend (positive/negative)
            odds_trend = np.polyfit(range(len(odds_values)), odds_values, 1)[0]
            features.append(odds_trend)
        else:
            features.extend([0.0, 0.0, 0.0])  # No history

        # 5. Volume/Sharp money features (if available)
        volume_history = odds_data.get("volume_history", [])
        if len(volume_history) >= 2:
            volume_values = [v["volume"] for v in volume_history]
            volume_avg = np.mean(volume_values)
            volume_last = volume_values[-1]

            # Volume spike
            volume_spike = volume_last / volume_avg if volume_avg > 0 else 1.0
            features.append(volume_spike)

            # Sharp money indicator
            sharp_money = (
                odds_movement < -0.05 and volume_spike > 2.0
            )
            features.append(float(sharp_money))
        else:
            features.extend([1.0, 0.0])

        # 6. Historical performance features
        similar_bets_roi = historical_context.get("similar_bets_roi", 0.0)
        similar_bets_count = historical_context.get("similar_bets_count", 0)
        similar_bets_winrate = historical_context.get("similar_bets_winrate", 0.5)

        features.append(similar_bets_roi)
        features.append(min(similar_bets_count / 100.0, 1.0))  # Normalize
        features.append(similar_bets_winrate)

        # 7. Market features
        league_quality = historical_context.get("league_quality", 0.7)
        market_efficiency = historical_context.get("market_efficiency", 0.7)
        features.append(league_quality)
        features.append(market_efficiency)

        # 8. Timing features
        time_to_kickoff = odds_data.get("time_to_kickoff_hours", 24.0)
        time_normalized = min(time_to_kickoff / 72.0, 1.0)  # 0-72h
        features.append(time_normalized)

        # Total features: 22
        return np.array(features, dtype=np.float32)

    def detect(
        self,
        calibrated_result: Dict,
        confidence_result: Dict,
        odds_data: Dict,
        historical_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Rileva value bet e classifica TRUE_VALUE vs TRAP.

        Returns:
            Dizionario con:
            - value_score: Score 0-100
            - value_type: TRUE_VALUE / TRAP / UNCERTAIN
            - expected_value: EV percentage
            - sharp_money_detected: bool
            - recommendation: STRONG_BET / BET / WATCH / SKIP
            - reasoning: Spiegazione
        """
        if historical_context is None:
            historical_context = {}

        try:
            # Basic EV calculation
            prob_calibrated = calibrated_result.get("prob_calibrated", 0.5)
            odds_current = odds_data.get("odds_current", 2.0)
            ev = (prob_calibrated * odds_current) - 1.0

            # Se EV negativo, è chiaramente un bad bet
            if ev < -0.05:
                return {
                    "value_score": 0.0,
                    "value_type": "TRAP",
                    "expected_value": ev,
                    "sharp_money_detected": False,
                    "recommendation": "SKIP",
                    "reasoning": f"Negative EV ({ev:.1%}). Clear bad bet."
                }

            # Check sharp money
            sharp_money_detected = self._detect_sharp_money(odds_data)

            # Se non trainato, usa rule-based
            if not self.is_trained:
                return self._rule_based_detection(
                    calibrated_result,
                    confidence_result,
                    odds_data,
                    historical_context,
                    ev,
                    sharp_money_detected
                )

            # ML-based detection
            features = self.extract_features(
                calibrated_result,
                confidence_result,
                odds_data,
                historical_context
            )

            # Normalize and predict
            features_normalized = self.scaler.transform(features.reshape(1, -1))
            value_proba = self.model.predict_proba(features_normalized)[0]

            # Classes: [TRAP, UNCERTAIN, TRUE_VALUE]
            trap_prob = value_proba[0]
            uncertain_prob = value_proba[1]
            true_value_prob = value_proba[2]

            value_score = true_value_prob * 100.0

            # Determine value type
            if true_value_prob > 0.6:
                value_type = "TRUE_VALUE"
            elif trap_prob > 0.6:
                value_type = "TRAP"
            else:
                value_type = "UNCERTAIN"

        except Exception as e:
            logger.error(f"❌ Error in ML detection, using rule-based: {e}")
            return self._rule_based_detection(
                calibrated_result,
                confidence_result,
                odds_data,
                historical_context,
                ev,
                False
            )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            value_score, value_type, ev, sharp_money_detected
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            value_score, value_type, ev, sharp_money_detected,
            calibrated_result, confidence_result
        )

        return {
            "value_score": float(value_score),
            "value_type": value_type,
            "expected_value": float(ev),
            "sharp_money_detected": sharp_money_detected,
            "recommendation": recommendation,
            "reasoning": reasoning,
            "ml_probabilities": {
                "trap": float(trap_prob),
                "uncertain": float(uncertain_prob),
                "true_value": float(true_value_prob)
            } if self.is_trained else None
        }

    def _detect_sharp_money(self, odds_data: Dict) -> bool:
        """Rileva sharp money movement"""
        odds_history = odds_data.get("odds_history") or []
        volume_history = odds_data.get("volume_history") or []

        if len(odds_history) < 3:
            return False

        # Check: odds dropping + volume increasing = sharp money
        odds_values = [o["odds"] for o in odds_history]
        odds_first = odds_values[0]
        odds_last = odds_values[-1]
        odds_drop = (odds_last - odds_first) / odds_first

        if odds_drop < self.config.sharp_money_threshold:
            # Odds dropped significantly
            if len(volume_history) >= 2:
                volume_values = [v["volume"] for v in volume_history]
                volume_increase = volume_values[-1] / np.mean(volume_values)
                if volume_increase > self.config.volume_sharp_threshold:
                    return True

        return False

    def _rule_based_detection(
        self,
        calibrated_result: Dict,
        confidence_result: Dict,
        odds_data: Dict,
        historical_context: Dict,
        ev: float,
        sharp_money_detected: bool
    ) -> Dict[str, Any]:
        """Rule-based value detection (fallback)"""

        # Score components
        score = 50.0  # Base

        # EV component (40 points)
        if ev > self.config.excellent_ev_threshold:
            score += 40
        elif ev > self.config.good_ev_threshold:
            score += 25
        elif ev > self.config.min_ev_to_bet:
            score += 10

        # Confidence component (30 points)
        confidence_score = confidence_result.get("confidence_score", 50.0)
        score += (confidence_score / 100.0) * 30

        # Sharp money bonus (20 points)
        if sharp_money_detected:
            score += 20

        # Historical ROI component (10 points)
        similar_roi = historical_context.get("similar_bets_roi", 0.0)
        if similar_roi > 0.10:
            score += 10
        elif similar_roi > 0:
            score += 5

        # Clamp
        score = np.clip(score, 0, 100)

        # Classify
        if score >= self.config.value_true_value_threshold:
            value_type = "TRUE_VALUE"
        elif score >= self.config.value_uncertain_threshold:
            value_type = "UNCERTAIN"
        else:
            value_type = "TRAP"

        # Recommendation
        recommendation = self._generate_recommendation(
            score, value_type, ev, sharp_money_detected
        )

        # Reasoning
        reasoning = self._generate_reasoning(
            score, value_type, ev, sharp_money_detected,
            calibrated_result, confidence_result
        )

        return {
            "value_score": float(score),
            "value_type": value_type,
            "expected_value": float(ev),
            "sharp_money_detected": sharp_money_detected,
            "recommendation": recommendation,
            "reasoning": reasoning,
            "method": "rule_based"
        }

    def _generate_recommendation(
        self,
        value_score: float,
        value_type: str,
        ev: float,
        sharp_money: bool
    ) -> str:
        """Generate betting recommendation"""
        if value_type == "TRUE_VALUE" and ev > 0.10:
            return "STRONG_BET"
        elif value_type == "TRUE_VALUE" and ev > 0.03:
            return "BET"
        elif value_type == "UNCERTAIN" and sharp_money:
            return "WATCH"
        else:
            return "SKIP"

    def _generate_reasoning(
        self,
        value_score: float,
        value_type: str,
        ev: float,
        sharp_money: bool,
        calibrated_result: Dict,
        confidence_result: Dict
    ) -> str:
        """Generate textual reasoning"""
        parts = []

        # Value type
        if value_type == "TRUE_VALUE":
            parts.append(f"True value detected (score: {value_score:.0f}/100)")
        elif value_type == "TRAP":
            parts.append(f"Potential trap bet (score: {value_score:.0f}/100)")
        else:
            parts.append(f"Uncertain value (score: {value_score:.0f}/100)")

        # EV
        parts.append(f"EV: {ev:+.1%}")

        # Sharp money
        if sharp_money:
            parts.append("Sharp money detected (strong indicator)")

        # Confidence
        confidence = confidence_result.get("confidence_level", "MEDIUM")
        parts.append(f"Confidence: {confidence}")

        return ". ".join(parts) + "."

    def train(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Train XGBoost classifier"""
        logger.warning("⚠️ Value Detector training not yet implemented")
        return {"status": "not_implemented"}

    def save(self, filepath: Optional[str] = None):
        """Save model"""
        if filepath is None:
            filepath = self.config.models_dir / "value_detector.pkl"
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained
        }, filepath)
        logger.info(f"💾 Value Detector saved to {filepath}")

    def load(self, filepath: Optional[str] = None):
        """Load model"""
        if filepath is None:
            filepath = self.config.models_dir / "value_detector.pkl"
        if not Path(filepath).exists():
            logger.warning(f"⚠️ Model not found: {filepath}")
            return
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_trained = data["is_trained"]
        logger.info(f"✅ Value Detector loaded from {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = ValueDetector()

    # Test
    calibrated = {"prob_calibrated": 0.58, "prob_raw": 0.65, "calibration_shift": -0.07}
    confidence = {"confidence_score": 85.0, "confidence_level": "HIGH"}
    odds = {
        "odds_current": 1.85,
        "odds_history": [
            {"odds": 1.90, "time": "10:00"},
            {"odds": 1.88, "time": "11:00"},
            {"odds": 1.85, "time": "12:00"}
        ],
        "volume_history": [
            {"volume": 1000, "time": "10:00"},
            {"volume": 2500, "time": "11:00"},
            {"volume": 5000, "time": "12:00"}
        ]
    }
    historical = {"similar_bets_roi": 0.15, "similar_bets_count": 50, "similar_bets_winrate": 0.62}

    result = detector.detect(calibrated, confidence, odds, historical)
    print(f"\n{result}")
