"""
BLOCCO 2: Confidence Scorer
============================

Valuta l'affidabilità di una predizione usando Random Forest.

Funzionalità:
- Multi-factor confidence scoring
- Considera: model agreement, data quality, odds stability, historical performance
- Output: confidence score 0-100 + livello (LOW/MEDIUM/HIGH/VERY_HIGH)
- Identifica red flags e green flags

Input: probabilità calibrata + context completo
Output: confidence score + flags + reasoning
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib
from datetime import datetime

# Machine Learning
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("⚠️ scikit-learn not available")

from .config import AIConfig

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Valutatore di confidenza per predizioni.

    Usa Random Forest per combinare multiple feature e produrre
    uno score di confidence affidabile.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        """Inizializza Confidence Scorer"""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for Confidence Scorer")

        self.config = config or AIConfig()
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

        # Statistics
        self.scoring_history = []

        logger.info("✅ Confidence Scorer initialized")

    def extract_features(
        self,
        calibrated_result: Dict,
        api_context: Dict,
        additional_context: Dict
    ) -> np.ndarray:
        """
        Estrae features per confidence scoring.

        Args:
            calibrated_result: Output da Probability Calibrator
            api_context: Context da API Data Engine
            additional_context: Info aggiuntive (odds, market, etc)

        Returns:
            Array di features
        """
        features = []

        # 1. Model agreement features
        prob_calibrated = calibrated_result.get("prob_calibrated", 0.5)
        prob_raw = calibrated_result.get("prob_raw", 0.5)
        calibration_shift = abs(calibrated_result.get("calibration_shift", 0.0))

        # Agreement = quanto i modelli sono allineati (low shift = high agreement)
        model_agreement = 1.0 - min(calibration_shift / 0.15, 1.0)
        features.append(model_agreement)

        # Uncertainty from calibrator
        uncertainty = calibrated_result.get("uncertainty", 0.10)
        features.append(1.0 - min(uncertainty / 0.20, 1.0))

        # 2. Data completeness features
        data_quality = api_context.get("metadata", {}).get("data_quality", 0.5)
        features.append(data_quality)

        # API source quality
        sources = api_context.get("metadata", {}).get("sources", [])
        source_quality = 0.5
        if "api-football" in sources:
            source_quality = 0.9
        elif "thesportsdb" in sources:
            source_quality = 0.7
        elif "cache" in sources:
            source_quality = 0.6
        features.append(source_quality)

        # 3. Odds stability features
        odds_current = additional_context.get("odds_current", 2.0)
        odds_history = additional_context.get("odds_history") or []

        if len(odds_history) >= 3:
            # Calculate volatility
            odds_values = [o["odds"] for o in odds_history]
            odds_std = np.std(odds_values)
            odds_stability = 1.0 - min(odds_std / 0.50, 1.0)
        else:
            odds_stability = 0.5  # Unknown

        features.append(odds_stability)

        # Implied probability from odds
        implied_prob = 1.0 / odds_current if odds_current > 1.0 else 0.5
        prob_odds_diff = abs(prob_calibrated - implied_prob)
        features.append(1.0 - min(prob_odds_diff / 0.30, 1.0))

        # 4. Historical accuracy (se disponibile)
        historical_accuracy = additional_context.get("historical_accuracy", 0.7)
        features.append(historical_accuracy)

        # Similar bets performance
        similar_bets_roi = additional_context.get("similar_bets_roi", 0.0)
        # Normalize ROI to 0-1 (assume -0.5 to +0.5 range)
        similar_bets_score = max(0, min(1, (similar_bets_roi + 0.5)))
        features.append(similar_bets_score)

        # 5. API freshness
        cache_used = api_context.get("metadata", {}).get("cache_used", True)
        api_freshness = 0.5 if cache_used else 1.0  # Fresh API = better
        features.append(api_freshness)

        # 6. Probability extremity (very low or very high = less confident)
        prob_extremity = abs(prob_calibrated - 0.5) * 2  # 0 to 1
        prob_confidence = 1.0 - (prob_extremity ** 2)  # Penalize extreme probs
        features.append(prob_confidence)

        # 7. League and market quality
        league_quality = additional_context.get("league_quality", 0.7)
        market_liquidity = additional_context.get("market_liquidity", 0.7)
        features.append(league_quality)
        features.append(market_liquidity)

        # 8. Context-specific features
        match_importance = api_context.get("metadata", {}).get("importance", 0.5)
        features.append(match_importance)

        # Red flags count (normalized)
        red_flags = additional_context.get("red_flags", [])
        red_flags_score = 1.0 - min(len(red_flags) / 5.0, 1.0)
        features.append(red_flags_score)

        # Green flags count (normalized)
        green_flags = additional_context.get("green_flags", [])
        green_flags_score = min(len(green_flags) / 5.0, 1.0)
        features.append(green_flags_score)

        # Total features: 16
        return np.array(features, dtype=np.float32)

    def score(
        self,
        calibrated_result: Dict,
        api_context: Dict,
        additional_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calcola confidence score per una predizione.

        Args:
            calibrated_result: Output da Probability Calibrator
            api_context: Context da API Data Engine
            additional_context: Context aggiuntivo (odds, historical, etc)

        Returns:
            Dizionario con:
            - confidence_score: Score 0-100
            - confidence_level: LOW/MEDIUM/HIGH/VERY_HIGH
            - risk_factors: Lista di fattori negativi
            - strength_factors: Lista di fattori positivi
            - recommendation: Textual recommendation
        """
        if additional_context is None:
            additional_context = {}

        try:
            # Se non trainato, usa rule-based scoring
            if not self.is_trained:
                return self._rule_based_scoring(
                    calibrated_result, api_context, additional_context
                )

            # ML-based scoring
            features = self.extract_features(
                calibrated_result, api_context, additional_context
            )

            # Normalize
            features_normalized = self.scaler.transform(features.reshape(1, -1))

            # Predict
            confidence_score = self.model.predict(features_normalized)[0]
            confidence_score = np.clip(confidence_score, 0, 100)

        except Exception as e:
            logger.error(f"❌ Error in ML scoring, falling back to rules: {e}")
            return self._rule_based_scoring(
                calibrated_result, api_context, additional_context
            )

        # Determine confidence level
        if confidence_score >= self.config.confidence_very_high:
            level = "VERY_HIGH"
        elif confidence_score >= self.config.confidence_high:
            level = "HIGH"
        elif confidence_score >= self.config.confidence_medium:
            level = "MEDIUM"
        elif confidence_score >= self.config.confidence_low:
            level = "LOW"
        else:
            level = "VERY_LOW"

        # Identify risk and strength factors
        risk_factors, strength_factors = self._identify_factors(
            calibrated_result, api_context, additional_context
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            confidence_score, level, risk_factors, strength_factors
        )

        result = {
            "confidence_score": float(confidence_score),
            "confidence_level": level,
            "risk_factors": risk_factors,
            "strength_factors": strength_factors,
            "recommendation": recommendation,
            "method": "ml" if self.is_trained else "rule_based"
        }

        # Track
        self.scoring_history.append({
            "timestamp": datetime.now().isoformat(),
            "score": confidence_score,
            "level": level
        })

        return result

    def _rule_based_scoring(
        self,
        calibrated_result: Dict,
        api_context: Dict,
        additional_context: Dict
    ) -> Dict[str, Any]:
        """
        Confidence scoring basato su regole (fallback quando model non trainato).
        """
        weights = self.config.confidence_weights
        score = 0.0

        # 1. Model agreement (30%)
        calibration_shift = abs(calibrated_result.get("calibration_shift", 0.0))
        model_agreement = (1.0 - min(calibration_shift / 0.15, 1.0)) * 100
        score += model_agreement * weights["model_agreement"]

        # 2. Data completeness (25%)
        data_quality = api_context.get("metadata", {}).get("data_quality", 0.5) * 100
        score += data_quality * weights["data_completeness"]

        # 3. Odds stability (20%)
        odds_history = additional_context.get("odds_history") or []
        if len(odds_history) >= 3:
            odds_values = [o["odds"] for o in odds_history]
            odds_std = np.std(odds_values)
            odds_stability = (1.0 - min(odds_std / 0.50, 1.0)) * 100
        else:
            odds_stability = 50.0
        score += odds_stability * weights["odds_stability"]

        # 4. Historical accuracy (15%)
        historical_accuracy = additional_context.get("historical_accuracy", 0.7) * 100
        score += historical_accuracy * weights["historical_accuracy"]

        # 5. API freshness (10%)
        cache_used = api_context.get("metadata", {}).get("cache_used", True)
        api_freshness = 50 if cache_used else 100
        score += api_freshness * weights["api_freshness"]

        # Clamp to 0-100
        score = np.clip(score, 0, 100)

        # Determine level
        if score >= self.config.confidence_very_high:
            level = "VERY_HIGH"
        elif score >= self.config.confidence_high:
            level = "HIGH"
        elif score >= self.config.confidence_medium:
            level = "MEDIUM"
        elif score >= self.config.confidence_low:
            level = "LOW"
        else:
            level = "VERY_LOW"

        # Identify factors
        risk_factors, strength_factors = self._identify_factors(
            calibrated_result, api_context, additional_context
        )

        # Recommendation
        recommendation = self._generate_recommendation(
            score, level, risk_factors, strength_factors
        )

        return {
            "confidence_score": float(score),
            "confidence_level": level,
            "risk_factors": risk_factors,
            "strength_factors": strength_factors,
            "recommendation": recommendation,
            "method": "rule_based"
        }

    def _identify_factors(
        self,
        calibrated_result: Dict,
        api_context: Dict,
        additional_context: Dict
    ) -> Tuple[List[str], List[str]]:
        """Identifica risk factors e strength factors"""
        risk_factors = []
        strength_factors = []

        # Check calibration shift
        calibration_shift = abs(calibrated_result.get("calibration_shift", 0.0))
        if calibration_shift > 0.10:
            risk_factors.append(f"Large calibration shift ({calibration_shift:.1%})")
        elif calibration_shift < 0.03:
            strength_factors.append("Models in strong agreement")

        # Check data quality
        data_quality = api_context.get("metadata", {}).get("data_quality", 0.5)
        if data_quality < 0.5:
            risk_factors.append(f"Low data quality ({data_quality:.0%})")
        elif data_quality > 0.85:
            strength_factors.append(f"High data quality ({data_quality:.0%})")

        # Check odds stability
        odds_history = additional_context.get("odds_history") or []
        if len(odds_history) >= 3:
            odds_values = [o["odds"] for o in odds_history]
            odds_std = np.std(odds_values)
            if odds_std > 0.30:
                risk_factors.append(f"Volatile odds (σ={odds_std:.2f})")
            elif odds_std < 0.05:
                strength_factors.append("Stable odds")

        # Check uncertainty
        uncertainty = calibrated_result.get("uncertainty", 0.10)
        if uncertainty > 0.15:
            risk_factors.append(f"High uncertainty (±{uncertainty:.1%})")
        elif uncertainty < 0.05:
            strength_factors.append(f"Low uncertainty (±{uncertainty:.1%})")

        # Check red/green flags
        red_flags = additional_context.get("red_flags", [])
        if red_flags:
            risk_factors.extend(red_flags[:3])  # Max 3

        green_flags = additional_context.get("green_flags", [])
        if green_flags:
            strength_factors.extend(green_flags[:3])  # Max 3

        return risk_factors, strength_factors

    def _generate_recommendation(
        self,
        score: float,
        level: str,
        risk_factors: List[str],
        strength_factors: List[str]
    ) -> str:
        """Genera raccomandazione testuale"""
        if level == "VERY_HIGH":
            base = "Extremely reliable prediction. Strong bet candidate."
        elif level == "HIGH":
            base = "Reliable prediction. Good bet candidate."
        elif level == "MEDIUM":
            base = "Moderate confidence. Consider carefully."
        elif level == "LOW":
            base = "Low confidence. High risk bet."
        else:
            base = "Very low confidence. Skip this bet."

        # Add context
        if risk_factors:
            base += f" Warning: {', '.join(risk_factors[:2])}."
        if strength_factors:
            base += f" Strengths: {', '.join(strength_factors[:2])}."

        return base

    def train(
        self,
        historical_data: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Addestra il confidence scorer su dati storici.

        Args:
            historical_data: DataFrame con features + actual confidence
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        logger.info(f"📚 Training Confidence Scorer on {len(historical_data)} samples...")

        # Extract features and targets
        # NOTE: Questo richiede dati storici con "true confidence" (difficile da ottenere)
        # Per ora usiamo un proxy: outcome match + ROI come confidence indicator

        # TODO: Implementare feature extraction da historical_data
        # Per ora training è opzionale, si usa rule-based scoring

        logger.warning("⚠️ ML training for Confidence Scorer not yet implemented")
        logger.warning("   Using rule-based scoring instead")

        return {"status": "not_implemented", "using": "rule_based"}

    def save(self, filepath: Optional[str] = None):
        """Salva il modello"""
        if filepath is None:
            filepath = self.config.models_dir / "confidence_scorer.pkl"

        save_dict = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "config": self.config.confidence_weights
        }

        joblib.dump(save_dict, filepath)
        logger.info(f"💾 Confidence Scorer saved to {filepath}")

    def load(self, filepath: Optional[str] = None):
        """Carica il modello"""
        if filepath is None:
            filepath = self.config.models_dir / "confidence_scorer.pkl"

        if not Path(filepath).exists():
            logger.warning(f"⚠️ Model file not found: {filepath}")
            return

        save_dict = joblib.load(filepath)
        self.model = save_dict["model"]
        self.scaler = save_dict["scaler"]
        self.is_trained = save_dict["is_trained"]

        logger.info(f"✅ Confidence Scorer loaded from {filepath}")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def test_confidence_scorer():
    """Test del Confidence Scorer"""
    print("=" * 70)
    print("TEST: Confidence Scorer")
    print("=" * 70)

    scorer = ConfidenceScorer()

    # Test case 1: High confidence
    print("\nTest 1: High confidence scenario")
    calibrated = {
        "prob_calibrated": 0.62,
        "prob_raw": 0.65,
        "calibration_shift": -0.03,
        "uncertainty": 0.04
    }
    api_context = {
        "metadata": {
            "data_quality": 0.92,
            "sources": ["api-football", "thesportsdb"],
            "cache_used": False,
            "importance": 0.85
        }
    }
    additional = {
        "odds_current": 1.75,
        "odds_history": [
            {"odds": 1.78, "time": "10:00"},
            {"odds": 1.76, "time": "11:00"},
            {"odds": 1.75, "time": "12:00"}
        ],
        "historical_accuracy": 0.82,
        "similar_bets_roi": 0.15,
        "green_flags": ["Strong form", "Stable odds"],
        "red_flags": []
    }

    result = scorer.score(calibrated, api_context, additional)
    print(f"  Confidence: {result['confidence_score']:.1f}/100 ({result['confidence_level']})")
    print(f"  Strengths: {result['strength_factors']}")
    print(f"  Risks: {result['risk_factors']}")
    print(f"  Recommendation: {result['recommendation']}")

    # Test case 2: Low confidence
    print("\nTest 2: Low confidence scenario")
    calibrated_low = {
        "prob_calibrated": 0.48,
        "prob_raw": 0.65,
        "calibration_shift": -0.17,
        "uncertainty": 0.18
    }
    api_context_low = {
        "metadata": {
            "data_quality": 0.35,
            "sources": ["fallback"],
            "cache_used": True,
            "importance": 0.40
        }
    }
    additional_low = {
        "odds_current": 2.10,
        "odds_history": [
            {"odds": 1.90, "time": "10:00"},
            {"odds": 2.05, "time": "11:00"},
            {"odds": 2.10, "time": "12:00"}
        ],
        "historical_accuracy": 0.55,
        "similar_bets_roi": -0.05,
        "green_flags": [],
        "red_flags": ["Key player injured", "Volatile odds", "Poor form"]
    }

    result_low = scorer.score(calibrated_low, api_context_low, additional_low)
    print(f"  Confidence: {result_low['confidence_score']:.1f}/100 ({result_low['confidence_level']})")
    print(f"  Strengths: {result_low['strength_factors']}")
    print(f"  Risks: {result_low['risk_factors']}")
    print(f"  Recommendation: {result_low['recommendation']}")

    print("\n" + "=" * 70)
    print("✅ Confidence Scorer tests completed")
    print("=" * 70)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_confidence_scorer()
