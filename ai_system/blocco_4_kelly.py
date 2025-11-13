"""
BLOCCO 4: Smart Kelly Optimizer
================================

Ottimizzazione stake usando Kelly Criterion con adjustments dinamici.

Features:
- Fractional Kelly con fraction dinamica
- Adjustment per confidence level
- Adjustment per API data quality
- Correlation penalty (evita overexposure)
- Portfolio-aware stake sizing

Input: value detection + confidence + portfolio state
Output: optimal stake + kelly fraction + reasoning
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from .config import AIConfig

logger = logging.getLogger(__name__)


class SmartKellyOptimizer:
    """Kelly Criterion optimizer con adjustments intelligenti"""

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        logger.info("✅ Smart Kelly Optimizer initialized")

    def optimize(
        self,
        value_result: Dict,
        confidence_result: Dict,
        odds: float,
        bankroll: float,
        portfolio_state: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calcola stake ottimale usando Kelly Criterion.

        Returns:
            - optimal_stake: Stake in currency
            - stake_percentage: Stake as % of bankroll
            - kelly_fraction: Kelly fraction used
            - adjustments: Dict con tutti gli adjustment applicati
        """
        if portfolio_state is None:
            portfolio_state = {}

        # Get probability and EV
        prob = value_result.get("prob_calibrated", 0.5)
        ev = value_result.get("expected_value", 0.0)

        # Basic Kelly formula: f* = (bp - q) / b
        # where b = odds - 1, p = prob, q = 1 - p
        b = odds - 1
        q = 1 - prob

        if b <= 0 or prob <= 0:
            logger.warning("⚠️ Invalid odds or probability for Kelly")
            return self._zero_stake_result("invalid_inputs")

        # Pure Kelly fraction
        kelly_pure = (b * prob - q) / b

        if kelly_pure <= 0:
            return self._zero_stake_result("negative_kelly")

        # Start with default fractional Kelly
        kelly_fraction = self.config.kelly_default_fraction
        adjustments = {"base_fraction": kelly_fraction}

        # Adjustment 1: Confidence level
        confidence_level = confidence_result.get("confidence_level", "MEDIUM")
        confidence_mult = self.config.kelly_confidence_multiplier.get(
            confidence_level.lower().replace(" ", "_"),
            1.0
        )
        kelly_fraction *= confidence_mult
        adjustments["confidence_multiplier"] = confidence_mult

        # Adjustment 2: API data quality
        data_quality = value_result.get("data_quality", 0.5)
        if data_quality >= 0.9:
            quality_mult = self.config.kelly_api_quality_multiplier["excellent"]
        elif data_quality >= 0.7:
            quality_mult = self.config.kelly_api_quality_multiplier["good"]
        elif data_quality >= 0.5:
            quality_mult = self.config.kelly_api_quality_multiplier["medium"]
        else:
            quality_mult = self.config.kelly_api_quality_multiplier["poor"]

        kelly_fraction *= quality_mult
        adjustments["quality_multiplier"] = quality_mult

        # Adjustment 3: Value type
        value_type = value_result.get("value_type", "UNCERTAIN")
        if value_type == "TRUE_VALUE":
            value_mult = 1.2
        elif value_type == "TRAP":
            value_mult = 0.3  # Heavy penalty
        else:
            value_mult = 0.8

        kelly_fraction *= value_mult
        adjustments["value_type_multiplier"] = value_mult

        # Adjustment 4: Correlation penalty
        correlation_penalty = self._calculate_correlation_penalty(portfolio_state)
        kelly_fraction *= (1.0 - correlation_penalty)
        adjustments["correlation_penalty"] = correlation_penalty

        # Apply fraction to pure Kelly
        kelly_final = kelly_pure * kelly_fraction

        # Calculate stake
        stake_pct = np.clip(
            kelly_final * 100,
            self.config.min_stake_pct,
            self.config.max_stake_pct
        )

        stake_amount = (stake_pct / 100) * bankroll

        # Apply absolute limits
        stake_amount = np.clip(
            stake_amount,
            self.config.absolute_min_stake,
            self.config.absolute_max_stake
        )

        # Recalculate percentage after absolute limits
        stake_pct_final = (stake_amount / bankroll) * 100

        return {
            "optimal_stake": float(stake_amount),
            "stake_percentage": float(stake_pct_final),
            "kelly_fraction": float(kelly_fraction),
            "kelly_pure": float(kelly_pure),
            "kelly_final": float(kelly_final),
            "adjustments": adjustments,
            "reasoning": self._generate_reasoning(
                stake_amount, stake_pct_final, kelly_fraction, adjustments
            )
        }

    def _calculate_correlation_penalty(self, portfolio_state: Dict) -> float:
        """Calculate penalty for correlated bets in portfolio"""
        if not self.config.correlation_penalty_enabled:
            return 0.0

        # TODO: Implement sophisticated correlation analysis
        # For now, simple exposure check

        active_bets = portfolio_state.get("active_bets", [])
        if not active_bets:
            return 0.0

        # Calculate total exposure
        total_exposure = sum(bet.get("stake", 0) for bet in active_bets)
        bankroll = portfolio_state.get("bankroll", 1000)
        exposure_pct = total_exposure / bankroll

        # Apply penalty if exposure high
        if exposure_pct > self.config.max_correlation_exposure:
            penalty = min((exposure_pct - self.config.max_correlation_exposure) / 0.10, 0.5)
            return penalty

        return 0.0

    def _zero_stake_result(self, reason: str) -> Dict[str, Any]:
        """Return zero stake with reason"""
        return {
            "optimal_stake": 0.0,
            "stake_percentage": 0.0,
            "kelly_fraction": 0.0,
            "kelly_pure": 0.0,
            "kelly_final": 0.0,
            "adjustments": {},
            "reasoning": f"Zero stake: {reason}"
        }

    def _generate_reasoning(
        self,
        stake: float,
        stake_pct: float,
        kelly_fraction: float,
        adjustments: Dict
    ) -> str:
        """Generate textual explanation"""
        parts = [f"Optimal stake: €{stake:.2f} ({stake_pct:.1f}% bankroll)"]
        parts.append(f"Kelly fraction: {kelly_fraction:.2f}")

        # Key adjustments
        if adjustments.get("confidence_multiplier", 1.0) != 1.0:
            parts.append(
                f"Confidence adj: {adjustments['confidence_multiplier']:.2f}×"
            )

        if adjustments.get("correlation_penalty", 0.0) > 0:
            parts.append(
                f"Correlation penalty: {adjustments['correlation_penalty']:.1%}"
            )

        return ". ".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    optimizer = SmartKellyOptimizer()

    value_result = {
        "prob_calibrated": 0.58,
        "expected_value": 0.073,
        "value_type": "TRUE_VALUE",
        "data_quality": 0.92
    }
    confidence_result = {
        "confidence_score": 85,
        "confidence_level": "HIGH"
    }
    result = optimizer.optimize(value_result, confidence_result, odds=1.85, bankroll=1000)
    print(f"\n{result}")
