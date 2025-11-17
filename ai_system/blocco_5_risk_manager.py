"""
BLOCCO 5: Risk Manager & Filter
================================

Gestione rischio e filtri finali prima di approvare una scommessa.

Features:
- Portfolio limits enforcement
- Red/green flags identification
- Stop-loss protection
- Exposure management (league, team, correlations)
- Final GO/NO-GO decision

Input: Tutti i risultati blocchi precedenti + portfolio
Output: Decision (BET/SKIP/WATCH) + final stake + reasoning
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from .config import AIConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """Gestore del rischio e filtri finali"""

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self.daily_stats = {"bets": 0, "losses": 0.0, "date": None}
        logger.info("✅ Risk Manager initialized")

    def decide(
        self,
        value_result: Dict,
        confidence_result: Dict,
        kelly_result: Dict,
        match_info: Dict,
        portfolio_state: Optional[Dict] = None,
        regime_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Decisione finale: scommettere o no?

        Returns:
            - decision: BET / SKIP / WATCH
            - final_stake: Stake approvato (può essere ridotto)
            - priority: LOW / MEDIUM / HIGH
            - red_flags: Lista di problemi
            - green_flags: Lista di punti di forza
            - reasoning: Spiegazione dettagliata
        """
        if portfolio_state is None:
            portfolio_state = {}

        # Reset daily stats if new day
        self._check_daily_reset()

        # Collect all flags
        red_flags = []
        green_flags = []

        # 1. Check minimum thresholds
        value_score = value_result.get("value_score", 0)
        confidence_score = confidence_result.get("confidence_score", 0)
        ev = value_result.get("expected_value", 0)

        if value_score < self.config.min_value_score_to_bet:
            red_flags.append(f"Value score too low ({value_score:.0f} < {self.config.min_value_score_to_bet})")

        if confidence_score < self.config.min_confidence_to_bet:
            red_flags.append(f"Confidence too low ({confidence_score:.0f} < {self.config.min_confidence_to_bet})")

        if ev < self.config.min_ev_to_bet:
            red_flags.append(f"EV too low ({ev:.1%} < {self.config.min_ev_to_bet:.1%})")

        # 2. Check portfolio limits
        active_bets = portfolio_state.get("active_bets", [])

        if len(active_bets) >= self.config.max_active_bets:
            red_flags.append(f"Max active bets reached ({len(active_bets)}/{self.config.max_active_bets})")

        if self.daily_stats["bets"] >= self.config.max_daily_bets:
            red_flags.append(f"Max daily bets reached ({self.daily_stats['bets']}/{self.config.max_daily_bets})")

        # 3. Check exposure limits
        league = match_info.get("league", "")
        home_team = match_info.get("home", "")

        league_exposure = max(0.0, self._calculate_league_exposure(active_bets, league, portfolio_state))
        if league_exposure > self.config.max_same_league_exposure:
            red_flags.append(f"League exposure too high ({league_exposure:.1%})")

        team_exposure = max(0.0, self._calculate_team_exposure(active_bets, home_team, portfolio_state))
        if team_exposure > self.config.max_same_team_exposure:
            red_flags.append(f"Team exposure too high ({team_exposure:.1%})")

        # 4. Check stop-loss
        if self.config.stop_loss_trigger:
            daily_loss_pct = abs(self.daily_stats["losses"]) / portfolio_state.get("bankroll", 1000)
            if daily_loss_pct > self.config.max_daily_loss_pct:
                red_flags.append(f"Daily stop-loss triggered ({daily_loss_pct:.1%})")

        # 5. Context-specific red flags
        red_flags.extend(confidence_result.get("risk_factors", [])[:2])

        # 6. Green flags
        if value_result.get("value_type") == "TRUE_VALUE":
            green_flags.append("True value bet identified")

        if value_result.get("sharp_money_detected"):
            green_flags.append("Sharp money detected")

        if confidence_result.get("confidence_level") in ["HIGH", "VERY_HIGH"]:
            green_flags.append(f"High confidence ({confidence_result['confidence_level']})")

        green_flags.extend(confidence_result.get("strength_factors", [])[:2])

        if regime_result:
            regime_label = regime_result.get("label")
            if regime_label == "sharp_rush":
                green_flags.append("Market regime: sharp rush (pro value)")
            elif regime_label in {"public_hype", "chaotic"}:
                red_flags.append(f"Market regime risk: {regime_label}")

        # Decision logic
        decision = "SKIP"
        final_stake = 0.0
        priority = "LOW"

        if len(red_flags) > self.config.max_red_flags_allowed:
            decision = "SKIP"
            reasoning = f"Too many red flags ({len(red_flags)}). {', '.join(red_flags[:3])}"

        elif len(red_flags) > 0 and len(green_flags) == 0:
            decision = "SKIP"
            reasoning = f"Red flags with no compensating strengths. {', '.join(red_flags[:2])}"

        elif value_result.get("recommendation") in ["STRONG_BET", "BET"]:
            decision = "BET"
            final_stake = kelly_result.get("optimal_stake", 0.0)

            # Determine priority
            if value_score > 80 and confidence_score > 80:
                priority = "HIGH"
            elif value_score > 60 and confidence_score > 65:
                priority = "MEDIUM"
            else:
                priority = "LOW"

            reasoning = f"Approved bet. Value: {value_score:.0f}, Confidence: {confidence_score:.0f}, EV: {ev:.1%}"

            # Apply risk reduction if needed
            if len(red_flags) > 0:
                reduction_factor = 1.0 - (len(red_flags) * 0.15)  # -15% per red flag
                final_stake *= max(reduction_factor, 0.5)  # Min 50% of original
                reasoning += f" (Stake reduced due to {len(red_flags)} red flag(s))"

        else:
            decision = "WATCH"
            reasoning = f"Uncertain value. Monitor odds movement. {value_result.get('reasoning', '')}"

        return {
            "decision": decision,
            "final_stake": float(final_stake),
            "priority": priority,
            "red_flags": red_flags,
            "green_flags": green_flags,
            "reasoning": reasoning,
            "risk_score": self._calculate_risk_score(red_flags, green_flags),
            "market_regime": regime_result,
            "checks_passed": {
                "min_thresholds": len([f for f in red_flags if "too low" in f.lower()]) == 0,
                "portfolio_limits": len([f for f in red_flags if "max" in f.lower() or "exposure" in f.lower()]) == 0,
                "stop_loss": "stop-loss" not in " ".join(red_flags).lower()
            }
        }

    def _check_daily_reset(self):
        """Reset daily stats if new day"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.daily_stats["date"] != today:
            self.daily_stats = {"bets": 0, "losses": 0.0, "date": today}

    def _calculate_league_exposure(self, active_bets: List, league: str, portfolio: Dict) -> float:
        """Calculate exposure to same league"""
        if not active_bets:
            return 0.0
        bankroll = portfolio.get("bankroll", 1000)
        league_stake = sum(
            bet.get("stake", 0) for bet in active_bets
            if bet.get("league", "").lower() == league.lower()
        )
        return league_stake / bankroll

    def _calculate_team_exposure(self, active_bets: List, team: str, portfolio: Dict) -> float:
        """Calculate exposure to same team"""
        if not active_bets:
            return 0.0
        bankroll = portfolio.get("bankroll", 1000)
        team_stake = sum(
            bet.get("stake", 0) for bet in active_bets
            if team.lower() in [bet.get("home", "").lower(), bet.get("away", "").lower()]
        )
        return team_stake / bankroll

    def _calculate_risk_score(self, red_flags: List, green_flags: List) -> float:
        """Calculate overall risk score 0-100 (100 = high risk)"""
        risk = 50.0  # Base
        risk += len(red_flags) * 10
        risk -= len(green_flags) * 8
        return float(max(0, min(100, risk)))

    def record_bet_result(self, stake: float, outcome: str, payout: float):
        """Record bet result for daily tracking"""
        self.daily_stats["bets"] += 1
        if outcome == "loss":
            self.daily_stats["losses"] += stake
        logger.info(f"📊 Daily stats updated: {self.daily_stats}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = RiskManager()

    value = {"value_score": 78, "value_type": "TRUE_VALUE", "expected_value": 0.12, "sharp_money_detected": True, "recommendation": "BET"}
    confidence = {"confidence_score": 85, "confidence_level": "HIGH", "risk_factors": [], "strength_factors": ["Strong form", "Stable odds"]}
    kelly = {"optimal_stake": 32}
    match = {"home": "Inter", "away": "Genoa", "league": "Serie A"}

    result = manager.decide(value, confidence, kelly, match)
    print(f"\n{result}")
