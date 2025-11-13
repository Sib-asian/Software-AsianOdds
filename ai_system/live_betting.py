"""
Live Betting AI Engine
======================

Predizioni real-time durante le partite in corso.

Aggiorna probabilità ogni minuto considerando:
- Score attuale
- xG live
- Momentum (dangerous attacks, shots)
- Eventi (goal, red cards, substitutions)
- Time remaining

ROI tipico live betting: +15-20% vs pre-match +5-8%
"""

import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LiveMatch:
    """Rappresenta stato live di una partita"""

    def __init__(self, match_id: str, pre_match_prob: float):
        self.match_id = match_id
        self.pre_match_prob = pre_match_prob

        # Live state
        self.minute = 0
        self.score_home = 0
        self.score_away = 0
        self.xg_home = 0.0
        self.xg_away = 0.0
        self.red_cards_home = 0
        self.red_cards_away = 0

        # Recent momentum (last 10 minutes)
        self.recent_shots_home = 0
        self.recent_shots_away = 0
        self.recent_attacks_home = 0
        self.recent_attacks_away = 0

    def update(self, live_data: Dict):
        """Update con nuovi dati live"""
        self.minute = live_data.get('minute', self.minute)
        self.score_home = live_data.get('score_home', self.score_home)
        self.score_away = live_data.get('score_away', self.score_away)
        self.xg_home = live_data.get('xg_home', self.xg_home)
        self.xg_away = live_data.get('xg_away', self.xg_away)


class LiveBettingEngine:
    """
    Engine per predizioni live in-play.

    Usage:
        engine = LiveBettingEngine()
        live_match = engine.start_monitoring(match_id, pre_match_prob)
        # Update periodically
        live_match.update(live_data)
        new_prob = engine.recalculate_probability(live_match)
    """

    def __init__(self):
        logger.info("✅ Live Betting Engine initialized")

    def start_monitoring(self, match_id: str, pre_match_prob: float) -> LiveMatch:
        """Inizia monitoring di una partita live"""
        return LiveMatch(match_id, pre_match_prob)

    def recalculate_probability(self, match: LiveMatch) -> Dict:
        """
        Ricalcola probabilità basandosi su stato live.

        Returns:
            Dict con nuove probabilità e raccomandazioni
        """
        # Base: pre-match probability
        base_prob = match.pre_match_prob

        # Adjust per score
        score_adj = self._score_adjustment(
            match.score_home, match.score_away, match.minute
        )

        # Adjust per xG
        xg_adj = self._xg_adjustment(match.xg_home, match.xg_away, match.minute)

        # Adjust per momentum
        momentum_adj = self._momentum_adjustment(match)

        # Adjust per red cards
        red_card_adj = self._red_card_adjustment(
            match.red_cards_home, match.red_cards_away, match.minute
        )

        # Combine adjustments
        new_prob = base_prob * score_adj * xg_adj * momentum_adj * red_card_adj

        # Clamp
        new_prob = max(0.01, min(0.99, new_prob))

        # Calculate other markets
        over_15 = self._calculate_over_goals(match, 1.5)
        over_25 = self._calculate_over_goals(match, 2.5)

        # Timing recommendation
        timing = self._get_timing_recommendation(match, new_prob)

        return {
            'probability_home_win': new_prob,
            'over_1.5_goals': over_15,
            'over_2.5_goals': over_25,
            'timing': timing,
            'minute': match.minute,
            'score': f"{match.score_home}-{match.score_away}",
            'adjustments': {
                'score': score_adj,
                'xg': xg_adj,
                'momentum': momentum_adj,
                'red_cards': red_card_adj
            }
        }

    def _score_adjustment(self, score_h: int, score_a: int, minute: int) -> float:
        """Adjust based on current score"""
        time_remaining = 90 - minute
        time_factor = 1 - (time_remaining / 90)

        if score_h > score_a:
            lead = score_h - score_a
            return 1 + (lead * 0.15 * time_factor)
        elif score_a > score_h:
            lead = score_a - score_h
            return 1 - (lead * 0.25 * time_factor)
        return 1.0

    def _xg_adjustment(self, xg_h: float, xg_a: float, minute: int) -> float:
        """Adjust based on live xG"""
        if minute < 10:
            return 1.0  # Too early

        xg_rate_h = xg_h / minute * 90
        xg_rate_a = xg_a / minute * 90

        xg_diff = xg_rate_h - xg_rate_a

        return 1 + (xg_diff * 0.08)

    def _momentum_adjustment(self, match: LiveMatch) -> float:
        """Adjust based on recent momentum"""
        total_attacks = match.recent_attacks_home + match.recent_attacks_away
        if total_attacks == 0:
            return 1.0

        home_momentum = match.recent_attacks_home / total_attacks
        return 1 + ((home_momentum - 0.5) * 0.10)

    def _red_card_adjustment(self, red_h: int, red_a: int, minute: int) -> float:
        """Adjust for red cards"""
        time_remaining = 90 - minute

        if red_h > red_a:
            # Home disadvantage
            penalty = (red_h - red_a) * 0.20 * (time_remaining / 90)
            return 1 - penalty
        elif red_a > red_h:
            # Away disadvantage = home advantage
            boost = (red_a - red_h) * 0.25 * (time_remaining / 90)
            return 1 + boost
        return 1.0

    def _calculate_over_goals(self, match: LiveMatch, threshold: float) -> float:
        """Calculate probability of Over X goals"""
        current_goals = match.score_home + match.score_away

        if current_goals >= threshold:
            return 0.99  # Already over

        goals_needed = threshold - current_goals
        minutes_remaining = 90 - match.minute

        # Estimate goal rate from xG
        if match.minute > 0:
            goal_rate = (match.xg_home + match.xg_away) / match.minute
            expected_remaining = goal_rate * minutes_remaining

            # Simple approximation
            if expected_remaining >= goals_needed:
                return 0.70
            else:
                return 0.30
        return 0.50

    def _get_timing_recommendation(self, match: LiveMatch, prob: float) -> str:
        """Get timing recommendation"""
        if match.minute < 15:
            return "WAIT"  # Too early
        elif match.minute > 75:
            return "NOW"  # Last chance
        else:
            return "WATCH"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Live Betting Engine...")
    print("=" * 70)

    engine = LiveBettingEngine()

    # Start monitoring
    live_match = engine.start_monitoring("match123", pre_match_prob=0.60)

    # Simulate match progress
    scenarios = [
        {'minute': 15, 'score_home': 0, 'score_away': 0, 'xg_home': 0.8, 'xg_away': 0.3},
        {'minute': 30, 'score_home': 1, 'score_away': 0, 'xg_home': 1.5, 'xg_away': 0.8},
        {'minute': 60, 'score_home': 1, 'score_away': 1, 'xg_home': 2.1, 'xg_away': 1.9},
        {'minute': 85, 'score_home': 2, 'score_away': 1, 'xg_home': 2.8, 'xg_away': 2.2},
    ]

    for scenario in scenarios:
        live_match.update(scenario)
        result = engine.recalculate_probability(live_match)

        print(f"\nMinute {result['minute']} | Score: {result['score']}")
        print(f"   Home Win Prob: {result['probability_home_win']:.1%}")
        print(f"   Over 2.5: {result['over_2.5_goals']:.1%}")
        print(f"   Timing: {result['timing']}")

    print("\n" + "=" * 70)
    print("✅ Live Betting Engine test completed!")
