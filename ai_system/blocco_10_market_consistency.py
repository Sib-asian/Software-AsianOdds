"""
BLOCCO 10: Market Consistency Validator

Sistema che verifica la coerenza probabilistica tra diversi mercati correlati
usando vincoli matematici e logical consistency checks.

Features:
- Constraint checking su mercati correlati (1X2, Over/Under, BTTS, AH)
- Probability coherence validation
- Arbitrage detection
- Market inefficiency identification
- Cross-market validation
- Logical consistency enforcement
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ConsistencyResult:
    """Risultato del check di consistenza"""
    is_consistent: bool
    consistency_score: float  # 0-100 (100 = perfetta consistenza)
    violations: List[Dict]
    warnings: List[str]
    arbitrage_opportunities: List[Dict]
    adjustments_recommended: Dict[str, float]
    confidence: float  # 0-1


class MarketConsistencyValidator:
    """
    Validatore di consistenza tra mercati correlati.

    Verifica che le probabilità assegnate a diversi mercati correlati
    rispettino i vincoli matematici e logici fondamentali.
    """

    def __init__(self, tolerance: float = 0.01, strict_mode: bool = False):
        """
        Args:
            tolerance: Tolleranza per errori numerici
            strict_mode: Se True, applica vincoli più stretti
        """
        self.tolerance = tolerance
        self.strict_mode = strict_mode

        # Thresholds
        if strict_mode:
            self.consistency_threshold = 0.005
            self.arbitrage_threshold = 0.02
        else:
            self.consistency_threshold = 0.01
            self.arbitrage_threshold = 0.03

    def validate_1x2_market(
        self,
        prob_home: float,
        prob_draw: float,
        prob_away: float
    ) -> ConsistencyResult:
        """
        Valida coerenza del mercato 1X2.

        Args:
            prob_home: P(Home win)
            prob_draw: P(Draw)
            prob_away: P(Away win)

        Returns:
            ConsistencyResult
        """
        violations = []
        warnings_list = []

        # Check 1: Tutte le prob devono essere in [0, 1]
        for name, prob in [("Home", prob_home), ("Draw", prob_draw), ("Away", prob_away)]:
            if not (0 <= prob <= 1):
                violations.append({
                    "type": "OUT_OF_BOUNDS",
                    "market": "1X2",
                    "detail": f"{name} probability {prob:.4f} outside [0,1]"
                })

        # Check 2: Somma deve essere ≈ 1
        total = prob_home + prob_draw + prob_away
        deviation = abs(total - 1.0)

        if deviation > self.consistency_threshold:
            violations.append({
                "type": "SUM_CONSTRAINT",
                "market": "1X2",
                "detail": f"Sum = {total:.4f}, deviation = {deviation:.4f}"
            })

        # Check 3: Nessuna prob dovrebbe essere troppo vicina a 0 o 1
        for name, prob in [("Home", prob_home), ("Draw", prob_draw), ("Away", prob_away)]:
            if prob < 0.01:
                warnings_list.append(f"{name} probability very low: {prob:.4f}")
            elif prob > 0.95:
                warnings_list.append(f"{name} probability very high: {prob:.4f}")

        # Consistency score
        if len(violations) == 0:
            consistency_score = 100 * (1 - deviation)
        else:
            consistency_score = max(0, 50 - len(violations) * 20)

        is_consistent = len(violations) == 0 and deviation <= self.consistency_threshold

        # Adjustments (normalizza se necessario)
        if not is_consistent and total > 0:
            adjustments = {
                "prob_home": prob_home / total,
                "prob_draw": prob_draw / total,
                "prob_away": prob_away / total
            }
        else:
            adjustments = {}

        return ConsistencyResult(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            violations=violations,
            warnings=warnings_list,
            arbitrage_opportunities=[],
            adjustments_recommended=adjustments,
            confidence=1.0
        )

    def validate_over_under_consistency(
        self,
        prob_over: float,
        prob_under: float,
        line: float
    ) -> ConsistencyResult:
        """
        Valida coerenza del mercato Over/Under.

        Args:
            prob_over: P(Over line)
            prob_under: P(Under line)
            line: Linea (es. 2.5)

        Returns:
            ConsistencyResult
        """
        violations = []
        warnings_list = []

        # Check 1: Bounds
        if not (0 <= prob_over <= 1):
            violations.append({
                "type": "OUT_OF_BOUNDS",
                "market": "Over/Under",
                "detail": f"Over probability {prob_over:.4f} outside [0,1]"
            })

        if not (0 <= prob_under <= 1):
            violations.append({
                "type": "OUT_OF_BOUNDS",
                "market": "Over/Under",
                "detail": f"Under probability {prob_under:.4f} outside [0,1]"
            })

        # Check 2: Over + Under ≈ 1
        total = prob_over + prob_under
        deviation = abs(total - 1.0)

        if deviation > self.consistency_threshold:
            violations.append({
                "type": "SUM_CONSTRAINT",
                "market": "Over/Under",
                "detail": f"Over + Under = {total:.4f}, should be 1.0"
            })

        # Check 3: Sanity check sulla linea
        if line < 0.5:
            warnings_list.append(f"Unusually low line: {line}")
        elif line > 7.5:
            warnings_list.append(f"Unusually high line: {line}")

        # Check 4: Prob Over dovrebbe diminuire con linee più alte
        # (questo check richiederebbe dati storici su più linee)

        # Consistency score
        consistency_score = 100 * (1 - min(1.0, deviation * 10))

        is_consistent = len(violations) == 0

        # Adjustments
        if not is_consistent and total > 0:
            adjustments = {
                "prob_over": prob_over / total,
                "prob_under": prob_under / total
            }
        else:
            adjustments = {}

        return ConsistencyResult(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            violations=violations,
            warnings=warnings_list,
            arbitrage_opportunities=[],
            adjustments_recommended=adjustments,
            confidence=1.0
        )

    def validate_btts_with_score_probabilities(
        self,
        prob_btts_yes: float,
        prob_btts_no: float,
        prob_home_scores: float,
        prob_away_scores: float
    ) -> ConsistencyResult:
        """
        Valida coerenza tra BTTS e probabilità che ogni squadra segni.

        Constraint: P(BTTS=Yes) = P(Home≥1) × P(Away≥1)

        Args:
            prob_btts_yes: P(Both teams score)
            prob_btts_no: P(At least one team doesn't score)
            prob_home_scores: P(Home goals ≥ 1)
            prob_away_scores: P(Away goals ≥ 1)

        Returns:
            ConsistencyResult
        """
        violations = []
        warnings_list = []

        # Check 1: BTTS Yes + BTTS No ≈ 1
        total_btts = prob_btts_yes + prob_btts_no
        deviation_btts = abs(total_btts - 1.0)

        if deviation_btts > self.consistency_threshold:
            violations.append({
                "type": "SUM_CONSTRAINT",
                "market": "BTTS",
                "detail": f"BTTS Yes + No = {total_btts:.4f}"
            })

        # Check 2: Logical constraint
        # P(BTTS=Yes) dovrebbe essere ≈ P(Home≥1) × P(Away≥1)
        expected_btts_yes = prob_home_scores * prob_away_scores
        btts_deviation = abs(prob_btts_yes - expected_btts_yes)

        if btts_deviation > self.consistency_threshold * 2:
            violations.append({
                "type": "LOGICAL_CONSTRAINT",
                "market": "BTTS",
                "detail": (
                    f"BTTS Yes = {prob_btts_yes:.4f}, "
                    f"expected ≈ {expected_btts_yes:.4f} "
                    f"(P(H≥1)×P(A≥1))"
                )
            })

        # Check 3: P(BTTS=No) dovrebbe essere ≈ 1 - P(BTTS=Yes)
        expected_btts_no = 1 - expected_btts_yes
        if abs(prob_btts_no - expected_btts_no) > self.consistency_threshold * 2:
            warnings_list.append(
                f"BTTS No = {prob_btts_no:.4f}, expected ≈ {expected_btts_no:.4f}"
            )

        # Consistency score
        total_deviation = deviation_btts + btts_deviation
        consistency_score = max(0, 100 * (1 - total_deviation * 5))

        is_consistent = len(violations) == 0

        # Adjustments
        adjustments = {}
        if not is_consistent:
            # Usa il constraint logico per ajustare
            adjustments["prob_btts_yes"] = expected_btts_yes
            adjustments["prob_btts_no"] = 1 - expected_btts_yes

        return ConsistencyResult(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            violations=violations,
            warnings=warnings_list,
            arbitrage_opportunities=[],
            adjustments_recommended=adjustments,
            confidence=0.9
        )

    def validate_asian_handicap_consistency(
        self,
        prob_ah_home: float,
        prob_ah_push: float,
        prob_ah_away: float,
        handicap: float,
        prob_1x2_home: float,
        prob_1x2_draw: float,
        prob_1x2_away: float
    ) -> ConsistencyResult:
        """
        Valida coerenza tra Asian Handicap e mercato 1X2.

        Args:
            prob_ah_home: P(Home covers handicap)
            prob_ah_push: P(Push)
            prob_ah_away: P(Away covers handicap)
            handicap: Valore handicap (es. -0.5, -1.0)
            prob_1x2_*: Probabilità dal mercato 1X2

        Returns:
            ConsistencyResult
        """
        violations = []
        warnings_list = []

        # Check 1: AH probabilities sum to 1
        total_ah = prob_ah_home + prob_ah_push + prob_ah_away
        deviation_ah = abs(total_ah - 1.0)

        if deviation_ah > self.consistency_threshold:
            violations.append({
                "type": "SUM_CONSTRAINT",
                "market": "Asian Handicap",
                "detail": f"Sum = {total_ah:.4f}"
            })

        # Check 2: Logical consistency con 1X2
        # Per handicap intero (es. -1), push è possibile
        # Per handicap .5 (es. -0.5), push dovrebbe essere ≈ 0

        if abs(handicap - round(handicap)) > 0.01:
            # Handicap non intero (.5, .25, .75) => no push
            if prob_ah_push > self.tolerance:
                violations.append({
                    "type": "LOGICAL_CONSTRAINT",
                    "market": "Asian Handicap",
                    "detail": f"Non-integer handicap {handicap} should have push ≈ 0, got {prob_ah_push:.4f}"
                })

        # Check 3: Per handicap 0, AH dovrebbe matchare 1X2
        if abs(handicap) < self.tolerance:
            # AH(0) Home = 1X2 Home + Draw/2
            expected_ah_home = prob_1x2_home + prob_1x2_draw / 2
            expected_ah_away = prob_1x2_away + prob_1x2_draw / 2

            if abs(prob_ah_home - expected_ah_home) > self.consistency_threshold * 2:
                warnings_list.append(
                    f"AH(0) Home = {prob_ah_home:.3f}, "
                    f"expected ≈ {expected_ah_home:.3f} from 1X2"
                )

        # Consistency score
        consistency_score = max(0, 100 * (1 - deviation_ah * 10))

        is_consistent = len(violations) == 0

        adjustments = {}
        if not is_consistent and total_ah > 0:
            adjustments["prob_ah_home"] = prob_ah_home / total_ah
            adjustments["prob_ah_push"] = prob_ah_push / total_ah
            adjustments["prob_ah_away"] = prob_ah_away / total_ah

        return ConsistencyResult(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            violations=violations,
            warnings=warnings_list,
            arbitrage_opportunities=[],
            adjustments_recommended=adjustments,
            confidence=0.85
        )

    def detect_arbitrage(
        self,
        markets: Dict[str, Dict[str, float]]
    ) -> List[Dict]:
        """
        Rileva opportunità di arbitraggio tra mercati.

        Args:
            markets: Dict {market_name: {outcome: odds}}

        Returns:
            Lista di arbitrage opportunities
        """
        arbitrage_opps = []

        # Check 1X2 market
        if "1x2" in markets:
            odds = markets["1x2"]
            if all(k in odds for k in ["home", "draw", "away"]):
                implied_probs = {
                    k: 1/v for k, v in odds.items()
                }
                total_implied = sum(implied_probs.values())

                # Se somma < 1, c'è arbitraggio
                if total_implied < 1.0 - self.arbitrage_threshold:
                    arbitrage_opps.append({
                        "market": "1X2",
                        "type": "SURE_BET",
                        "total_implied_prob": total_implied,
                        "guaranteed_profit_pct": (1 - total_implied) * 100,
                        "stakes": {
                            k: v / total_implied
                            for k, v in implied_probs.items()
                        }
                    })

        # Check Over/Under
        if "over_under" in markets:
            odds = markets["over_under"]
            if "over" in odds and "under" in odds:
                implied_over = 1 / odds["over"]
                implied_under = 1 / odds["under"]
                total_implied = implied_over + implied_under

                if total_implied < 1.0 - self.arbitrage_threshold:
                    arbitrage_opps.append({
                        "market": "Over/Under",
                        "type": "SURE_BET",
                        "total_implied_prob": total_implied,
                        "guaranteed_profit_pct": (1 - total_implied) * 100,
                        "stakes": {
                            "over": implied_over / total_implied,
                            "under": implied_under / total_implied
                        }
                    })

        return arbitrage_opps

    def comprehensive_consistency_check(
        self,
        markets_data: Dict
    ) -> Dict[str, ConsistencyResult]:
        """
        Esegue check completo di consistenza su tutti i mercati.

        Args:
            markets_data: Dict con dati di tutti i mercati

        Returns:
            Dict {market: ConsistencyResult}
        """
        results = {}

        # Check 1X2
        if all(k in markets_data for k in ["prob_home", "prob_draw", "prob_away"]):
            results["1x2"] = self.validate_1x2_market(
                markets_data["prob_home"],
                markets_data["prob_draw"],
                markets_data["prob_away"]
            )

        # Check Over/Under
        if all(k in markets_data for k in ["prob_over", "prob_under", "ou_line"]):
            results["over_under"] = self.validate_over_under_consistency(
                markets_data["prob_over"],
                markets_data["prob_under"],
                markets_data["ou_line"]
            )

        # Check BTTS
        if all(k in markets_data for k in [
            "prob_btts_yes", "prob_btts_no",
            "prob_home_scores", "prob_away_scores"
        ]):
            results["btts"] = self.validate_btts_with_score_probabilities(
                markets_data["prob_btts_yes"],
                markets_data["prob_btts_no"],
                markets_data["prob_home_scores"],
                markets_data["prob_away_scores"]
            )

        # Check Asian Handicap
        if all(k in markets_data for k in [
            "prob_ah_home", "prob_ah_push", "prob_ah_away", "handicap"
        ]) and "1x2" in results:
            results["asian_handicap"] = self.validate_asian_handicap_consistency(
                markets_data["prob_ah_home"],
                markets_data["prob_ah_push"],
                markets_data["prob_ah_away"],
                markets_data["handicap"],
                markets_data["prob_home"],
                markets_data["prob_draw"],
                markets_data["prob_away"]
            )

        # Arbitrage detection
        if "odds" in markets_data:
            arbitrage_opps = self.detect_arbitrage(markets_data["odds"])
            if arbitrage_opps:
                for market_name in results:
                    results[market_name].arbitrage_opportunities.extend(arbitrage_opps)

        return results

    def get_overall_consistency_score(
        self,
        results: Dict[str, ConsistencyResult]
    ) -> float:
        """
        Calcola consistency score complessivo.

        Args:
            results: Dict di ConsistencyResult per ogni mercato

        Returns:
            Score 0-100
        """
        if not results:
            return 0.0

        scores = [r.consistency_score for r in results.values()]
        confidences = [r.confidence for r in results.values()]

        # Weighted average
        weighted_score = sum(
            s * c for s, c in zip(scores, confidences)
        ) / sum(confidences)

        return weighted_score


if __name__ == "__main__":
    # Test del sistema
    validator = MarketConsistencyValidator(tolerance=0.01, strict_mode=False)

    # Test 1: 1X2 consistency
    print("=== TEST 1: 1X2 Market Consistency ===")
    result_1x2 = validator.validate_1x2_market(
        prob_home=0.45,
        prob_draw=0.30,
        prob_away=0.26  # Sum = 1.01, slight deviation
    )
    print(f"Consistent: {result_1x2.is_consistent}")
    print(f"Score: {result_1x2.consistency_score:.1f}")
    print(f"Violations: {len(result_1x2.violations)}")

    # Test 2: BTTS logical consistency
    print("\n=== TEST 2: BTTS Logical Consistency ===")
    result_btts = validator.validate_btts_with_score_probabilities(
        prob_btts_yes=0.55,
        prob_btts_no=0.45,
        prob_home_scores=0.85,
        prob_away_scores=0.70
    )
    # Expected: 0.85 * 0.70 = 0.595, given is 0.55 (small deviation)
    print(f"Consistent: {result_btts.is_consistent}")
    print(f"Score: {result_btts.consistency_score:.1f}")
    if result_btts.violations:
        print(f"Violations: {result_btts.violations[0]['detail']}")

    # Test 3: Comprehensive check
    print("\n=== TEST 3: Comprehensive Consistency Check ===")
    markets_data = {
        "prob_home": 0.50,
        "prob_draw": 0.27,
        "prob_away": 0.23,
        "prob_over": 0.58,
        "prob_under": 0.42,
        "ou_line": 2.5,
        "prob_btts_yes": 0.54,
        "prob_btts_no": 0.46,
        "prob_home_scores": 0.82,
        "prob_away_scores": 0.68
    }

    comprehensive_results = validator.comprehensive_consistency_check(markets_data)
    overall_score = validator.get_overall_consistency_score(comprehensive_results)

    print(f"Overall Consistency Score: {overall_score:.1f}")
    for market, result in comprehensive_results.items():
        print(f"\n{market}:")
        print(f"  Consistent: {result.is_consistent}")
        print(f"  Score: {result.consistency_score:.1f}")
        if result.violations:
            print(f"  Violations: {len(result.violations)}")

    print("\n✓ Market Consistency Validator Test Completed")
