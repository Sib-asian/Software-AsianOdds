"""
BLOCCO 13: Statistical Arbitrage Detector

Sistema avanzato per rilevare inefficienze di mercato e opportunità
di arbitraggio statistico tra mercati correlati e bookmakers diversi.

Features:
- Sure bet detection (arbitraggio classico)
- Statistical arbitrage opportunities
- Cross-market inefficiencies
- Value bet identification attraverso cross-validation
- Bookmaker comparison
- Market efficiency scoring
- Mispricing detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ArbitrageOpportunity:
    """Opportunità di arbitraggio rilevata"""
    opportunity_type: str  # "SURE_BET", "STATISTICAL_ARB", "VALUE", "MISPRICING"
    markets_involved: List[str]
    bookmakers_involved: List[str]
    guaranteed_profit_pct: float  # Per sure bets
    expected_value_pct: float  # Per statistical arb
    confidence: float  # 0-1
    risk_level: str  # LOW, MEDIUM, HIGH
    stakes_allocation: Dict[str, float]
    reasoning: List[str]
    time_sensitivity: str  # IMMEDIATE, HOUR, DAY
    details: Dict


@dataclass
class MarketEfficiencyScore:
    """Score di efficienza del mercato"""
    efficiency_score: float  # 0-100 (100 = perfettamente efficiente)
    overround: float  # Margine del bookmaker
    value_opportunities: int
    mispricing_detected: bool
    liquidity_assessment: str  # LOW, MEDIUM, HIGH
    recommendation: str


class StatisticalArbitrageDetector:
    """
    Detector di arbitraggio e inefficienze di mercato.

    Identifica opportunità dove le quote del mercato non riflettono
    correttamente le vere probabilità, permettendo profitti sistemici.
    """

    def __init__(
        self,
        sure_bet_threshold: float = 0.01,
        value_threshold: float = 0.05,
        confidence_threshold: float = 0.75
    ):
        """
        Args:
            sure_bet_threshold: Minimo profitto garantito per sure bet (%)
            value_threshold: Minimo EV per value bet (%)
            confidence_threshold: Minima confidence per segnalare opportunità
        """
        self.sure_bet_threshold = sure_bet_threshold
        self.value_threshold = value_threshold
        self.confidence_threshold = confidence_threshold

    def detect_sure_bet(
        self,
        odds_by_bookmaker: Dict[str, Dict[str, float]]
    ) -> Optional[ArbitrageOpportunity]:
        """
        Rileva sure bet (arbitraggio classico) tra bookmakers.

        Args:
            odds_by_bookmaker: {bookmaker: {outcome: odds}}
                              es. {"Bet365": {"home": 2.1, "draw": 3.5, "away": 3.2}}

        Returns:
            ArbitrageOpportunity se trovato, None altrimenti
        """
        if len(odds_by_bookmaker) < 2:
            return None

        # Trova le quote migliori per ogni outcome
        all_outcomes = set()
        for book_odds in odds_by_bookmaker.values():
            all_outcomes.update(book_odds.keys())

        best_odds = {}
        best_bookmaker = {}

        for outcome in all_outcomes:
            best_odd = 0
            best_book = None

            for bookmaker, odds_dict in odds_by_bookmaker.items():
                if outcome in odds_dict and odds_dict[outcome] > best_odd:
                    best_odd = odds_dict[outcome]
                    best_book = bookmaker

            if best_odd > 0:
                best_odds[outcome] = best_odd
                best_bookmaker[outcome] = best_book

        # Calcola implied probabilities
        implied_probs = {
            outcome: 1 / odd
            for outcome, odd in best_odds.items()
        }

        total_implied_prob = sum(implied_probs.values())

        # Se somma < 1, c'è arbitraggio!
        if total_implied_prob < 1.0 - self.sure_bet_threshold:
            # Calcola profitto garantito
            guaranteed_profit = (1 / total_implied_prob - 1) * 100

            # Calcola stakes ottimali
            stakes = {
                outcome: (implied_probs[outcome] / total_implied_prob) * 100
                for outcome in best_odds.keys()
            }

            # Bookmakers coinvolti
            bookmakers = list(set(best_bookmaker.values()))

            reasoning = [
                f"Sure bet detected with {guaranteed_profit:.2f}% guaranteed profit",
                f"Total implied probability: {total_implied_prob:.4f}",
                f"Best odds combination across {len(bookmakers)} bookmakers"
            ]

            details = {
                "best_odds": best_odds,
                "best_bookmakers": best_bookmaker,
                "implied_probabilities": implied_probs,
                "total_implied_prob": total_implied_prob
            }

            return ArbitrageOpportunity(
                opportunity_type="SURE_BET",
                markets_involved=list(best_odds.keys()),
                bookmakers_involved=bookmakers,
                guaranteed_profit_pct=guaranteed_profit,
                expected_value_pct=guaranteed_profit,
                confidence=1.0,  # Sure bet = 100% confidence
                risk_level="LOW",
                stakes_allocation=stakes,
                reasoning=reasoning,
                time_sensitivity="IMMEDIATE",  # Sure bets vanish quickly
                details=details
            )

        return None

    def detect_statistical_arbitrage(
        self,
        model_probability: float,
        market_odds: float,
        model_confidence: float,
        historical_edge: Optional[float] = None
    ) -> Optional[ArbitrageOpportunity]:
        """
        Rileva opportunità di arbitraggio statistico.

        Statistical arbitrage: quando model probability è significativamente
        diversa dalla implied probability del mercato.

        Args:
            model_probability: Probabilità stimata dal modello
            market_odds: Quote offerte dal mercato
            model_confidence: Confidence del modello (0-1)
            historical_edge: Edge storico osservato (opzionale)

        Returns:
            ArbitrageOpportunity se trovato
        """
        implied_prob = 1 / market_odds

        # Calcola discrepanza
        discrepancy = model_probability - implied_prob

        # Expected value
        ev = (model_probability * market_odds) - 1
        ev_pct = ev * 100

        # Deve superare threshold e confidence minima
        if ev_pct < self.value_threshold or model_confidence < self.confidence_threshold:
            return None

        # Assess risk
        if model_confidence > 0.85 and ev_pct > 10:
            risk_level = "LOW"
        elif model_confidence > 0.70 and ev_pct > 5:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # Kelly criterion per stake allocation
        kelly_fraction = ev / (market_odds - 1)
        kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap al 25%

        stakes = {
            "bet": kelly_fraction * 100,
            "skip": (1 - kelly_fraction) * 100
        }

        reasoning = [
            f"Model probability ({model_probability:.3f}) vs Market ({implied_prob:.3f})",
            f"Expected value: {ev_pct:.2f}%",
            f"Model confidence: {model_confidence:.2f}",
            f"Recommended Kelly stake: {kelly_fraction*100:.1f}%"
        ]

        if historical_edge:
            reasoning.append(f"Historical edge: {historical_edge:.2f}%")

        # Time sensitivity basato su EV e confidence
        if ev_pct > 15 and model_confidence > 0.85:
            time_sensitivity = "IMMEDIATE"
        elif ev_pct > 8:
            time_sensitivity = "HOUR"
        else:
            time_sensitivity = "DAY"

        details = {
            "model_probability": model_probability,
            "implied_probability": implied_prob,
            "market_odds": market_odds,
            "discrepancy": discrepancy,
            "expected_value": ev,
            "kelly_fraction": kelly_fraction
        }

        return ArbitrageOpportunity(
            opportunity_type="STATISTICAL_ARB",
            markets_involved=["primary_market"],
            bookmakers_involved=["current"],
            guaranteed_profit_pct=0.0,
            expected_value_pct=ev_pct,
            confidence=model_confidence,
            risk_level=risk_level,
            stakes_allocation=stakes,
            reasoning=reasoning,
            time_sensitivity=time_sensitivity,
            details=details
        )

    def detect_cross_market_inefficiency(
        self,
        market_1x2: Dict[str, float],
        market_ou: Dict[str, float],
        market_btts: Dict[str, float],
        model_predictions: Dict
    ) -> List[ArbitrageOpportunity]:
        """
        Rileva inefficienze tra mercati correlati.

        Args:
            market_1x2: Odds per 1X2 {home, draw, away}
            market_ou: Odds per Over/Under {over, under}
            market_btts: Odds per BTTS {yes, no}
            model_predictions: Predizioni del modello per tutti i mercati

        Returns:
            Lista di opportunità rilevate
        """
        opportunities = []

        # Check 1: Inconsistenza tra 1X2 e BTTS
        # Se home win ha odds alte MA BTTS=Yes ha odds basse,
        # potrebbe essere inconsistenza (home dovrebbe segnare)

        implied_home_win = 1 / market_1x2.get("home", 2.0)
        implied_btts_yes = 1 / market_btts.get("yes", 2.0)

        model_home_win = model_predictions.get("prob_home", 0.5)
        model_btts_yes = model_predictions.get("prob_btts_yes", 0.5)

        # Se model dice home win è probabile E btts_yes è probabile,
        # ma il mercato sottostima uno dei due, c'è opportunità

        home_undervalued = model_home_win > implied_home_win * 1.15
        btts_undervalued = model_btts_yes > implied_btts_yes * 1.15

        if home_undervalued and btts_undervalued:
            # Possibile hedge opportunity
            ev_combined = (
                (model_home_win * market_1x2["home"] - 1) +
                (model_btts_yes * market_btts["yes"] - 1)
            ) / 2 * 100

            if ev_combined > self.value_threshold:
                opportunities.append(ArbitrageOpportunity(
                    opportunity_type="MISPRICING",
                    markets_involved=["1X2", "BTTS"],
                    bookmakers_involved=["current"],
                    guaranteed_profit_pct=0.0,
                    expected_value_pct=ev_combined,
                    confidence=0.75,
                    risk_level="MEDIUM",
                    stakes_allocation={"1x2_home": 50, "btts_yes": 50},
                    reasoning=[
                        "Cross-market inefficiency detected",
                        "Home win and BTTS both undervalued by market",
                        f"Combined EV: {ev_combined:.2f}%"
                    ],
                    time_sensitivity="HOUR",
                    details={
                        "home_ev": (model_home_win * market_1x2["home"] - 1) * 100,
                        "btts_ev": (model_btts_yes * market_btts["yes"] - 1) * 100
                    }
                ))

        # Check 2: Over/Under vs score probabilities
        implied_over = 1 / market_ou.get("over", 2.0)
        model_over = model_predictions.get("prob_over", 0.5)

        if model_over > implied_over * 1.20:
            ev_over = (model_over * market_ou["over"] - 1) * 100

            if ev_over > self.value_threshold:
                opportunities.append(ArbitrageOpportunity(
                    opportunity_type="VALUE",
                    markets_involved=["Over/Under"],
                    bookmakers_involved=["current"],
                    guaranteed_profit_pct=0.0,
                    expected_value_pct=ev_over,
                    confidence=0.80,
                    risk_level="MEDIUM",
                    stakes_allocation={"over": 100},
                    reasoning=[
                        f"Over significantly undervalued",
                        f"Model: {model_over:.3f}, Implied: {implied_over:.3f}",
                        f"EV: {ev_over:.2f}%"
                    ],
                    time_sensitivity="HOUR",
                    details={"model_over": model_over, "implied_over": implied_over}
                ))

        return opportunities

    def calculate_market_efficiency(
        self,
        odds_dict: Dict[str, float],
        model_probabilities: Dict[str, float]
    ) -> MarketEfficiencyScore:
        """
        Calcola score di efficienza del mercato.

        Args:
            odds_dict: {outcome: odds}
            model_probabilities: {outcome: probability}

        Returns:
            MarketEfficiencyScore
        """
        # Calcola overround (margine del bookmaker)
        implied_probs = {k: 1/v for k, v in odds_dict.items()}
        overround = sum(implied_probs.values()) - 1.0
        overround_pct = overround * 100

        # Confronta con model predictions
        discrepancies = []
        value_opportunities = 0

        for outcome in model_probabilities.keys():
            if outcome in implied_probs:
                model_prob = model_probabilities[outcome]
                implied_prob = implied_probs[outcome]
                discrepancy = abs(model_prob - implied_prob)
                discrepancies.append(discrepancy)

                # Check value
                if model_prob > implied_prob * 1.15:
                    value_opportunities += 1

        # Efficiency score
        # Alta efficienza = basso overround + piccole discrepanze
        avg_discrepancy = np.mean(discrepancies) if discrepancies else 0

        # Score components
        overround_score = max(0, 100 - overround_pct * 500)  # Penalizza alto overround
        discrepancy_score = max(0, 100 - avg_discrepancy * 200)  # Penalizza grandi discrepanze

        efficiency_score = (overround_score + discrepancy_score) / 2

        # Mispricing detected?
        mispricing = value_opportunities > 0 or avg_discrepancy > 0.10

        # Liquidity assessment (basato su overround come proxy)
        if overround_pct < 3:
            liquidity = "HIGH"
        elif overround_pct < 7:
            liquidity = "MEDIUM"
        else:
            liquidity = "LOW"

        # Recommendation
        if efficiency_score > 80 and not mispricing:
            recommendation = "Market highly efficient - be cautious"
        elif value_opportunities > 0:
            recommendation = f"Found {value_opportunities} value opportunity(ies)"
        elif efficiency_score < 50:
            recommendation = "Market inefficient - good for finding value"
        else:
            recommendation = "Market moderately efficient"

        return MarketEfficiencyScore(
            efficiency_score=efficiency_score,
            overround=overround,
            value_opportunities=value_opportunities,
            mispricing_detected=mispricing,
            liquidity_assessment=liquidity,
            recommendation=recommendation
        )

    def scan_for_all_opportunities(
        self,
        data: Dict
    ) -> Dict[str, List[ArbitrageOpportunity]]:
        """
        Scansione completa per tutte le opportunità.

        Args:
            data: Dict con tutti i dati necessari

        Returns:
            Dict {opportunity_type: [opportunities]}
        """
        all_opportunities = {
            "sure_bets": [],
            "statistical_arb": [],
            "cross_market": [],
            "value_bets": []
        }

        # Check sure bets
        if "odds_by_bookmaker" in data:
            sure_bet = self.detect_sure_bet(data["odds_by_bookmaker"])
            if sure_bet:
                all_opportunities["sure_bets"].append(sure_bet)

        # Check statistical arbitrage
        if all(k in data for k in ["model_probability", "market_odds", "model_confidence"]):
            stat_arb = self.detect_statistical_arbitrage(
                data["model_probability"],
                data["market_odds"],
                data["model_confidence"],
                data.get("historical_edge")
            )
            if stat_arb:
                all_opportunities["statistical_arb"].append(stat_arb)

        # Check cross-market
        if all(k in data for k in ["market_1x2", "market_ou", "market_btts", "model_predictions"]):
            cross_market_opps = self.detect_cross_market_inefficiency(
                data["market_1x2"],
                data["market_ou"],
                data["market_btts"],
                data["model_predictions"]
            )
            all_opportunities["cross_market"].extend(cross_market_opps)

        return all_opportunities


if __name__ == "__main__":
    # Test del sistema
    detector = StatisticalArbitrageDetector(
        sure_bet_threshold=0.01,
        value_threshold=0.05
    )

    # Test 1: Sure bet detection
    print("=== TEST 1: Sure Bet Detection ===")
    odds_multi_bookmaker = {
        "Bet365": {"home": 2.10, "draw": 3.40, "away": 3.50},
        "Pinnacle": {"home": 2.05, "draw": 3.60, "away": 3.80},
        "William Hill": {"home": 2.15, "draw": 3.50, "away": 3.60}
    }

    sure_bet = detector.detect_sure_bet(odds_multi_bookmaker)
    if sure_bet:
        print(f"Sure Bet Found!")
        print(f"  Guaranteed Profit: {sure_bet.guaranteed_profit_pct:.2f}%")
        print(f"  Stakes: {sure_bet.stakes_allocation}")
        print(f"  Bookmakers: {sure_bet.bookmakers_involved}")
    else:
        print("No sure bet found")

    # Test 2: Statistical arbitrage
    print("\n=== TEST 2: Statistical Arbitrage ===")
    stat_arb = detector.detect_statistical_arbitrage(
        model_probability=0.58,
        market_odds=2.0,  # Implied = 0.50
        model_confidence=0.85,
        historical_edge=7.5
    )

    if stat_arb:
        print(f"Statistical Arbitrage Found!")
        print(f"  Expected Value: {stat_arb.expected_value_pct:.2f}%")
        print(f"  Risk Level: {stat_arb.risk_level}")
        print(f"  Recommended Stake: {stat_arb.stakes_allocation['bet']:.1f}%")
        print(f"  Reasoning: {stat_arb.reasoning[0]}")

    # Test 3: Market efficiency
    print("\n=== TEST 3: Market Efficiency ===")
    market_odds = {"home": 2.10, "draw": 3.40, "away": 3.80}
    model_probs = {"home": 0.52, "draw": 0.27, "away": 0.21}

    efficiency = detector.calculate_market_efficiency(market_odds, model_probs)
    print(f"Efficiency Score: {efficiency.efficiency_score:.1f}/100")
    print(f"Overround: {efficiency.overround*100:.2f}%")
    print(f"Value Opportunities: {efficiency.value_opportunities}")
    print(f"Recommendation: {efficiency.recommendation}")

    print("\n✓ Statistical Arbitrage Detector Test Completed")
