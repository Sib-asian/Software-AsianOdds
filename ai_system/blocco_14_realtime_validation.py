"""
BLOCCO 14: Real-time Validation Engine

Sistema di validazione in tempo reale che usa multiple metodologie
per verificare la correttezza e l'affidabilità dei calcoli.

Features:
- Multi-methodology validation
- Cross-validation tra approcci diversi
- Sanity checks automatici
- Numerical stability verification
- Logical consistency enforcement
- Real-time error detection
- Automatic correction suggestions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ValidationResult:
    """Risultato di una validazione"""
    is_valid: bool
    confidence: float  # 0-1
    validation_score: float  # 0-100
    errors: List[Dict]
    warnings: List[str]
    corrections: Dict[str, float]
    methodology_results: Dict[str, bool]
    timestamp: datetime


class RealtimeValidationEngine:
    """
    Engine di validazione real-time multi-metodologia.

    Valida calcoli usando diversi approcci indipendenti e richiede
    consenso tra metodologie per confermare la correttezza.
    """

    def __init__(
        self,
        strict_mode: bool = False,
        tolerance: float = 0.001,
        require_consensus: bool = True
    ):
        """
        Args:
            strict_mode: Se True, applica validazioni più rigorose
            tolerance: Tolleranza per errori numerici
            require_consensus: Se True, richiede accordo tra metodologie
        """
        self.strict_mode = strict_mode
        self.tolerance = tolerance
        self.require_consensus = require_consensus

        # Soglie
        if strict_mode:
            self.max_deviation = 0.005
            self.min_confidence = 0.90
            self.consensus_threshold = 0.90
        else:
            self.max_deviation = 0.01
            self.min_confidence = 0.75
            self.consensus_threshold = 0.75

    def validate_probability_calculation(
        self,
        probability: float,
        lambda_home: float,
        lambda_away: float,
        calculation_method: str,
        market_type: str
    ) -> ValidationResult:
        """
        Valida un calcolo di probabilità usando multiple metodologie.

        Args:
            probability: Probabilità calcolata da validare
            lambda_home: Expected goals casa
            lambda_away: Expected goals trasferta
            calculation_method: Metodo usato per calcolo
            market_type: Tipo di mercato (1X2, O/U, etc.)

        Returns:
            ValidationResult
        """
        errors = []
        warnings_list = []
        methodology_results = {}

        # Method 1: Bounds check
        bounds_valid = self._validate_bounds(probability)
        methodology_results["bounds_check"] = bounds_valid

        if not bounds_valid:
            errors.append({
                "type": "OUT_OF_BOUNDS",
                "detail": f"Probability {probability:.4f} outside [0,1]"
            })

        # Method 2: Sanity check basato su lambda
        sanity_valid = self._sanity_check_probability(
            probability, lambda_home, lambda_away, market_type
        )
        methodology_results["sanity_check"] = sanity_valid

        if not sanity_valid:
            warnings_list.append(
                f"Probability seems inconsistent with lambda values"
            )

        # Method 3: Cross-validation con metodo alternativo
        if calculation_method == "poisson":
            # Verifica con approssimazione normale
            cross_calc = self._cross_validate_poisson(
                probability, lambda_home, lambda_away, market_type
            )
            methodology_results["cross_validation"] = cross_calc["valid"]

            if not cross_calc["valid"]:
                warnings_list.append(
                    f"Cross-validation disagreement: {cross_calc['deviation']:.4f}"
                )

        # Method 4: Numerical stability check
        stability_valid = self._check_numerical_stability(probability)
        methodology_results["numerical_stability"] = stability_valid

        if not stability_valid:
            errors.append({
                "type": "NUMERICAL_INSTABILITY",
                "detail": "Calculation may have numerical issues"
            })

        # Calcola validation score
        n_valid = sum(methodology_results.values())
        n_total = len(methodology_results)
        validation_score = (n_valid / n_total) * 100

        # Consensus check
        consensus = n_valid / n_total >= self.consensus_threshold

        # Is valid?
        is_valid = (
            len(errors) == 0 and
            (not self.require_consensus or consensus)
        )

        # Confidence
        confidence = validation_score / 100

        # Corrections se necessario
        corrections = {}
        if not is_valid and bounds_valid:
            # Suggerisci correzione
            corrections["probability"] = max(0.01, min(0.99, probability))

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            validation_score=validation_score,
            errors=errors,
            warnings=warnings_list,
            corrections=corrections,
            methodology_results=methodology_results,
            timestamp=datetime.now()
        )

    def _validate_bounds(self, value: float) -> bool:
        """Verifica che il valore sia in bounds validi."""
        return 0.0 <= value <= 1.0

    def _sanity_check_probability(
        self,
        probability: float,
        lambda_home: float,
        lambda_away: float,
        market_type: str
    ) -> bool:
        """
        Sanity check: verifica se la probabilità è ragionevole
        dato i lambda values.
        """
        # Per mercato 1X2 Home Win
        if market_type == "home_win":
            # Se lambda_home >> lambda_away, prob dovrebbe essere alta
            if lambda_home > lambda_away * 2 and probability < 0.40:
                return False
            # Se lambda_home << lambda_away, prob dovrebbe essere bassa
            if lambda_home < lambda_away * 0.5 and probability > 0.30:
                return False

        # Per mercato Over 2.5
        elif market_type == "over_2.5":
            total_lambda = lambda_home + lambda_away
            # Se total_lambda > 3.5, Over 2.5 dovrebbe essere > 0.50
            if total_lambda > 3.5 and probability < 0.50:
                return False
            # Se total_lambda < 1.5, Over 2.5 dovrebbe essere < 0.30
            if total_lambda < 1.5 and probability > 0.30:
                return False

        return True

    def _cross_validate_poisson(
        self,
        probability: float,
        lambda_home: float,
        lambda_away: float,
        market_type: str
    ) -> Dict:
        """
        Cross-valida con approssimazione alternativa.
        """
        # Usa approssimazione normale per goals totali
        if market_type == "over_2.5":
            total_lambda = lambda_home + lambda_away
            total_std = np.sqrt(total_lambda)

            # P(Total > 2.5) con normale
            z_score = (2.5 - total_lambda) / total_std
            approx_prob = 1 - 0.5 * (1 + np.tanh(z_score / np.sqrt(2)))

            deviation = abs(probability - approx_prob)
            valid = deviation < self.max_deviation * 2

            return {"valid": valid, "deviation": deviation, "approx": approx_prob}

        # Per altri mercati, default valid
        return {"valid": True, "deviation": 0.0, "approx": probability}

    def _check_numerical_stability(self, value: float) -> bool:
        """Verifica stabilità numerica."""
        # Check per NaN, Inf
        if np.isnan(value) or np.isinf(value):
            return False

        # Check per valori troppo vicini a 0 o 1 (possibile underflow/overflow)
        if value < 1e-10 or value > 1 - 1e-10:
            return False

        return True

    def validate_odds_calculation(
        self,
        odds: float,
        probability: float,
        margin: float = 0.05
    ) -> ValidationResult:
        """
        Valida calcolo odds da probabilità.

        Args:
            odds: Odds calcolate
            probability: Probabilità di partenza
            margin: Margine del bookmaker

        Returns:
            ValidationResult
        """
        errors = []
        warnings_list = []
        methodology_results = {}

        # Method 1: Odds = 1 / (probability - margin)
        expected_odds = 1 / (probability + margin) if probability + margin < 1 else 1.01
        deviation = abs(odds - expected_odds)

        calc_valid = deviation < self.max_deviation * expected_odds
        methodology_results["calculation_check"] = calc_valid

        if not calc_valid:
            errors.append({
                "type": "CALCULATION_ERROR",
                "detail": f"Odds {odds:.2f} deviate from expected {expected_odds:.2f}"
            })

        # Method 2: Reverse check
        implied_prob = 1 / odds
        reverse_deviation = abs(implied_prob - (probability + margin))

        reverse_valid = reverse_deviation < self.max_deviation
        methodology_results["reverse_check"] = reverse_valid

        if not reverse_valid:
            warnings_list.append("Reverse calculation shows inconsistency")

        # Method 3: Range check
        if odds < 1.01:
            errors.append({
                "type": "OUT_OF_RANGE",
                "detail": f"Odds {odds:.2f} too low"
            })
            methodology_results["range_check"] = False
        elif odds > 100:
            warnings_list.append(f"Very high odds: {odds:.2f}")
            methodology_results["range_check"] = True
        else:
            methodology_results["range_check"] = True

        # Validation score
        n_valid = sum(methodology_results.values())
        validation_score = (n_valid / len(methodology_results)) * 100

        is_valid = len(errors) == 0

        corrections = {}
        if not is_valid:
            corrections["odds"] = expected_odds

        return ValidationResult(
            is_valid=is_valid,
            confidence=validation_score / 100,
            validation_score=validation_score,
            errors=errors,
            warnings=warnings_list,
            corrections=corrections,
            methodology_results=methodology_results,
            timestamp=datetime.now()
        )

    def validate_market_coherence(
        self,
        probabilities: Dict[str, float],
        market_name: str
    ) -> ValidationResult:
        """
        Valida coerenza di un mercato completo.

        Args:
            probabilities: Dict {outcome: probability}
            market_name: Nome del mercato

        Returns:
            ValidationResult
        """
        errors = []
        warnings_list = []
        methodology_results = {}

        # Method 1: Sum check
        total = sum(probabilities.values())
        sum_valid = abs(total - 1.0) < self.tolerance

        methodology_results["sum_check"] = sum_valid

        if not sum_valid:
            errors.append({
                "type": "SUM_CONSTRAINT",
                "detail": f"Probabilities sum to {total:.4f}, expected 1.0"
            })

        # Method 2: Individual bounds
        all_in_bounds = all(0 <= p <= 1 for p in probabilities.values())
        methodology_results["bounds_check"] = all_in_bounds

        if not all_in_bounds:
            errors.append({
                "type": "OUT_OF_BOUNDS",
                "detail": "Some probabilities outside [0,1]"
            })

        # Method 3: Logical constraints per tipo di mercato
        logical_valid = self._check_logical_constraints(probabilities, market_name)
        methodology_results["logical_check"] = logical_valid

        if not logical_valid:
            warnings_list.append("Logical constraints violated")

        # Method 4: Entropy check (misura della certezza)
        if all_in_bounds and sum_valid:
            entropy = -sum(
                p * np.log(p + 1e-10) for p in probabilities.values()
            )
            max_entropy = np.log(len(probabilities))

            # Entropy troppo bassa = troppo certo (possibile errore)
            # Entropy troppo alta = troppo incerto
            if entropy < max_entropy * 0.1:
                warnings_list.append("Very low entropy - almost certain outcome")
            elif entropy > max_entropy * 0.95:
                warnings_list.append("Very high entropy - very uncertain")

            methodology_results["entropy_check"] = True
        else:
            methodology_results["entropy_check"] = False

        # Validation score
        n_valid = sum(methodology_results.values())
        validation_score = (n_valid / len(methodology_results)) * 100

        is_valid = len(errors) == 0

        # Corrections
        corrections = {}
        if not sum_valid and total > 0:
            # Normalizza
            corrections = {
                k: v / total for k, v in probabilities.items()
            }

        return ValidationResult(
            is_valid=is_valid,
            confidence=validation_score / 100,
            validation_score=validation_score,
            errors=errors,
            warnings=warnings_list,
            corrections=corrections,
            methodology_results=methodology_results,
            timestamp=datetime.now()
        )

    def _check_logical_constraints(
        self,
        probabilities: Dict[str, float],
        market_name: str
    ) -> bool:
        """Verifica vincoli logici specifici per tipo di mercato."""

        if market_name == "1X2":
            # Per 1X2, nessuna prob dovrebbe dominare completamente
            # a meno di casi estremi
            max_prob = max(probabilities.values())
            if max_prob > 0.95:
                return False

        elif market_name == "over_under":
            # Over + Under dovrebbe essere ≈ 1
            if "over" in probabilities and "under" in probabilities:
                total = probabilities["over"] + probabilities["under"]
                if abs(total - 1.0) > 0.01:
                    return False

        return True

    def comprehensive_validation(
        self,
        calculation_data: Dict
    ) -> Dict[str, ValidationResult]:
        """
        Validazione comprensiva di tutti gli aspetti.

        Args:
            calculation_data: Dict con tutti i dati da validare

        Returns:
            Dict {aspect: ValidationResult}
        """
        results = {}

        # Valida probabilità
        if all(k in calculation_data for k in [
            "probability", "lambda_home", "lambda_away", "market_type"
        ]):
            results["probability"] = self.validate_probability_calculation(
                calculation_data["probability"],
                calculation_data["lambda_home"],
                calculation_data["lambda_away"],
                calculation_data.get("calculation_method", "poisson"),
                calculation_data["market_type"]
            )

        # Valida odds
        if "odds" in calculation_data and "probability" in calculation_data:
            results["odds"] = self.validate_odds_calculation(
                calculation_data["odds"],
                calculation_data["probability"],
                calculation_data.get("margin", 0.05)
            )

        # Valida coerenza mercato
        if "market_probabilities" in calculation_data:
            results["market_coherence"] = self.validate_market_coherence(
                calculation_data["market_probabilities"],
                calculation_data.get("market_name", "unknown")
            )

        return results

    def get_overall_validation_score(
        self,
        results: Dict[str, ValidationResult]
    ) -> float:
        """
        Calcola score di validazione complessivo.

        Args:
            results: Dict di ValidationResult

        Returns:
            Score 0-100
        """
        if not results:
            return 0.0

        scores = [r.validation_score for r in results.values()]
        return np.mean(scores)


if __name__ == "__main__":
    # Test del sistema
    engine = RealtimeValidationEngine(strict_mode=False, tolerance=0.001)

    # Test 1: Probability validation
    print("=== TEST 1: Probability Validation ===")
    result_prob = engine.validate_probability_calculation(
        probability=0.55,
        lambda_home=1.8,
        lambda_away=1.2,
        calculation_method="poisson",
        market_type="home_win"
    )
    print(f"Valid: {result_prob.is_valid}")
    print(f"Score: {result_prob.validation_score:.1f}")
    print(f"Confidence: {result_prob.confidence:.2f}")
    print(f"Methodologies: {result_prob.methodology_results}")

    # Test 2: Odds validation
    print("\n=== TEST 2: Odds Validation ===")
    result_odds = engine.validate_odds_calculation(
        odds=1.85,
        probability=0.50,
        margin=0.04
    )
    print(f"Valid: {result_odds.is_valid}")
    print(f"Score: {result_odds.validation_score:.1f}")
    if result_odds.errors:
        print(f"Errors: {result_odds.errors[0]['detail']}")

    # Test 3: Market coherence
    print("\n=== TEST 3: Market Coherence ===")
    market_probs = {
        "home": 0.45,
        "draw": 0.28,
        "away": 0.27
    }
    result_market = engine.validate_market_coherence(market_probs, "1X2")
    print(f"Valid: {result_market.is_valid}")
    print(f"Score: {result_market.validation_score:.1f}")
    if result_market.warnings:
        print(f"Warnings: {result_market.warnings}")

    # Test 4: Comprehensive validation
    print("\n=== TEST 4: Comprehensive Validation ===")
    calc_data = {
        "probability": 0.55,
        "lambda_home": 1.8,
        "lambda_away": 1.2,
        "market_type": "home_win",
        "odds": 1.85,
        "margin": 0.04,
        "market_probabilities": market_probs,
        "market_name": "1X2"
    }

    comprehensive_results = engine.comprehensive_validation(calc_data)
    overall_score = engine.get_overall_validation_score(comprehensive_results)

    print(f"Overall Validation Score: {overall_score:.1f}/100")
    for aspect, result in comprehensive_results.items():
        print(f"\n{aspect}:")
        print(f"  Valid: {result.is_valid}")
        print(f"  Score: {result.validation_score:.1f}")

    print("\n✓ Real-time Validation Engine Test Completed")
