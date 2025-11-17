"""
Advanced Precision Pipeline - Orchestratore dei Nuovi Sistemi IA

Questo modulo integra tutti i nuovi sistemi IA avanzati (Blocchi 7-14)
con la pipeline esistente per migliorare drasticamente precisione e affidabilità.

NUOVI BLOCCHI IMPLEMENTATI:
- Blocco 7: Bayesian Uncertainty Quantification
- Blocco 8: Monte Carlo Simulation Engine
- Blocco 9: Advanced Anomaly Detection
- Blocco 10: Market Consistency Validator
- Blocco 11: Adaptive Calibration System
- Blocco 12: Multi-Model Consensus Validator
- Blocco 13: Statistical Arbitrage Detector
- Blocco 14: Real-time Validation Engine
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import dei nuovi blocchi
try:
    from .blocco_7_bayesian_uncertainty import (
        BayesianUncertaintyQuantifier,
        run_bayesian_analysis
    )
    from .blocco_8_monte_carlo import MonteCarloSimulator
    from .blocco_9_anomaly_detection import AnomalyDetector
    from .blocco_10_market_consistency import MarketConsistencyValidator
    from .blocco_11_adaptive_calibration import AdaptiveCalibrationSystem
    from .blocco_12_consensus_validator import ConsensusValidator, ModelPrediction
    from .blocco_13_arbitrage_detector import StatisticalArbitrageDetector
    from .blocco_14_realtime_validation import RealtimeValidationEngine
except ImportError:
    # Fallback per import diretti
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))

    from blocco_7_bayesian_uncertainty import (
        BayesianUncertaintyQuantifier,
        run_bayesian_analysis
    )
    from blocco_8_monte_carlo import MonteCarloSimulator
    from blocco_9_anomaly_detection import AnomalyDetector
    from blocco_10_market_consistency import MarketConsistencyValidator
    from blocco_11_adaptive_calibration import AdaptiveCalibrationSystem
    from blocco_12_consensus_validator import ConsensusValidator, ModelPrediction
    from blocco_13_arbitrage_detector import StatisticalArbitrageDetector
    from blocco_14_realtime_validation import RealtimeValidationEngine


@dataclass
class AdvancedPredictionResult:
    """
    Risultato avanzato con tutte le analisi supplementari.
    """
    # Predizione base
    original_probability: float
    calibrated_probability: float
    consensus_probability: float
    recommended_probability: float

    # Uncertainty quantification
    credible_interval_95: Tuple[float, float]
    uncertainty_level: str
    reliability_index: float

    # Robustness
    robustness_score: float
    monte_carlo_percentile_5: float
    monte_carlo_percentile_95: float

    # Validation
    validation_passed: bool
    validation_score: float
    consistency_score: float

    # Risk assessment
    anomaly_detected: bool
    anomaly_severity: str
    risk_level: str

    # Consensus
    consensus_reached: bool
    agreement_score: float
    outlier_models: List[str]

    # Opportunità
    arbitrage_opportunities: List[Dict]
    expected_value_pct: float

    # Meta
    confidence_score: float  # 0-100 score complessivo
    recommendation: str  # BET, SKIP, WATCH, INVESTIGATE
    reasoning: List[str]
    timestamp: datetime


class AdvancedPrecisionPipeline:
    """
    Pipeline avanzata che integra tutti i nuovi sistemi IA.

    Questa pipeline wrappa la pipeline esistente aggiungendo:
    - Quantificazione incertezza Bayesiana
    - Simulazioni Monte Carlo
    - Rilevamento anomalie
    - Validazione consistenza mercati
    - Calibrazione adattiva
    - Validazione consenso multi-modello
    - Rilevamento arbitraggi
    - Validazione real-time
    """

    def __init__(
        self,
        strict_mode: bool = False,
        enable_all_systems: bool = True
    ):
        """
        Args:
            strict_mode: Modalità rigorosa (thresholds più stretti)
            enable_all_systems: Se True, abilita tutti i sistemi avanzati
        """
        self.strict_mode = strict_mode
        self.enable_all_systems = enable_all_systems

        # Inizializza tutti i sistemi
        self.bayesian_quantifier = BayesianUncertaintyQuantifier(
            n_samples=10000
        )

        self.monte_carlo_simulator = MonteCarloSimulator(
            n_simulations=10000
        )

        self.anomaly_detector = AnomalyDetector(
            sensitivity="high" if strict_mode else "medium"
        )

        self.consistency_validator = MarketConsistencyValidator(
            tolerance=0.005 if strict_mode else 0.01,
            strict_mode=strict_mode
        )

        self.calibration_system = AdaptiveCalibrationSystem(
            update_frequency=50,
            min_samples=30
        )

        self.consensus_validator = ConsensusValidator(
            consensus_threshold=0.80 if strict_mode else 0.75,
            min_models_required=3
        )

        self.arbitrage_detector = StatisticalArbitrageDetector(
            sure_bet_threshold=0.01,
            value_threshold=0.05
        )

        self.realtime_validator = RealtimeValidationEngine(
            strict_mode=strict_mode,
            tolerance=0.001
        )

        # Statistiche
        self.predictions_processed = 0
        self.anomalies_detected = 0
        self.validations_failed = 0

    def process_prediction(
        self,
        prediction_data: Dict,
        enable_monte_carlo: bool = True,
        enable_anomaly_detection: bool = True,
        enable_consensus: bool = True
    ) -> AdvancedPredictionResult:
        """
        Processa una predizione attraverso tutti i sistemi avanzati.

        Args:
            prediction_data: Dict con dati della predizione
                Required keys:
                - probability: probabilità predetta
                - lambda_home: expected goals casa
                - lambda_away: expected goals trasferta
                - market_type: tipo di mercato
                Optional keys:
                - ensemble_predictions: lista di predizioni da diversi modelli
                - model_confidences: confidence di ciascun modello
                - market_odds: odds del mercato
                - historical_data: dati storici per calibrazione

        Returns:
            AdvancedPredictionResult con analisi completa
        """
        self.predictions_processed += 1

        probability = prediction_data["probability"]
        lambda_home = prediction_data["lambda_home"]
        lambda_away = prediction_data["lambda_away"]
        market_type = prediction_data["market_type"]

        reasoning = []

        # ===== STEP 1: BAYESIAN UNCERTAINTY QUANTIFICATION =====
        ensemble_preds = prediction_data.get("ensemble_predictions", [probability])
        model_reliabilities = prediction_data.get("model_confidences", None)

        bayesian_analysis = run_bayesian_analysis(
            prediction=probability,
            ensemble_predictions=ensemble_preds,
            model_reliabilities=model_reliabilities
        )

        uncertainty_level = bayesian_analysis["uncertainty_level"]
        reliability_index = bayesian_analysis["reliability_index"]
        credible_interval = bayesian_analysis["credible_interval_95"]

        reasoning.append(
            f"Bayesian analysis: {uncertainty_level} uncertainty, "
            f"reliability {reliability_index:.2f}"
        )

        # ===== STEP 2: MONTE CARLO ROBUSTNESS CHECK =====
        robustness_score = 75.0  # Default
        mc_p5, mc_p95 = probability, probability

        if enable_monte_carlo:
            mc_result = self.monte_carlo_simulator.simulate_match_outcome(
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                uncertainty_home=0.10,
                uncertainty_away=0.10
            )

            robustness_score = mc_result.robustness_score
            mc_p5 = mc_result.percentile_5
            mc_p95 = mc_result.percentile_95

            reasoning.append(
                f"Monte Carlo: robustness {robustness_score:.1f}, "
                f"range [{mc_p5:.3f}, {mc_p95:.3f}]"
            )

        # ===== STEP 3: ANOMALY DETECTION =====
        anomaly_detected = False
        anomaly_severity = "LOW"

        if enable_anomaly_detection and "market_odds" in prediction_data:
            market_odds = prediction_data["market_odds"]

            anomaly_result = self.anomaly_detector.detect_market_anomalies(
                predicted_prob=probability,
                market_odds=market_odds,
                historical_margins=prediction_data.get("historical_margins", [])
            )

            anomaly_detected = anomaly_result.is_anomaly
            anomaly_severity = anomaly_result.severity

            if anomaly_detected:
                self.anomalies_detected += 1
                reasoning.append(
                    f"⚠ Anomaly detected: {anomaly_severity} - "
                    f"{anomaly_result.recommendations[0]}"
                )

        # ===== STEP 4: MARKET CONSISTENCY VALIDATION =====
        consistency_score = 100.0

        if "market_probabilities" in prediction_data:
            market_probs = prediction_data["market_probabilities"]
            market_name = prediction_data.get("market_name", "unknown")

            consistency_result = self.consistency_validator.validate_1x2_market(
                **market_probs
            ) if market_name == "1X2" else self.consistency_validator.validate_over_under_consistency(
                **market_probs
            )

            consistency_score = consistency_result.consistency_score

            if not consistency_result.is_consistent:
                reasoning.append(
                    f"⚠ Consistency issues: score {consistency_score:.1f}"
                )

        # ===== STEP 5: ADAPTIVE CALIBRATION =====
        calibrated_prob = probability

        if "league" in prediction_data:
            calibration_result = self.calibration_system.calibrate_probability(
                predicted_prob=probability,
                method="auto",
                league=prediction_data.get("league")
            )

            calibrated_prob = calibration_result.calibrated_probability

            if abs(calibrated_prob - probability) > 0.05:
                reasoning.append(
                    f"Calibration adjusted prob from {probability:.3f} to "
                    f"{calibrated_prob:.3f}"
                )

        # ===== STEP 6: MULTI-MODEL CONSENSUS =====
        consensus_reached = True
        consensus_prob = calibrated_prob
        agreement_score = 100.0
        outlier_models = []

        if enable_consensus and "model_predictions" in prediction_data:
            model_preds = prediction_data["model_predictions"]

            # Converti in ModelPrediction objects
            predictions_list = [
                ModelPrediction(
                    model_name=p["name"],
                    probability=p["probability"],
                    confidence=p.get("confidence", 0.8),
                    features_used=p.get("features", []),
                    model_type=p.get("type", "unknown")
                )
                for p in model_preds
            ]

            consensus_result = self.consensus_validator.check_consensus(
                predictions_list
            )

            consensus_reached = consensus_result.consensus_reached
            consensus_prob = consensus_result.consensus_probability
            agreement_score = consensus_result.agreement_score
            outlier_models = consensus_result.outlier_models

            if not consensus_reached:
                reasoning.append(
                    f"⚠ No consensus: disagreement level "
                    f"{consensus_result.disagreement_level}"
                )

        # ===== STEP 7: ARBITRAGE DETECTION =====
        arbitrage_opportunities = []
        ev_pct = 0.0

        if "market_odds" in prediction_data:
            market_odds = prediction_data["market_odds"]

            arb_opp = self.arbitrage_detector.detect_statistical_arbitrage(
                model_probability=consensus_prob,
                market_odds=market_odds,
                model_confidence=reliability_index
            )

            if arb_opp:
                arbitrage_opportunities.append({
                    "type": arb_opp.opportunity_type,
                    "ev_pct": arb_opp.expected_value_pct,
                    "risk": arb_opp.risk_level
                })
                ev_pct = arb_opp.expected_value_pct

                reasoning.append(
                    f"💰 {arb_opp.opportunity_type}: EV {ev_pct:.2f}%"
                )

        # ===== STEP 8: REAL-TIME VALIDATION =====
        validation_passed = True
        validation_score = 100.0

        validation_data = {
            "probability": consensus_prob,
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "market_type": market_type,
            "calculation_method": "poisson"
        }

        validation_result = self.realtime_validator.validate_probability_calculation(
            **validation_data
        )

        validation_passed = validation_result.is_valid
        validation_score = validation_result.validation_score

        if not validation_passed:
            self.validations_failed += 1
            reasoning.append(
                f"⚠ Validation failed: score {validation_score:.1f}"
            )

        # ===== FINAL DECISION LOGIC =====
        # Calcola confidence score complessivo
        confidence_components = [
            reliability_index * 100,
            robustness_score,
            consistency_score,
            validation_score,
            agreement_score
        ]

        # Penalità per anomalie
        if anomaly_detected and anomaly_severity in ["HIGH", "CRITICAL"]:
            confidence_components.append(0.0)

        confidence_score = np.mean(confidence_components)

        # Determine risk level
        if anomaly_detected and anomaly_severity == "CRITICAL":
            risk_level = "CRITICAL"
        elif not validation_passed or not consensus_reached:
            risk_level = "HIGH"
        elif uncertainty_level in ["HIGH", "VERY_HIGH"]:
            risk_level = "MEDIUM"
        elif robustness_score < 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Final recommendation
        if confidence_score >= 80 and risk_level == "LOW" and ev_pct > 5:
            recommendation = "BET"
            reasoning.append("✓ High confidence, low risk, positive EV → BET")
        elif confidence_score >= 70 and risk_level in ["LOW", "MEDIUM"] and ev_pct > 3:
            recommendation = "BET"
            reasoning.append("✓ Good confidence and EV → BET (conservative stake)")
        elif confidence_score >= 60 and not anomaly_detected:
            recommendation = "WATCH"
            reasoning.append("⚡ Moderate confidence → WATCH for better opportunities")
        elif anomaly_detected or not validation_passed:
            recommendation = "SKIP"
            reasoning.append("✗ Issues detected → SKIP")
        else:
            recommendation = "INVESTIGATE"
            reasoning.append("⚠ Needs further analysis → INVESTIGATE")

        # Recommended probability (most conservative)
        recommended_prob = min(
            consensus_prob,
            calibrated_prob,
            bayesian_analysis["conservative_probability"]
        )

        return AdvancedPredictionResult(
            original_probability=probability,
            calibrated_probability=calibrated_prob,
            consensus_probability=consensus_prob,
            recommended_probability=recommended_prob,
            credible_interval_95=credible_interval,
            uncertainty_level=uncertainty_level,
            reliability_index=reliability_index,
            robustness_score=robustness_score,
            monte_carlo_percentile_5=mc_p5,
            monte_carlo_percentile_95=mc_p95,
            validation_passed=validation_passed,
            validation_score=validation_score,
            consistency_score=consistency_score,
            anomaly_detected=anomaly_detected,
            anomaly_severity=anomaly_severity,
            risk_level=risk_level,
            consensus_reached=consensus_reached,
            agreement_score=agreement_score,
            outlier_models=outlier_models,
            arbitrage_opportunities=arbitrage_opportunities,
            expected_value_pct=ev_pct,
            confidence_score=confidence_score,
            recommendation=recommendation,
            reasoning=reasoning,
            timestamp=datetime.now()
        )

    def get_pipeline_statistics(self) -> Dict:
        """
        Ritorna statistiche della pipeline.

        Returns:
            Dict con statistiche
        """
        return {
            "predictions_processed": self.predictions_processed,
            "anomalies_detected": self.anomalies_detected,
            "validations_failed": self.validations_failed,
            "anomaly_rate": (
                self.anomalies_detected / self.predictions_processed
                if self.predictions_processed > 0 else 0
            ),
            "validation_failure_rate": (
                self.validations_failed / self.predictions_processed
                if self.predictions_processed > 0 else 0
            ),
            "calibration_status": self.calibration_system.get_calibration_status()
        }


if __name__ == "__main__":
    # Test della pipeline avanzata
    print("=== TEST: Advanced Precision Pipeline ===\n")

    pipeline = AdvancedPrecisionPipeline(strict_mode=False)

    # Test con dati simulati
    prediction_data = {
        "probability": 0.65,
        "lambda_home": 1.8,
        "lambda_away": 1.2,
        "market_type": "home_win",
        "ensemble_predictions": [0.63, 0.65, 0.67, 0.64],
        "model_confidences": [0.85, 0.90, 0.82, 0.88],
        "market_odds": 1.75,
        "league": "Premier League",
        "model_predictions": [
            {
                "name": "Dixon-Coles",
                "probability": 0.63,
                "confidence": 0.85,
                "type": "statistical"
            },
            {
                "name": "XGBoost",
                "probability": 0.67,
                "confidence": 0.90,
                "type": "ml"
            },
            {
                "name": "LSTM",
                "probability": 0.64,
                "confidence": 0.82,
                "type": "dl"
            }
        ]
    }

    result = pipeline.process_prediction(prediction_data)

    print("=" * 60)
    print("ADVANCED PREDICTION ANALYSIS REPORT")
    print("=" * 60)
    print(f"\n📊 PROBABILITIES:")
    print(f"  Original:     {result.original_probability:.4f}")
    print(f"  Calibrated:   {result.calibrated_probability:.4f}")
    print(f"  Consensus:    {result.consensus_probability:.4f}")
    print(f"  Recommended:  {result.recommended_probability:.4f}")
    print(f"  95% CI:       [{result.credible_interval_95[0]:.4f}, {result.credible_interval_95[1]:.4f}]")

    print(f"\n🎯 CONFIDENCE & RELIABILITY:")
    print(f"  Confidence Score:   {result.confidence_score:.1f}/100")
    print(f"  Reliability Index:  {result.reliability_index:.2f}")
    print(f"  Robustness Score:   {result.robustness_score:.1f}/100")
    print(f"  Uncertainty Level:  {result.uncertainty_level}")

    print(f"\n✅ VALIDATION:")
    print(f"  Validation Passed:  {result.validation_passed}")
    print(f"  Validation Score:   {result.validation_score:.1f}/100")
    print(f"  Consistency Score:  {result.consistency_score:.1f}/100")

    print(f"\n🚨 RISK ASSESSMENT:")
    print(f"  Risk Level:         {result.risk_level}")
    print(f"  Anomaly Detected:   {result.anomaly_detected}")
    if result.anomaly_detected:
        print(f"  Anomaly Severity:   {result.anomaly_severity}")

    print(f"\n🤝 CONSENSUS:")
    print(f"  Consensus Reached:  {result.consensus_reached}")
    print(f"  Agreement Score:    {result.agreement_score:.1f}/100")
    if result.outlier_models:
        print(f"  Outlier Models:     {', '.join(result.outlier_models)}")

    print(f"\n💰 OPPORTUNITIES:")
    print(f"  Expected Value:     {result.expected_value_pct:.2f}%")
    if result.arbitrage_opportunities:
        for opp in result.arbitrage_opportunities:
            print(f"  {opp['type']}: EV {opp['ev_pct']:.2f}% (Risk: {opp['risk']})")

    print(f"\n🎯 FINAL RECOMMENDATION: {result.recommendation}")
    print(f"\n📝 REASONING:")
    for i, reason in enumerate(result.reasoning, 1):
        print(f"  {i}. {reason}")

    print(f"\n" + "=" * 60)

    # Statistics
    stats = pipeline.get_pipeline_statistics()
    print(f"\n📈 PIPELINE STATISTICS:")
    print(f"  Predictions Processed: {stats['predictions_processed']}")
    print(f"  Anomalies Detected:    {stats['anomalies_detected']}")
    print(f"  Validations Failed:    {stats['validations_failed']}")

    print("\n✓ Advanced Precision Pipeline Test Completed!")
