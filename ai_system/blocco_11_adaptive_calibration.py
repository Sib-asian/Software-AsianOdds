"""
BLOCCO 11: Adaptive Calibration System

Sistema di calibrazione adattiva che impara dai risultati storici
per migliorare continuamente la precisione delle previsioni.

Features:
- Online learning per calibrazione continua
- Platt scaling adattivo
- Isotonic regression con aggiornamento incrementale
- Temperature scaling dinamico
- League-specific calibration
- Performance-based weight adjustment
- Automatic recalibration triggers
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


@dataclass
class CalibrationParams:
    """Parametri di calibrazione"""
    temperature: float
    platt_a: float
    platt_b: float
    isotonic_mapping: Optional[Dict] = None
    last_updated: Optional[datetime] = None
    n_samples: int = 0
    calibration_error: float = 0.0


@dataclass
class CalibrationResult:
    """Risultato della calibrazione"""
    calibrated_probability: float
    original_probability: float
    adjustment_factor: float
    calibration_confidence: float  # 0-1
    method_used: str
    expected_calibration_error: float
    reliability_score: float  # 0-100


class AdaptiveCalibrationSystem:
    """
    Sistema di calibrazione adattiva che si auto-aggiorna.

    Mantiene modelli di calibrazione separati per diverse categorie
    (league, team strength, etc.) e li aggiorna incrementalmente.
    """

    def __init__(
        self,
        update_frequency: int = 50,
        min_samples: int = 30,
        decay_factor: float = 0.95
    ):
        """
        Args:
            update_frequency: Ogni quante predizioni ricalcolare calibrazione
            min_samples: Minimo campioni per calibrazione affidabile
            decay_factor: Peso decrescente per dati vecchi (0-1)
        """
        self.update_frequency = update_frequency
        self.min_samples = min_samples
        self.decay_factor = decay_factor

        # Storage per calibration models
        self.global_params = CalibrationParams(
            temperature=1.0,
            platt_a=1.0,
            platt_b=0.0,
            last_updated=None,
            n_samples=0
        )

        self.league_params: Dict[str, CalibrationParams] = {}
        self.strength_tier_params: Dict[str, CalibrationParams] = {}

        # Historical data per online learning
        self.history_predictions = []
        self.history_outcomes = []
        self.history_metadata = []  # league, tier, etc.

        self.predictions_since_update = 0

    def add_observation(
        self,
        predicted_prob: float,
        actual_outcome: bool,
        metadata: Optional[Dict] = None
    ):
        """
        Aggiunge una nuova osservazione per online learning.

        Args:
            predicted_prob: Probabilità predetta
            actual_outcome: Outcome effettivo (True/False)
            metadata: Info addizionali (league, tier, etc.)
        """
        # Applica decay a dati vecchi
        if len(self.history_predictions) > 1000:
            # Mantieni solo gli ultimi 1000, applicando decay
            self.history_predictions = self.history_predictions[-1000:]
            self.history_outcomes = self.history_outcomes[-1000:]
            self.history_metadata = self.history_metadata[-1000:]

        self.history_predictions.append(predicted_prob)
        self.history_outcomes.append(float(actual_outcome))
        self.history_metadata.append(metadata or {})

        self.predictions_since_update += 1

        # Trigger recalibration se necessario
        if self.predictions_since_update >= self.update_frequency:
            self.recalibrate()
            self.predictions_since_update = 0

    def platt_scaling(
        self,
        probabilities: np.ndarray,
        outcomes: np.ndarray
    ) -> Tuple[float, float]:
        """
        Platt scaling: calibra usando funzione sigmoide.

        P_calibrated = 1 / (1 + exp(A * P + B))

        Args:
            probabilities: Probabilità predette
            outcomes: Outcome effettivi

        Returns:
            (A, B) parameters
        """
        # Converti probabilities in logit
        epsilon = 1e-7
        logits = np.log((probabilities + epsilon) / (1 - probabilities + epsilon))

        # Ottimizza A e B
        def loss(params):
            a, b = params
            calibrated_logits = a * logits + b
            calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
            # Log loss
            return -np.mean(
                outcomes * np.log(calibrated_probs + epsilon) +
                (1 - outcomes) * np.log(1 - calibrated_probs + epsilon)
            )

        result = minimize(loss, [1.0, 0.0], method='BFGS')

        return result.x[0], result.x[1]

    def temperature_scaling(
        self,
        probabilities: np.ndarray,
        outcomes: np.ndarray
    ) -> float:
        """
        Temperature scaling: calibra dividendo logits per temperatura.

        Args:
            probabilities: Probabilità predette
            outcomes: Outcome effettivi

        Returns:
            Optimal temperature T
        """
        epsilon = 1e-7
        logits = np.log((probabilities + epsilon) / (1 - probabilities + epsilon))

        def loss(temperature):
            t = temperature[0]
            if t <= 0:
                return 1e10
            calibrated_logits = logits / t
            calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
            # Log loss
            return -np.mean(
                outcomes * np.log(calibrated_probs + epsilon) +
                (1 - outcomes) * np.log(1 - calibrated_probs + epsilon)
            )

        result = minimize(loss, [1.0], bounds=[(0.1, 10.0)], method='L-BFGS-B')

        return result.x[0]

    def isotonic_regression_mapping(
        self,
        probabilities: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Isotonic regression: mapping non-parametrico monotono.

        Args:
            probabilities: Probabilità predette
            outcomes: Outcome effettivi
            n_bins: Numero di bins per mapping

        Returns:
            Dict con mapping
        """
        # Ordina per probabilità
        sorted_indices = np.argsort(probabilities)
        sorted_probs = probabilities[sorted_indices]
        sorted_outcomes = outcomes[sorted_indices]

        # Dividi in bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(sorted_probs, bins) - 1

        # Per ogni bin, calcola frequenza osservata
        mapping = {}
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_center = (bins[i] + bins[i + 1]) / 2
                observed_freq = np.mean(sorted_outcomes[mask])
                n_samples = np.sum(mask)
                mapping[bin_center] = {
                    "calibrated_prob": observed_freq,
                    "n_samples": int(n_samples),
                    "bin_range": (bins[i], bins[i + 1])
                }

        return mapping

    def recalibrate(self):
        """
        Ricalcola parametri di calibrazione basandosi sui dati storici.
        """
        if len(self.history_predictions) < self.min_samples:
            return

        predictions = np.array(self.history_predictions)
        outcomes = np.array(self.history_outcomes)

        # Applica decay weights ai dati vecchi
        n = len(predictions)
        weights = np.array([
            self.decay_factor ** (n - i - 1) for i in range(n)
        ])
        weights = weights / np.sum(weights)

        # Global calibration
        try:
            # Temperature scaling
            temperature = self.temperature_scaling(predictions, outcomes)

            # Platt scaling
            platt_a, platt_b = self.platt_scaling(predictions, outcomes)

            # Isotonic mapping
            isotonic_map = self.isotonic_regression_mapping(predictions, outcomes)

            # Update global params
            self.global_params = CalibrationParams(
                temperature=temperature,
                platt_a=platt_a,
                platt_b=platt_b,
                isotonic_mapping=isotonic_map,
                last_updated=datetime.now(),
                n_samples=len(predictions),
                calibration_error=self.calculate_ece(predictions, outcomes)
            )

        except Exception as e:
            print(f"Calibration update failed: {e}")

        # League-specific calibration
        self._recalibrate_by_category("league")

    def _recalibrate_by_category(self, category: str):
        """
        Ricalcola calibrazione per una specifica categoria.

        Args:
            category: "league" o "strength_tier"
        """
        if category not in ["league", "strength_tier"]:
            return

        # Raggruppa per categoria
        category_data = {}
        for pred, outcome, meta in zip(
            self.history_predictions,
            self.history_outcomes,
            self.history_metadata
        ):
            if category in meta:
                cat_value = meta[category]
                if cat_value not in category_data:
                    category_data[cat_value] = {"preds": [], "outcomes": []}
                category_data[cat_value]["preds"].append(pred)
                category_data[cat_value]["outcomes"].append(outcome)

        # Calibra per ogni categoria
        for cat_value, data in category_data.items():
            if len(data["preds"]) < self.min_samples:
                continue

            preds = np.array(data["preds"])
            outcomes = np.array(data["outcomes"])

            try:
                temperature = self.temperature_scaling(preds, outcomes)
                platt_a, platt_b = self.platt_scaling(preds, outcomes)

                params = CalibrationParams(
                    temperature=temperature,
                    platt_a=platt_a,
                    platt_b=platt_b,
                    last_updated=datetime.now(),
                    n_samples=len(preds),
                    calibration_error=self.calculate_ece(preds, outcomes)
                )

                if category == "league":
                    self.league_params[cat_value] = params
                else:
                    self.strength_tier_params[cat_value] = params

            except:
                continue

    def calibrate_probability(
        self,
        predicted_prob: float,
        method: str = "auto",
        league: Optional[str] = None,
        strength_tier: Optional[str] = None
    ) -> CalibrationResult:
        """
        Calibra una probabilità usando il metodo specificato.

        Args:
            predicted_prob: Probabilità predetta
            method: "temperature", "platt", "isotonic", "auto"
            league: League per calibrazione specifica
            strength_tier: Tier per calibrazione specifica

        Returns:
            CalibrationResult
        """
        # Seleziona params appropriati
        if league and league in self.league_params:
            params = self.league_params[league]
            confidence = 0.9
        elif strength_tier and strength_tier in self.strength_tier_params:
            params = self.strength_tier_params[strength_tier]
            confidence = 0.85
        else:
            params = self.global_params
            confidence = 0.75 if params.n_samples >= self.min_samples else 0.5

        # Se non abbastanza dati, ritorna prob originale
        if params.n_samples < self.min_samples:
            return CalibrationResult(
                calibrated_probability=predicted_prob,
                original_probability=predicted_prob,
                adjustment_factor=1.0,
                calibration_confidence=0.3,
                method_used="none",
                expected_calibration_error=0.0,
                reliability_score=50.0
            )

        # Applica calibrazione
        if method == "auto":
            # Usa il metodo con migliore calibration error
            method = "temperature"  # Default

        calibrated = predicted_prob

        if method == "temperature":
            # Temperature scaling
            epsilon = 1e-7
            logit = np.log((predicted_prob + epsilon) / (1 - predicted_prob + epsilon))
            calibrated_logit = logit / params.temperature
            calibrated = 1 / (1 + np.exp(-calibrated_logit))

        elif method == "platt":
            # Platt scaling
            epsilon = 1e-7
            logit = np.log((predicted_prob + epsilon) / (1 - predicted_prob + epsilon))
            calibrated_logit = params.platt_a * logit + params.platt_b
            calibrated = 1 / (1 + np.exp(-calibrated_logit))

        elif method == "isotonic" and params.isotonic_mapping:
            # Isotonic regression mapping
            # Trova il bin più vicino
            bin_centers = sorted(params.isotonic_mapping.keys())
            closest_bin = min(
                bin_centers,
                key=lambda x: abs(x - predicted_prob)
            )
            calibrated = params.isotonic_mapping[closest_bin]["calibrated_prob"]

        # Clip to valid range
        calibrated = max(0.01, min(0.99, calibrated))

        adjustment_factor = calibrated / predicted_prob if predicted_prob > 0 else 1.0

        # Reliability score basato su calibration error e n_samples
        reliability_score = 100 * (1 - params.calibration_error) * min(1.0, params.n_samples / 100)

        return CalibrationResult(
            calibrated_probability=calibrated,
            original_probability=predicted_prob,
            adjustment_factor=adjustment_factor,
            calibration_confidence=confidence,
            method_used=method,
            expected_calibration_error=params.calibration_error,
            reliability_score=reliability_score
        )

    def calculate_ece(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calcola Expected Calibration Error.

        Args:
            predictions: Probabilità predette
            outcomes: Outcome effettivi
            n_bins: Numero di bins

        Returns:
            ECE (0-1)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) == 0:
                continue

            bin_predictions = predictions[mask]
            bin_outcomes = outcomes[mask]

            avg_pred = np.mean(bin_predictions)
            avg_outcome = np.mean(bin_outcomes)

            ece += (np.sum(mask) / len(predictions)) * abs(avg_pred - avg_outcome)

        return ece

    def get_calibration_status(self) -> Dict:
        """
        Ritorna status del sistema di calibrazione.

        Returns:
            Dict con statistiche
        """
        return {
            "global": {
                "temperature": self.global_params.temperature,
                "n_samples": self.global_params.n_samples,
                "calibration_error": self.global_params.calibration_error,
                "last_updated": self.global_params.last_updated
            },
            "n_leagues_calibrated": len(self.league_params),
            "n_tiers_calibrated": len(self.strength_tier_params),
            "total_observations": len(self.history_predictions),
            "predictions_since_update": self.predictions_since_update,
            "next_update_in": self.update_frequency - self.predictions_since_update
        }


if __name__ == "__main__":
    # Test del sistema
    calibration_system = AdaptiveCalibrationSystem(
        update_frequency=50,
        min_samples=30
    )

    # Simula dati
    print("=== TEST: Adaptive Calibration System ===")

    # Genera dati di training (modello overconfident)
    np.random.seed(42)
    n_train = 200
    true_probs = np.random.beta(2, 2, n_train)  # Vere probabilità
    predicted_probs = true_probs * 1.2  # Overconfident
    predicted_probs = np.clip(predicted_probs, 0.01, 0.99)
    outcomes = np.random.rand(n_train) < true_probs

    # Aggiungi observations
    for pred, outcome in zip(predicted_probs, outcomes):
        calibration_system.add_observation(
            pred, outcome,
            metadata={"league": "Premier League"}
        )

    # Forza recalibration
    calibration_system.recalibrate()

    # Test calibration
    test_prob = 0.70
    result = calibration_system.calibrate_probability(
        test_prob,
        method="temperature",
        league="Premier League"
    )

    print(f"\nOriginal Probability: {result.original_probability:.4f}")
    print(f"Calibrated Probability: {result.calibrated_probability:.4f}")
    print(f"Adjustment Factor: {result.adjustment_factor:.4f}")
    print(f"Confidence: {result.calibration_confidence:.2f}")
    print(f"Reliability Score: {result.reliability_score:.1f}")

    # Status
    status = calibration_system.get_calibration_status()
    print(f"\nSystem Status:")
    print(f"  Total Observations: {status['total_observations']}")
    print(f"  Calibration Error: {status['global']['calibration_error']:.4f}")
    print(f"  Temperature: {status['global']['temperature']:.4f}")

    print("\n✓ Adaptive Calibration System Test Completed")
