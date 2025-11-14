"""
BLOCCO 12: Multi-Model Consensus Validator

Sistema che richiede consenso tra diversi modelli indipendenti
prima di confermare una previsione come affidabile.

Features:
- Voting mechanism tra modelli diversi
- Consensus threshold configurabile
- Disagreement analysis
- Confidence weighting
- Outlier model detection
- Ensemble uncertainty quantification
- Majority voting con soft/hard thresholds
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ModelPrediction:
    """Predizione da un singolo modello"""
    model_name: str
    probability: float
    confidence: float  # 0-1
    features_used: List[str]
    model_type: str  # "statistical", "ml", "dl", "ensemble"


@dataclass
class ConsensusResult:
    """Risultato dell'analisi di consenso"""
    consensus_reached: bool
    consensus_probability: float
    consensus_confidence: float  # 0-1
    agreement_score: float  # 0-100
    disagreement_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    voting_results: Dict[str, float]
    outlier_models: List[str]
    reliability_score: float  # 0-100
    recommendation: str  # BET, SKIP, WATCH, INVESTIGATE
    reasoning: List[str]


class ConsensusValidator:
    """
    Validatore che richiede consenso tra modelli multipli.

    Analizza le predizioni di diversi modelli e determina se c'è
    sufficiente accordo per considerare la previsione affidabile.
    """

    def __init__(
        self,
        consensus_threshold: float = 0.75,
        min_models_required: int = 3,
        disagreement_tolerance: float = 0.10,
        outlier_threshold: float = 2.5
    ):
        """
        Args:
            consensus_threshold: Frazione minima di modelli che devono essere d'accordo
            min_models_required: Numero minimo di modelli necessari
            disagreement_tolerance: Max differenza accettabile tra predizioni
            outlier_threshold: Z-score per identificare modelli outlier
        """
        self.consensus_threshold = consensus_threshold
        self.min_models_required = min_models_required
        self.disagreement_tolerance = disagreement_tolerance
        self.outlier_threshold = outlier_threshold

    def check_consensus(
        self,
        predictions: List[ModelPrediction],
        use_confidence_weighting: bool = True
    ) -> ConsensusResult:
        """
        Verifica se c'è consenso tra le predizioni dei modelli.

        Args:
            predictions: Lista di predizioni dai diversi modelli
            use_confidence_weighting: Se pesare per confidence

        Returns:
            ConsensusResult
        """
        if len(predictions) < self.min_models_required:
            return ConsensusResult(
                consensus_reached=False,
                consensus_probability=0.0,
                consensus_confidence=0.0,
                agreement_score=0.0,
                disagreement_level="CRITICAL",
                voting_results={},
                outlier_models=[],
                reliability_score=0.0,
                recommendation="SKIP",
                reasoning=[
                    f"Insufficient models: {len(predictions)} < {self.min_models_required}"
                ]
            )

        # Estrai probabilità e confidence
        probs = np.array([p.probability for p in predictions])
        confidences = np.array([p.confidence for p in predictions])
        model_names = [p.model_name for p in predictions]

        # Identifica outliers
        outlier_models = self._detect_outlier_models(probs, model_names)

        # Rimuovi outliers per calcolo consenso
        non_outlier_mask = np.array([
            name not in outlier_models for name in model_names
        ])
        probs_filtered = probs[non_outlier_mask]
        confidences_filtered = confidences[non_outlier_mask]

        if len(probs_filtered) < 2:
            # Troppi outliers
            return ConsensusResult(
                consensus_reached=False,
                consensus_probability=np.mean(probs),
                consensus_confidence=0.2,
                agreement_score=0.0,
                disagreement_level="CRITICAL",
                voting_results={},
                outlier_models=outlier_models,
                reliability_score=0.0,
                recommendation="SKIP",
                reasoning=["Too many outlier models detected"]
            )

        # Calcola consenso
        if use_confidence_weighting:
            # Weighted average
            weights = confidences_filtered / np.sum(confidences_filtered)
            consensus_prob = np.sum(probs_filtered * weights)
            consensus_conf = np.mean(confidences_filtered)
        else:
            # Simple average
            consensus_prob = np.mean(probs_filtered)
            consensus_conf = np.mean(confidences_filtered)

        # Calcola agreement score
        std_dev = np.std(probs_filtered)
        cv = std_dev / consensus_prob if consensus_prob > 0 else 1.0
        agreement_score = max(0, 100 * (1 - cv * 2))

        # Calcola disagreement level
        max_diff = np.max(probs_filtered) - np.min(probs_filtered)
        if max_diff <= self.disagreement_tolerance:
            disagreement_level = "LOW"
        elif max_diff <= self.disagreement_tolerance * 2:
            disagreement_level = "MEDIUM"
        elif max_diff <= self.disagreement_tolerance * 3:
            disagreement_level = "HIGH"
        else:
            disagreement_level = "CRITICAL"

        # Consensus raggiunto?
        # Verifica quanti modelli sono vicini al consenso
        close_to_consensus = np.sum(
            np.abs(probs_filtered - consensus_prob) <= self.disagreement_tolerance
        )
        consensus_fraction = close_to_consensus / len(probs_filtered)

        consensus_reached = (
            consensus_fraction >= self.consensus_threshold and
            disagreement_level in ["LOW", "MEDIUM"]
        )

        # Voting results
        voting_results = {
            name: float(prob)
            for name, prob in zip(model_names, probs)
        }

        # Reliability score
        reliability_score = self._calculate_reliability(
            agreement_score,
            consensus_conf,
            len(probs_filtered),
            outlier_models
        )

        # Recommendation
        recommendation, reasoning = self._generate_recommendation(
            consensus_reached,
            disagreement_level,
            reliability_score,
            outlier_models
        )

        return ConsensusResult(
            consensus_reached=consensus_reached,
            consensus_probability=consensus_prob,
            consensus_confidence=consensus_conf,
            agreement_score=agreement_score,
            disagreement_level=disagreement_level,
            voting_results=voting_results,
            outlier_models=outlier_models,
            reliability_score=reliability_score,
            recommendation=recommendation,
            reasoning=reasoning
        )

    def _detect_outlier_models(
        self,
        probabilities: np.ndarray,
        model_names: List[str]
    ) -> List[str]:
        """
        Identifica modelli outlier usando Z-score.

        Args:
            probabilities: Array di probabilità
            model_names: Nomi dei modelli

        Returns:
            Lista di nomi di modelli outlier
        """
        if len(probabilities) < 3:
            return []

        mean = np.mean(probabilities)
        std = np.std(probabilities)

        if std < 1e-6:
            return []

        z_scores = np.abs((probabilities - mean) / std)

        outliers = [
            model_names[i]
            for i in range(len(model_names))
            if z_scores[i] > self.outlier_threshold
        ]

        return outliers

    def _calculate_reliability(
        self,
        agreement_score: float,
        consensus_confidence: float,
        n_models: int,
        outlier_models: List[str]
    ) -> float:
        """
        Calcola reliability score complessivo.

        Args:
            agreement_score: Score di accordo tra modelli
            consensus_confidence: Confidence media
            n_models: Numero di modelli (non-outlier)
            outlier_models: Lista outlier

        Returns:
            Reliability score 0-100
        """
        # Base score da agreement
        base_score = agreement_score

        # Bonus per alta confidence
        confidence_bonus = consensus_confidence * 10

        # Bonus per numero di modelli
        models_bonus = min(20, n_models * 3)

        # Penalità per outliers
        outliers_penalty = len(outlier_models) * 10

        reliability = base_score + confidence_bonus + models_bonus - outliers_penalty

        return max(0, min(100, reliability))

    def _generate_recommendation(
        self,
        consensus_reached: bool,
        disagreement_level: str,
        reliability_score: float,
        outlier_models: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Genera raccomandazione basata sull'analisi.

        Args:
            consensus_reached: Se consenso raggiunto
            disagreement_level: Livello di disaccordo
            reliability_score: Score di affidabilità
            outlier_models: Modelli outlier

        Returns:
            (recommendation, reasoning)
        """
        reasoning = []

        if not consensus_reached:
            recommendation = "SKIP"
            reasoning.append("No consensus reached among models")

            if disagreement_level == "CRITICAL":
                reasoning.append("CRITICAL disagreement - models strongly disagree")
                reasoning.append("Possible data issues or extreme uncertainty")

            if len(outlier_models) > 0:
                reasoning.append(f"Outlier models detected: {', '.join(outlier_models)}")

            return recommendation, reasoning

        # Consensus raggiunto
        if reliability_score >= 80 and disagreement_level == "LOW":
            recommendation = "BET"
            reasoning.append("Strong consensus with high reliability")
            reasoning.append(f"Reliability score: {reliability_score:.1f}/100")

        elif reliability_score >= 65 and disagreement_level in ["LOW", "MEDIUM"]:
            recommendation = "BET"
            reasoning.append("Good consensus reached")
            reasoning.append(f"Reliability score: {reliability_score:.1f}/100")

            if disagreement_level == "MEDIUM":
                reasoning.append("Some disagreement present - use conservative stake")

        elif reliability_score >= 50:
            recommendation = "WATCH"
            reasoning.append("Moderate consensus")
            reasoning.append("Consider for betting but with caution")
            reasoning.append(f"Reliability score: {reliability_score:.1f}/100")

        else:
            recommendation = "INVESTIGATE"
            reasoning.append("Weak consensus")
            reasoning.append("Further analysis recommended")
            reasoning.append(f"Low reliability score: {reliability_score:.1f}/100")

        if len(outlier_models) > 0:
            reasoning.append(f"Note: {len(outlier_models)} model(s) disagreed significantly")

        return recommendation, reasoning

    def analyze_disagreement(
        self,
        predictions: List[ModelPrediction]
    ) -> Dict:
        """
        Analisi approfondita del disaccordo tra modelli.

        Args:
            predictions: Lista di predizioni

        Returns:
            Dict con dettagli del disaccordo
        """
        probs = np.array([p.probability for p in predictions])
        model_names = [p.model_name for p in predictions]
        model_types = [p.model_type for p in predictions]

        # Statistiche base
        mean_prob = np.mean(probs)
        std_prob = np.std(probs)
        min_prob = np.min(probs)
        max_prob = np.max(probs)
        range_prob = max_prob - min_prob

        # Disaccordo per tipo di modello
        type_disagreement = {}
        for mtype in set(model_types):
            mask = np.array([t == mtype for t in model_types])
            if np.sum(mask) > 0:
                type_probs = probs[mask]
                type_disagreement[mtype] = {
                    "mean": float(np.mean(type_probs)),
                    "std": float(np.std(type_probs)),
                    "n_models": int(np.sum(mask))
                }

        # Pairwise disagreement matrix
        n = len(predictions)
        disagreement_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                disagreement_matrix[i, j] = abs(probs[i] - probs[j])
                disagreement_matrix[j, i] = disagreement_matrix[i, j]

        # Trova i modelli più in disaccordo
        avg_disagreement_per_model = np.mean(disagreement_matrix, axis=1)
        most_disagreeing_idx = np.argmax(avg_disagreement_per_model)

        return {
            "overall_statistics": {
                "mean": float(mean_prob),
                "std": float(std_prob),
                "min": float(min_prob),
                "max": float(max_prob),
                "range": float(range_prob),
                "cv": float(std_prob / mean_prob) if mean_prob > 0 else 0
            },
            "by_model_type": type_disagreement,
            "most_disagreeing_model": {
                "name": model_names[most_disagreeing_idx],
                "probability": float(probs[most_disagreeing_idx]),
                "avg_disagreement": float(avg_disagreement_per_model[most_disagreeing_idx])
            },
            "pairwise_max_disagreement": float(np.max(disagreement_matrix))
        }

    def get_consensus_strength(
        self,
        consensus_result: ConsensusResult
    ) -> str:
        """
        Classifica la forza del consenso.

        Args:
            consensus_result: Risultato del consensus check

        Returns:
            "VERY_STRONG", "STRONG", "MODERATE", "WEAK", "NONE"
        """
        if not consensus_result.consensus_reached:
            return "NONE"

        score = consensus_result.reliability_score

        if score >= 85:
            return "VERY_STRONG"
        elif score >= 70:
            return "STRONG"
        elif score >= 55:
            return "MODERATE"
        else:
            return "WEAK"


if __name__ == "__main__":
    # Test del sistema
    validator = ConsensusValidator(
        consensus_threshold=0.75,
        min_models_required=3,
        disagreement_tolerance=0.10
    )

    # Test 1: Buon consenso
    print("=== TEST 1: Good Consensus ===")
    predictions_good = [
        ModelPrediction("Dixon-Coles", 0.65, 0.85, ["lambda", "rho"], "statistical"),
        ModelPrediction("XGBoost", 0.67, 0.90, ["form", "h2h"], "ml"),
        ModelPrediction("LSTM", 0.63, 0.80, ["sequences"], "dl"),
        ModelPrediction("Ensemble", 0.66, 0.88, ["all"], "ensemble")
    ]

    result_good = validator.check_consensus(predictions_good)
    print(f"Consensus Reached: {result_good.consensus_reached}")
    print(f"Consensus Probability: {result_good.consensus_probability:.4f}")
    print(f"Agreement Score: {result_good.agreement_score:.1f}")
    print(f"Disagreement Level: {result_good.disagreement_level}")
    print(f"Reliability: {result_good.reliability_score:.1f}")
    print(f"Recommendation: {result_good.recommendation}")
    print(f"Reasoning: {result_good.reasoning[0]}")

    # Test 2: Alto disaccordo
    print("\n=== TEST 2: High Disagreement ===")
    predictions_bad = [
        ModelPrediction("Dixon-Coles", 0.45, 0.85, ["lambda", "rho"], "statistical"),
        ModelPrediction("XGBoost", 0.75, 0.90, ["form", "h2h"], "ml"),
        ModelPrediction("LSTM", 0.55, 0.80, ["sequences"], "dl"),
        ModelPrediction("Ensemble", 0.30, 0.70, ["all"], "ensemble")
    ]

    result_bad = validator.check_consensus(predictions_bad)
    print(f"Consensus Reached: {result_bad.consensus_reached}")
    print(f"Agreement Score: {result_bad.agreement_score:.1f}")
    print(f"Disagreement Level: {result_bad.disagreement_level}")
    print(f"Outlier Models: {result_bad.outlier_models}")
    print(f"Recommendation: {result_bad.recommendation}")

    # Test 3: Disagreement analysis
    print("\n=== TEST 3: Disagreement Analysis ===")
    disagreement_analysis = validator.analyze_disagreement(predictions_bad)
    print(f"Overall CV: {disagreement_analysis['overall_statistics']['cv']:.4f}")
    print(f"Range: {disagreement_analysis['overall_statistics']['range']:.4f}")
    print(f"Most Disagreeing: {disagreement_analysis['most_disagreeing_model']['name']}")

    # Test 4: Consensus strength
    print("\n=== TEST 4: Consensus Strength ===")
    strength = validator.get_consensus_strength(result_good)
    print(f"Consensus Strength: {strength}")

    print("\n✓ Multi-Model Consensus Validator Test Completed")
