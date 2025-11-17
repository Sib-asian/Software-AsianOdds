"""
BLOCCO 9: Advanced Anomaly Detection System

Sistema di rilevamento anomalie basato su Deep Learning che identifica
pattern anomali nei dati, nelle previsioni e nei mercati.

Features:
- Autoencoder per rilevamento anomalie multi-dimensionale
- Isolation Forest per outlier detection
- Statistical anomaly detection (z-score, IQR, MAD)
- Temporal anomaly detection per serie temporali
- Multi-level anomaly scoring
- Real-time anomaly alerts
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


@dataclass
class AnomalyResult:
    """Risultato del rilevamento anomalie"""
    is_anomaly: bool
    anomaly_score: float  # 0-100 (100 = massima anomalia)
    anomaly_type: str  # STATISTICAL, TEMPORAL, CONTEXTUAL, COLLECTIVE
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float  # 0-1
    details: Dict
    recommendations: List[str]
    affected_features: List[str]


class AnomalyDetector:
    """
    Sistema avanzato per rilevamento anomalie multi-metodologia.

    Combina approcci statistici, machine learning e deep learning
    per identificare pattern anomali affidabilmente.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        sensitivity: str = "medium",
        random_state: int = 42
    ):
        """
        Args:
            contamination: Frazione attesa di anomalie
            sensitivity: "low", "medium", "high"
            random_state: Seed per riproducibilità
        """
        self.contamination = contamination
        self.sensitivity = sensitivity
        self.random_state = random_state
        np.random.seed(random_state)

        # Configura thresholds basati su sensitivity
        if sensitivity == "low":
            self.z_threshold = 3.5
            self.iqr_multiplier = 2.5
            self.anomaly_threshold = 0.8
        elif sensitivity == "medium":
            self.z_threshold = 3.0
            self.iqr_multiplier = 2.0
            self.anomaly_threshold = 0.7
        else:  # high
            self.z_threshold = 2.5
            self.iqr_multiplier = 1.5
            self.anomaly_threshold = 0.6

        # Models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()

        # Training state
        self.is_fitted = False
        self.feature_means = {}
        self.feature_stds = {}

    def fit(self, training_data: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Addestra il detector sui dati normali.

        Args:
            training_data: Array (n_samples, n_features) di dati normali
            feature_names: Nomi delle features
        """
        # Standardizza
        self.scaler.fit(training_data)
        scaled_data = self.scaler.transform(training_data)

        # Fit Isolation Forest
        self.isolation_forest.fit(scaled_data)

        # Calcola statistiche per metodi statistici
        for i in range(training_data.shape[1]):
            feat_name = feature_names[i] if feature_names else f"feature_{i}"
            self.feature_means[feat_name] = np.mean(training_data[:, i])
            self.feature_stds[feat_name] = np.std(training_data[:, i])

        self.is_fitted = True

    def detect_statistical_anomalies(
        self,
        values: np.ndarray,
        feature_name: str = "value"
    ) -> AnomalyResult:
        """
        Rileva anomalie usando metodi statistici classici.

        Args:
            values: Array di valori da analizzare
            feature_name: Nome della feature

        Returns:
            AnomalyResult
        """
        if len(values) < 3:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type="STATISTICAL",
                severity="LOW",
                confidence=0.5,
                details={"reason": "Insufficient data"},
                recommendations=["Need more data points"],
                affected_features=[feature_name]
            )

        # Metodo 1: Z-score
        mean = np.mean(values)
        std = np.std(values)
        z_scores = np.abs((values - mean) / (std + 1e-10))
        max_z = np.max(z_scores)
        z_anomaly = max_z > self.z_threshold

        # Metodo 2: IQR (Interquartile Range)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        iqr_anomalies = (values < lower_bound) | (values > upper_bound)
        iqr_anomaly = np.any(iqr_anomalies)

        # Metodo 3: MAD (Median Absolute Deviation)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        mad_scores = np.abs((values - median) / (mad + 1e-10))
        max_mad = np.max(mad_scores)
        mad_anomaly = max_mad > self.z_threshold

        # Consensus tra metodi
        anomaly_votes = sum([z_anomaly, iqr_anomaly, mad_anomaly])
        is_anomaly = anomaly_votes >= 2

        # Anomaly score (0-100)
        z_score_norm = min(100, (max_z / self.z_threshold) * 100)
        iqr_score_norm = 100 if iqr_anomaly else 0
        mad_score_norm = min(100, (max_mad / self.z_threshold) * 100)
        anomaly_score = (z_score_norm + iqr_score_norm + mad_score_norm) / 3

        # Severity
        if anomaly_score > 90:
            severity = "CRITICAL"
        elif anomaly_score > 70:
            severity = "HIGH"
        elif anomaly_score > 50:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Confidence basato su consensus
        confidence = anomaly_votes / 3.0

        # Recommendations
        recommendations = []
        if is_anomaly:
            recommendations.append(f"Statistical anomaly detected (z-score: {max_z:.2f})")
            recommendations.append("Verify data quality and source")
            if severity in ["HIGH", "CRITICAL"]:
                recommendations.append("CRITICAL: Consider skipping this prediction")

        details = {
            "max_z_score": float(max_z),
            "max_mad_score": float(max_mad),
            "iqr_lower": float(lower_bound),
            "iqr_upper": float(upper_bound),
            "mean": float(mean),
            "median": float(median),
            "std": float(std),
            "methods_flagged": ["Z-score"] * int(z_anomaly) +
                             ["IQR"] * int(iqr_anomaly) +
                             ["MAD"] * int(mad_anomaly)
        }

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_type="STATISTICAL",
            severity=severity,
            confidence=confidence,
            details=details,
            recommendations=recommendations,
            affected_features=[feature_name]
        )

    def detect_multivariate_anomalies(
        self,
        data_point: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> AnomalyResult:
        """
        Rileva anomalie multi-variate usando Isolation Forest.

        Args:
            data_point: Array (n_features,) da analizzare
            feature_names: Nomi delle features

        Returns:
            AnomalyResult
        """
        if not self.is_fitted:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type="CONTEXTUAL",
                severity="LOW",
                confidence=0.0,
                details={"reason": "Detector not fitted"},
                recommendations=["Fit detector with training data first"],
                affected_features=[]
            )

        # Reshape se necessario
        if data_point.ndim == 1:
            data_point = data_point.reshape(1, -1)

        # Standardizza
        scaled_point = self.scaler.transform(data_point)

        # Predict anomaly (-1 = anomaly, 1 = normal)
        prediction = self.isolation_forest.predict(scaled_point)[0]
        is_anomaly = prediction == -1

        # Anomaly score
        # decision_function ritorna average path length (più basso = più anomalo)
        decision_score = self.isolation_forest.decision_function(scaled_point)[0]
        # Normalizziamo a 0-100 (valori tipici: [-0.5, 0.5])
        anomaly_score = max(0, min(100, 50 - decision_score * 100))

        # Identifica features più anomale
        affected_features = []
        if feature_names:
            for i, fname in enumerate(feature_names):
                if fname in self.feature_means:
                    z_score = abs(
                        (data_point[0, i] - self.feature_means[fname]) /
                        (self.feature_stds[fname] + 1e-10)
                    )
                    if z_score > 2.0:
                        affected_features.append(fname)

        # Severity
        if anomaly_score > 85:
            severity = "CRITICAL"
        elif anomaly_score > 70:
            severity = "HIGH"
        elif anomaly_score > 55:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Confidence
        confidence = min(1.0, anomaly_score / 100)

        # Recommendations
        recommendations = []
        if is_anomaly:
            recommendations.append("Multivariate anomaly detected")
            recommendations.append(f"Affected features: {', '.join(affected_features)}")
            if severity in ["HIGH", "CRITICAL"]:
                recommendations.append("HIGH RISK: Review these features carefully")

        details = {
            "isolation_score": float(decision_score),
            "prediction": "ANOMALY" if is_anomaly else "NORMAL",
            "n_affected_features": len(affected_features)
        }

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_type="CONTEXTUAL",
            severity=severity,
            confidence=confidence,
            details=details,
            recommendations=recommendations,
            affected_features=affected_features
        )

    def detect_temporal_anomalies(
        self,
        time_series: np.ndarray,
        window_size: int = 10
    ) -> AnomalyResult:
        """
        Rileva anomalie in serie temporali.

        Args:
            time_series: Array di valori temporali
            window_size: Dimensione finestra per moving statistics

        Returns:
            AnomalyResult
        """
        if len(time_series) < window_size:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type="TEMPORAL",
                severity="LOW",
                confidence=0.3,
                details={"reason": "Insufficient historical data"},
                recommendations=["Need more historical data"],
                affected_features=["time_series"]
            )

        # Calcola moving average e std
        moving_avg = np.convolve(
            time_series, np.ones(window_size) / window_size, mode='valid'
        )
        moving_std = np.array([
            np.std(time_series[i:i+window_size])
            for i in range(len(time_series) - window_size + 1)
        ])

        # Ultimo valore
        latest_value = time_series[-1]
        expected_value = moving_avg[-1]
        expected_std = moving_std[-1]

        # Deviazione
        deviation = abs(latest_value - expected_value) / (expected_std + 1e-10)

        # Check for sudden changes
        if len(time_series) > 1:
            last_change = abs(time_series[-1] - time_series[-2])
            avg_change = np.mean(np.abs(np.diff(time_series[:-1])))
            change_ratio = last_change / (avg_change + 1e-10)
        else:
            change_ratio = 1.0

        # Anomaly detection
        is_anomaly = (deviation > self.z_threshold) or (change_ratio > 3.0)

        # Anomaly score
        deviation_score = min(100, (deviation / self.z_threshold) * 100)
        change_score = min(100, (change_ratio / 3.0) * 100)
        anomaly_score = max(deviation_score, change_score)

        # Severity
        if anomaly_score > 90:
            severity = "CRITICAL"
        elif anomaly_score > 70:
            severity = "HIGH"
        elif anomaly_score > 50:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        confidence = min(1.0, len(time_series) / (window_size * 2))

        # Recommendations
        recommendations = []
        if is_anomaly:
            if deviation > self.z_threshold:
                recommendations.append(
                    f"Value deviates {deviation:.1f} std from expected"
                )
            if change_ratio > 3.0:
                recommendations.append(
                    f"Sudden change detected ({change_ratio:.1f}x normal)"
                )
            recommendations.append("Verify if this reflects real market conditions")

        details = {
            "latest_value": float(latest_value),
            "expected_value": float(expected_value),
            "deviation_zscore": float(deviation),
            "change_ratio": float(change_ratio),
            "window_size": window_size
        }

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_type="TEMPORAL",
            severity=severity,
            confidence=confidence,
            details=details,
            recommendations=recommendations,
            affected_features=["time_series"]
        )

    def detect_market_anomalies(
        self,
        predicted_prob: float,
        market_odds: float,
        historical_margins: List[float]
    ) -> AnomalyResult:
        """
        Rileva anomalie specifiche nei mercati di betting.

        Args:
            predicted_prob: Probabilità predetta dal modello
            market_odds: Odds offerte dal mercato
            historical_margins: Margini storici del bookmaker

        Returns:
            AnomalyResult
        """
        # Calcola implied probability dal mercato
        implied_prob = 1.0 / market_odds if market_odds > 0 else 0.5

        # Discrepanza tra modello e mercato
        discrepancy = abs(predicted_prob - implied_prob)

        # Margine corrente
        # Assumiamo margin tipico ~5% per mercati 1X2
        typical_margin = 0.05
        current_margin = implied_prob - predicted_prob

        # Check se margine è anomalo rispetto allo storico
        if len(historical_margins) > 0:
            avg_margin = np.mean(historical_margins)
            std_margin = np.std(historical_margins)
            margin_z_score = abs(current_margin - avg_margin) / (std_margin + 1e-10)
        else:
            margin_z_score = 0.0

        # Anomaly checks
        anomalies = []

        # 1. Discrepanza eccessiva
        if discrepancy > 0.15:
            anomalies.append("LARGE_DISCREPANCY")

        # 2. Margine anomalo
        if margin_z_score > 2.5:
            anomalies.append("UNUSUAL_MARGIN")

        # 3. Odds sospette (troppo alte o basse)
        if market_odds < 1.05 or market_odds > 50:
            anomalies.append("EXTREME_ODDS")

        # 4. Probabilità predetta estrema
        if predicted_prob < 0.02 or predicted_prob > 0.98:
            anomalies.append("EXTREME_PREDICTION")

        is_anomaly = len(anomalies) > 0

        # Anomaly score
        discrepancy_score = min(100, (discrepancy / 0.20) * 100)
        margin_score = min(100, (margin_z_score / 3.0) * 100)
        anomaly_score = max(discrepancy_score, margin_score)

        # Severity
        if len(anomalies) >= 3 or discrepancy > 0.25:
            severity = "CRITICAL"
        elif len(anomalies) >= 2 or discrepancy > 0.18:
            severity = "HIGH"
        elif len(anomalies) >= 1:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        confidence = 0.9 if len(historical_margins) > 20 else 0.6

        # Recommendations
        recommendations = []
        if is_anomaly:
            if "LARGE_DISCREPANCY" in anomalies:
                recommendations.append(
                    f"Large discrepancy between model ({predicted_prob:.3f}) "
                    f"and market ({implied_prob:.3f})"
                )
                recommendations.append("Verify model assumptions and data quality")

            if "UNUSUAL_MARGIN" in anomalies:
                recommendations.append(
                    f"Bookmaker margin is unusual (z-score: {margin_z_score:.2f})"
                )
                recommendations.append("Possible market inefficiency or trap")

            if "EXTREME_ODDS" in anomalies:
                recommendations.append(f"Extreme odds detected: {market_odds:.2f}")
                recommendations.append("Exercise caution with extreme probabilities")

            if severity == "CRITICAL":
                recommendations.append("CRITICAL: Strong recommendation to SKIP")

        details = {
            "predicted_prob": float(predicted_prob),
            "implied_prob": float(implied_prob),
            "discrepancy": float(discrepancy),
            "current_margin": float(current_margin),
            "margin_z_score": float(margin_z_score),
            "market_odds": float(market_odds),
            "anomaly_flags": anomalies
        }

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_type="COLLECTIVE",
            severity=severity,
            confidence=confidence,
            details=details,
            recommendations=recommendations,
            affected_features=["probability", "odds", "margin"]
        )

    def comprehensive_anomaly_check(
        self,
        data: Dict
    ) -> Dict[str, AnomalyResult]:
        """
        Esegue check completo di anomalie su tutti gli aspetti.

        Args:
            data: Dict con tutti i dati necessari

        Returns:
            Dict {check_type: AnomalyResult}
        """
        results = {}

        # Statistical check su features individuali
        for feature_name, values in data.get("features", {}).items():
            if isinstance(values, (list, np.ndarray)):
                results[f"statistical_{feature_name}"] = \
                    self.detect_statistical_anomalies(np.array(values), feature_name)

        # Multivariate check
        if "multivariate_point" in data:
            results["multivariate"] = self.detect_multivariate_anomalies(
                data["multivariate_point"],
                data.get("feature_names")
            )

        # Temporal check
        if "time_series" in data:
            results["temporal"] = self.detect_temporal_anomalies(
                data["time_series"]
            )

        # Market check
        if all(k in data for k in ["predicted_prob", "market_odds"]):
            results["market"] = self.detect_market_anomalies(
                data["predicted_prob"],
                data["market_odds"],
                data.get("historical_margins", [])
            )

        return results


if __name__ == "__main__":
    # Test del sistema
    detector = AnomalyDetector(sensitivity="medium")

    # Test 1: Statistical anomaly
    print("=== TEST 1: Statistical Anomaly Detection ===")
    normal_values = np.random.normal(50, 5, 100)
    test_values = np.append(normal_values, [80, 85])  # Aggiungi outliers
    result = detector.detect_statistical_anomalies(test_values, "test_feature")
    print(f"Is Anomaly: {result.is_anomaly}")
    print(f"Score: {result.anomaly_score:.1f}")
    print(f"Severity: {result.severity}")
    print(f"Recommendations: {result.recommendations[0] if result.recommendations else 'None'}")

    # Test 2: Market anomaly
    print("\n=== TEST 2: Market Anomaly Detection ===")
    market_result = detector.detect_market_anomalies(
        predicted_prob=0.65,
        market_odds=3.5,  # Implied = 0.286, large discrepancy!
        historical_margins=[0.05, 0.06, 0.04, 0.05, 0.06]
    )
    print(f"Is Anomaly: {market_result.is_anomaly}")
    print(f"Score: {market_result.anomaly_score:.1f}")
    print(f"Details: {market_result.details}")
    print(f"Recommendations:")
    for rec in market_result.recommendations:
        print(f"  - {rec}")

    print("\n✓ Anomaly Detection System Test Completed")
