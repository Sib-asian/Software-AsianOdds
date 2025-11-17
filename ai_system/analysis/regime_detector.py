"""
Regime Detector
===============

Analizza il regime di mercato (sharp rush, public hype, ecc.) combinando
trend delle quote, volatilità, spike di volume e qualità dei dati.

Può essere allenato offline (batch) e poi caricatto runtime per fornire
un'etichetta rapida da usare nei blocchi Value e Risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from ..config import AIConfig

logger = logging.getLogger(__name__)


@dataclass
class RegimeSample:
    """Campione grezzo utilizzato per il training."""

    trend: float
    volatility: float
    volume_spike: float
    data_quality: float
    sentiment: float
    expected_value: float
    time_to_kickoff: float


class RegimeDetector:
    """
    Riconoscitore di regimi di mercato.

    OUTPUT:
        {
            "label": "sharp_rush",
            "score": 0.82,
            "features": {...}
        }
    """

    DEFAULT_LABELS = {
        "sharp_rush": "sharp_rush",
        "public_hype": "public_hype",
        "stable_market": "stable_market",
        "chaotic": "chaotic",
        "baseline": "baseline",
    }

    def __init__(self, config: Optional[AIConfig] = None, n_clusters: int = 4):
        self.config = config or AIConfig()
        self.model_dir = Path(self.config.models_dir)
        self.model_path = self.model_dir / "regime_detector.pkl"
        self.n_clusters = n_clusters

        self.model: Optional[MiniBatchKMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_labels: Dict[int, str] = {}
        self.is_trained = False

        self._auto_load()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        movement: Optional[Dict[str, Any]],
        odds_history: Optional[List[Dict[str, Any]]],
        odds_current: float,
        time_to_kickoff_hours: float,
        api_context: Dict[str, Any],
        sentiment_result: Optional[Dict[str, Any]] = None,
        value_result: Optional[Dict[str, Any]] = None,
        volume_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Restituisce label regime per l'analisi corrente."""

        features = self._build_feature_vector(
            movement or {},
            odds_history or [],
            odds_current,
            time_to_kickoff_hours,
            api_context,
            sentiment_result or {},
            value_result or {},
            volume_history or [],
        ).astype(np.float64)

        score = 0.5
        label = "baseline"

        if self.is_trained and self.model and self.scaler:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            cluster_idx = int(self.model.predict(features_scaled)[0])
            label = self.cluster_labels.get(cluster_idx, "baseline")
            distances = self.model.transform(features_scaled)[0]
            max_dist = float(np.max(distances)) or 1.0
            score = 1.0 - float(distances[cluster_idx] / max_dist)
        else:
            label, score = self._heuristic_label(features)

        return {
            "label": label,
            "score": float(np.clip(score, 0.0, 1.0)),
            "features": {
                "trend": features[0],
                "volatility": features[1],
                "volume_spike": features[2],
                "data_quality": features[3],
                "sentiment": features[4],
                "expected_value": features[5],
                "time_to_kickoff": features[6],
            },
        }

    def train(self, samples: Iterable[RegimeSample]) -> Dict[str, Any]:
        """Allena il modello su campioni forniti."""
        df = self._samples_to_dataframe(samples)
        if df.empty:
            raise ValueError("RegimeDetector: nessun campione valido per il training")

        features = df[
            [
                "trend",
                "volatility",
                "volume_spike",
                "data_quality",
                "sentiment",
                "expected_value",
                "time_to_kickoff",
            ]
        ].to_numpy(dtype=np.float64)

        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        self.model = MiniBatchKMeans(
            n_clusters=min(self.n_clusters, len(features)),
            random_state=42,
            batch_size=min(1024, len(features)),
            max_iter=100,
        )
        self.model.fit(features_scaled)
        self.cluster_labels = self._label_clusters()
        self.is_trained = True

        logger.info(
            "✅ RegimeDetector addestrato su %s campioni (cluster=%s)",
            len(features),
            len(self.cluster_labels),
        )
        return {"samples": len(features), "clusters": len(self.cluster_labels)}

    def save(self, path: Optional[Path] = None):
        """Salva modello + scaler."""
        if not self.is_trained or not self.model or not self.scaler:
            raise RuntimeError("RegimeDetector non allenato, impossibile salvare")

        path = Path(path) if path else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "cluster_labels": self.cluster_labels,
            "n_clusters": self.n_clusters,
        }
        import joblib

        joblib.dump(payload, path)
        logger.info("💾 RegimeDetector salvato in %s", path)

    def load(self, path: Optional[Path] = None):
        """Carica modello da file."""
        path = Path(path) if path else self.model_path
        if not path.exists():
            raise FileNotFoundError(path)

        import joblib

        payload = joblib.load(path)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.cluster_labels = payload.get("cluster_labels", {})
        self.n_clusters = payload.get("n_clusters", self.n_clusters)
        self.is_trained = True
        logger.info("✅ RegimeDetector caricato da %s", path)

    # ------------------------------------------------------------------ #
    # Dataset utilities
    # ------------------------------------------------------------------ #
    def build_dataset_from_history(self, df: pd.DataFrame) -> List[RegimeSample]:
        """
        Prova a ricostruire campioni da uno storico pipeline.
        Richiede colonne compatibili (trend/volatilità o odds_history serializzate).
        """
        samples: List[RegimeSample] = []

        if df.empty:
            return samples

        for _, row in df.iterrows():
            try:
                movement = row.get("movement")
                odds_history = row.get("odds_history") or []
                if isinstance(odds_history, str):
                    odds_history = self._safe_json_load(odds_history)

                odds_current = float(row.get("odds_current", row.get("odds", 2.0) or 2.0))
                time_to_kickoff = float(row.get("time_to_kickoff_hours", 6.0))
                api_quality = float(row.get("data_quality", 0.6))
                sentiment = float(row.get("sentiment_score", 0.0))
                ev = float(row.get("expected_value", row.get("ev", 0.0)))

                sample = self._build_sample_from_inputs(
                    movement or {},
                    odds_history,
                    odds_current,
                    time_to_kickoff,
                    api_quality,
                    sentiment,
                    ev,
                    row.get("volume_history"),
                )
                samples.append(sample)
            except Exception:
                continue

        return samples

    def generate_synthetic_samples(self, n_samples: int = 2000) -> List[RegimeSample]:
        """Genera dataset sintetico per bootstrap/CI testing."""
        rng = np.random.default_rng(42)
        samples: List[RegimeSample] = []

        for _ in range(n_samples):
            regime = rng.choice(
                ["sharp_rush", "public_hype", "stable_market", "chaotic", "baseline"],
                p=[0.2, 0.2, 0.3, 0.2, 0.1],
            )
            if regime == "sharp_rush":
                trend = rng.uniform(-0.12, -0.04)
                volatility = rng.uniform(0.03, 0.08)
                volume = rng.uniform(1.5, 3.5)
                sentiment = rng.uniform(-0.2, 0.2)
                ev = rng.uniform(0.08, 0.20)
            elif regime == "public_hype":
                trend = rng.uniform(0.04, 0.12)
                volatility = rng.uniform(0.02, 0.05)
                volume = rng.uniform(0.8, 1.5)
                sentiment = rng.uniform(0.3, 0.8)
                ev = rng.uniform(-0.05, 0.08)
            elif regime == "stable_market":
                trend = rng.uniform(-0.01, 0.01)
                volatility = rng.uniform(0.005, 0.02)
                volume = rng.uniform(0.8, 1.2)
                sentiment = rng.uniform(-0.1, 0.3)
                ev = rng.uniform(-0.02, 0.08)
            elif regime == "chaotic":
                trend = rng.uniform(-0.08, 0.08)
                volatility = rng.uniform(0.05, 0.12)
                volume = rng.uniform(1.0, 2.5)
                sentiment = rng.uniform(-0.5, 0.5)
                ev = rng.uniform(-0.1, 0.15)
            else:  # baseline
                trend = rng.uniform(-0.03, 0.03)
                volatility = rng.uniform(0.02, 0.04)
                volume = rng.uniform(0.9, 1.4)
                sentiment = rng.uniform(-0.2, 0.2)
                ev = rng.uniform(-0.05, 0.10)

            sample = RegimeSample(
                trend=trend,
                volatility=volatility,
                volume_spike=volume,
                data_quality=float(rng.uniform(0.4, 0.95)),
                sentiment=float(sentiment),
                expected_value=float(ev),
                time_to_kickoff=float(rng.uniform(0.5, 48.0)),
            )
            samples.append(sample)

        return samples

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _auto_load(self):
        if self.model_path.exists():
            try:
                self.load(self.model_path)
            except Exception as exc:
                logger.warning("⚠️  RegimeDetector: impossibile caricare modello (%s)", exc)

    def _build_feature_vector(
        self,
        movement: Dict[str, Any],
        odds_history: List[Dict[str, Any]],
        odds_current: float,
        time_to_kickoff_hours: float,
        api_context: Dict[str, Any],
        sentiment_result: Dict[str, Any],
        value_result: Dict[str, Any],
        volume_history: List[Dict[str, Any]],
    ) -> np.ndarray:
        trend = float(movement.get("trend", 0.0))
        volatility = float(movement.get("volatility", 0.0))

        volume_spike = 1.0
        volumes = [entry.get("volume") for entry in volume_history if isinstance(entry, dict) and "volume" in entry]
        if volumes:
            avg = float(np.mean(volumes[:-1])) if len(volumes) > 1 else float(volumes[0])
            last = float(volumes[-1])
            if avg > 0:
                volume_spike = last / avg

        data_quality = float(api_context.get("metadata", {}).get("data_quality", 0.5))
        sentiment = float(
            sentiment_result.get("overall_sentiment_home")
            or sentiment_result.get("overall_sentiment_away")
            or 0.0
        )
        sentiment /= 100.0  # normalize -1..1

        ev = float(value_result.get("expected_value", 0.0))
        time_to_kickoff = float(min(max(time_to_kickoff_hours, 0.0), 72.0))

        # Fallback volatility if not provided
        if volatility == 0.0 and len(odds_history) >= 3:
            odds_values = [entry.get("odds") for entry in odds_history if entry.get("odds")]
            if len(odds_values) >= 3:
                volatility = float(np.std(odds_values))
                first = odds_values[0]
                last = odds_values[-1]
                if first:
                    trend = float((last - first) / first)

        return np.array(
            [
                trend,
                volatility,
                volume_spike,
                data_quality,
                sentiment,
                ev,
                time_to_kickoff,
            ],
            dtype=np.float32,
        )

    def _heuristic_label(self, features: np.ndarray) -> Tuple[str, float]:
        trend, volatility, volume_spike, _, sentiment, ev, _ = features

        if trend < -0.04 and volume_spike > 1.5:
            return "sharp_rush", 0.8
        if trend > 0.04 and sentiment > 0.2:
            return "public_hype", 0.7
        if volatility < 0.02 and abs(trend) < 0.015:
            return "stable_market", 0.6
        if volatility > 0.06:
            return "chaotic", 0.65
        if ev > 0.12 and volume_spike > 1.2:
            return "sharp_rush", 0.6
        return "baseline", 0.5

    def _label_clusters(self) -> Dict[int, str]:
        labels = {}
        if not self.model or not self.scaler:
            return labels

        centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        for idx, center in enumerate(centers):
            trend, volatility, volume_spike, _, sentiment, ev, _ = center
            label, _ = self._heuristic_label(
                np.array([trend, volatility, volume_spike, 0, sentiment, ev, 0])
            )
            labels[idx] = label
        return labels

    def _build_sample_from_inputs(
        self,
        movement: Dict[str, Any],
        odds_history: List[Dict[str, Any]],
        odds_current: float,
        time_to_kickoff: float,
        data_quality: float,
        sentiment_score: float,
        expected_value: float,
        volume_history: Optional[List[Dict[str, Any]]] = None,
    ) -> RegimeSample:
        features = self._build_feature_vector(
            movement,
            odds_history,
            odds_current,
            time_to_kickoff,
            {"metadata": {"data_quality": data_quality}},
            {"overall_sentiment_home": sentiment_score * 100},
            {"expected_value": expected_value},
            volume_history or [],
        )

        return RegimeSample(
            trend=float(features[0]),
            volatility=float(features[1]),
            volume_spike=float(features[2]),
            data_quality=float(features[3]),
            sentiment=float(features[4]),
            expected_value=float(features[5]),
            time_to_kickoff=float(features[6]),
        )

    def _samples_to_dataframe(self, samples: Iterable[RegimeSample]) -> pd.DataFrame:
        rows = [
            {
                "trend": s.trend,
                "volatility": s.volatility,
                "volume_spike": s.volume_spike,
                "data_quality": s.data_quality,
                "sentiment": s.sentiment,
                "expected_value": s.expected_value,
                "time_to_kickoff": s.time_to_kickoff,
            }
            for s in samples
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _safe_json_load(raw: Any) -> Any:
        import json

        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return []
        return raw


__all__ = ["RegimeDetector", "RegimeSample"]
