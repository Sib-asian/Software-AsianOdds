"""
Chronos Forecaster
==================

Wrapper per utilizzare i modelli transformer Chronos di Nixtla/Amazon
per prevedere il movimento delle quote (o qualsiasi time-series) a breve termine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import AIConfig

logger = logging.getLogger(__name__)

try:
    from chronos import ChronosPipeline

    CHRONOS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CHRONOS_AVAILABLE = False
    ChronosPipeline = None  # type: ignore


class ChronosForecaster:
    """Lazy loader intorno a ChronosPipeline."""

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self.enabled = self.config.chronos_enabled and CHRONOS_AVAILABLE
        self._pipeline: Optional["ChronosPipeline"] = None

        if not CHRONOS_AVAILABLE and self.config.chronos_enabled:
            logger.warning("⚠️ Libreria 'chronos' non trovata. Installa dipendenza per abilitare le previsioni.")

    def forecast(
        self,
        series: List[float],
        prediction_length: Optional[int] = None,
        quantiles: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """Restituisce la previsione Chronos (median + quantili)."""
        if not self.enabled:
            return None

        if len(series) < 3:  # Serve un minimo di osservazioni
            return None

        pipeline = self._get_pipeline()
        if pipeline is None:
            return None

        pred_len = prediction_length or self.config.chronos_prediction_length
        quantiles = quantiles or self.config.chronos_quantiles
        try:
            result = pipeline.predict(
                time_series=[series],
                prediction_length=pred_len,
                quantiles=quantiles,
            )
        except Exception as exc:  # pragma: no cover - inference issues
            logger.debug(f"Chronos forecast failed: {exc}")
            return None

        quantile_map = result.get("quantiles") or {}
        summary = {
            "prediction_length": pred_len,
            "quantiles": {
                str(q): self._extract_value(quantile_map.get(str(q)))
                for q in quantiles
            },
        }
        summary["median"] = summary["quantiles"].get("0.5")
        summary["mean"] = self._extract_value(result.get("mean"))
        return summary

    def _get_pipeline(self) -> Optional["ChronosPipeline"]:
        if self._pipeline is not None:
            return self._pipeline

        if not self.enabled:
            return None

        try:
            self._pipeline = ChronosPipeline.from_pretrained(
                self.config.chronos_model_name,
                device_map=self.config.chronos_device
            )
        except Exception as exc:  # pragma: no cover - download failure
            logger.warning(f"⚠️ Chronos model init failed: {exc}")
            self.enabled = False
            self._pipeline = None
        return self._pipeline

    @staticmethod
    def _extract_value(raw: Any) -> Optional[float]:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple, np.ndarray)):
            return float(np.array(raw).flatten()[0])
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
