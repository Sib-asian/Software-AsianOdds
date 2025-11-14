"""
BLOCCO 6: Odds Movement Tracker
================================

Monitora movimenti quote in real-time e suggerisce timing ottimale.

Features:
- LSTM per previsione movimento quote
- Sharp money detection
- Timing recommendations (BET_NOW / WAIT / WATCH)
- Urgency levels based on time to kickoff

Input: match + current odds + decision from risk manager
Output: timing recommendation + predicted odds + urgency
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone

# PyTorch per LSTM
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import AIConfig
from .models.chronos_forecaster import ChronosForecaster
from .utils.theodds_api_client import TheOddsAPIClient

logger = logging.getLogger(__name__)


# Define classes only if PyTorch is available
if TORCH_AVAILABLE:
    class OddsLSTM(nn.Module):
        """LSTM per previsione movimento quote"""

        def __init__(self, input_size: int, hidden_size: int, num_layers: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)  # Predict single odds value

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            predictions = self.fc(lstm_out[:, -1, :])  # Last timestep
            return predictions
else:
    # Dummy class when PyTorch is not available
    class OddsLSTM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for OddsLSTM. Install with: pip install torch")


class OddsMovementTracker:
    """Tracker per movimenti quote e timing ottimale"""

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self.model: Optional[OddsLSTM] = None
        self.is_trained = False
        self.chronos = ChronosForecaster(self.config) if self.config.chronos_enabled else None
        self.theodds_client = TheOddsAPIClient(
            api_key=self.config.theodds_api_key,
            regions=self.config.theodds_regions,
            markets=self.config.theodds_markets,
            primary_market=self.config.theodds_primary_market,
            odds_format=self.config.theodds_odds_format,
            date_format=self.config.theodds_date_format,
            sport_mapping=self.config.theodds_sport_mapping,
            history_window_hours=self.config.theodds_history_window_hours
        ) if self.config.theodds_enabled else None

        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch not available, using rule-based tracking")

        logger.info("✅ Odds Movement Tracker initialized")

    def monitor(
        self,
        match: Dict,
        decision: Dict,
        current_odds: float,
        odds_history: List[Dict],
        time_to_kickoff_hours: float,
        market: str = "h2h",
        selection: str = "home"
    ) -> Dict[str, Any]:
        """
        Monitora quote e fornisce raccomandazione timing.

        Returns:
            - timing_recommendation: BET_NOW / WAIT / WATCH
            - predicted_odds_1h: Previsione tra 1h
            - urgency: LOW / MEDIUM / HIGH
            - sharp_money_detected: bool
            - reasoning: Spiegazione
        """
        # Ensure odds_history is never None
        if odds_history is None:
            odds_history = []

        # Se decisione è SKIP, non serve monitoring
        if decision.get("decision") == "SKIP":
            return {
                "timing_recommendation": "SKIP",
                "predicted_odds_1h": current_odds,
                "urgency": "NONE",
                "sharp_money_detected": False,
                "reasoning": "Bet skipped by risk manager"
            }

        market = market or "h2h"
        selection = selection or "home"

        live_snapshot = None
        if self.theodds_client and self.config.theodds_auto_refresh:
            odds_history, refreshed_price, live_snapshot = self._enhance_history_with_live_feed(
                odds_history,
                match,
                market,
                selection
            )
            if refreshed_price is not None:
                current_odds = refreshed_price

        # Analizza movimento quote
        movement_analysis = self._analyze_movement(odds_history, current_odds)

        # Detect sharp money
        sharp_money = movement_analysis.get("sharp_money_detected", False)

        # Predict future odds (se model trainato)
        if self.is_trained and len(odds_history) >= self.config.odds_min_data_points:
            predicted_odds_1h = self._predict_odds(odds_history, current_odds)
        else:
            # Rule-based prediction
            predicted_odds_1h = self._rule_based_prediction(
                odds_history, current_odds, movement_analysis
            )

        # Determine timing recommendation
        timing_rec = self._determine_timing(
            movement_analysis,
            sharp_money,
            current_odds,
            predicted_odds_1h,
            time_to_kickoff_hours
        )

        # Determine urgency
        urgency = self._determine_urgency(
            time_to_kickoff_hours,
            sharp_money,
            movement_analysis
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            timing_rec,
            sharp_money,
            movement_analysis,
            current_odds,
            predicted_odds_1h
        )

        return {
            "timing_recommendation": timing_rec,
            "predicted_odds_1h": float(predicted_odds_1h),
            "current_odds": float(current_odds),
            "urgency": urgency,
            "sharp_money_detected": sharp_money,
            "movement_pattern": movement_analysis.get("pattern", "STABLE"),
            "reasoning": reasoning,
            "live_odds_snapshot": live_snapshot
        }

    def _analyze_movement(
        self,
        odds_history: List[Dict],
        current_odds: float
    ) -> Dict:
        """Analizza pattern di movimento quote"""
        if len(odds_history) < 2:
            return {
                "pattern": "UNKNOWN",
                "trend": 0.0,
                "volatility": 0.0,
                "sharp_money_detected": False
            }

        odds_values = [o["odds"] for o in odds_history] + [current_odds]
        odds_first = odds_values[0]
        odds_last = odds_values[-1]

        # Trend
        trend = (odds_last - odds_first) / odds_first

        # Volatility
        volatility = np.std(odds_values) if len(odds_values) > 2 else 0.0

        # Pattern classification
        if abs(trend) < 0.02:
            pattern = "STABLE"
        elif trend < -0.05:
            pattern = "FALLING"  # Sharp money likely
        elif trend > 0.05:
            pattern = "RISING"   # Public money / value trap
        else:
            pattern = "VOLATILE"

        # Sharp money detection
        sharp_money = (
            pattern == "FALLING" and
            abs(trend) > abs(self.config.sharp_money_threshold)
        )

        return {
            "pattern": pattern,
            "trend": float(trend),
            "volatility": float(volatility),
            "sharp_money_detected": sharp_money,
            "odds_first": odds_first,
            "odds_last": odds_last
        }

    def _rule_based_prediction(
        self,
        odds_history: List[Dict],
        current_odds: float,
        movement_analysis: Dict
    ) -> float:
        """Previsione basata su regole"""
        trend = movement_analysis.get("trend", 0.0)
        pattern = movement_analysis.get("pattern", "STABLE")

        # Simple extrapolation
        if pattern == "STABLE":
            return current_odds  # No change expected

        elif pattern == "FALLING":
            # Continue falling (dampened)
            predicted = current_odds * (1 + trend * 0.5)  # 50% of trend

        elif pattern == "RISING":
            # Continue rising (dampened)
            predicted = current_odds * (1 + trend * 0.5)

        else:  # VOLATILE
            return current_odds  # Too uncertain

        # Clamp to reasonable range
        return np.clip(predicted, current_odds * 0.90, current_odds * 1.10)

    def _predict_odds(
        self,
        odds_history: List[Dict],
        current_odds: float
    ) -> float:
        """Prediction using Chronos or fallback."""
        series = [entry.get("odds", current_odds) for entry in odds_history[-self.config.odds_lookback_window:]]
        series.append(current_odds)

        if self.chronos and self.chronos.enabled:
            forecast = self.chronos.forecast(series)
            if forecast and forecast.get("median") is not None:
                return float(forecast["median"])

        # TODO: Implement LSTM prediction
        logger.debug("Chronos unavailable, falling back to current odds for prediction.")
        return current_odds

    def _determine_timing(
        self,
        movement_analysis: Dict,
        sharp_money: bool,
        current_odds: float,
        predicted_odds: float,
        time_to_kickoff: float
    ) -> str:
        """Determina timing raccomandato"""
        trend = movement_analysis.get("trend", 0.0)
        pattern = movement_analysis.get("pattern", "STABLE")

        # BET NOW conditions
        if sharp_money:
            return "BET_NOW"  # Sharp money = bet immediately

        if trend < self.config.odds_bet_now_threshold:
            return "BET_NOW"  # Falling fast

        if time_to_kickoff < 1.0:
            return "BET_NOW"  # Too close to kickoff

        # WAIT conditions
        if trend > self.config.odds_wait_threshold:
            return "WAIT"  # Rising, wait for better odds

        if pattern == "VOLATILE" and time_to_kickoff > 6.0:
            return "WAIT"  # Let market stabilize

        # WATCH conditions (neutral/uncertain)
        if abs(trend) < self.config.odds_watch_threshold:
            return "WATCH"  # Stable, monitor

        return "WATCH"  # Default

    def _determine_urgency(
        self,
        time_to_kickoff: float,
        sharp_money: bool,
        movement_analysis: Dict
    ) -> str:
        """Determina urgency level"""
        if sharp_money:
            return "HIGH"

        if time_to_kickoff < self.config.urgency_high_time_hours:
            return "HIGH"

        if time_to_kickoff < self.config.urgency_medium_time_hours:
            return "MEDIUM"

        if movement_analysis.get("pattern") == "FALLING":
            return "MEDIUM"

        return "LOW"

    def _generate_reasoning(
        self,
        timing_rec: str,
        sharp_money: bool,
        movement_analysis: Dict,
        current_odds: float,
        predicted_odds: float
    ) -> str:
        """Generate explanation"""
        pattern = movement_analysis.get("pattern", "STABLE")
        trend = movement_analysis.get("trend", 0.0)

        parts = []

        if timing_rec == "BET_NOW":
            if sharp_money:
                parts.append("Sharp money detected - odds dropping with high volume.")
            else:
                parts.append(f"Odds falling ({trend:+.1%}).")
            parts.append("Bet immediately before further drop.")

        elif timing_rec == "WAIT":
            parts.append(f"Odds rising ({trend:+.1%}).")
            parts.append("Wait for better odds or skip if trend continues.")

        else:  # WATCH
            parts.append(f"Odds {pattern.lower()} ({trend:+.1%}).")
            parts.append("Monitor for changes before betting.")

        parts.append(f"Current: {current_odds:.2f}, Predicted 1h: {predicted_odds:.2f}")

        return " ".join(parts)

    def _enhance_history_with_live_feed(
        self,
        odds_history: List[Dict],
        match: Dict[str, Any],
        market: str,
        selection: str
    ) -> Tuple[List[Dict], Optional[float], Optional[Dict]]:
        if not self.theodds_client:
            return odds_history, None, None

        snapshot = self.theodds_client.fetch_latest_snapshot(match, market)
        if not snapshot:
            return odds_history, None, None

        price = self._select_price(snapshot["prices"], selection)
        if price is None:
            return odds_history, None, None

        history = list(odds_history or [])
        history.append({
            "odds": float(price),
            "timestamp": snapshot.get("timestamp"),
            "source": "theoddsapi",
            "market": snapshot.get("market"),
            "bookmaker": snapshot["prices"][selection].get("bookmaker") if snapshot["prices"].get(selection) else None
        })
        history = self._trim_history(history)

        snapshot_summary = {
            "market": snapshot.get("market"),
            "timestamp": snapshot.get("timestamp"),
            "bookmakers_queried": snapshot.get("bookmakers_queried"),
            "sport_key": snapshot.get("sport_key"),
            "selection": selection,
            "price": price,
            "bookmaker": snapshot["prices"][selection].get("bookmaker") if snapshot["prices"].get(selection) else None
        }

        return history, float(price), snapshot_summary

    @staticmethod
    def _select_price(prices: Dict[str, Dict[str, Any]], selection: str) -> Optional[float]:
        selection = selection.lower()
        target = None
        if selection in {"home", "1", "home_win"}:
            target = "home"
        elif selection in {"away", "2", "away_win"}:
            target = "away"
        elif selection in {"draw", "x"}:
            target = "draw"
        else:
            target = "home"

        entry = prices.get(target)
        if not entry:
            return None
        price = entry.get("price")
        return float(price) if price is not None else None

    def _trim_history(self, history: List[Dict]) -> List[Dict]:
        window = timedelta(hours=self.config.theodds_history_window_hours)
        cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - window
        trimmed: List[Dict] = []
        for entry in history:
            ts = self._parse_timestamp(entry.get("timestamp"))
            if ts is not None and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts is None or ts >= cutoff:
                trimmed.append(entry)
        return trimmed

    @staticmethod
    def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None

    def train(self, historical_odds_data: List[Dict]) -> Dict:
        """Train LSTM model"""
        logger.warning("⚠️ LSTM training not yet implemented")
        return {"status": "not_implemented"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tracker = OddsMovementTracker()

    odds_history = [
        {"odds": 1.90, "time": "08:00", "volume": 1000},
        {"odds": 1.88, "time": "10:00", "volume": 2500},
        {"odds": 1.85, "time": "12:00", "volume": 5000}
    ]
    decision = {"decision": "BET"}

    result = tracker.monitor(
        match={"home": "Inter", "away": "Genoa"},
        decision=decision,
        current_odds=1.85,
        odds_history=odds_history,
        time_to_kickoff_hours=3.0
    )
    print(f"\n{result}")
