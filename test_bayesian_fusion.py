#!/usr/bin/env python3
"""
Test per il nuovo layer di Bayesian Fusion e la sua integrazione con Telegram.
"""

from ai_system.models.bayesian_fusion import BayesianFusionModel
from ai_system.utils.bayesian_fusion_utils import attach_bayesian_fusion
from ai_system.telegram_notifier import TelegramNotifier


def test_bayesian_fusion_respects_reliability():
    model = BayesianFusionModel()
    signals = {"manual": 0.62, "xgboost": 0.55, "sentiment": 0.58}
    fused = model.fuse(signals, league="SERIE A", market="1X2")
    assert 0.55 < fused["probability"] < 0.62

    # Potenziamo manual (successi ripetuti) e verifichiamo che pesi di più
    for _ in range(5):
        model.update_reliability("SERIE A", "1X2", "manual", success=True, weight=1.0)

    fused_after = model.fuse(signals, league="SERIE A", market="1X2")
    assert fused_after["probability"] > fused["probability"]
    assert fused_after["confidence"] >= fused["confidence"]


def test_attach_bayesian_fusion_enriches_analysis():
    model = BayesianFusionModel()
    analysis = {"market": "1X2", "probability": 0.58}
    signals = {"ensemble": analysis["probability"], "manual": 0.60}

    enriched = attach_bayesian_fusion(
        analysis,
        signals=signals,
        league="Serie A",
        market="1X2",
        fusion_model=model,
    )

    assert "bayesian_fusion" in enriched
    fusion = enriched["bayesian_fusion"]
    assert 0 <= fusion["probability"] <= 1
    assert fusion["ci_low"] <= fusion["probability"] <= fusion["ci_high"]


def test_telegram_notifier_includes_bayesian_block():
    class DummyNotifier(TelegramNotifier):
        def __init__(self):
            super().__init__(
                bot_token="TEST",
                chat_id="CHAT",
                min_ev=1.0,
                min_confidence=1.0,
            )
            self.last_message = None

        def _send_message(self, message: str, parse_mode: str = "HTML") -> bool:  # type: ignore[override]
            self.last_message = message
            return True

    notifier = DummyNotifier()
    match = {"home": "Inter", "away": "Milan", "league": "Serie A"}
    analysis = {
        "action": "BET",
        "market": "1X2",
        "stake_amount": 100.0,
        "ev": 12.0,
        "probability": 0.61,
        "odds": 1.85,
        "confidence_level": 80.0,
        "bayesian_fusion": {
            "probability": 0.63,
            "ci_low": 0.58,
            "ci_high": 0.68,
            "confidence": 88.0,
        },
    }

    sent = notifier.send_betting_opportunity(match, analysis, "PRE-MATCH")
    assert sent is True
    assert notifier.last_message is not None
    assert "Bayesian Fusion" in notifier.last_message
    assert "CI 95%" in notifier.last_message
