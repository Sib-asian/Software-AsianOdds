#!/usr/bin/env python3
"""
Script di Verifica Integrazione Blocchi IA
==========================================

Verifica che tutti i 15 blocchi IA siano correttamente importabili
senza richiedere installazioni aggiuntive. Il file può essere usato
sia con pytest che come script standalone (`python test_ai_imports.py`).
"""

import importlib
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

# Aggiungi il percorso del progetto se non già presente
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

MODULES_TO_TEST: List[Tuple[str, str]] = [
    # Config
    ("ai_system.config", "Config System"),
    # Blocchi core
    ("ai_system.blocco_0_api_engine", "Blocco 0: API Data Engine"),
    ("ai_system.blocco_1_calibrator", "Blocco 1: Probability Calibrator"),
    ("ai_system.blocco_2_confidence", "Blocco 2: Confidence Scorer"),
    ("ai_system.blocco_3_value_detector", "Blocco 3: Value Detector"),
    ("ai_system.blocco_4_kelly", "Blocco 4: Smart Kelly Optimizer"),
    ("ai_system.blocco_5_risk_manager", "Blocco 5: Risk Manager"),
    ("ai_system.blocco_6_odds_tracker", "Blocco 6: Odds Movement Tracker"),
    # Blocchi avanzati
    ("ai_system.blocco_7_bayesian_uncertainty", "Blocco 7: Bayesian Uncertainty"),
    ("ai_system.blocco_8_monte_carlo", "Blocco 8: Monte Carlo Simulator"),
    ("ai_system.blocco_9_anomaly_detection", "Blocco 9: Anomaly Detection"),
    ("ai_system.blocco_10_market_consistency", "Blocco 10: Market Consistency"),
    ("ai_system.blocco_11_adaptive_calibration", "Blocco 11: Adaptive Calibration"),
    ("ai_system.blocco_12_consensus_validator", "Blocco 12: Consensus Validator"),
    ("ai_system.blocco_13_arbitrage_detector", "Blocco 13: Arbitrage Detector"),
    ("ai_system.blocco_14_realtime_validation", "Blocco 14: Realtime Validation"),
    # Modelli
    ("ai_system.models.ensemble", "Ensemble Meta-Model"),
    ("ai_system.models.xgboost_predictor", "XGBoost Predictor"),
    ("ai_system.models.lstm_predictor", "LSTM Predictor"),
    ("ai_system.models.meta_learner", "Meta-Learner"),
    # Pipeline
    ("ai_system.pipeline", "Main Pipeline (Blocchi 0-6)"),
    ("ai_system.advanced_precision_pipeline", "Advanced Pipeline (Blocchi 7-14)"),
    # Moduli di supporto
    ("ai_system.llm_analyst", "LLM Sports Analyst"),
    ("ai_system.sentiment_analyzer", "Sentiment Analyzer"),
    ("ai_system.live_betting", "Live Betting Engine"),
    ("ai_system.live_monitor", "Live Monitor"),
    ("ai_system.telegram_notifier", "Telegram Notifier"),
    ("ai_system.backtesting", "Backtesting System"),
]


def _check_import(module_name: str) -> Tuple[bool, str]:
    """Prova ad importare un modulo e restituisce esito + errore eventuale."""
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - vogliamo il messaggio originale
        return False, f"{type(exc).__name__}: {exc}"


@pytest.mark.parametrize("module_name,description", MODULES_TO_TEST)
def test_module_imports(module_name: str, description: str) -> None:
    """Test parametrico per tutti i moduli dichiarati sopra."""
    success, error_message = _check_import(module_name)
    assert success, f"{description} import failed → {error_message}"


def main() -> int:
    """Entry point manuale, utile per esecuzioni rapide da CLI."""
    print("=" * 80)
    print("VERIFICA INTEGRAZIONE BLOCCHI IA")
    print("=" * 80)
    print()

    ok_modules = []
    for module_name, description in MODULES_TO_TEST:
        success, error_message = _check_import(module_name)
        symbol = "✅" if success else "❌"
        detail = "OK" if success else error_message
        print(f"{symbol} {description:50} {detail}")
        ok_modules.append(success)

    print("\n" + "=" * 80)
    total = len(ok_modules)
    passed = sum(ok_modules)
    print(f"📊 RIEPILOGO: {passed}/{total} moduli importati correttamente")

    if passed == total:
        print("✅ TUTTI I BLOCCHI IA SONO CORRETTAMENTE INTEGRATI!")
        return 0

    print("⚠️  Alcuni moduli hanno problemi di import. Vedi log sopra.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
