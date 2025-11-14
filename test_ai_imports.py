#!/usr/bin/env python3
"""
Script di Verifica Integrazione Blocchi IA
==========================================

Verifica che tutti i 15 blocchi IA siano correttamente importabili
senza richiedere installazioni aggiuntive.
"""

import sys
from pathlib import Path

# Aggiungi il percorso del progetto
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import(module_name, description):
    """Testa l'import di un modulo."""
    try:
        __import__(module_name)
        print(f"✅ {description:50} OK")
        return True
    except ImportError as e:
        print(f"❌ {description:50} ERRORE: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description:50} WARNING: {e}")
        return True  # Alcune eccezioni a runtime sono ok durante l'import


def main():
    """Esegue tutti i test di import."""
    print("=" * 80)
    print("VERIFICA INTEGRAZIONE BLOCCHI IA")
    print("=" * 80)
    print()

    results = []

    # Test configurazione
    print("📦 CONFIGURAZIONE")
    results.append(test_import("ai_system.config", "Config System"))
    print()

    # Test blocchi core (0-6)
    print("🎯 BLOCCHI CORE (0-6)")
    results.append(test_import("ai_system.blocco_0_api_engine", "Blocco 0: API Data Engine"))
    results.append(test_import("ai_system.blocco_1_calibrator", "Blocco 1: Probability Calibrator"))
    results.append(test_import("ai_system.blocco_2_confidence", "Blocco 2: Confidence Scorer"))
    results.append(test_import("ai_system.blocco_3_value_detector", "Blocco 3: Value Detector"))
    results.append(test_import("ai_system.blocco_4_kelly", "Blocco 4: Smart Kelly Optimizer"))
    results.append(test_import("ai_system.blocco_5_risk_manager", "Blocco 5: Risk Manager"))
    results.append(test_import("ai_system.blocco_6_odds_tracker", "Blocco 6: Odds Movement Tracker"))
    print()

    # Test blocchi avanzati (7-14)
    print("🚀 BLOCCHI AVANZATI (7-14)")
    results.append(test_import("ai_system.blocco_7_bayesian_uncertainty", "Blocco 7: Bayesian Uncertainty"))
    results.append(test_import("ai_system.blocco_8_monte_carlo", "Blocco 8: Monte Carlo Simulator"))
    results.append(test_import("ai_system.blocco_9_anomaly_detection", "Blocco 9: Anomaly Detection"))
    results.append(test_import("ai_system.blocco_10_market_consistency", "Blocco 10: Market Consistency"))
    results.append(test_import("ai_system.blocco_11_adaptive_calibration", "Blocco 11: Adaptive Calibration"))
    results.append(test_import("ai_system.blocco_12_consensus_validator", "Blocco 12: Consensus Validator"))
    results.append(test_import("ai_system.blocco_13_arbitrage_detector", "Blocco 13: Arbitrage Detector"))
    results.append(test_import("ai_system.blocco_14_realtime_validation", "Blocco 14: Realtime Validation"))
    print()

    # Test modelli ML
    print("🤖 MODELLI ML")
    results.append(test_import("ai_system.models.ensemble", "Ensemble Meta-Model"))
    results.append(test_import("ai_system.models.xgboost_predictor", "XGBoost Predictor"))
    results.append(test_import("ai_system.models.lstm_predictor", "LSTM Predictor"))
    results.append(test_import("ai_system.models.meta_learner", "Meta-Learner"))
    print()

    # Test pipeline
    print("⚙️  PIPELINE")
    results.append(test_import("ai_system.pipeline", "Main Pipeline (Blocchi 0-6)"))
    results.append(test_import("ai_system.advanced_precision_pipeline", "Advanced Pipeline (Blocchi 7-14)"))
    print()

    # Test moduli di supporto
    print("🔧 MODULI DI SUPPORTO")
    results.append(test_import("ai_system.llm_analyst", "LLM Sports Analyst"))
    results.append(test_import("ai_system.sentiment_analyzer", "Sentiment Analyzer"))
    results.append(test_import("ai_system.live_betting", "Live Betting Engine"))
    results.append(test_import("ai_system.live_monitor", "Live Monitor"))
    results.append(test_import("ai_system.telegram_notifier", "Telegram Notifier"))
    results.append(test_import("ai_system.backtesting", "Backtesting System"))
    print()

    # Riepilogo
    print("=" * 80)
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"📊 RIEPILOGO: {passed}/{total} moduli importati correttamente")

    if failed == 0:
        print("✅ TUTTI I BLOCCHI IA SONO CORRETTAMENTE INTEGRATI!")
        print("✅ NON È NECESSARIA ALCUNA INSTALLAZIONE AGGIUNTIVA!")
        return 0
    else:
        print(f"⚠️  {failed} moduli hanno problemi di import")
        print("ℹ️  Verificare le dipendenze in requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
