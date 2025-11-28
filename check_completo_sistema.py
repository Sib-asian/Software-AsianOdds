#!/usr/bin/env python3
"""
Check Completo Sistema
======================

Verifica che tutto il sistema funzioni correttamente.
"""

import sys
import os
from pathlib import Path

# Aggiungi path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("🔍 CHECK COMPLETO SISTEMA AUTOMAZIONE 24/7")
print("=" * 70)

errors = []
warnings = []
success = []

# 1. Check Import Moduli Base
print("\n📦 1. VERIFICA IMPORT MODULI BASE")
print("-" * 70)

try:
    from ai_system.pipeline import AIPipeline
    from ai_system.config import AIConfig
    success.append("✅ AI Pipeline")
except Exception as e:
    errors.append(f"❌ AI Pipeline: {e}")

try:
    from ai_system.telegram_notifier import TelegramNotifier
    success.append("✅ Telegram Notifier")
except Exception as e:
    warnings.append(f"⚠️  Telegram Notifier: {e}")

try:
    from api_manager import APIManager
    success.append("✅ API Manager")
except Exception as e:
    errors.append(f"❌ API Manager: {e}")

# 2. Check Import Moduli Avanzati AI
print("\n🧠 2. VERIFICA SISTEMI AI AVANZATI")
print("-" * 70)

try:
    from ai_system.multi_model_consensus import MultiModelConsensus
    success.append("✅ Multi-Model Consensus")
except Exception as e:
    warnings.append(f"⚠️  Multi-Model Consensus: {e}")

try:
    from ai_system.intelligent_alert_system import IntelligentAlertSystem
    success.append("✅ Intelligent Alert System")
except Exception as e:
    warnings.append(f"⚠️  Intelligent Alert System: {e}")

try:
    from ai_system.pattern_analyzer_llm import PatternAnalyzerLLM
    success.append("✅ Pattern Analyzer LLM")
except Exception as e:
    warnings.append(f"⚠️  Pattern Analyzer LLM: {e}")

try:
    from ai_system.parameter_optimizer import ParameterOptimizer
    success.append("✅ Parameter Optimizer")
except Exception as e:
    warnings.append(f"⚠️  Parameter Optimizer: {e}")

# 3. Check Import Sistemi Avanzati 24/7
print("\n🚀 3. VERIFICA SISTEMI AVANZATI 24/7")
print("-" * 70)

try:
    from odds_monitor import OddsMonitor
    success.append("✅ Odds Monitor")
except Exception as e:
    errors.append(f"❌ Odds Monitor: {e}")

try:
    from result_tracker_auto import ResultTrackerAuto
    success.append("✅ Result Tracker Auto")
except Exception as e:
    errors.append(f"❌ Result Tracker Auto: {e}")

try:
    from pre_match_alerter import PreMatchAlerter
    success.append("✅ Pre-Match Alerter")
except Exception as e:
    errors.append(f"❌ Pre-Match Alerter: {e}")

try:
    from arbitrage_detector_auto import ArbitrageDetectorAuto
    success.append("✅ Arbitrage Detector")
except Exception as e:
    errors.append(f"❌ Arbitrage Detector: {e}")

try:
    from news_sentiment_analyzer import NewsSentimentAnalyzer
    success.append("✅ News Sentiment Analyzer")
except Exception as e:
    errors.append(f"❌ News Sentiment Analyzer: {e}")

# 4. Check Import Moduli Supporto
print("\n📊 4. VERIFICA MODULI SUPPORTO")
print("-" * 70)

try:
    from betting_results_tracker import BettingResultsTracker
    success.append("✅ Betting Results Tracker")
except Exception as e:
    warnings.append(f"⚠️  Betting Results Tracker: {e}")

try:
    from match_filters import MatchFilters
    success.append("✅ Match Filters")
except Exception as e:
    warnings.append(f"⚠️  Match Filters: {e}")

try:
    from bankroll_manager import BankrollManager
    success.append("✅ Bankroll Manager")
except Exception as e:
    warnings.append(f"⚠️  Bankroll Manager: {e}")

try:
    from automated_reports import AutomatedReports
    success.append("✅ Automated Reports")
except Exception as e:
    warnings.append(f"⚠️  Automated Reports: {e}")

# 5. Check Inizializzazione Automation24H
print("\n⚙️  5. VERIFICA INIZIALIZZAZIONE AUTOMATION24H")
print("-" * 70)

try:
    from automation_24h import Automation24H
    from dotenv import load_dotenv
    
    load_dotenv()
    
    auto = Automation24H()
    
    # Verifica componenti
    checks = {
        'AI Pipeline': auto.ai_pipeline is not None,
        'Consensus Analyzer': auto.consensus_analyzer is not None,
        'Alert System': auto.alert_system is not None,
        'Pattern Analyzer': auto.pattern_analyzer is not None,
        'Parameter Optimizer': auto.parameter_optimizer is not None,
        'Odds Monitor': auto.odds_monitor is not None,
        'Result Tracker Auto': auto.result_tracker_auto is not None,
        'Pre-Match Alerter': auto.pre_match_alerter is not None,
        'Arbitrage Detector': auto.arbitrage_detector is not None,
        'News Analyzer': auto.news_analyzer is not None,
        'API Manager': auto.api_manager is not None,
        'Results Tracker': auto.results_tracker is not None,
        'Match Filters': auto.match_filters is not None,
        'Bankroll Manager': auto.bankroll_manager is not None,
    }
    
    for name, status in checks.items():
        if status:
            success.append(f"✅ {name} inizializzato")
        else:
            warnings.append(f"⚠️  {name} non inizializzato")
    
    success.append("✅ Automation24H inizializzato correttamente")
    
except Exception as e:
    errors.append(f"❌ Automation24H inizializzazione: {e}")
    import traceback
    errors.append(f"   Traceback: {traceback.format_exc()}")

# 6. Check Configurazione
print("\n🔧 6. VERIFICA CONFIGURAZIONE")
print("-" * 70)

from dotenv import load_dotenv
load_dotenv()

config_checks = {
    'THEODDS_API_KEY': os.getenv('THEODDS_API_KEY'),
    'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
    'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
    'AUTOMATION_MIN_EV': os.getenv('AUTOMATION_MIN_EV'),
    'AUTOMATION_MIN_CONFIDENCE': os.getenv('AUTOMATION_MIN_CONFIDENCE'),
    'AUTOMATION_UPDATE_INTERVAL': os.getenv('AUTOMATION_UPDATE_INTERVAL'),
}

for key, value in config_checks.items():
    if value:
        success.append(f"✅ {key}: Configurato")
    else:
        warnings.append(f"⚠️  {key}: Non configurato")

# 7. Check File Importanti
print("\n📁 7. VERIFICA FILE IMPORTANTI")
print("-" * 70)

important_files = [
    'automation_24h.py',
    'automation_service_wrapper.py',
    'odds_monitor.py',
    'result_tracker_auto.py',
    'pre_match_alerter.py',
    'arbitrage_detector_auto.py',
    'news_sentiment_analyzer.py',
    '.env'
]

for file in important_files:
    if Path(file).exists():
        success.append(f"✅ {file}: Esiste")
    else:
        errors.append(f"❌ {file}: Non trovato")

# 8. Riepilogo
print("\n" + "=" * 70)
print("📊 RIEPILOGO CHECK")
print("=" * 70)

print(f"\n✅ Successi: {len(success)}")
for item in success:
    print(f"   {item}")

if warnings:
    print(f"\n⚠️  Warning: {len(warnings)}")
    for item in warnings:
        print(f"   {item}")

if errors:
    print(f"\n❌ Errori: {len(errors)}")
    for item in errors:
        print(f"   {item}")

print("\n" + "=" * 70)

if errors:
    print("❌ CI SONO ERRORI - Sistema potrebbe non funzionare correttamente")
    sys.exit(1)
elif warnings:
    print("⚠️  CI SONO WARNING - Sistema funziona ma alcune funzionalità potrebbero essere limitate")
    sys.exit(0)
else:
    print("✅ TUTTO OK - Sistema completamente funzionante!")
    sys.exit(0)

