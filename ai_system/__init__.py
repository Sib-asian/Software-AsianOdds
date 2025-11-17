"""
AI System for AsianOdds Prediction Platform
============================================

Sistema modulare di Intelligenza Artificiale per migliorare le predizioni
di betting attraverso 7 blocchi interconnessi + 5 features avanzate:

BLOCCO 0: API Data Engine - Raccolta e caching intelligente dati real-time
BLOCCO 1: Probability Calibrator - Calibrazione probabilità con Neural Network
BLOCCO 2: Confidence Scorer - Valutazione affidabilità predizioni
BLOCCO 3: Value Detector - Identificazione true value vs trap bets
BLOCCO 4: Smart Kelly Optimizer - Ottimizzazione stake dinamica
BLOCCO 5: Risk Manager - Gestione rischio e filtri di sicurezza
BLOCCO 6: Odds Movement Tracker - Timing ottimale per scommesse

NUOVE FEATURES:
- Ensemble Meta-Model: Combina Dixon-Coles + XGBoost + LSTM
- LLM Sports Analyst: Spiegazioni AI in linguaggio naturale
- Sentiment Analyzer: Monitoring social media per insider info
- Live Betting Engine: Predizioni real-time durante le partite
- Historical Backtesting: Test strategie su dati storici
- Telegram Notifier: Notifiche automatiche opportunità
- Live Monitor: Monitoring continuo partite con alert

Autore: AsianOdds AI Team
Versione: 2.0.0
Data: 2025-11-13
"""

__version__ = "2.0.0"
__author__ = "AsianOdds AI Team"

# Import lightweight components eagerly
from .config import AIConfig

__all__ = [
    'AIConfig',
    'AIPipeline',
    'TelegramNotifier',
    'LiveMonitor',
]


def __getattr__(name):
    if name == "AIPipeline":
        from .pipeline import AIPipeline
        return AIPipeline
    if name == "TelegramNotifier":
        from .telegram_notifier import TelegramNotifier
        return TelegramNotifier
    if name == "LiveMonitor":
        from .live_monitor import LiveMonitor
        return LiveMonitor
    raise AttributeError(f"module 'ai_system' has no attribute '{name}'")
