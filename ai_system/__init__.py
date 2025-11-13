"""
AI System for AsianOdds Prediction Platform
============================================

Sistema modulare di Intelligenza Artificiale per migliorare le predizioni
di betting attraverso 7 blocchi interconnessi:

BLOCCO 0: API Data Engine - Raccolta e caching intelligente dati real-time
BLOCCO 1: Probability Calibrator - Calibrazione probabilità con Neural Network
BLOCCO 2: Confidence Scorer - Valutazione affidabilità predizioni
BLOCCO 3: Value Detector - Identificazione true value vs trap bets
BLOCCO 4: Smart Kelly Optimizer - Ottimizzazione stake dinamica
BLOCCO 5: Risk Manager - Gestione rischio e filtri di sicurezza
BLOCCO 6: Odds Movement Tracker - Timing ottimale per scommesse

Autore: AsianOdds AI Team
Versione: 1.0.0
Data: 2025-11-13
"""

__version__ = "1.0.0"
__author__ = "AsianOdds AI Team"

# Import main components for easy access
from .pipeline import AIPipeline
from .config import AIConfig
from .news_sentiment import NewsSentimentMonitor
from .minor_league_data import MinorLeagueDataPipeline
from .chat_assistant import AIAssistantChat
from .odds_anomaly_detector import OddsAnomalyDetector

__all__ = [
    'AIPipeline',
    'AIConfig',
    'NewsSentimentMonitor',
    'MinorLeagueDataPipeline',
    'AIAssistantChat',
    'OddsAnomalyDetector',
]
