#!/usr/bin/env python3
"""
Sistema Automazione 24/7
========================

Sistema completamente autonomo che:
1. Monitora partite 24/7
2. Analizza solo VALUE BET reali (non consigli basati su score)
3. Notifica Telegram solo per vere opportunità
4. Aggiorna dati automaticamente
5. Gestisce API quota intelligente

Usage:
    python automation_24h.py --config config.json
"""

import asyncio
import logging
import json
import time
import signal
import sys
import os
import sqlite3
import math
from decimal import Decimal, InvalidOperation, getcontext
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
import argparse

# Carica variabili d'ambiente da .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging using centralized configuration
from logging_setup import init_logging
init_logging()
getcontext().prec = 12
logger = logging.getLogger(__name__)

# Import sistema esistente
try:
    from ai_system.pipeline import AIPipeline
    from ai_system.config import AIConfig
    AI_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    AI_SYSTEM_AVAILABLE = False

# Import componenti essenziali (sempre necessari)
try:
    from ai_system.telegram_notifier import TelegramNotifier
except ImportError as e:
    logger.error(f"❌ TelegramNotifier not found: {e}")
    TelegramNotifier = None

try:
    from api_manager import APIManager
except ImportError as e:
    logger.error(f"❌ APIManager not found: {e}")
    APIManager = None

# Import nuovi moduli
try:
    from betting_results_tracker import BettingResultsTracker
    from match_filters import MatchFilters
    from bankroll_manager import BankrollManager
    from automated_reports import AutomatedReports
    NEW_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  Nuovi moduli non disponibili: {e}")
    NEW_MODULES_AVAILABLE = False

# Import live betting performance tracking
try:
    from live_betting_performance_tracker import LiveBettingPerformanceTracker
    from live_betting_reports import LiveBettingReports
    LIVE_BETTING_TRACKING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  Live betting tracking non disponibile: {e}")
    LIVE_BETTING_TRACKING_AVAILABLE = False

# Import nuovi sistemi AI avanzati
try:
    from ai_system.multi_model_consensus import MultiModelConsensus
    from ai_system.intelligent_alert_system import IntelligentAlertSystem, AlertLevel
    from ai_system.pattern_analyzer_llm import PatternAnalyzerLLM
    from ai_system.parameter_optimizer import ParameterOptimizer
    ADVANCED_AI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  Sistemi AI avanzati non disponibili: {e}")
    ADVANCED_AI_AVAILABLE = False

# Import nuovi sistemi avanzati 24/7
try:
    from odds_monitor import OddsMonitor
    from result_tracker_auto import ResultTrackerAuto
    from pre_match_alerter import PreMatchAlerter
    from arbitrage_detector_auto import ArbitrageDetectorAuto
    from news_sentiment_analyzer import NewsSentimentAnalyzer
    ADVANCED_SYSTEMS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  Sistemi avanzati 24/7 non disponibili: {e}")
    ADVANCED_SYSTEMS_AVAILABLE = False

# Import MultiSourceMatchFinder per trovare partite da API-SPORTS
try:
    from multi_source_match_finder import MultiSourceMatchFinder
    MULTI_SOURCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  MultiSourceMatchFinder non disponibile: {e}")
    MULTI_SOURCE_AVAILABLE = False

# Import LiveBettingAdvisor per analisi partite live
try:
    from live_betting_advisor import LiveBettingAdvisor
    LIVE_BETTING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  LiveBettingAdvisor non disponibile: {e}")
    LIVE_BETTING_AVAILABLE = False


class Automation24H:
    """
    Sistema automazione 24/7 per betting intelligente.
    
    Caratteristiche:
    - Analizza solo VALUE BET reali (EV > soglia)
    - Non consiglia basandosi su score (es: "1-0 quindi gioca 1")
    - Notifica solo opportunità con alta confidence
    - Aggiorna dati automaticamente
    - Gestisce API quota intelligente
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        min_ev: float = 8.0,  # EV minimo 8%
        min_confidence: float = 70.0,  # Confidence minima 70%
        update_interval: int = 600,  # 10 minuti (600 secondi)
        api_budget_per_day: int = 7500,  # Piano Pro: 7500 chiamate/giorno
        max_notifications_per_cycle: int = 1  # Max notifiche per ciclo (solo la migliore in assoluto)
    ):
        self.config_path = config_path
        self.min_ev = min_ev
        self.min_confidence = min_confidence
        self.update_interval = update_interval
        self.api_budget_per_day = api_budget_per_day
        self.max_notifications_per_cycle = max_notifications_per_cycle
        
        # Stato
        self.running = False
        self.monitored_matches: Dict[str, Dict] = {}
        self.notified_opportunities: Set[str] = set()  # Evita duplicati
        self.notified_opportunities_timestamps: Dict[str, datetime] = {}  # Timestamp delle notifiche
        self.notified_matches_timestamps: Dict[str, datetime] = {}  # Timestamp per partita (max 1 notifica ogni 30 min per partita)
        self.last_global_notification_time: Optional[datetime] = None  # 🆕 Limite globale 10 minuti tra qualsiasi notifica
        self._load_last_global_notification_time()  # 🆕 Carica timestamp persistente
        # 🔧 OPZIONE 4: Tracking mercati già suggeriti per partita (per penalizzazione/bonus)
        self.match_markets_history: Dict[str, List[Dict[str, Any]]] = {}  # match_id -> lista di {market, timestamp}
        # 🆕 Cache Quality Score per evitare doppio calcolo
        self.quality_score_cache: Dict[str, Any] = {}  # opp_key -> QualityScore
        # 🆕 Signal Quality Gate (inizializzato lazy)
        self.signal_quality_gate = None
        # 🆕 Signal Quality Learner per apprendimento automatico
        self.signal_quality_learner = None
        self.last_learning_update = None
        # 🆕 Tracking progresso apprendimento (per notifiche)
        self.last_progress_notification = {}  # {threshold: datetime}
        self.last_signal_count = 0  # Ultimo conteggio segnali per rilevare nuovi
        self.start_time = datetime.now()  # Per calcolo stima giorni
        self.api_usage_today = 0
        self.last_api_reset = datetime.now().date()
        self.last_daily_report = datetime.now().date()
        self.last_weekly_report = datetime.now() - timedelta(days=7)
        
        # Inizializza componenti
        self._init_components(telegram_token, telegram_chat_id)
        
        # 🆕 Inizializza MultiSourceMatchFinder per trovare partite da API-SPORTS
        if MULTI_SOURCE_AVAILABLE:
            try:
                self.multi_source_finder = MultiSourceMatchFinder()
                logger.info("✅ MultiSourceMatchFinder initialized")
            except Exception as e:
                logger.warning(f"⚠️  MultiSourceMatchFinder error: {e}")
                self.multi_source_finder = None
        else:
            self.multi_source_finder = None
        
        # 🆕 Inizializza LiveBettingAdvisor per analisi partite live
        if LIVE_BETTING_AVAILABLE:
            try:
                # Per partite live, abbassa soglia EV (odds spesso basse)
                # Con confidence 72% e odds 1.3, EV = (0.72 * 1.3 - 1) * 100 = -6.4%
                # Con confidence 75% e odds 1.4, EV = (0.75 * 1.4 - 1) * 100 = 5%
                # 🔧 ABBASSATO: Per live betting, EV minimo più basso (6% invece di 5%)
                # 🎯 RIMOSSO: Non passa più min_confidence e min_ev al LiveBettingAdvisor
                # L'utente vuole calcolare confidence ed EV senza soglie minime
                # 🔧 NUOVO: Passa performance tracker per soglie dinamiche
                live_tracker = self.live_performance_tracker if hasattr(self, 'live_performance_tracker') else None
                self.live_betting_advisor = LiveBettingAdvisor(
                    notifier=self.notifier,
                    min_confidence=0.0,  # 🎯 Nessuna soglia minima
                    min_ev=0.0,  # 🎯 Nessuna soglia minima
                    performance_tracker=live_tracker  # 🔧 NUOVO: Passa tracker
                )
                logger.info(f"   LiveBettingAdvisor: nessuna soglia minima (min_confidence=0%, min_ev=0%)")
                logger.info("✅ LiveBettingAdvisor initialized")
            except Exception as e:
                logger.warning(f"⚠️  LiveBettingAdvisor error: {e}")
                self.live_betting_advisor = None
        else:
            self.live_betting_advisor = None
        
        logger.info("✅ Automation24H initialized")
        logger.info(f"   Min EV: {min_ev}%")
        logger.info(f"   Min Confidence: {min_confidence}%")
        logger.info(f"   Update Interval: {update_interval}s")
    
    def _init_components(self, telegram_token: Optional[str], telegram_chat_id: Optional[str]):
        """Inizializza componenti sistema"""
        
        # AI Pipeline
        if AI_SYSTEM_AVAILABLE:
            ai_config = AIConfig()
            ai_config.use_ensemble = True
            ai_config.llm_playbook_enabled = False  # Disabilita per risparmiare API
            self.ai_pipeline = AIPipeline(ai_config)
            logger.info("✅ AI Pipeline initialized")
        else:
            self.ai_pipeline = None
            logger.warning("⚠️  AI Pipeline not available")
        
        # Sistemi AI avanzati
        if ADVANCED_AI_AVAILABLE:
            self.consensus_analyzer = MultiModelConsensus()
            self.alert_system = IntelligentAlertSystem()
            
            # Pattern Analyzer con LLM (se disponibile)
            llm_analyst = None
            if self.ai_pipeline and hasattr(self.ai_pipeline, 'llm_analyst'):
                llm_analyst = self.ai_pipeline.llm_analyst
            self.pattern_analyzer = PatternAnalyzerLLM(llm_analyst=llm_analyst)
            
            self.parameter_optimizer = ParameterOptimizer()
            logger.info("✅ Advanced AI systems initialized")
        else:
            self.consensus_analyzer = None
            self.alert_system = None
            self.pattern_analyzer = None
            self.parameter_optimizer = None
            logger.warning("⚠️  Advanced AI systems not available")
        
        # API Manager (deve essere prima di tutto)
        if APIManager is None:
            logger.error("❌ APIManager not available")
            self.api_manager = None
        else:
            try:
                self.api_manager = APIManager()
                logger.info("✅ API Manager initialized")
            except Exception as e:
                logger.error(f"❌ API Manager error: {e}")
                self.api_manager = None
        
        # Telegram Notifier
        if TelegramNotifier is None:
            logger.error("❌ TelegramNotifier not available")
            self.notifier = None
        elif telegram_token and telegram_chat_id:
            try:
                self.notifier = TelegramNotifier(
                    bot_token=telegram_token,
                    chat_id=telegram_chat_id,
                    min_ev=0.0,  # 🎯 Nessuna soglia minima
                    min_confidence=0.0,  # 🎯 Nessuna soglia minima
                    rate_limit_seconds=3,
                    live_alerts_enabled=True  # ✅ Abilita notifiche live
                )
                logger.info("✅ Telegram Notifier initialized")
            except Exception as e:
                logger.error(f"❌ Telegram Notifier error: {e}")
                self.notifier = None
        else:
            self.notifier = None
            logger.warning("⚠️  Telegram not configured")
        
        # Sistemi avanzati 24/7 (dopo notifier e api_manager)
        if ADVANCED_SYSTEMS_AVAILABLE:
            self.odds_monitor = OddsMonitor()
            self.result_tracker_auto = ResultTrackerAuto(api_manager=self.api_manager)
            self.pre_match_alerter = PreMatchAlerter(notifier=self.notifier)
            self.arbitrage_detector = ArbitrageDetectorAuto(min_profit_pct=1.0)
            
            # News analyzer (richiede sentiment analyzer se disponibile)
            sentiment_analyzer = None
            if self.ai_pipeline and hasattr(self.ai_pipeline, 'sentiment_analyzer'):
                sentiment_analyzer = self.ai_pipeline.sentiment_analyzer
            self.news_analyzer = NewsSentimentAnalyzer(
                newsapi_key=os.getenv('NEWSAPI_KEY'),
                sentiment_analyzer=sentiment_analyzer
            )
            logger.info("✅ Advanced 24/7 systems initialized")
        else:
            self.odds_monitor = None
            self.result_tracker_auto = None
            self.pre_match_alerter = None
            self.arbitrage_detector = None
            self.news_analyzer = None
            logger.warning("⚠️  Advanced 24/7 systems not available")
        
        # Nuovi moduli (se disponibili)
        if NEW_MODULES_AVAILABLE:
            try:
                self.results_tracker = BettingResultsTracker()
                logger.info("✅ Results Tracker initialized")
            except Exception as e:
                logger.warning(f"⚠️  Results Tracker error: {e}")
                self.results_tracker = None
            
            # 🔧 NUOVO: Live betting performance tracker
            if LIVE_BETTING_TRACKING_AVAILABLE:
                try:
                    self.live_performance_tracker = LiveBettingPerformanceTracker()
                    self.live_betting_reports = LiveBettingReports(
                        tracker=self.live_performance_tracker,
                        notifier=self.notifier
                    )
                    logger.info("✅ Live Betting Performance Tracker initialized")
                except Exception as e:
                    logger.warning(f"⚠️  Live Betting Tracker error: {e}")
                    self.live_performance_tracker = None
                    self.live_betting_reports = None
            else:
                self.live_performance_tracker = None
                self.live_betting_reports = None
            
            # 🆕 Inizializza Signal Quality Learner per apprendimento automatico
            try:
                from ai_system.signal_quality_learner import SignalQualityLearner
                self.signal_quality_learner = SignalQualityLearner()
                logger.info("✅ Signal Quality Learner inizializzato per apprendimento automatico")
            except Exception as e:
                logger.warning(f"⚠️  Signal Quality Learner non disponibile: {e}")
                self.signal_quality_learner = None
            
            try:
                self.match_filters = MatchFilters()
                logger.info("✅ Match Filters initialized")
            except Exception as e:
                logger.warning(f"⚠️  Match Filters error: {e}")
                self.match_filters = None
            
            try:
                self.bankroll_manager = BankrollManager(initial_bankroll=1000.0)
                logger.info("✅ Bankroll Manager initialized")
            except Exception as e:
                logger.warning(f"⚠️  Bankroll Manager error: {e}")
                self.bankroll_manager = None
            
            # Automated Reports (richiede notifier)
            if self.notifier and self.results_tracker:
                try:
                    self.automated_reports = AutomatedReports(self.notifier, self.results_tracker)
                    logger.info("✅ Automated Reports initialized")
                except Exception as e:
                    logger.warning(f"⚠️  Automated Reports error: {e}")
                    self.automated_reports = None
            else:
                self.automated_reports = None
        else:
            self.results_tracker = None
            self.match_filters = None
            self.bankroll_manager = None
            self.automated_reports = None
    
    def start(self, single_run: bool = False):
        """
        Avvia sistema 24/7
        
        Args:
            single_run: Se True, esegue un solo ciclo (utile per cron jobs)
        """
        logger.info("🚀 Starting Automation24H system...")
        self.running = True
        
        # Gestione shutdown graceful (solo se non single_run)
        if not single_run:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Loop principale
            while self.running:
                # 🔧 NUOVO: Controlla se il servizio è attivo prima di ogni ciclo
                if not self._is_service_active():
                    logger.warning("⚠️  Servizio non attivo, interrompendo loop")
                    break
                
                try:
                    self._run_cycle()
                except Exception as e:
                    logger.error(f"❌ Error in cycle: {e}", exc_info=True)
                    if single_run:
                        break  # In single_run, esci dopo errore
                    time.sleep(60)  # Attendi 1 minuto prima di riprovare
                
                # Se single_run, esci dopo primo ciclo
                if single_run:
                    logger.info("✅ Single run completed")
                    break
                
                # Attendi prima del prossimo ciclo (con controllo running per shutdown immediato)
                # Suddividi il sleep in piccoli intervalli per rispondere rapidamente ai segnali
                sleep_interval = min(60, self.update_interval)  # Max 60 secondi per controllo
                elapsed = 0
                while elapsed < self.update_interval and self.running:
                    # 🔧 NUOVO: Controlla servizio attivo anche durante il sleep
                    if not self._is_service_active():
                        logger.warning("⚠️  Servizio non attivo durante sleep, interrompendo")
                        self.running = False
                        break
                    time.sleep(sleep_interval)
                    elapsed += sleep_interval
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested")
        finally:
            self.stop()
    
    def _run_cycle(self):
        """Esegue un ciclo di analisi"""
        # 🔧 NUOVO: Controlla se il servizio è attivo prima di iniziare
        if not self._is_service_active():
            logger.warning("⚠️  Servizio non attivo, interrompendo ciclo")
            self.running = False
            return
        
        logger.info("🔄 Running analysis cycle...")
        
        # Reset API usage se nuovo giorno
        self._reset_api_usage_if_needed()
        
        # 🆕 NUOVO: Monitoraggio quote real-time
        if self.odds_monitor:
            try:
                self._monitor_odds_movements()
            except Exception as e:
                logger.debug(f"⚠️  Odds monitoring error: {e}")
        
        # 🆕 NUOVO: Tracking risultati automatico
        if self.result_tracker_auto:
            try:
                self._update_match_results()
            except Exception as e:
                logger.debug(f"⚠️  Result tracking error: {e}")
        
        # 🆕 NUOVO: Alert pre-partita
        if self.pre_match_alerter:
            try:
                self.pre_match_alerter.check_and_send_alerts()
            except Exception as e:
                logger.debug(f"⚠️  Pre-match alert error: {e}")
        
        # 🆕 NUOVO: Analisi news (ogni 30 minuti)
        if self.news_analyzer:
            if not hasattr(self, 'last_news_check'):
                self.last_news_check = datetime.now() - timedelta(minutes=31)
            time_since_news = (datetime.now() - self.last_news_check).total_seconds()
            if time_since_news > 1800:  # 30 minuti
                try:
                    self._check_news_alerts()
                    self.last_news_check = datetime.now()
                except Exception as e:
                    logger.debug(f"⚠️  News check error: {e}")
        
        # 🆕 Controlla progresso apprendimento e invia notifiche
        if self.signal_quality_learner:
            try:
                # Conta segnali con risultati
                conn = sqlite3.connect(self.signal_quality_learner.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM signal_records WHERE was_correct IS NOT NULL")
                signals_with_results = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM signal_records WHERE was_correct IS NULL")
                signals_pending = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM signal_records")
                total_signals = cursor.fetchone()[0]
                conn.close()
                
                min_samples = 50
                progress_percent = (signals_with_results / min_samples * 100) if min_samples > 0 else 0
                
                # 🆕 Soglie di notifica ogni 5 segnali (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
                thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                current_threshold = None
                for threshold in thresholds:
                    if signals_with_results >= threshold:
                        current_threshold = threshold
                
                # 🆕 Notifica se nuovo segnale registrato (con statistiche database)
                if total_signals > self.last_signal_count and self.notifier:
                    new_signals = total_signals - self.last_signal_count
                    try:
                        # Recupera statistiche complete
                        conn = sqlite3.connect(self.signal_quality_learner.db_path)
                        cursor = conn.cursor()
                        
                        # Conta approvati e bloccati
                        cursor.execute("SELECT COUNT(*) FROM signal_records WHERE was_approved = 1")
                        approved_count = cursor.fetchone()[0]
                        cursor.execute("SELECT COUNT(*) FROM signal_records WHERE was_approved = 0")
                        blocked_count = cursor.fetchone()[0]
                        
                        # Recupera ultimi segnali registrati
                        cursor.execute("""
                            SELECT match_id, market, was_approved, quality_score, notified_at
                            FROM signal_records
                            ORDER BY notified_at DESC
                            LIMIT ?
                        """, (min(new_signals, 3),))  # Max 3 segnali per notifica
                        recent_signals = cursor.fetchall()
                        conn.close()
                        
                        # Notifica con statistiche complete
                        message = (
                            f"📊 <b>IA: Statistiche Database</b>\n\n"
                            f"📈 Totale segnali: <b>{total_signals}</b>\n"
                            f"✅ Approvati: {approved_count}\n"
                            f"❌ Bloccati: {blocked_count}\n"
                            f"⏳ In attesa risultati: {signals_pending}\n"
                            f"✅ Con risultati: {signals_with_results}/50\n\n"
                        )
                        
                        if recent_signals:
                            message += f"📝 Ultimi {len(recent_signals)} segnali:\n"
                            for signal in recent_signals:
                                match_id, market, was_approved, quality_score, notified_at = signal
                                status = "✅" if was_approved else "❌"
                                message += f"{status} {match_id[:20]}/{market} (QS: {quality_score:.1f})\n"
                        
                        self.notifier._send_message(message, parse_mode="HTML")
                        logger.info(f"📊 Notifica statistiche database: {total_signals} totali ({approved_count} approvati, {blocked_count} bloccati)")
                    except Exception as e:
                        logger.debug(f"⚠️  Errore notifica statistiche database: {e}")
                    
                    self.last_signal_count = total_signals
                
                # Notifica se raggiunta nuova soglia (max 1 notifica ogni 1 ora per soglia)
                if current_threshold and self.notifier:
                    threshold_key = f"{current_threshold}%"
                    last_notif = self.last_progress_notification.get(threshold_key)
                    should_notify = False
                    
                    if not last_notif:
                        should_notify = True
                    else:
                        time_since = (datetime.now() - last_notif).total_seconds()
                        if time_since > 3600:  # 1 ora (ridotto per notifiche più frequenti)
                            should_notify = True
                    
                    if should_notify:
                        try:
                            # Calcola giorni stimati per raggiungere 50 (basato su media giornaliera)
                            days_estimate = ""
                            if signals_with_results > 0 and total_signals > 0:
                                # Stima basata su segnali totali e tempo
                                avg_per_day = signals_with_results / max(1, (datetime.now() - self.start_time).days) if hasattr(self, 'start_time') else signals_with_results
                                if avg_per_day > 0:
                                    remaining = min_samples - signals_with_results
                                    days_needed = remaining / avg_per_day
                                    days_estimate = f"\n⏱️ Stima: ~{days_needed:.1f} giorni per raggiungere 50"
                            
                            message = (
                                f"📊 <b>IA: Progresso Apprendimento</b>\n\n"
                                f"🎯 Soglia raggiunta: <b>{current_threshold}/{min_samples}</b>\n"
                                f"✅ Segnali con risultati: <b>{signals_with_results}/{min_samples}</b> ({progress_percent:.1f}%)\n"
                                f"⏳ Segnali in attesa: {signals_pending}\n"
                                f"📈 Totale tracciati: {total_signals}\n"
                                f"{days_estimate}\n\n"
                            )
                            
                            # Aggiungi barra progresso visiva
                            bar_length = 20
                            filled = int(progress_percent / 100 * bar_length)
                            bar = "█" * filled + "░" * (bar_length - filled)
                            message += f"<code>{bar}</code> {progress_percent:.0f}%"
                            
                            self.notifier._send_message(message, parse_mode="HTML")
                            self.last_progress_notification[threshold_key] = datetime.now()
                            logger.info(f"📊 Notifica progresso: {signals_with_results}/{min_samples} ({progress_percent:.1f}%)")
                        except Exception as e:
                            logger.debug(f"⚠️  Errore notifica progresso: {e}")
            except Exception as e:
                logger.debug(f"⚠️  Errore controllo progresso: {e}")
        
        # 🆕 Apprendimento automatico Signal Quality Gate (ogni 6 ore)
        if self.signal_quality_learner:
            if not self.last_learning_update:
                self.last_learning_update = datetime.now() - timedelta(hours=7)
            time_since_learning = (datetime.now() - self.last_learning_update).total_seconds()
            if time_since_learning > 21600:  # 6 ore
                try:
                    logger.info("🧠 Eseguendo apprendimento automatico Signal Quality Gate...")
                    
                    # 🆕 Notifica inizio apprendimento
                    if self.notifier:
                        try:
                            self.notifier._send_message(
                                "🧠 <b>IA: Apprendimento Automatico</b>\n\n"
                                "🔄 Inizio apprendimento Signal Quality Gate...\n"
                                "📊 Analizzando risultati segnali precedenti...",
                                parse_mode="HTML"
                            )
                        except Exception as e:
                            logger.debug(f"⚠️  Errore notifica inizio apprendimento: {e}")
                    
                    results = self.signal_quality_learner.learn_from_results(min_samples=50)
                    if results.get('status') == 'success':
                        logger.info(
                            f"✅ Apprendimento completato: "
                            f"Precision={results['precision']:.2%}, "
                            f"Recall={results['recall']:.2%}, "
                            f"Accuracy={results['accuracy']:.2%} | "
                            f"Nuovi pesi: Context={results['weights']['context']:.2%}, "
                            f"Data={results['weights']['data_quality']:.2%}, "
                            f"Logic={results['weights']['logic']:.2%}, "
                            f"Timing={results['weights']['timing']:.2%} | "
                            f"Nuova soglia: {results['min_quality_score']:.1f}"
                        )
                        
                        # 🆕 Notifica completamento apprendimento con risultati
                        if self.notifier:
                            try:
                                message = (
                                    "✅ <b>IA: Apprendimento Completato</b>\n\n"
                                    f"📊 <b>Metriche:</b>\n"
                                    f"   • Precision: {results['precision']:.1%}\n"
                                    f"   • Recall: {results['recall']:.1%}\n"
                                    f"   • Accuracy: {results['accuracy']:.1%}\n\n"
                                    f"⚖️ <b>Nuovi Pesi Quality Score:</b>\n"
                                    f"   • Context: {results['weights']['context']:.1%}\n"
                                    f"   • Data Quality: {results['weights']['data_quality']:.1%}\n"
                                    f"   • Logic: {results['weights']['logic']:.1%}\n"
                                    f"   • Timing: {results['weights']['timing']:.1%}\n\n"
                                    f"🎯 <b>Nuova Soglia Minima:</b> {results['min_quality_score']:.1f}/100\n\n"
                                    f"📈 Sistema aggiornato e pronto!"
                                )
                                self.notifier._send_message(message, parse_mode="HTML")
                            except Exception as e:
                                logger.debug(f"⚠️  Errore notifica completamento apprendimento: {e}")
                        
                        # Ricarica Signal Quality Gate con nuovi parametri
                        if hasattr(self, 'signal_quality_gate'):
                            from ai_system.signal_quality_scorer import SignalQualityGate
                            self.signal_quality_gate = SignalQualityGate(
                                ai_pipeline=self.ai_pipeline,
                                min_quality_score=results['min_quality_score'],
                                learner=self.signal_quality_learner
                            )
                    elif results.get('status') == 'insufficient_samples':
                        logger.info(f"ℹ️  Campioni insufficienti per apprendere ({results['samples']} < {results['min_samples']})")
                        
                        # 🆕 Notifica campioni insufficienti
                        if self.notifier:
                            try:
                                self.notifier._send_message(
                                    f"⚠️ <b>IA: Apprendimento Posticipato</b>\n\n"
                                    f"📊 Campioni insufficienti: {results['samples']}/{results['min_samples']}\n"
                                    f"⏳ Attendo più dati per apprendere...",
                                    parse_mode="HTML"
                                )
                            except Exception as e:
                                logger.debug(f"⚠️  Errore notifica campioni insufficienti: {e}")
                    
                    self.last_learning_update = datetime.now()
                except Exception as e:
                    logger.error(f"❌ Errore apprendimento automatico: {e}")
                    
                    # 🆕 Notifica errore apprendimento
                    if self.notifier:
                        try:
                            self.notifier._send_message(
                                f"❌ <b>IA: Errore Apprendimento</b>\n\n"
                                f"⚠️ Errore durante l'apprendimento automatico:\n"
                                f"<code>{str(e)[:200]}</code>",
                                parse_mode="HTML"
                            )
                        except:
                            pass
        
        # 1. Ottieni partite da monitorare
        all_matches = self._get_matches_to_monitor()
        logger.info(f"   Found {len(all_matches)} total matches")
        
        # 🆕 FILTRO: Analizza SOLO partite LIVE (no pre-match)
        matches = [m for m in all_matches if m.get('is_live', False)]
        pre_match_count = len(all_matches) - len(matches)
        
        if pre_match_count > 0:
            logger.info(f"   ⏭️  Saltate {pre_match_count} partite pre-match (solo live betting attivo)")
        
        logger.info(f"   Found {len(matches)} LIVE matches to monitor")
        
        if not matches:
            logger.info("   No LIVE matches to monitor, skipping cycle")
            return
        
        # 🔧 RIMOSSO: Filtri match_filters - analizziamo tutte le partite LIVE con statistiche e quote
        # Le partite vengono già filtrate per avere statistiche e quote disponibili in _fetch_matches_with_odds_from_api_football
        
        if not matches:
            logger.info("   No LIVE matches to monitor, skipping cycle")
            return
        
        # 🕵️  Controllo precisione quote prima di analizzare
        try:
            self._run_odds_precision_watchdog(matches)
        except Exception as e:
            logger.debug(f"⚠️  Odds precision watchdog error: {e}")
        
        # 2. Analizza ogni partita e raccogli tutte le opportunità
        all_opportunities = []  # Raccogli tutte le opportunità per selezionare le migliori
        opportunities_found = 0
        matches_analyzed = 0
        matches_with_opportunities = 0
        matches_without_opportunities = 0
        
        # 🔧 LOG VISIBILE: Inizio ciclo analisi
        logger.info("=" * 80)
        logger.info(f"📊 CICLO ANALISI LIVE BETTING - {len(matches)} partite da analizzare")
        logger.info("=" * 80)
        for match in matches:
            try:
                matches_analyzed += 1
                match_name = f"{match.get('home', '?')} vs {match.get('away', '?')}"
                
                # 🔧 FIX: Disabilitato arbitraggio per partite LIVE (l'utente vuole solo live betting)
                # if self.arbitrage_detector:
                #     self._check_arbitrage(match)
                
                # 🆕 SOLO LIVE BETTING: Analizza solo partite live
                if not self.live_betting_advisor:
                    logger.warning("⚠️  LiveBettingAdvisor not available, skipping live match")
                    continue
                
                # 🔧 DEBUG: Verifica dati partita prima di analizzare
                has_stats = bool(match.get('home_total_shots') is not None or match.get('home_shots_on_target') is not None)
                has_odds = bool(match.get('odds_1') or match.get('over_0.5') or match.get('all_odds'))
                score = f"{match.get('score_home', 0)}-{match.get('score_away', 0)}"
                minute = match.get('minute', 0)
                logger.info(f"🔍 Analizzando {match_name}: score={score}, min={minute}', stats={'✅' if has_stats else '❌'}, odds={'✅' if has_odds else '❌'}")
                
                opportunities = self._analyze_live_match(match)
                if opportunities:
                    matches_with_opportunities += 1
                    logger.info(f"📊 {match_name}: trovate {len(opportunities)} opportunità")
                    for opp in opportunities:
                        if not opp:
                            continue

                        # 🔧 FIX: estrai sempre l'oggetto opportunità reale (anche se annidato in un dict)
                        live_opp = opp.get('live_opportunity', opp) if isinstance(opp, dict) else opp

                        # 🔧 FIX: supporta sia LiveBettingOpportunity che dict
                        if isinstance(live_opp, dict):
                            market = live_opp.get('market', 'unknown')
                            ev = live_opp.get('ev', 0.0)
                            conf = live_opp.get('confidence', 0.0)
                            quality = live_opp.get('signal_quality_score', 0.0)
                        else:
                            market = getattr(live_opp, 'market', 'unknown')
                            ev = getattr(live_opp, 'ev', 0.0)
                            conf = getattr(live_opp, 'confidence', 0.0)
                            quality = getattr(live_opp, 'signal_quality_score', 0.0)

                        opportunities_found += 1
                        all_opportunities.append(opp)  # Mantieni struttura originale per selezione successiva
                        logger.info(f"   ✅ {market}: EV={ev:.1f}%, Conf={conf:.1f}%, Quality={quality:.1f}")
                else:
                    matches_without_opportunities += 1
                    # 🔧 DEBUG: Log dettagliato perché non ci sono opportunità
                    logger.info(f"📊 {match_name}: nessuna opportunità trovata")
                    logger.info(f"   Dettagli: score={score}, min={minute}', stats={'✅' if has_stats else '❌'}, odds={'✅' if has_odds else '❌'}")
                    if not has_stats:
                        logger.info(f"   ⚠️  Motivo: partita senza statistiche disponibili")
                    elif not has_odds:
                        logger.info(f"   ⚠️  Motivo: partita senza quote disponibili")
                    else:
                        logger.info(f"   ℹ️  Motivo: partita analizzata ma nessuna opportunità generata dal LiveBettingAdvisor")
            except Exception as e:
                logger.error(f"❌ Error analyzing match {match.get('id', 'unknown')}: {e}")
                continue
        
        # 🔧 LOG VISIBILE: Riepilogo analisi
        logger.info("=" * 80)
        logger.info(f"📊 RIEPILOGO ANALISI PARTITE:")
        logger.info(f"   - Partite analizzate: {matches_analyzed}")
        logger.info(f"   - Partite con opportunità: {matches_with_opportunities}")
        logger.info(f"   - Partite senza opportunità: {matches_without_opportunities}")
        logger.info(f"   - Opportunità totali generate: {opportunities_found}")
        logger.info("=" * 80)
        
        # 🆕 NUOVO: Seleziona e invia solo le migliori opportunità
        notified_count = 0
        if all_opportunities:
            logger.info(f"📊 SELEZIONE OPPORTUNITÀ:")
            logger.info(f"   - Prima della selezione: {opportunities_found} opportunità totali")
            best_opportunities = self._select_best_opportunities(all_opportunities)
            logger.info(f"   - Dopo la selezione: {len(best_opportunities)} opportunità selezionate per notifica")
            
            if len(best_opportunities) < opportunities_found:
                logger.info(f"   - Opportunità scartate dalla selezione: {opportunities_found - len(best_opportunities)} (selezionata solo la migliore)")
            
            for opp_dict in best_opportunities:
                self._handle_live_opportunity(opp_dict)
                notified_count += 1
        else:
            logger.info(f"📊 Nessuna opportunità trovata in questo ciclo")
        
        logger.info("=" * 80)
        logger.info(f"✅ CYCLE COMPLETE: {opportunities_found} opportunities found, {notified_count} notified")
        logger.info("=" * 80)
    
    def _get_matches_to_monitor(self) -> List[Dict]:
        """
        Ottiene partite da monitorare.
        
        Strategia:
        - Partite nelle prossime 24h (pre-match)
        - Partite in corso (live)
        - Evita partite già analizzate di recente
        """
        matches = []
        
        if not self.api_manager:
            logger.warning("⚠️  API Manager not available, using mock data")
            return self._get_mock_matches()
        
        try:
            # 🔧 NUOVO: Controlla se il servizio è attivo prima di fare chiamate API
            if not self._is_service_active():
                logger.warning("⚠️  Servizio non attivo, skip get matches")
                return []
            
            # Reset quota se nuovo giorno
            if self.api_usage_today >= self.api_budget_per_day:
                logger.warning(f"⚠️  API quota exhausted ({self.api_usage_today}/{self.api_budget_per_day})")
                return []
            
            # Prova a ottenere partite reali da API-Football
            matches = self._fetch_real_matches()
            
            # Se non ci sono partite reali, usa mock per testing
            if not matches:
                logger.info("ℹ️  No real matches found, using mock data for testing")
                matches = self._get_mock_matches()
            
        except Exception as e:
            logger.error(f"❌ Error fetching matches: {e}")
            # Fallback a mock in caso di errore
            matches = self._get_mock_matches()
        
        return matches
    
    def _fetch_real_matches(self) -> List[Dict]:
        """
        Ottiene partite reali da API-Football con TUTTE le quote disponibili.
        
        🔧 MODIFICATO: Usa API-Football come fonte principale per le quote.
        Estrae TUTTI i mercati disponibili (1X2, Over/Under, BTTS, Asian Handicap, ecc.).
        """
        # Controlla se il servizio è attivo (evita chiamate quando Render è sospeso)
        if not self._is_service_active():
            logger.warning("⚠️  Servizio non attivo, skip fetch matches")
            return []
        
        # Usa API-Football per ottenere partite con tutte le quote
        if not self.api_manager:
            logger.warning("⚠️  API Manager not available, using mock data")
            return self._get_mock_matches()
        
        try:
            # Ottieni partite da API-Football con tutte le quote
            matches = self._fetch_matches_with_odds_from_api_football()
            
            if matches:
                logger.info(f"✅ Trovate {len(matches)} partite con quote da API-Football")
                return matches
            else:
                logger.info("ℹ️  Nessuna partita trovata da API-Football")
                return []
            
        except Exception as e:
            logger.error(f"❌ Error fetching matches from API-Football: {e}")
            return []
    
    def _fetch_matches_with_odds_from_api_football(self) -> List[Dict]:
        """
        Estrae partite da API-Football con TUTTE le quote disponibili.
        
        Cerca tutti i mercati possibili:
        - Match Winner (1X2)
        - Over/Under (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
        - BTTS (Both Teams To Score)
        - Asian Handicap
        - Double Chance
        - Draw No Bet
        - Altri mercati disponibili
        """
        import os
        import urllib.request
        import urllib.parse
        import json
        from datetime import datetime, timedelta
        
        api_key = os.getenv("API_FOOTBALL_KEY", "")
        if not api_key:
            logger.warning("⚠️  API_FOOTBALL_KEY non configurata")
            return []
        
        try:
            base_url = "https://v3.football.api-sports.io"

            # ✅ NUOVO: Usa endpoint /fixtures?live=all per ottenere TUTTE le partite live in corso
            # Questo è più efficiente e diretto rispetto a cercare per data e filtrare
            params = {
                "live": "all"  # Tutte le partite live in corso al mondo
            }
            
            query = urllib.parse.urlencode(params)
            url = f"{base_url}/fixtures?{query}"
            headers = {
                "x-rapidapi-key": api_key,
                "x-rapidapi-host": "v3.football.api-sports.io"
            }
            
            logger.info(f"📡 Fetching LIVE fixtures from API-Football (endpoint: /fixtures?live=all)...")
            self.api_usage_today += 1  # Conta chiamata API per fixtures
            req = urllib.request.Request(url, headers=headers)
            
            # 🎯 Inizializza matches_found prima del retry per evitare errori
            matches_found = []
            
            # 🎯 RETRY LOGIC: Usa retry con backoff esponenziale per resilienza
            def _make_fixtures_request():
                with urllib.request.urlopen(req, timeout=15) as response:
                    response_data = response.read().decode()
                    return json.loads(response_data)
            
            data = self._retry_api_call(_make_fixtures_request, max_retries=3, base_delay=1.0)
            if data is None:
                logger.error("❌ Impossibile recuperare fixtures dopo retry")
                return []
            
            if data.get("errors"):
                logger.error(f"❌ API-Football ha restituito errori: {data.get('errors')}")
                return []
            
            if not data.get("response"):
                logger.info(f"ℹ️  Nessuna partita LIVE trovata in questo momento (response vuoto)")
                matches_found = []
            else:
                matches_found = data["response"]
                logger.info(f"📊 Trovate {len(matches_found)} partite LIVE in corso dall'API!")
                
                # 🎯 DEBUG: Log dettagliato delle partite trovate
                for i, fixture in enumerate(matches_found[:3]):  # Prime 3 per debug
                    fixture_data = fixture.get("fixture", {})
                    teams_data = fixture.get("teams", {})
                    status_short = fixture_data.get("status", {}).get("short", "N/A")
                    home = teams_data.get("home", {}).get("name", "?")
                    away = teams_data.get("away", {}).get("name", "?")
                    logger.info(f"   Partita {i+1}: {home} vs {away} - Status: {status_short}")

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()
            except:
                pass
            logger.error(f"❌ API-Football HTTP error: {e.code} - {e.reason}")
            if error_body:
                logger.error(f"   Response body: {error_body[:500]}")
            if e.code == 429:
                logger.error("⚠️  Rate limit raggiunto, aspetta prima di riprovare")
            elif e.code == 401:
                logger.error("⚠️  API key non valida o scaduta")
            elif e.code == 403:
                logger.error("⚠️  Accesso negato - verifica API key e permessi")
            matches_found = []
            return []
        except Exception as e:
            logger.error(f"❌ Errore chiamata API-Football: {e}")
            matches_found = []
            return []
            
            # Se non ci sono partite, ritorna lista vuota
            if not matches_found:
                return []
            
            # Le partite sono già LIVE (filtrate dall'API), non serve filtrarle di nuovo
            data = {"response": matches_found}
            
            matches = []
            live_count = 0
            skipped_finished = 0
            skipped_not_live = 0
            skipped_no_stats = 0
            skipped_no_odds = 0
            
            for fixture in data["response"]:
                try:
                    fixture_data = fixture.get("fixture", {})
                    teams_data = fixture.get("teams", {})
                    league_data = fixture.get("league", {})
                    odds_data = fixture.get("odds", [])  # Lista di bookmaker con quote
                    goals_data = fixture.get("goals", {})  # 🔧 FIX: Score potrebbe essere qui!
                    
                    # 🔧 DEBUG: Log struttura completa fixture per vedere dove è lo score
                    logger.info(f"🔍 RAW fixture completo per {teams_data.get('home', {}).get('name', '?')} vs {teams_data.get('away', {}).get('name', '?')}:")
                    logger.info(f"   fixture.keys(): {list(fixture.keys())}")
                    logger.info(f"   fixture['goals']: {goals_data}")
                    logger.info(f"   fixture['teams']: {teams_data}")
                    
                    # Estrai informazioni base
                    fixture_id = fixture_data.get("id")
                    if not fixture_id:
                        continue
                    
                    date_str = fixture_data.get("date")
                    if not date_str:
                        continue
                    
                    # Parse datetime
                    fixture_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    
                    # ✅ SEMPLIFICATO: L'endpoint ?live=all restituisce già solo partite live
                    # Non serve più filtrare per data o status - sono tutte live!
                    status_short = fixture_data.get("status", {}).get("short", "")

                    # Doppio check di sicurezza (dovrebbe essere sempre live)
                    is_live = status_short in ["1H", "HT", "2H", "ET", "P", "LIVE"]
                    is_finished = status_short in ["FT", "AET", "PEN"]
                    is_not_started = status_short in ["NS", "TBD", "CANC", "SUSP", "INT", "PST", "ABAN"]
                    
                    if is_finished:
                        skipped_finished += 1
                        continue  # Salta partite finite
                    
                    if is_not_started:
                        skipped_not_live += 1
                        continue  # Salta partite non iniziate o sospese
                    
                    # Verifica che sia live (doppio check di sicurezza)
                    if not is_live:
                        skipped_not_live += 1
                        continue  # Salta partite non LIVE
                    
                    live_count += 1
                    
                    home_team = teams_data.get("home", {}).get("name", "")
                    away_team = teams_data.get("away", {}).get("name", "")
                    league_name = league_data.get("name", "Unknown League")
                    
                    if not home_team or not away_team:
                        continue
                    
                    # 🎯 FILTRO ESSENZIALE: Richiede statistiche per calcolare confidence precisa
                    logger.info(f"🔍 Verificando statistiche per {home_team} vs {away_team} (fixture {fixture_id}, status: {status_short})...")
                    statistics = self._fetch_statistics_from_api_football(fixture_id, api_key, base_url)
                    if not statistics:
                        skipped_no_stats += 1
                        logger.warning(f"⚠️  Partita LIVE {home_team} vs {away_team} (status: {status_short}) senza statistiche disponibili, skip (necessarie per confidence precisa)")
                        continue  # Salta questa partita, serve statistiche per confidence precisa
                    logger.info(f"✅ Statistiche disponibili per {home_team} vs {away_team} (status: {status_short}), procedo con estrazione quote")
                    
                    # 🔧 FIX: Per partite LIVE, dobbiamo fare una chiamata separata per le quote
                    # L'endpoint /fixtures non include sempre le quote per partite LIVE, dobbiamo richiederle
                    logger.info(f"🔍 Verificando quote per {home_team} vs {away_team} (fixture {fixture_id})...")
                    logger.info(f"   Quote iniziali da /fixtures: {len(odds_data) if odds_data else 0} bookmaker")
                    
                    odds_pending = False
                    if not odds_data or len(odds_data) == 0:
                        logger.info(f"   ⚠️  Nessuna quota in /fixtures, provo a recuperare con /odds?fixture={fixture_id}")
                        try:
                            odds_fetch = self._fetch_fixture_odds_from_api_football(fixture_id, api_key, base_url)
                        except Exception as e:
                            logger.warning(f"⚠️  Errore recupero quote per fixture {fixture_id}: {e}")
                            odds_fetch = None
                        if odds_fetch:
                            odds_data = odds_fetch
                            self.api_usage_today += 1  # Conta chiamata API per quote
                            logger.info(f"✅ Quote recuperate per {home_team} vs {away_team} (fixture {fixture_id}, {len(odds_data)} bookmaker)")
                        else:
                            skipped_no_odds += 1
                            odds_pending = True
                            odds_data = []
                            logger.warning(f"⚠️  Nessuna quota disponibile per fixture {fixture_id}, lascio che il watchdog ritenti")
                    else:
                        logger.debug(f"✅ Quote già presenti in /fixtures per {home_team} vs {away_team} ({len(odds_data)} bookmaker)")
                    
                    # Estrai TUTTE le quote disponibili
                    logger.info(f"🔍 Estraendo quote per {home_team} vs {away_team} (fixture {fixture_id})...")
                    logger.info(f"   odds_data type: {type(odds_data)}, length: {len(odds_data) if isinstance(odds_data, list) else 'N/A'}")
                    if odds_data and len(odds_data) > 0:
                        logger.info(f"   Primo bookmaker keys: {list(odds_data[0].keys()) if isinstance(odds_data[0], dict) else 'NOT_DICT'}")
                    
                    all_odds = self._extract_all_odds_from_api_football(odds_data)
                    
                    # 🔧 DEBUG: Log dettagliato quote estratte (INFO per vedere nei log)
                    logger.info(f"📊 Quote estratte per {home_team} vs {away_team}:")
                    logger.info(f"   Quote 1X2: home={all_odds.get('match_winner', {}).get('home')}, draw={all_odds.get('match_winner', {}).get('draw')}, away={all_odds.get('match_winner', {}).get('away')}")
                    logger.info(f"   Over/Under FT: {list(all_odds.get('over_under', {}).keys())} ({len(all_odds.get('over_under', {}))} thresholds)")
                    logger.info(f"   Over/Under HT: {list(all_odds.get('over_under_ht', {}).keys())} ({len(all_odds.get('over_under_ht', {}))} thresholds)")
                    logger.info(f"   First Half Goals: {list(all_odds.get('first_half_goals', {}).keys())} ({len(all_odds.get('first_half_goals', {}))} thresholds)")
                    logger.info(f"   Second Half Goals: {list(all_odds.get('second_half_goals', {}).keys())} ({len(all_odds.get('second_half_goals', {}))} thresholds)")
                    logger.info(f"   BTTS: {all_odds.get('btts', {}).get('yes')} (yes) / {all_odds.get('btts', {}).get('no')} (no)")
                    logger.info(f"   BTTS HT: {all_odds.get('btts_ht', {}).get('yes')} (yes) / {all_odds.get('btts_ht', {}).get('no')} (no)")
                    logger.info(f"   Double Chance: {all_odds.get('double_chance', {})}")
                    logger.info(f"   Draw No Bet: {all_odds.get('draw_no_bet', {})}")
                    logger.info(f"   Asian Handicap: {bool(all_odds.get('asian_handicap'))}")
                    
                    # Log quanti mercati sono stati trovati
                    markets_found = []
                    if all_odds.get('match_winner', {}).get('home'):
                        markets_found.append('1X2')
                    if all_odds.get('over_under'):
                        markets_found.append(f"Over/Under FT ({len(all_odds['over_under'])} thresholds)")
                    if all_odds.get('over_under_ht'):
                        markets_found.append(f"Over/Under HT ({len(all_odds['over_under_ht'])} thresholds)")
                    if all_odds.get('first_half_goals'):
                        markets_found.append(f"First Half Goals ({len(all_odds['first_half_goals'])} thresholds)")
                    if all_odds.get('second_half_goals'):
                        markets_found.append(f"Second Half Goals ({len(all_odds['second_half_goals'])} thresholds)")
                    if all_odds.get('btts', {}).get('yes'):
                        markets_found.append('BTTS FT')
                    if all_odds.get('btts_ht', {}).get('yes'):
                        markets_found.append('BTTS HT')
                    if all_odds.get('double_chance', {}).get('1x'):
                        markets_found.append('Double Chance')
                    if all_odds.get('draw_no_bet', {}).get('home'):
                        markets_found.append('Draw No Bet')
                    if all_odds.get('asian_handicap'):
                        markets_found.append(f"Asian Handicap ({len(all_odds['asian_handicap'])} options)")
                    if all_odds.get('other_markets'):
                        markets_found.append(f"Altri mercati ({len(all_odds['other_markets'])} markets)")
                    
                    if markets_found:
                        logger.info(f"📊 Mercati trovati per {home_team} vs {away_team}: {', '.join(markets_found)}")
                    else:
                        logger.warning(f"⚠️  Nessun mercato trovato per {home_team} vs {away_team} (fixture {fixture_id})")
                    
                    # Estrai score e minute dalla fixture
                    # 🔧 FIX CRITICO: Lo score NON è in fixture_data['score'] ma probabilmente in fixture['goals']
                    score_data = fixture_data.get("score", {})
                    
                    # 🔧 DEBUG: Log RAW score_data completo
                    logger.info(f"⚽ RAW score_data per {home_team} vs {away_team}:")
                    logger.info(f"   score_data type: {type(score_data)}")
                    if isinstance(score_data, dict):
                        logger.info(f"   score_data completo: {json.dumps(score_data, indent=2, default=str)}")
                    else:
                        logger.warning(f"   ⚠️ score_data non è un dict: {score_data}")
                    
                    # 🔧 FIX CRITICO: Prova a estrarre lo score da fixture['goals'] (non fixture_data['score'])
                    score_home = 0
                    score_away = 0
                    
                    # 1. Prova da fixture['goals'] (score corrente LIVE)
                    if isinstance(goals_data, dict):
                        g_home = goals_data.get("home")
                        g_away = goals_data.get("away")
                        if g_home is not None or g_away is not None:
                            score_home = g_home or 0
                            score_away = g_away or 0
                            logger.info(f"   ⚽ Score da fixture['goals']: {score_home}-{score_away}")
                    
                    # 2. Prova da fixture_data['score'] (se presente)
                    if isinstance(score_data, dict) and score_data:
                        # Prova fulltime
                        fulltime = score_data.get("fulltime", {})
                        if isinstance(fulltime, dict):
                            ft_home = fulltime.get("home")
                            ft_away = fulltime.get("away")
                            if ft_home is not None or ft_away is not None:
                                score_home = ft_home or score_home
                                score_away = ft_away or score_away
                                logger.info(f"   ⚽ Score da fulltime: {score_home}-{score_away}")
                        
                        # Prova extratime
                        extratime = score_data.get("extratime", {})
                        if isinstance(extratime, dict):
                            et_home = extratime.get("home")
                            et_away = extratime.get("away")
                            if et_home is not None or et_away is not None:
                                score_home = et_home or score_home
                                score_away = et_away or score_away
                                logger.info(f"   ⚽ Score da extratime: {score_home}-{score_away}")
                        
                        # Prova halftime
                        halftime = score_data.get("halftime", {})
                        if isinstance(halftime, dict):
                            ht_home = halftime.get("home")
                            ht_away = halftime.get("away")
                            if ht_home is not None or ht_away is not None:
                                score_home = ht_home or score_home
                                score_away = ht_away or score_away
                                logger.info(f"   ⚽ Score da halftime: {score_home}-{score_away}")
                    
                    logger.info(f"   ⚽ Score FINALE estratto: {score_home}-{score_away}")
                    
                    # 🔧 DEBUG: Log RAW fixture_data per capire struttura completa
                    logger.info(f"🔍 RAW fixture_data per {home_team} vs {away_team}:")
                    logger.info(f"   fixture_data.keys(): {list(fixture_data.keys())}")
                    logger.info(f"   fixture_data['status']: {fixture_data.get('status', 'NOT_FOUND')}")
                    logger.info(f"   fixture_data['date']: {fixture_data.get('date', 'NOT_FOUND')}")
                    
                    # 🔧 FIX: Estrai minuto da status - prova diversi campi
                    status_data = fixture_data.get("status", {})
                    
                    # 🔧 DEBUG: Log RAW status PRIMA di estrarre
                    logger.info(f"🔍 RAW status_data per {home_team} vs {away_team}:")
                    logger.info(f"   type(status_data): {type(status_data)}")
                    if isinstance(status_data, dict):
                        logger.info(f"   status_data.keys(): {list(status_data.keys())}")
                        logger.info(f"   status_data completo: {json.dumps(status_data, indent=2, default=str)}")
                    else:
                        logger.warning(f"   ⚠️ status_data non è un dict: {status_data}")
                    
                    # Estrai minuto - prova TUTTI i possibili campi
                    minute = 0
                    if isinstance(status_data, dict):
                        # Prova tutti i possibili nomi di campo
                        minute = (status_data.get("elapsed") or 
                                 status_data.get("elapsed_time") or 
                                 status_data.get("elapsedTime") or
                                 status_data.get("minute") or
                                 status_data.get("time") or 0)
                        
                        # Se è None, converti a 0
                        if minute is None:
                            minute = 0
                        else:
                            try:
                                minute = int(minute)
                            except (ValueError, TypeError):
                                minute = 0
                    
                    status_short = status_data.get("short", "") if isinstance(status_data, dict) else ""
                    status_long = status_data.get("long", "") if isinstance(status_data, dict) else ""
                    
                    logger.info(f"🔍 Minuto estratto: {minute}' (status: {status_short}, long: {status_long})")
                    
                    # 🔧 FIX CRITICO: Gestisci status speciali PRIMA di calcolare dalla data
                    # Se status è HT (Half Time), il minuto è sempre 45, non calcolarlo dalla data
                    if status_short == "HT":
                        minute = 45
                        logger.info(f"⏰ Minuto impostato da status HT: 45' (intervallo)")
                    elif status_short == "2H":
                        # Secondo tempo: calcola dalla data SOTTRARRE 15 minuti di intervallo
                        # Se partita iniziata 90 minuti fa: 45' (1T) + 15' (intervallo) + 30' (2T) = 90 minuti totali
                        # Ma minuto di gioco = 45 + 30 = 75', non 90'!
                        try:
                            now = datetime.now(timezone.utc)
                            time_diff = (now - fixture_date).total_seconds() / 60
                            
                            if time_diff > 0 and time_diff < 120:
                                # Sottrai 15 minuti di intervallo e aggiungi 45 minuti del primo tempo
                                # minute = 45 (primo tempo) + max(0, time_diff - 60) (secondo tempo dopo intervallo)
                                # Se time_diff < 60, siamo ancora nel primo tempo o intervallo
                                if time_diff <= 60:
                                    # Ancora nel primo tempo o intervallo, ma status dice 2H → usa minimo 46
                                    minute = 46
                                    logger.info(f"⏰ Minuto per 2H: {minute}' (time_diff={time_diff:.1f} <= 60, minimo 46')")
                                else:
                                    # Dopo intervallo: primo tempo (45') + secondo tempo (time_diff - 60)
                                    calculated_minute = 45 + int(time_diff - 60)
                                    # Limita a 90 minuti massimo (prima dei tempi supplementari)
                                    minute = min(90, max(46, calculated_minute))
                                    logger.info(f"⏰ Minuto per 2H: {minute}' (time_diff={time_diff:.1f}, calcolato: {calculated_minute}', intervallo sottratto)")
                            else:
                                minute = 46  # Fallback minimo
                                logger.info(f"⏰ Minuto per 2H: {minute}' (fallback minimo)")
                        except Exception as e:
                            minute = 46  # Fallback minimo
                            logger.warning(f"   ⚠️ Errore calcolo minuto 2H: {e}, uso 46'")
                    elif status_short in ["1H", "ET", "P", "LIVE"]:
                        # Primo tempo, tempi supplementari, rigori, o LIVE generico: calcola dalla data
                        try:
                            now = datetime.now(timezone.utc)
                            time_diff = (now - fixture_date).total_seconds() / 60
                            
                            logger.info(f"   Calcolo minuto dalla data: now={now}, fixture_date={fixture_date}, time_diff={time_diff:.1f} minuti")
                            
                            # Se la partita è iniziata (time_diff positivo) e siamo entro 2 ore
                            if time_diff > 0 and time_diff < 120:
                                calculated_minute = int(time_diff)
                                
                                # 🔧 FIX: Se il minuto estratto è già presente e valido, usalo come base
                                # Altrimenti usa quello calcolato
                                minute_extracted = minute  # Minuto estratto da status_data
                                
                                if status_short == "1H":
                                    # Per 1H: usa il minuto estratto se valido (tra 1-45), altrimenti calcola
                                    if minute_extracted > 0 and 1 <= minute_extracted <= 45:
                                        # Minuto estratto valido, usalo
                                        minute = minute_extracted
                                        logger.info(f"⏰ Minuto per 1H da status_data: {minute}' (valido)")
                                    else:
                                        # Minuto estratto non valido o 0, calcola dalla data ma limita a 45
                                        minute = max(1, min(45, calculated_minute))
                                        logger.info(f"⏰ Minuto per 1H calcolato dalla data: {minute}' (limitato a 45')")
                                else:
                                    # Per ET, P, LIVE: usa il minuto estratto se valido, altrimenti calcola
                                    if minute_extracted > 0:
                                        # Se il minuto estratto è valido, usalo solo se è maggiore di quello calcolato
                                        # (il minuto estratto potrebbe essere più recente)
                                        minute = max(minute_extracted, calculated_minute)
                                        logger.info(f"⏰ Minuto per {status_short}: {minute}' (estratto: {minute_extracted}', calcolato: {calculated_minute}')")
                                    else:
                                        # Minuto estratto non valido, usa quello calcolato
                                        minute = calculated_minute
                                        logger.info(f"⏰ Minuto per {status_short} calcolato dalla data: {minute}'")
                            elif time_diff < 0:
                                # Partita non ancora iniziata - ma se status è LIVE, potrebbe essere un problema di timezone
                                logger.warning(f"   ⚠️ Partita {home_team} vs {away_team} con status {status_short} ma time_diff negativo ({time_diff:.1f} minuti) - possibile problema timezone")
                                if status_short == "1H" and minute == 0:
                                    minute = 1  # Fallback minimo per 1H solo se minuto è 0
                            else:
                                logger.info(f"   Partita {home_team} vs {away_team} iniziata più di 2 ore fa ({time_diff:.1f} minuti)")
                                if status_short == "1H" and minute == 0:
                                    minute = 45  # Se è 1H ma partita iniziata più di 2 ore fa, probabilmente è finita
                        except Exception as e:
                            logger.warning(f"   ⚠️ Errore calcolo minuto dalla data: {e}")
                            import traceback
                            logger.debug(f"   Traceback: {traceback.format_exc()}")
                            if status_short == "1H" and minute == 0:
                                minute = 1  # Fallback minimo solo se minuto è 0
                    
                    # 🔧 FIX: Se ancora 0, prova a dedurlo dallo status (fallback finale)
                    if minute == 0:
                        if status_short == "HT":
                            minute = 45
                            logger.info(f"⏰ Minuto dedotto da status HT (fallback): 45'")
                        elif status_short == "2H":
                            minute = 46
                            logger.info(f"⏰ Minuto dedotto da status 2H (fallback): 46'")
                        elif status_short == "1H":
                            minute = 1
                            logger.info(f"⏰ Minuto dedotto da status 1H (fallback): 1'")
                        elif status_short == "LIVE":
                            minute = 1
                            logger.info(f"⏰ Minuto dedotto da status LIVE (fallback): 1'")
                        else:
                            logger.warning(f"⚠️ Minuto ancora 0 per {home_team} vs {away_team} con status={status_short}")
                    
                    logger.info(f"✅ Minuto FINALE per {home_team} vs {away_team}: {minute}'")
                    
                    # Crea match dict con tutte le quote
                    match = {
                        'id': str(fixture_id),
                        'home': home_team,
                        'away': away_team,
                        'league': league_name,
                        'date': fixture_date,
                        'is_live': is_live,
                        'match_status': 'LIVE' if is_live else 'PRE-MATCH',
                        'status': status_short,
                        'fixture_id': fixture_id,
                        # Score e minute
                        'score_home': score_home,
                        'score_away': score_away,
                        'minute': minute,
                        # Quote base 1X2 (sempre presenti se disponibili)
                        'odds_1': all_odds.get('match_winner', {}).get('home'),
                        'odds_x': all_odds.get('match_winner', {}).get('draw'),
                        'odds_2': all_odds.get('match_winner', {}).get('away'),
                        # Tutte le altre quote disponibili
                        'all_odds': all_odds,
                        'odds_pending': odds_pending,
                        'all_odds_precision': all_odds.get('_precision_snapshot')
                    }
                    
                    # Aggiungi quote specifiche per compatibilità con codice esistente
                    if all_odds.get('over_under'):
                        for threshold, odds in all_odds['over_under'].items():
                            match[f'over_{threshold}'] = odds.get('over')
                            match[f'under_{threshold}'] = odds.get('under')
                    
                    # Aggiungi quote HT
                    if all_odds.get('over_under_ht'):
                        for threshold, odds in all_odds['over_under_ht'].items():
                            match[f'over_{threshold}_ht'] = odds.get('over')
                            match[f'under_{threshold}_ht'] = odds.get('under')
                    
                    # Aggiungi First Half Goals
                    if all_odds.get('first_half_goals'):
                        for threshold, odds in all_odds['first_half_goals'].items():
                            match[f'over_{threshold}_1h'] = odds.get('over')
                            match[f'under_{threshold}_1h'] = odds.get('under')
                    
                    # Aggiungi Second Half Goals
                    if all_odds.get('second_half_goals'):
                        for threshold, odds in all_odds['second_half_goals'].items():
                            match[f'over_{threshold}_2h'] = odds.get('over')
                            match[f'under_{threshold}_2h'] = odds.get('under')
                    
                    if all_odds.get('btts'):
                        match['btts_yes'] = all_odds['btts'].get('yes')
                        match['btts_no'] = all_odds['btts'].get('no')
                    
                    if all_odds.get('btts_ht'):
                        match['btts_yes_ht'] = all_odds['btts_ht'].get('yes')
                        match['btts_no_ht'] = all_odds['btts_ht'].get('no')
                    
                    if all_odds.get('double_chance'):
                        match['odds_1x'] = all_odds['double_chance'].get('1x')
                        match['odds_12'] = all_odds['double_chance'].get('12')
                        match['odds_x2'] = all_odds['double_chance'].get('x2')
                    
                    if all_odds.get('draw_no_bet'):
                        match['dnb_home'] = all_odds['draw_no_bet'].get('home')
                        match['dnb_away'] = all_odds['draw_no_bet'].get('away')
                    
                    if all_odds.get('asian_handicap'):
                        match['asian_handicap'] = all_odds['asian_handicap']
                    
                    # 🔧 FIX CRITICO: Statistiche già recuperate sopra, aggiungile al match dict
                    # ⚠️ ATTENZIONE: match.update(stats_dict) sovrascrive TUTTI i campi, incluso minute!
                    # Devo salvare il minuto PRIMA di fare update e ripristinarlo dopo se stats_dict['minute'] è 0
                    if statistics:
                        # Estrai statistiche e aggiungile al match dict
                        stats_dict = self._parse_statistics_from_api_football(statistics)
                        
                        # 🔧 DEBUG: Log statistiche raw per capire struttura (solo primi 500 caratteri)
                        try:
                            stats_preview = json.dumps(statistics[:2] if len(statistics) >= 2 else statistics, indent=2, default=str)[:500]
                            logger.info(f"📊 Statistiche raw per {home_team} vs {away_team}: {stats_preview}...")
                        except:
                            logger.info(f"📊 Statistiche presenti per {home_team} vs {away_team} ma non serializzabili")
                        
                        # 🔧 FIX CRITICO: Salva il minuto e score correnti PRIMA di fare update
                        minute_before_update = match.get('minute', 0)
                        score_home_before = match.get('score_home', 0)
                        score_away_before = match.get('score_away', 0)
                        logger.info(f"🔍 PRIMA di update statistiche: minuto={minute_before_update}', score={score_home_before}-{score_away_before}")
                        
                        # 🔧 FIX CRITICO: Rimuovi 'minute' e 'score' da stats_dict se sono 0 per evitare sovrascrittura
                        stats_minute = stats_dict.get('minute', 0)
                        stats_score_home = stats_dict.get('score_home', 0)
                        stats_score_away = stats_dict.get('score_away', 0)
                        
                        # Prepara stats_dict senza campi che non vogliamo sovrascrivere
                        stats_dict_filtered = {}
                        for k, v in stats_dict.items():
                            if k == 'minute':
                                # Aggiorna minuto solo se stats_minute > 0 e maggiore di quello corrente
                                if stats_minute > 0 and (stats_minute > minute_before_update or minute_before_update == 0):
                                    stats_dict_filtered[k] = stats_minute
                                    logger.info(f"⏰ Minuto aggiornato da statistiche: {minute_before_update}' -> {stats_minute}'")
                                else:
                                    logger.info(f"🔍 Minuto statistiche ({stats_minute}') non migliore, mantenuto fixture ({minute_before_update}')")
                            elif k in ['score_home', 'score_away']:
                                # Aggiorna score solo se stats_score > 0 (score corrente dalle statistiche)
                                if k == 'score_home' and stats_score_home > 0:
                                    stats_dict_filtered[k] = stats_score_home
                                    logger.info(f"⚽ Score home aggiornato da statistiche: {score_home_before} -> {stats_score_home}")
                                elif k == 'score_away' and stats_score_away > 0:
                                    stats_dict_filtered[k] = stats_score_away
                                    logger.info(f"⚽ Score away aggiornato da statistiche: {score_away_before} -> {stats_score_away}")
                                else:
                                    logger.info(f"🔍 Score {k} statistiche ({stats_dict.get(k, 0)}) non migliore, mantenuto fixture ({match.get(k, 0)})")
                            else:
                                # Altri campi: aggiungi sempre
                                stats_dict_filtered[k] = v
                        
                        match.update(stats_dict_filtered)
                        logger.info(f"✅ Statistiche aggiunte per {home_team} vs {away_team}, minuto finale: {match.get('minute', 0)}', score finale: {match.get('score_home', 0)}-{match.get('score_away', 0)}")
                    
                    # 🔧 FIX: Accetta partita se ha statistiche E almeno alcune quote (non tutte, ma almeno qualcosa)
                    # Le quote sono necessarie per calcolare EV, quindi dobbiamo avere almeno qualche quota disponibile
                    odds_1 = match.get('odds_1')
                    odds_x = match.get('odds_x')
                    odds_2 = match.get('odds_2')
                    has_1x2_complete = odds_1 and odds_x and odds_2
                    has_1x2_partial = sum([bool(odds_1), bool(odds_x), bool(odds_2)]) >= 1
                    
                    # Verifica TUTTE le quote disponibili
                    has_over_under_ft = bool(all_odds.get('over_under'))
                    has_over_under_ht = bool(all_odds.get('over_under_ht'))
                    has_first_half_goals = bool(all_odds.get('first_half_goals'))
                    has_second_half_goals = bool(all_odds.get('second_half_goals'))
                    has_btts = bool(all_odds.get('btts', {}).get('yes'))
                    has_btts_ht = bool(all_odds.get('btts_ht', {}).get('yes'))
                    has_double_chance = bool(all_odds.get('double_chance', {}).get('1x'))
                    has_dnb = bool(all_odds.get('draw_no_bet', {}).get('home'))
                    has_asian_handicap = bool(all_odds.get('asian_handicap'))
                    
                    has_other_odds = (
                        has_over_under_ft or 
                        has_over_under_ht or 
                        has_first_half_goals or
                        has_second_half_goals or
                        has_btts or 
                        has_btts_ht or
                        has_double_chance or
                        has_dnb or
                        has_asian_handicap
                    )
                    
                    # 🔧 DEBUG: Log dettagliato quote disponibili
                    logger.info(f"📊 Quote disponibili per {home_team} vs {away_team}:")
                    logger.info(f"   1X2: {bool(odds_1)}/{bool(odds_x)}/{bool(odds_2)} (complete: {has_1x2_complete}, partial: {has_1x2_partial})")
                    logger.info(f"   Over/Under FT: {has_over_under_ft}")
                    logger.info(f"   Over/Under HT: {has_over_under_ht}")
                    logger.info(f"   First Half Goals: {has_first_half_goals}")
                    logger.info(f"   Second Half Goals: {has_second_half_goals}")
                    logger.info(f"   BTTS: {has_btts}")
                    logger.info(f"   BTTS HT: {has_btts_ht}")
                    logger.info(f"   Double Chance: {has_double_chance}")
                    logger.info(f"   Draw No Bet: {has_dnb}")
                    logger.info(f"   Asian Handicap: {has_asian_handicap}")
                    logger.info(f"   Ha statistiche: {bool(statistics)}")
                    logger.info(f"   Ha almeno qualche quota: {has_1x2_partial or has_other_odds}")
                    
                    # 🎯 FILTRO ESSENZIALE: Aggiungi solo partite con statistiche E quote (necessarie per confidence ed EV precisi)
                    # L'utente vuole calcolare confidence ed EV solo per partite complete
                    if statistics and (has_1x2_complete or has_1x2_partial or has_other_odds):
                        matches.append(match)
                        if has_1x2_complete:
                            logger.info(f"✅ Match {home_team} vs {away_team} aggiunto (ha statistiche e quote 1X2 complete)")
                        elif has_1x2_partial and has_other_odds:
                            logger.info(f"✅ Match {home_team} vs {away_team} aggiunto (ha statistiche, quote 1X2 parziali e altre quote)")
                        elif has_1x2_partial:
                            logger.info(f"✅ Match {home_team} vs {away_team} aggiunto (ha statistiche e almeno 1 quota 1X2)")
                        elif has_other_odds:
                            logger.info(f"✅ Match {home_team} vs {away_team} aggiunto (ha statistiche e altre quote disponibili)")
                    elif not statistics:
                        skipped_no_stats += 1
                        logger.debug(f"⏭️  Match {home_team} vs {away_team} senza statistiche, skip (necessarie per confidence precisa)")
                    else:
                        skipped_no_odds += 1
                        logger.debug(f"⏭️  Match {home_team} vs {away_team} senza quote sufficienti (1X2: {bool(odds_1)}/{bool(odds_x)}/{bool(odds_2)}, altre: {has_other_odds}), skip (necessarie per EV preciso)")
                
                except Exception as e:
                    logger.debug(f"⚠️  Error processing fixture: {e}")
                    continue
            
            logger.info(f"✅ Riepilogo estrazione partite LIVE:")
            logger.info(f"   - Partite LIVE totali trovate: {len(data['response'])}")
            logger.info(f"📊 RIEPILOGO FILTRAGGIO PARTITE LIVE:")
            logger.info(f"   ✅ Partite LIVE processate: {live_count}")
            logger.info(f"   ⏭️  Partite finite (skipped): {skipped_finished}")
            logger.info(f"   ⏭️  Partite non LIVE (skipped): {skipped_not_live}")
            logger.info(f"   ⚠️  Partite LIVE senza statistiche (skipped): {skipped_no_stats}")
            logger.info(f"   ⚠️  Partite LIVE senza quote (skipped): {skipped_no_odds}")
            logger.info(f"   ✅ Partite LIVE con quote e statistiche (VALIDATE): {len(matches)}")
            
            if len(matches) == 0:
                if live_count == 0:
                    logger.info(f"ℹ️  Nessuna partita LIVE trovata in questo momento. Questo è normale se non ci sono partite in corso.")
                else:
                    logger.warning(f"⚠️  PROBLEMA: Trovate {live_count} partite LIVE ma tutte scartate!")
                    if skipped_no_stats > 0:
                        logger.warning(f"   - {skipped_no_stats} partite senza statistiche disponibili (API potrebbe non avere ancora dati)")
                    if skipped_no_odds > 0:
                        logger.warning(f"   - {skipped_no_odds} partite senza quote disponibili (bookmaker potrebbero non offrire quote live)")
                    logger.warning(f"💡 SUGGERIMENTO: Le partite appena iniziate potrebbero non avere ancora statistiche/quote. Riprova tra qualche minuto.")
            
            return matches
            
        except urllib.error.HTTPError as e:
            logger.error(f"❌ API-Football HTTP error: {e.code} - {e.reason}")
            return []
        except Exception as e:
            logger.error(f"❌ Error fetching from API-Football: {e}")
            return []
    
    def _fetch_fixture_odds_from_api_football(self, fixture_id: int, api_key: str, base_url: str) -> Optional[List[Dict]]:
        """
        Recupera la lista di bookmaker/quote per un singolo fixture live.
        Restituisce una lista di bookmaker pronta per _extract_all_odds_from_api_football.
        """
        import urllib.request
        
        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "v3.football.api-sports.io"
        }
        odds_url = f"{base_url}/odds?fixture={fixture_id}"
        
        def _make_odds_request():
            odds_req = urllib.request.Request(odds_url, headers=headers)
            with urllib.request.urlopen(odds_req, timeout=10) as odds_response:
                return json.loads(odds_response.read().decode())
        
        odds_data_response = self._retry_api_call(_make_odds_request, max_retries=3, base_delay=1.0)
        if odds_data_response is None:
            return None
        
        response = odds_data_response.get("response")
        if not response:
            return None
        
        bookmakers_list: List[Dict[str, Any]] = []
        for item in response:
            if isinstance(item, dict) and item.get("bookmakers"):
                bookmakers_list.extend(item.get("bookmakers") or [])
            else:
                bookmakers_list.append(item)
        
        return bookmakers_list or None
    
    def _validate_odds(self, odd: Any) -> Optional[Decimal]:
        """
        🎯 PRECISIONE MANIACALE: Valida una quota con controlli rigorosi.
        
        Args:
            odd: Quota da validare (può essere str, int, float)
        
        Returns:
            Quota validata come Decimal, o None se invalida
        """
        if odd is None:
            return None
        
        try:
            if isinstance(odd, Decimal):
                decimal_odd = odd
            elif isinstance(odd, str):
                decimal_odd = Decimal(odd.strip())
            elif isinstance(odd, (int, float)):
                if isinstance(odd, float) and (math.isnan(odd) or math.isinf(odd)):
                    logger.debug(f"⚠️  Quota NaN/Inf ignorata: {odd}")
                    return None
                decimal_odd = Decimal(str(odd))
            else:
                return None
            
            if decimal_odd.is_nan():
                logger.debug(f"⚠️  Quota NaN ignorata: {odd}")
                return None
            if not decimal_odd.is_finite():
                logger.debug(f"⚠️  Quota non finita ignorata: {odd}")
                return None
            
            if decimal_odd <= Decimal("1.0"):
                logger.debug(f"⚠️  Quota <= 1.0 ignorata: {odd}")
                return None
            
            if decimal_odd > Decimal("1000"):
                logger.warning(f"⚠️  Quota sospetta > 1000 ignorata: {odd}")
                return None
            
            return decimal_odd
            
        except (InvalidOperation, ValueError, TypeError) as e:
            logger.debug(f"⚠️  Errore validazione quota: {odd}, errore: {e}")
            return None
    
    def _get_trusted_bookmakers(self) -> Dict[str, float]:
        """
        🎯 BOOKMAKER AFFIDABILI: Restituisce un dizionario di bookmaker affidabili con i loro pesi.
        
        I bookmaker più affidabili hanno peso più alto.
        I bookmaker meno conosciuti hanno peso più basso ma non vengono completamente esclusi.
        """
        return {
            'bet365': 1.0,           # Massima fiducia
            'pinnacle': 0.95,        # Bookmaker professionale
            'betfair': 0.90,         # Exchange affidabile
            'william hill': 0.85,    # Bookmaker storico
            'bwin': 0.85,
            'unibet': 0.85,
            '888sport': 0.80,
            'betway': 0.80,
            'ladbrokes': 0.80,
            'coral': 0.75,
            'betvictor': 0.75,
            'marathonbet': 0.75,
            'sportingbet': 0.70,
            'interwetten': 0.70,
            '10bet': 0.70,
            'betsson': 0.70,
            '1xbet': 0.65,           # Meno affidabile ma comune
            'betfred': 0.65,
            'skybet': 0.65,
            'dafabet': 0.60,
            'betonline': 0.60,
            'bodog': 0.60,
            'sportsbet.io': 0.60,
            'leovegas': 0.55,
            'mr green': 0.55,
        }
    
    def _get_bookmaker_weight(self, bookmaker_name: str) -> float:
        """
        Restituisce il peso di affidabilità di un bookmaker.
        Bookmaker sconosciuti ricevono peso base 0.5.
        """
        if not bookmaker_name:
            return 0.5
        
        bookmaker_lower = bookmaker_name.lower().strip()
        trusted_bookmakers = self._get_trusted_bookmakers()
        
        # Cerca corrispondenza esatta o parziale
        for trusted_name, weight in trusted_bookmakers.items():
            if trusted_name in bookmaker_lower or bookmaker_lower in trusted_name:
                return weight
        
        # Bookmaker non nella lista: peso medio-basso ma non zero
        return 0.5
    
    def _calculate_odds_quality_score(self, odd: Decimal, bookmaker: str, total_odds_count: int, 
                                      has_trusted_bookmakers: bool = True) -> float:
        """
        🎯 CALCOLA QUALITÀ QUOTA: Restituisce uno score (0-1) per la qualità di una quota.
        
        Fattori considerati:
        1. Affidabilità bookmaker (peso 40%)
        2. Numero totale quote disponibili (peso 20%)
        3. Presenza di bookmaker affidabili (peso 20%)
        4. Valore quota (peso 20%)
        
        Args:
            odd: Quota da valutare
            bookmaker: Nome del bookmaker
            total_odds_count: Numero totale di quote disponibili per questo mercato
            has_trusted_bookmakers: Se ci sono bookmaker affidabili tra le quote
        
        Returns:
            Score di qualità (0.0-1.0), dove 1.0 è la migliore qualità
        """
        try:
            quality_score = 0.0
            
            # 1. Affidabilità bookmaker (40%)
            bookmaker_weight = self._get_bookmaker_weight(bookmaker)
            quality_score += bookmaker_weight * 0.4
            
            # 2. Numero quote disponibili (20%)
            # Più quote = più affidabile (più bookmaker concordano)
            if total_odds_count >= 10:
                odds_availability_score = 1.0
            elif total_odds_count >= 5:
                odds_availability_score = 0.7
            elif total_odds_count >= 3:
                odds_availability_score = 0.5
            else:
                odds_availability_score = 0.3
            quality_score += odds_availability_score * 0.2
            
            # 3. Presenza bookmaker affidabili (20%)
            if has_trusted_bookmakers:
                quality_score += 1.0 * 0.2
            else:
                quality_score += 0.5 * 0.2
            
            # 4. Valore quota (20%)
            # Quote troppo basse (< 1.1) o troppo alte (> 100) sono sospette
            odd_float = float(odd)
            if 1.1 <= odd_float <= 50:
                value_score = 1.0
            elif 1.05 <= odd_float < 1.1 or 50 < odd_float <= 100:
                value_score = 0.7
            else:
                value_score = 0.3
            quality_score += value_score * 0.2
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.debug(f"⚠️  Errore calcolo quality score: {e}")
            return 0.5  # Score neutro in caso di errore
    
    def _validate_market_implied_probabilities(self, odds_dict: Dict[str, Any], market_type: str = "1x2") -> bool:
        """
        🎯 VALIDAZIONE PROBABILITÀ IMPLICITE: Verifica che le quote di un mercato siano coerenti.
        
        Calcola la somma delle probabilità implicite (1/odd) per tutte le quote di un mercato.
        Per un mercato equilibrato, la somma dovrebbe essere tra 1.01 e 1.08 (margine del bookmaker).
        
        Args:
            odds_dict: Dict con le quote del mercato (es. {'home': 2.5, 'draw': 3.0, 'away': 2.8})
            market_type: Tipo di mercato ('1x2', 'over_under', 'btts', etc.)
        
        Returns:
            True se le quote sono coerenti, False altrimenti
        """
        if not odds_dict:
            return False
        
        try:
            total_implied_prob = Decimal('0')
            valid_outcomes = 0
            
            for outcome, odd_value in odds_dict.items():
                if odd_value is None:
                    continue
                
                odd_decimal = self._validate_odds(odd_value)
                if odd_decimal is None:
                    continue
                
                # Probabilità implicita = 1 / quota
                implied_prob = Decimal('1') / odd_decimal
                total_implied_prob += implied_prob
                valid_outcomes += 1
            
            if valid_outcomes == 0:
                return False
            
            total_implied_prob_float = float(total_implied_prob)
            
            # 🎯 VALIDAZIONE MIGLIORATA: Range diversi per tipo di mercato
            # Per mercati 1X2: margine tipico 1.03-1.07
            # Per Over/Under: margine tipico 1.04-1.08 (più variabilità)
            # Per BTTS: margine tipico 1.03-1.08
            
            if 'over_under' in market_type.lower() or 'goals' in market_type.lower():
                # Over/Under: accetta range più ampio
                min_total = Decimal('1.02')
                max_total = Decimal('1.10')  # Leggermente più permissivo per live
            else:
                # 1X2, BTTS, etc.: range più stretto
                min_total = Decimal('1.01')
                max_total = Decimal('1.08')
            
            if total_implied_prob < min_total:
                logger.debug(
                    f"⚠️  {market_type}: Probabilità implicita totale molto bassa ({total_implied_prob_float:.4f}), "
                    f"possibile errore o opportunità"
                )
                # Accettiamo comunque se è molto vicino a 1.0 (potrebbe essere un'opportunità rara)
                # Ma solo se abbiamo almeno 2 outcomes (altrimenti potrebbe essere errore)
                if valid_outcomes >= 2:
                    return total_implied_prob >= Decimal('0.98')
                return False
            
            if total_implied_prob > max_total:
                logger.warning(
                    f"⚠️  {market_type}: Probabilità implicita totale troppo alta ({total_implied_prob_float:.4f} > {float(max_total):.2f}), "
                    f"quote probabilmente errate o margine troppo alto. Scarto."
                )
                return False
            
            # 🆕 VALIDAZIONE AGGIUNTIVA: Controlla se le quote sono troppo vicine tra loro
            # (potrebbe indicare che non sono state aggiornate o sono errate)
            if valid_outcomes >= 2:
                odds_list = [float(self._validate_odds(v)) for v in odds_dict.values() if v is not None]
                odds_list = [o for o in odds_list if o is not None]
                if len(odds_list) >= 2:
                    odds_range = max(odds_list) - min(odds_list)
                    odds_avg = statistics.mean(odds_list)
                    # Se le quote sono troppo vicine (< 5% di variazione), potrebbe essere sospetto
                    if odds_avg > 0 and (odds_range / odds_avg) < 0.05:
                        logger.debug(
                            f"🔍 {market_type}: Quote molto vicine tra loro (range={odds_range:.3f}, avg={odds_avg:.3f}), "
                            f"possibile problema, ma accetto se probabilità implicita è valida"
                        )
                        # Accettiamo comunque se la probabilità implicita è valida
            
            # Se siamo nel range accettabile
            return True
            
        except Exception as e:
            logger.debug(f"⚠️  Errore validazione probabilità implicite per {market_type}: {e}")
            return True  # In caso di errore, accettiamo per non escludere quote valide
    
    def _select_realistic_odds(self, odds_dict: Dict[str, Decimal], market_name: str = "unknown") -> Tuple[Optional[Decimal], Optional[str]]:
        """
        🎯 SELEZIONE INTELLIGENTE QUOTE: Seleziona una quota "realistica" evitando outlier.
        
        Strategia MIGLIORATA:
        1. Raccoglie tutte le quote valide
        2. Filtra per bookmaker affidabili (preferenza)
        3. Calcola statistiche (media, mediana, deviazione standard)
        4. Filtra outlier (> 2 deviazioni standard dalla media)
        5. Calcola media ponderata per affidabilità bookmaker
        6. Seleziona tra media ponderata, 75° percentile o mediana
        
        Args:
            odds_dict: Dict {bookmaker_name: quota} con tutte le quote disponibili
            market_name: Nome del mercato per logging
        
        Returns:
            Tuple (quota_selezionata, bookmaker_name) o (None, None) se nessuna quota valida
        """
        if not odds_dict:
            return None, None
        
        # Raccogli tutte le quote valide con i loro pesi
        valid_odds: List[Tuple[Decimal, str, float]] = []
        for bookmaker, odd in odds_dict.items():
            validated = self._validate_odds(odd)
            if validated is not None:
                weight = self._get_bookmaker_weight(bookmaker)
                valid_odds.append((validated, bookmaker, weight))
        
        if not valid_odds:
            return None, None
        
        # Se c'è solo una quota valida, usala
        if len(valid_odds) == 1:
            return valid_odds[0][0], valid_odds[0][1]
        
        # Separa quote da bookmaker affidabili (peso >= 0.75) e altri
        trusted_odds = [(odd, bm, w) for odd, bm, w in valid_odds if w >= 0.75]
        other_odds = [(odd, bm, w) for odd, bm, w in valid_odds if w < 0.75]
        
        # Estrai solo i valori numerici per calcoli statistici
        odds_values = [float(odd) for odd, _, _ in valid_odds]
        trusted_odds_values = [float(odd) for odd, _, _ in trusted_odds] if trusted_odds else []
        
        # Calcola statistiche
        mean_odds = statistics.mean(odds_values)
        median_odds = statistics.median(odds_values)
        
        # Calcola deviazione standard (se ci sono almeno 2 quote)
        if len(odds_values) >= 2:
            try:
                std_dev = statistics.stdev(odds_values)
            except statistics.StatisticsError:
                std_dev = 0
        else:
            std_dev = 0
        
        # Filtra outlier: rimuovi quote > 2 deviazioni standard dalla media
        filtered_odds = []
        outlier_threshold = mean_odds + (2 * std_dev) if std_dev > 0 else float('inf')
        outlier_threshold_low = mean_odds - (2 * std_dev) if std_dev > 0 else 0.0
        
        for odd, bookmaker, weight in valid_odds:
            odd_float = float(odd)
            # Accetta quote nel range [mean - 2*std, mean + 2*std]
            if outlier_threshold_low <= odd_float <= outlier_threshold:
                filtered_odds.append((odd, bookmaker, weight))
        
        # Se tutte le quote sono outlier, usa comunque quelle da bookmaker affidabili
        if not filtered_odds:
            if trusted_odds:
                # Usa la migliore tra i bookmaker affidabili
                best_trusted = max(trusted_odds, key=lambda x: (x[2], x[0]))  # Max peso, poi quota
                logger.warning(
                    f"⚠️  QUOTE ANOMALE per {market_name}: tutte le quote sono outlier. "
                    f"Uso bookmaker affidabile: {float(best_trusted[0]):.3f} da {best_trusted[1]} "
                    f"(media={mean_odds:.3f})"
                )
                return best_trusted[0], best_trusted[1]
            else:
                # Fallback: usa la migliore disponibile
                max_odd, max_bookmaker = max(valid_odds, key=lambda x: x[0])
                diff_pct = 0.0
                if mean_odds > 0:
                    diff_pct = ((float(max_odd) - mean_odds) / mean_odds) * 100
                logger.warning(
                    f"⚠️  QUOTE ANOMALE per {market_name}: tutte le quote sono outlier "
                    f"(media={mean_odds:.3f}, max={float(max_odd):.3f}, diff={diff_pct:.1f}%). "
                    f"Uso comunque la migliore: {float(max_odd):.3f} da {max_bookmaker}"
                )
                return max_odd, max_bookmaker
        
        # 🆕 NUOVO: Calcola media ponderata per affidabilità bookmaker
        # Le quote da bookmaker più affidabili hanno più peso
        weighted_sum = Decimal('0')
        total_weight = Decimal('0')
        
        for odd, bookmaker, weight in filtered_odds:
            weighted_sum += odd * Decimal(str(weight))
            total_weight += Decimal(str(weight))
        
        weighted_average = float(weighted_sum / total_weight) if total_weight > 0 else mean_odds
        
        # Filtra quote trusted che sono nel range filtrato
        filtered_trusted = [(odd, bm, w) for odd, bm, w in filtered_odds if w >= 0.75]
        
        # 🆕 Calcola quality score per ogni quota filtrata
        has_trusted = len(filtered_trusted) > 0
        filtered_odds_with_quality = []
        for odd, bookmaker, weight in filtered_odds:
            quality_score = self._calculate_odds_quality_score(
                odd, bookmaker, len(filtered_odds), has_trusted
            )
            filtered_odds_with_quality.append((odd, bookmaker, weight, quality_score))
        
        # Strategia di selezione:
        # 1. Se ci sono bookmaker affidabili filtrati, preferisci la loro media ponderata o migliore
        # 2. Altrimenti usa 75° percentile o mediana, considerando anche quality score
        if filtered_trusted:
            # Calcola media ponderata solo per bookmaker affidabili
            trusted_weighted_sum = Decimal('0')
            trusted_total_weight = Decimal('0')
            
            for odd, bookmaker, weight in filtered_trusted:
                trusted_weighted_sum += odd * Decimal(str(weight))
                trusted_total_weight += Decimal(str(weight))
            
            trusted_weighted_avg = float(trusted_weighted_sum / trusted_total_weight) if trusted_total_weight > 0 else None
            
            # 🆕 Trova la quota trusted con il miglior quality score che è vicina alla media
            if trusted_weighted_avg:
                # Calcola quality score per trusted odds
                trusted_with_quality = []
                for odd, bookmaker, weight in filtered_trusted:
                    quality_score = self._calculate_odds_quality_score(
                        odd, bookmaker, len(filtered_odds), True
                    )
                    diff_from_avg = abs(float(odd) - trusted_weighted_avg) / trusted_weighted_avg * 100
                    # Combina quality score e prossimità alla media
                    combined_score = quality_score * 0.6 + (1.0 - min(diff_from_avg / 10.0, 1.0)) * 0.4
                    trusted_with_quality.append((odd, bookmaker, weight, quality_score, combined_score, diff_from_avg))
                
                # Seleziona la quota con il miglior combined score
                if trusted_with_quality:
                    best_trusted = max(trusted_with_quality, key=lambda x: x[4])  # Max combined score
                    odd, bookmaker, weight, quality, combined, diff_avg = best_trusted
                    
                    # Usa se è entro 5% dalla media o se ha quality score molto alto (> 0.8)
                    if diff_avg < 5.0 or quality > 0.8:
                        logger.info(
                            f"✅ {market_name}: Selezionata quota da bookmaker affidabile "
                            f"{float(odd):.3f} da {bookmaker} "
                            f"(quality={quality:.2f}, media trusted={trusted_weighted_avg:.3f}, diff={diff_avg:.1f}%)"
                        )
                        return odd, bookmaker
        
        # Strategia fallback: usa 75° percentile o mediana, considerando quality score
        # 🆕 Se abbiamo quality scores, preferisci quote con quality score alto
        if filtered_odds_with_quality and len(filtered_odds_with_quality) >= 2:
            # Ordina per quality score (decrescente), poi per quota (decrescente)
            sorted_by_quality = sorted(
                filtered_odds_with_quality,
                key=lambda x: (x[3], float(x[0])),  # (quality_score, odd)
                reverse=True
            )
            
            # Prendi le top 3 quote per quality score
            top_quality_odds = sorted_by_quality[:min(3, len(sorted_by_quality))]
            
            # Tra queste, scegli quella con la quota più alta (più competitiva)
            if top_quality_odds:
                selected_odd, selected_bookmaker, selected_weight, selected_quality = max(
                    top_quality_odds,
                    key=lambda x: float(x[0])  # Max quota tra le top quality
                )
                logger.info(
                    f"✅ {market_name}: Selezionata quota con quality score alto "
                    f"{float(selected_odd):.3f} da {selected_bookmaker} "
                    f"(quality={selected_quality:.2f})"
                )
                return selected_odd, selected_bookmaker
        
        # Fallback: usa 75° percentile o mediana (strategia originale)
        sorted_odds = sorted(filtered_odds, key=lambda x: float(x[0]))
        
        # Calcola 75° percentile
        percentile_75_idx = int(len(sorted_odds) * 0.75)
        if percentile_75_idx >= len(sorted_odds):
            percentile_75_idx = len(sorted_odds) - 1
        
        selected_odd, selected_bookmaker, selected_weight = sorted_odds[percentile_75_idx]
        
        # Se la differenza tra 75° percentile e max è < 5%, preferisci la max (più competitiva)
        max_odd, max_bookmaker, max_weight = sorted_odds[-1]
        diff_pct = float(((max_odd - selected_odd) / selected_odd) * 100) if selected_odd > 0 else 0.0
        
        if diff_pct < 5.0 and len(filtered_odds) >= 3:
            # Usa la quota massima se è vicina al 75° percentile (non è un outlier)
            selected_odd, selected_bookmaker, selected_weight = max_odd, max_bookmaker, max_weight
        
        # Log dettagliato
        outliers_count = len(valid_odds) - len(filtered_odds)
        trusted_count = len([x for x in filtered_odds if x[2] >= 0.75])
        selected_quality = next(
            (q for o, b, w, q in filtered_odds_with_quality 
             if o == selected_odd and b == selected_bookmaker),
            0.0
        ) if filtered_odds_with_quality else 0.0
        
        logger.info(
            f"📊 {market_name}: {outliers_count} outlier filtrati su {len(valid_odds)} quote. "
            f"Media={mean_odds:.3f}, Mediana={median_odds:.3f}, MediaPonderata={weighted_average:.3f}, "
            f"BookmakerAffidabili={trusted_count}/{len(filtered_odds)}, "
            f"Selezionata={float(selected_odd):.3f} (75° percentile) da {selected_bookmaker} "
            f"(affidabilità={selected_weight:.2f}, quality={selected_quality:.2f})"
        )
        
        return selected_odd, selected_bookmaker
    
    def _build_precision_snapshot(self, all_odds: Dict[str, Any], bookmaker_tracker: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea un riepilogo di precisione per tutte le quote presenti, includendo decimali esatti e bookmaker.
        """
        snapshot_time = datetime.now(timezone.utc).isoformat()
        snapshot: Dict[str, Any] = {
            'generated_at': snapshot_time,
            'source': 'api-football',
            'markets': {}
        }
        
        def _store_precision(market: str, qualifier: str, value: Any, bookmaker: Optional[str]):
            if value is None:
                return
            try:
                decimal_value = Decimal(str(value))
            except (InvalidOperation, ValueError):
                return
            market_entry = snapshot['markets'].setdefault(market, {})
            market_entry[qualifier] = {
                'decimal': format(decimal_value, 'f'),
                'bookmaker': bookmaker,
                'updated_at': snapshot_time
            }
        
        for outcome in ['home', 'draw', 'away']:
            _store_precision('match_winner', outcome, all_odds.get('match_winner', {}).get(outcome), bookmaker_tracker['match_winner'].get(outcome))
        
        for market_key in ['over_under', 'over_under_ht', 'first_half_goals', 'second_half_goals']:
            market_dict = all_odds.get(market_key, {})
            tracker_dict = bookmaker_tracker.get(market_key, {})
            for threshold, odds in market_dict.items():
                for outcome_type in ['over', 'under']:
                    qualifier = f"{threshold}:{outcome_type}"
                    bm = tracker_dict.get(threshold, {}).get(outcome_type) if isinstance(tracker_dict.get(threshold), dict) else None
                    _store_precision(market_key, qualifier, odds.get(outcome_type), bm)
        
        for market_key in ['btts', 'btts_ht']:
            for outcome in ['yes', 'no']:
                _store_precision(market_key, outcome, all_odds.get(market_key, {}).get(outcome), bookmaker_tracker.get(market_key, {}).get(outcome))
        
        for outcome in ['1x', '12', 'x2']:
            _store_precision('double_chance', outcome, all_odds.get('double_chance', {}).get(outcome), bookmaker_tracker.get('double_chance', {}).get(outcome))
        
        for outcome in ['home', 'away']:
            _store_precision('draw_no_bet', outcome, all_odds.get('draw_no_bet', {}).get(outcome), bookmaker_tracker.get('draw_no_bet', {}).get(outcome))
        
        if all_odds.get('asian_handicap'):
            for handicap, odd_value in all_odds['asian_handicap'].items():
                _store_precision('asian_handicap', handicap, odd_value, bookmaker_tracker.get('asian_handicap', {}).get(handicap))
        
        return snapshot
    
    def _run_odds_precision_watchdog(self, matches: List[Dict], freshness_seconds: int = 120):
        """Verifica che le quote di ogni match siano fresche e complete, altrimenti forza un refresh."""
        if not matches:
            return
        
        now = datetime.now(timezone.utc)
        for match in matches:
            precision_meta = match.get('all_odds_precision') or match.get('all_odds', {}).get('_precision_snapshot')
            fixture_id = match.get('fixture_id') or match.get('id')
            
            if not precision_meta:
                logger.warning(f"🕵️  Watchdog: fixture {fixture_id} senza precision metadata, provo refresh quote")
                self._refresh_match_odds(match)
                continue
            
            generated_at = precision_meta.get('generated_at')
            markets_meta = precision_meta.get('markets', {})
            stale = False
            age_seconds = None
            if generated_at:
                try:
                    generated_dt = datetime.fromisoformat(generated_at)
                    if generated_dt.tzinfo is None:
                        generated_dt = generated_dt.replace(tzinfo=timezone.utc)
                    age_seconds = (now - generated_dt).total_seconds()
                    if age_seconds > freshness_seconds:
                        stale = True
                except ValueError:
                    stale = True
            else:
                stale = True
            
            match_winner_meta = markets_meta.get('match_winner', {})
            missing_core = not all(match_winner_meta.get(outcome) for outcome in ['home', 'draw', 'away'])
            
            if not stale and not missing_core:
                continue
            
            reasons = []
            if stale:
                if age_seconds is not None:
                    reasons.append(f"stale {int(age_seconds)}s")
                else:
                    reasons.append("stale")
            if missing_core:
                reasons.append("missing_1x2")
            
            logger.warning(f"🕵️  Odds Precision Watchdog: fixture {fixture_id} richiede refresh ({', '.join(reasons)})")
            self._refresh_match_odds(match)
    
    def _refresh_match_odds(self, match: Dict[str, Any]):
        """Forza il refresh delle quote per un singolo match aggiornando il dict in-place."""
        fixture_id = match.get('fixture_id') or match.get('id')
        if not fixture_id:
            logger.debug("🕵️  Watchdog: impossibile refresh, fixture_id mancante")
            return
        
        api_key = os.getenv("API_FOOTBALL_KEY", "")
        if not api_key:
            logger.warning("🕵️  Watchdog: impossibile refresh quote, API_FOOTBALL_KEY non configurata")
            return
        
        base_url = "https://v3.football.api-sports.io"
        try:
            odds_data = self._fetch_fixture_odds_from_api_football(int(fixture_id), api_key, base_url)
        except Exception as e:
            logger.warning(f"🕵️  Watchdog: errore imprevisto refresh quote fixture {fixture_id}: {e}")
            return
        
        if not odds_data:
            logger.warning(f"🕵️  Watchdog: nessuna quota disponibile per fixture {fixture_id} durante il refresh")
            return
        
        refreshed_odds = self._extract_all_odds_from_api_football(odds_data)
        if not refreshed_odds:
            logger.warning(f"🕵️  Watchdog: estrazione quote fallita per fixture {fixture_id}")
            return
        
        self.api_usage_today += 1
        match['all_odds'] = refreshed_odds
        match['all_odds_precision'] = refreshed_odds.get('_precision_snapshot')
        if refreshed_odds.get('match_winner'):
            match['odds_1'] = refreshed_odds['match_winner'].get('home')
            match['odds_x'] = refreshed_odds['match_winner'].get('draw')
            match['odds_2'] = refreshed_odds['match_winner'].get('away')
        
        logger.info(f"🔁 Watchdog: quote aggiornate per fixture {fixture_id}")
    
    def _retry_api_call(self, func, max_retries: int = 3, base_delay: float = 1.0, *args, **kwargs):
        """
        🎯 RETRY LOGIC: Esegue una chiamata API con retry e backoff esponenziale.
        
        Args:
            func: Funzione da eseguire (deve essere una funzione che fa chiamate API)
            max_retries: Numero massimo di tentativi (default: 3)
            base_delay: Delay iniziale in secondi (default: 1.0)
            *args, **kwargs: Argomenti da passare alla funzione
        
        Returns:
            Risultato della funzione, o None se tutti i tentativi falliscono
        """
        import urllib.error
        import urllib.request
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    if attempt > 0:
                        logger.info(f"✅ API call riuscita al tentativo {attempt + 1}/{max_retries}")
                    return result
                # Se result è None, riprova (potrebbe essere un errore silenzioso)
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Backoff esponenziale: 1s, 2s, 4s
                    logger.warning(f"⚠️  API call restituito None, retry {attempt + 1}/{max_retries} tra {delay:.1f}s")
                    time.sleep(delay)
                    
            except urllib.error.HTTPError as e:
                last_exception = e
                # Per errori HTTP, controlla se è un errore recuperabile
                if e.code in [429, 500, 502, 503, 504]:  # Rate limit, server errors
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"⚠️  HTTP error {e.code} al tentativo {attempt + 1}/{max_retries}, "
                            f"retry tra {delay:.1f}s: {e.reason}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"❌ HTTP error {e.code} dopo {max_retries} tentativi: {e.reason}")
                else:
                    # Errori non recuperabili (404, 401, ecc.) - non riprovare
                    logger.error(f"❌ HTTP error non recuperabile {e.code}: {e.reason}")
                    return None
                    
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"⚠️  Errore connessione al tentativo {attempt + 1}/{max_retries}, "
                        f"retry tra {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Errore connessione dopo {max_retries} tentativi: {e}")
                    
            except Exception as e:
                last_exception = e
                logger.error(f"❌ Errore inatteso al tentativo {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
        
        logger.error(f"❌ Tutti i {max_retries} tentativi falliti, ultimo errore: {last_exception}")
        return None
    
    def _extract_all_odds_from_api_football(self, odds_list: List[Dict]) -> Dict[str, Any]:
        """
        Estrae TUTTE le quote disponibili da API-Football.
        
        API-Football restituisce una lista di bookmaker, ognuno con i suoi mercati.
        Estrae le migliori quote per ogni mercato disponibile.
        
        Mercati cercati:
        - Match Winner (1X2) - id: 1
        - Over/Under FT/HT - id: 5
        - BTTS FT/HT - id: 8
        - First Half Goals - id: 16
        - Second Half Goals - id: 17
        - Asian Handicap - id: 2
        - Double Chance - id: 12
        - Draw No Bet - id: 13
        """
        import re  # Import una sola volta
        
        all_odds = {
            'match_winner': {'home': None, 'draw': None, 'away': None},
            'over_under': {},  # FT (Full Time)
            'over_under_ht': {},  # HT (First Half)
            'btts': {'yes': None, 'no': None},
            'btts_ht': {'yes': None, 'no': None},  # BTTS First Half
            'double_chance': {'1x': None, '12': None, 'x2': None},
            'draw_no_bet': {'home': None, 'away': None},
            'asian_handicap': {},
            'first_half_goals': {},  # First Half Goals Over/Under
            'second_half_goals': {},  # Second Half Goals Over/Under
            'other_markets': {}  # Altri mercati non categorizzati
        }
        
        # 🔧 NUOVO: Traccia quale bookmaker fornisce ogni quota
        bookmaker_tracker = {
            'match_winner': {'home': None, 'draw': None, 'away': None},
            'over_under': {},
            'over_under_ht': {},
            'btts': {'yes': None, 'no': None},
            'btts_ht': {'yes': None, 'no': None},
            'double_chance': {'1x': None, '12': None, 'x2': None},
            'draw_no_bet': {'home': None, 'away': None},
            'asian_handicap': {},
            'first_half_goals': {},
            'second_half_goals': {},
            'other_markets': {}
        }
        
        if not odds_list:
            return all_odds
        
        # 🔧 OPZIONE 4: Raccogli tutte le quote da tutti i bookmaker (non solo la migliore)
        # Struttura: {mercato: {outcome: {bookmaker: quota}}}
        all_bookmaker_odds = {
            'match_winner': {'home': {}, 'draw': {}, 'away': {}},
            'over_under': {},
            'over_under_ht': {},
            'first_half_goals': {},
            'second_half_goals': {},
            'btts': {'yes': {}, 'no': {}},
            'btts_ht': {'yes': {}, 'no': {}},
            'double_chance': {'1x': {}, '12': {}, 'x2': {}},
            'draw_no_bet': {'home': {}, 'away': {}},
            'asian_handicap': {}
        }
        
        # Itera su tutti i bookmaker per raccogliere tutte le quote
        for bookmaker in odds_list:
            bookmaker_name = bookmaker.get("bookmaker", {}).get("name", "")
            bets = bookmaker.get("bets", [])
            
            for bet in bets:
                bet_id = bet.get("id")
                bet_name = bet.get("name", "").lower()
                values = bet.get("values", [])
                
                # Match Winner (1X2) - id: 1
                if bet_id == 1 or "match winner" in bet_name or "1x2" in bet_name:
                    for value in values:
                        outcome = value.get("value", "").lower()
                        odd = value.get("odd")
                        # 🎯 PRECISIONE MANIACALE: Validazione rigorosa quote prima di usarle
                        odd_decimal = self._validate_odds(odd)
                        if odd_decimal is None:
                            continue
                        odd_float = float(odd_decimal)
                            
                        if outcome in ["home", "1"]:
                            # Raccogli quota da questo bookmaker
                            all_bookmaker_odds['match_winner']['home'][bookmaker_name] = odd_decimal
                            # Aggiorna se è la migliore (verrà poi sovrascritta da selezione intelligente)
                            if all_odds['match_winner']['home'] is None or odd_float > all_odds['match_winner']['home']:
                                all_odds['match_winner']['home'] = odd_float
                                bookmaker_tracker['match_winner']['home'] = bookmaker_name
                        elif outcome in ["draw", "x"]:
                            all_bookmaker_odds['match_winner']['draw'][bookmaker_name] = odd_decimal
                            if all_odds['match_winner']['draw'] is None or odd_float > all_odds['match_winner']['draw']:
                                all_odds['match_winner']['draw'] = odd_float
                                bookmaker_tracker['match_winner']['draw'] = bookmaker_name
                        elif outcome in ["away", "2"]:
                            all_bookmaker_odds['match_winner']['away'][bookmaker_name] = odd_decimal
                            if all_odds['match_winner']['away'] is None or odd_float > all_odds['match_winner']['away']:
                                all_odds['match_winner']['away'] = odd_float
                                bookmaker_tracker['match_winner']['away'] = bookmaker_name
                
                # Over/Under - id: 5 (può essere FT o HT)
                elif bet_id == 5 or "over/under" in bet_name or "total goals" in bet_name:
                    # Determina se è FT o HT
                    is_ht = "first half" in bet_name or "1st half" in bet_name or "ht" in bet_name or "half time" in bet_name
                    is_ft = "full time" in bet_name or "ft" in bet_name or (not is_ht and "over/under" in bet_name)
                    
                    target_dict = all_odds['over_under_ht'] if is_ht else all_odds['over_under']
                    
                    for value in values:
                        outcome = value.get("value", "").lower()
                        odd = value.get("odd")
                        # 🎯 PRECISIONE MANIACALE: Validazione rigorosa quote
                        odd_decimal = self._validate_odds(odd)
                        if odd_decimal is None:
                                continue
                        odd_float = float(odd_decimal)
                        
                        # Estrai threshold da qualsiasi valore (non solo hardcoded)
                        threshold = None
                        # Cerca pattern numerici come "0.5", "1.5", "2.5", ecc.
                        threshold_match = re.search(r'(\d+\.?\d*)', outcome)
                        if threshold_match:
                            threshold = threshold_match.group(1)
                        
                        if threshold and odd:
                            # Traccia bookmaker per over/under
                            tracker_key = 'over_under_ht' if is_ht else 'over_under'
                            if threshold not in bookmaker_tracker[tracker_key]:
                                bookmaker_tracker[tracker_key][threshold] = {'over': None, 'under': None}
                            
                            if "over" in outcome:
                                if threshold not in target_dict:
                                    target_dict[threshold] = {'over': None, 'under': None}
                                if target_dict[threshold]['over'] is None or odd_float > target_dict[threshold]['over']:
                                    target_dict[threshold]['over'] = odd_float
                                    bookmaker_tracker[tracker_key][threshold]['over'] = bookmaker_name
                            elif "under" in outcome:
                                if threshold not in target_dict:
                                    target_dict[threshold] = {'over': None, 'under': None}
                                if target_dict[threshold]['under'] is None or odd_float > target_dict[threshold]['under']:
                                    target_dict[threshold]['under'] = odd_float
                                    bookmaker_tracker[tracker_key][threshold]['under'] = bookmaker_name
                
                # First Half Goals - id: 16 o varianti
                elif bet_id == 16 or "first half goals" in bet_name or "1st half goals" in bet_name:
                    for value in values:
                        outcome = value.get("value", "").lower()
                        odd = value.get("odd")
                        if odd:
                            try:
                                odd = float(odd) if isinstance(odd, str) else odd
                            except (ValueError, TypeError):
                                continue
                        
                        threshold_match = re.search(r'(\d+\.?\d*)', outcome)
                        if threshold_match:
                            threshold = threshold_match.group(1)
                            if threshold and odd:
                                if threshold not in bookmaker_tracker['first_half_goals']:
                                    bookmaker_tracker['first_half_goals'][threshold] = {'over': None, 'under': None}
                                if threshold not in all_bookmaker_odds['first_half_goals']:
                                    all_bookmaker_odds['first_half_goals'][threshold] = {'over': {}, 'under': {}}
                                
                                if "over" in outcome:
                                    all_bookmaker_odds['first_half_goals'][threshold]['over'][bookmaker_name] = odd
                                    if threshold not in all_odds['first_half_goals']:
                                        all_odds['first_half_goals'][threshold] = {'over': None, 'under': None}
                                    if all_odds['first_half_goals'][threshold]['over'] is None or odd > all_odds['first_half_goals'][threshold]['over']:
                                        all_odds['first_half_goals'][threshold]['over'] = odd
                                        bookmaker_tracker['first_half_goals'][threshold]['over'] = bookmaker_name
                                elif "under" in outcome:
                                    all_bookmaker_odds['first_half_goals'][threshold]['under'][bookmaker_name] = odd
                                    if threshold not in all_odds['first_half_goals']:
                                        all_odds['first_half_goals'][threshold] = {'over': None, 'under': None}
                                    if all_odds['first_half_goals'][threshold]['under'] is None or odd > all_odds['first_half_goals'][threshold]['under']:
                                        all_odds['first_half_goals'][threshold]['under'] = odd
                                        bookmaker_tracker['first_half_goals'][threshold]['under'] = bookmaker_name
                
                # Second Half Goals - id: 17 o varianti
                elif bet_id == 17 or "second half goals" in bet_name or "2nd half goals" in bet_name:
                    for value in values:
                        outcome = value.get("value", "").lower()
                        odd = value.get("odd")
                        if odd:
                            try:
                                odd = float(odd) if isinstance(odd, str) else odd
                            except (ValueError, TypeError):
                                continue
                        
                        threshold_match = re.search(r'(\d+\.?\d*)', outcome)
                        if threshold_match:
                            threshold = threshold_match.group(1)
                            if threshold and odd:
                                if threshold not in bookmaker_tracker['second_half_goals']:
                                    bookmaker_tracker['second_half_goals'][threshold] = {'over': None, 'under': None}
                                if threshold not in all_bookmaker_odds['second_half_goals']:
                                    all_bookmaker_odds['second_half_goals'][threshold] = {'over': {}, 'under': {}}
                                
                                if "over" in outcome:
                                    all_bookmaker_odds['second_half_goals'][threshold]['over'][bookmaker_name] = odd
                                    if threshold not in all_odds['second_half_goals']:
                                        all_odds['second_half_goals'][threshold] = {'over': None, 'under': None}
                                    if all_odds['second_half_goals'][threshold]['over'] is None or odd > all_odds['second_half_goals'][threshold]['over']:
                                        all_odds['second_half_goals'][threshold]['over'] = odd
                                        bookmaker_tracker['second_half_goals'][threshold]['over'] = bookmaker_name
                                elif "under" in outcome:
                                    all_bookmaker_odds['second_half_goals'][threshold]['under'][bookmaker_name] = odd
                                    if threshold not in all_odds['second_half_goals']:
                                        all_odds['second_half_goals'][threshold] = {'over': None, 'under': None}
                                    if all_odds['second_half_goals'][threshold]['under'] is None or odd > all_odds['second_half_goals'][threshold]['under']:
                                        all_odds['second_half_goals'][threshold]['under'] = odd
                                        bookmaker_tracker['second_half_goals'][threshold]['under'] = bookmaker_name
                
                # BTTS - id: 8 (può essere FT o HT)
                elif bet_id == 8 or "both teams to score" in bet_name or "btts" in bet_name:
                    # Determina se è FT o HT
                    is_ht = "first half" in bet_name or "1st half" in bet_name or "ht" in bet_name
                    target_dict = all_odds['btts_ht'] if is_ht else all_odds['btts']
                    
                    for value in values:
                        outcome = value.get("value", "").lower()
                        odd = value.get("odd")
                        # Converti odd a float se è stringa
                        if odd:
                            try:
                                odd = float(odd) if isinstance(odd, str) else odd
                            except (ValueError, TypeError):
                                continue
                        
                        if odd:
                            if "yes" in outcome or "1" in outcome:
                                if target_dict['yes'] is None or odd > target_dict['yes']:
                                    target_dict['yes'] = odd
                            elif "no" in outcome or "0" in outcome:
                                if target_dict['no'] is None or odd > target_dict['no']:
                                    target_dict['no'] = odd
                
                # Double Chance - id: 12
                elif bet_id == 12 or "double chance" in bet_name:
                    for value in values:
                        outcome = value.get("value", "").lower()
                        odd = value.get("odd")
                        # Converti odd a float se è stringa
                        if odd:
                            try:
                                odd = float(odd) if isinstance(odd, str) else odd
                            except (ValueError, TypeError):
                                continue
                        
                        if odd:
                            if "1x" in outcome or "home or draw" in outcome:
                                if all_odds['double_chance']['1x'] is None or odd > all_odds['double_chance']['1x']:
                                    all_odds['double_chance']['1x'] = odd
                            elif "12" in outcome or "home or away" in outcome:
                                if all_odds['double_chance']['12'] is None or odd > all_odds['double_chance']['12']:
                                    all_odds['double_chance']['12'] = odd
                            elif "x2" in outcome or "draw or away" in outcome:
                                if all_odds['double_chance']['x2'] is None or odd > all_odds['double_chance']['x2']:
                                    all_odds['double_chance']['x2'] = odd
                
                # Draw No Bet - id: 13
                elif bet_id == 13 or "draw no bet" in bet_name:
                    for value in values:
                        outcome = value.get("value", "").lower()
                        odd_decimal = self._validate_odds(value.get("odd"))
                        if odd_decimal is None:
                                continue
                        odd_float = float(odd_decimal)
                        
                        if odd_float:
                            if "home" in outcome or "1" in outcome:
                                if all_odds['draw_no_bet']['home'] is None or odd_float > all_odds['draw_no_bet']['home']:
                                    all_odds['draw_no_bet']['home'] = odd_float
                            elif "away" in outcome or "2" in outcome:
                                if all_odds['draw_no_bet']['away'] is None or odd_float > all_odds['draw_no_bet']['away']:
                                    all_odds['draw_no_bet']['away'] = odd_float
                
                # Asian Handicap - id: 2
                elif bet_id == 2 or "asian handicap" in bet_name:
                    for value in values:
                        outcome = value.get("value", "")
                        odd_decimal = self._validate_odds(value.get("odd"))
                        if odd_decimal is None:
                                continue
                        odd_float = float(odd_decimal)
                        
                        if odd_float and outcome:
                            # Salva con il valore dell'handicap come chiave
                            if outcome not in all_odds['asian_handicap']:
                                all_odds['asian_handicap'][outcome] = odd_float
                            elif odd_float > all_odds['asian_handicap'][outcome]:
                                all_odds['asian_handicap'][outcome] = odd_float
                
                # Altri mercati non categorizzati - salva per riferimento futuro
                else:
                    # Salva il mercato per riferimento futuro
                    market_key = f"{bet_id}_{bet_name}"
                    if market_key not in all_odds['other_markets']:
                        all_odds['other_markets'][market_key] = {
                            'id': bet_id,
                            'name': bet_name,
                            'values': []
                        }
                    # Aggiungi valori se interessanti
                    for value in values:
                        outcome = value.get("value", "")
                        odd = value.get("odd")
                        if odd:
                            try:
                                odd = float(odd) if isinstance(odd, str) else odd
                                all_odds['other_markets'][market_key]['values'].append({
                                    'outcome': outcome,
                                    'odd': odd
                                })
                            except (ValueError, TypeError):
                                continue
        
        # 🔧 NUOVO: Calcola numero di bookmaker disponibili per ogni mercato/outcome
        bookmaker_counts = {
            'match_winner': {'home': 0, 'draw': 0, 'away': 0},
            'over_under': {},
            'over_under_ht': {},
            'first_half_goals': {},
            'second_half_goals': {},
            'btts': {'yes': 0, 'no': 0},
            'btts_ht': {'yes': 0, 'no': 0},
            'double_chance': {'1x': 0, '12': 0, 'x2': 0},
            'draw_no_bet': {'home': 0, 'away': 0}
        }
        bookmaker_counts_flat = {}

        for outcome in ['home', 'draw', 'away']:
            count = len(all_bookmaker_odds['match_winner'][outcome])
            bookmaker_counts['match_winner'][outcome] = count
            bookmaker_counts_flat[f"match_winner:{outcome}"] = count

        def _populate_counts_for_threshold(source_dict, target_key: str):
            for threshold, outcome_dict in source_dict.items():
                if threshold not in bookmaker_counts[target_key]:
                    bookmaker_counts[target_key][threshold] = {}
                for outcome_type in ['over', 'under']:
                    count = len(outcome_dict.get(outcome_type, {})) if isinstance(outcome_dict, dict) else 0
                    bookmaker_counts[target_key][threshold][outcome_type] = count
                    bookmaker_counts_flat[f"{target_key}:{threshold}:{outcome_type}"] = count

        _populate_counts_for_threshold(all_bookmaker_odds['over_under'], 'over_under')
        _populate_counts_for_threshold(all_bookmaker_odds['over_under_ht'], 'over_under_ht')
        _populate_counts_for_threshold(all_bookmaker_odds['first_half_goals'], 'first_half_goals')
        _populate_counts_for_threshold(all_bookmaker_odds['second_half_goals'], 'second_half_goals')

        for outcome in ['yes', 'no']:
            count = len(all_bookmaker_odds['btts'][outcome])
            bookmaker_counts['btts'][outcome] = count
            bookmaker_counts_flat[f"btts:{outcome}"] = count

            count_ht = len(all_bookmaker_odds['btts_ht'][outcome])
            bookmaker_counts['btts_ht'][outcome] = count_ht
            bookmaker_counts_flat[f"btts_ht:{outcome}"] = count_ht

        for outcome in ['1x', '12', 'x2']:
            count = len(all_bookmaker_odds['double_chance'][outcome])
            bookmaker_counts['double_chance'][outcome] = count
            bookmaker_counts_flat[f"double_chance:{outcome}"] = count

        for outcome in ['home', 'away']:
            count = len(all_bookmaker_odds['draw_no_bet'][outcome])
            bookmaker_counts['draw_no_bet'][outcome] = count
            bookmaker_counts_flat[f"draw_no_bet:{outcome}"] = count
        
        # 🎯 NUOVO: Applica selezione intelligente quote (evita outlier)
        # Sostituisce le quote massime con quote "realistiche" (75° percentile, filtra outlier)
        logger.debug("🔍 Applicazione selezione intelligente quote (filtro outlier)...")
        
        # Match Winner (1X2)
        for outcome in ['home', 'draw', 'away']:
            if all_bookmaker_odds['match_winner'][outcome]:
                selected_odd, selected_bookmaker = self._select_realistic_odds(
                    all_bookmaker_odds['match_winner'][outcome],
                    f"1X2_{outcome}"
                )
                if selected_odd is not None:
                    all_odds['match_winner'][outcome] = float(selected_odd)
                    bookmaker_tracker['match_winner'][outcome] = selected_bookmaker
        
        # Over/Under FT e HT
        for threshold_dict, market_key, odds_key, tracker_key in [
            (all_bookmaker_odds['over_under'], 'over_under', all_odds['over_under'], bookmaker_tracker['over_under']),
            (all_bookmaker_odds['over_under_ht'], 'over_under_ht', all_odds['over_under_ht'], bookmaker_tracker['over_under_ht']),
            (all_bookmaker_odds['first_half_goals'], 'first_half_goals', all_odds['first_half_goals'], bookmaker_tracker['first_half_goals']),
            (all_bookmaker_odds['second_half_goals'], 'second_half_goals', all_odds['second_half_goals'], bookmaker_tracker['second_half_goals'])
        ]:
            for threshold, outcomes in threshold_dict.items():
                for outcome_type in ['over', 'under']:
                    if outcome_type in outcomes and outcomes[outcome_type]:
                        selected_odd, selected_bookmaker = self._select_realistic_odds(
                            outcomes[outcome_type],
                            f"{market_key}_{threshold}_{outcome_type}"
                        )
                        if selected_odd is not None:
                            if threshold not in odds_key:
                                odds_key[threshold] = {'over': None, 'under': None}
                            odds_key[threshold][outcome_type] = float(selected_odd)
                            if threshold not in tracker_key:
                                tracker_key[threshold] = {'over': None, 'under': None}
                            tracker_key[threshold][outcome_type] = selected_bookmaker
        
        # BTTS FT e HT
        for outcome in ['yes', 'no']:
            for market_key, target_dict in [
                ('btts', all_odds['btts']),
                ('btts_ht', all_odds['btts_ht'])
            ]:
                if all_bookmaker_odds[market_key][outcome]:
                    selected_odd, selected_bookmaker = self._select_realistic_odds(
                        all_bookmaker_odds[market_key][outcome],
                        f"{market_key}_{outcome}"
                    )
                    if selected_odd is not None:
                        target_dict[outcome] = float(selected_odd)
        
        # Double Chance
        for outcome in ['1x', '12', 'x2']:
            if all_bookmaker_odds['double_chance'][outcome]:
                selected_odd, selected_bookmaker = self._select_realistic_odds(
                    all_bookmaker_odds['double_chance'][outcome],
                    f"double_chance_{outcome}"
                )
                if selected_odd is not None:
                    all_odds['double_chance'][outcome] = float(selected_odd)
        
        # Draw No Bet
        for outcome in ['home', 'away']:
            if all_bookmaker_odds['draw_no_bet'][outcome]:
                selected_odd, selected_bookmaker = self._select_realistic_odds(
                    all_bookmaker_odds['draw_no_bet'][outcome],
                    f"dnb_{outcome}"
                )
                if selected_odd is not None:
                    all_odds['draw_no_bet'][outcome] = float(selected_odd)
        
        # 🔧 OPZIONE 4: Applica logica ibrida - preferisci bet365 se differenza < 5%
        # Cerca bet365 in tutti i bookmaker (case-insensitive)
        bet365_names = ['bet365', 'bet 365', 'bet-365']
        bet365_odds = {}
        
        def find_bet365_odds(market_dict, market_type):
            """Trova quote bet365 per un mercato"""
            result = {}
            for bookmaker_name, quota in market_dict.items():
                if any(name.lower() in bookmaker_name.lower() for name in bet365_names):
                    result[bookmaker_name] = quota
            return result
        
        # Trova quote bet365 per ogni mercato
        for outcome in ['home', 'draw', 'away']:
            bet365_quota = find_bet365_odds(all_bookmaker_odds['match_winner'][outcome], 'match_winner')
            if bet365_quota:
                bet365_odds[f'match_winner_{outcome}'] = list(bet365_quota.values())[0]  # Prendi la prima (dovrebbe essere una sola)
        
        # Trova quote bet365 per over/under
        for threshold in all_bookmaker_odds['over_under'].keys():
            for outcome_type in ['over', 'under']:
                if threshold in all_bookmaker_odds['over_under'] and outcome_type in all_bookmaker_odds['over_under'][threshold]:
                    bet365_quota = find_bet365_odds(all_bookmaker_odds['over_under'][threshold][outcome_type], 'over_under')
                    if bet365_quota:
                        bet365_odds[f'over_under_{threshold}_{outcome_type}'] = list(bet365_quota.values())[0]
        
        # Trova quote bet365 per second half goals
        for threshold in all_bookmaker_odds['second_half_goals'].keys():
            for outcome_type in ['over', 'under']:
                if threshold in all_bookmaker_odds['second_half_goals'] and outcome_type in all_bookmaker_odds['second_half_goals'][threshold]:
                    bet365_quota = find_bet365_odds(all_bookmaker_odds['second_half_goals'][threshold][outcome_type], 'second_half_goals')
                    if bet365_quota:
                        bet365_odds[f'second_half_goals_{threshold}_{outcome_type}'] = list(bet365_quota.values())[0]
        
        # Applica logica ibrida: se bet365 disponibile e differenza < 5%, usa bet365
        def apply_hybrid_logic(best_odd, bet365_odd_key, market_path, outcome_key=None):
            """Applica logica ibrida: preferisci bet365 se differenza < 5%"""
            if bet365_odd_key not in bet365_odds:
                return best_odd, None  # Nessuna quota bet365 disponibile
            
            bet365_odd = bet365_odds[bet365_odd_key]
            if best_odd is None:
                return bet365_odd, 'bet365'
            
            # Calcola differenza percentuale
            diff_pct = ((best_odd - bet365_odd) / bet365_odd) * 100
            
            if diff_pct < 5.0:  # Differenza < 5%, preferisci bet365
                # Aggiorna all_odds con quota bet365
                if outcome_key:
                    if isinstance(market_path, dict) and outcome_key in market_path:
                        market_path[outcome_key] = bet365_odd
                elif isinstance(market_path, dict) and 'over' in market_path and 'under' in market_path:
                    # Per over/under, devo sapere quale outcome
                    pass  # Gestito separatamente
                return bet365_odd, 'bet365'
            else:
                return best_odd, bookmaker_tracker.get(market_path, {}).get(outcome_key) if outcome_key else None
        
        # Applica logica ibrida per match_winner
        for outcome in ['home', 'draw', 'away']:
            best_odd = all_odds['match_winner'][outcome]
            bet365_key = f'match_winner_{outcome}'
            new_odd, used_bookmaker = apply_hybrid_logic(best_odd, bet365_key, all_odds['match_winner'], outcome)
            if used_bookmaker == 'bet365':
                all_odds['match_winner'][outcome] = new_odd
                bookmaker_tracker['match_winner'][outcome] = 'bet365'
                logger.info(f"✅ Preferita bet365 per 1X2 {outcome}: {new_odd} (differenza < 5% dalla quota migliore {best_odd})")
        
        # Applica logica ibrida per over/under e second_half_goals
        for market_type in ['over_under', 'second_half_goals']:
            market_dict = all_odds[market_type]
            for threshold in market_dict.keys():
                for outcome_type in ['over', 'under']:
                    if outcome_type in market_dict[threshold] and market_dict[threshold][outcome_type] is not None:
                        best_odd = market_dict[threshold][outcome_type]
                        bet365_key = f'{market_type}_{threshold}_{outcome_type}'
                        if bet365_key in bet365_odds:
                            bet365_odd = bet365_odds[bet365_key]
                            diff_pct = ((best_odd - bet365_odd) / bet365_odd) * 100
                            if diff_pct < 5.0:
                                market_dict[threshold][outcome_type] = bet365_odd
                                bookmaker_tracker[market_type][threshold][outcome_type] = 'bet365'
                                logger.info(f"✅ Preferita bet365 per {market_type} {threshold} {outcome_type}: {bet365_odd} (differenza {diff_pct:.1f}% < 5%)")
        
        # 🎯 VALIDAZIONE PROBABILITÀ IMPLICITE: Verifica coerenza delle quote
        # Rimuovi quote incoerenti prima di restituirle
        logger.debug("🔍 Validazione probabilità implicite per coerenza quote...")
        
        # Valida 1X2
        if all_odds['match_winner']:
            match_winner_valid = self._validate_market_implied_probabilities(
                all_odds['match_winner'], '1X2'
            )
            if not match_winner_valid:
                logger.warning(
                    f"⚠️  Quote 1X2 incoerenti (probabilità implicite non valide), "
                    f"mantenute solo se almeno 2 quote valide: "
                    f"home={all_odds['match_winner'].get('home')}, "
                    f"draw={all_odds['match_winner'].get('draw')}, "
                    f"away={all_odds['match_winner'].get('away')}"
                )
                # Se non valide ma abbiamo almeno 2 quote, manteniamole ma logga warning
                # (potrebbero essere quote live che cambiano rapidamente)
                valid_count = sum([
                    1 for v in all_odds['match_winner'].values() if v is not None
                ])
                if valid_count < 2:
                    # Reset se abbiamo meno di 2 quote valide
                    logger.warning("⚠️  Reset quote 1X2: meno di 2 quote valide dopo validazione")
                    all_odds['match_winner'] = {'home': None, 'draw': None, 'away': None}
        
        # Valida Over/Under per ogni threshold
        for market_type in ['over_under', 'over_under_ht', 'first_half_goals', 'second_half_goals']:
            market_dict = all_odds.get(market_type, {})
            for threshold, odds_pair in list(market_dict.items()):
                if isinstance(odds_pair, dict) and ('over' in odds_pair or 'under' in odds_pair):
                    is_valid = self._validate_market_implied_probabilities(odds_pair, f"{market_type}_{threshold}")
                    if not is_valid:
                        logger.warning(
                            f"⚠️  Quote {market_type} {threshold} incoerenti, reset"
                        )
                        # Reset le quote per questo threshold
                        all_odds[market_type][threshold] = {'over': None, 'under': None}
        
        # Valida BTTS
        for market_type in ['btts', 'btts_ht']:
            btts_dict = all_odds.get(market_type, {})
            if btts_dict:
                is_valid = self._validate_market_implied_probabilities(btts_dict, market_type)
                if not is_valid:
                    logger.warning(f"⚠️  Quote {market_type} incoerenti, reset")
                    all_odds[market_type] = {'yes': None, 'no': None}
        
        logger.debug("✅ Validazione probabilità implicite completata")
        
        # 🔧 NUOVO: Aggiungi tracker, conteggi e riepilogo offerte a all_odds per uso futuro
        best_offer_summary = []
        for outcome in ['home', 'draw', 'away']:
            odd = all_odds['match_winner'][outcome]
            bm = bookmaker_tracker['match_winner'][outcome]
            if odd:
                best_offer_summary.append({
                    'market': f'1x2_{outcome}',
                    'odd': odd,
                    'bookmaker': bm,
                    'bookmakers_available': bookmaker_counts['match_winner'][outcome]
                })

        for threshold, odds in list(all_odds['over_under'].items())[:3]:
            over_odd = odds.get('over')
            under_odd = odds.get('under')
            if over_odd:
                best_offer_summary.append({
                    'market': f'over_{threshold}',
                    'odd': over_odd,
                    'bookmaker': bookmaker_tracker['over_under'].get(threshold, {}).get('over'),
                    'bookmakers_available': bookmaker_counts['over_under'].get(threshold, {}).get('over', 0)
                })
            if under_odd:
                best_offer_summary.append({
                    'market': f'under_{threshold}',
                    'odd': under_odd,
                    'bookmaker': bookmaker_tracker['over_under'].get(threshold, {}).get('under'),
                    'bookmakers_available': bookmaker_counts['over_under'].get(threshold, {}).get('under', 0)
                })

        all_odds['_bookmakers'] = bookmaker_tracker
        all_odds['_bet365_odds'] = bet365_odds  # Salva quote bet365 per mostrare nelle notifiche
        all_odds['_bookmaker_counts'] = bookmaker_counts
        all_odds['_bookmaker_counts_flat'] = bookmaker_counts_flat
        all_odds['_best_offer_summary'] = best_offer_summary
        all_odds['_precision_snapshot'] = self._build_precision_snapshot(all_odds, bookmaker_tracker)
        
        # 🔧 LOGGING: Mostra quale bookmaker fornisce le quote principali
        logger.info(f"📊 Bookmaker utilizzati per le quote:")
        if bookmaker_tracker['match_winner']['home']:
            logger.info(f"   1X2 Home: {all_odds['match_winner']['home']} ({bookmaker_tracker['match_winner']['home']})")
        if bookmaker_tracker['match_winner']['draw']:
            logger.info(f"   1X2 Draw: {all_odds['match_winner']['draw']} ({bookmaker_tracker['match_winner']['draw']})")
        if bookmaker_tracker['match_winner']['away']:
            logger.info(f"   1X2 Away: {all_odds['match_winner']['away']} ({bookmaker_tracker['match_winner']['away']})")
        
        # Log Over/Under principali
        for threshold in sorted(all_odds['over_under'].keys(), key=lambda x: float(x)):
            over_odd = all_odds['over_under'][threshold].get('over')
            under_odd = all_odds['over_under'][threshold].get('under')
            over_bm = bookmaker_tracker['over_under'].get(threshold, {}).get('over', 'N/A')
            under_bm = bookmaker_tracker['over_under'].get(threshold, {}).get('under', 'N/A')
            if over_odd:
                logger.info(f"   Over {threshold} FT: {over_odd} ({over_bm})")
            if under_odd:
                logger.info(f"   Under {threshold} FT: {under_odd} ({under_bm})")
        
        # Log Second Half Goals (quello che l'utente sta vedendo)
        for threshold in sorted(all_odds['second_half_goals'].keys(), key=lambda x: float(x)):
            over_odd = all_odds['second_half_goals'][threshold].get('over')
            under_odd = all_odds['second_half_goals'][threshold].get('under')
            over_bm = bookmaker_tracker['second_half_goals'].get(threshold, {}).get('over', 'N/A')
            under_bm = bookmaker_tracker['second_half_goals'].get(threshold, {}).get('under', 'N/A')
            if over_odd:
                logger.info(f"   Over {threshold} 2H: {over_odd} ({over_bm})")
            if under_odd:
                logger.info(f"   Under {threshold} 2H: {under_odd} ({under_bm})")
        
        return all_odds
    
    def _fetch_statistics_from_api_football(self, fixture_id: int, api_key: str, base_url: str) -> Optional[List[Dict]]:
        """
        Recupera le statistiche complete per una partita da API-Football.
        Restituisce None se le statistiche non sono disponibili o non valide.
        Fa UNA SOLA chiamata API invece di due (controllo + fetch).
        
        🔧 FIX: Conta la chiamata API solo se le statistiche sono valide.
        """
        try:
            import urllib.request
            
            url = f"{base_url}/fixtures/statistics?fixture={fixture_id}"
            headers = {
                "x-rapidapi-key": api_key,
                "x-rapidapi-host": "v3.football.api-sports.io"
            }
            
            # 🎯 RETRY LOGIC: Usa retry con backoff esponenziale per resilienza
            def _make_request():
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as response:
                    return json.loads(response.read().decode())
            
            data = self._retry_api_call(_make_request, max_retries=3, base_delay=1.0)
            if data is None:
                return None
            
            # Verifica se ci sono statistiche disponibili e valide
            if data.get("response") and len(data["response"]) > 0:
                self.api_usage_today += 1
                # Controlla se tutte le stats sono zero solo per logging
                all_zero = True
                for team_stats in data["response"]:
                    stats_list = team_stats.get("statistics", [])
                    for stat in stats_list or []:
                        value = stat.get("value")
                        if value not in (None, 0, "0"):
                            all_zero = False
                            break
                    if not all_zero:
                        break
                if all_zero:
                    logger.debug(f"⚠️  Statistiche presenti ma ancora tutte a 0 per fixture {fixture_id}, accetto comunque il feed")
                else:
                    logger.debug(f"✅ Statistiche valide trovate per fixture {fixture_id}")
                return data["response"]
            logger.debug(f"⚠️  Nessuna statistica disponibile per fixture {fixture_id}")
            return None
            
        except urllib.error.HTTPError as e:
            # Errore HTTP (es. 404, 429) - NON contare chiamata API se errore
            logger.debug(f"⚠️  HTTP error recupero statistiche per fixture {fixture_id}: {e.code} - {e.reason}")
            return None
        except Exception as e:
            logger.debug(f"⚠️  Errore recupero statistiche per fixture {fixture_id}: {e}")
            return None
    
    def _parse_statistics_from_api_football(self, statistics: List[Dict]) -> Dict[str, Any]:
        """
        Estrae le statistiche da API-Football e le converte nel formato interno.
        """
        stats_dict = {
            'score_home': 0,
            'score_away': 0,
            'minute': 0,
            'status': 'LIVE',
            'home_total_shots': 0,
            'away_total_shots': 0,
            'home_shots_on_target': 0,
            'away_shots_on_target': 0,
            'home_possession': None,
            'away_possession': None,
            'home_xg': 0.0,
            'away_xg': 0.0,
            'home_dangerous_attacks': 0,
            'away_dangerous_attacks': 0,
            'home_corners': 0,
            'away_corners': 0,
            'home_yellow_cards': 0,
            'away_yellow_cards': 0,
            'home_red_cards': 0,
            'away_red_cards': 0,
            'home_fouls': 0,
            'away_fouls': 0,
        }
        
        if not statistics or len(statistics) < 2:
            return stats_dict
        
        # 🔧 FIX: Prova a estrarre minuto e score dalle statistiche se disponibili
        # Le statistiche potrebbero contenere informazioni sul minuto corrente
        try:
            # Cerca in tutti gli elementi delle statistiche
            for team_stat in statistics:
                team_info = team_stat.get("team", {})
                # Prova a estrarre da vari campi possibili
                if "minute" in team_stat:
                    stats_dict['minute'] = int(team_stat.get("minute", 0))
                if "elapsed" in team_stat:
                    stats_dict['minute'] = int(team_stat.get("elapsed", 0))
                if "time" in team_stat:
                    time_val = team_stat.get("time")
                    if isinstance(time_val, (int, float)):
                        stats_dict['minute'] = int(time_val)
                    elif isinstance(time_val, str) and time_val.isdigit():
                        stats_dict['minute'] = int(time_val)
        except Exception as e:
            logger.debug(f"   Errore estrazione minuto da statistiche: {e}")
        
        home_stats = statistics[0].get("statistics", [])
        away_stats = statistics[1].get("statistics", [])
        
        # Estrai statistiche home
        for stat in home_stats:
            stat_type = stat.get("type", "").lower()
            value = stat.get("value")
            
            # 🔧 FIX: Estrai score dalle statistiche se disponibile
            if "goals" in stat_type and "expected" not in stat_type:
                try:
                    # Se è il campo "Goals" delle statistiche, potrebbe essere lo score corrente
                    goals_value = int(value) if value else 0
                    if goals_value > stats_dict.get('score_home', 0):
                        stats_dict['score_home'] = goals_value
                        logger.debug(f"   Score home estratto da statistiche: {goals_value}")
                except:
                    pass
            
            if "shots on goal" in stat_type or "shots on target" in stat_type:
                try:
                    stats_dict['home_shots_on_target'] = int(value) if value else 0
                except:
                    pass
            elif "total shots" in stat_type or "shots total" in stat_type or stat_type == "shots":
                try:
                    stats_dict['home_total_shots'] = int(value) if value else 0
                except:
                    pass
            elif "ball possession" in stat_type or stat_type == "possession":
                try:
                    if isinstance(value, str):
                        stats_dict['home_possession'] = float(value.replace("%", "").strip())
                    else:
                        stats_dict['home_possession'] = float(value) if value else None
                except:
                    pass
            elif "expected goals" in stat_type or stat_type == "xg":
                try:
                    stats_dict['home_xg'] = float(value) if value else 0.0
                except:
                    pass
            elif "dangerous attacks" in stat_type or stat_type == "attacks":
                try:
                    stats_dict['home_dangerous_attacks'] = int(value) if value else 0
                except:
                    pass
            elif "corner kicks" in stat_type or stat_type == "corners":
                try:
                    stats_dict['home_corners'] = int(value) if value else 0
                except:
                    pass
            elif "yellow cards" in stat_type or stat_type == "yellow":
                try:
                    stats_dict['home_yellow_cards'] = int(value) if value else 0
                except:
                    pass
            elif "red cards" in stat_type or stat_type == "red":
                try:
                    stats_dict['home_red_cards'] = int(value) if value else 0
                except:
                    pass
            elif "fouls" in stat_type:
                try:
                    stats_dict['home_fouls'] = int(value) if value else 0
                except:
                    pass
        
        # Estrai statistiche away
        for stat in away_stats:
            stat_type = stat.get("type", "").lower()
            value = stat.get("value")
            
            # 🔧 FIX: Estrai score dalle statistiche se disponibile
            if "goals" in stat_type and "expected" not in stat_type:
                try:
                    # Se è il campo "Goals" delle statistiche, potrebbe essere lo score corrente
                    goals_value = int(value) if value else 0
                    if goals_value > stats_dict.get('score_away', 0):
                        stats_dict['score_away'] = goals_value
                        logger.debug(f"   Score away estratto da statistiche: {goals_value}")
                except:
                    pass
            
            if "shots on goal" in stat_type or "shots on target" in stat_type:
                try:
                    stats_dict['away_shots_on_target'] = int(value) if value else 0
                except:
                    pass
            elif "total shots" in stat_type or "shots total" in stat_type or stat_type == "shots":
                try:
                    stats_dict['away_total_shots'] = int(value) if value else 0
                except:
                    pass
            elif "ball possession" in stat_type or stat_type == "possession":
                try:
                    if isinstance(value, str):
                        stats_dict['away_possession'] = float(value.replace("%", "").strip())
                    else:
                        stats_dict['away_possession'] = float(value) if value else None
                except:
                    pass
            elif "expected goals" in stat_type or stat_type == "xg":
                try:
                    stats_dict['away_xg'] = float(value) if value else 0.0
                except:
                    pass
            elif "dangerous attacks" in stat_type or stat_type == "attacks":
                try:
                    stats_dict['away_dangerous_attacks'] = int(value) if value else 0
                except:
                    pass
            elif "corner kicks" in stat_type or stat_type == "corners":
                try:
                    stats_dict['away_corners'] = int(value) if value else 0
                except:
                    pass
            elif "yellow cards" in stat_type or stat_type == "yellow":
                try:
                    stats_dict['away_yellow_cards'] = int(value) if value else 0
                except:
                    pass
            elif "red cards" in stat_type or stat_type == "red":
                try:
                    stats_dict['away_red_cards'] = int(value) if value else 0
                except:
                    pass
            elif "fouls" in stat_type:
                try:
                    stats_dict['away_fouls'] = int(value) if value else 0
                except:
                    pass
        
        return stats_dict
    
    def _is_service_active(self) -> bool:
        """
        Verifica se il servizio è attivo (non sospeso su Render).
        
        Controlla se il processo può continuare a funzionare.
        Se Render sospende il servizio, questo controllo dovrebbe rilevarlo.
        
        Strategia:
        1. Controlla flag self.running
        2. Controlla variabile d'ambiente Render (se disponibile)
        3. Verifica se il processo può ancora eseguire operazioni base
        """
        # Controlla se self.running è False (fermato manualmente)
        if not self.running:
            return False
        
        # Controlla variabile d'ambiente Render (se disponibile)
        import os
        render_service_status = os.getenv("RENDER_SERVICE_STATUS", "").lower()
        if render_service_status == "suspended":
            logger.warning("⚠️  Render service is suspended (env var), stopping operations")
            self.running = False
            return False
        
        # Controlla se c'è un file di stato che indica sospensione
        # (utile se Render crea un file quando sospende)
        try:
            status_file = os.getenv("RENDER_STATUS_FILE", "/tmp/render_status")
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = f.read().strip().lower()
                    if status == "suspended":
                        logger.warning("⚠️  Render service is suspended (status file), stopping operations")
                        self.running = False
                        return False
        except:
            pass  # Ignora errori di lettura file
        
        # Verifica se il processo può ancora eseguire operazioni base
        # (se Render sospende, alcune operazioni potrebbero fallire)
        try:
            # Test semplice: verifica se possiamo accedere a una variabile d'ambiente
            # Se il processo è completamente sospeso, anche questo potrebbe fallire
            _ = os.getenv("PATH")
        except:
            logger.warning("⚠️  Processo non può eseguire operazioni base, potrebbe essere sospeso")
            return False
        
        return True
    
    def _get_mock_matches(self) -> List[Dict]:
        """Mock matches per testing"""
        return [
            {
                'id': 'match_1',
                'home': 'Team A',
                'away': 'Team B',
                'league': 'Serie A',
                'date': datetime.now() + timedelta(hours=2),
                'odds_1': 2.10,
                'odds_x': 3.40,
                'odds_2': 3.20
            }
        ]
    
    def _analyze_match(self, match: Dict) -> Optional[Dict]:
        """
        Analizza partita e cerca opportunità VALUE BET.
        
        IMPORTANTE: 
        - Analizza solo VALUE BET reali (EV > soglia)
        - NON consiglia basandosi su score
        - Considera solo probabilità vs quote
        """
        match_id = match.get('id')
        
        if not self.ai_pipeline:
            logger.warning("⚠️  AI Pipeline not available")
            return None
        
        try:
            # Prepara dati match
            match_date = match.get('date')
            is_live = match.get('is_live', False)
            match_status = match.get('match_status', 'PRE-MATCH')
            
            # Normalizza match_date: converte datetime a stringa ISO se necessario
            if match_date is not None:
                if isinstance(match_date, datetime):
                    match_date_str = match_date.isoformat()
                else:
                    match_date_str = str(match_date)
            else:
                match_date_str = None
            
            match_data = {
                'home': match.get('home'),
                'away': match.get('away'),
                'league': match.get('league'),
                'date': match_date_str,  # Usa 'date' invece di 'match_date' per compatibilità
                'match_date': match_date_str,  # Mantieni anche 'match_date' per retrocompatibilità
                'odds_1': match.get('odds_1'),
                'odds_x': match.get('odds_x'),
                'odds_2': match.get('odds_2'),
                'is_live': is_live,
                'match_status': match_status
            }
            
            # Prepara odds data
            odds_data = {
                'current_odds': {
                    'home': match.get('odds_1'),
                    'draw': match.get('odds_x'),
                    'away': match.get('odds_2')
                },
                'odds_history': []  # Nessuna storia disponibile per ora
            }
            
            # Analizza con AI Pipeline
            # Usa bankroll manager se disponibile
            if self.bankroll_manager:
                bankroll = self.bankroll_manager.get_current_bankroll()
            else:
                bankroll = 1000.0  # Bankroll di default
            
            prob_dixon_coles = 0.33  # Default: probabilità uniforme (verrà calcolata dalla pipeline)
            ai_result = self.ai_pipeline.analyze(match_data, prob_dixon_coles, odds_data, bankroll)
            
            # 🧠 NUOVO: Analisi Consensus Multi-Modello
            consensus_result = None
            if self.consensus_analyzer:
                try:
                    consensus_result = self.consensus_analyzer.analyze_consensus(
                        predictions={},  # Le predizioni sono già in ai_result
                        ai_results=ai_result
                    )
                    
                    # Boost confidence se consensus alto
                    should_boost, boost_amount = self.consensus_analyzer.should_boost_confidence(consensus_result)
                    if should_boost and boost_amount != 0:
                        current_conf = ai_result.get('summary', {}).get('confidence', 0)
                        new_conf = max(0, min(100, current_conf + boost_amount))
                        if 'summary' not in ai_result:
                            ai_result['summary'] = {}
                        ai_result['summary']['confidence'] = new_conf
                        ai_result['summary']['confidence_boosted'] = True
                        ai_result['summary']['consensus_boost'] = boost_amount
                    
                    # Aggiungi consensus result
                    ai_result['consensus'] = consensus_result
                except Exception as e:
                    logger.debug(f"⚠️  Consensus analysis error: {e}")
            
            # 🚨 NUOVO: Calcola Alert Level Intelligente
            alert_info = None
            if self.alert_system:
                try:
                    # Raccogli dati per alert system
                    odds_movement = None  # TODO: implementare tracking quote
                    arbitrage = None  # TODO: implementare arbitrage detection
                    anomaly = None  # TODO: implementare anomaly detection
                    
                    alert_info = self.alert_system.calculate_alert_level(
                        ai_result=ai_result,
                        consensus_result=consensus_result or {},
                        odds_movement=odds_movement,
                        arbitrage=arbitrage,
                        anomaly=anomaly
                    )
                    ai_result['alert'] = alert_info
                except Exception as e:
                    logger.debug(f"⚠️  Alert calculation error: {e}")
            
            # Ricalcola stake usando bankroll manager se disponibile
            if self.bankroll_manager and ai_result.get('final_decision', {}).get('action') == 'BET':
                final_decision = ai_result.get('final_decision', {})
                ev = ai_result.get('summary', {}).get('expected_value', 0)
                odds = final_decision.get('odds', 0)
                
                if ev > 0 and odds > 1.0:
                    # Calcola stake ottimale (25% di Kelly)
                    optimal_stake = self.bankroll_manager.calculate_stake(
                        bankroll=bankroll,
                        kelly_fraction=0.25,
                        ev=ev * 100 if ev < 1.0 else ev,  # Converti a percentuale se necessario
                        odds=odds
                    )
                    final_decision['stake'] = optimal_stake
                    ai_result['final_decision'] = final_decision
            
            # 🔧 RIMOSSO: Filtri market e validazioni - calcoliamo confidence ed EV per tutti i mercati disponibili
            # Non filtriamo più per action, min_ev, min_confidence, score-based, real value
            # Tutti i calcoli vengono fatti e la migliore opportunità viene selezionata in _select_best_opportunities
            
            # Costruisci opportunità
            opportunity = {
                'match_id': match_id,
                'match_data': match_data,
                'ai_result': ai_result,
                'timestamp': datetime.now()
            }
            
            return opportunity
            
        except Exception as e:
            logger.error(f"❌ Error analyzing match {match_id}: {e}")
            return None
    
    def _analyze_live_match(self, match: Dict) -> List[Optional[Dict]]:
        """
        Analizza partita live usando LiveBettingAdvisor.
        
        Restituisce lista di opportunità (può essere più di una per partita).
        """
        if not self.live_betting_advisor:
            return []
        
        try:
            match_id = str(match.get('id', ''))
            
            # Prepara match_data (dati base partita)
            # 🔧 FIX: Passa TUTTE le quote disponibili, incluse HT/FT e tutti i threshold
            match_data = {
                'home': match.get('home', ''),
                'away': match.get('away', ''),
                'league': match.get('league', ''),
                'odds_1': match.get('odds_1'),
                'odds_x': match.get('odds_x'),
                'odds_2': match.get('odds_2'),
                # 🆕 NUOVO: Quote Over/Under FT (tutti i threshold disponibili)
                'odds_over_0_5': match.get('over_0.5'),
                'odds_under_0_5': match.get('under_0.5'),
                'odds_over_1_5': match.get('over_1.5'),
                'odds_under_1_5': match.get('under_1.5'),
                'odds_over_2_5': match.get('over_2.5'),
                'odds_under_2_5': match.get('under_2.5'),
                'odds_over_3_5': match.get('over_3.5'),
                'odds_under_3_5': match.get('under_3.5'),
                'odds_over_4_5': match.get('over_4.5'),
                'odds_under_4_5': match.get('under_4.5'),
                # 🆕 NUOVO: Quote Over/Under HT (tutti i threshold disponibili)
                'odds_over_0_5_ht': match.get('over_0.5_ht'),
                'odds_under_0_5_ht': match.get('under_0.5_ht'),
                'odds_over_1_5_ht': match.get('over_1.5_ht'),
                'odds_under_1_5_ht': match.get('under_1.5_ht'),
                'odds_over_2_5_ht': match.get('over_2.5_ht'),
                'odds_under_2_5_ht': match.get('under_2.5_ht'),
                # 🆕 NUOVO: Quote First Half Goals (tutti i threshold)
                'odds_over_0_5_1h': match.get('over_0.5_1h'),
                'odds_under_0_5_1h': match.get('under_0.5_1h'),
                'odds_over_1_5_1h': match.get('over_1.5_1h'),
                'odds_under_1_5_1h': match.get('under_1.5_1h'),
                # 🆕 NUOVO: Quote Second Half Goals (tutti i threshold)
                'odds_over_0_5_2h': match.get('over_0.5_2h'),
                'odds_under_0_5_2h': match.get('under_0.5_2h'),
                'odds_over_1_5_2h': match.get('over_1.5_2h'),
                'odds_under_1_5_2h': match.get('under_1.5_2h'),
                # BTTS
                'odds_btts_yes': match.get('btts_yes'),
                'odds_btts_no': match.get('btts_no'),
                'odds_btts_yes_ht': match.get('btts_yes_ht'),
                'odds_btts_no_ht': match.get('btts_no_ht'),
                # 🆕 NUOVO: Quote Double Chance, DNB
                'odds_1x': match.get('odds_1x'),
                'odds_x2': match.get('odds_x2'),
                'odds_12': match.get('odds_12'),
                'odds_dnb_home': match.get('dnb_home'),
                'odds_dnb_away': match.get('dnb_away'),
                # 🆕 NUOVO: Passa anche all_odds per accesso completo a tutti i mercati
                'all_odds': match.get('all_odds', {}),
            }
            
            # Prepara live_data (dati live partita)
            # 🔧 FIX: Usa score_home e score_away da multi_source_match_finder
            home_score = match.get('score_home', 0)
            away_score = match.get('score_away', 0)
            
            # Fallback: prova anche current_score se score_home/score_away non esistono
            if home_score == 0 and away_score == 0:
                current_score = match.get('current_score', '0-0')
                try:
                    home_score, away_score = map(int, current_score.split('-'))
                except:
                    home_score, away_score = 0, 0
            
            # Estrai minute e status
            minute = match.get('minute', 0)
            match_status = match.get('status', 'LIVE')
            
            # 🔧 DEBUG: Log minuto estratto da match
            logger.info(f"🔍 Minuto estratto da match per {match.get('home', '?')} vs {match.get('away', '?')}:")
            logger.info(f"   match.get('minute'): {match.get('minute')}")
            logger.info(f"   match.get('status'): {match.get('status')}")
            logger.info(f"   match.get('match_status'): {match.get('match_status')}")
            
            if minute is None:
                minute = 0
            elif isinstance(minute, str):
                try:
                    minute = int(minute.replace("'", "").replace("+", "").split()[0])
                    logger.info(f"   Minuto convertito da stringa: {minute}'")
                except:
                    minute = 0
                    logger.warning(f"   ⚠️ Errore conversione minuto da stringa: {match.get('minute')}")
            elif not isinstance(minute, int):
                try:
                    minute = int(minute)
                    logger.info(f"   Minuto convertito da numero: {minute}'")
                except:
                    minute = 0
                    logger.warning(f"   ⚠️ Errore conversione minuto da numero: {match.get('minute')}")
            
            logger.info(f"   Minuto finale dopo conversione: {minute}'")
            
            # 🔧 FILTRO: Escludi partite finite prima di analizzarle
            if minute > 90 or (match_status and match_status.upper() in ["FINISHED", "FT", "AET", "PEN"]):
                logger.debug(f"⏭️  Partita saltata (finita): {match.get('home', '?')} vs {match.get('away', '?')} - minuto {minute}, status {match_status}")
                return []
            
            # 🔧 LOG per debug: verifica score passato a LiveBettingAdvisor
            logger.info(f"📊 Analizzando LIVE {match.get('home', '?')} vs {match.get('away', '?')}: {home_score}-{away_score} (min {minute})")
            
            live_data = {
                'minute': minute,
                # 🔧 FIX: Usa score_home/score_away (non home_score/away_score) per compatibilità con LiveBettingAdvisor
                'score_home': home_score,
                'score_away': away_score,
                'home_score': home_score,  # Mantieni anche per retrocompatibilità
                'away_score': away_score,  # Mantieni anche per retrocompatibilità
                'status': match_status,  # Usa match_status già estratto sopra
                # Statistiche live (se disponibili) - usa nomi corretti per LiveBettingAdvisor
                'shots_on_target_home': match.get('home_shots_on_target', 0),
                'shots_on_target_away': match.get('away_shots_on_target', 0),
                'shots_home': match.get('home_total_shots', 0),
                'shots_away': match.get('away_total_shots', 0),
                'xg_home': match.get('home_xg', 0.0),
                'xg_away': match.get('away_xg', 0.0),
                'dangerous_attacks_home': match.get('home_dangerous_attacks', 0),
                'dangerous_attacks_away': match.get('away_dangerous_attacks', 0),
                # 🔧 NUOVO: Possesso (se disponibile)
                'possession_home': match.get('home_possession'),
                'possession_away': match.get('away_possession'),
                # Mantieni anche nomi alternativi per retrocompatibilità
                'home_shots_on_target': match.get('home_shots_on_target', 0),
                'away_shots_on_target': match.get('away_shots_on_target', 0),
                'home_total_shots': match.get('home_total_shots', 0),
                'away_total_shots': match.get('away_total_shots', 0),
                'home_xg': match.get('home_xg', 0.0),
                'away_xg': match.get('away_xg', 0.0),
                'home_dangerous_attacks': match.get('home_dangerous_attacks', 0),
                'away_dangerous_attacks': match.get('away_dangerous_attacks', 0),
                'home_possession': match.get('home_possession'),
                'away_possession': match.get('away_possession'),
                # 🔧 NUOVO: Statistiche aggiuntive (priorità 1-4)
                'corners_home': match.get('home_corners', 0),
                'corners_away': match.get('away_corners', 0),
                'yellow_cards_home': match.get('home_yellow_cards', 0),
                'yellow_cards_away': match.get('away_yellow_cards', 0),
                'red_cards_home': match.get('home_red_cards', 0),
                'red_cards_away': match.get('away_red_cards', 0),
                'fouls_home': match.get('home_fouls', 0),
                'fouls_away': match.get('away_fouls', 0),
            }
            
            # 🔧 LOG: Verifica live_data prima di passarlo a analyze_live_match
            logger.info(f"📊 live_data passato a analyze_live_match per {match_id}:")
            logger.info(f"   shots_on_target_home: {live_data.get('shots_on_target_home', 0)}, shots_on_target_away: {live_data.get('shots_on_target_away', 0)}")
            logger.info(f"   shots_home: {live_data.get('shots_home', 0)}, shots_away: {live_data.get('shots_away', 0)}")
            logger.info(f"   xg_home: {live_data.get('xg_home', 0.0)}, xg_away: {live_data.get('xg_away', 0.0)}")
            logger.info(f"   match.get('home_shots_on_target'): {match.get('home_shots_on_target', 'NOT_FOUND')}")
            logger.info(f"   match.get('away_shots_on_target'): {match.get('away_shots_on_target', 'NOT_FOUND')}")
            logger.info(f"   score_home: {live_data.get('score_home', 'N/A')}, score_away: {live_data.get('score_away', 'N/A')}")
            logger.info(f"   minute: {live_data.get('minute', 'N/A')}")
            # Verifica se le statistiche sono presenti
            has_stats = any([
                live_data.get('shots_on_target_home', 0) > 0,
                live_data.get('shots_on_target_away', 0) > 0,
                live_data.get('shots_home', 0) > 0,
                live_data.get('shots_away', 0) > 0,
                live_data.get('xg_home', 0.0) > 0,
                live_data.get('xg_away', 0.0) > 0
            ])
            logger.info(f"   Statistiche presenti: {has_stats}")
            
            # Analizza con LiveBettingAdvisor
            opportunities = self.live_betting_advisor.analyze_live_match(
                match_id=match_id,
                match_data=match_data,
                live_data=live_data
            )
            
            # Converti in formato standard
            result = []
            for opp in opportunities:
                if opp:
                    result.append({
                        'match_id': match.get('id'),
                        'match_data': match,
                        'live_opportunity': opp,
                        'timestamp': datetime.now()
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing live match {match.get('id', 'unknown')}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    # 🔧 RIMOSSO: _is_real_value_opportunity, _is_score_based_recommendation, _has_real_value
    # Non filtriamo più le opportunità - calcoliamo confidence ed EV per tutti i mercati
    # La selezione della migliore opportunità avviene in _select_best_opportunities basandosi solo su EV, Confidence e Quality Score
    
    def _handle_opportunity(self, opportunity: Dict):
        """Gestisce opportunità trovata"""
        match_id = opportunity['match_id']
        
        # Evita duplicati
        if match_id in self.notified_opportunities:
            logger.debug(f"   Opportunity {match_id} already notified, skipping")
            return
        
        # Salva opportunità nel tracker (se disponibile)
        if self.results_tracker:
            try:
                self.results_tracker.save_opportunity(opportunity)
                logger.debug(f"   Opportunity saved to tracker: {match_id}")
            except Exception as e:
                logger.warning(f"⚠️  Error saving opportunity to tracker: {e}")
        
        # Notifica Telegram
        if self.notifier:
            try:
                # Determina tipo opportunità (PRE-MATCH o LIVE)
                match_data = opportunity.get('match_data', {})
                match_status = match_data.get('match_status', 'PRE-MATCH')
                ai_result = opportunity.get('ai_result', {})
                
                # 🚨 NUOVO: Usa alert system per priorità intelligente
                alert_info = ai_result.get('alert', {})
                alert_level = alert_info.get('alert_level', 'INFO')
                should_notify = alert_info.get('should_notify', True)
                
                # Determina se notificare basandosi su alert level
                if alert_level in ['CRITICAL', 'HIGH']:
                    opportunity_type = f"🚨 {alert_level} - AUTO-24H {match_status}"
                    # Usa messaggio alert personalizzato se disponibile
                    if alert_info.get('message'):
                        # Invia alert speciale
                        self.notifier.send_message(alert_info['message'])
                elif alert_level == 'MEDIUM':
                    opportunity_type = f"⚡ {match_status} - AUTO-24H"
                else:
                    opportunity_type = f"AUTO-24H {match_status}"
                
                # Notifica solo se alert level è sufficiente
                if should_notify or alert_level in ['CRITICAL', 'HIGH', 'MEDIUM']:
                    success = self.notifier.send_betting_opportunity(
                        opportunity['match_data'],
                        opportunity['ai_result'],
                        opportunity_type=opportunity_type
                    )
                else:
                    logger.debug(f"   Alert level {alert_level} too low, skipping notification")
                    success = False
                
                if success:
                    self.notified_opportunities.add(match_id)
                    logger.info(f"✅ Notified opportunity: {match_id}")
                else:
                    logger.warning(f"⚠️  Failed to notify opportunity: {match_id}")
            except Exception as e:
                logger.error(f"❌ Error notifying opportunity: {e}")
        else:
            logger.warning("⚠️  Telegram notifier not available")
    
    def _select_best_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Seleziona la migliore opportunità in assoluto da notificare.
        
        Ranking basato su:
        - EV (30%): Profittabilità
        - Confidence (30%): Affidabilità
        - Quality Score (30%): Qualità segnale (dal Signal Quality Gate)
        - Stats Bonus (10%): Qualità dati statistici disponibili
        
        Restituisce max 1 opportunità (la migliore in assoluto).
        """
        if not opportunities:
            return []
        
        def _extract_threshold_from_market_name(market_name: Optional[str]) -> Optional[str]:
            if not market_name:
                return None
            match = re.search(r'(\d+(\.\d+)?)', market_name)
            return match.group(1) if match else None

        def _map_market_to_liquidity_key(market_name: Optional[str]) -> Optional[str]:
            if not market_name:
                return None
            lower = market_name.lower()
            threshold = _extract_threshold_from_market_name(lower)

            def _is_first_half(name: str) -> bool:
                return any(tag in name for tag in ['_ht', '_1h', 'first_half', '1st_half'])

            def _is_second_half(name: str) -> bool:
                return any(tag in name for tag in ['_2h', 'second_half', '2nd_half'])

            if lower.startswith('over_') or lower.startswith('under_'):
                base = 'over_under'
                if _is_first_half(lower):
                    base = 'over_under_ht'
                elif _is_second_half(lower):
                    base = 'second_half_goals'
                outcome = 'over' if lower.startswith('over_') else 'under'
                if threshold:
                    return f"{base}:{threshold}:{outcome}"

            if lower.startswith('btts'):
                outcome = 'yes'
                if 'no' in lower:
                    outcome = 'no'
                base = 'btts'
                if _is_first_half(lower):
                    base = 'btts_ht'
                return f"{base}:{outcome}"

            if 'next_goal' in lower or lower.startswith('next_goal'):
                if 'away' in lower:
                    return "match_winner:away"
                if 'home' in lower:
                    return "match_winner:home"
                return None

            if lower.startswith('home_win') or '1x2_home' in lower:
                return "match_winner:home"
            if lower.startswith('away_win') or '1x2_away' in lower:
                return "match_winner:away"

            if 'draw_no_bet' in lower or lower.startswith('dnb_'):
                if 'away' in lower:
                    return "draw_no_bet:away"
                return "draw_no_bet:home"

            if 'double_chance' in lower or lower.startswith('1x') or lower.startswith('x2') or lower.startswith('12'):
                if '1x' in lower:
                    return "double_chance:1x"
                if 'x2' in lower:
                    return "double_chance:x2"
                if '12' in lower:
                    return "double_chance:12"

            return None

        def _compute_liquidity_factor_for_opportunity(market_name: Optional[str], opp_dict: Dict) -> float:
            if not market_name:
                return 1.0
            match_data = opp_dict.get('match_data') or {}
            all_odds_data = match_data.get('all_odds') or {}
            counts_flat = all_odds_data.get('_bookmaker_counts_flat') or {}
            liquidity_key = _map_market_to_liquidity_key(market_name)
            if not liquidity_key or not counts_flat:
                return 1.0
            count = counts_flat.get(liquidity_key)
            if not count:
                return 1.0
            if count >= 5:
                return 1.08
            if count == 4:
                return 1.05
            if count == 3:
                return 1.03
            if count == 2:
                return 1.01
            return 0.97  # Solo un bookmaker → meno affidabile

        def _compute_pressure_factor(stats_dict: Optional[Dict[str, Any]]) -> float:
            if not isinstance(stats_dict, dict):
                return 1.0
            shots_home = stats_dict.get('shots_home') or stats_dict.get('home_total_shots') or 0
            shots_away = stats_dict.get('shots_away') or stats_dict.get('away_total_shots') or 0
            sot_home = stats_dict.get('shots_on_target_home') or stats_dict.get('home_shots_on_target') or 0
            sot_away = stats_dict.get('shots_on_target_away') or stats_dict.get('away_shots_on_target') or 0
            xg_home = stats_dict.get('xg_home') or 0.0
            xg_away = stats_dict.get('xg_away') or 0.0
            attacks_home = stats_dict.get('dangerous_attacks_home') or 0
            attacks_away = stats_dict.get('dangerous_attacks_away') or 0

            total_shots = shots_home + shots_away
            total_sot = sot_home + sot_away
            total_xg = (xg_home or 0.0) + (xg_away or 0.0)
            total_attacks = attacks_home + attacks_away

            metrics_available = sum([
                1 if total_shots > 0 else 0,
                1 if total_sot > 0 else 0,
                1 if total_xg > 0 else 0,
                1 if total_attacks > 0 else 0
            ])
            if metrics_available <= 1:
                return 1.0

            factor = 1.0
            if total_sot >= 6 or total_xg >= 2.5:
                factor += 0.08
            elif total_sot <= 1 and total_shots <= 8:
                factor -= 0.05

            if abs(sot_home - sot_away) >= 4:
                factor += 0.03

            if total_attacks >= 80 or total_shots >= 22:
                factor += 0.04

            return max(0.9, min(1.12, factor))

        def _compute_realism_factor(confidence_value: float, odds_value: float) -> float:
            if not odds_value or odds_value <= 1.0:
                return 1.0
            try:
                implied_prob = (1.0 / odds_value) * 100.0
            except ZeroDivisionError:
                return 1.0
            prob_gap = abs(confidence_value - implied_prob)
            if prob_gap <= 5:
                return 1.05
            if prob_gap <= 15:
                return 1.02
            if prob_gap >= 35:
                return 0.90
            if prob_gap >= 25:
                return 0.95
            return 1.0

        def _adjust_weights_for_market(market_type: str, market_name: str, minute_value: int, weights: Dict[str, float]) -> Dict[str, float]:
            adjusted = weights.copy()
            lower = (market_name or '').lower()
            if market_type in {'over', 'under', 'btts'}:
                adjusted['stats'] += 0.05
            if market_type in {'next', 'goal'} or 'next_goal' in lower:
                adjusted['confidence'] += 0.05
            if 'card' in lower or 'corner' in lower:
                adjusted['quality'] += 0.05
            if 'second_half' in lower or minute_value >= 60:
                adjusted['quality'] += 0.02
            if minute_value >= 75 and market_type in {'over', 'goal', 'next'}:
                adjusted['ev'] += 0.05
            total = sum(adjusted.values()) or 1.0
            for key in adjusted:
                adjusted[key] = adjusted[key] / total
            return adjusted
        
        # 🔧 RIMOSSO: Filtro has_live_stats - accettiamo tutte le opportunità
        # Le partite vengono già filtrate per avere statistiche e quote in _fetch_matches_with_odds_from_api_football
        # Qui calcoliamo confidence ed EV per tutte le opportunità e selezioniamo la migliore
        valid_opportunities = []
        for opp_dict in opportunities:
            live_opp = opp_dict.get('live_opportunity')
            if not live_opp:
                continue
            valid_opportunities.append(opp_dict)
        
        # Se non ci sono opportunità valide, ritorna lista vuota
        if not valid_opportunities:
            logger.info(f"⚠️  Nessuna opportunità valida tra {len(opportunities)} totali")
            return []
        
        # 🆕 Inizializza Signal Quality Gate se non esiste (verrà inizializzato in _handle_live_opportunity se necessario)
        # Non inizializziamo qui per evitare duplicazioni
        
        # 🔧 OPZIONE 4: Identifica mercati alternativi
        alternative_market_types = {'over', 'under', 'btts', 'clean', 'exact', 'goal', 'odd', 'ht'}
        
        # 🆕 Calcola score completo per ogni opportunità (incluso Quality Score)
        scored_opportunities = []
        now = datetime.now()
        
        for opp_dict in valid_opportunities:
            live_opp = opp_dict.get('live_opportunity')
            if not live_opp:
                continue
            
            match_id = opp_dict.get('match_id', 'unknown')
            market = getattr(live_opp, 'market', opp_dict.get('market', 'unknown'))
            market_type = market.split('_')[0] if '_' in market else market
            
            # 🔧 FIX: Estrai stats da live_opp.match_stats o opp_dict
            stats = {}
            if hasattr(live_opp, 'match_stats') and live_opp.match_stats:
                stats = live_opp.match_stats if isinstance(live_opp.match_stats, dict) else {}
            elif 'match_stats' in opp_dict:
                stats = opp_dict['match_stats'] if isinstance(opp_dict['match_stats'], dict) else {}
            
            # 1. 🎯 PRECISIONE MANIACALE: Ricalcola EV con precisione massima ad ogni ciclo
            confidence = getattr(live_opp, 'confidence', 0.0)
            odds = getattr(live_opp, 'odds', 1.0)
            minute = stats.get('minute', 0) if isinstance(stats, dict) else opp_dict.get('minute', 0)
            
            # 🎯 PRECISIONE MANIACALE: Ricalcola EV con precisione massima ad ogni ciclo
            # Questo assicura che EV sia sempre aggiornato con quote/confidence correnti
            
            # 🎯 Validazione rigorosa quote e confidence prima del calcolo EV
            validated_odds = None
            validated_odds_decimal = self._validate_odds(odds)
            if validated_odds_decimal is None:
                logger.warning(
                    f"⚠️  Quote invalide per {match_id}/{market}: odds={odds}, "
                    f"conf={confidence:.2f}%, salto calcolo EV"
                )
                ev = 0.0
            elif not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 100:
                logger.warning(
                    f"⚠️  Confidence invalida per {match_id}/{market}: conf={confidence}, "
                    f"odds={odds:.4f}, salto calcolo EV"
                )
                ev = 0.0
            elif self.live_betting_advisor and hasattr(self.live_betting_advisor, '_calculate_ev_from_values'):
                try:
                    # Ricalcola EV con funzione precisa (usa Decimal per precisione assoluta)
                    ev_old = getattr(live_opp, 'ev', 0.0)
                    validated_odds = float(validated_odds_decimal)
                    ev = self.live_betting_advisor._calculate_ev_from_values(confidence, validated_odds)
                    # Aggiorna anche l'oggetto live_opp per coerenza
                    live_opp.ev = ev
                    live_opp.odds = validated_odds  # Aggiorna anche odds validati
                    
                    # 🎯 Log dettagliato per tracciabilità precisione
                    ev_diff = ev - ev_old
                    if abs(ev_diff) > 0.01:  # Se differenza > 0.01%
                        logger.info(
                            f"🔢 EV RICALCOLATO per {match_id}/{market}: "
                            f"conf={confidence:.4f}%, odds={validated_odds:.6f} → "
                            f"EV={ev:.4f}% (precedente: {ev_old:.4f}%, diff: {ev_diff:+.4f}%)"
                        )
                    else:
                        logger.debug(
                            f"🔢 EV ricalcolato per {match_id}/{market}: "
                            f"conf={confidence:.2f}%, odds={validated_odds:.4f} → EV={ev:.4f}% (invariato)"
                        )
                except Exception as e:
                    logger.error(
                        f"❌ Errore ricalcolo EV per {match_id}/{market}: {e}, "
                        f"conf={confidence:.2f}%, odds={(validated_odds or odds):.4f}, uso valore cached"
                    )
                    ev = getattr(live_opp, 'ev', 0.0)
            else:
                # Fallback: usa valore cached se live_betting_advisor non disponibile
                ev = getattr(live_opp, 'ev', 0.0)
                logger.warning(
                    f"⚠️  LiveBettingAdvisor non disponibile per {match_id}/{market}, "
                    f"uso EV cached: {ev:.4f}% (conf={confidence:.2f}%, odds={(validated_odds or odds):.4f})"
                )
            
            # 🆕 Normalizzazione EV intelligente (funzione sigmoide/logaritmica)
            # EV positivo alto → più peso, EV negativo → penalità forte
            if ev > 0:
                # EV positivo: usa funzione logaritmica per dare più peso a valori alti
                # EV 5% → ~1.05, EV 15% → ~1.25, EV 25% → ~1.50
                ev_normalized = 1.0 + (ev / 100.0) * (1.0 + ev / 50.0)  # Crescita più che lineare
            elif ev < 0:
                # EV negativo: penalità forte (esponenziale)
                # EV -5% → ~0.70, EV -10% → ~0.50
                ev_normalized = 1.0 + (ev / 100.0) * 1.5  # Penalità più forte
            else:
                ev_normalized = 1.0  # EV = 0 → neutro
            
            # Limita valori estremi
            ev_normalized = max(0.3, min(2.0, ev_normalized))
            
            confidence_normalized = confidence / 100.0

            pressure_factor = _compute_pressure_factor(stats or opp_dict.get('match_stats'))
            liquidity_factor = _compute_liquidity_factor_for_opportunity(market, opp_dict)
            odds_for_gap = validated_odds if validated_odds is not None else odds
            realism_factor = _compute_realism_factor(confidence, odds_for_gap if odds_for_gap else 0.0)
            
            # 2. 🆕 Calcola Quality Score (se disponibile)
            quality_score_normalized = 0.0
            quality_score_obj = None
            opp_key = None
            
            # 🆕 Inizializza Signal Quality Gate se necessario (PRIMA del controllo)
            if not hasattr(self, 'signal_quality_gate') or not self.signal_quality_gate:
                try:
                    from ai_system.signal_quality_scorer import SignalQualityGate
                    # Verifica che il learner sia disponibile
                    if not hasattr(self, 'signal_quality_learner') or not self.signal_quality_learner:
                        logger.warning("⚠️  SignalQualityLearner non disponibile per SignalQualityGate!")
                    self.signal_quality_gate = SignalQualityGate(
                        ai_pipeline=self.ai_pipeline,
                        min_quality_score=0.0,  # 🎯 Nessuna soglia minima
                        learner=getattr(self, 'signal_quality_learner', None)
                    )
                    if self.signal_quality_gate.learner:
                        logger.debug("✅ SignalQualityGate inizializzato con learner in _select_best_opportunities")
                    else:
                        logger.warning("⚠️  SignalQualityGate inizializzato SENZA learner in _select_best_opportunities!")
                except Exception as e:
                    logger.error(f"❌ Errore inizializzazione SignalQualityGate: {e}")
                    self.signal_quality_gate = None
            
            if self.signal_quality_gate:
                try:
                    # Prepara dati per validazione
                    match_data = {
                        'home': opp_dict.get('home', ''),
                        'away': opp_dict.get('away', ''),
                        'league': opp_dict.get('league', '')
                    }
                    
                    # Estrai live_data
                    live_data = {}
                    if hasattr(live_opp, 'match_stats') and live_opp.match_stats:
                        live_data = live_opp.match_stats.copy()
                    else:
                        live_data = {
                            'minute': opp_dict.get('minute', 0),
                            'score_home': opp_dict.get('score_home', 0),
                            'score_away': opp_dict.get('score_away', 0),
                            'shots_home': opp_dict.get('shots_home', 0),
                            'shots_away': opp_dict.get('shots_away', 0),
                            'shots_on_target_home': opp_dict.get('shots_on_target_home', 0),
                            'shots_on_target_away': opp_dict.get('shots_on_target_away', 0),
                            'possession_home': opp_dict.get('possession_home'),
                            'xg_home': opp_dict.get('xg_home', 0.0),
                            'xg_away': opp_dict.get('xg_away', 0.0)
                        }
                    
                    _, quality_score_obj = self.signal_quality_gate.should_send_signal(
                        opportunity=opp_dict,
                        match_data=match_data,
                        live_data=live_data
                    )
                    
                    # Normalizza Quality Score (0-100 -> 0-1)
                    if quality_score_obj:
                        quality_score_normalized = quality_score_obj.total_score / 100.0
                        # 🆕 Salva in cache per evitare doppio calcolo
                        minute = live_data.get('minute', 0)
                        minute_rounded = (minute // 5) * 5
                        opp_key = f"{match_id}_{market}_{minute_rounded}"
                        self.quality_score_cache[opp_key] = quality_score_obj
                    else:
                        logger.warning(f"⚠️  quality_score_obj è None per {match_id}/{market}, uso default 0.5")
                        quality_score_normalized = 0.5
                    
                except Exception as e:
                    logger.debug(f"⚠️  Errore calcolo Quality Score per {match_id}/{market}: {e}")
                    quality_score_normalized = 0.5  # Default neutro se errore
            
            # 3. Calcola Stats Bonus (qualità dati statistici)
            stats_bonus = 0.8  # Default: penalità se statistiche insufficienti
            stats = getattr(live_opp, 'match_stats', {}) or {}
            if isinstance(stats, dict):
                stats_count = 0
                if stats.get('shots_home', 0) > 0 or stats.get('shots_away', 0) > 0:
                    stats_count += 1
                if stats.get('shots_on_target_home', 0) > 0 or stats.get('shots_on_target_away', 0) > 0:
                    stats_count += 1
                if stats.get('possession_home') is not None:
                    stats_count += 1
                if stats.get('xg_home', 0) > 0 or stats.get('xg_away', 0) > 0:
                    stats_count += 1
                
                # Bonus basato su numero di statistiche disponibili
                if stats_count >= 4:
                    stats_bonus = 1.2  # +20% se statistiche complete
                elif stats_count >= 3:
                    stats_bonus = 1.0  # Neutro se 3 statistiche
                elif stats_count >= 2:
                    stats_bonus = 0.9  # -10% se solo 2 statistiche
                else:
                    stats_bonus = 0.8  # -20% se statistiche insufficienti
            
            # 4. 🔧 OPZIONE 4: Applica penalizzazioni e bonus per mercati
            score_modifier = 1.0
            modifier_reason = ""
            
            if match_id in self.match_markets_history:
                market_already_used = False
                for market_entry in self.match_markets_history[match_id]:
                    if market_entry['market'] == market:
                        market_already_used = True
                        break
                
                if market_already_used:
                    score_modifier *= 0.7
                    modifier_reason = " (penalizzato -30%: mercato già suggerito)"
                elif market_type in alternative_market_types:
                    score_modifier *= 1.2
                    modifier_reason = " (bonus +20%: mercato alternativo)"
            
            # 5. 🆕 MIGLIORATO: Calcola Final Score con pesi dinamici e bonus sinergia
            
            # 🆕 5.1 Pesi dinamici basati su contesto
            weight_ev = 0.30
            weight_confidence = 0.30
            weight_quality = 0.30
            weight_stats = 0.10
            
            # Se EV è molto alto (>15%), dargli più peso
            if ev > 15.0:
                weight_ev = 0.40
                weight_confidence = 0.25
                weight_quality = 0.25
                weight_stats = 0.10
            # Se Confidence è molto alta (>85%), dargli più peso
            elif confidence > 85.0:
                weight_confidence = 0.40
                weight_ev = 0.25
                weight_quality = 0.25
                weight_stats = 0.10
            # Se Quality Score è molto alto (>90), dargli più peso
            elif quality_score_obj and quality_score_obj.total_score > 90.0:
                weight_quality = 0.40
                weight_ev = 0.25
                weight_confidence = 0.25
                weight_stats = 0.10
            
            # 🆕 5.2 Calcola score base con pesi dinamici e profili mercato
            weights = _adjust_weights_for_market(
                market_type,
                market,
                minute,
                {
                    'ev': weight_ev,
                    'confidence': weight_confidence,
                    'quality': weight_quality,
                    'stats': weight_stats
                }
            )
            weight_ev = weights['ev']
            weight_confidence = weights['confidence']
            weight_quality = weights['quality']
            weight_stats = weights['stats']
            
            base_score = (
                ev_normalized * weight_ev +
                confidence_normalized * weight_confidence +
                quality_score_normalized * weight_quality +
                stats_bonus * weight_stats
            )
            
            # 🆕 5.3 Bonus sinergia (tutti i fattori alti insieme)
            synergy_bonus = 1.0
            synergy_factors = 0
            
            if ev > 10.0:
                synergy_factors += 1
            if confidence > 80.0:
                synergy_factors += 1
            if quality_score_obj and quality_score_obj.total_score > 85.0:
                synergy_factors += 1
            
            # Bonus progressivo: 2 fattori alti = +10%, tutti e 3 = +20%
            if synergy_factors == 2:
                synergy_bonus = 1.10
            elif synergy_factors == 3:
                synergy_bonus = 1.20
            
            # 🆕 5.4 Fattore tempo (minuto del match)
            time_factor = 1.0
            if minute > 0:
                if minute <= 20:
                    # Minuti iniziali: statistiche meno affidabili
                    time_factor = 0.90  # -10%
                elif minute >= 60:
                    # Minuti avanzati: statistiche più affidabili
                    time_factor = 1.05  # +5%
                # Minuti 20-60: neutro (time_factor = 1.0)
            
            # 🆕 5.5 Fattore qualità quote
            odds_factor = 1.0
            if odds > 0:
                if odds >= 2.0:
                    # Quote molto favorevoli: bonus
                    odds_factor = 1.10  # +10%
                elif odds < 1.3:
                    # Quote troppo basse: penalità
                    odds_factor = 0.95  # -5%
                # Quote 1.3-2.0: neutro (odds_factor = 1.0)
            
            # 🎯 RIMOSSO: Penalità per EV negativo o troppo basso
            # L'utente vuole la miglior partita senza soglie minime
            ev_penalty = 1.0  # Nessuna penalità
            
            # 🆕 5.7 Calcola Final Score composito con tutti i fattori
            final_score = (
                base_score * 
                synergy_bonus * 
                time_factor * 
                odds_factor * 
                ev_penalty * 
                score_modifier *
                pressure_factor *
                liquidity_factor *
                realism_factor
            )
            
            minute = stats.get('minute', 0) if isinstance(stats, dict) else 0
            
            scored_opportunities.append({
                'opportunity': opp_dict,
                'score': final_score,
                'score_original': final_score,  # 🆕 Salva score originale prima di diversificazione
                'ev': ev,
                'confidence': confidence,
                'quality_score': quality_score_obj.total_score if quality_score_obj else 0.0,
                'stats_bonus': stats_bonus,
                'minute': minute,
                'odds': odds,
                'base_score': base_score,  # 🆕 Score base (prima di bonus/penalità)
                'synergy_bonus': synergy_bonus,  # 🆕 Bonus sinergia
                'time_factor': time_factor,  # 🆕 Fattore tempo
                'odds_factor': odds_factor,  # 🆕 Fattore quote
                'ev_penalty': ev_penalty,  # 🆕 Penalità EV
                'score_modifier': score_modifier,
                'pressure_factor': pressure_factor,
                'liquidity_factor': liquidity_factor,
                'realism_factor': realism_factor,
                'modifier_reason': modifier_reason,
                'opp_key': opp_key  # Per cache
            })
        
        # Ordina per score decrescente
        scored_opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # 🆕 FIX: Diversificazione intelligente per TIPO di mercato
        # 1. Conta quante volte ogni TIPO di mercato è stato inviato di recente
        market_type_counts = {}
        if hasattr(self, 'last_global_notification_time') and self.last_global_notification_time:
            # Conta tipi di mercato inviati nelle ultime 60 minuti (circa 6 notifiche)
            cutoff_time = datetime.now() - timedelta(minutes=60)
            for match_id_history, markets_list in self.match_markets_history.items():
                for market_entry in markets_list:
                    if market_entry['timestamp'] > cutoff_time:
                        market = market_entry['market']
                        # Estrai tipo di mercato (over, under, btts, next_goal, cards, ecc.)
                        market_type = market.split('_')[0] if '_' in market else market
                        market_type_counts[market_type] = market_type_counts.get(market_type, 0) + 1
        
        # 2. Applica penalizzazione per TIPO di mercato già inviato di recente
        for opp in scored_opportunities:
            live_opp = opp['opportunity'].get('live_opportunity')
            if live_opp:
                market = getattr(live_opp, 'market', None)
                if market:
                    # Estrai tipo di mercato
                    market_type = market.split('_')[0] if '_' in market else market
                    
                    # Penalizza tipo di mercato già inviato di recente
                    if market_type in market_type_counts:
                        type_count = market_type_counts[market_type]
                        # Penalizzazione progressiva: -15% per ogni volta che il TIPO è stato inviato
                        penalty = 0.15 * type_count
                        opp['score'] *= (1.0 - min(penalty, 0.70))  # Max -70% di penalizzazione
                        current_reason = opp.get('modifier_reason', '')
                        opp['modifier_reason'] = f"{current_reason} (penalizzato -{penalty*100:.0f}%: tipo '{market_type}' già inviato {type_count} volte)"
                    else:
                        # Bonus per tipo di mercato non inviato di recente
                        bonus = 1.15  # +15% bonus
                        opp['score'] *= bonus
                        current_reason = opp.get('modifier_reason', '')
                        opp['modifier_reason'] = f"{current_reason} (bonus +15%: tipo '{market_type}' non inviato di recente)"
        
        # 3. Riordina dopo penalizzazioni/bonus
        scored_opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # 4. 🆕 NUOVO: Seleziona la migliore per ogni TIPO di mercato (diversificazione intelligente)
        best_by_type = {}  # market_type -> best opportunity
        for opp in scored_opportunities:
            live_opp = opp['opportunity'].get('live_opportunity')
            if live_opp:
                market = getattr(live_opp, 'market', None)
                if market:
                    market_type = market.split('_')[0] if '_' in market else market
                    # Se non abbiamo ancora una opportunità per questo tipo, o questa è migliore
                    if market_type not in best_by_type or opp['score'] > best_by_type[market_type]['score']:
                        best_by_type[market_type] = opp
        
        # 5. Seleziona la migliore tra le migliori di ogni tipo (diversificazione intelligente)
        # IMPORTANTE: La diversificazione è solo un modificatore, la QUALITÀ rimane prioritaria
        best = []
        if best_by_type:
            # Ordina per score ORIGINALE (prima di diversificazione) per mantenere qualità come priorità
            # Poi usa lo score finale (dopo diversificazione) come tie-breaker
            sorted_by_type = sorted(
                best_by_type.items(), 
                key=lambda x: (
                    x[1].get('score_original', x[1].get('score', 0)),  # Priorità: score originale (qualità)
                    x[1].get('score', 0)  # Tie-breaker: score finale (diversificazione)
                ),
                reverse=True
            )
            
            # Prendi la migliore in assoluto (basata su qualità originale)
            # Ma verifica che lo score finale (dopo diversificazione) sia ancora ragionevole
            for market_type, opp in sorted_by_type:
                score_original = opp.get('score_original', opp.get('score', 0))
                score_final = opp.get('score', 0)
                
                # 🎯 RIMOSSO: Filtro soglia minima score - l'utente vuole vedere tutte le opportunità
                # Aggiungi sempre la migliore opportunità senza controllare soglie
                best.append(opp)
                live_opp = opp['opportunity'].get('live_opportunity')
                market = getattr(live_opp, 'market', 'unknown') if live_opp else 'unknown'
                logger.info(f"   ✅ Selezionata opportunità tipo '{market_type}' ({market})")
                logger.info(f"      📊 Score originale: {score_original:.3f} | Score finale: {score_final:.3f}")
                logger.info(f"      📈 EV: {opp['ev']:.1f}% | Conf: {opp['confidence']:.1f}% | Quality: {opp.get('quality_score', 0):.1f} | Odds: {opp.get('odds', 0):.2f}")
                    # 🆕 Log dettagliato dei fattori di calcolo
                logger.debug(
                    "      🔍 Dettagli calcolo: Base=%.3f | Sinergia=%.2fx | Tempo=%.2fx | Quote=%.2fx | EV_penalty=%.2fx | Pressione=%.2fx | Liquidità=%.2fx | Realismo=%.2fx",
                    opp.get('base_score', 0),
                    opp.get('synergy_bonus', 1.0),
                    opp.get('time_factor', 1.0),
                    opp.get('odds_factor', 1.0),
                    opp.get('ev_penalty', 1.0),
                    opp.get('pressure_factor', 1.0),
                    opp.get('liquidity_factor', 1.0),
                    opp.get('realism_factor', 1.0)
                )
                break
            
            # Log delle migliori per tipo (per debug)
            logger.debug(f"   📊 Top 5 opportunità per tipo di mercato:")
            for market_type, opp in sorted(best_by_type.items(), key=lambda x: x[1].get('score_original', x[1].get('score', 0)), reverse=True)[:5]:
                live_opp = opp['opportunity'].get('live_opportunity')
                market = getattr(live_opp, 'market', 'unknown') if live_opp else 'unknown'
                score_orig = opp.get('score_original', opp.get('score', 0))
                score_fin = opp.get('score', 0)
                logger.debug(f"      {market_type}: {market} | Score orig: {score_orig:.3f} | Score fin: {score_fin:.3f} | EV: {opp['ev']:.1f}%")
        elif scored_opportunities:
            # Fallback: se non riusciamo a categorizzare, prendi la migliore in assoluto
            best.append(scored_opportunities[0])
        
        # Log dettagliato
        logger.info(f"   🏆 Migliori opportunità selezionate:")
        for i, item in enumerate(best, 1):
            live_opp = item['opportunity'].get('live_opportunity')
            match_id = item['opportunity'].get('match_id', 'unknown')
            market = getattr(live_opp, 'market', 'unknown')
            modifier_info = item.get('modifier_reason', '')
            quality_info = f", quality={item['quality_score']:.1f}/100" if item['quality_score'] > 0 else ""
            logger.info(
                f"      {i}. {match_id} - {market}: "
                f"final_score={item['score']:.3f}, "
                f"ev={item['ev']:.1f}%, conf={item['confidence']:.1f}%{quality_info}{modifier_info}"
            )
        
        return [item['opportunity'] for item in best]
    
    def _handle_live_opportunity(self, opportunity: Dict):
        """Gestisce opportunità live trovata da LiveBettingAdvisor"""
        match_id = opportunity['match_id']
        live_opp = opportunity.get('live_opportunity')
        
        if not live_opp:
            return

        # 🔧 RIMOSSO: Controllo has_live_stats - le partite vengono già filtrate per avere statistiche e quote
        # in _fetch_matches_with_odds_from_api_football, quindi tutte le opportunità qui hanno statistiche valide
        
        # 🆕 AI SIGNAL QUALITY GATE: Validazione finale qualità segnale
        # 🆕 Ottimizzazione: Usa cache se disponibile per evitare doppio calcolo
        market = getattr(live_opp, 'market', opportunity.get('market', 'unknown'))
        minute = 0
        if hasattr(live_opp, 'match_stats') and live_opp.match_stats:
            if isinstance(live_opp.match_stats, dict):
                minute = live_opp.match_stats.get('minute', 0)
        minute_rounded = (minute // 5) * 5
        opp_key = f"{match_id}_{market}_{minute_rounded}"
        
        # Verifica se Quality Score è già in cache (calcolato in _select_best_opportunities)
        quality_score = None
        if opp_key in self.quality_score_cache:
            quality_score = self.quality_score_cache[opp_key]
            if quality_score:
                logger.debug(f"✅ Quality Score da cache per {opp_key}: {quality_score.total_score:.1f}/100")
            else:
                logger.debug(f"⚠️  Quality Score in cache è None per {opp_key}")
        else:
            # Calcola Quality Score se non in cache
            try:
                from ai_system.signal_quality_scorer import SignalQualityGate
                if not hasattr(self, 'signal_quality_gate'):
                    # 🆕 Inizializza learner se non esiste
                    if not hasattr(self, 'signal_quality_learner'):
                        try:
                            from ai_system.signal_quality_learner import SignalQualityLearner
                            self.signal_quality_learner = SignalQualityLearner()
                            logger.info("✅ Signal Quality Learner inizializzato")
                        except Exception as e:
                            logger.debug(f"⚠️  Signal Quality Learner non disponibile: {e}")
                            self.signal_quality_learner = None
                    
                    # Verifica che il learner sia disponibile
                    learner = getattr(self, 'signal_quality_learner', None)
                    if not learner:
                        logger.warning("⚠️  SignalQualityLearner non disponibile per SignalQualityGate in _handle_live_opportunity!")
                    self.signal_quality_gate = SignalQualityGate(
                        ai_pipeline=self.ai_pipeline,
                        min_quality_score=75.0,
                        learner=learner
                    )
                    if self.signal_quality_gate.learner:
                        logger.debug("✅ SignalQualityGate inizializzato con learner in _handle_live_opportunity")
                    else:
                        logger.warning("⚠️  SignalQualityGate inizializzato SENZA learner in _handle_live_opportunity!")
                
                # Prepara dati per validazione
                match_data = {
                    'home': opportunity.get('home', ''),
                    'away': opportunity.get('away', ''),
                    'league': opportunity.get('league', '')
                }
                
                # Estrai live_data da live_opp
                live_data = {}
                if hasattr(live_opp, 'match_stats') and live_opp.match_stats:
                    live_data = live_opp.match_stats.copy()
                else:
                    # Fallback: usa dati da opportunity
                    live_data = {
                        'minute': opportunity.get('minute', 0),
                        'score_home': opportunity.get('score_home', 0),
                        'score_away': opportunity.get('score_away', 0),
                        'shots_home': opportunity.get('shots_home', 0),
                        'shots_away': opportunity.get('shots_away', 0),
                        'shots_on_target_home': opportunity.get('shots_on_target_home', 0),
                        'shots_on_target_away': opportunity.get('shots_on_target_away', 0),
                        'possession_home': opportunity.get('possession_home'),
                        'xg_home': opportunity.get('xg_home', 0.0),
                        'xg_away': opportunity.get('xg_away', 0.0)
                    }
                
                # Valida qualità segnale (solo se Signal Quality Gate è disponibile)
                if self.signal_quality_gate:
                    try:
                        should_send, quality_score = self.signal_quality_gate.should_send_signal(
                            opportunity=opportunity,
                            match_data=match_data,
                            live_data=live_data
                        )
                        # Verifica che quality_score sia valido
                        if quality_score is not None and not hasattr(quality_score, 'total_score'):
                            logger.error(f"❌ quality_score non ha attributo total_score: {type(quality_score)}")
                            quality_score = None
                    except Exception as e:
                        logger.error(f"❌ Errore durante should_send_signal: {e}", exc_info=True)
                        should_send = True
                        quality_score = None
                else:
                    # Se Signal Quality Gate non disponibile, approva sempre
                    should_send = True
                    quality_score = None
                
                # Salva in cache solo se quality_score è valido
                if quality_score is not None:
                    self.quality_score_cache[opp_key] = quality_score
                
            except ImportError as e:
                logger.warning(f"⚠️  Signal Quality Gate non disponibile: {e}")
                quality_score = None
            except Exception as e:
                logger.error(f"❌ Errore Signal Quality Gate: {e}", exc_info=True)
                quality_score = None
                # In caso di errore, continua comunque (non bloccare tutto il sistema)
        
        # Valida Quality Score se disponibile
        if quality_score is not None:
            # Verifica che quality_score abbia gli attributi necessari
            if not hasattr(quality_score, 'total_score'):
                logger.error(f"❌ quality_score non ha attributo total_score: {type(quality_score)}")
                quality_score = None
            elif not hasattr(quality_score, 'is_approved'):
                logger.error(f"❌ quality_score non ha attributo is_approved: {type(quality_score)}")
                quality_score = None
        
        # 🆕 FIX: Registra segnale PRIMA del controllo should_send (così anche i bloccati vengono registrati)
        if not hasattr(self, 'signal_quality_learner'):
            logger.warning(f"⚠️  signal_quality_learner non esiste come attributo")
        elif self.signal_quality_learner is None:
            logger.warning(f"⚠️  signal_quality_learner è None")
        
        if hasattr(self, 'signal_quality_learner') and self.signal_quality_learner:
            try:
                # Usa quality_score dalla cache o quello appena calcolato
                cached_quality_score = self.quality_score_cache.get(opp_key) if opp_key in self.quality_score_cache else quality_score
                if cached_quality_score and hasattr(cached_quality_score, 'total_score'):
                    # Segnale valutato, registra con quality_score
                    record_id = self.signal_quality_learner.record_signal(
                        match_id=match_id,
                        market=market,
                        minute=minute,
                        score_home=live_opp.match_stats.get('score_home', 0) if live_opp.match_stats else 0,
                        score_away=live_opp.match_stats.get('score_away', 0) if live_opp.match_stats else 0,
                        quality_score=cached_quality_score.total_score,
                        context_score=cached_quality_score.context_score if hasattr(cached_quality_score, 'context_score') else 0.0,
                        data_quality_score=cached_quality_score.data_quality_score if hasattr(cached_quality_score, 'data_quality_score') else 0.0,
                        logic_score=cached_quality_score.logic_score if hasattr(cached_quality_score, 'logic_score') else 0.0,
                        timing_score=cached_quality_score.timing_score if hasattr(cached_quality_score, 'timing_score') else 0.0,
                        was_approved=cached_quality_score.is_approved if hasattr(cached_quality_score, 'is_approved') else True,
                        block_reasons=cached_quality_score.reasons if hasattr(cached_quality_score, 'reasons') else [],
                        confidence=getattr(live_opp, 'confidence', 0.0),
                        ev=getattr(live_opp, 'ev', 0.0)
                    )
                    status_text = "APPROVATO" if cached_quality_score.is_approved else "BLOCCATO"
                    source_text = "da cache" if opp_key in self.quality_score_cache else "calcolato"
                    logger.info(f"📝 Segnale registrato nel database (ID: {record_id}): {match_id}/{market} - {status_text} (QS: {cached_quality_score.total_score:.1f}, {source_text})")
                elif quality_score and hasattr(quality_score, 'total_score'):
                    # Quality score appena calcolato ma non in cache
                    record_id = self.signal_quality_learner.record_signal(
                        match_id=match_id,
                        market=market,
                        minute=minute,
                        score_home=live_opp.match_stats.get('score_home', 0) if live_opp.match_stats else 0,
                        score_away=live_opp.match_stats.get('score_away', 0) if live_opp.match_stats else 0,
                        quality_score=quality_score.total_score,
                        context_score=quality_score.context_score if hasattr(quality_score, 'context_score') else 0.0,
                        data_quality_score=quality_score.data_quality_score if hasattr(quality_score, 'data_quality_score') else 0.0,
                        logic_score=quality_score.logic_score if hasattr(quality_score, 'logic_score') else 0.0,
                        timing_score=quality_score.timing_score if hasattr(quality_score, 'timing_score') else 0.0,
                        was_approved=quality_score.is_approved if hasattr(quality_score, 'is_approved') else True,
                        block_reasons=quality_score.reasons if hasattr(quality_score, 'reasons') else [],
                        confidence=getattr(live_opp, 'confidence', 0.0),
                        ev=getattr(live_opp, 'ev', 0.0)
                    )
                    status_text = "APPROVATO" if quality_score.is_approved else "BLOCCATO"
                    logger.info(f"📝 Segnale registrato nel database (ID: {record_id}): {match_id}/{market} - {status_text} (QS: {quality_score.total_score:.1f})")
                else:
                    # Segnale non valutato, registra con valori di default
                    logger.warning(f"⚠️  quality_score non disponibile per {opp_key}, registro con valori di default")
                    record_id = self.signal_quality_learner.record_signal(
                        match_id=match_id,
                        market=market,
                        minute=minute,
                        score_home=live_opp.match_stats.get('score_home', 0) if live_opp.match_stats else 0,
                        score_away=live_opp.match_stats.get('score_away', 0) if live_opp.match_stats else 0,
                        quality_score=75.0,  # Default
                        context_score=75.0,
                        data_quality_score=75.0,
                        logic_score=75.0,
                        timing_score=75.0,
                        was_approved=True,
                        block_reasons=[],
                        confidence=getattr(live_opp, 'confidence', 0.0),
                        ev=getattr(live_opp, 'ev', 0.0)
                    )
                    logger.info(f"📝 Segnale registrato nel database (ID: {record_id}): {match_id}/{market} - APPROVATO (default)")
            except Exception as e:
                logger.error(f"❌ Errore registrazione segnale: {e}", exc_info=True)
        
        # 🎯 RIMOSSO: Controllo should_send basato su quality_score - l'utente vuole vedere tutte le opportunità
        # Forza sempre should_send = True per inviare tutte le opportunità
        should_send = True
        if quality_score is not None:
                try:
                    score_value = quality_score.total_score if hasattr(quality_score, 'total_score') else 0.0
                    logger.info(
                    f"✅ Segnale {match_id}/{market} approvato (Quality Score: {score_value:.1f}/100)"
                    )
                except Exception as e:
                    logger.debug(f"⚠️  Errore durante log quality_score: {e}")
        
        # 🔧 MIGLIORATO: Evita duplicati usando match_id + market + minuto
        # Questo evita di inviare la stessa opportunità più volte anche se rilevata in cicli diversi
        # (market e minute già definiti sopra)
        status = None
        if live_opp.match_stats:
            status = live_opp.match_stats.get('status', None)
            # 🔧 FIX: Estrai minute anche da match_stats se disponibile
            if minute is None or minute == 0:
                minute = live_opp.match_stats.get('minute', 0)
        
        # 🔧 FIX TIMING: Filtra partite finite - NON inviare notifiche per partite già terminate
        # Verifica status PRIMA del minuto (più affidabile)
        if status and status.upper() in ["FINISHED", "FT", "AET", "PEN"]:
            logger.warning(f"⏭️  Partita {match_id} saltata: partita già finita (status: {status}) - notifica non inviata")
            return
        # Verifica anche minuto (se > 90, partita probabilmente finita)
        if minute and minute > 90:
            logger.warning(f"⏭️  Partita {match_id} saltata: partita già finita (minuto: {minute} > 90) - notifica non inviata")
            return
        
        # 🔧 FIX: Definisci 'now' prima di usarlo
        now = datetime.now()
        
        # 🔧 FIX: Rimosso controllo globale troppo restrittivo - ora usa solo controlli per partita/mercato
        # Il controllo globale di 10 minuti bloccava TUTTE le notifiche, anche per partite diverse
        # Ora usiamo solo controlli più specifici (per partita, mercato, opportunità)
        
        # 🆕 NUOVO: Blocca partita per 15 minuti (max 1 notifica ogni 15 minuti per partita)
        if match_id in self.notified_matches_timestamps:
            last_match_notification = self.notified_matches_timestamps[match_id]
            time_diff_match = (now - last_match_notification).total_seconds() / 60  # Differenza in minuti
            if time_diff_match < 15:  # Blocco 15 minuti per partita
                logger.info(f"⏭️  Match {match_id} already notified {time_diff_match:.1f} minutes ago (blocked for 15 min), skipping")
                return
        
        # 🔧 OPZIONE 4: Blocca stesso mercato per partita per 30 minuti (invece di bloccare tutta la partita)
        # Controlla se questo mercato è stato già suggerito per questa partita di recente
        if match_id in self.match_markets_history:
            for market_entry in self.match_markets_history[match_id]:
                if market_entry['market'] == market:
                    time_diff_market = (now - market_entry['timestamp']).total_seconds() / 60
                    if time_diff_market < 30:  # 🔧 OPZIONE 4: Blocco 30 minuti per stesso mercato
                        logger.info(f"⏭️  Market {market} for match {match_id} already notified {time_diff_market:.1f} minutes ago (blocked for 30 min), skipping")
                        return
        
        # Crea chiave più specifica: match_id + market + minuto (arrotondato a multipli di 5 per evitare duplicati per minuti simili)
        minute_rounded = (minute // 5) * 5  # Arrotonda a multipli di 5 (es. 23' -> 20', 27' -> 25')
        opp_key = f"{match_id}_{market}_{minute_rounded}"
        
        # 🔧 NUOVO: Controlla se questa specifica opportunità è stata già notificata di recente (ultimi 15 minuti)
        if opp_key in self.notified_opportunities_timestamps:
            last_notified = self.notified_opportunities_timestamps[opp_key]
            time_diff = (now - last_notified).total_seconds() / 60  # Differenza in minuti
            
            # Se è stata notificata meno di 15 minuti fa, salta
            if time_diff < 15:
                logger.info(f"⏭️  Live opportunity {opp_key} already notified {time_diff:.1f} minutes ago, skipping")
                return
            else:
                # Rimuovi dalla cache se è passato più di 15 minuti (per liberare memoria)
                del self.notified_opportunities_timestamps[opp_key]
                self.notified_opportunities.discard(opp_key)
        
        # Controllo classico (backup)
        if opp_key in self.notified_opportunities:
            logger.info(f"⏭️  Live opportunity {opp_key} already notified, skipping")
            return
        
        # Notifica Telegram (la registrazione è già stata fatta prima del controllo should_send)
        if self.notifier:
            try:
                # Formatta messaggio e invia con _send_message (metodo privato ma usato internamente)
                message = self.live_betting_advisor.format_live_betting_message(live_opp)
                
                if message:
                    # 🔧 VERIFICA MESSAGGIO: Log completo del messaggio prima dell'invio
                    logger.info(f"📱 MESSAGGIO TELEGRAM (prima dell'invio):")
                    logger.info(f"   Match: {match_id}")
                    logger.info(f"   Market: {market}")
                    # Verifica dati in live_opp
                    logger.info(f"   live_opp.match_stats: {live_opp.match_stats}")
                    logger.info(f"   live_opp.match_data: {live_opp.match_data}")
                    # Estrai score dal messaggio o da live_opp
                    if live_opp.match_stats:
                        stats = live_opp.match_stats
                        score_home = stats.get('score_home', 0)
                        score_away = stats.get('score_away', 0)
                        minute = stats.get('minute', 0)
                        logger.info(f"   Score da match_stats: {score_home}-{score_away} al {minute}'")
                    else:
                        logger.warning(f"   ⚠️  live_opp.match_stats è None o vuoto!")
                    # Mostra prime 15 righe del messaggio
                    message_lines = message.split('\n')
                    total_lines = len(message_lines)
                    message_lines_preview = message_lines[:15]
                    logger.info(f"   Contenuto messaggio (prime 15 righe):")
                    for i, line in enumerate(message_lines_preview, 1):
                        logger.info(f"      {i:2d}. {line}")
                    if total_lines > 15:
                        remaining_lines = total_lines - 15
                        logger.info(f"      ... (altre {remaining_lines} righe)")
                    
                    # Usa _send_message (metodo privato ma usato in altri punti del codice)
                    # 🔧 FIX: Rimosso aggiornamento timestamp globale (non più necessario)
                    success = self.notifier._send_message(message, parse_mode="HTML")
                    if success:
                        self.notified_opportunities.add(opp_key)
                        self.notified_opportunities_timestamps[opp_key] = datetime.now()
                        self.notified_matches_timestamps[match_id] = datetime.now()  # Traccia anche per partita
                        logger.info(f"✅ Notifica inviata con successo: {opp_key}")
                        
                        # 🔧 OPZIONE 4: Traccia mercato suggerito per questa partita
                        if match_id not in self.match_markets_history:
                            self.match_markets_history[match_id] = []
                        self.match_markets_history[match_id].append({
                            'market': market,
                            'timestamp': datetime.now()
                        })
                        # Pulisci entry vecchie (> 60 minuti) per liberare memoria
                        self.match_markets_history[match_id] = [
                            entry for entry in self.match_markets_history[match_id]
                            if (datetime.now() - entry['timestamp']).total_seconds() / 60 < 60
                        ]
                        
                        # 🆕 Pulisci cache Quality Score vecchia (> 30 minuti) per liberare memoria
                        # Rimuovi solo entry vecchie dalla cache
                        keys_to_remove = []
                        for cached_key in list(self.quality_score_cache.keys()):
                            # Estrai timestamp dalla chiave se possibile, altrimenti rimuovi dopo 30 minuti
                            # Per semplicità, limitiamo la cache a 100 entry
                            if len(self.quality_score_cache) > 100:
                                keys_to_remove.append(cached_key)
                        for key in keys_to_remove[:50]:  # Rimuovi max 50 alla volta
                            del self.quality_score_cache[key]
                        
                        # 🔧 NUOVO: Salva opportunità nel performance tracker
                        if hasattr(self, 'live_performance_tracker') and self.live_performance_tracker:
                            try:
                                self.live_performance_tracker.save_live_opportunity(live_opp, opportunity.get('match_data', {}))
                                logger.debug(f"✅ Live opportunity salvata nel tracker: {opp_key}")
                            except Exception as e:
                                logger.warning(f"⚠️  Errore salvataggio nel tracker: {e}")
                        
                        logger.info(f"✅ Notified live opportunity: {opp_key} (minute: {minute}')")
                    else:
                        logger.warning(f"⚠️  Failed to notify live opportunity: {opp_key} (ma segnale già registrato: ID {record_id})")
                else:
                    logger.warning(f"⚠️  Empty message for live opportunity: {opp_key}")
            except Exception as e:
                logger.error(f"❌ Error notifying live opportunity: {e}")
        else:
            logger.warning("⚠️  Telegram notifier not available")
    
    def _reset_api_usage_if_needed(self):
        """Reset API usage se nuovo giorno"""
        today = datetime.now().date()
        if today > self.last_api_reset:
            self.api_usage_today = 0
            self.last_api_reset = today
            self.notified_opportunities.clear()  # Reset notifiche
            self.notified_opportunities_timestamps.clear()  # Reset timestamp
            self.notified_matches_timestamps.clear()  # Reset timestamp partite
            self.match_markets_history.clear()  # 🔧 OPZIONE 4: Reset storico mercati
            self.quality_score_cache.clear()  # 🆕 Reset cache Quality Score
            logger.info("🔄 New day: API usage reset")
            
            # Invia report giornaliero (se disponibile)
            if self.automated_reports and today > self.last_daily_report:
                try:
                    # 🧠 NUOVO: Aggiungi analisi pattern e ottimizzazione parametri
                    enhanced_report = self._generate_enhanced_report('daily')
                    # Invia report (additional_content sarà aggiunto se supportato)
                    try:
                        self.automated_reports.send_daily_report(additional_content=enhanced_report)
                    except TypeError:
                        # Se non supporta additional_content, invia normale e poi invia enhanced separatamente
                        self.automated_reports.send_daily_report()
                        if enhanced_report and self.notifier:
                            self.notifier.send_message(f"📊 AI Insights Daily:\n{enhanced_report}")
                    self.last_daily_report = today
                    logger.info("✅ Daily report sent (with AI insights)")
                except Exception as e:
                    logger.warning(f"⚠️  Error sending daily report: {e}")
            
            # Invia report settimanale (se disponibile)
            if self.automated_reports:
                days_since_weekly = (datetime.now() - self.last_weekly_report).days
                if days_since_weekly >= 7:
                    try:
                        # 🧠 NUOVO: Aggiungi analisi pattern e ottimizzazione parametri
                        enhanced_report = self._generate_enhanced_report('weekly')
                        # Invia report (additional_content sarà aggiunto se supportato)
                        try:
                            self.automated_reports.send_weekly_report(additional_content=enhanced_report)
                        except TypeError:
                            # Se non supporta additional_content, invia normale e poi invia enhanced separatamente
                            self.automated_reports.send_weekly_report()
                            if enhanced_report and self.notifier:
                                self.notifier.send_message(f"📊 AI Insights Weekly:\n{enhanced_report}")
                        self.last_weekly_report = datetime.now()
                        logger.info("✅ Weekly report sent (with AI insights)")
                    except Exception as e:
                        logger.warning(f"⚠️  Error sending weekly report: {e}")
            
            # 🔧 NUOVO: Invia report giornaliero live betting
            if hasattr(self, 'live_betting_reports') and self.live_betting_reports:
                try:
                    self.live_betting_reports.send_daily_report()
                    logger.info("✅ Live betting daily report sent")
                except Exception as e:
                    logger.warning(f"⚠️  Errore invio report giornaliero live betting: {e}")
            
            # 🔧 NUOVO: Invia report settimanale live betting
            if hasattr(self, 'live_betting_reports') and self.live_betting_reports:
                try:
                    days_since_weekly = (datetime.now() - self.last_weekly_report).days
                    if days_since_weekly >= 7:
                        self.live_betting_reports.send_weekly_report()
                        # Verifica e invia alert se win rate basso
                        self.live_betting_reports.check_and_send_alerts()
                        self.last_weekly_report = datetime.now()
                        logger.info("✅ Live betting weekly report sent")
                except Exception as e:
                    logger.warning(f"⚠️  Errore invio report settimanale live betting: {e}")
    
    def _generate_enhanced_report(self, report_type: str) -> str:
        """
        Genera report potenziato con analisi pattern e ottimizzazione parametri.
        
        Args:
            report_type: 'daily' o 'weekly'
            
        Returns:
            Stringa con contenuto aggiuntivo per report
        """
        try:
            content_parts = []
            
            # 1. Analisi Pattern con LLM
            if self.pattern_analyzer and self.results_tracker:
                try:
                    # Ottieni risultati storici
                    all_results = self.results_tracker.get_all_opportunities()
                    if all_results:
                        days = 7 if report_type == 'daily' else 30
                        pattern_analysis = self.pattern_analyzer.analyze_performance_patterns(
                            all_results, days=days
                        )
                        
                        if pattern_analysis.get('insights'):
                            content_parts.append("\n📊 ANALISI PATTERN AI:\n")
                            for insight in pattern_analysis['insights'][:5]:
                                content_parts.append(f"  • {insight.get('message', '')}")
                            
                            if pattern_analysis.get('recommendations'):
                                content_parts.append("\n💡 RACCOMANDAZIONI:\n")
                                for rec in pattern_analysis['recommendations'][:3]:
                                    content_parts.append(f"  • {rec.get('action', '')}: {rec.get('reason', '')}")
                except Exception as e:
                    logger.debug(f"⚠️  Pattern analysis error: {e}")
            
            # 2. Ottimizzazione Parametri
            if self.parameter_optimizer and self.results_tracker:
                try:
                    all_results = self.results_tracker.get_all_opportunities()
                    if all_results:
                        current_params = {
                            'min_ev': self.min_ev,
                            'min_confidence': self.min_confidence
                        }
                        
                        days = 7 if report_type == 'daily' else 30
                        optimization = self.parameter_optimizer.optimize_parameters(
                            all_results, current_params, days=days
                        )
                        
                        if optimization.get('recommendation') == 'UPDATE':
                            content_parts.append("\n⚙️ OTTIMIZZAZIONE PARAMETRI:\n")
                            content_parts.append(f"  Parametri Attuali: EV≥{current_params['min_ev']}%, Conf≥{current_params['min_confidence']}%")
                            suggested = optimization.get('suggested_params', {})
                            content_parts.append(f"  Parametri Suggeriti: EV≥{suggested.get('min_ev', 0)}%, Conf≥{suggested.get('min_confidence', 0)}%")
                            improvement = optimization.get('improvement_estimate', 0)
                            if improvement > 0:
                                content_parts.append(f"  Miglioramento Stimato: +{improvement:.1f}% ROI")
                except Exception as e:
                    logger.debug(f"⚠️  Parameter optimization error: {e}")
            
            return "\n".join(content_parts) if content_parts else ""
            
        except Exception as e:
            logger.warning(f"⚠️  Error generating enhanced report: {e}")
            return ""
    
    def _monitor_odds_movements(self):
        """Monitora movimenti quote e invia alert"""
        if not self.odds_monitor:
            return
        
        try:
            # Ottieni partite monitorate
            matches = list(self.monitored_matches.values())
            
            for match in matches:
                match_id = match.get('id')
                if not match_id:
                    continue
                
                # Registra quote attuali
                movements = self.odds_monitor.record_odds(
                    match_id=match_id,
                    odds_home=match.get('odds_1', 0),
                    odds_draw=match.get('odds_x', 0),
                    odds_away=match.get('odds_2', 0)
                )
                
                # Invia alert per movimenti significativi
                for movement in movements:
                    if movement.is_sharp_money or abs(movement.movement_percent) > 5:
                        if self.notifier:
                            message = (
                                f"⚡ Movimento Quote Rilevato!\n\n"
                                f"Match: {match.get('home')} vs {match.get('away')}\n"
                                f"Market: {movement.market.upper()}\n"
                                f"Quote: {movement.old_odds:.2f} → {movement.new_odds:.2f}\n"
                                f"Movimento: {movement.movement_percent:+.2f}%\n"
                                f"{'🚨 SHARP MONEY!' if movement.is_sharp_money else ''}"
                            )
                            try:
                                self.notifier._send_message(message)
                            except Exception as e:
                                logger.debug(f"Failed to send odds movement notification: {e}")
        except Exception as e:
            logger.debug(f"⚠️  Error monitoring odds: {e}")
    
    def _update_match_results(self):
        """Aggiorna risultati partite automaticamente"""
        if not self.result_tracker_auto:
            logger.debug("⚠️  ResultTrackerAuto non disponibile")
            return
        
        try:
            # Aggiungi partite da tracciare
            tracked_count = 0
            for match_id, match in self.monitored_matches.items():
                self.result_tracker_auto.add_match_to_track(
                    match_id=match_id,
                    home_team=match.get('home', ''),
                    away_team=match.get('away', ''),
                    league=match.get('league'),
                    match_date=match.get('date')
                )
                tracked_count += 1
            
            if tracked_count > 0:
                logger.info(f"📊 Tracciate {tracked_count} partite per aggiornamento risultati")
            
            # Aggiorna risultati
            updated_results = self.result_tracker_auto.update_results()
            
            if updated_results:
                logger.info(f"✅ Aggiornati {len(updated_results)} risultati partite")
            else:
                logger.debug(f"ℹ️  Nessun risultato aggiornato (partite tracciate: {len(self.result_tracker_auto.tracked_matches)})")
            
            # 🆕 Aggiorna Signal Quality Learner con risultati finali
            for result in updated_results:
                if result.status == "FINISHED":
                    # Aggiorna Signal Quality Learner con risultati finali
                    if hasattr(self, 'signal_quality_learner') and self.signal_quality_learner:
                        try:
                            match_id = result.match_id if hasattr(result, 'match_id') else None
                            if match_id and result.home_score is not None and result.away_score is not None:
                                # 🆕 Passa eventi e statistiche per valutazione mercati complessi
                                events = getattr(result, 'events', None)
                                statistics = getattr(result, 'statistics', None)
                                updated_count = self.signal_quality_learner.update_signal_result(
                                    match_id=match_id,
                                    final_score_home=result.home_score,
                                    final_score_away=result.away_score,
                                    events=events,
                                    statistics=statistics
                                )
                                if updated_count > 0:
                                    logger.info(f"✅ Aggiornati {updated_count} segnali per partita {match_id} (risultato: {result.home_score}-{result.away_score})")
                                    
                                    # 🆕 Notifica aggiornamento risultati per apprendimento
                                    if self.notifier and updated_count > 0:
                                        try:
                                            home_team = result.home_team if hasattr(result, 'home_team') else 'Home'
                                            away_team = result.away_team if hasattr(result, 'away_team') else 'Away'
                                            # Conta segnali totali con risultati dopo aggiornamento
                                            conn = sqlite3.connect(self.signal_quality_learner.db_path)
                                            cursor = conn.cursor()
                                            cursor.execute("SELECT COUNT(*) FROM signal_records WHERE was_correct IS NOT NULL")
                                            total_with_results = cursor.fetchone()[0]
                                            
                                            # Conta quanti erano corretti e quanti sbagliati
                                            cursor.execute("SELECT COUNT(*) FROM signal_records WHERE match_id = ? AND was_correct = 1", (match_id,))
                                            correct_count = cursor.fetchone()[0]
                                            cursor.execute("SELECT COUNT(*) FROM signal_records WHERE match_id = ? AND was_correct = 0", (match_id,))
                                            wrong_count = cursor.fetchone()[0]
                                            
                                            # Conta totale segnali per questa partita
                                            cursor.execute("SELECT COUNT(*) FROM signal_records WHERE match_id = ?", (match_id,))
                                            total_match_signals = cursor.fetchone()[0]
                                            
                                            conn.close()
                                            
                                            # Calcola percentuale corretti
                                            correct_percent = (correct_count / total_match_signals * 100) if total_match_signals > 0 else 0
                                            
                                            message = (
                                                f"⚽ <b>IA: Partita Finita - Risultati Aggiornati</b>\n\n"
                                                f"🏆 {home_team} {result.home_score}-{result.away_score} {away_team}\n\n"
                                                f"📊 <b>Segnali per questa partita:</b>\n"
                                                f"   • Totale: {total_match_signals}\n"
                                                f"   • ✅ Corretti: {correct_count} ({correct_percent:.1f}%)\n"
                                                f"   • ❌ Sbagliati: {wrong_count}\n"
                                                f"   • Aggiornati: {updated_count}\n\n"
                                                f"📈 <b>Progresso Apprendimento:</b>\n"
                                                f"   • Segnali con risultati: {total_with_results}/50\n"
                                            )
                                            
                                            # Aggiungi barra progresso
                                            progress_percent = (total_with_results / 50 * 100) if 50 > 0 else 0
                                            bar_length = 20
                                            filled = int(progress_percent / 100 * bar_length)
                                            bar = "█" * filled + "░" * (bar_length - filled)
                                            message += f"   <code>{bar}</code> {progress_percent:.0f}%\n\n"
                                            
                                            if total_with_results >= 50:
                                                message += "🎉 <b>Pronto per apprendimento automatico!</b>"
                                            else:
                                                remaining = 50 - total_with_results
                                                message += f"⏳ Mancano {remaining} segnali per iniziare l'apprendimento"
                                            
                                            self.notifier._send_message(message, parse_mode="HTML")
                                            logger.info(f"📊 Notifica risultato partita: {home_team} {result.home_score}-{result.away_score} {away_team} - {correct_count}/{total_match_signals} corretti")
                                        except Exception as e:
                                            logger.debug(f"⚠️  Errore notifica aggiornamento risultato: {e}")
                        except Exception as e:
                            logger.debug(f"⚠️  Errore aggiornamento learner: {e}")
                    
                    # Aggiorna betting results tracker se partita finita
                    if self.results_tracker:
                        # Determina outcome scommessa (da implementare con logica specifica)
                        # self.result_tracker_auto.update_betting_results(...)
                        pass
                    
                    # 🔧 NUOVO: Aggiorna live performance tracker se partita finita
                    if hasattr(self, 'live_performance_tracker') and self.live_performance_tracker:
                        try:
                            match_id = result.match_id if hasattr(result, 'match_id') else None
                            if match_id and result.home_score is not None and result.away_score is not None:
                                self.live_performance_tracker.update_live_result(
                                    match_id=match_id,
                                    final_score_home=result.home_score,
                                    final_score_away=result.away_score
                                )
                                logger.info(f"✅ Live performance tracker aggiornato per match {match_id} (risultato: {result.home_score}-{result.away_score})")
                        except Exception as e:
                            logger.warning(f"⚠️  Errore aggiornamento live performance tracker: {e}")
        except Exception as e:
            logger.debug(f"⚠️  Error updating results: {e}")
    
    def _check_arbitrage(self, match: Dict):
        """Controlla arbitraggi per una partita"""
        if not self.arbitrage_detector:
            return
        
        try:
            # Per ora usa quote singole (da espandere con multiple bookmaker)
            # TheOddsAPI supporta multiple bookmaker, ma per ora usiamo best odds
            bookmaker_odds = {
                'best': {
                    'home': match.get('odds_1', 0),
                    'draw': match.get('odds_x', 0),
                    'away': match.get('odds_2', 0)
                }
            }
            
            # Rileva arbitraggio
            arbitrage = self.arbitrage_detector.detect_arbitrage(
                match_id=match.get('id'),
                match_data=match,
                bookmaker_odds=bookmaker_odds
            )
            
            # 🔧 FIX: Disabilitato invio notifiche arbitraggio (l'utente vuole solo live betting)
            if arbitrage:
                logger.debug(f"💰 Arbitraggio trovato (NON notificato): {match.get('id')} - Profitto: {arbitrage.get('profit_pct', 0):.2f}%")
                # Non inviare notifica arbitraggio
                # if self.notifier:
                #     message = self.arbitrage_detector.format_arbitrage_message(arbitrage)
                #     try:
                #         self.notifier._send_message(message)
                #         logger.info(f"💰 Arbitraggio trovato e notificato: {match.get('id')}")
                #     except Exception as e:
                #         logger.debug(f"Failed to send arbitrage notification: {e}")
        except Exception as e:
            logger.debug(f"⚠️  Error checking arbitrage: {e}")
    
    def _check_news_alerts(self):
        """Controlla news e invia alert per notizie importanti"""
        if not self.news_analyzer:
            return
        
        try:
            # Recupera news sportive
            news_items = self.news_analyzer.fetch_sports_news(query="football", max_results=10)
            
            # Filtra solo notizie importanti
            important_news = [
                news for news in news_items
                if news.importance in ['HIGH', 'CRITICAL']
            ]
            
            # Invia alert per notizie importanti
            for news in important_news:
                if self.notifier:
                    message = self.news_analyzer.format_news_alert(news)
                    try:
                        self.notifier._send_message(message)
                        logger.info(f"📰 News importante notificata: {news.title[:50]}")
                    except Exception as e:
                        logger.debug(f"Failed to send news alert notification: {e}")
        except Exception as e:
            logger.debug(f"⚠️  Error checking news: {e}")
    
    def _signal_handler(self, signum, frame):
        """Gestisce segnali di shutdown"""
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT" if signum == signal.SIGINT else f"Signal {signum}"
        logger.info(f"🛑 Received {signal_name}, shutting down gracefully...")
        self.running = False
        # Forza uscita immediata (non aspetta sleep)
        logger.info("✅ Shutdown signal processed, exiting...")
    
    def _load_last_global_notification_time(self):
        """Carica ultimo timestamp notifica globale da database persistente"""
        try:
            if hasattr(self, 'signal_quality_learner') and self.signal_quality_learner:
                conn = sqlite3.connect(self.signal_quality_learner.db_path)
                cursor = conn.cursor()
                
                # Crea tabella se non esiste
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS automation_state (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Carica timestamp
                cursor.execute("SELECT value FROM automation_state WHERE key = 'last_global_notification_time'")
                result = cursor.fetchone()
                conn.close()
                
                if result and result[0]:
                    timestamp_str = result[0]
                    self.last_global_notification_time = datetime.fromisoformat(timestamp_str)
                    logger.info(f"📥 Caricato timestamp globale persistente da database: {self.last_global_notification_time}")
                else:
                    logger.debug("ℹ️  Nessun timestamp globale persistente trovato nel database")
        except Exception as e:
            logger.debug(f"⚠️  Errore caricamento stato persistente: {e}")
            self.last_global_notification_time = None
    
    def _save_last_global_notification_time(self):
        """Salva ultimo timestamp notifica globale in database persistente"""
        try:
            if hasattr(self, 'signal_quality_learner') and self.signal_quality_learner and self.last_global_notification_time:
                conn = sqlite3.connect(self.signal_quality_learner.db_path)
                cursor = conn.cursor()
                
                # Crea tabella se non esiste
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS automation_state (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Salva timestamp
                timestamp_str = self.last_global_notification_time.isoformat()
                cursor.execute("""
                    INSERT OR REPLACE INTO automation_state (key, value, updated_at)
                    VALUES ('last_global_notification_time', ?, CURRENT_TIMESTAMP)
                """, (timestamp_str,))
                conn.commit()
                conn.close()
                logger.debug(f"💾 Timestamp globale salvato nel database: {self.last_global_notification_time}")
        except Exception as e:
            logger.debug(f"⚠️  Errore salvataggio stato persistente: {e}")
    
    def stop(self):
        """Ferma sistema"""
        logger.info("🛑 Stopping Automation24H system...")
        self.running = False
        
        # Salva stato persistente prima di chiudere
        if self.last_global_notification_time:
            self._save_last_global_notification_time()
        
        # Chiudi connessioni database se presenti
        if hasattr(self, 'signal_quality_learner') and self.signal_quality_learner:
            try:
                # Chiudi eventuali connessioni aperte
                pass
            except Exception as e:
                logger.debug(f"⚠️  Errore chiusura database: {e}")
        
        logger.info("✅ Automation24H stopped")


def main():
    """Main entry point con autoretry continuo"""
    parser = argparse.ArgumentParser(description='Automation 24/7 System')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--telegram-token', type=str, help='Telegram bot token')
    parser.add_argument('--telegram-chat-id', type=str, help='Telegram chat ID')
    parser.add_argument('--min-ev', type=float, default=8.0, help='Min EV % (default: 8.0)')
    parser.add_argument('--min-confidence', type=float, default=70.0, help='Min confidence % (default: 70.0)')
    parser.add_argument('--max-notifications', type=int, default=2, help='Max notifications per cycle (default: 2)')
    parser.add_argument('--update-interval', type=int, default=600, help='Update interval seconds (default: 600 = 10 min)')
    parser.add_argument('--single-run', action='store_true', help='Run once and exit (for cron jobs)')
    
    args = parser.parse_args()
    
    # Carica config se fornito o se esiste config.json nella directory corrente
    config = {}
    config_path = args.config or (Path('config.json') if Path('config.json').exists() else None)
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"✅ Config caricato da {config_path}")
    
    retry_delay = int(os.getenv('AUTOMATION_RETRY_DELAY', config.get('retry_delay', 180)))
    
    attempt = 0
    while True:
        attempt += 1
        logger.info(f"🔁 Avvio Automation24H (tentativo {attempt})")
        
        automation = Automation24H(
            config_path=args.config,
            telegram_token=args.telegram_token or config.get('telegram_token') or os.getenv('TELEGRAM_BOT_TOKEN') or os.getenv('TELEGRAM_TOKEN'),
            telegram_chat_id=args.telegram_chat_id or config.get('telegram_chat_id') or os.getenv('TELEGRAM_CHAT_ID'),
            min_ev=args.min_ev or config.get('min_ev', 8.0),
            min_confidence=args.min_confidence or config.get('min_confidence', 70.0),
            update_interval=args.update_interval or config.get('update_interval', 600),
            api_budget_per_day=config.get('api_budget_per_day', 7500),  # Piano Pro: 7500 chiamate/giorno
            max_notifications_per_cycle=args.max_notifications or config.get('max_notifications_per_cycle', 2)
        )
        
        try:
            automation.start(single_run=args.single_run)
            if args.single_run:
                break
        except Exception as e:
            logger.error(f"❌ Errore critico nel main loop: {e}", exc_info=True)
        
        if args.single_run:
            break
    
    logger.info(f"⏳ Riavvio automatico tra {retry_delay} secondi...")
    time.sleep(retry_delay)


if __name__ == '__main__':
    main()


