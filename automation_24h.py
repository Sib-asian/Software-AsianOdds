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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
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
                live_min_ev = max(6.0, min_ev - 2.0)  # Abbassa di 2% rispetto a pre-match (minimo 6%)
                # 🔧 NUOVO: Passa performance tracker per soglie dinamiche
                live_tracker = self.live_performance_tracker if hasattr(self, 'live_performance_tracker') else None
                self.live_betting_advisor = LiveBettingAdvisor(
                    notifier=self.notifier,
                    min_confidence=min_confidence,
                    min_ev=live_min_ev,
                    performance_tracker=live_tracker  # 🔧 NUOVO: Passa tracker
                )
                logger.info(f"   LiveBettingAdvisor: min_confidence={min_confidence}%, min_ev={live_min_ev}% (abbassato per live betting)")
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
                    min_ev=self.min_ev,
                    min_confidence=self.min_confidence,
                    rate_limit_seconds=3
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
                    time.sleep(sleep_interval)
                    elapsed += sleep_interval
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested")
        finally:
            self.stop()
    
    def _run_cycle(self):
        """Esegue un ciclo di analisi"""
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
        
        # 1.5. Applica filtri (se disponibili)
        if self.match_filters:
            filtered_matches = [m for m in matches if self.match_filters.should_analyze_match(m)]
            logger.info(f"   After filters: {len(filtered_matches)} matches")
            matches = filtered_matches
        
        if not matches:
            logger.info("   No matches after filters, skipping cycle")
            return
        
        # 2. Analizza ogni partita e raccogli tutte le opportunità
        all_opportunities = []  # Raccogli tutte le opportunità per selezionare le migliori
        opportunities_found = 0
        
        for match in matches:
            try:
                # 🆕 NUOVO: Rileva arbitraggi prima di analisi normale
                if self.arbitrage_detector:
                    self._check_arbitrage(match)
                
                # 🆕 SOLO LIVE BETTING: Analizza solo partite live
                if not self.live_betting_advisor:
                    logger.warning("⚠️  LiveBettingAdvisor not available, skipping live match")
                    continue
                
                opportunities = self._analyze_live_match(match)
                for opp in opportunities:
                    if opp:
                        opportunities_found += 1
                        all_opportunities.append(opp)  # Raccogli invece di inviare subito
            except Exception as e:
                logger.error(f"❌ Error analyzing match {match.get('id', 'unknown')}: {e}")
                continue
        
        # 🆕 NUOVO: Seleziona e invia solo le migliori opportunità
        notified_count = 0
        if all_opportunities:
            best_opportunities = self._select_best_opportunities(all_opportunities)
            logger.info(f"📊 Trovate {opportunities_found} opportunità, selezionate le migliori {len(best_opportunities)} per notifica")
            
            for opp_dict in best_opportunities:
                self._handle_live_opportunity(opp_dict)
                notified_count += 1
        
        logger.info(f"✅ Cycle complete: {opportunities_found} opportunities found, {notified_count} notified")
    
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
            # Reset quota se nuovo giorno
            if self.api_usage_today >= self.api_budget_per_day:
                logger.warning(f"⚠️  API quota exhausted ({self.api_usage_today}/{self.api_budget_per_day})")
                return []
            
            # Prova a ottenere partite reali da TheOddsAPI
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
        Ottiene partite reali da multiple fonti (API-SPORTS, TheOddsAPI, ecc.).
        
        🆕 PRIORITÀ: Usa MultiSourceMatchFinder (API-SPORTS) come primario.
        TheOddsAPI come fallback se MultiSource non disponibile.
        """
        # 🆕 NUOVO: Usa MultiSourceMatchFinder come primario (API-SPORTS)
        if self.multi_source_finder:
            try:
                logger.info("🔍 Usando sistema multi-fonte per trovare partite (TheOddsAPI + API-SPORTS + Football-Data.org)...")
                matches = self.multi_source_finder.find_all_matches(
                    days_ahead=1,
                    include_minor_leagues=True,
                    include_live=True
                )
                
                if matches:
                    logger.info(f"✅ Trovate {len(matches)} partite da sistema multi-fonte")
                    self.api_usage_today += 1  # Conta chiamata API
                    return matches
                else:
                    logger.info("ℹ️  Nessuna partita trovata da sistema multi-fonte")
            except Exception as e:
                logger.warning(f"⚠️  Errore sistema multi-fonte: {e}, fallback a TheOddsAPI")
        
        # Fallback a TheOddsAPI se MultiSource non disponibile o fallisce
        import os
        import requests
        from datetime import datetime, timedelta
        
        theodds_api_key = os.getenv("THEODDS_API_KEY", "")
        if not theodds_api_key:
            logger.debug("ℹ️  THEODDS_API_KEY non configurata, usando mock data")
            return []
        
        try:
            # TheOddsAPI endpoint per partite di calcio
            # Sport key: "soccer" per calcio
            url = "https://api.the-odds-api.com/v4/sports/soccer/odds"
            params = {
                "apiKey": theodds_api_key,
                "regions": "eu",  # Regione EU (include bookmaker italiani)
                "markets": "h2h",  # Head-to-head (1X2)
                "oddsFormat": "decimal",
                "dateFormat": "iso"
            }
            
            logger.debug(f"📡 Fetching matches from TheOddsAPI...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            events = response.json()
            if not events:
                logger.info("ℹ️  Nessuna partita trovata su TheOddsAPI")
                return []
            
            # Converti eventi TheOddsAPI in formato interno
            matches = []
            now = datetime.now(timezone.utc)  # Use timezone-aware UTC datetime
            max_future = now + timedelta(hours=24)  # Prossime 24h
            min_past = now - timedelta(hours=2)  # Partite iniziate nelle ultime 2h (live)
            
            for event in events:
                try:
                    # Estrai data partita
                    commence_time_str = event.get("commence_time")
                    if not commence_time_str:
                        continue
                    
                    # Parse ISO datetime as timezone-aware (UTC)
                    commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
                    
                    # Determina se è live o pre-match (compare timezone-aware datetimes)
                    is_live = commence_time < now
                    is_prematch = commence_time >= now
                    
                    # Filtra:
                    # - Pre-match: partite nelle prossime 24h
                    # - Live: partite iniziate nelle ultime 2h (per evitare partite già finite)
                    if is_prematch:
                        if commence_time > max_future:
                            continue  # Troppo in futuro
                    elif is_live:
                        if commence_time < min_past:
                            continue  # Troppo tempo fa (probabilmente finita)
                    else:
                        continue  # Non dovrebbe succedere
                    
                    # Estrai quote migliori da tutti i bookmaker
                    bookmakers = event.get("bookmakers", [])
                    best_odds = {"home": None, "draw": None, "away": None}
                    
                    for bookmaker in bookmakers:
                        markets = bookmaker.get("markets", [])
                        h2h_market = next((m for m in markets if m.get("key") == "h2h"), None)
                        if not h2h_market:
                            continue
                        
                        outcomes = h2h_market.get("outcomes", [])
                        for outcome in outcomes:
                            name = outcome.get("name", "").lower()
                            price = outcome.get("price")
                            
                            if price is None:
                                continue
                            
                            # Identifica outcome (home/away/draw)
                            home_team = event.get("home_team", "").lower()
                            away_team = event.get("away_team", "").lower()
                            
                            if name == home_team and (best_odds["home"] is None or price > best_odds["home"]):
                                best_odds["home"] = price
                            elif name == away_team and (best_odds["away"] is None or price > best_odds["away"]):
                                best_odds["away"] = price
                            elif name in ["draw", "pareggio", "x"] and (best_odds["draw"] is None or price > best_odds["draw"]):
                                best_odds["draw"] = price
                    
                    # Crea match solo se ha quote valide
                    if best_odds["home"] and best_odds["away"] and best_odds["draw"]:
                        match = {
                            'id': event.get("id", f"match_{len(matches)}"),
                            'home': event.get("home_team", ""),
                            'away': event.get("away_team", ""),
                            'league': event.get("sport_title", "Soccer"),
                            'date': commence_time,  # Keep timezone-aware datetime
                            'odds_1': best_odds["home"],
                            'odds_x': best_odds["draw"],
                            'odds_2': best_odds["away"],
                            'sport_key': event.get("sport_key", "soccer"),
                            'is_live': is_live,  # Flag per distinguere live/pre-match
                            'match_status': 'LIVE' if is_live else 'PRE-MATCH'
                        }
                        matches.append(match)
                
                except Exception as e:
                    logger.debug(f"⚠️  Error processing event: {e}")
                    continue
            
            logger.info(f"✅ Trovate {len(matches)} partite reali da TheOddsAPI")
            self.api_usage_today += 1  # Conta chiamata API
            
            return matches
            
        except requests.RequestException as e:
            logger.warning(f"⚠️  TheOddsAPI request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error fetching real matches: {e}")
            return []
    
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
            
            # Verifica filtro market (se disponibile)
            if self.match_filters:
                market = ai_result.get('final_decision', {}).get('market', '')
                if market and not self.match_filters.should_analyze_market(market):
                    logger.debug(f"   Market {market} filtered out")
                    return None
            
            # Verifica se è una vera opportunità VALUE BET
            if not self._is_real_value_opportunity(ai_result, match):
                return None
            
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
            match_data = {
                'home': match.get('home', ''),
                'away': match.get('away', ''),
                'league': match.get('league', ''),
                'odds_1': match.get('odds_1'),
                'odds_x': match.get('odds_x'),
                'odds_2': match.get('odds_2'),
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
            if minute is None:
                minute = 0
            elif isinstance(minute, str):
                try:
                    minute = int(minute.replace("'", "").replace("+", "").split()[0])
                except:
                    minute = 0
            elif not isinstance(minute, int):
                try:
                    minute = int(minute)
                except:
                    minute = 0
            
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
            logger.info(f"   score_home: {live_data.get('score_home', 'N/A')}")
            logger.info(f"   score_away: {live_data.get('score_away', 'N/A')}")
            logger.info(f"   minute: {live_data.get('minute', 'N/A')}")
            logger.info(f"   shots_on_target_home: {live_data.get('shots_on_target_home', 'N/A')}")
            
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
    
    def _is_real_value_opportunity(self, ai_result: Dict, match: Dict) -> bool:
        """
        Verifica se è una VERA opportunità VALUE BET.
        
        Criteri:
        1. Action deve essere BET (non WATCH/SKIP)
        2. EV > soglia minima
        3. Confidence > soglia minima
        4. NON basato su score (se live)
        5. Probabilità vs Quote deve avere vero valore
        """
        # 1. Check action
        action = ai_result.get('action') or ai_result.get('final_decision', {}).get('action')
        if action != 'BET':
            return False
        
        # 2. Check EV
        ev = ai_result.get('ev') or ai_result.get('summary', {}).get('expected_value', 0)
        if isinstance(ev, float) and ev < 1.0:
            ev = ev * 100  # Convert to %
        if ev < self.min_ev:
            logger.debug(f"   EV too low: {ev:.1f}% < {self.min_ev}%")
            return False
        
        # 3. Check confidence
        confidence = ai_result.get('confidence_level') or ai_result.get('summary', {}).get('confidence', 0)
        if confidence < self.min_confidence:
            logger.debug(f"   Confidence too low: {confidence:.1f}% < {self.min_confidence}%")
            return False
        
        # 4. Check se è basato su score (se live)
        # IMPORTANTE: Non vogliamo consigli tipo "1-0 quindi gioca 1"
        if self._is_score_based_recommendation(ai_result, match):
            logger.warning(f"   ⚠️  Rejecting score-based recommendation for {match.get('id')}")
            return False
        
        # 5. Verifica vero valore (probabilità vs quote)
        if not self._has_real_value(ai_result):
            logger.debug(f"   No real value detected")
            return False
        
        return True
    
    def _is_score_based_recommendation(self, ai_result: Dict, match: Dict) -> bool:
        """
        Verifica se la raccomandazione è basata solo su score.
        
        Questo è il problema che vogliamo evitare:
        "1-0 quindi gioca 1" - NON ha senso!
        """
        # Se non c'è score, non è basato su score
        current_score = match.get('current_score')
        if not current_score:
            return False
        
        # Estrai score
        try:
            home_score, away_score = map(int, current_score.split('-'))
        except:
            return False
        
        # Se score è 0-0, non è basato su score
        if home_score == 0 and away_score == 0:
            return False
        
        # Verifica se la raccomandazione è troppo correlata allo score
        market = ai_result.get('market') or ai_result.get('final_decision', {}).get('market', '')
        
        # Pattern da evitare:
        # - Score 1-0 e raccomanda HOME
        # - Score 0-1 e raccomanda AWAY
        # - Score 2-0 e raccomanda HOME
        # etc.
        
        if 'HOME' in market.upper() and home_score > away_score:
            # Score favorisce home, raccomanda home - potrebbe essere basato su score
            # Verifica se c'è altro reasoning oltre allo score
            reasoning = ai_result.get('llm_playbook', {}).get('text', '') if isinstance(ai_result.get('llm_playbook'), dict) else ''
            if 'score' in reasoning.lower() and len(reasoning) < 100:
                # Reasoning troppo breve e menziona score - probabilmente basato su score
                return True
        
        if 'AWAY' in market.upper() and away_score > home_score:
            reasoning = ai_result.get('llm_playbook', {}).get('text', '') if isinstance(ai_result.get('llm_playbook'), dict) else ''
            if 'score' in reasoning.lower() and len(reasoning) < 100:
                return True
        
        return False
    
    def _has_real_value(self, ai_result: Dict) -> bool:
        """
        Verifica se c'è vero valore (probabilità vs quote).
        
        True value = probabilità > probabilità implicita dalla quota
        """
        probability = ai_result.get('probability') or ai_result.get('summary', {}).get('probability')
        odds = ai_result.get('odds') or ai_result.get('summary', {}).get('odds')
        
        if not probability or not odds or odds <= 1.0:
            return False
        
        # Probabilità implicita dalla quota
        implied_prob = 1.0 / odds
        
        # Se probabilità reale > probabilità implicita + margine, c'è valore
        margin = 0.05  # 5% margine minimo
        if probability > implied_prob + margin:
            return True
        
        return False
    
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
        
        # 🔧 Filtra PRIMA le opportunità senza statistiche live
        valid_opportunities = []
        for opp_dict in opportunities:
            live_opp = opp_dict.get('live_opportunity')
            if not live_opp:
                continue
            
            # Salta opportunità senza statistiche live significative
            if hasattr(live_opp, "has_live_stats") and not live_opp.has_live_stats:
                continue
            
            valid_opportunities.append(opp_dict)
        
        # Se non ci sono opportunità valide, ritorna lista vuota
        if not valid_opportunities:
            logger.info(f"⚠️  Nessuna opportunità valida con statistiche live tra {len(opportunities)} totali")
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
            market = getattr(live_opp, 'market', 'unknown')
            market_type = market.split('_')[0] if '_' in market else market
            
            # 1. Calcola EV e Confidence
            ev = getattr(live_opp, 'ev', 0.0)
            confidence = getattr(live_opp, 'confidence', 0.0)
            
            # Normalizza EV (può essere negativo)
            ev_normalized = (ev / 100.0) + 1.0
            confidence_normalized = confidence / 100.0
            
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
                        min_quality_score=75.0,
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
            
            # 5. 🆕 Calcola Final Score composito
            final_score = (
                ev_normalized * 0.30 +
                confidence_normalized * 0.30 +
                quality_score_normalized * 0.30 +
                stats_bonus * 0.10
            ) * score_modifier
            
            minute = stats.get('minute', 0) if isinstance(stats, dict) else 0
            
            scored_opportunities.append({
                'opportunity': opp_dict,
                'score': final_score,
                'ev': ev,
                'confidence': confidence,
                'quality_score': quality_score_obj.total_score if quality_score_obj else 0.0,
                'stats_bonus': stats_bonus,
                'minute': minute,
                'score_modifier': score_modifier,
                'modifier_reason': modifier_reason,
                'opp_key': opp_key  # Per cache
            })
        
        # Ordina per score decrescente
        scored_opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # 🆕 FIX: Diversificazione mercati - penalizza mercati già inviati di recente
        # Conta quante volte ogni mercato è stato inviato nelle ultime notifiche
        market_counts = {}
        if hasattr(self, 'last_global_notification_time') and self.last_global_notification_time:
            # Conta mercati inviati nelle ultime 30 minuti (circa 3 notifiche)
            cutoff_time = datetime.now() - timedelta(minutes=30)
            for match_id_history, markets_list in self.match_markets_history.items():
                for market_entry in markets_list:
                    if market_entry['timestamp'] > cutoff_time:
                        market = market_entry['market']
                        market_counts[market] = market_counts.get(market, 0) + 1
        
        # Applica penalizzazione aggiuntiva per mercati già inviati di recente
        for opp in scored_opportunities:
            live_opp = opp['opportunity'].get('live_opportunity')
            if live_opp:
                market = getattr(live_opp, 'market', None)
                if market and market in market_counts:
                    # Penalizza in base a quante volte è stato inviato
                    penalty = 0.1 * market_counts[market]  # -10% per ogni volta inviato
                    opp['score'] *= (1.0 - min(penalty, 0.5))  # Max -50% di penalizzazione
                    current_reason = opp.get('modifier_reason', '')
                    opp['modifier_reason'] = f"{current_reason} (penalizzato -{penalty*100:.0f}%: mercato già inviato {market_counts[market]} volte)"
        
        # Riordina dopo penalizzazioni
        scored_opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # 🆕 Seleziona SOLO la migliore in assoluto (max 1)
        best = []
        if scored_opportunities:
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

        # 🔍 Assicura che ci siano statistiche live reali prima di notificare
        # NOTA: Questo controllo è ridondante ora che filtriamo in _select_best_opportunities,
        # ma lo manteniamo come sicurezza aggiuntiva
        if hasattr(live_opp, "has_live_stats") and not live_opp.has_live_stats:
            logger.debug(f"⏭️  Opportunità {match_id}/{live_opp.market} saltata: nessuna statistica live disponibile (dovrebbe essere già filtrata)")
            return
        
        # 🆕 AI SIGNAL QUALITY GATE: Validazione finale qualità segnale
        # 🆕 Ottimizzazione: Usa cache se disponibile per evitare doppio calcolo
        market = live_opp.market
        minute = 0
        if live_opp.match_stats:
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
        
        if quality_score is not None:
            should_send = quality_score.is_approved
            if not should_send:
                try:
                    score_value = quality_score.total_score if hasattr(quality_score, 'total_score') else 0.0
                    logger.info(
                        f"⏭️  Segnale {match_id}/{market} BLOCCATO da Signal Quality Gate "
                        f"(score: {score_value:.1f}/100, min: 75.0)"
                    )
                    if hasattr(quality_score, 'reasons') and quality_score.reasons:
                        logger.info(f"   Motivi blocco: {', '.join(quality_score.reasons)}")
                except Exception as e:
                    logger.error(f"❌ Errore durante logging blocco segnale: {e}", exc_info=True)
                return
        
        # 🔧 MIGLIORATO: Evita duplicati usando match_id + market + minuto
        # Questo evita di inviare la stessa opportunità più volte anche se rilevata in cicli diversi
        # (market e minute già definiti sopra)
        status = None
        if live_opp.match_stats:
            status = live_opp.match_stats.get('status', None)
        
        # 🔧 NUOVO: Filtra partite finite - NON inviare notifiche per partite già terminate
        if minute > 90 or (status and status.upper() in ["FINISHED", "FT", "AET", "PEN"]):
            logger.debug(f"⏭️  Partita {match_id} saltata: partita finita (minuto: {minute}, status: {status})")
            return
        
        # 🔧 FIX: Definisci 'now' prima di usarlo
        now = datetime.now()
        
        # 🆕 FIX: Limite globale 10 minuti tra qualsiasi notifica (CONTROLLO PRIMA DI TUTTO)
        if hasattr(self, 'last_global_notification_time') and self.last_global_notification_time:
            time_since_global = (now - self.last_global_notification_time).total_seconds() / 60
            if time_since_global < 10:  # Blocco globale 10 minuti
                logger.info(f"⏭️  Notifica globale bloccata: ultima notifica {time_since_global:.1f} minuti fa (minimo 10 minuti richiesti) - Match: {match_id}, Market: {market}")
                logger.debug(f"   Timestamp ultima notifica: {self.last_global_notification_time}, Ora attuale: {now}")
                return
        else:
            logger.debug(f"ℹ️  Nessun timestamp globale precedente, prima notifica consentita - Match: {match_id}, Market: {market}")
        
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
                    # 🆕 FIX: Aggiorna timestamp globale PRIMA di inviare (così il limite funziona anche se la notifica fallisce)
                    self.last_global_notification_time = datetime.now()  # 🆕 Aggiorna timestamp globale
                    self._save_last_global_notification_time()  # 🆕 Salva in modo persistente
                    
                    success = self.notifier._send_message(message, parse_mode="HTML")
                    if success:
                        self.notified_opportunities.add(opp_key)
                        self.notified_opportunities_timestamps[opp_key] = datetime.now()
                        self.notified_matches_timestamps[match_id] = datetime.now()  # Traccia anche per partita
                        logger.info(f"✅ Notifica inviata e timestamp globale aggiornato: {self.last_global_notification_time}")
                        
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
            
            # 🔧 NUOVO: Invia report settimanale live betting
            if hasattr(self, 'live_betting_reports') and self.live_betting_reports:
                try:
                    self.live_betting_reports.send_weekly_report()
                    # Verifica e invia alert se win rate basso
                    self.live_betting_reports.check_and_send_alerts()
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
            
            if arbitrage and self.notifier:
                message = self.arbitrage_detector.format_arbitrage_message(arbitrage)
                try:
                    self.notifier._send_message(message)
                    logger.info(f"💰 Arbitraggio trovato e notificato: {match.get('id')}")
                except Exception as e:
                    logger.debug(f"Failed to send arbitrage notification: {e}")
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
    """Main entry point"""
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
    
    # Crea sistema
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
    
    # Avvia (single_run per cron jobs)
    automation.start(single_run=args.single_run)


if __name__ == '__main__':
    main()

