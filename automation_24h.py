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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
import argparse

# Setup logging using centralized configuration
from logging_setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

# Import sistema esistente
try:
    from ai_system.pipeline import AIPipeline
    from ai_system.config import AIConfig
    from ai_system.telegram_notifier import TelegramNotifier
    from api_manager import APIManager
    AI_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    AI_SYSTEM_AVAILABLE = False

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
        update_interval: int = 300,  # 5 minuti
        api_budget_per_day: int = 100
    ):
        self.config_path = config_path
        self.min_ev = min_ev
        self.min_confidence = min_confidence
        self.update_interval = update_interval
        self.api_budget_per_day = api_budget_per_day
        
        # Stato
        self.running = False
        self.monitored_matches: Dict[str, Dict] = {}
        self.notified_opportunities: Set[str] = set()  # Evita duplicati
        self.api_usage_today = 0
        self.last_api_reset = datetime.now().date()
        self.last_daily_report = datetime.now().date()
        self.last_weekly_report = datetime.now() - timedelta(days=7)
        
        # Inizializza componenti
        self._init_components(telegram_token, telegram_chat_id)
        
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
        try:
            self.api_manager = APIManager()
            logger.info("✅ API Manager initialized")
        except Exception as e:
            logger.error(f"❌ API Manager error: {e}")
            self.api_manager = None
        
        # Telegram Notifier
        if telegram_token and telegram_chat_id:
            self.notifier = TelegramNotifier(
                bot_token=telegram_token,
                chat_id=telegram_chat_id,
                min_ev=self.min_ev,
                min_confidence=self.min_confidence,
                rate_limit_seconds=3
            )
            logger.info("✅ Telegram Notifier initialized")
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
                
                # Attendi prima del prossimo ciclo
                time.sleep(self.update_interval)
                
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
        
        # 1. Ottieni partite da monitorare
        matches = self._get_matches_to_monitor()
        logger.info(f"   Found {len(matches)} matches to monitor")
        
        if not matches:
            logger.info("   No matches to monitor, skipping cycle")
            return
        
        # 1.5. Applica filtri (se disponibili)
        if self.match_filters:
            filtered_matches = [m for m in matches if self.match_filters.should_analyze_match(m)]
            logger.info(f"   After filters: {len(filtered_matches)} matches")
            matches = filtered_matches
        
        if not matches:
            logger.info("   No matches after filters, skipping cycle")
            return
        
        # 2. Analizza ogni partita
        opportunities_found = 0
        for match in matches:
            try:
                # 🆕 NUOVO: Rileva arbitraggi prima di analisi normale
                if self.arbitrage_detector:
                    self._check_arbitrage(match)
                
                opportunity = self._analyze_match(match)
                if opportunity:
                    opportunities_found += 1
                    self._handle_opportunity(opportunity)
                    
                    # 🆕 NUOVO: Schedula alert pre-partita
                    if self.pre_match_alerter and match.get('date'):
                        try:
                            match_date = match.get('date')
                            if isinstance(match_date, str):
                                match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                            self.pre_match_alerter.schedule_alert(
                                match_id=match.get('id'),
                                match_data=opportunity.get('match_data', {}),
                                match_date=match_date,
                                alert_type='OPPORTUNITY'
                            )
                        except Exception as e:
                            logger.debug(f"⚠️  Error scheduling pre-match alert: {e}")
            except Exception as e:
                logger.error(f"❌ Error analyzing match {match.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"✅ Cycle complete: {opportunities_found} opportunities found")
    
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
        Ottiene partite reali da TheOddsAPI.
        
        TheOddsAPI fornisce:
        - Partite di calcio (soccer) con quote
        - Partite nelle prossime 24h
        - Quote da vari bookmaker
        """
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
    
    def _reset_api_usage_if_needed(self):
        """Reset API usage se nuovo giorno"""
        today = datetime.now().date()
        if today > self.last_api_reset:
            self.api_usage_today = 0
            self.last_api_reset = today
            self.notified_opportunities.clear()  # Reset notifiche
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
            return
        
        try:
            # Aggiungi partite da tracciare
            for match_id, match in self.monitored_matches.items():
                self.result_tracker_auto.add_match_to_track(
                    match_id=match_id,
                    home_team=match.get('home', ''),
                    away_team=match.get('away', ''),
                    league=match.get('league'),
                    match_date=match.get('date')
                )
            
            # Aggiorna risultati
            updated_results = self.result_tracker_auto.update_results()
            
            # Aggiorna betting results tracker se partita finita
            for result in updated_results:
                if result.status == "FINISHED" and self.results_tracker:
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
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.running = False
    
    def stop(self):
        """Ferma sistema"""
        logger.info("🛑 Stopping Automation24H system...")
        self.running = False
        logger.info("✅ Automation24H stopped")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Automation 24/7 System')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--telegram-token', type=str, help='Telegram bot token')
    parser.add_argument('--telegram-chat-id', type=str, help='Telegram chat ID')
    parser.add_argument('--min-ev', type=float, default=8.0, help='Min EV % (default: 8.0)')
    parser.add_argument('--min-confidence', type=float, default=70.0, help='Min confidence % (default: 70.0)')
    parser.add_argument('--update-interval', type=int, default=300, help='Update interval seconds (default: 300)')
    parser.add_argument('--single-run', action='store_true', help='Run once and exit (for cron jobs)')
    
    args = parser.parse_args()
    
    # Carica config se fornito
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Crea sistema
    automation = Automation24H(
        config_path=args.config,
        telegram_token=args.telegram_token or config.get('telegram_token'),
        telegram_chat_id=args.telegram_chat_id or config.get('telegram_chat_id'),
        min_ev=args.min_ev or config.get('min_ev', 8.0),
        min_confidence=args.min_confidence or config.get('min_confidence', 70.0),
        update_interval=args.update_interval or config.get('update_interval', 300)
    )
    
    # Avvia (single_run per cron jobs)
    automation.start(single_run=args.single_run)


if __name__ == '__main__':
    main()

