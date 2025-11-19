#!/usr/bin/env python3
"""
Sistema Tracking Performance Live Betting
==========================================

Estende BettingResultsTracker per tracciare performance specifiche del live betting:
- Win rate per mercato
- Performance per minuto
- Soglie dinamiche basate su risultati storici
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class LiveBettingPerformanceTracker:
    """Traccia performance live betting e calcola soglie dinamiche"""
    
    def __init__(self, db_path: str = "betting_results.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inizializza tabelle aggiuntive per live betting"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabella opportunità live
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                league TEXT,
                market TEXT NOT NULL,
                minute INTEGER,
                score_home INTEGER,
                score_away INTEGER,
                odds REAL NOT NULL,
                confidence REAL NOT NULL,
                expected_value REAL,
                notified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result TEXT,
                is_winner INTEGER,
                profit_loss REAL,
                updated_at TIMESTAMP
            )
        """)
        
        # Tabella statistiche per mercato
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_performance (
                market TEXT PRIMARY KEY,
                total_count INTEGER DEFAULT 0,
                winners_count INTEGER DEFAULT 0,
                losers_count INTEGER DEFAULT 0,
                pending_count INTEGER DEFAULT 0,
                total_profit_loss REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                avg_ev REAL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabella soglie dinamiche
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_thresholds (
                market TEXT PRIMARY KEY,
                min_confidence REAL DEFAULT 75.0,
                min_ev REAL DEFAULT 10.0,
                adjustment_reason TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indici
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_market ON live_opportunities(market)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_minute ON live_opportunities(minute)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_result ON live_opportunities(result)")
        
        conn.commit()
        conn.close()
        logger.info("✅ Live betting performance tracker inizializzato")
    
    def save_live_opportunity(self, opportunity: Any, match_data: Dict) -> int:
        """
        Salva opportunità live notificata
        
        Args:
            opportunity: LiveBettingOpportunity object
            match_data: Dict con dati partita
            
        Returns:
            ID opportunità salvata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Estrai dati
        match_id = opportunity.match_id
        home = match_data.get('home', '')
        away = match_data.get('away', '')
        league = match_data.get('league', '')
        market = opportunity.market
        minute = 0
        score_home = 0
        score_away = 0
        
        if opportunity.match_stats:
            minute = opportunity.match_stats.get('minute', 0)
            score_home = opportunity.match_stats.get('score_home', 0)
            score_away = opportunity.match_stats.get('score_away', 0)
        
        odds = opportunity.odds or 0
        confidence = opportunity.confidence or 0
        # 🔧 FIX: LiveBettingOpportunity usa 'ev' non 'expected_value'
        ev = getattr(opportunity, 'ev', None) or getattr(opportunity, 'expected_value', None) or 0
        
        cursor.execute("""
            INSERT INTO live_opportunities 
            (match_id, home_team, away_team, league, market, minute, 
             score_home, score_away, odds, confidence, expected_value, notified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (match_id, home, away, league, market, minute, 
              score_home, score_away, odds, confidence, ev, datetime.now()))
        
        opp_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.debug(f"✅ Live opportunity salvata: {market} (ID: {opp_id})")
        return opp_id
    
    def update_live_result(self, match_id: str, final_score_home: int, final_score_away: int):
        """
        Aggiorna risultato partita live
        
        Args:
            match_id: ID partita
            final_score_home: Gol finali casa
            final_score_away: Gol finali trasferta
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trova tutte le opportunità live per questa partita
        cursor.execute("""
            SELECT id, market FROM live_opportunities 
            WHERE match_id = ? AND result IS NULL
        """, (match_id,))
        
        opportunities = cursor.fetchall()
        
        for opp_id, market in opportunities:
            is_winner = self._check_live_winner(market, final_score_home, final_score_away)
            
            # Calcola profit/loss (stima basata su odds)
            cursor.execute("SELECT odds FROM live_opportunities WHERE id = ?", (opp_id,))
            row = cursor.fetchone()
            odds = row[0] if row else 0
            
            if is_winner:
                result = 'W'
                # Stima profit (assumendo stake di 1 unità)
                profit_loss = odds - 1
            else:
                result = 'L'
                profit_loss = -1  # Perdita stake
        
            cursor.execute("""
                UPDATE live_opportunities 
                SET result = ?, is_winner = ?, profit_loss = ?, updated_at = ?
                WHERE id = ?
            """, (result, 1 if is_winner else 0, profit_loss, datetime.now(), opp_id))
        
        conn.commit()
        conn.close()
        
        # Aggiorna statistiche mercati
        self._update_market_performance()
        
        logger.info(f"✅ Risultati live aggiornati per match {match_id}")
    
    def _check_live_winner(self, market: str, home_score: int, away_score: int) -> bool:
        """Verifica se scommessa live è vincente"""
        market_lower = market.lower()
        total_goals = home_score + away_score
        
        # Over/Under
        if 'over' in market_lower:
            if '0.5' in market_lower:
                return total_goals > 0
            elif '1.5' in market_lower:
                return total_goals > 1
            elif '2.5' in market_lower:
                return total_goals > 2
            elif '3.5' in market_lower:
                return total_goals > 3
        elif 'under' in market_lower:
            if '0.5' in market_lower:
                return total_goals < 1
            elif '1.5' in market_lower:
                return total_goals < 2
            elif '2.5' in market_lower:
                return total_goals < 3
            elif '3.5' in market_lower:
                return total_goals < 4
        
        # BTTS
        if 'btts' in market_lower:
            if 'yes' in market_lower or 'si' in market_lower:
                return home_score > 0 and away_score > 0
            elif 'no' in market_lower:
                return home_score == 0 or away_score == 0
        
        # Clean Sheet
        if 'clean_sheet' in market_lower:
            if 'home' in market_lower:
                return away_score == 0
            elif 'away' in market_lower:
                return home_score == 0
        
        # Win to Nil
        if 'win_to_nil' in market_lower:
            if 'home' in market_lower:
                return home_score > away_score and away_score == 0
            elif 'away' in market_lower:
                return away_score > home_score and home_score == 0
        
        # Match Winner
        if 'home' in market_lower and ('win' in market_lower or '1x2_home' in market_lower):
            return home_score > away_score
        elif 'away' in market_lower and ('win' in market_lower or '1x2_away' in market_lower):
            return away_score > home_score
        
        return False
    
    def _update_market_performance(self):
        """Aggiorna statistiche performance per mercato"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calcola statistiche per ogni mercato
        cursor.execute("""
            SELECT 
                market,
                COUNT(*) as total,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losers,
                SUM(CASE WHEN result IS NULL THEN 1 ELSE 0 END) as pending,
                SUM(COALESCE(profit_loss, 0)) as total_pl,
                AVG(confidence) as avg_conf,
                AVG(expected_value) as avg_ev
            FROM live_opportunities
            WHERE notified_at >= datetime('now', '-30 days')
            GROUP BY market
        """)
        
        for row in cursor.fetchall():
            market, total, winners, losers, pending, total_pl, avg_conf, avg_ev = row
            
            win_rate = (winners / (winners + losers) * 100) if (winners + losers) > 0 else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO market_performance
                (market, total_count, winners_count, losers_count, pending_count,
                 total_profit_loss, win_rate, avg_confidence, avg_ev, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (market, total, winners, losers, pending, total_pl, win_rate, 
                  avg_conf or 0, avg_ev or 0, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_market_performance(self, market: str, days: int = 30) -> Optional[Dict[str, Any]]:
        """
        Ottiene performance per un mercato specifico
        
        Args:
            market: Nome mercato
            days: Giorni da analizzare
            
        Returns:
            Dict con statistiche o None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losers,
                SUM(COALESCE(profit_loss, 0)) as total_pl,
                AVG(confidence) as avg_conf,
                AVG(expected_value) as avg_ev
            FROM live_opportunities
            WHERE market = ? AND notified_at >= ?
        """, (market, start_date))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row or row[0] == 0:
            return None
        
        total, winners, losers, total_pl, avg_conf, avg_ev = row
        win_rate = (winners / (winners + losers) * 100) if (winners + losers) > 0 else 0
        
        return {
            'market': market,
            'total': total,
            'winners': winners or 0,
            'losers': losers or 0,
            'win_rate': win_rate,
            'total_profit_loss': total_pl or 0,
            'avg_confidence': avg_conf or 0,
            'avg_ev': avg_ev or 0
        }
    
    def calculate_dynamic_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Calcola soglie dinamiche basate su performance storiche
        
        Returns:
            Dict con soglie per mercato: {market: {min_confidence: float, min_ev: float}}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ottieni performance per tutti i mercati
        cursor.execute("""
            SELECT market, win_rate, total_count, avg_confidence, avg_ev
            FROM market_performance
            WHERE total_count >= 5  -- Minimo 5 scommesse per considerare
        """)
        
        thresholds = {}
        
        for market, win_rate, total_count, avg_conf, avg_ev in cursor.fetchall():
            # Soglie base
            base_confidence = 75.0
            base_ev = 10.0
            
            # Aggiustamento basato su win rate
            if win_rate < 50:
                # Win rate basso: aumenta soglie
                confidence_adjustment = (50 - win_rate) * 0.5  # +0.5% per ogni % sotto 50
                ev_adjustment = (50 - win_rate) * 0.3  # +0.3% per ogni % sotto 50
                min_confidence = base_confidence + confidence_adjustment
                min_ev = base_ev + ev_adjustment
                reason = f"Win rate basso ({win_rate:.1f}%)"
            elif win_rate > 65:
                # Win rate alto: abbassa leggermente soglie
                confidence_adjustment = (win_rate - 65) * 0.3  # -0.3% per ogni % sopra 65
                ev_adjustment = (win_rate - 65) * 0.2  # -0.2% per ogni % sopra 65
                min_confidence = max(70.0, base_confidence - confidence_adjustment)
                min_ev = max(8.0, base_ev - ev_adjustment)
                reason = f"Win rate alto ({win_rate:.1f}%)"
            else:
                # Win rate normale: mantieni soglie base
                min_confidence = base_confidence
                min_ev = base_ev
                reason = f"Win rate normale ({win_rate:.1f}%)"
            
            # Limiti minimi e massimi
            min_confidence = max(70.0, min(85.0, min_confidence))
            min_ev = max(8.0, min(20.0, min_ev))
            
            thresholds[market] = {
                'min_confidence': round(min_confidence, 1),
                'min_ev': round(min_ev, 1),
                'reason': reason,
                'win_rate': win_rate,
                'total_count': total_count
            }
            
            # Salva nel database
            cursor.execute("""
                INSERT OR REPLACE INTO dynamic_thresholds
                (market, min_confidence, min_ev, adjustment_reason, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (market, min_confidence, min_ev, reason, datetime.now()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Soglie dinamiche calcolate per {len(thresholds)} mercati")
        return thresholds
    
    def get_dynamic_threshold(self, market: str) -> Dict[str, float]:
        """
        Ottiene soglia dinamica per un mercato
        
        Args:
            market: Nome mercato
            
        Returns:
            Dict con min_confidence e min_ev, o valori default
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT min_confidence, min_ev FROM dynamic_thresholds
            WHERE market = ?
        """, (market,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'min_confidence': row[0],
                'min_ev': row[1]
            }
        
        # Valori default se non trovato
        return {
            'min_confidence': 75.0,
            'min_ev': 10.0
        }
    
    def get_all_market_performance(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """Ottiene performance per tutti i mercati"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT 
                market,
                COUNT(*) as total,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losers,
                SUM(COALESCE(profit_loss, 0)) as total_pl,
                AVG(confidence) as avg_conf,
                AVG(expected_value) as avg_ev
            FROM live_opportunities
            WHERE notified_at >= ?
            GROUP BY market
            ORDER BY total DESC
        """, (start_date,))
        
        results = {}
        for row in cursor.fetchall():
            market, total, winners, losers, total_pl, avg_conf, avg_ev = row
            win_rate = (winners / (winners + losers) * 100) if (winners + losers) > 0 else 0
            
            results[market] = {
                'total': total,
                'winners': winners or 0,
                'losers': losers or 0,
                'win_rate': win_rate,
                'total_profit_loss': total_pl or 0,
                'avg_confidence': avg_conf or 0,
                'avg_ev': avg_ev or 0
            }
        
        conn.close()
        return results

