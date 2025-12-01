#!/usr/bin/env python3
"""
Sistema Tracking Risultati Scommesse
=====================================

Traccia se le scommesse consigliate sono state vincenti.
Calcola ROI reale e statistiche performance.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)


class BettingResultsTracker:
    """Traccia risultati scommesse e calcola performance"""
    
    def __init__(self, db_path: str = "betting_results.db"):
        # Use /data persistent disk on Render if available
        import os
        if os.path.exists('/data') and os.path.isdir('/data'):
            db_path = os.path.join('/data', os.path.basename(db_path))
            logger.info(f"📁 Using persistent disk: {db_path}")

        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inizializza database SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabella opportunità notificate
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                league TEXT,
                match_date TIMESTAMP NOT NULL,
                market TEXT NOT NULL,
                recommended_stake REAL,
                odds REAL NOT NULL,
                expected_value REAL,
                confidence REAL,
                notified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result TEXT,
                is_winner INTEGER,
                profit_loss REAL,
                updated_at TIMESTAMP
            )
        """)
        
        # Tabella statistiche giornaliere
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date DATE PRIMARY KEY,
                opportunities_count INTEGER DEFAULT 0,
                winners_count INTEGER DEFAULT 0,
                losers_count INTEGER DEFAULT 0,
                pending_count INTEGER DEFAULT 0,
                total_stake REAL DEFAULT 0,
                total_profit_loss REAL DEFAULT 0,
                roi REAL DEFAULT 0,
                win_rate REAL DEFAULT 0
            )
        """)
        
        # Indici per performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_match_id ON opportunities(match_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_match_date ON opportunities(match_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_result ON opportunities(result)")
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Database inizializzato: {self.db_path}")
    
    def save_opportunity(self, opportunity: Dict[str, Any]) -> int:
        """
        Salva opportunità notificata
        
        Args:
            opportunity: Dict con dati opportunità
            
        Returns:
            ID opportunità salvata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        match_data = opportunity.get('match_data', {})
        ai_result = opportunity.get('ai_result', {})
        
        # Estrai dati
        match_id = opportunity.get('match_id', 'unknown')
        home = match_data.get('home', '')
        away = match_data.get('away', '')
        league = match_data.get('league', '')
        match_date = opportunity.get('timestamp', datetime.now())
        
        # Estrai raccomandazione
        final_decision = ai_result.get('final_decision', {})
        market = final_decision.get('market', '')
        stake = final_decision.get('stake', 0)
        odds = final_decision.get('odds', 0)
        ev = ai_result.get('summary', {}).get('expected_value', 0)
        confidence = ai_result.get('summary', {}).get('confidence', 0)
        
        cursor.execute("""
            INSERT INTO opportunities 
            (match_id, home_team, away_team, league, match_date, market, 
             recommended_stake, odds, expected_value, confidence, notified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (match_id, home, away, league, match_date, market, 
              stake, odds, ev, confidence, datetime.now()))
        
        opp_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Opportunità salvata: {home} vs {away} (ID: {opp_id})")
        return opp_id
    
    def update_result(self, match_id: str, result: str, home_score: int, away_score: int, market: str = None):
        """
        Aggiorna risultato partita
        
        Args:
            match_id: ID partita
            result: Risultato (W, L, P per pending)
            home_score: Gol casa
            away_score: Gol trasferta
            market: Market specifico (opzionale)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trova opportunità per questa partita
        query = "SELECT id, market, odds, recommended_stake FROM opportunities WHERE match_id = ?"
        params = [match_id]
        
        if market:
            query += " AND market = ?"
            params.append(market)
        
        cursor.execute(query, params)
        opportunities = cursor.fetchall()
        
        for opp_id, opp_market, odds, stake in opportunities:
            # Determina se vinta
            is_winner = self._check_winner(result, opp_market, home_score, away_score)
            
            # Calcola profit/loss
            if result == 'P':  # Pending
                profit_loss = None
            elif is_winner:
                profit_loss = (odds - 1) * stake  # Profitto
            else:
                profit_loss = -stake  # Perdita
            
            # Aggiorna
            cursor.execute("""
                UPDATE opportunities 
                SET result = ?, is_winner = ?, profit_loss = ?, updated_at = ?
                WHERE id = ?
            """, (result, is_winner, profit_loss, datetime.now(), opp_id))
        
        conn.commit()
        conn.close()
        
        # Aggiorna statistiche giornaliere
        self._update_daily_stats()
        
        logger.info(f"✅ Risultato aggiornato per match {match_id}: {result}")
    
    def _check_winner(self, result: str, market: str, home_score: int, away_score: int) -> bool:
        """Verifica se scommessa è vincente"""
        if result == 'P':  # Pending
            return None
        
        market_upper = market.upper()
        
        # Market 1X2
        if 'HOME' in market_upper or '1' in market_upper:
            return home_score > away_score
        elif 'AWAY' in market_upper or '2' in market_upper:
            return away_score > home_score
        elif 'DRAW' in market_upper or 'X' in market_upper:
            return home_score == away_score
        
        # Market Over/Under
        if 'OVER' in market_upper:
            total = home_score + away_score
            threshold = self._extract_number_from_market(market)
            return total > threshold if threshold else False
        elif 'UNDER' in market_upper:
            total = home_score + away_score
            threshold = self._extract_number_from_market(market)
            return total < threshold if threshold else False
        
        return False
    
    def _extract_number_from_market(self, market: str) -> Optional[float]:
        """Estrae numero da market (es: OVER_2.5 -> 2.5)"""
        import re
        match = re.search(r'(\d+\.?\d*)', market)
        return float(match.group(1)) if match else None
    
    def _update_daily_stats(self):
        """Aggiorna statistiche giornaliere"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        # Calcola statistiche per oggi
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losers,
                SUM(CASE WHEN result = 'P' OR result IS NULL THEN 1 ELSE 0 END) as pending,
                SUM(recommended_stake) as total_stake,
                SUM(COALESCE(profit_loss, 0)) as total_pl
            FROM opportunities
            WHERE DATE(notified_at) = ?
        """, (today,))
        
        row = cursor.fetchone()
        if row:
            total, winners, losers, pending, total_stake, total_pl = row
            
            roi = (total_pl / total_stake * 100) if total_stake > 0 else 0
            win_rate = (winners / (winners + losers) * 100) if (winners + losers) > 0 else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO daily_stats
                (date, opportunities_count, winners_count, losers_count, pending_count,
                 total_stake, total_profit_loss, roi, win_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (today, total or 0, winners or 0, losers or 0, pending or 0,
                  total_stake or 0, total_pl or 0, roi, win_rate))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Ottiene statistiche performance
        
        Args:
            days: Numero di giorni da analizzare
            
        Returns:
            Dict con statistiche
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Statistiche generali
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losers,
                SUM(CASE WHEN result = 'P' OR result IS NULL THEN 1 ELSE 0 END) as pending,
                SUM(recommended_stake) as total_stake,
                SUM(COALESCE(profit_loss, 0)) as total_pl,
                AVG(COALESCE(profit_loss, 0)) as avg_pl
            FROM opportunities
            WHERE notified_at >= ?
        """, (start_date,))
        
        row = cursor.fetchone()
        total, winners, losers, pending, total_stake, total_pl, avg_pl = row or (0, 0, 0, 0, 0, 0, 0)
        
        # ROI e win rate
        roi = (total_pl / total_stake * 100) if total_stake > 0 else 0
        win_rate = (winners / (winners + losers) * 100) if (winners + losers) > 0 else 0
        
        # Per market
        cursor.execute("""
            SELECT market, 
                   COUNT(*) as count,
                   SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                   SUM(COALESCE(profit_loss, 0)) as total_pl
            FROM opportunities
            WHERE notified_at >= ?
            GROUP BY market
        """, (start_date,))
        
        by_market = {}
        for market, count, market_winners, market_pl in cursor.fetchall():
            by_market[market] = {
                'count': count,
                'winners': market_winners,
                'win_rate': (market_winners / count * 100) if count > 0 else 0,
                'profit_loss': market_pl
            }
        
        # Per lega
        cursor.execute("""
            SELECT league,
                   COUNT(*) as count,
                   SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as winners,
                   SUM(COALESCE(profit_loss, 0)) as total_pl
            FROM opportunities
            WHERE notified_at >= ? AND league IS NOT NULL
            GROUP BY league
        """, (start_date,))
        
        by_league = {}
        for league, count, league_winners, league_pl in cursor.fetchall():
            by_league[league] = {
                'count': count,
                'winners': league_winners,
                'win_rate': (league_winners / count * 100) if count > 0 else 0,
                'profit_loss': league_pl
            }
        
        conn.close()
        
        return {
            'period_days': days,
            'total_opportunities': total or 0,
            'winners': winners or 0,
            'losers': losers or 0,
            'pending': pending or 0,
            'total_stake': total_stake or 0,
            'total_profit_loss': total_pl or 0,
            'average_profit_loss': avg_pl or 0,
            'roi_percent': roi,
            'win_rate_percent': win_rate,
            'by_market': by_market,
            'by_league': by_league
        }
    
    def get_recent_opportunities(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Ottiene opportunità recenti"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM opportunities
            ORDER BY notified_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

