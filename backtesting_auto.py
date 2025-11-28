"""
Backtesting Automatico 24/7
============================

Sistema di backtesting automatico che:
- Raccoglie dati storici automaticamente
- Testa strategie su dati passati
- Ottimizza parametri
- Genera report e grafici
- Si integra con il sistema 24/7
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import json

logger = logging.getLogger(__name__)


class BacktestingAuto:
    """
    Sistema di backtesting automatico per strategie betting.
    """
    
    def __init__(self, db_path: str = "backtesting.db"):
        """
        Args:
            db_path: Path al database SQLite per storage dati storici
        """
        self.db_path = db_path
        self._init_database()
        logger.info("✅ Backtesting Auto initialized")
    
    def _init_database(self):
        """Inizializza database per storage dati storici"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabella per match storici
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_matches (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                league TEXT,
                home_score INTEGER,
                away_score INTEGER,
                odds_1 REAL,
                odds_x REAL,
                odds_2 REAL,
                odds_over_25 REAL,
                odds_under_25 REAL,
                odds_btts_yes REAL,
                odds_btts_no REAL,
                result TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabella per risultati backtest
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                total_bets INTEGER,
                winners INTEGER,
                losers INTEGER,
                win_rate REAL,
                total_stake REAL,
                total_profit REAL,
                roi REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                profit_factor REAL,
                parameters TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def collect_historical_data(self, api_manager=None, days: int = 90):
        """
        Raccoglie dati storici da API e li salva nel database.
        
        Args:
            api_manager: API Manager per ottenere dati
            days: Numero di giorni di storia da raccogliere
        """
        if not api_manager:
            logger.warning("⚠️  API Manager non disponibile per raccolta dati storici")
            return
        
        try:
            logger.info(f"📊 Raccogliendo dati storici per ultimi {days} giorni...")
            
            # Usa TheOddsAPI per ottenere match storici
            theodds_provider = api_manager.providers.get("theoddsapi")
            if not theodds_provider:
                logger.warning("⚠️  TheOddsAPI non disponibile")
                return
            
            # Raccogli dati per ogni giorno
            end_date = datetime.now()
            matches_collected = 0
            
            for i in range(days):
                target_date = end_date - timedelta(days=i)
                date_str = target_date.strftime("%Y-%m-%d")
                
                try:
                    # Ottieni match per questa data
                    matches = theodds_provider.get_matches(
                        sport="soccer",
                        date=date_str
                    )
                    
                    if matches:
                        self._save_matches_to_db(matches, date_str)
                        matches_collected += len(matches)
                        
                except Exception as e:
                    logger.debug(f"⚠️  Errore raccolta dati per {date_str}: {e}")
                    continue
            
            logger.info(f"✅ Raccolti {matches_collected} match storici")
            
        except Exception as e:
            logger.error(f"❌ Errore raccolta dati storici: {e}")
    
    def _save_matches_to_db(self, matches: List[Dict], date_str: str):
        """Salva match nel database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for match in matches:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO historical_matches 
                    (id, date, home_team, away_team, league, 
                     odds_1, odds_x, odds_2, odds_over_25, odds_under_25,
                     odds_btts_yes, odds_btts_no, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match.get('id', ''),
                    date_str,
                    match.get('home', ''),
                    match.get('away', ''),
                    match.get('league', ''),
                    match.get('odds_1'),
                    match.get('odds_x'),
                    match.get('odds_2'),
                    match.get('odds_over_25'),
                    match.get('odds_under_25'),
                    match.get('odds_btts_yes'),
                    match.get('odds_btts_no'),
                    match.get('result')  # W, D, L, O, U, etc.
                ))
            except Exception as e:
                logger.debug(f"⚠️  Errore salvataggio match: {e}")
        
        conn.commit()
        conn.close()
    
    def run_backtest(
        self,
        strategy_name: str,
        strategy_func: callable,
        start_date: str,
        end_date: str,
        min_ev: float = 8.0,
        min_confidence: float = 70.0,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.25
    ) -> Dict:
        """
        Esegue backtest di una strategia.
        
        Args:
            strategy_name: Nome della strategia
            strategy_func: Funzione che prende match data e ritorna decisione
            start_date: Data inizio (YYYY-MM-DD)
            end_date: Data fine (YYYY-MM-DD)
            min_ev: EV minimo per scommettere
            min_confidence: Confidence minima per scommettere
            initial_bankroll: Bankroll iniziale
            kelly_fraction: Frazione Kelly per calcolo stake
        
        Returns:
            Dict con risultati backtest
        """
        logger.info(f"🔄 Eseguendo backtest: {strategy_name} ({start_date} - {end_date})")
        
        # Carica dati storici
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM historical_matches
            WHERE date >= ? AND date <= ?
            ORDER BY date ASC
        """, conn, params=(start_date, end_date))
        conn.close()
        
        if df.empty:
            logger.warning("⚠️  Nessun dato storico disponibile per il periodo")
            return None
        
        # Esegui backtest
        bankroll = initial_bankroll
        bets = []
        cumulative_profit = []
        max_bankroll = bankroll
        max_drawdown = 0.0
        
        for _, match in df.iterrows():
            try:
                # Applica strategia
                decision = strategy_func(match.to_dict())
                
                if not decision or decision.get('action') != 'BET':
                    continue
                
                # Verifica soglie
                ev = decision.get('expected_value', 0)
                confidence = decision.get('confidence', 0)
                
                if ev < min_ev or confidence < min_confidence:
                    continue
                
                # Calcola stake (Kelly Criterion)
                prob_win = decision.get('probability', 0.5)
                odds = decision.get('odds', 2.0)
                kelly_pct = (prob_win * odds - 1) / (odds - 1) if odds > 1 else 0
                kelly_pct = max(0, min(kelly_pct, 0.25))  # Cap a 25%
                stake = bankroll * kelly_pct * kelly_fraction
                
                # Determina risultato
                result = match.get('result', '')
                market = decision.get('market', '1X2')
                won = self._check_bet_result(match, market, result)
                
                # Calcola profit
                if won:
                    profit = stake * (odds - 1)
                else:
                    profit = -stake
                
                bankroll += profit
                max_bankroll = max(max_bankroll, bankroll)
                drawdown = (max_bankroll - bankroll) / max_bankroll if max_bankroll > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
                
                bets.append({
                    'date': match['date'],
                    'match': f"{match['home_team']} vs {match['away_team']}",
                    'market': market,
                    'odds': odds,
                    'stake': stake,
                    'profit': profit,
                    'won': won,
                    'bankroll': bankroll
                })
                
                cumulative_profit.append({
                    'date': match['date'],
                    'profit': profit,
                    'cumulative': bankroll - initial_bankroll,
                    'bankroll': bankroll
                })
                
            except Exception as e:
                logger.debug(f"⚠️  Errore processando match: {e}")
                continue
        
        # Calcola metriche
        if not bets:
            logger.warning("⚠️  Nessuna scommessa eseguita nel backtest")
            return None
        
        df_bets = pd.DataFrame(bets)
        winners = df_bets['won'].sum()
        losers = (~df_bets['won']).sum()
        total_stake = df_bets['stake'].sum()
        total_profit = df_bets['profit'].sum()
        win_rate = (winners / len(bets)) * 100 if bets else 0
        roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
        
        # Sharpe Ratio (semplificato)
        if len(cumulative_profit) > 1:
            df_profit = pd.DataFrame(cumulative_profit)
            returns = df_profit['profit'].values
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Profit Factor
        winning_bets = df_bets[df_bets['profit'] > 0]['profit'].sum()
        losing_bets = abs(df_bets[df_bets['profit'] < 0]['profit'].sum())
        profit_factor = winning_bets / losing_bets if losing_bets > 0 else 0
        
        # Risultati
        results = {
            'strategy_name': strategy_name,
            'start_date': start_date,
            'end_date': end_date,
            'total_bets': len(bets),
            'winners': int(winners),
            'losers': int(losers),
            'win_rate': win_rate,
            'total_stake': total_stake,
            'total_profit': total_profit,
            'roi': roi,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'profit_factor': profit_factor,
            'final_bankroll': bankroll,
            'bankroll_change': ((bankroll - initial_bankroll) / initial_bankroll) * 100,
            'bets': bets,
            'cumulative_profit': cumulative_profit
        }
        
        # Salva risultati
        self._save_backtest_results(results, {
            'min_ev': min_ev,
            'min_confidence': min_confidence,
            'kelly_fraction': kelly_fraction
        })
        
        logger.info(f"✅ Backtest completato: {len(bets)} scommesse, ROI: {roi:.2f}%, Win Rate: {win_rate:.1f}%")
        
        return results
    
    def _check_bet_result(self, match: Dict, market: str, result: str) -> bool:
        """Verifica se una scommessa ha vinto"""
        home_score = match.get('home_score', 0)
        away_score = match.get('away_score', 0)
        
        if market == '1X2':
            if result == 'W':
                return home_score > away_score
            elif result == 'D':
                return home_score == away_score
            elif result == 'L':
                return away_score > home_score
        elif market == 'OVER_25':
            return (home_score + away_score) > 2.5
        elif market == 'UNDER_25':
            return (home_score + away_score) < 2.5
        elif market == 'BTTS_YES':
            return home_score > 0 and away_score > 0
        elif market == 'BTTS_NO':
            return home_score == 0 or away_score == 0
        
        return False
    
    def _save_backtest_results(self, results: Dict, parameters: Dict):
        """Salva risultati backtest nel database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO backtest_results
            (strategy_name, start_date, end_date, total_bets, winners, losers,
             win_rate, total_stake, total_profit, roi, sharpe_ratio, max_drawdown,
             profit_factor, parameters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            results['strategy_name'],
            results['start_date'],
            results['end_date'],
            results['total_bets'],
            results['winners'],
            results['losers'],
            results['win_rate'],
            results['total_stake'],
            results['total_profit'],
            results['roi'],
            results['sharpe_ratio'],
            results['max_drawdown'],
            results['profit_factor'],
            json.dumps(parameters)
        ))
        
        conn.commit()
        conn.close()
    
    def optimize_parameters(
        self,
        strategy_func: callable,
        start_date: str,
        end_date: str,
        param_grid: Dict[str, List]
    ) -> Dict:
        """
        Ottimizza parametri di una strategia usando grid search.
        
        Args:
            strategy_func: Funzione strategia
            start_date: Data inizio
            end_date: Data fine
            param_grid: Dict con parametri da testare
                Es: {'min_ev': [5, 8, 10], 'min_confidence': [60, 70, 80]}
        
        Returns:
            Migliori parametri trovati
        """
        logger.info("🔍 Ottimizzando parametri...")
        
        best_params = None
        best_roi = -float('inf')
        best_results = None
        
        # Genera tutte le combinazioni
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Esegui backtest con questi parametri
            results = self.run_backtest(
                strategy_name="optimization",
                strategy_func=strategy_func,
                start_date=start_date,
                end_date=end_date,
                **params
            )
            
            if results and results['roi'] > best_roi:
                best_roi = results['roi']
                best_params = params
                best_results = results
        
        logger.info(f"✅ Parametri ottimali trovati: {best_params} (ROI: {best_roi:.2f}%)")
        
        return {
            'best_params': best_params,
            'best_roi': best_roi,
            'results': best_results
        }
    
    def get_backtest_history(self, limit: int = 10) -> List[Dict]:
        """Ottiene storico backtest eseguiti"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM backtest_results
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

