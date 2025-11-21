"""
Signal Quality Gate - Sistema di Apprendimento
===============================================

Sistema di apprendimento automatico per il Signal Quality Gate che:
1. Traccia segnali inviati/bloccati con Quality Score
2. Confronta con risultati finali delle partite
3. Aggiorna automaticamente:
   - Pesi del Quality Score (35% contesto, 25% dati, ecc.)
   - Soglie dei filtri (minuti limite, differenze gol, ecc.)
   - Min quality score threshold (75.0)

Funziona sia H24 (aggiornamento continuo) che batch (aggiornamento periodico).
"""

import logging
import sqlite3
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """Record di un segnale valutato"""
    match_id: str
    market: str
    minute: int
    score_home: int
    score_away: int
    quality_score: float
    context_score: float
    data_quality_score: float
    logic_score: float
    timing_score: float
    was_approved: bool
    was_blocked: bool
    block_reasons: List[str]
    confidence: float
    ev: float
    timestamp: datetime
    # Risultato finale (compilato dopo)
    final_score_home: Optional[int] = None
    final_score_away: Optional[int] = None
    was_correct: Optional[bool] = None
    outcome: Optional[str] = None  # "win", "loss", "void"


class SignalQualityLearner:
    """
    Sistema di apprendimento per Signal Quality Gate.
    
    Meccanismo:
    1. Traccia ogni segnale valutato (inviato o bloccato)
    2. Quando la partita finisce, confronta risultato con previsione
    3. Calcola metriche (precision, recall, accuracy)
    4. Aggiorna automaticamente parametri per migliorare performance
    """
    
    def __init__(self, db_path: str = "signal_quality_learning.db"):
        self.db_path = db_path
        self._init_database()
        
        # Parametri attuali (verranno aggiornati dall'apprendimento)
        self.current_weights = {
            'context': 0.35,
            'data_quality': 0.25,
            'logic': 0.25,
            'timing': 0.15
        }
        self.min_quality_score = 75.0
        
        # Carica parametri salvati
        self._load_learned_parameters()
    
    def _init_database(self):
        """Inizializza database per tracciare segnali"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabella segnali valutati
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                market TEXT NOT NULL,
                minute INTEGER,
                score_home INTEGER,
                score_away INTEGER,
                quality_score REAL,
                context_score REAL,
                data_quality_score REAL,
                logic_score REAL,
                timing_score REAL,
                was_approved INTEGER,
                was_blocked INTEGER,
                block_reasons TEXT,
                confidence REAL,
                ev REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                final_score_home INTEGER,
                final_score_away INTEGER,
                was_correct INTEGER,
                outcome TEXT
            )
        """)
        
        # Tabella parametri appresi
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parameter_name TEXT UNIQUE NOT NULL,
                parameter_value REAL,
                metadata TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabella metriche performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                total_signals INTEGER,
                approved_signals INTEGER,
                blocked_signals INTEGER,
                correct_predictions INTEGER,
                wrong_predictions INTEGER,
                precision REAL,
                recall REAL,
                accuracy REAL,
                avg_quality_score REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indici
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_match ON signal_records(match_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_market ON signal_records(market)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_approved ON signal_records(was_approved)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_correct ON signal_records(was_correct)")
        
        conn.commit()
        conn.close()
        logger.info("✅ Signal Quality Learner database inizializzato")
    
    def record_signal(
        self,
        match_id: str,
        market: str,
        minute: int,
        score_home: int,
        score_away: int,
        quality_score: float,
        context_score: float,
        data_quality_score: float,
        logic_score: float,
        timing_score: float,
        was_approved: bool,
        block_reasons: List[str],
        confidence: float,
        ev: float
    ) -> int:
        """
        Registra un segnale valutato (inviato o bloccato)
        
        Returns:
            ID del record creato
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO signal_records (
                match_id, market, minute, score_home, score_away,
                quality_score, context_score, data_quality_score,
                logic_score, timing_score, was_approved, was_blocked,
                block_reasons, confidence, ev
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match_id, market, minute, score_home, score_away,
            quality_score, context_score, data_quality_score,
            logic_score, timing_score, 1 if was_approved else 0,
            1 if not was_approved else 0,
            json.dumps(block_reasons), confidence, ev
        ))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return record_id
    
    def update_signal_result(
        self,
        match_id: str,
        final_score_home: int,
        final_score_away: int,
        market: str = None,
        events: List[Dict[str, Any]] = None,
        statistics: Dict[str, Any] = None
    ) -> int:
        """
        Aggiorna risultato finale per segnali di una partita.
        
        Calcola se il segnale era corretto basandosi sul mercato.
        
        Returns:
            Numero di segnali aggiornati
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trova tutti i segnali per questa partita (non ancora aggiornati)
        query = "SELECT id, market, score_home, score_away, minute FROM signal_records WHERE match_id = ? AND final_score_home IS NULL"
        params = [match_id]
        
        if market:
            query += " AND market = ?"
            params.append(market)
        
        cursor.execute(query, params)
        signals = cursor.fetchall()
        
        updated_count = 0
        for signal_id, signal_market, signal_score_home, signal_score_away, signal_minute in signals:
            # 🆕 Determina se il segnale era corretto (con eventi e statistiche)
            was_correct, outcome = self._evaluate_signal_correctness(
                signal_market,
                signal_score_home,
                signal_score_away,
                final_score_home,
                final_score_away,
                signal_minute=signal_minute,
                events=events,
                statistics=statistics
            )
            
            # Aggiorna record (solo se was_correct non è None)
            if was_correct is not None:
                cursor.execute("""
                    UPDATE signal_records
                    SET final_score_home = ?,
                        final_score_away = ?,
                        was_correct = ?,
                        outcome = ?
                    WHERE id = ?
                """, (final_score_home, final_score_away, 1 if was_correct else 0, outcome, signal_id))
            else:
                # Se non possiamo valutare (mercato sconosciuto), aggiorna solo il risultato
                cursor.execute("""
                    UPDATE signal_records
                    SET final_score_home = ?,
                        final_score_away = ?,
                        outcome = ?
                    WHERE id = ?
                """, (final_score_home, final_score_away, outcome, signal_id))
            
            updated_count += 1
        
        conn.commit()
        conn.close()
        
        if updated_count > 0:
            logger.info(f"✅ Aggiornati {updated_count} segnali per partita {match_id}")
        
        return updated_count
    
    def _evaluate_signal_correctness(
        self,
        market: str,
        signal_score_home: int,
        signal_score_away: int,
        final_score_home: int,
        final_score_away: int,
        signal_minute: int = None,
        events: List[Dict[str, Any]] = None,
        statistics: Dict[str, Any] = None
    ) -> Tuple[Optional[bool], str]:
        """
        Valuta se un segnale era corretto basandosi sul mercato.
        
        Returns:
            (was_correct, outcome)
        """
        total_goals_final = final_score_home + final_score_away
        total_goals_signal = signal_score_home + signal_score_away
        
        # Over markets
        if 'over_0.5' in market:
            return (total_goals_final >= 1, "win" if total_goals_final >= 1 else "loss")
        elif 'over_1.5' in market:
            return (total_goals_final >= 2, "win" if total_goals_final >= 2 else "loss")
        elif 'over_2.5' in market:
            return (total_goals_final >= 3, "win" if total_goals_final >= 3 else "loss")
        elif 'over_3.5' in market:
            return (total_goals_final >= 4, "win" if total_goals_final >= 4 else "loss")
        elif 'over_4.5' in market:
            return (total_goals_final >= 5, "win" if total_goals_final >= 5 else "loss")
        
        # Under markets
        elif 'under_1.5' in market:
            return (total_goals_final <= 1, "win" if total_goals_final <= 1 else "loss")
        elif 'under_2.5' in market:
            return (total_goals_final <= 2, "win" if total_goals_final <= 2 else "loss")
        elif 'under_3.5' in market:
            return (total_goals_final <= 3, "win" if total_goals_final <= 3 else "loss")
        
        # BTTS
        elif 'btts_yes' in market:
            return (final_score_home > 0 and final_score_away > 0, "win" if (final_score_home > 0 and final_score_away > 0) else "loss")
        elif 'btts_no' in market:
            return (final_score_home == 0 or final_score_away == 0, "win" if (final_score_home == 0 or final_score_away == 0) else "loss")
        
        # 1X2
        elif '1x2_home' in market or 'home_win' in market:
            return (final_score_home > final_score_away, "win" if final_score_home > final_score_away else "loss")
        elif '1x2_away' in market or 'away_win' in market:
            return (final_score_away > final_score_home, "win" if final_score_away > final_score_home else "loss")
        elif '1x2_draw' in market or 'draw' in market:
            return (final_score_home == final_score_away, "win" if final_score_home == final_score_away else "loss")
        
        # Odd/Even
        elif 'total_goals_odd' in market:
            return (total_goals_final % 2 == 1, "win" if total_goals_final % 2 == 1 else "loss")
        elif 'total_goals_even' in market:
            return (total_goals_final % 2 == 0, "win" if total_goals_final % 2 == 0 else "loss")
        
        # 🆕 Next Goal markets (richiede eventi)
        elif 'next_goal' in market:
            if not events or signal_minute is None:
                return (None, "unknown")
            
            # Trova primo gol dopo il minuto del segnale
            next_goal = None
            for event in events:
                event_type = event.get("type", {})
                if isinstance(event_type, dict):
                    event_type_name = event_type.get("name", "")
                else:
                    event_type_name = str(event_type)
                
                if event_type_name == "Goal":
                    time_info = event.get("time", {})
                    if isinstance(time_info, dict):
                        goal_minute = time_info.get("elapsed")
                    else:
                        goal_minute = None
                    
                    # 🆕 Filtra solo gol nel tempo regolamentare e dopo il segnale
                    if goal_minute and goal_minute > signal_minute and goal_minute <= 90:
                        next_goal = event
                        break
            
            if not next_goal:
                # Nessun gol dopo il segnale
                return (False, "loss")
            
            # 🆕 Determina se è home o away usando team IDs (CORRETTO)
            team_info = next_goal.get("team", {})
            if isinstance(team_info, dict):
                event_team_id = team_info.get("id")
                # Usa team IDs salvati negli eventi
                home_team_id = next_goal.get("_home_team_id")
                away_team_id = next_goal.get("_away_team_id")
                
                if home_team_id and event_team_id == home_team_id:
                    is_home_goal = True
                elif away_team_id and event_team_id == away_team_id:
                    is_home_goal = False
                else:
                    # Fallback: usa nome squadra se disponibile
                    team_name = team_info.get("name", "").lower()
                    is_home_goal = "home" in team_name or event_team_id == home_team_id
            else:
                is_home_goal = False
            
            if 'next_goal_home' in market or 'next_goal_pressure_home' in market:
                return (is_home_goal, "win" if is_home_goal else "loss")
            elif 'next_goal_away' in market or 'next_goal_pressure_away' in market:
                return (not is_home_goal, "win" if not is_home_goal else "loss")
            elif 'next_goal_before_75' in market:
                time_info = next_goal.get("time", {})
                goal_minute = time_info.get("elapsed", 999) if isinstance(time_info, dict) else 999
                return (goal_minute <= 75, "win" if goal_minute <= 75 else "loss")
            elif 'next_goal_after_75' in market:
                time_info = next_goal.get("time", {})
                goal_minute = time_info.get("elapsed", 0) if isinstance(time_info, dict) else 0
                return (goal_minute > 75, "win" if goal_minute > 75 else "loss")
        
        # 🆕 Corner markets (richiede statistiche)
        elif 'corner' in market:
            if not statistics:
                return (None, "unknown")
            
            total_corners = statistics.get("total_corners", 0)
            if total_corners is None:
                return (None, "unknown")
            
            # Estrai threshold dal nome mercato (es. "corner_over_9.5" -> 9.5)
            threshold_match = re.search(r'(\d+\.?\d*)', market)
            if threshold_match:
                threshold = float(threshold_match.group(1))
                if 'over' in market:
                    return (total_corners > threshold, "win" if total_corners > threshold else "loss")
                elif 'under' in market:
                    return (total_corners < threshold, "win" if total_corners < threshold else "loss")
            else:
                logger.warning(f"⚠️  Impossibile estrarre threshold da mercato corner: {market}")
                return (None, "unknown")
        
        # 🆕 Cards markets (richiede statistiche)
        elif 'card' in market or 'cartellin' in market:
            if not statistics:
                return (None, "unknown")
            
            total_cards = statistics.get("total_cards", 0)
            if total_cards is None:
                return (None, "unknown")
            
            # Estrai threshold dal nome mercato
            threshold_match = re.search(r'(\d+\.?\d*)', market)
            if threshold_match:
                threshold = float(threshold_match.group(1))
                if 'over' in market:
                    return (total_cards > threshold, "win" if total_cards > threshold else "loss")
                elif 'under' in market:
                    return (total_cards < threshold, "win" if total_cards < threshold else "loss")
            else:
                logger.warning(f"⚠️  Impossibile estrarre threshold da mercato cards: {market}")
                return (None, "unknown")
        
        # 🆕 Team Goal Anytime (richiede eventi)
        elif 'team_goal_anytime' in market or 'goal_anytime' in market:
            if not events or signal_minute is None:
                return (None, "unknown")
            
            # Controlla se la squadra ha segnato dopo il minuto del segnale
            team_scored = False
            for event in events:
                event_type = event.get("type", {})
                if isinstance(event_type, dict):
                    event_type_name = event_type.get("name", "")
                else:
                    event_type_name = str(event_type)
                
                if event_type_name == "Goal":
                    time_info = event.get("time", {})
                    goal_minute = time_info.get("elapsed") if isinstance(time_info, dict) else None
                    
                    # 🆕 Filtra solo gol nel tempo regolamentare (escludi supplementari se necessario)
                    if goal_minute and goal_minute > signal_minute and goal_minute <= 90:
                        team_info = event.get("team", {})
                        if isinstance(team_info, dict):
                            event_team_id = team_info.get("id")
                            # Usa team IDs salvati negli eventi
                            home_team_id = event.get("_home_team_id")
                            away_team_id = event.get("_away_team_id")
                            
                            if home_team_id and event_team_id == home_team_id:
                                is_home_goal = True
                            elif away_team_id and event_team_id == away_team_id:
                                is_home_goal = False
                            else:
                                # Fallback
                                is_home_goal = False
                        else:
                            is_home_goal = False
                        
                        if 'home' in market and is_home_goal:
                            team_scored = True
                            break
                        elif 'away' in market and not is_home_goal:
                            team_scored = True
                            break
            
            return (team_scored, "win" if team_scored else "loss")
        
        # Default: non possiamo valutare
        return (None, "unknown")
    
    def learn_from_results(self, min_samples: int = 50) -> Dict[str, Any]:
        """
        Apprende dai risultati e aggiorna parametri.
        
        Meccanismo:
        1. Analizza segnali con risultati noti
        2. Calcola metriche (precision, recall, accuracy)
        3. Identifica pattern (es. contesto più importante di timing)
        4. Aggiorna pesi e soglie
        
        Args:
            min_samples: Numero minimo di campioni per apprendere
            
        Returns:
            Dict con metriche e aggiornamenti
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ottieni segnali con risultati
        cursor.execute("""
            SELECT 
                was_approved,
                was_correct,
                quality_score,
                context_score,
                data_quality_score,
                logic_score,
                timing_score
            FROM signal_records
            WHERE was_correct IS NOT NULL
        """)
        
        signals = cursor.fetchall()
        
        if len(signals) < min_samples:
            conn.close()
            logger.info(f"⚠️  Campioni insufficienti per apprendere ({len(signals)} < {min_samples})")
            return {
                'status': 'insufficient_samples',
                'samples': len(signals),
                'min_samples': min_samples
            }
        
        # Analizza performance
        approved_correct = sum(1 for s in signals if s[0] == 1 and s[1] == 1)
        approved_wrong = sum(1 for s in signals if s[0] == 1 and s[1] == 0)
        blocked_correct = sum(1 for s in signals if s[0] == 0 and s[1] == 1)  # Falsi negativi
        blocked_wrong = sum(1 for s in signals if s[0] == 0 and s[1] == 0)  # Buoni blocchi
        
        total_approved = approved_correct + approved_wrong
        total_blocked = blocked_correct + blocked_wrong
        
        # Calcola metriche
        precision = approved_correct / total_approved if total_approved > 0 else 0
        recall = approved_correct / (approved_correct + blocked_correct) if (approved_correct + blocked_correct) > 0 else 0
        accuracy = (approved_correct + blocked_wrong) / len(signals) if len(signals) > 0 else 0
        
        # Analizza importanza dei componenti
        # Se segnali corretti hanno context_score alto, aumenta peso contesto
        correct_signals = [s for s in signals if s[1] == 1]
        wrong_signals = [s for s in signals if s[1] == 0]
        
        if len(correct_signals) > 0 and len(wrong_signals) > 0:
            # Calcola differenza media tra corretti e sbagliati per ogni componente
            avg_context_correct = sum(s[3] for s in correct_signals) / len(correct_signals)
            avg_context_wrong = sum(s[3] for s in wrong_signals) / len(wrong_signals)
            context_importance = max(0, avg_context_correct - avg_context_wrong) / 100.0
            
            avg_data_correct = sum(s[4] for s in correct_signals) / len(correct_signals)
            avg_data_wrong = sum(s[4] for s in wrong_signals) / len(wrong_signals)
            data_importance = max(0, avg_data_correct - avg_data_wrong) / 100.0
            
            avg_logic_correct = sum(s[5] for s in correct_signals) / len(correct_signals)
            avg_logic_wrong = sum(s[5] for s in wrong_signals) / len(wrong_signals)
            logic_importance = max(0, avg_logic_correct - avg_logic_wrong) / 100.0
            
            avg_timing_correct = sum(s[6] for s in correct_signals) / len(correct_signals)
            avg_timing_wrong = sum(s[6] for s in wrong_signals) / len(wrong_signals)
            timing_importance = max(0, avg_timing_correct - avg_timing_wrong) / 100.0
            
            # Normalizza importanze
            total_importance = context_importance + data_importance + logic_importance + timing_importance
            if total_importance > 0:
                # Aggiorna pesi (media mobile con learning rate 0.1)
                learning_rate = 0.1
                self.current_weights['context'] = (
                    self.current_weights['context'] * (1 - learning_rate) +
                    (context_importance / total_importance) * learning_rate
                )
                self.current_weights['data_quality'] = (
                    self.current_weights['data_quality'] * (1 - learning_rate) +
                    (data_importance / total_importance) * learning_rate
                )
                self.current_weights['logic'] = (
                    self.current_weights['logic'] * (1 - learning_rate) +
                    (logic_importance / total_importance) * learning_rate
                )
                self.current_weights['timing'] = (
                    self.current_weights['timing'] * (1 - learning_rate) +
                    (timing_importance / total_importance) * learning_rate
                )
                
                # Normalizza pesi
                total_weight = sum(self.current_weights.values())
                for key in self.current_weights:
                    self.current_weights[key] /= total_weight
        
        # Aggiorna min_quality_score basandosi su precision
        # Se precision è alta, possiamo abbassare soglia (più segnali)
        # Se precision è bassa, alziamo soglia (meno segnali ma più precisi)
        if precision > 0.8 and recall < 0.5:
            # Troppi falsi negativi, abbassa soglia
            self.min_quality_score = max(70.0, self.min_quality_score - 1.0)
        elif precision < 0.6:
            # Troppi falsi positivi, alza soglia
            self.min_quality_score = min(85.0, self.min_quality_score + 1.0)
        
        # Salva parametri appresi
        self._save_learned_parameters()
        
        # Salva metriche
        cursor.execute("""
            INSERT INTO performance_metrics (
                date, total_signals, approved_signals, blocked_signals,
                correct_predictions, wrong_predictions,
                precision, recall, accuracy, avg_quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().date(),
            len(signals),
            total_approved,
            total_blocked,
            approved_correct,
            approved_wrong,
            precision,
            recall,
            accuracy,
            sum(s[2] for s in signals) / len(signals) if len(signals) > 0 else 0
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(
            f"✅ Apprendimento completato: "
            f"Precision={precision:.2%}, Recall={recall:.2%}, Accuracy={accuracy:.2%} | "
            f"Pesi: Context={self.current_weights['context']:.2%}, "
            f"Data={self.current_weights['data_quality']:.2%}, "
            f"Logic={self.current_weights['logic']:.2%}, "
            f"Timing={self.current_weights['timing']:.2%} | "
            f"Min Score={self.min_quality_score:.1f}"
        )
        
        return {
            'status': 'success',
            'samples': len(signals),
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'weights': self.current_weights.copy(),
            'min_quality_score': self.min_quality_score,
            'approved_correct': approved_correct,
            'approved_wrong': approved_wrong,
            'blocked_correct': blocked_correct,
            'blocked_wrong': blocked_wrong
        }
    
    def _save_learned_parameters(self):
        """Salva parametri appresi nel database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Salva pesi
        for param_name, param_value in self.current_weights.items():
            cursor.execute("""
                INSERT OR REPLACE INTO learned_parameters (parameter_name, parameter_value, metadata)
                VALUES (?, ?, ?)
            """, (f"weight_{param_name}", param_value, json.dumps({'updated_at': datetime.now().isoformat()})))
        
        # Salva min_quality_score
        cursor.execute("""
            INSERT OR REPLACE INTO learned_parameters (parameter_name, parameter_value, metadata)
            VALUES (?, ?, ?)
        """, ("min_quality_score", self.min_quality_score, json.dumps({'updated_at': datetime.now().isoformat()})))
        
        conn.commit()
        conn.close()
    
    def _load_learned_parameters(self):
        """Carica parametri appresi dal database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT parameter_name, parameter_value FROM learned_parameters")
        params = cursor.fetchall()
        
        for param_name, param_value in params:
            if param_name.startswith("weight_"):
                key = param_name.replace("weight_", "")
                if key in self.current_weights:
                    self.current_weights[key] = param_value
            elif param_name == "min_quality_score":
                self.min_quality_score = param_value
        
        conn.close()
        
        if params:
            logger.info(f"✅ Caricati parametri appresi: {len(params)} parametri")
    
    def get_learned_weights(self) -> Dict[str, float]:
        """Restituisce pesi appresi"""
        return self.current_weights.copy()
    
    def get_learned_min_score(self) -> float:
        """Restituisce min quality score appreso"""
        return self.min_quality_score

