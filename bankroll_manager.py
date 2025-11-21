#!/usr/bin/env python3
"""
Gestione Portfolio/Bankroll
============================

Traccia bankroll e gestisce stake.
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BankrollManager:
    """Gestisce bankroll e calcolo stake"""
    
    def __init__(self, db_path: str = "bankroll.db", initial_bankroll: float = 1000.0):
        # Use /data persistent disk on Render if available
        import os
        if os.path.exists('/data') and os.path.isdir('/data'):
            db_path = os.path.join('/data', os.path.basename(db_path))
            logger.info(f"📁 Using persistent disk: {db_path}")

        self.db_path = db_path
        self.initial_bankroll = initial_bankroll
        self._init_database()
    
    def _init_database(self):
        """Inizializza database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bankroll_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                amount REAL NOT NULL,
                change_amount REAL,
                change_reason TEXT,
                note TEXT
            )
        """)
        
        # Inserisci bankroll iniziale se non esiste
        cursor.execute("SELECT COUNT(*) FROM bankroll_history")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO bankroll_history (amount, change_amount, change_reason, note)
                VALUES (?, ?, ?, ?)
            """, (self.initial_bankroll, 0, "INITIAL", "Bankroll iniziale"))
        
        conn.commit()
        conn.close()
    
    def get_current_bankroll(self) -> float:
        """Ottiene bankroll attuale"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT amount FROM bankroll_history ORDER BY date DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else self.initial_bankroll
    
    def update_bankroll(self, change_amount: float, reason: str, note: str = ""):
        """Aggiorna bankroll"""
        current = self.get_current_bankroll()
        new_amount = current + change_amount
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO bankroll_history (amount, change_amount, change_reason, note)
            VALUES (?, ?, ?, ?)
        """, (new_amount, change_amount, reason, note))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Bankroll aggiornato: €{current:.2f} → €{new_amount:.2f} ({reason})")
    
    def calculate_stake(self, bankroll: float, kelly_fraction: float, ev: float, odds: float) -> float:
        """
        Calcola stake ottimale usando Kelly Criterion
        
        Args:
            bankroll: Bankroll attuale
            kelly_fraction: Frazione Kelly da usare (es: 0.25 = 25% di Kelly)
            ev: Expected Value (in percentuale, es: 8.0 per 8%)
            odds: Quote scommessa
            
        Returns:
            Stake consigliato
        """
        if odds <= 1.0 or ev <= 0:
            return 0.0
        
        # Probabilità implicita
        implied_prob = 1.0 / odds
        
        # Probabilità reale (da EV)
        # EV = (prob * odds - 1) * stake
        # prob = (EV/stake + 1) / odds
        # Approssimiamo: prob ≈ implied_prob * (1 + EV/100)
        real_prob = implied_prob * (1 + ev / 100)
        
        # Kelly: f = (p * odds - 1) / (odds - 1)
        kelly = (real_prob * odds - 1) / (odds - 1)
        
        # Applica frazione Kelly e calcola stake
        stake_fraction = kelly * kelly_fraction
        stake = bankroll * stake_fraction
        
        # Limiti di sicurezza
        max_stake_percent = 0.10  # Max 10% del bankroll
        min_stake = 1.0  # Minimo €1
        max_stake = bankroll * max_stake_percent
        
        stake = max(min_stake, min(stake, max_stake))
        
        return round(stake, 2)
    
    def get_statistics(self) -> Dict:
        """Ottiene statistiche bankroll"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM bankroll_history ORDER BY date")
        history = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if not history:
            return {
                'current': self.initial_bankroll,
                'initial': self.initial_bankroll,
                'total_change': 0,
                'roi_percent': 0
            }
        
        current = history[-1]['amount']
        initial = history[0]['amount']
        total_change = current - initial
        roi = (total_change / initial * 100) if initial > 0 else 0
        
        return {
            'current': current,
            'initial': initial,
            'total_change': total_change,
            'roi_percent': roi,
            'history': history
        }

