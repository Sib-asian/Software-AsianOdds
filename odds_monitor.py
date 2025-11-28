"""
Sistema Monitoraggio Quote Real-Time
====================================

Monitora quote ogni 5-10 minuti e alert quando cambiano significativamente.
Identifica "sharp money" e movimenti sospetti.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OddsSnapshot:
    """Snapshot di quote in un momento specifico"""
    match_id: str
    timestamp: datetime
    odds_home: float
    odds_draw: float
    odds_away: float
    bookmaker: str = "best"


@dataclass
class OddsMovement:
    """Movimento di quote rilevato"""
    match_id: str
    market: str  # 'home', 'draw', 'away'
    old_odds: float
    new_odds: float
    movement_percent: float
    movement_type: str  # 'UP', 'DOWN', 'STABLE'
    timestamp: datetime
    is_sharp_money: bool = False


class OddsMonitor:
    """
    Monitora quote real-time e rileva movimenti significativi.
    """
    
    def __init__(self, storage_path: str = "odds_history.json"):
        self.storage_path = Path(storage_path)
        self.odds_history: Dict[str, List[OddsSnapshot]] = {}
        self.movement_threshold = 0.05  # 5% movimento minimo
        self.sharp_money_threshold = 0.10  # 10% movimento = sharp money
        self._load_history()
    
    def _load_history(self):
        """Carica storico quote da file"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    # Converti timestamp string a datetime
                    for match_id, snapshots in data.items():
                        self.odds_history[match_id] = [
                            OddsSnapshot(
                                match_id=s['match_id'],
                                timestamp=datetime.fromisoformat(s['timestamp']),
                                odds_home=s['odds_home'],
                                odds_draw=s['odds_draw'],
                                odds_away=s['odds_away'],
                                bookmaker=s.get('bookmaker', 'best')
                            )
                            for s in snapshots
                        ]
                logger.info(f"✅ Caricato storico quote: {len(self.odds_history)} partite")
            except Exception as e:
                logger.warning(f"⚠️  Errore caricamento storico: {e}")
                self.odds_history = {}
    
    def _save_history(self):
        """Salva storico quote su file"""
        try:
            data = {}
            for match_id, snapshots in self.odds_history.items():
                data[match_id] = [
                    {
                        'match_id': s.match_id,
                        'timestamp': s.timestamp.isoformat(),
                        'odds_home': s.odds_home,
                        'odds_draw': s.odds_draw,
                        'odds_away': s.odds_away,
                        'bookmaker': s.bookmaker
                    }
                    for s in snapshots
                ]
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Errore salvataggio storico: {e}")
    
    def record_odds(
        self,
        match_id: str,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        bookmaker: str = "best"
    ) -> List[OddsMovement]:
        """
        Registra quote attuali e rileva movimenti.
        
        Returns:
            Lista di movimenti rilevati
        """
        try:
            now = datetime.now()
            snapshot = OddsSnapshot(
                match_id=match_id,
                timestamp=now,
                odds_home=odds_home,
                odds_draw=odds_draw,
                odds_away=odds_away,
                bookmaker=bookmaker
            )
            
            # Aggiungi a storico
            if match_id not in self.odds_history:
                self.odds_history[match_id] = []
            
            self.odds_history[match_id].append(snapshot)
            
            # Mantieni solo ultime 24h
            cutoff = now - timedelta(hours=24)
            self.odds_history[match_id] = [
                s for s in self.odds_history[match_id]
                if s.timestamp > cutoff
            ]
            
            # Rileva movimenti
            movements = self._detect_movements(match_id, snapshot)
            
            # Salva storico
            self._save_history()
            
            return movements
            
        except Exception as e:
            logger.error(f"❌ Errore registrazione quote: {e}")
            return []
    
    def _detect_movements(
        self,
        match_id: str,
        current_snapshot: OddsSnapshot
    ) -> List[OddsMovement]:
        """Rileva movimenti significativi nelle quote"""
        movements = []
        
        if match_id not in self.odds_history or len(self.odds_history[match_id]) < 2:
            return movements
        
        # Confronta con snapshot precedente
        previous_snapshot = self.odds_history[match_id][-2]
        
        # Controlla ogni market
        markets = [
            ('home', current_snapshot.odds_home, previous_snapshot.odds_home),
            ('draw', current_snapshot.odds_draw, previous_snapshot.odds_draw),
            ('away', current_snapshot.odds_away, previous_snapshot.odds_away)
        ]
        
        for market, new_odds, old_odds in markets:
            if old_odds <= 0 or new_odds <= 0:
                continue
            
            # Calcola movimento percentuale
            movement_pct = ((new_odds - old_odds) / old_odds) * 100
            
            # Rileva solo movimenti significativi
            if abs(movement_pct) >= (self.movement_threshold * 100):
                movement_type = 'DOWN' if movement_pct < 0 else 'UP'
                is_sharp_money = abs(movement_pct) >= (self.sharp_money_threshold * 100)
                
                movement = OddsMovement(
                    match_id=match_id,
                    market=market,
                    old_odds=old_odds,
                    new_odds=new_odds,
                    movement_percent=movement_pct,
                    movement_type=movement_type,
                    timestamp=current_snapshot.timestamp,
                    is_sharp_money=is_sharp_money
                )
                movements.append(movement)
        
        return movements
    
    def get_odds_trend(
        self,
        match_id: str,
        market: str = 'home',
        hours: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Ottiene trend quote per una partita.
        
        Returns:
            Dict con trend, volatilità, direzione
        """
        if match_id not in self.odds_history:
            return None
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [
            s for s in self.odds_history[match_id]
            if s.timestamp > cutoff
        ]
        
        if len(recent_snapshots) < 2:
            return None
        
        # Estrai quote per market
        odds_values = []
        if market == 'home':
            odds_values = [s.odds_home for s in recent_snapshots]
        elif market == 'draw':
            odds_values = [s.odds_draw for s in recent_snapshots]
        elif market == 'away':
            odds_values = [s.odds_away for s in recent_snapshots]
        
        if not odds_values:
            return None
        
        # Calcola trend
        first_odds = odds_values[0]
        last_odds = odds_values[-1]
        trend_pct = ((last_odds - first_odds) / first_odds) * 100
        
        # Calcola volatilità (std dev)
        import numpy as np
        volatility = float(np.std(odds_values)) if len(odds_values) > 1 else 0.0
        
        # Direzione
        if trend_pct < -2:
            direction = 'FALLING'
        elif trend_pct > 2:
            direction = 'RISING'
        else:
            direction = 'STABLE'
        
        return {
            'match_id': match_id,
            'market': market,
            'first_odds': first_odds,
            'last_odds': last_odds,
            'trend_percent': trend_pct,
            'volatility': volatility,
            'direction': direction,
            'snapshots_count': len(recent_snapshots),
            'hours': hours
        }
    
    def get_sharp_money_alerts(self, hours: int = 1) -> List[OddsMovement]:
        """Ottiene alert per sharp money rilevati"""
        alerts = []
        cutoff = datetime.now() - timedelta(hours=hours)
        
        for match_id, snapshots in self.odds_history.items():
            for i in range(1, len(snapshots)):
                current = snapshots[i]
                previous = snapshots[i-1]
                
                if current.timestamp < cutoff:
                    continue
                
                # Controlla movimenti sharp money
                for market in ['home', 'draw', 'away']:
                    old_odds = getattr(previous, f'odds_{market}')
                    new_odds = getattr(current, f'odds_{market}')
                    
                    if old_odds <= 0 or new_odds <= 0:
                        continue
                    
                    movement_pct = ((new_odds - old_odds) / old_odds) * 100
                    
                    if abs(movement_pct) >= (self.sharp_money_threshold * 100):
                        movement = OddsMovement(
                            match_id=match_id,
                            market=market,
                            old_odds=old_odds,
                            new_odds=new_odds,
                            movement_percent=movement_pct,
                            movement_type='DOWN' if movement_pct < 0 else 'UP',
                            timestamp=current.timestamp,
                            is_sharp_money=True
                        )
                        alerts.append(movement)
        
        return alerts

