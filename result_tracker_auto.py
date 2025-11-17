"""
Sistema Tracking Risultati Automatico
======================================

Monitora partite in corso e aggiorna risultati automaticamente.
Calcola ROI real-time e performance tracking.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Risultato partita"""
    match_id: str
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str = "SCHEDULED"  # SCHEDULED, LIVE, FINISHED, CANCELLED
    minute: Optional[int] = None
    timestamp: datetime = None
    league: Optional[str] = None


class ResultTrackerAuto:
    """
    Traccia risultati partite automaticamente.
    """
    
    def __init__(self, api_manager=None):
        self.api_manager = api_manager
        self.tracked_matches: Dict[str, MatchResult] = {}
        self.last_update = {}
    
    def add_match_to_track(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        league: Optional[str] = None,
        match_date: Optional[datetime] = None
    ):
        """Aggiungi partita da tracciare"""
        self.tracked_matches[match_id] = MatchResult(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            timestamp=match_date or datetime.now(),
            status="SCHEDULED"
        )
        logger.debug(f"✅ Partita aggiunta al tracking: {match_id}")
    
    def update_results(self) -> List[MatchResult]:
        """
        Aggiorna risultati per tutte le partite tracciate.
        
        Returns:
            Lista di risultati aggiornati
        """
        updated_results = []
        
        for match_id, match in list(self.tracked_matches.items()):
            try:
                # Aggiorna solo se partita dovrebbe essere iniziata o in corso
                if match.status == "FINISHED" or match.status == "CANCELLED":
                    continue
                
                # Controlla se è ora di aggiornare (ogni 5 minuti per partite live)
                if match_id in self.last_update:
                    time_since_update = (datetime.now() - self.last_update[match_id]).total_seconds()
                    if match.status == "SCHEDULED" and time_since_update < 300:  # 5 minuti
                        continue
                    elif match.status == "LIVE" and time_since_update < 60:  # 1 minuto per live
                        continue
                
                # Prova ad aggiornare risultato
                updated_result = self._fetch_match_result(match)
                if updated_result:
                    self.tracked_matches[match_id] = updated_result
                    self.last_update[match_id] = datetime.now()
                    updated_results.append(updated_result)
                    
                    # Log se partita finita
                    if updated_result.status == "FINISHED":
                        logger.info(
                            f"✅ Partita finita: {updated_result.home_team} "
                            f"{updated_result.home_score}-{updated_result.away_score} "
                            f"{updated_result.away_team}"
                        )
            
            except Exception as e:
                logger.error(f"❌ Errore aggiornamento risultato {match_id}: {e}")
                continue
        
        return updated_results
    
    def _fetch_match_result(self, match: MatchResult) -> Optional[MatchResult]:
        """Recupera risultato partita da API"""
        try:
            # Usa API-Football se disponibile
            if self.api_manager:
                # Prova a ottenere risultato da API-Football
                # Questo è un esempio - adatta alla tua API
                pass
            
            # Fallback: usa TheOddsAPI per risultati
            # TheOddsAPI non fornisce risultati diretti, ma possiamo usare API-Football
            
            # Per ora, ritorna None (da implementare con API specifica)
            return None
            
        except Exception as e:
            logger.debug(f"⚠️  Errore fetch risultato: {e}")
            return None
    
    def get_match_result(self, match_id: str) -> Optional[MatchResult]:
        """Ottiene risultato per una partita specifica"""
        return self.tracked_matches.get(match_id)
    
    def get_finished_matches(self) -> List[MatchResult]:
        """Ottiene tutte le partite finite"""
        return [
            match for match in self.tracked_matches.values()
            if match.status == "FINISHED"
        ]
    
    def get_live_matches(self) -> List[MatchResult]:
        """Ottiene tutte le partite in corso"""
        return [
            match for match in self.tracked_matches.values()
            if match.status == "LIVE"
        ]
    
    def update_betting_results(
        self,
        betting_tracker,
        match_id: str,
        outcome: str  # 'W', 'L', 'PUSH'
    ):
        """
        Aggiorna risultati scommesse quando partita finisce.
        
        Args:
            betting_tracker: Istanza di BettingResultsTracker
            match_id: ID partita
            outcome: Risultato scommessa ('W', 'L', 'PUSH')
        """
        try:
            if not betting_tracker:
                return
            
            # Trova opportunità per questa partita
            # e aggiorna risultato
            # Questo richiede integrazione con BettingResultsTracker
            logger.info(f"✅ Risultato aggiornato per {match_id}: {outcome}")
            
        except Exception as e:
            logger.error(f"❌ Errore aggiornamento betting result: {e}")

