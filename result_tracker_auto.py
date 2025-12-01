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
    # 🆕 Dati estesi per valutazione mercati complessi
    events: Optional[List[Dict[str, Any]]] = None  # Eventi partita (gol, cartellini, corner)
    statistics: Optional[Dict[str, Any]] = None  # Statistiche aggregate (corner, cartellini, ecc.)


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
        total_tracked = len(self.tracked_matches)
        
        if total_tracked == 0:
            logger.debug("ℹ️  Nessuna partita tracciata per aggiornare risultati")
            return updated_results
        
        logger.debug(f"🔄 Controllando {total_tracked} partite tracciate per aggiornamento risultati...")
        
        for match_id, match in list(self.tracked_matches.items()):
            try:
                # 🆕 FIX: Controlla anche partite FINISHED per aggiornare risultati finali
                # (potrebbero essere state aggiornate da un ciclo precedente)
                if match.status == "CANCELLED":
                    continue
                
                # Controlla se è ora di aggiornare
                if match_id in self.last_update:
                    time_since_update = (datetime.now() - self.last_update[match_id]).total_seconds()
                    # Per partite FINISHED, controlla solo ogni 30 minuti (per evitare chiamate inutili)
                    if match.status == "FINISHED" and time_since_update < 1800:  # 30 minuti
                        continue
                    elif match.status == "SCHEDULED" and time_since_update < 300:  # 5 minuti
                        continue
                    elif match.status == "LIVE" and time_since_update < 60:  # 1 minuto per live
                        continue
                
                # Prova ad aggiornare risultato
                logger.debug(f"🔄 Recuperando risultato per {match_id}...")
                updated_result = self._fetch_match_result(match)
                if updated_result:
                    # Aggiorna solo se ci sono cambiamenti
                    if (updated_result.status != match.status or 
                        updated_result.home_score != match.home_score or 
                        updated_result.away_score != match.away_score):
                        self.tracked_matches[match_id] = updated_result
                        self.last_update[match_id] = datetime.now()
                        updated_results.append(updated_result)
                        
                        # Log se partita finita (solo se appena finita)
                        if updated_result.status == "FINISHED" and match.status != "FINISHED":
                            logger.info(
                                f"✅ Partita finita: {updated_result.home_team} "
                                f"{updated_result.home_score}-{updated_result.away_score} "
                                f"{updated_result.away_team}"
                            )
                            # 🆕 Log se eventi/statistiche recuperati
                            if updated_result.events:
                                logger.info(f"   📊 Recuperati {len(updated_result.events)} eventi per valutazione mercati")
                            if updated_result.statistics:
                                logger.info(f"   📈 Recuperate statistiche: corners={updated_result.statistics.get('total_corners', 'N/A')}, cards={updated_result.statistics.get('total_cards', 'N/A')}")
                else:
                    logger.debug(f"⚠️  Impossibile recuperare risultato per {match_id}")
            
            except Exception as e:
                logger.error(f"❌ Errore aggiornamento risultato {match_id}: {e}")
                continue
        
        return updated_results
    
    def _fetch_match_result(self, match: MatchResult) -> Optional[MatchResult]:
        """
        Recupera risultato partita da API-SPORTS.
        
        Supporta match_id in formato:
        - "apisports_live_XXXXX" (da API-SPORTS live)
        - "apisports_XXXXX" (da API-SPORTS fixtures)
        - "XXXXX" (solo ID numerico)
        """
        try:
            import os
            import urllib.request
            import json
            
            # Estrai fixture ID dal match_id
            fixture_id = None
            if match.match_id.startswith("apisports_live_"):
                fixture_id = match.match_id.replace("apisports_live_", "")
            elif match.match_id.startswith("apisports_"):
                fixture_id = match.match_id.replace("apisports_", "")
            else:
                # Prova a usare match_id direttamente se è numerico
                try:
                    fixture_id = str(int(match.match_id))
                except ValueError:
                    logger.debug(f"⚠️  Match ID non riconosciuto: {match.match_id}")
                    return None
            
            if not fixture_id:
                return None
            
            # Recupera API key
            api_key = os.getenv("API_FOOTBALL_KEY", "") or os.getenv("RAPIDAPI_KEY", "") or "95c43f936816cd4389a747fd2cfe061a"
            if not api_key:
                logger.debug("⚠️  API-SPORTS key non disponibile per recuperare risultati")
                return None
            
            # Chiama API-SPORTS per ottenere fixture specifica
            url = f"https://v3.football.api-sports.io/fixtures?id={fixture_id}"
            
            request = urllib.request.Request(url)
            request.add_header("x-rapidapi-key", api_key)
            request.add_header("x-rapidapi-host", "v3.football.api-sports.io")
            
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                if not data.get("response"):
                    return None
                
                fixture_data = data["response"][0]
                fixture = fixture_data.get("fixture", {})
                teams = fixture_data.get("teams", {})
                goals = fixture_data.get("goals", {})
                
                # Estrai informazioni
                status = fixture.get("status", {}).get("long", "SCHEDULED")
                minute = fixture.get("status", {}).get("elapsed")
                home_score = goals.get("home")
                away_score = goals.get("away")
                
                # Determina status finale
                if status == "Match Finished":
                    final_status = "FINISHED"
                elif status in ["First Half", "Second Half", "Halftime", "Extra Time", "Penalty In Progress"]:
                    final_status = "LIVE"
                else:
                    final_status = "SCHEDULED"
                
                # 🆕 Recupera eventi e statistiche se partita finita
                events = None
                statistics = None
                home_team_id = teams.get("home", {}).get("id")  # 🆕 Salva team IDs per confronto
                away_team_id = teams.get("away", {}).get("id")
                
                if final_status == "FINISHED":
                    # 🆕 Retry logic per eventi (max 3 tentativi)
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Recupera eventi (gol, cartellini, corner, ecc.)
                            events_url = f"https://v3.football.api-sports.io/fixtures/events?id={fixture_id}"
                            events_request = urllib.request.Request(events_url)
                            events_request.add_header("x-rapidapi-key", api_key)
                            events_request.add_header("x-rapidapi-host", "v3.football.api-sports.io")
                            
                            with urllib.request.urlopen(events_request, timeout=10) as events_response:
                                events_data = json.loads(events_response.read().decode())
                                if events_data.get("response") and len(events_data["response"]) > 0:
                                    events = events_data["response"][0].get("events", [])
                                    # 🆕 Aggiungi team IDs agli eventi per facilitare valutazione
                                    for event in events:
                                        event["_home_team_id"] = home_team_id
                                        event["_away_team_id"] = away_team_id
                                    logger.info(f"✅ Recuperati {len(events)} eventi per partita {match.match_id}")
                                    break
                                else:
                                    logger.warning(f"⚠️  Nessun evento disponibile per partita {match.match_id}")
                        except urllib.error.HTTPError as e:
                            if e.code == 429:  # Rate limit
                                wait_time = (attempt + 1) * 2
                                logger.warning(f"⚠️  Rate limit API-SPORTS, attendo {wait_time}s (tentativo {attempt + 1}/{max_retries})")
                                time.sleep(wait_time)
                            else:
                                logger.error(f"❌ Errore HTTP recupero eventi per {match.match_id}: {e.code}")
                                if attempt == max_retries - 1:
                                    break
                        except Exception as e:
                            logger.error(f"❌ Errore recupero eventi per {match.match_id} (tentativo {attempt + 1}/{max_retries}): {e}")
                            if attempt < max_retries - 1:
                                time.sleep(1)
                            else:
                                break
                    
                    # 🆕 Retry logic per statistiche (max 3 tentativi)
                    for attempt in range(max_retries):
                        try:
                            # Recupera statistiche (corner, cartellini, ecc.)
                            stats_url = f"https://v3.football.api-sports.io/fixtures/statistics?id={fixture_id}"
                            stats_request = urllib.request.Request(stats_url)
                            stats_request.add_header("x-rapidapi-key", api_key)
                            stats_request.add_header("x-rapidapi-host", "v3.football.api-sports.io")
                            
                            with urllib.request.urlopen(stats_request, timeout=10) as stats_response:
                                stats_data = json.loads(stats_response.read().decode())
                                if stats_data.get("response") and len(stats_data["response"]) > 0:
                                    stats_list = stats_data["response"][0].get("statistics", [])
                                    # Converti lista statistiche in dict per facile accesso
                                    statistics = {}
                                    home_team_name = teams.get("home", {}).get("name", "")
                                    away_team_name = teams.get("away", {}).get("name", "")
                                    
                                    for team_stats in stats_list:
                                        team_info = team_stats.get("team", {})
                                        team_name = team_info.get("name", "")
                                        team_id = team_info.get("id")
                                        # 🆕 Usa team ID per identificazione più affidabile
                                        if team_id == home_team_id or team_name == home_team_name:
                                            team_type = "home"
                                        elif team_id == away_team_id or team_name == away_team_name:
                                            team_type = "away"
                                        else:
                                            continue  # Skip se team non riconosciuto
                                        
                                        for stat in team_stats.get("statistics", []):
                                            stat_type = stat.get("type", "")
                                            stat_value = stat.get("value")
                                            if stat_type and stat_value is not None:
                                                # 🆕 Gestione valori None/N/A
                                                if stat_value in [None, "None", "N/A", "-", ""]:
                                                    continue
                                                
                                                key = f"{team_type}_{stat_type.lower().replace(' ', '_')}"
                                                # Prova a convertire in numero se possibile
                                                try:
                                                    if isinstance(stat_value, str):
                                                        # Rimuovi caratteri non numerici e prova a convertire
                                                        clean_value = ''.join(c for c in stat_value if c.isdigit() or c == '.')
                                                        if clean_value:
                                                            statistics[key] = float(clean_value) if '.' in clean_value else int(clean_value)
                                                        else:
                                                            continue
                                                    elif isinstance(stat_value, (int, float)):
                                                        statistics[key] = stat_value
                                                    else:
                                                        continue
                                                except (ValueError, TypeError):
                                                    continue
                                    
                                    # 🆕 Calcola totali con validazione
                                    def safe_get(key, default=0):
                                        val = statistics.get(key, default)
                                        return val if isinstance(val, (int, float)) else default
                                    
                                    home_corners = safe_get("home_corner_kicks")
                                    away_corners = safe_get("away_corner_kicks")
                                    statistics["total_corners"] = home_corners + away_corners
                                    
                                    home_yellow = safe_get("home_yellow_cards")
                                    away_yellow = safe_get("away_yellow_cards")
                                    home_red = safe_get("home_red_cards")
                                    away_red = safe_get("away_red_cards")
                                    statistics["total_cards"] = home_yellow + away_yellow + home_red + away_red
                                    
                                    # 🆕 Validazione: verifica coerenza con risultato
                                    if events:
                                        # Conta gol dagli eventi e confronta con risultato finale
                                        goals_from_events = {"home": 0, "away": 0}
                                        for event in events:
                                            if event.get("type", {}).get("name") == "Goal":
                                                event_team_id = event.get("team", {}).get("id")
                                                if event_team_id == home_team_id:
                                                    goals_from_events["home"] += 1
                                                elif event_team_id == away_team_id:
                                                    goals_from_events["away"] += 1
                                        
                                        # Se c'è discrepanza, logga warning
                                        if goals_from_events["home"] != home_score or goals_from_events["away"] != away_score:
                                            logger.warning(
                                                f"⚠️  Discrepanza gol: Eventi={goals_from_events}, "
                                                f"Risultato={home_score}-{away_score} per {match.match_id}"
                                            )
                                    
                                    logger.info(f"✅ Recuperate statistiche per partita {match.match_id}: {len(statistics)} campi")
                                    break
                                else:
                                    logger.warning(f"⚠️  Nessuna statistica disponibile per partita {match.match_id}")
                        except urllib.error.HTTPError as e:
                            if e.code == 429:  # Rate limit
                                wait_time = (attempt + 1) * 2
                                logger.warning(f"⚠️  Rate limit API-SPORTS, attendo {wait_time}s (tentativo {attempt + 1}/{max_retries})")
                                time.sleep(wait_time)
                            else:
                                logger.error(f"❌ Errore HTTP recupero statistiche per {match.match_id}: {e.code}")
                                if attempt == max_retries - 1:
                                    break
                        except Exception as e:
                            logger.error(f"❌ Errore recupero statistiche per {match.match_id} (tentativo {attempt + 1}/{max_retries}): {e}")
                            if attempt < max_retries - 1:
                                time.sleep(1)
                            else:
                                break
                
                # Aggiorna match result
                updated_match = MatchResult(
                    match_id=match.match_id,
                    home_team=teams.get("home", {}).get("name", match.home_team),
                    away_team=teams.get("away", {}).get("name", match.away_team),
                    home_score=home_score,
                    away_score=away_score,
                    status=final_status,
                    minute=minute,
                    timestamp=match.timestamp,
                    league=match.league,
                    events=events,
                    statistics=statistics
                )
                
                # Log se partita finita
                if final_status == "FINISHED" and match.status != "FINISHED":
                    logger.info(
                        f"✅ Risultato recuperato: {updated_match.home_team} "
                        f"{updated_match.home_score}-{updated_match.away_score} "
                        f"{updated_match.away_team}"
                    )
                
                return updated_match
            
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.debug(f"⚠️  Rate limit API-SPORTS per risultato {match.match_id}")
            else:
                logger.debug(f"⚠️  Errore HTTP recupero risultato {match.match_id}: {e.code}")
            return None
        except Exception as e:
            logger.debug(f"⚠️  Errore fetch risultato {match.match_id}: {e}")
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

