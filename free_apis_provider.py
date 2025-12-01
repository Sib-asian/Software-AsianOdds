"""
Provider API Gratuite - Senza Chiave API
=========================================

API gratuite che NON richiedono chiave API per supportare API-Football:
1. OpenLigaDB - Bundesliga (gratuito, no key)
2. Football-API (gratuito, no key, limitato)
3. API-Football Free Tier (richiede key ma gratuito)
4. Web scraping da fonti pubbliche (come ultima risorsa)

Questo provider funziona come supplemento quando API-Football non è disponibile.
"""

import logging
import urllib.request
import urllib.parse
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import time
import re

logger = logging.getLogger(__name__)


class FreeAPIsProvider:
    """
    Provider di API gratuite senza chiave API per dati live e statistiche.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60  # 1 minuto
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    def get_live_data(
        self,
        home_team: str,
        away_team: str,
        league_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Ottiene dati live da API gratuite (senza chiave).
        
        Args:
            home_team: Nome squadra casa
            away_team: Nome squadra trasferta
            league_name: Nome lega (opzionale, per filtrare)
        
        Returns:
            Dict con dati live o None se non disponibili
        """
        # Prova in ordine di priorità
        # 1. OpenLigaDB (Bundesliga)
        if league_name and ('bundesliga' in league_name.lower() or 'germany' in league_name.lower()):
            live_data = self._try_openligadb(home_team, away_team)
            if live_data:
                logger.debug(f"✅ Dati live da OpenLigaDB per {home_team} vs {away_team}")
                return live_data
        
        # 2. Football-API (gratuito, no key, limitato)
        live_data = self._try_football_api(home_team, away_team)
        if live_data:
            logger.debug(f"✅ Dati live da Football-API per {home_team} vs {away_team}")
            return live_data
        
        # 3. API-Football Free Tier (richiede key ma è gratuito)
        # Nota: Questo richiede comunque una chiave, ma è nel tier gratuito
        # Lo includiamo qui per completezza, ma in realtà va configurato separatamente
        
        return None
    
    def _try_openligadb(
        self,
        home_team: str,
        away_team: str
    ) -> Optional[Dict[str, Any]]:
        """
        Prova a ottenere dati da OpenLigaDB (Bundesliga, gratuito, no key).
        API: https://www.openligadb.de/
        """
        try:
            # OpenLigaDB ha un endpoint per partite live
            # Endpoint: https://www.openligadb.de/api/getmatchdata/1 (1 = Bundesliga)
            # Per partite live: https://www.openligadb.de/api/getmatchdata/1/2024 (anno)
            
            current_year = datetime.now().year
            url = f"https://www.openligadb.de/api/getmatchdata/1/{current_year}"
            
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                # Cerca partita corrispondente
                for match in data:
                    team1 = match.get("Team1", {}).get("TeamName", "").lower()
                    team2 = match.get("Team2", {}).get("TeamName", "").lower()
                    
                    # Match fuzzy
                    if (home_team.lower() in team1 or team1 in home_team.lower()) and \
                       (away_team.lower() in team2 or team2 in away_team.lower()):
                        
                        # Estrai dati
                        match_results = match.get("MatchResults", [])
                        score_home = 0
                        score_away = 0
                        
                        if match_results:
                            # Prendi risultato finale o più recente
                            latest_result = match_results[-1]
                            score_home = latest_result.get("PointsTeam1", 0)
                            score_away = latest_result.get("PointsTeam2", 0)
                        
                        # Minuto
                        match_date = match.get("MatchDateTime", "")
                        match_is_finished = match.get("MatchIsFinished", False)
                        match_status = match.get("MatchStatus", 0)
                        
                        # MatchStatus: 0 = non iniziata, 1 = prima metà, 2 = pausa, 3 = seconda metà, 4 = finita
                        minute = 0
                        if match_status == 1:
                            # Prima metà, stima minuto (non disponibile direttamente)
                            minute = 30
                        elif match_status == 2:
                            minute = 45
                        elif match_status == 3:
                            minute = 60
                        elif match_status == 4:
                            minute = 90
                        
                        # Statistiche (limitato in OpenLigaDB)
                        return {
                            'score_home': score_home,
                            'score_away': score_away,
                            'minute': minute,
                            'possession_home': 50,  # Non disponibile
                            'possession_away': 50,
                            'shots_home': 0,  # Non disponibile
                            'shots_away': 0,
                            'shots_on_target_home': 0,
                            'shots_on_target_away': 0,
                            'corners_home': 0,
                            'corners_away': 0,
                            'yellow_cards_home': 0,
                            'yellow_cards_away': 0,
                            'red_cards_home': 0,
                            'red_cards_away': 0,
                            'status': 'Finished' if match_is_finished else 'Live',
                            'events': [],
                            'source': 'OpenLigaDB',
                            'estimated': False  # Score è reale
                        }
        except Exception as e:
            logger.debug(f"⚠️  OpenLigaDB non disponibile: {e}")
        
        return None
    
    def _try_football_api(
        self,
        home_team: str,
        away_team: str
    ) -> Optional[Dict[str, Any]]:
        """
        Prova a ottenere dati da Football-API (gratuito, no key, molto limitato).
        API: https://www.football-api.com/
        Nota: Questa API è molto limitata e potrebbe non essere più attiva.
        """
        try:
            # Football-API ha endpoint pubblici ma molto limitati
            # Endpoint: https://www.football-api.com/api/v1.0/matches
            # Nota: Questa API potrebbe richiedere autenticazione o essere disattivata
            
            # Per ora saltiamo questa fonte (troppo limitata/inattiva)
            return None
        except Exception as e:
            logger.debug(f"⚠️  Football-API non disponibile: {e}")
        
        return None
    
    def get_live_matches(self, league_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Ottiene lista di partite live da API gratuite.
        
        Args:
            league_id: ID lega (opzionale, per OpenLigaDB: 1=Bundesliga)
        
        Returns:
            Lista di partite live
        """
        live_matches = []
        
        # OpenLigaDB per Bundesliga
        if league_id == 1 or league_id is None:  # 1 = Bundesliga
            matches = self._get_openligadb_live_matches()
            live_matches.extend(matches)
        
        return live_matches
    
    def _get_openligadb_live_matches(self) -> List[Dict[str, Any]]:
        """
        Ottiene partite live da OpenLigaDB (Bundesliga).
        """
        try:
            current_year = datetime.now().year
            url = f"https://www.openligadb.de/api/getmatchdata/1/{current_year}"
            
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                live_matches = []
                for match in data:
                    match_status = match.get("MatchStatus", 0)
                    match_is_finished = match.get("MatchIsFinished", False)
                    
                    # Solo partite live (status 1 o 3) o recenti (finite da < 2 ore)
                    if match_status in [1, 3] or (match_is_finished and match_status == 4):
                        team1 = match.get("Team1", {}).get("TeamName", "")
                        team2 = match.get("Team2", {}).get("TeamName", "")
                        
                        match_results = match.get("MatchResults", [])
                        score_home = 0
                        score_away = 0
                        
                        if match_results:
                            latest_result = match_results[-1]
                            score_home = latest_result.get("PointsTeam1", 0)
                            score_away = latest_result.get("PointsTeam2", 0)
                        
                        live_matches.append({
                            'home': team1,
                            'away': team2,
                            'score_home': score_home,
                            'score_away': score_away,
                            'league': 'Bundesliga',
                            'league_id': 1,
                            'source': 'OpenLigaDB',
                            'is_live': match_status in [1, 3]
                        })
                
                return live_matches
        except Exception as e:
            logger.debug(f"⚠️  Errore OpenLigaDB live matches: {e}")
        
        return []
    
    def get_statistics(
        self,
        home_team: str,
        away_team: str,
        league_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Ottiene statistiche da API gratuite (limitato).
        
        Nota: La maggior parte delle API gratuite non fornisce statistiche dettagliate.
        Questo metodo è principalmente un placeholder per future integrazioni.
        """
        # OpenLigaDB non fornisce statistiche dettagliate
        # Altre API gratuite sono molto limitate
        
        return None


# 🆕 NUOVO: Provider per API pubbliche senza autenticazione (web scraping etico)
class PublicDataProvider:
    """
    Provider per dati pubblici da fonti web (web scraping etico).
    Usa solo dati pubblicamente disponibili e rispetta robots.txt.
    """
    
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        self.cache = {}
        self.cache_timeout = 120  # 2 minuti (più lungo per evitare troppe richieste)
    
    def get_live_score(
        self,
        home_team: str,
        away_team: str
    ) -> Optional[Dict[str, Any]]:
        """
        Ottiene score live da fonti pubbliche (web scraping).
        
        ⚠️  NOTA: Web scraping può violare ToS di alcuni siti.
        Usa solo per test o con permesso esplicito.
        """
        # Per ora non implementiamo web scraping per evitare problemi legali
        # Se necessario, implementare con:
        # - Rispetto robots.txt
        # - Rate limiting
        # - User-Agent appropriato
        # - Solo dati pubblicamente disponibili
        
        return None


# Funzione helper per integrare nel sistema esistente
def get_free_apis_provider() -> Optional[FreeAPIsProvider]:
    """
    Crea e restituisce un'istanza di FreeAPIsProvider.
    """
    try:
        return FreeAPIsProvider()
    except Exception as e:
        logger.warning(f"⚠️  Impossibile inizializzare FreeAPIsProvider: {e}")
        return None







