#!/usr/bin/env python3
"""
Sistema Whitelist/Blacklist per Filtri Partite
===============================================

Filtra partite/leghe da analizzare o ignorare.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime, time

logger = logging.getLogger(__name__)


class MatchFilters:
    """Gestisce filtri per partite"""
    
    def __init__(self, config_path: str = "filters_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Carica configurazione filtri"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Errore caricamento config: {e}")
        
        # Config default
        return {
            "whitelist": {
                "leagues": [],
                "teams": [],
                "enabled": False
            },
            "blacklist": {
                "leagues": [],
                "teams": [],
                "enabled": True
            },
            "time_filters": {
                "enabled": False,
                "start_hour": 0,
                "end_hour": 23
            },
            "market_filters": {
                "enabled": False,
                "allowed_markets": ["1X2_HOME", "1X2_AWAY", "1X2_DRAW"]
            }
        }
    
    def save_config(self):
        """Salva configurazione"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"✅ Config salvata: {self.config_path}")
        except Exception as e:
            logger.error(f"❌ Errore salvataggio config: {e}")
    
    def should_analyze_match(self, match: Dict) -> bool:
        """
        Verifica se partita deve essere analizzata
        
        Args:
            match: Dict con dati partita
            
        Returns:
            True se deve essere analizzata
        """
        # Whitelist
        if self.config['whitelist']['enabled']:
            if not self._check_whitelist(match):
                logger.debug(f"Match {match.get('id')} non in whitelist, skip")
                return False
        
        # Blacklist
        if self.config['blacklist']['enabled']:
            if self._check_blacklist(match):
                logger.debug(f"Match {match.get('id')} in blacklist, skip")
                return False
        
        # Filtri orario
        if self.config['time_filters']['enabled']:
            if not self._check_time_filter(match):
                logger.debug(f"Match {match.get('id')} fuori orario, skip")
                return False
        
        return True
    
    def _check_whitelist(self, match: Dict) -> bool:
        """Verifica whitelist"""
        league = match.get('league', '').lower()
        home = match.get('home', '').lower()
        away = match.get('away', '').lower()
        
        # Check leghe
        whitelist_leagues = [l.lower() for l in self.config['whitelist']['leagues']]
        if whitelist_leagues and league not in whitelist_leagues:
            return False
        
        # Check teams
        whitelist_teams = [t.lower() for t in self.config['whitelist']['teams']]
        if whitelist_teams:
            if home not in whitelist_teams and away not in whitelist_teams:
                return False
        
        return True
    
    def _check_blacklist(self, match: Dict) -> bool:
        """Verifica blacklist"""
        league = match.get('league', '').lower()
        home = match.get('home', '').lower()
        away = match.get('away', '').lower()
        
        # Check leghe
        blacklist_leagues = [l.lower() for l in self.config['blacklist']['leagues']]
        if league in blacklist_leagues:
            return True
        
        # Check teams
        blacklist_teams = [t.lower() for t in self.config['blacklist']['teams']]
        if home in blacklist_teams or away in blacklist_teams:
            return True
        
        return False
    
    def _check_time_filter(self, match: Dict) -> bool:
        """Verifica filtro orario"""
        match_date = match.get('date')
        if not match_date:
            return True
        
        if isinstance(match_date, str):
            match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
        
        match_hour = match_date.hour
        start_hour = self.config['time_filters']['start_hour']
        end_hour = self.config['time_filters']['end_hour']
        
        if start_hour <= end_hour:
            return start_hour <= match_hour <= end_hour
        else:  # Oltre mezzanotte
            return match_hour >= start_hour or match_hour <= end_hour
    
    def should_analyze_market(self, market: str) -> bool:
        """Verifica se market deve essere analizzato"""
        if not self.config['market_filters']['enabled']:
            return True
        
        allowed = self.config['market_filters']['allowed_markets']
        return market in allowed
    
    def add_to_whitelist(self, leagues: List[str] = None, teams: List[str] = None):
        """Aggiunge a whitelist"""
        if leagues:
            self.config['whitelist']['leagues'].extend(leagues)
            self.config['whitelist']['leagues'] = list(set(self.config['whitelist']['leagues']))
        if teams:
            self.config['whitelist']['teams'].extend(teams)
            self.config['whitelist']['teams'] = list(set(self.config['whitelist']['teams']))
        self.save_config()
    
    def add_to_blacklist(self, leagues: List[str] = None, teams: List[str] = None):
        """Aggiunge a blacklist"""
        if leagues:
            self.config['blacklist']['leagues'].extend(leagues)
            self.config['blacklist']['leagues'] = list(set(self.config['blacklist']['leagues']))
        if teams:
            self.config['blacklist']['teams'].extend(teams)
            self.config['blacklist']['teams'] = list(set(self.config['blacklist']['teams']))
        self.save_config()

