"""
Sistema Multi-Fonte per Trovare Partite
========================================

Combina multiple fonti per trovare più partite, incluse leghe minori e altre nazionalità:
1. TheOddsAPI - Partite con quote
2. API-SPORTS - Oltre 2000 competizioni (leghe minori incluse)
3. Football-Data.org - Leghe europee principali
4. Sistema combinato per massima copertura
"""

import logging
import os
import urllib.request
import urllib.parse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import time

logger = logging.getLogger(__name__)


class MultiSourceMatchFinder:
    """
    Trova partite da multiple fonti per massima copertura.
    """
    
    def __init__(self):
        # 🆕 Prova multiple variabili d'ambiente per API-SPORTS
        self.api_sports_key = os.getenv("API_FOOTBALL_KEY", "") or os.getenv("RAPIDAPI_KEY", "") or "95c43f936816cd4389a747fd2cfe061a"
        self.football_data_key = os.getenv("FOOTBALL_DATA_KEY", "")
        self.theodds_key = os.getenv("THEODDS_API_KEY", "")
        
        # Cache per evitare duplicati
        self.found_match_ids: Set[str] = set()
        
        # Log per debug
        if self.api_sports_key:
            logger.debug(f"✅ API-SPORTS key configurata: {self.api_sports_key[:15]}...")
        else:
            logger.warning("⚠️  API-SPORTS key non trovata")
    
    def find_all_matches(
        self,
        days_ahead: int = 1,
        include_minor_leagues: bool = True,
        countries: Optional[List[str]] = None,
        include_live: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Trova partite da tutte le fonti disponibili.
        
        Args:
            days_ahead: Quanti giorni in avanti cercare (default: 1 = oggi)
            include_minor_leagues: Includere leghe minori (default: True)
            countries: Lista paesi da includere (None = tutti)
            include_live: Includere partite live (default: True)
        
        Returns:
            Lista di partite uniche (senza duplicati)
        """
        all_matches = []
        self.found_match_ids.clear()
        
        # 🆕 OTTIMIZZAZIONE: Usa API-SPORTS come primario (7500 chiamate/giorno)
        # TheOddsAPI solo come supplemento per quote (500 chiamate/mese limitate)
        
        # 1. API-SPORTS PRIMA (massima copertura, leghe minori, budget generoso)
        if self.api_sports_key:
            logger.info("📡 Cercando partite da API-SPORTS (primario - 7500 chiamate/giorno)...")
            api_sports_matches = self._fetch_from_api_sports(days_ahead, include_minor_leagues, countries)
            all_matches.extend(api_sports_matches)
            logger.info(f"   ✅ Trovate {len(api_sports_matches)} partite da API-SPORTS")
            
            # 🆕 NUOVO: Cerca anche partite LIVE se richiesto
            if include_live:
                logger.info("📡 Cercando partite LIVE da API-SPORTS...")
                live_matches = self._fetch_live_from_api_sports(include_minor_leagues, countries)
                all_matches.extend(live_matches)
                logger.info(f"   ✅ Trovate {len(live_matches)} partite LIVE da API-SPORTS")
        else:
            logger.warning("⚠️  API-SPORTS non configurata, salto questa fonte")
        
        # 2. TheOddsAPI SOLO se necessario (per quote aggiuntive, budget limitato: 500/mese = ~20/giorno)
        # Strategia conservativa: usa solo quando API-SPORTS ha trovato pochissime partite
        # Limita a massimo 1 chiamata ogni 2-3 cicli per rispettare budget di 20 chiamate/giorno
        use_theodds = False
        if len(all_matches) < 5:  # Solo se API-SPORTS ha trovato molto poche partite (< 5)
            use_theodds = True
            logger.info("📡 API-SPORTS ha trovato pochissime partite (<5), integrando con TheOddsAPI per quote...")
        # NON usare TheOddsAPI per quote aggiuntive se API-SPORTS ha già trovato partite sufficienti
        # Questo riduce l'uso a ~5-10 chiamate/giorno invece di 20+
        
        if use_theodds and self.theodds_key:
            try:
                theodds_matches = self._fetch_from_theodds()
                all_matches.extend(theodds_matches)
                logger.info(f"   ✅ Trovate {len(theodds_matches)} partite aggiuntive da TheOddsAPI")
            except Exception as e:
                logger.debug(f"⚠️  TheOddsAPI non disponibile: {e} (non critico, API-SPORTS è primario)")
        else:
            logger.debug("ℹ️  TheOddsAPI saltata (API-SPORTS è sufficiente o budget limitato)")
        
        # 3. Football-Data.org (leghe europee principali)
        if self.football_data_key:
            logger.info("📡 Cercando partite da Football-Data.org...")
            football_data_matches = self._fetch_from_football_data(days_ahead, countries)
            all_matches.extend(football_data_matches)
            logger.info(f"   ✅ Trovate {len(football_data_matches)} partite da Football-Data.org")
        else:
            logger.debug("ℹ️  Football-Data.org non configurata, salto questa fonte")
        
        # Rimuovi duplicati e ordina
        unique_matches = self._deduplicate_matches(all_matches)
        
        # Conta partite live
        live_count = sum(1 for m in unique_matches if m.get('is_live'))
        
        logger.info(f"📊 Totale partite uniche trovate: {len(unique_matches)}")
        logger.info(f"🎯 Partite LIVE: {live_count}")
        return unique_matches
    
    def _fetch_from_theodds(self) -> List[Dict[str, Any]]:
        """Recupera partite da TheOddsAPI"""
        if not self.theodds_key:
            return []
        
        try:
            url = "https://api.the-odds-api.com/v4/sports/soccer/odds"
            params = {
                "apiKey": self.theodds_key,
                "regions": "eu",
                "markets": "h2h",
                "oddsFormat": "decimal"
            }
            
            query = urllib.parse.urlencode(params)
            full_url = f"{url}?{query}"
            
            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                matches = []
                now = datetime.now()
                max_future = now + timedelta(hours=24)
                min_past = now - timedelta(hours=3)
                
                for event in data:
                    try:
                        commence_time_str = event.get("commence_time")
                        if not commence_time_str:
                            continue
                        
                        commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
                        commence_time_local = commence_time.replace(tzinfo=None)
                        
                        # Filtra per tempo
                        if commence_time_local > max_future:
                            continue
                        if commence_time_local < min_past:
                            continue
                        
                        # Estrai quote
                        bookmakers = event.get("bookmakers", [])
                        best_odds = {"home": None, "draw": None, "away": None}
                        
                        for bookmaker in bookmakers:
                            markets = bookmaker.get("markets", [])
                            h2h_market = next((m for m in markets if m.get("key") == "h2h"), None)
                            if not h2h_market:
                                continue
                            
                            outcomes = h2h_market.get("outcomes", [])
                            home_team = event.get("home_team", "").lower()
                            away_team = event.get("away_team", "").lower()
                            
                            for outcome in outcomes:
                                name = outcome.get("name", "").lower()
                                price = outcome.get("price")
                                
                                if price is None:
                                    continue
                                
                                if name == home_team:
                                    if best_odds["home"] is None or price > best_odds["home"]:
                                        best_odds["home"] = price
                                elif name == away_team:
                                    if best_odds["away"] is None or price > best_odds["away"]:
                                        best_odds["away"] = price
                                elif name in ["draw", "pareggio", "x"]:
                                    if best_odds["draw"] is None or price > best_odds["draw"]:
                                        best_odds["draw"] = price
                        
                        if best_odds["home"] and best_odds["away"] and best_odds["draw"]:
                            matches.append({
                                'id': f"theodds_{event.get('id', '')}",
                                'home': event.get("home_team", ""),
                                'away': event.get("away_team", ""),
                                'league': event.get("sport_title", "Soccer"),
                                'date': commence_time_local,
                                'odds_1': best_odds["home"],
                                'odds_x': best_odds["draw"],
                                'odds_2': best_odds["away"],
                                'source': 'theodds',
                                'is_live': commence_time_local < now
                            })
                    except Exception as e:
                        logger.debug(f"⚠️  Errore processing TheOddsAPI event: {e}")
                        continue
                
                return matches
        except Exception as e:
            logger.error(f"❌ Errore TheOddsAPI: {e}")
            return []
    
    def _fetch_from_api_sports(
        self,
        days_ahead: int,
        include_minor_leagues: bool,
        countries: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Recupera partite da API-SPORTS.
        Supporta oltre 2000 competizioni incluse leghe minori.
        """
        if not self.api_sports_key:
            return []
        
        matches = []
        now = datetime.now()
        target_date = now + timedelta(days=days_ahead)
        
        try:
            # API-SPORTS endpoint per fixtures
            # Possiamo cercare per data o per competizione
            url = "https://v3.football.api-sports.io/fixtures"
            
            # Cerca partite per data
            date_str = target_date.strftime("%Y-%m-%d")
            params = {
                "date": date_str
            }
            
            # Se specificati paesi, filtra per competizioni di quei paesi
            # (API-SPORTS supporta filtri per country)
            if countries:
                # Per ora prendiamo tutte, poi filtriamo
                pass
            
            query = urllib.parse.urlencode(params)
            full_url = f"{url}?{query}"
            
            headers = {
                "x-rapidapi-key": self.api_sports_key,
                "x-rapidapi-host": "v3.football.api-sports.io"
            }
            
            req = urllib.request.Request(full_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
                
                if not data.get("response"):
                    return []
                
                for fixture_data in data["response"]:
                    try:
                        fixture = fixture_data.get("fixture", {})
                        teams = fixture_data.get("teams", {})
                        league = fixture_data.get("league", {})
                        
                        home_team = teams.get("home", {}).get("name", "")
                        away_team = teams.get("away", {}).get("name", "")
                        league_name = league.get("name", "")
                        country = league.get("country", "")
                        league_id = league.get("id", "")
                        
                        if not home_team or not away_team:
                            continue
                        
                        # Filtra per leghe minori se richiesto
                        if not include_minor_leagues:
                            # Escludi leghe minori (puoi personalizzare questa logica)
                            minor_league_keywords = ["U19", "U21", "Youth", "Reserve", "B Team"]
                            if any(keyword in league_name for keyword in minor_league_keywords):
                                continue
                        
                        # Filtra per paese se specificato
                        if countries and country not in countries:
                            continue
                        
                        # Estrai data
                        fixture_date_str = fixture.get("date", "")
                        if not fixture_date_str:
                            continue
                        
                        try:
                            fixture_date = datetime.fromisoformat(fixture_date_str.replace("Z", "+00:00"))
                            fixture_date = fixture_date.replace(tzinfo=None)
                        except:
                            continue
                        
                        # Estrai quote se disponibili
                        odds_data = fixture_data.get("odds", {})
                        odds_1 = None
                        odds_x = None
                        odds_2 = None
                        
                        if odds_data:
                            # API-SPORTS può avere quote in diversi formati
                            # Per ora usiamo valori di default
                            pass
                        
                        # Crea match
                        match_id = f"apisports_{fixture.get('id', '')}"
                        matches.append({
                            'id': match_id,
                            'home': home_team,
                            'away': away_team,
                            'league': league_name,
                            'country': country,
                            'date': fixture_date,
                            'odds_1': odds_1 or 2.0,  # Default se non disponibili
                            'odds_x': odds_x or 3.0,
                            'odds_2': odds_2 or 2.0,
                            'source': 'api-sports',
                            'league_id': league_id,
                            'is_live': fixture.get("status", {}).get("long") == "Match Finished" or False
                        })
                    except Exception as e:
                        logger.debug(f"⚠️  Errore processing API-SPORTS fixture: {e}")
                        continue
                
                return matches
        except Exception as e:
            logger.error(f"❌ Errore API-SPORTS: {e}")
            return []
    
    def _fetch_live_from_api_sports(
        self,
        include_minor_leagues: bool,
        countries: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Recupera partite LIVE da API-SPORTS.
        """
        if not self.api_sports_key:
            return []
        
        matches = []
        
        try:
            # API-SPORTS endpoint per partite live
            url = "https://v3.football.api-sports.io/fixtures"
            params = {
                "live": "all"  # Tutte le partite live
            }
            
            query = urllib.parse.urlencode(params)
            full_url = f"{url}?{query}"
            
            headers = {
                "x-rapidapi-key": self.api_sports_key,
                "x-rapidapi-host": "v3.football.api-sports.io"
            }
            
            req = urllib.request.Request(full_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
                
                if not data.get("response"):
                    return []
                
                for fixture_data in data["response"]:
                    try:
                        fixture = fixture_data.get("fixture", {})
                        teams = fixture_data.get("teams", {})
                        league = fixture_data.get("league", {})
                        
                        home_team = teams.get("home", {}).get("name", "")
                        away_team = teams.get("away", {}).get("name", "")
                        league_name = league.get("name", "")
                        country = league.get("country", "")
                        league_id = league.get("id", "")
                        
                        if not home_team or not away_team:
                            continue
                        
                        # Filtra per leghe minori se richiesto
                        if not include_minor_leagues:
                            minor_league_keywords = ["U19", "U21", "Youth", "Reserve", "B Team"]
                            if any(keyword in league_name for keyword in minor_league_keywords):
                                continue
                        
                        # Filtra per paese se specificato
                        if countries and country not in countries:
                            continue
                        
                        # Estrai score e minuto
                        score = fixture_data.get("goals", {})
                        score_home = score.get("home", 0)
                        score_away = score.get("away", 0)
                        
                        # 🔧 LOG per debug: verifica score recuperato
                        if score_home > 0 or score_away > 0:
                            logger.info(f"📊 LIVE {home_team} vs {away_team}: {score_home}-{score_away} (min {minute})")
                        
                        fixture_status = fixture.get("status", {})
                        minute = fixture_status.get("elapsed", 0)
                        status_long = fixture_status.get("long", "Live")
                        
                        # 🔧 Estrai statistiche live se disponibili
                        # NOTA: Le statistiche potrebbero non essere incluse nell'endpoint /fixtures
                        # Potrebbero richiedere una chiamata separata all'endpoint /statistics
                        statistics = fixture_data.get("statistics", [])
                        
                        # 🔧 LOG: Verifica se le statistiche sono presenti
                        if not statistics:
                            logger.debug(f"⚠️  Statistiche non presenti in fixture_data per {home_team} vs {away_team} (fixture_id: {fixture.get('id', 'N/A')})")
                            # Le statistiche potrebbero essere in un campo diverso o richiedere un endpoint separato
                        
                        home_shots_on_target = 0
                        away_shots_on_target = 0
                        home_total_shots = 0
                        away_total_shots = 0
                        home_xg = 0.0
                        away_xg = 0.0
                        home_dangerous_attacks = 0
                        away_dangerous_attacks = 0
                        
                        # 🔧 PROVA: Cerca statistiche anche in altri campi possibili
                        # Alcune API potrebbero mettere le statistiche in fixture_data direttamente
                        if not statistics:
                            # Prova a cercare in altri formati possibili
                            if "statistics" in fixture_data:
                                statistics = fixture_data["statistics"]
                            elif "stats" in fixture_data:
                                statistics = fixture_data["stats"]
                        
                        # API-SPORTS restituisce statistiche come array di oggetti per home/away
                        if statistics and len(statistics) >= 2:
                            home_stats = statistics[0].get("statistics", [])
                            away_stats = statistics[1].get("statistics", [])
                            
                            # 🔧 LOG: Verifica statistiche disponibili
                            logger.debug(f"📊 Statistiche disponibili per {home_team} vs {away_team}: {len(home_stats)} home, {len(away_stats)} away")
                            
                            # Estrai statistiche home
                            for stat in home_stats:
                                stat_type = stat.get("type", "")
                                value = stat.get("value")
                                # 🔧 FIX: Prova anche varianti del nome (API-SPORTS può usare nomi diversi)
                                if stat_type in ["Shots on Goal", "Shots on Target", "Shots On Target"]:
                                    try:
                                        home_shots_on_target = int(value) if value is not None else 0
                                    except:
                                        home_shots_on_target = 0
                                elif stat_type in ["Total Shots", "Shots Total", "Shots"]:
                                    try:
                                        home_total_shots = int(value) if value is not None else 0
                                    except:
                                        home_total_shots = 0
                                elif stat_type in ["Expected Goals", "xG", "Expected goals"]:
                                    try:
                                        home_xg = float(value) if value is not None else 0.0
                                    except:
                                        home_xg = 0.0
                                elif stat_type in ["Dangerous Attacks", "Dangerous attacks", "Attacks"]:
                                    try:
                                        home_dangerous_attacks = int(value) if value is not None else 0
                                    except:
                                        home_dangerous_attacks = 0
                            
                            # Estrai statistiche away
                            for stat in away_stats:
                                stat_type = stat.get("type", "")
                                value = stat.get("value")
                                # 🔧 FIX: Prova anche varianti del nome
                                if stat_type in ["Shots on Goal", "Shots on Target", "Shots On Target"]:
                                    try:
                                        away_shots_on_target = int(value) if value is not None else 0
                                    except:
                                        away_shots_on_target = 0
                                elif stat_type in ["Total Shots", "Shots Total", "Shots"]:
                                    try:
                                        away_total_shots = int(value) if value is not None else 0
                                    except:
                                        away_total_shots = 0
                                elif stat_type in ["Expected Goals", "xG", "Expected goals"]:
                                    try:
                                        away_xg = float(value) if value is not None else 0.0
                                    except:
                                        away_xg = 0.0
                                elif stat_type in ["Dangerous Attacks", "Dangerous attacks", "Attacks"]:
                                    try:
                                        away_dangerous_attacks = int(value) if value is not None else 0
                                    except:
                                        away_dangerous_attacks = 0
                            
                            # 🔧 LOG: Verifica statistiche estratte
                            if home_shots_on_target > 0 or away_shots_on_target > 0:
                                logger.info(f"✅ Statistiche estratte per {home_team} vs {away_team}: SOT {home_shots_on_target}-{away_shots_on_target}, Shots {home_total_shots}-{away_total_shots}, xG {home_xg:.2f}-{away_xg:.2f}")
                        else:
                            logger.debug(f"⚠️  Statistiche non disponibili o incomplete per {home_team} vs {away_team}")
                        
                        # Estrai data (per deduplicazione)
                        fixture_date_str = fixture.get("date", "")
                        if not fixture_date_str:
                            continue
                        
                        try:
                            fixture_date = datetime.fromisoformat(fixture_date_str.replace("Z", "+00:00"))
                            fixture_date = fixture_date.replace(tzinfo=None)
                        except:
                            continue
                        
                        # Crea match
                        match_id = f"apisports_live_{fixture.get('id', '')}"
                        fixture_id = fixture.get('id', '')
                        
                        # 🔧 NUOVO: Se le statistiche non sono disponibili, fai una chiamata separata
                        if not statistics and fixture_id:
                            statistics = self._fetch_statistics_from_api_sports(fixture_id)
                            if statistics:
                                # Estrai statistiche dalla chiamata separata
                                home_stats = statistics[0].get("statistics", []) if len(statistics) > 0 else []
                                away_stats = statistics[1].get("statistics", []) if len(statistics) > 1 else []
                                
                                # Estrai statistiche home
                                for stat in home_stats:
                                    stat_type = stat.get("type", "")
                                    value = stat.get("value")
                                    if stat_type in ["Shots on Goal", "Shots on Target", "Shots On Target"]:
                                        try:
                                            home_shots_on_target = int(value) if value is not None else 0
                                        except:
                                            home_shots_on_target = 0
                                    elif stat_type in ["Total Shots", "Shots Total", "Shots"]:
                                        try:
                                            home_total_shots = int(value) if value is not None else 0
                                        except:
                                            home_total_shots = 0
                                    elif stat_type in ["Expected Goals", "xG", "Expected goals"]:
                                        try:
                                            home_xg = float(value) if value is not None else 0.0
                                        except:
                                            home_xg = 0.0
                                    elif stat_type in ["Dangerous Attacks", "Dangerous attacks", "Attacks"]:
                                        try:
                                            home_dangerous_attacks = int(value) if value is not None else 0
                                        except:
                                            home_dangerous_attacks = 0
                                
                                # Estrai statistiche away
                                for stat in away_stats:
                                    stat_type = stat.get("type", "")
                                    value = stat.get("value")
                                    if stat_type in ["Shots on Goal", "Shots on Target", "Shots On Target"]:
                                        try:
                                            away_shots_on_target = int(value) if value is not None else 0
                                        except:
                                            away_shots_on_target = 0
                                    elif stat_type in ["Total Shots", "Shots Total", "Shots"]:
                                        try:
                                            away_total_shots = int(value) if value is not None else 0
                                        except:
                                            away_total_shots = 0
                                    elif stat_type in ["Expected Goals", "xG", "Expected goals"]:
                                        try:
                                            away_xg = float(value) if value is not None else 0.0
                                        except:
                                            away_xg = 0.0
                                    elif stat_type in ["Dangerous Attacks", "Dangerous attacks", "Attacks"]:
                                        try:
                                            away_dangerous_attacks = int(value) if value is not None else 0
                                        except:
                                            away_dangerous_attacks = 0
                                
                                if home_shots_on_target > 0 or away_shots_on_target > 0:
                                    logger.info(f"✅ Statistiche ottenute da endpoint separato per {home_team} vs {away_team}: SOT {home_shots_on_target}-{away_shots_on_target}, Shots {home_total_shots}-{away_total_shots}, xG {home_xg:.2f}-{away_xg:.2f}")
                        
                        matches.append({
                            'id': match_id,
                            'home': home_team,
                            'away': away_team,
                            'league': league_name,
                            'country': country,
                            'date': fixture_date,
                            'odds_1': 2.0,  # Default, quote live non sempre disponibili
                            'odds_x': 3.0,
                            'odds_2': 2.0,
                            'source': 'api-sports',
                            'league_id': league_id,
                            'is_live': True,  # Marca come live
                            'score_home': score_home,
                            'score_away': score_away,
                            'minute': minute,
                            'status': status_long,
                            # 🔧 Statistiche live
                            'home_shots_on_target': home_shots_on_target,
                            'away_shots_on_target': away_shots_on_target,
                            'home_total_shots': home_total_shots,
                            'away_total_shots': away_total_shots,
                            'home_xg': home_xg,
                            'away_xg': away_xg,
                            'home_dangerous_attacks': home_dangerous_attacks,
                            'away_dangerous_attacks': away_dangerous_attacks
                        })
                    except Exception as e:
                        logger.debug(f"⚠️  Errore processing API-SPORTS live fixture: {e}")
                        continue
                
                return matches
        except Exception as e:
            logger.error(f"❌ Errore API-SPORTS live: {e}")
            return []
    
    def _fetch_statistics_from_api_sports(self, fixture_id: int) -> Optional[List[Dict[str, Any]]]:
        """
        Recupera statistiche per una specifica partita da API-SPORTS.
        Endpoint separato: /fixtures/statistics
        """
        if not self.api_sports_key or not fixture_id:
            return None
        
        try:
            url = "https://v3.football.api-sports.io/fixtures/statistics"
            params = {
                "fixture": str(fixture_id)
            }
            
            query = urllib.parse.urlencode(params)
            full_url = f"{url}?{query}"
            
            headers = {
                "x-rapidapi-key": self.api_sports_key,
                "x-rapidapi-host": "v3.football.api-sports.io"
            }
            
            req = urllib.request.Request(full_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                if not data.get("response"):
                    return None
                
                # API-SPORTS restituisce un array con statistiche per home e away
                statistics = data.get("response", [])
                if len(statistics) >= 2:
                    logger.debug(f"✅ Statistiche ottenute da endpoint separato per fixture {fixture_id}")
                    return statistics
                else:
                    logger.debug(f"⚠️  Statistiche incomplete per fixture {fixture_id}")
                    return None
                    
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.warning(f"⚠️  Rate limit raggiunto per statistiche fixture {fixture_id}")
            else:
                logger.debug(f"⚠️  Errore HTTP ottenendo statistiche per fixture {fixture_id}: {e.code}")
            return None
        except Exception as e:
            logger.debug(f"⚠️  Errore ottenendo statistiche per fixture {fixture_id}: {e}")
            return None
    
    def _fetch_from_football_data(
        self,
        days_ahead: int,
        countries: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Recupera partite da Football-Data.org.
        Copre principalmente leghe europee principali.
        """
        if not self.football_data_key:
            return []
        
        matches = []
        now = datetime.now()
        
        try:
            # Football-Data.org endpoint per matches
            # Cerca nelle competizioni principali
            url = "https://api.football-data.org/v4/matches"
            
            # Cerca partite per date range
            date_from = now.strftime("%Y-%m-%d")
            date_to = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            
            params = {
                "dateFrom": date_from,
                "dateTo": date_to
            }
            
            query = urllib.parse.urlencode(params)
            full_url = f"{url}?{query}"
            
            headers = {
                "X-Auth-Token": self.football_data_key
            }
            
            req = urllib.request.Request(full_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                if not data.get("matches"):
                    return []
                
                for match_data in data["matches"]:
                    try:
                        home_team = match_data.get("homeTeam", {}).get("name", "")
                        away_team = match_data.get("awayTeam", {}).get("name", "")
                        competition = match_data.get("competition", {})
                        league_name = competition.get("name", "")
                        country = competition.get("area", {}).get("name", "")
                        
                        if not home_team or not away_team:
                            continue
                        
                        # Filtra per paese se specificato
                        if countries and country not in countries:
                            continue
                        
                        # Estrai data
                        utc_date_str = match_data.get("utcDate", "")
                        if not utc_date_str:
                            continue
                        
                        try:
                            match_date = datetime.fromisoformat(utc_date_str.replace("Z", "+00:00"))
                            match_date = match_date.replace(tzinfo=None)
                        except:
                            continue
                        
                        # Estrai quote se disponibili
                        odds = match_data.get("odds", {})
                        odds_1 = odds.get("homeWin") if odds else None
                        odds_x = odds.get("draw") if odds else None
                        odds_2 = odds.get("awayWin") if odds else None
                        
                        match_id = f"footballdata_{match_data.get('id', '')}"
                        matches.append({
                            'id': match_id,
                            'home': home_team,
                            'away': away_team,
                            'league': league_name,
                            'country': country,
                            'date': match_date,
                            'odds_1': odds_1 or 2.0,
                            'odds_x': odds_x or 3.0,
                            'odds_2': odds_2 or 2.0,
                            'source': 'football-data',
                            'is_live': match_data.get("status") == "LIVE"
                        })
                    except Exception as e:
                        logger.debug(f"⚠️  Errore processing Football-Data match: {e}")
                        continue
                
                return matches
        except Exception as e:
            logger.error(f"❌ Errore Football-Data.org: {e}")
            return []
    
    def _deduplicate_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rimuove duplicati basandosi su squadre e data.
        Mantiene la partita con più informazioni (quote, ecc.)
        """
        unique_matches = {}
        
        for match in matches:
            home = match.get('home', '').lower().strip()
            away = match.get('away', '').lower().strip()
            match_date = match.get('date')
            
            if not home or not away or not match_date:
                continue
            
            # Crea chiave unica
            if isinstance(match_date, datetime):
                date_key = match_date.strftime("%Y-%m-%d %H:%M")
            else:
                date_key = str(match_date)
            
            key = f"{home}_{away}_{date_key}"
            
            # Se già esiste, mantieni quello con più info (quote, ecc.)
            if key in unique_matches:
                existing = unique_matches[key]
                # Preferisci quello con quote valide
                if match.get('odds_1') and not existing.get('odds_1'):
                    unique_matches[key] = match
                # Preferisci TheOddsAPI (ha sempre quote)
                elif match.get('source') == 'theodds' and existing.get('source') != 'theodds':
                    unique_matches[key] = match
            else:
                unique_matches[key] = match
        
        # Ordina per data
        result = list(unique_matches.values())
        result.sort(key=lambda x: x.get('date', datetime.min))
        
        return result
    
    def get_leagues_available(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Restituisce lista competizioni disponibili per fonte.
        Utile per vedere quali leghe sono coperte.
        """
        leagues = {
            'api-sports': [],
            'football-data': [],
            'theodds': []
        }
        
        # API-SPORTS: lista competizioni
        if self.api_sports_key:
            try:
                url = "https://v3.football.api-sports.io/leagues"
                headers = {
                    "x-rapidapi-key": self.api_sports_key,
                    "x-rapidapi-host": "v3.football.api-sports.io"
                }
                
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = json.loads(response.read().decode())
                    if data.get("response"):
                        for league_data in data["response"][:100]:  # Prime 100
                            league = league_data.get("league", {})
                            country = league_data.get("country", {})
                            leagues['api-sports'].append({
                                'name': league.get("name", ""),
                                'country': country.get("name", ""),
                                'id': league.get("id", "")
                            })
            except Exception as e:
                logger.debug(f"⚠️  Errore recupero leghe API-SPORTS: {e}")
        
        # Football-Data.org: lista competizioni
        if self.football_data_key:
            try:
                url = "https://api.football-data.org/v4/competitions"
                headers = {"X-Auth-Token": self.football_data_key}
                
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())
                    if data.get("competitions"):
                        for comp in data["competitions"]:
                            leagues['football-data'].append({
                                'name': comp.get("name", ""),
                                'country': comp.get("area", {}).get("name", ""),
                                'id': comp.get("id", "")
                            })
            except Exception as e:
                logger.debug(f"⚠️  Errore recupero leghe Football-Data: {e}")
        
        return leagues

