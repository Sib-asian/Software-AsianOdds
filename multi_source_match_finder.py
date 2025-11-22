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

            # 🔧 DEBUG: Log della chiamata API
            logger.info(f"📡 Chiamando API-SPORTS fixtures per data {date_str}")
            logger.debug(f"   Full URL: {full_url}")

            req = urllib.request.Request(full_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                # 🔧 DEBUG: Log dello status code
                status_code = response.status
                logger.info(f"✅ API-SPORTS fixtures ha risposto con status code: {status_code}")

                response_body = response.read().decode()
                data = json.loads(response_body)

                # 🔧 DEBUG: Log della struttura della risposta
                logger.info(f"📊 Struttura risposta API-SPORTS fixtures: keys={list(data.keys())}")
                if "errors" in data and data["errors"]:
                    logger.error(f"❌ API-SPORTS fixtures ha restituito errori: {data['errors']}")
                if "results" in data:
                    logger.info(f"   results={data.get('results', 'N/A')}")

                if not data.get("response"):
                    logger.warning(f"⚠️  Nessuna partita restituita da API-SPORTS per data {date_str}")
                    logger.warning(f"   Response: {json.dumps(data, indent=2)[:500]}...")
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
                        
                        # 🔧 FIX: Determina correttamente se la partita è live
                        fixture_status = fixture.get("status", {})
                        status_long = fixture_status.get("long", "")
                        status_short = fixture_status.get("short", "")
                        
                        # Partita è LIVE se:
                        # - Status è "First Half", "Second Half", "Halftime", "Extra Time", "Penalties", "Break Time", "In Play"
                        # - Oppure se la data è passata e non è "Match Finished", "Not Started", "Time To Be Defined"
                        now = datetime.now()
                        is_live = False
                        if status_long in ["First Half", "Second Half", "Halftime", "Extra Time", "Penalties", "Break Time", "In Play", "Live"]:
                            is_live = True
                        elif status_long not in ["Match Finished", "Finished", "Not Started", "Time To Be Defined", "Match Postponed", "Match Cancelled", "Match Suspended", "Match Interrupted"]:
                            # Se lo status non è uno di questi, e la partita è iniziata, potrebbe essere live
                            if fixture_date < now:
                                is_live = True
                        
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
                            'is_live': is_live
                        })
                    except Exception as e:
                        logger.debug(f"⚠️  Errore processing API-SPORTS fixture: {e}")
                        continue
                
                return matches
        except urllib.error.HTTPError as e:
            # 🔧 DEBUG: Log dettagliato degli errori HTTP
            logger.error(f"❌ Errore HTTP API-SPORTS fixtures: {e.code} - {e.reason}")
            logger.error(f"   URL: {full_url}")
            try:
                error_body = e.read().decode()
                logger.error(f"   Error body: {error_body[:500]}")
            except:
                pass
            if e.code == 401:
                logger.error("❌ API key non valida o non autorizzata!")
            elif e.code == 429:
                logger.error("❌ Rate limit superato!")
            elif e.code == 403:
                logger.error("❌ Accesso negato! Verifica il tuo piano API-SPORTS")
            return []
        except urllib.error.URLError as e:
            logger.error(f"❌ Errore connessione API-SPORTS fixtures: {e.reason}")
            return []
        except Exception as e:
            logger.error(f"❌ Errore API-SPORTS fixtures: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
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

            # 🔧 DEBUG: Log della chiamata API
            logger.info(f"📡 Chiamando API-SPORTS live: {url}")
            logger.debug(f"   Headers: x-rapidapi-key={self.api_sports_key[:15]}..., x-rapidapi-host=v3.football.api-sports.io")
            logger.debug(f"   Full URL: {full_url}")

            req = urllib.request.Request(full_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                # 🔧 DEBUG: Log dello status code
                status_code = response.status
                logger.info(f"✅ API-SPORTS live ha risposto con status code: {status_code}")

                # 🔧 DEBUG: Log degli headers della risposta
                response_headers = dict(response.headers)
                logger.debug(f"   Response headers: {response_headers}")

                # 🔧 DEBUG: Leggi e logga la risposta raw
                response_body = response.read().decode()
                logger.debug(f"   Response body length: {len(response_body)} bytes")

                data = json.loads(response_body)

                # 🔧 DEBUG: Log della struttura della risposta
                logger.info(f"📊 Struttura risposta API-SPORTS live: keys={list(data.keys())}")
                if "errors" in data and data["errors"]:
                    logger.error(f"❌ API-SPORTS live ha restituito errori: {data['errors']}")
                if "results" in data:
                    logger.info(f"   results={data.get('results', 'N/A')}")
                if "paging" in data:
                    logger.info(f"   paging={data.get('paging', 'N/A')}")

                if not data.get("response"):
                    logger.warning(f"⚠️  Nessuna partita LIVE restituita da API-SPORTS")
                    logger.warning(f"   Response completa: {json.dumps(data, indent=2)[:500]}...")  # Prime 500 chars
                    return []

                logger.info(f"📡 API-SPORTS ha restituito {len(data.get('response', []))} partite LIVE")
                
                processed_count = 0
                filtered_count = 0
                
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
                            filtered_count += 1
                            logger.debug(f"⏭️  Partita LIVE saltata: team mancanti (home: {home_team}, away: {away_team})")
                            continue
                        
                        # Filtra per leghe minori se richiesto
                        if not include_minor_leagues:
                            minor_league_keywords = ["U19", "U21", "Youth", "Reserve", "B Team"]
                            if any(keyword in league_name for keyword in minor_league_keywords):
                                filtered_count += 1
                                logger.debug(f"⏭️  Partita LIVE saltata: lega minore ({league_name})")
                                continue
                        
                        # Filtra per paese se specificato
                        if countries and country not in countries:
                            filtered_count += 1
                            logger.debug(f"⏭️  Partita LIVE saltata: paese non incluso ({country})")
                            continue
                        
                        processed_count += 1
                        
                        # 🔧 FIX: Estrai data PRIMA dei filtri (serve per matches.append)
                        fixture_date_str = fixture.get("date", "")
                        if not fixture_date_str:
                            filtered_count += 1
                            logger.debug(f"⏭️  Partita LIVE saltata: data mancante ({home_team} vs {away_team})")
                            continue
                        
                        try:
                            fixture_date = datetime.fromisoformat(fixture_date_str.replace("Z", "+00:00"))
                            fixture_date = fixture_date.replace(tzinfo=None)
                        except:
                            filtered_count += 1
                            logger.debug(f"⏭️  Partita LIVE saltata: data non valida ({home_team} vs {away_team})")
                            continue
                        
                        # Estrai score e minuto
                        score = fixture_data.get("goals", {})
                        score_home = score.get("home", 0)
                        score_away = score.get("away", 0)
                        
                        fixture_status = fixture.get("status", {})
                        minute = fixture_status.get("elapsed", 0)
                        status_long = fixture_status.get("long", "Live")
                        status_short = fixture_status.get("short", "")
                        
                        # 🔧 LOG per debug: verifica score recuperato
                        if score_home > 0 or score_away > 0:
                            logger.info(f"📊 LIVE {home_team} vs {away_team}: {score_home}-{score_away} (min {minute})")
                        
                        # 🔧 FILTRO: Escludi partite finite
                        if status_long in ["Match Finished", "Finished"] or status_short in ["FT", "AET", "PEN"]:
                            filtered_count += 1
                            logger.debug(f"⏭️  Partita LIVE saltata: {home_team} vs {away_team} - partita finita (status: {status_long})")
                            continue
                        
                        # 🔧 FILTRO: Escludi partite con minuto > 90 (probabilmente finite)
                        if minute > 90:
                            filtered_count += 1
                            logger.debug(f"⏭️  Partita LIVE saltata: {home_team} vs {away_team} - minuto {minute} > 90 (probabilmente finita)")
                            continue
                        
                        # 🔧 LOG: Mostra tutte le partite live trovate
                        logger.info(f"✅ Partita LIVE trovata: {home_team} vs {away_team} - {score_home}-{score_away} (min {minute}, status: {status_long})")
                        
                        # 🔧 Estrai statistiche live se disponibili
                        # NOTA: Le statistiche potrebbero non essere incluse nell'endpoint /fixtures
                        # Potrebbero richiedere una chiamata separata all'endpoint /statistics
                        statistics = fixture_data.get("statistics", [])
                        
                        # 🔧 LOG: Verifica se le statistiche sono presenti
                        if not statistics:
                            logger.debug(f"⚠️  Statistiche non presenti in fixture_data per {home_team} vs {away_team} (fixture_id: {fixture.get('id', 'N/A')})")
                            # Le statistiche potrebbero essere in un campo diverso o richiedere un endpoint separato
                        else:
                            # 🔧 LOG DETTAGLIATO: Mostra tutte le statistiche disponibili per debug
                            logger.debug(f"📊 Statistiche presenti in fixture_data per {home_team} vs {away_team}: {len(statistics)} elementi")
                            if len(statistics) >= 2:
                                home_stats_list = statistics[0].get("statistics", [])
                                away_stats_list = statistics[1].get("statistics", [])
                                # 🔧 INFO: Mostra TUTTI i tipi di statistiche disponibili per verificare se API fornisce corner/cartellini/falli
                                all_home_types = [s.get('type', 'N/A') for s in home_stats_list]
                                all_away_types = [s.get('type', 'N/A') for s in away_stats_list]
                                logger.info(f"📊 Tipi statistiche disponibili per {home_team} vs {away_team}: {all_home_types}")
                                logger.debug(f"   Home stats types: {all_home_types[:10]}")
                                logger.debug(f"   Away stats types: {all_away_types[:10]}")
                        
                        home_shots_on_target = 0
                        away_shots_on_target = 0
                        home_total_shots = 0
                        away_total_shots = 0
                        home_xg = 0.0
                        away_xg = 0.0
                        home_dangerous_attacks = 0
                        away_dangerous_attacks = 0
                        home_possession = None
                        away_possession = None
                        # 🔧 NUOVO: Statistiche aggiuntive (priorità 1-4)
                        home_corners = 0
                        away_corners = 0
                        home_yellow_cards = 0
                        away_yellow_cards = 0
                        home_red_cards = 0
                        away_red_cards = 0
                        home_fouls = 0
                        away_fouls = 0
                        
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
                                elif stat_type in ["Ball Possession", "Possession", "Ball possession"]:
                                    try:
                                        # Il possesso può essere una percentuale (es. "65%") o un numero
                                        if isinstance(value, str):
                                            # Rimuovi il simbolo % e converti
                                            home_possession = float(value.replace("%", "").strip())
                                        else:
                                            home_possession = float(value) if value is not None else None
                                    except:
                                        home_possession = None
                                # 🔧 PRIORITÀ 1: Cartellini rossi (per filtri anti-banali)
                                elif stat_type in ["Red Cards", "Red cards", "Red"]:
                                    try:
                                        home_red_cards = int(value) if value is not None else 0
                                    except:
                                        home_red_cards = 0
                                # 🔧 PRIORITÀ 2: Corner (per mercati corner)
                                elif stat_type in ["Corner Kicks", "Corner kicks", "Corners", "Corner"]:
                                    try:
                                        home_corners = int(value) if value is not None else 0
                                    except:
                                        home_corners = 0
                                # 🔧 PRIORITÀ 3: Cartellini gialli (per mercati cartellini)
                                elif stat_type in ["Yellow Cards", "Yellow cards", "Yellow"]:
                                    try:
                                        home_yellow_cards = int(value) if value is not None else 0
                                    except:
                                        home_yellow_cards = 0
                                # 🔧 PRIORITÀ 4: Falli (per contesto partita)
                                elif stat_type in ["Fouls", "Total Fouls", "Foul"]:
                                    try:
                                        home_fouls = int(value) if value is not None else 0
                                    except:
                                        home_fouls = 0
                            
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
                                elif stat_type in ["Ball Possession", "Possession", "Ball possession"]:
                                    try:
                                        # Il possesso può essere una percentuale (es. "65%") o un numero
                                        if isinstance(value, str):
                                            # Rimuovi il simbolo % e converti
                                            away_possession = float(value.replace("%", "").strip())
                                        else:
                                            away_possession = float(value) if value is not None else None
                                    except:
                                        away_possession = None
                                # 🔧 PRIORITÀ 1: Cartellini rossi (per filtri anti-banali)
                                elif stat_type in ["Red Cards", "Red cards", "Red"]:
                                    try:
                                        away_red_cards = int(value) if value is not None else 0
                                    except:
                                        away_red_cards = 0
                                # 🔧 PRIORITÀ 2: Corner (per mercati corner)
                                elif stat_type in ["Corner Kicks", "Corner kicks", "Corners", "Corner"]:
                                    try:
                                        away_corners = int(value) if value is not None else 0
                                    except:
                                        away_corners = 0
                                # 🔧 PRIORITÀ 3: Cartellini gialli (per mercati cartellini)
                                elif stat_type in ["Yellow Cards", "Yellow cards", "Yellow"]:
                                    try:
                                        away_yellow_cards = int(value) if value is not None else 0
                                    except:
                                        away_yellow_cards = 0
                                # 🔧 PRIORITÀ 4: Falli (per contesto partita)
                                elif stat_type in ["Fouls", "Total Fouls", "Foul"]:
                                    try:
                                        away_fouls = int(value) if value is not None else 0
                                    except:
                                        away_fouls = 0
                            
                            # 🔧 LOG: Verifica statistiche estratte
                            if home_shots_on_target > 0 or away_shots_on_target > 0 or home_possession is not None:
                                poss_str = f", Possesso {home_possession:.0f}%-{away_possession:.0f}%" if home_possession is not None else ""
                                logger.info(f"✅ Statistiche estratte per {home_team} vs {away_team}: SOT {home_shots_on_target}-{away_shots_on_target}, Shots {home_total_shots}-{away_total_shots}, xG {home_xg:.2f}-{away_xg:.2f}{poss_str}")
                            else:
                                logger.debug(f"⚠️  Statistiche tutte a 0 per {home_team} vs {away_team} - verifica se API fornisce statistiche per questa partita")
                        else:
                            logger.debug(f"⚠️  Statistiche non disponibili o incomplete per {home_team} vs {away_team} (statistics array vuoto o < 2 elementi)")
                        
                        # Crea match (fixture_date già estratto prima)
                        match_id = f"apisports_live_{fixture.get('id', '')}"
                        fixture_id = fixture.get('id', '')
                        
                        # 🔧 NUOVO: Se le statistiche non sono disponibili O sono tutte a 0, fai una chiamata separata
                        # Questo è importante per partite importanti come Champions League che dovrebbero avere statistiche
                        should_fetch_separate = False
                        if not statistics or len(statistics) < 2:
                            should_fetch_separate = True
                            logger.debug(f"📡 Statistiche non presenti o incomplete in /fixtures (len={len(statistics) if statistics else 0}), richiamo endpoint /statistics per fixture {fixture_id}")
                        elif len(statistics) >= 2:
                            # Verifica se le statistiche sono tutte a 0 (potrebbero non essere aggiornate)
                            home_stats = statistics[0].get("statistics", [])
                            has_any_stats = False
                            for stat in home_stats:
                                stat_type = stat.get("type", "")
                                value = stat.get("value")
                                if stat_type in ["Shots on Goal", "Shots on Target", "Total Shots", "Ball Possession"]:
                                    try:
                                        if isinstance(value, str):
                                            val = float(value.replace("%", "").strip())
                                        else:
                                            val = float(value) if value is not None else 0
                                        if val > 0:
                                            has_any_stats = True
                                            break
                                    except:
                                        pass
                            if not has_any_stats and minute >= 4:
                                # Se siamo oltre il 4° minuto e non ci sono statistiche, richiama endpoint separato
                                should_fetch_separate = True
                                logger.debug(f"📡 Statistiche presenti ma tutte a 0 al minuto {minute}, richiamo endpoint /statistics per fixture {fixture_id}")
                        
                        if should_fetch_separate and fixture_id:
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
                                    elif stat_type in ["Ball Possession", "Possession", "Ball possession"]:
                                        try:
                                            if isinstance(value, str):
                                                home_possession = float(value.replace("%", "").strip())
                                            else:
                                                home_possession = float(value) if value is not None else None
                                        except:
                                            home_possession = None
                                    # 🔧 PRIORITÀ 1: Cartellini rossi (per filtri anti-banali)
                                    elif stat_type in ["Red Cards", "Red cards", "Red"]:
                                        try:
                                            home_red_cards = int(value) if value is not None else 0
                                        except:
                                            home_red_cards = 0
                                    # 🔧 PRIORITÀ 2: Corner (per mercati corner)
                                    elif stat_type in ["Corner Kicks", "Corner kicks", "Corners", "Corner"]:
                                        try:
                                            home_corners = int(value) if value is not None else 0
                                        except:
                                            home_corners = 0
                                    # 🔧 PRIORITÀ 3: Cartellini gialli (per mercati cartellini)
                                    elif stat_type in ["Yellow Cards", "Yellow cards", "Yellow"]:
                                        try:
                                            home_yellow_cards = int(value) if value is not None else 0
                                        except:
                                            home_yellow_cards = 0
                                    # 🔧 PRIORITÀ 4: Falli (per contesto partita)
                                    elif stat_type in ["Fouls", "Total Fouls", "Foul"]:
                                        try:
                                            home_fouls = int(value) if value is not None else 0
                                        except:
                                            home_fouls = 0
                                
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
                                    elif stat_type in ["Ball Possession", "Possession", "Ball possession"]:
                                        try:
                                            if isinstance(value, str):
                                                away_possession = float(value.replace("%", "").strip())
                                            else:
                                                away_possession = float(value) if value is not None else None
                                        except:
                                            away_possession = None
                                    # 🔧 PRIORITÀ 1: Cartellini rossi (per filtri anti-banali)
                                    elif stat_type in ["Red Cards", "Red cards", "Red"]:
                                        try:
                                            away_red_cards = int(value) if value is not None else 0
                                        except:
                                            away_red_cards = 0
                                    # 🔧 PRIORITÀ 2: Corner (per mercati corner)
                                    elif stat_type in ["Corner Kicks", "Corner kicks", "Corners", "Corner"]:
                                        try:
                                            away_corners = int(value) if value is not None else 0
                                        except:
                                            away_corners = 0
                                    # 🔧 PRIORITÀ 3: Cartellini gialli (per mercati cartellini)
                                    elif stat_type in ["Yellow Cards", "Yellow cards", "Yellow"]:
                                        try:
                                            away_yellow_cards = int(value) if value is not None else 0
                                        except:
                                            away_yellow_cards = 0
                                    # 🔧 PRIORITÀ 4: Falli (per contesto partita)
                                    elif stat_type in ["Fouls", "Total Fouls", "Foul"]:
                                        try:
                                            away_fouls = int(value) if value is not None else 0
                                        except:
                                            away_fouls = 0
                                
                                if home_shots_on_target > 0 or away_shots_on_target > 0 or home_possession is not None:
                                    poss_str = f", Possesso {home_possession:.0f}%-{away_possession:.0f}%" if home_possession is not None else ""
                                    logger.info(f"✅ Statistiche ottenute da endpoint separato per {home_team} vs {away_team}: SOT {home_shots_on_target}-{away_shots_on_target}, Shots {home_total_shots}-{away_total_shots}, xG {home_xg:.2f}-{away_xg:.2f}{poss_str}")
                                else:
                                    # 🔧 LOG: Mostra quali statistiche sono disponibili anche se sono 0
                                    logger.debug(f"⚠️  Statistiche endpoint separato per {home_team} vs {away_team}: tutte a 0 o None")
                                    if statistics and len(statistics) >= 2:
                                        home_stats_list = statistics[0].get("statistics", [])
                                        away_stats_list = statistics[1].get("statistics", [])
                                        # 🔧 INFO: Mostra TUTTI i tipi di statistiche disponibili dall'endpoint separato
                                        all_home_types = [s.get('type', 'N/A') for s in home_stats_list]
                                        all_away_types = [s.get('type', 'N/A') for s in away_stats_list]
                                        logger.info(f"📊 Tipi statistiche endpoint separato per {home_team} vs {away_team}: {all_home_types}")
                                        logger.debug(f"   Tipi statistiche home disponibili: {all_home_types}")
                                        logger.debug(f"   Tipi statistiche away disponibili: {all_away_types}")

                        # 🆕 NUOVO: Recupera quote live da API-SPORTS
                        odds_data = self._fetch_odds_from_api_sports(fixture_id) if fixture_id else None

                        # Usa quote reali se disponibili, altrimenti fallback a valori di default
                        odds_1 = odds_data.get('odds_1', 2.0) if odds_data else 2.0
                        odds_x = odds_data.get('odds_x', 3.0) if odds_data else 3.0
                        odds_2 = odds_data.get('odds_2', 2.0) if odds_data else 2.0
                        odds_over_0_5 = odds_data.get('odds_over_0_5') if odds_data else None
                        odds_under_0_5 = odds_data.get('odds_under_0_5') if odds_data else None
                        odds_over_1_5 = odds_data.get('odds_over_1_5') if odds_data else None
                        odds_under_1_5 = odds_data.get('odds_under_1_5') if odds_data else None
                        odds_over_2_5 = odds_data.get('odds_over_2_5') if odds_data else None
                        odds_under_2_5 = odds_data.get('odds_under_2_5') if odds_data else None
                        odds_over_3_5 = odds_data.get('odds_over_3_5') if odds_data else None
                        odds_under_3_5 = odds_data.get('odds_under_3_5') if odds_data else None
                        odds_btts_yes = odds_data.get('odds_btts_yes') if odds_data else None
                        odds_btts_no = odds_data.get('odds_btts_no') if odds_data else None
                        # 🆕 NUOVO: Quote Double Chance, DNB, HT/FT, Odd/Even
                        odds_1x = odds_data.get('odds_1x') if odds_data else None
                        odds_x2 = odds_data.get('odds_x2') if odds_data else None
                        odds_12 = odds_data.get('odds_12') if odds_data else None
                        odds_dnb_home = odds_data.get('odds_dnb_home') if odds_data else None
                        odds_dnb_away = odds_data.get('odds_dnb_away') if odds_data else None
                        odds_ht_1 = odds_data.get('odds_ht_1') if odds_data else None
                        odds_ht_x = odds_data.get('odds_ht_x') if odds_data else None
                        odds_ht_2 = odds_data.get('odds_ht_2') if odds_data else None
                        odds_2h_1 = odds_data.get('odds_2h_1') if odds_data else None
                        odds_2h_x = odds_data.get('odds_2h_x') if odds_data else None
                        odds_2h_2 = odds_data.get('odds_2h_2') if odds_data else None
                        odds_goals_odd = odds_data.get('odds_goals_odd') if odds_data else None
                        odds_goals_even = odds_data.get('odds_goals_even') if odds_data else None

                        # 🔧 LOG: Verifica prima di aggiungere
                        logger.debug(f"🔍 Aggiungendo partita LIVE: {home_team} vs {away_team} (id: {match_id})")
                        if odds_data:
                            logger.info(f"💰 Quote recuperate: 1X2 ({odds_1:.2f}/{odds_x:.2f}/{odds_2:.2f}), Under 2.5: {odds_under_2_5}, BTTS: {odds_btts_yes}")

                        matches.append({
                            'id': match_id,
                            'home': home_team,
                            'away': away_team,
                            'league': league_name,
                            'country': country,
                            'date': fixture_date,
                            'odds_1': odds_1,
                            'odds_x': odds_x,
                            'odds_2': odds_2,
                            'odds_over_0_5': odds_over_0_5,
                            'odds_under_0_5': odds_under_0_5,
                            'odds_over_1_5': odds_over_1_5,
                            'odds_under_1_5': odds_under_1_5,
                            'odds_over_2_5': odds_over_2_5,
                            'odds_under_2_5': odds_under_2_5,
                            'odds_over_3_5': odds_over_3_5,
                            'odds_under_3_5': odds_under_3_5,
                            'odds_btts_yes': odds_btts_yes,
                            'odds_btts_no': odds_btts_no,
                            # 🆕 Quote aggiuntive
                            'odds_1x': odds_1x,
                            'odds_x2': odds_x2,
                            'odds_12': odds_12,
                            'odds_dnb_home': odds_dnb_home,
                            'odds_dnb_away': odds_dnb_away,
                            'odds_ht_1': odds_ht_1,
                            'odds_ht_x': odds_ht_x,
                            'odds_ht_2': odds_ht_2,
                            'odds_2h_1': odds_2h_1,
                            'odds_2h_x': odds_2h_x,
                            'odds_2h_2': odds_2h_2,
                            'odds_goals_odd': odds_goals_odd,
                            'odds_goals_even': odds_goals_even,
                            'source': 'api-sports',
                            'league_id': league_id,
                            'is_live': True,  # Marca come live
                            'score_home': score_home,
                            'score_away': score_away,
                            'minute': minute,
                            'status': status_long,
                            # 🔧 Statistiche live (DOPPIO FORMATO per compatibilità)
                            'home_shots_on_target': home_shots_on_target,
                            'away_shots_on_target': away_shots_on_target,
                            'shots_on_target_home': home_shots_on_target,  # 🎯 Formato alternativo
                            'shots_on_target_away': away_shots_on_target,  # 🎯 Formato alternativo
                            'home_total_shots': home_total_shots,
                            'away_total_shots': away_total_shots,
                            'shots_home': home_total_shots,  # 🎯 Formato alternativo
                            'shots_away': away_total_shots,  # 🎯 Formato alternativo
                            'home_xg': home_xg,
                            'away_xg': away_xg,
                            'home_dangerous_attacks': home_dangerous_attacks,
                            'away_dangerous_attacks': away_dangerous_attacks,
                            'dangerous_attacks_home': home_dangerous_attacks,  # 🎯 Formato alternativo
                            'dangerous_attacks_away': away_dangerous_attacks,  # 🎯 Formato alternativo
                            'home_possession': home_possession,
                            'away_possession': away_possession,
                            # 🔧 FIX: Calcola possesso away se non disponibile ma home sì (e viceversa)
                            'possession_home': home_possession if home_possession is not None else (100 - away_possession if away_possession is not None else None),
                            'possession_away': away_possession if away_possession is not None else (100 - home_possession if home_possession is not None else None),
                            # 🔧 NUOVO: Statistiche aggiuntive (priorità 1-4)
                            'home_corners': home_corners,
                            'away_corners': away_corners,
                            'corners_home': home_corners,  # 🎯 Formato alternativo
                            'corners_away': away_corners,  # 🎯 Formato alternativo
                            'home_yellow_cards': home_yellow_cards,
                            'away_yellow_cards': away_yellow_cards,
                            'yellow_cards_home': home_yellow_cards,  # 🎯 Formato alternativo
                            'yellow_cards_away': away_yellow_cards,  # 🎯 Formato alternativo
                            'home_red_cards': home_red_cards,
                            'away_red_cards': away_red_cards,
                            'red_cards_home': home_red_cards,  # 🎯 Formato alternativo
                            'red_cards_away': away_red_cards,  # 🎯 Formato alternativo
                            'home_fouls': home_fouls,
                            'away_fouls': away_fouls,
                            'fouls_home': home_fouls,  # 🎯 Formato alternativo
                            'fouls_away': away_fouls  # 🎯 Formato alternativo
                        })
                    except Exception as e:
                        logger.error(f"❌ Errore processing API-SPORTS live fixture: {e}")
                        import traceback
                        logger.error(f"❌ Traceback: {traceback.format_exc()}")
                        continue
                
                # 🔧 LOG FINALE: Mostra quante partite sono state processate
                logger.info(f"📊 Partite LIVE processate: {processed_count} aggiunte, {filtered_count} filtrate, {len(matches)} totali in lista")
                return matches
        except urllib.error.HTTPError as e:
            # 🔧 DEBUG: Log dettagliato degli errori HTTP
            logger.error(f"❌ Errore HTTP API-SPORTS live: {e.code} - {e.reason}")
            logger.error(f"   URL: {full_url}")
            try:
                error_body = e.read().decode()
                logger.error(f"   Error body: {error_body[:500]}")  # Prime 500 chars
            except:
                pass
            if e.code == 401:
                logger.error("❌ API key non valida o non autorizzata!")
            elif e.code == 429:
                logger.error("❌ Rate limit superato!")
            elif e.code == 403:
                logger.error("❌ Accesso negato! Verifica il tuo piano API-SPORTS")
            return []
        except urllib.error.URLError as e:
            logger.error(f"❌ Errore connessione API-SPORTS live: {e.reason}")
            return []
        except Exception as e:
            logger.error(f"❌ Errore API-SPORTS live: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
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

    def _fetch_odds_from_api_sports(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """
        Recupera quote live per una specifica partita da API-SPORTS.
        Endpoint: /odds/live

        Returns:
            Dizionario con quote per vari mercati:
            {
                'odds_1': float,  # Vittoria casa
                'odds_x': float,  # Pareggio
                'odds_2': float,  # Vittoria ospite
                'odds_over_0_5': float,
                'odds_under_0_5': float,
                'odds_over_1_5': float,
                'odds_under_1_5': float,
                'odds_over_2_5': float,
                'odds_under_2_5': float,
                'odds_over_3_5': float,
                'odds_under_3_5': float,
                'odds_btts_yes': float,
                'odds_btts_no': float,
            }
        """
        if not self.api_sports_key or not fixture_id:
            return None

        try:
            # Prova prima odds/live (per partite in corso)
            url = "https://v3.football.api-sports.io/odds/live"
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

                if not data.get("response") or len(data.get("response", [])) == 0:
                    # Se non ci sono odds live, prova endpoint odds normale
                    logger.debug(f"⚠️  Nessuna quota live per fixture {fixture_id}, provo endpoint odds normale...")
                    return self._fetch_odds_from_api_sports_prematch(fixture_id)

                # Estrai quote dal bookmaker preferito (o primo disponibile)
                fixture_odds = data["response"][0]
                bookmakers = fixture_odds.get("bookmakers", [])

                if not bookmakers:
                    logger.debug(f"⚠️  Nessun bookmaker disponibile per fixture {fixture_id}")
                    return None

                # 🎯 Bookmaker IDs preferiti (in ordine di preferenza)
                # Bet365 (8), Pinnacle (12), William Hill (3), 1xBet (5), Betfair (11)
                preferred_bookmaker_ids = [8, 12, 3, 5, 11]

                # Cerca bookmaker preferito
                selected_bookmaker = None
                for pref_id in preferred_bookmaker_ids:
                    for bookmaker in bookmakers:
                        if bookmaker.get("id") == pref_id:
                            selected_bookmaker = bookmaker
                            logger.info(f"💰 Usando bookmaker: {bookmaker.get('name', 'N/A')} (ID: {pref_id}) per fixture {fixture_id}")
                            break
                    if selected_bookmaker:
                        break

                # Se nessun preferito trovato, usa il primo disponibile
                if not selected_bookmaker:
                    selected_bookmaker = bookmakers[0]
                    logger.info(f"💰 Usando bookmaker di default: {selected_bookmaker.get('name', 'N/A')} (ID: {selected_bookmaker.get('id')}) per fixture {fixture_id}")

                # Usa il bookmaker selezionato
                odds_dict = {}
                bookmaker = selected_bookmaker
                bets = bookmaker.get("bets", [])

                for bet in bets:
                    bet_name = bet.get("name", "")
                    bet_id = bet.get("id", 0)
                    values = bet.get("values", [])

                    # Match Winner (1X2)
                    if bet_name == "Match Winner" or bet_id == 1:
                        for value in values:
                            outcome = value.get("value", "")
                            odd = value.get("odd")
                            if odd:
                                if outcome == "Home":
                                    odds_dict['odds_1'] = float(odd)
                                elif outcome == "Draw":
                                    odds_dict['odds_x'] = float(odd)
                                elif outcome == "Away":
                                    odds_dict['odds_2'] = float(odd)

                    # Goals Over/Under
                    elif "Goals Over/Under" in bet_name or bet_id == 5:
                        for value in values:
                            outcome = value.get("value", "")
                            odd = value.get("odd")
                            if odd:
                                # Over/Under 0.5
                                if "Over 0.5" in outcome:
                                    odds_dict['odds_over_0_5'] = float(odd)
                                elif "Under 0.5" in outcome:
                                    odds_dict['odds_under_0_5'] = float(odd)
                                # Over/Under 1.5
                                elif "Over 1.5" in outcome:
                                    odds_dict['odds_over_1_5'] = float(odd)
                                elif "Under 1.5" in outcome:
                                    odds_dict['odds_under_1_5'] = float(odd)
                                # Over/Under 2.5
                                elif "Over 2.5" in outcome:
                                    odds_dict['odds_over_2_5'] = float(odd)
                                elif "Under 2.5" in outcome:
                                    odds_dict['odds_under_2_5'] = float(odd)
                                # Over/Under 3.5
                                elif "Over 3.5" in outcome:
                                    odds_dict['odds_over_3_5'] = float(odd)
                                elif "Under 3.5" in outcome:
                                    odds_dict['odds_under_3_5'] = float(odd)

                    # Both Teams Score (BTTS)
                    elif "Both Teams Score" in bet_name or "BTTS" in bet_name or bet_id == 8:
                        for value in values:
                            outcome = value.get("value", "")
                            odd = value.get("odd")
                            if odd:
                                if "Yes" in outcome:
                                    odds_dict['odds_btts_yes'] = float(odd)
                                elif "No" in outcome:
                                    odds_dict['odds_btts_no'] = float(odd)

                    # Double Chance (1X, X2, 12)
                    elif "Double Chance" in bet_name or bet_id == 3:
                        for value in values:
                            outcome = value.get("value", "")
                            odd = value.get("odd")
                            if odd:
                                if "Home/Draw" in outcome or "1X" in outcome:
                                    odds_dict['odds_1x'] = float(odd)
                                elif "Draw/Away" in outcome or "X2" in outcome:
                                    odds_dict['odds_x2'] = float(odd)
                                elif "Home/Away" in outcome or "12" in outcome:
                                    odds_dict['odds_12'] = float(odd)

                    # Draw No Bet (DNB)
                    elif "Draw No Bet" in bet_name or "DNB" in bet_name:
                        for value in values:
                            outcome = value.get("value", "")
                            odd = value.get("odd")
                            if odd:
                                if "Home" in outcome:
                                    odds_dict['odds_dnb_home'] = float(odd)
                                elif "Away" in outcome:
                                    odds_dict['odds_dnb_away'] = float(odd)

                    # First Half Result / Half Time Result
                    elif ("First Half" in bet_name or "Half Time Result" in bet_name or "HT Result" in bet_name) and "FT" not in bet_name:
                        for value in values:
                            outcome = value.get("value", "")
                            odd = value.get("odd")
                            if odd:
                                if "Home" in outcome:
                                    odds_dict['odds_ht_1'] = float(odd)
                                elif "Draw" in outcome:
                                    odds_dict['odds_ht_x'] = float(odd)
                                elif "Away" in outcome:
                                    odds_dict['odds_ht_2'] = float(odd)

                    # Second Half Result
                    elif "Second Half" in bet_name or "2nd Half Result" in bet_name:
                        for value in values:
                            outcome = value.get("value", "")
                            odd = value.get("odd")
                            if odd:
                                if "Home" in outcome:
                                    odds_dict['odds_2h_1'] = float(odd)
                                elif "Draw" in outcome:
                                    odds_dict['odds_2h_x'] = float(odd)
                                elif "Away" in outcome:
                                    odds_dict['odds_2h_2'] = float(odd)

                    # Goals Odd/Even
                    elif "Odd/Even" in bet_name or "Goals Odd or Even" in bet_name:
                        for value in values:
                            outcome = value.get("value", "")
                            odd = value.get("odd")
                            if odd:
                                if "Odd" in outcome:
                                    odds_dict['odds_goals_odd'] = float(odd)
                                elif "Even" in outcome:
                                    odds_dict['odds_goals_even'] = float(odd)

                # Quote recuperate con successo
                if odds_dict:
                    logger.info(f"✅ Quote recuperate per fixture {fixture_id}: {list(odds_dict.keys())}")
                    return odds_dict
                else:
                    logger.debug(f"⚠️  Nessuna quota utile trovata per fixture {fixture_id}")
                    return None

        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.warning(f"⚠️  Rate limit raggiunto per quote fixture {fixture_id}")
            else:
                logger.debug(f"⚠️  Errore HTTP ottenendo quote per fixture {fixture_id}: {e.code}")
            return None
        except Exception as e:
            logger.debug(f"⚠️  Errore ottenendo quote per fixture {fixture_id}: {e}")
            return None

    def _fetch_odds_from_api_sports_prematch(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """Fallback: prova endpoint odds pre-match se live non disponibile"""
        if not self.api_sports_key or not fixture_id:
            return None

        try:
            url = "https://v3.football.api-sports.io/odds"
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

                if not data.get("response") or len(data.get("response", [])) == 0:
                    return None

                # Usa stessa logica di parsing del metodo principale
                fixture_odds = data["response"][0]
                bookmakers = fixture_odds.get("bookmakers", [])

                if not bookmakers:
                    return None

                odds_dict = {}

                for bookmaker in bookmakers:
                    bets = bookmaker.get("bets", [])

                    for bet in bets:
                        bet_name = bet.get("name", "")
                        bet_id = bet.get("id", 0)
                        values = bet.get("values", [])

                        if bet_name == "Match Winner" or bet_id == 1:
                            for value in values:
                                outcome = value.get("value", "")
                                odd = value.get("odd")
                                if odd:
                                    if outcome == "Home":
                                        odds_dict['odds_1'] = float(odd)
                                    elif outcome == "Draw":
                                        odds_dict['odds_x'] = float(odd)
                                    elif outcome == "Away":
                                        odds_dict['odds_2'] = float(odd)

                        elif "Goals Over/Under" in bet_name or bet_id == 5:
                            for value in values:
                                outcome = value.get("value", "")
                                odd = value.get("odd")
                                if odd:
                                    if "Over 0.5" in outcome:
                                        odds_dict['odds_over_0_5'] = float(odd)
                                    elif "Under 0.5" in outcome:
                                        odds_dict['odds_under_0_5'] = float(odd)
                                    elif "Over 1.5" in outcome:
                                        odds_dict['odds_over_1_5'] = float(odd)
                                    elif "Under 1.5" in outcome:
                                        odds_dict['odds_under_1_5'] = float(odd)
                                    elif "Over 2.5" in outcome:
                                        odds_dict['odds_over_2_5'] = float(odd)
                                    elif "Under 2.5" in outcome:
                                        odds_dict['odds_under_2_5'] = float(odd)
                                    elif "Over 3.5" in outcome:
                                        odds_dict['odds_over_3_5'] = float(odd)
                                    elif "Under 3.5" in outcome:
                                        odds_dict['odds_under_3_5'] = float(odd)

                        elif "Both Teams Score" in bet_name or "BTTS" in bet_name or bet_id == 8:
                            for value in values:
                                outcome = value.get("value", "")
                                odd = value.get("odd")
                                if odd:
                                    if "Yes" in outcome:
                                        odds_dict['odds_btts_yes'] = float(odd)
                                    elif "No" in outcome:
                                        odds_dict['odds_btts_no'] = float(odd)

                    if len(odds_dict) >= 3:
                        break

                if odds_dict:
                    logger.debug(f"✅ Quote pre-match recuperate per fixture {fixture_id}")
                    return odds_dict
                else:
                    return None

        except Exception as e:
            logger.debug(f"⚠️  Errore ottenendo quote pre-match per fixture {fixture_id}: {e}")
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

