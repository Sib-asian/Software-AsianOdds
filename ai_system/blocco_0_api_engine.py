"""
BLOCCO 0: API Data Engine
==========================

Raccolta intelligente di dati real-time dalle API con:
- Caching ottimizzato per risparmiare quota
- Prioritizzazione basata su importanza match
- Data quality scoring
- Fallback cascade (API → Cache → DB → Default)
- Smart quota management

Utilizza:
- TheSportsDB (unlimited, gratis)
- API-Football (100 calls/day, priorità alta)
- Football-Data.org (opzionale)
"""

import asyncio
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import requests

# Imports from existing system
try:
    from api_manager import APIManager, CacheManager, QuotaManager
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from api_manager import APIManager, CacheManager, QuotaManager

from .config import AIConfig
from .utils.statsbomb_client import StatsBombClient, StatsBombSettings
from .utils.theodds_api_client import TheOddsAPIClient

logger = logging.getLogger(__name__)


class APIDataEngine:
    """
    Motore di raccolta dati intelligente per pipeline AI.

    Responsabilità:
    1. Raccogliere dati da multiple API sources
    2. Valutare qualità e completezza dati
    3. Gestire cache e quota intelligentemente
    4. Fornire context arricchito per AI pipeline
    """

    def __init__(self, config: Optional[AIConfig] = None):
        """
        Inizializza API Data Engine

        Args:
            config: Configurazione AI (usa default se None)
        """
        self.config = config or AIConfig()

        # Initialize managers
        self.api_manager = APIManager()
        self.cache = self.api_manager.cache
        self.quota = self.api_manager.quota
        self.api_football_provider = self.api_manager.providers.get("api-football")

        # External data helpers
        self.weather_api_key = self.config.openweather_api_key
        self._weather_cache: Dict[str, Dict[str, Any]] = {}
        self._xg_cache: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "fallbacks": 0
        }

        # StatsBomb Open Data
        self.statsbomb_client: Optional[StatsBombClient] = None
        if self.config.statsbomb_enabled:
            sb_settings = StatsBombSettings(
                enabled=True,
                max_matches=self.config.statsbomb_max_matches,
                cache_hours=self.config.statsbomb_cache_hours,
                min_matches_required=self.config.statsbomb_min_matches,
            )
            self.statsbomb_client = StatsBombClient(sb_settings)

        cache_dir = self.config.cache_dir
        if not isinstance(cache_dir, Path):
            cache_dir = Path(cache_dir)
        self.odds_history_dir = cache_dir / "odds_history"
        self.odds_history_dir.mkdir(parents=True, exist_ok=True)

        self.theodds_client: Optional[TheOddsAPIClient] = None
        if self.config.theodds_enabled and self.config.theodds_api_key:
            self.theodds_client = TheOddsAPIClient(
                api_key=self.config.theodds_api_key,
                regions=self.config.theodds_regions,
                markets=self.config.theodds_markets,
                primary_market=self.config.theodds_primary_market,
                odds_format=self.config.theodds_odds_format,
                date_format=self.config.theodds_date_format,
                sport_mapping=self.config.theodds_sport_mapping,
                history_window_hours=self.config.theodds_history_window_hours,
            )

        logger.info("✅ API Data Engine initialized")
        self._prefetch_executor = ThreadPoolExecutor(max_workers=self.config.api_prefetch_max_workers)

    def _normalize_match_dict(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizza match dict convertendo datetime objects a stringhe ISO.
        Questo previene errori quando il codice chiama metodi stringa su datetime.
        """
        normalized = dict(match)
        
        # Normalizza 'date' e 'match_date'
        for key in ['date', 'match_date']:
            if key in normalized and normalized[key] is not None:
                value = normalized[key]
                if isinstance(value, datetime):
                    normalized[key] = value.isoformat()
                elif not isinstance(value, str):
                    normalized[key] = str(value)
        
        return normalized

    def collect(
        self,
        match: Dict[str, Any],
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Raccoglie dati completi per un match.

        Args:
            match: Dizionario con info match (home, away, league, date, etc)
            force_refresh: Se True, bypassa cache

        Returns:
            Dizionario con dati arricchiti + metadata
        """
        self.stats["total_requests"] += 1

        # Normalizza match dict: converte datetime objects a stringhe
        match = self._normalize_match_dict(match)

        try:
            # Calculate match importance
            importance = self._calculate_importance(match)

            # Get context for both teams
            home_context = self._get_team_context(
                match["home"],
                match["league"],
                importance,
                force_refresh
            )

            away_context = self._get_team_context(
                match["away"],
                match["league"],
                importance,
                force_refresh
            )

            # Enrich with match-specific data (if high importance)
            match_data = {}
            if importance >= self.config.high_importance_threshold:
                match_data = self._get_match_specific_data(match)

            match_data = self._enrich_precision_signals(match, match_data)

            # Calculate data quality score
            quality_score = self._calculate_data_quality(
                home_context,
                away_context,
                match_data
            )

            # Assemble enriched context
            enriched = {
                "home_context": home_context,
                "away_context": away_context,
                "match_data": match_data,
                "metadata": {
                    "importance": importance,
                    "data_quality": quality_score,
                    "api_calls_used": (
                        home_context.get("api_calls_used", 0) +
                        away_context.get("api_calls_used", 0)
                    ),
                    "sources": self._get_sources_used(home_context, away_context),
                    "timestamp": datetime.now().isoformat(),
                    "cache_used": home_context.get("source") == "cache"
                }
            }

            precision_sources = match_data.get("precision_sources", [])
            if precision_sources:
                metadata_sources = set(enriched["metadata"].get("sources", []))
                for src in precision_sources:
                    if src not in metadata_sources:
                        metadata_sources.add(src)
                enriched["metadata"]["sources"] = list(metadata_sources)

            logger.info(
                f"✅ Data collected for {match['home']} vs {match['away']}: "
                f"quality={quality_score:.2f}, importance={importance:.2f}"
            )

            return enriched

        except Exception as e:
            import traceback
            logger.error(f"❌ Error collecting data for match: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return self._get_fallback_context(match)

    def _calculate_importance(self, match: Dict[str, Any]) -> float:
        """
        Calcola importanza del match (0-1) per prioritizzare API usage.

        Fattori:
        - Lega (Serie A > Serie B > minori)
        - Stage (playoff, finale = più importante)
        - Teams (top teams = più importante)
        - User specified importance

        Returns:
            Score 0.0-1.0 (1.0 = massima importanza)
        """
        importance = 0.5  # Base

        # User-specified importance (se presente)
        if "importance" in match:
            return float(match["importance"])

        # League importance
        league = match.get("league", "").lower()
        league_scores = {
            "serie a": 0.4,
            "premier league": 0.4,
            "la liga": 0.4,
            "bundesliga": 0.4,
            "ligue 1": 0.35,
            "champions league": 0.5,
            "europa league": 0.4,
            "serie b": 0.2,
        }
        for league_name, score in league_scores.items():
            if league_name in league:
                importance += score
                break
        else:
            importance += 0.1  # Default per altre leghe

        # Stage importance (se presente)
        stage = match.get("stage", "").lower()
        if any(keyword in stage for keyword in ["final", "finale", "playoff"]):
            importance += 0.15
        elif any(keyword in stage for keyword in ["semi", "quarter"]):
            importance += 0.10

        # Team importance (se presente nel LEVEL 1 DB)
        # TODO: Integrare con team database per ranking

        return min(importance, 1.0)  # Cap a 1.0

    def _get_team_context(
        self,
        team: str,
        league: str,
        importance: float,
        force_refresh: bool
    ) -> Dict[str, Any]:
        """
        Ottiene context per una squadra con strategia intelligente.

        Strategy:
        1. Check cache (se non force_refresh)
        2. Se high importance → usa API premium
        3. Se medium importance → usa solo free API
        4. Se low importance → usa cache o fallback
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = self.cache.get(team, league)
            if cached:
                self.stats["cache_hits"] += 1
                return {
                    "source": "cache",
                    "provider": "cache",
                    "data": cached,
                    "api_calls_used": 0
                }

        self.stats["cache_misses"] += 1

        # Determine API strategy based on importance and quota
        use_api_football = False
        quota_remaining = (
            self.config.api_daily_budget -
            self.quota.get_usage("api-football")
        )

        if importance >= self.config.high_importance_threshold:
            # High importance: use API-Football if quota available
            if quota_remaining > self.config.api_reserved_enrichment:
                use_api_football = True

        # Get data from API manager
        try:
            context = self.api_manager.get_team_context(
                team, league, force_refresh=True
            )

            self.stats["api_calls"] += context.get("api_calls_used", 0)

            # Enrich with additional data if using API-Football
            if use_api_football and context["source"] == "api":
                enriched = self._enrich_with_api_football(team, league, context)
                if enriched:
                    context["data"].update(enriched)
                    context["api_calls_used"] += 1

            return context

        except Exception as e:
            logger.error(f"❌ Error getting team context for {team}: {e}")
            self.stats["fallbacks"] += 1
            return {
                "source": "fallback",
                "provider": "fallback",
                "data": self._get_fallback_team_data(team, league),
                "api_calls_used": 0
            }

    def _enrich_with_api_football(
        self,
        team: str,
        league: str,
        existing_context: Dict
    ) -> Optional[Dict]:
        """
        Arricchisce context con dati da API-Football (injuries, form, xG).

        Returns:
            Dati aggiuntivi o None se non disponibili
        """
        provider = self.api_football_provider
        if not provider or not getattr(provider, "api_key", None):
            return None

        if not self.quota.can_use("api-football", calls=1):
            logger.debug("API-Football quota non disponibile per %s", team)
            return None

        try:
            team_data = provider.get_team_match_data(team, None)
        except Exception as exc:
            logger.debug("API-Football enrichment error (%s): %s", team, exc)
            return None

        if not team_data:
            return None

        self.quota.log_usage("api-football", calls=1)
        self.stats["api_calls"] += 1

        data = existing_context.setdefault("data", {})
        if team_data.get("injuries"):
            data["injuries"] = team_data["injuries"]
        if team_data.get("suspensions"):
            data["suspensions"] = team_data["suspensions"]
        if team_data.get("lineup"):
            data["lineup_prediction"] = team_data["lineup"]
        if team_data.get("form"):
            data["form"] = team_data["form"]

        existing_context["provider"] = "api-football"
        existing_context["source"] = "api"
        return data

    def _get_match_specific_data(self, match: Dict) -> Dict:
        """
        Raccoglie dati specifici per il match (H2H, odds history, etc).

        Args:
            match: Info del match

        Returns:
            Dati match-specific
        """
        match_data = {
            "h2h_history": [],
            "odds_history": [],
            "injuries": {"home": [], "away": []},
            "suspensions": {"home": [], "away": []},
            "lineup_prediction": {}
        }

        try:
            api_football_data = self._get_api_football_match_data(match)
            if api_football_data:
                match_data.update(api_football_data)

            h2h_stats = self._get_head_to_head_context(match)
            if h2h_stats:
                match_data["h2h"] = h2h_stats
                match_data.setdefault("precision_sources", []).append("api_football_h2h")

            odds_history, odds_snapshot = self._collect_odds_history(
                match,
                match.get("market", "h2h")
            )
            if odds_history:
                match_data["odds_history"] = odds_history
            if odds_snapshot:
                match_data["odds_snapshot"] = odds_snapshot
                match_data.setdefault("precision_sources", []).append("theoddsapi")

        except Exception as e:
            logger.error(f"❌ Error getting match-specific data: {e}")

        return match_data

    def prefetch_matches(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prefetch contexts for upcoming matches (fills cache asynchronously).
        """
        if not self.config.api_prefetch_enabled:
            return {"status": "disabled", "requested": 0}

        if not matches:
            return {"status": "empty", "requested": 0}

        batch = matches[: self.config.api_prefetch_batch_size]

        async def _run_prefetch():
            sem = asyncio.Semaphore(self.config.api_prefetch_max_workers)
            summary = {"requested": len(batch), "completed": 0, "errors": 0}

            async def _worker(match: Dict[str, Any]):
                async with sem:
                    try:
                        await asyncio.to_thread(self._prefetch_single_match, match)
                        summary["completed"] += 1
                    except Exception as exc:
                        summary["errors"] += 1
                        logger.debug("Prefetch failed for %s vs %s: %s", match.get("home"), match.get("away"), exc)

            await asyncio.gather(*[_worker(match) for match in batch])
            return summary

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_run_prefetch())
        finally:
            loop.close()
        logger.info("⚡ Prefetch completato: %s", result)
        return result

    def shutdown(self):
        try:
            self._prefetch_executor.shutdown(wait=False)
        except Exception:
            pass

    def __del__(self):
        self.shutdown()

    def _enrich_precision_signals(self, match: Dict, match_data: Optional[Dict]) -> Dict:
        """
        Aggiunge al contesto dati che aumentano la precisione dei calcoli
        (meteo, xG gratuiti, rumor/contesto extra).
        """
        enriched = dict(match_data) if match_data else {}

        weather_context = self._get_weather_context(match)
        if weather_context:
            enriched["weather"] = weather_context

        xg_context = self._get_understat_context(match)
        if xg_context:
            enriched["xg_metrics"] = xg_context

        if self.statsbomb_client:
            statsbomb_context = self.statsbomb_client.get_match_context(match)
            if statsbomb_context:
                enriched["statsbomb_metrics"] = statsbomb_context
                enriched.setdefault("precision_sources", []).append("statsbomb_open_data")
                self._merge_statsbomb_signals(enriched, statsbomb_context)

        return enriched

    def _get_api_football_match_data(self, match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch injuries/lineups using API-Football if available."""
        provider = self.api_football_provider
        if not provider or not getattr(provider, "api_key", None):
            return None

        season = match.get("season") or self._infer_season_from_match(match)
        season = str(season)

        def _fetch(team_name: str):
            if not team_name:
                return None
            if not self.quota.can_use("api-football", calls=1):
                logger.debug("Quota API-Football insufficiente per %s", team_name)
                return None
            try:
                data = provider.get_team_match_data(team_name, season)
            except Exception as exc:
                logger.debug("API-Football team fetch failed (%s): %s", team_name, exc)
                return None
            if data:
                self.quota.log_usage("api-football", calls=1)
                self.stats["api_calls"] += 1
            return data

        try:
            home_data = _fetch(match.get("home", ""))
            away_data = _fetch(match.get("away", ""))

            if not home_data and not away_data:
                return None

            enriched: Dict[str, Any] = {}
            if home_data:
                enriched.setdefault("injuries", {})["home"] = home_data.get("injuries", [])
                enriched.setdefault("suspensions", {})["home"] = home_data.get("suspensions", [])
                enriched.setdefault("lineup_prediction", {})["home"] = home_data.get("lineup", {})
                enriched["form_home"] = home_data.get("form")
            if away_data:
                enriched.setdefault("injuries", {})["away"] = away_data.get("injuries", [])
                enriched.setdefault("suspensions", {})["away"] = away_data.get("suspensions", [])
                enriched.setdefault("lineup_prediction", {})["away"] = away_data.get("lineup", {})
                enriched["form_away"] = away_data.get("form")

            if "injuries" in enriched:
                enriched["injuries"].setdefault("home", [])
                enriched["injuries"].setdefault("away", [])
            if "suspensions" in enriched:
                enriched["suspensions"].setdefault("home", [])
                enriched["suspensions"].setdefault("away", [])
            if "lineup_prediction" in enriched:
                enriched["lineup_prediction"].setdefault("home", {})
                enriched["lineup_prediction"].setdefault("away", {})

            return enriched
        except Exception as exc:
            logger.debug("API-Football match data unavailable: %s", exc)
            return None

    def _get_head_to_head_context(self, match: Dict[str, Any], last: int = 8) -> Optional[Dict[str, Any]]:
        provider = self.api_football_provider
        if not provider or not getattr(provider, "api_key", None):
            return None

        if not self.quota.can_use("api-football", calls=1):
            logger.debug(
                "Quota API-Football insufficiente per H2H %s vs %s",
                match.get("home"),
                match.get("away")
            )
            return None

        try:
            stats = provider.get_head_to_head_stats(
                match.get("home", ""),
                match.get("away", ""),
                last=last,
                season=match.get("season")
            )
        except Exception as exc:
            logger.debug("API-Football H2H fetch failed: %s", exc)
            return None

        if stats:
            self.quota.log_usage("api-football", calls=1)
            self.stats["api_calls"] += 1
        return stats

    def _prefetch_single_match(self, match: Dict[str, Any]) -> None:
        home = match.get("home")
        away = match.get("away")
        league = match.get("league", "")
        if not home or not away:
            return

        # Fetch team contexts (which populates cache)
        self._get_team_context(home, league, importance=self._calculate_importance(match), force_refresh=False)
        self._get_team_context(away, league, importance=self._calculate_importance(match), force_refresh=False)

        # Head-to-head to warm API-Football cache/quota permitting
        try:
            self._get_head_to_head_context(match)
        except Exception:
            pass

        # Odds snapshot caching
        try:
            self._collect_odds_history(match, match.get("market", "h2h"))
        except Exception:
            pass

    def _merge_statsbomb_signals(self, enriched: Dict[str, Any], statsbomb_context: Dict[str, Any]) -> None:
        """
        Converte le metriche StatsBomb in segnali operativi già usati dai blocchi
        (xG totals ultima finestra, shots, pressures) così da evitare fallback.
        """
        try:
            max_window = getattr(self.config, "statsbomb_max_matches", 5)
        except AttributeError:
            max_window = 5

        for side in ("home", "away"):
            metrics = statsbomb_context.get(side) or {}
            if not metrics:
                continue

            matches_used = int(metrics.get("matches") or 0)
            window = min(matches_used or 0, max_window)
            if window <= 0:
                continue

            avg_xg = metrics.get("avg_xg_per_match")
            avg_xga = metrics.get("avg_xga_per_match")

            if isinstance(avg_xg, (int, float)):
                enriched[f"xg_{side}"] = round(avg_xg * window, 2)
            if isinstance(avg_xga, (int, float)):
                enriched[f"xga_{side}"] = round(avg_xga * window, 2)

            # Pass through per-match advanced metrics for downstream blocks
            shots = metrics.get("avg_shots_per_match")
            shots_pct = metrics.get("shots_on_target_pct")
            pressures = metrics.get("pressures_per_match")

            if isinstance(shots, (int, float)):
                enriched[f"shots_{side}_per_match"] = round(float(shots), 2)
            if isinstance(shots_pct, (int, float)):
                enriched[f"shots_on_target_pct_{side}"] = round(float(shots_pct), 3)
            if isinstance(pressures, (int, float)):
                enriched[f"pressures_{side}_per_match"] = round(float(pressures), 1)

        # Match-level quality flag (entrambi i lati disponibili)
        if "xg_home" in enriched and "xg_away" in enriched:
            enriched["have_statsbomb_window"] = True

    def _get_weather_context(self, match: Dict) -> Optional[Dict[str, Any]]:
        """Recupera e normalizza dati meteo rilevanti per il match."""
        if not self.weather_api_key:
            return None

        match_dt = self._parse_match_datetime(match)
        if match_dt is None:
            match_dt = datetime.utcnow()

        city = match.get("city") or self._infer_city_from_match(match)
        if not city:
            return None

        cache_key = f"weather:{city.lower()}:{match_dt.date().isoformat()}"
        cached = self._weather_cache.get(cache_key)
        if cached and (datetime.utcnow() - cached["timestamp"]) < timedelta(hours=3):
            return cached["data"]

        try:
            params = {
                "q": city,
                "appid": self.weather_api_key,
                "units": "metric",
                "lang": "it"
            }
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            forecasts = response.json().get("list", [])
            if not forecasts:
                return None

            closest = None
            min_diff = float("inf")
            for forecast in forecasts:
                dt_ts = forecast.get("dt")
                if dt_ts is None:
                    continue
                forecast_dt = datetime.fromtimestamp(dt_ts)
                diff = abs((forecast_dt - match_dt).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest = forecast

            if not closest:
                return None

            main = closest.get("main", {})
            weather = (closest.get("weather") or [{}])[0]
            wind = closest.get("wind", {})
            rain = closest.get("rain", {})

            weather_data = {
                "city": city,
                "match_time": match_dt.isoformat(),
                "forecast_time": datetime.fromtimestamp(closest.get("dt", match_dt.timestamp())).isoformat(),
                "temperature": main.get("temp", 20.0),
                "feels_like": main.get("feels_like", 20.0),
                "humidity": main.get("humidity", 50.0),
                "rain_mm": rain.get("3h", 0.0),
                "wind_speed_kmh": wind.get("speed", 0.0) * 3.6,
                "description": weather.get("description", "clear"),
                "condition": weather.get("main", "Clear")
            }

            penalty = self._calculate_weather_penalty(weather_data)
            weather_data["over_penalty"] = penalty["total_penalty"]
            weather_data["impact_flags"] = penalty["flags"]
            weather_data["recommendation"] = penalty["recommendation"]

            context = {
                "provider": "openweather",
                "data": weather_data
            }

            self._weather_cache[cache_key] = {
                "timestamp": datetime.utcnow(),
                "data": context
            }

            return context

        except requests.RequestException as exc:
            logger.warning(f"⚠️ Weather fetch failed for {city}: {exc}")
            return None

    def _parse_match_datetime(self, match: Dict) -> Optional[datetime]:
        """Tenta di costruire un datetime usando date/time disponibili nel match."""
        date_str = match.get("date") or match.get("match_date")
        if not date_str:
            return None

        # Se è già un oggetto datetime, restituiscilo direttamente
        if isinstance(date_str, datetime):
            return date_str

        def _from_iso(value: str) -> Optional[datetime]:
            # Assicurati che value sia una stringa
            if not isinstance(value, str):
                value = str(value)
            value = value.strip()
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None

        dt = _from_iso(date_str)
        if dt:
            return dt

        time_str = match.get("time") or match.get("match_time") or "00:00"
        # Assicurati che date_str sia una stringa prima di concatenare
        date_str_for_combined = str(date_str) if not isinstance(date_str, str) else date_str
        combined = f"{date_str_for_combined} {time_str}".strip()
        dt = _from_iso(combined)
        if dt:
            return dt

        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(combined, fmt)
            except ValueError:
                continue
        return None

    def _infer_city_from_match(self, match: Dict) -> Optional[str]:
        """Deduce la città usando venue, city o nome squadra di casa."""
        if match.get("venue_city"):
            return match["venue_city"]
        if match.get("city"):
            return match["city"]
        home_team = match.get("home")
        if home_team:
            return self._infer_city_from_team(home_team)
        return None

    def _infer_city_from_team(self, team_name: str) -> str:
        """Mapping veloce team -> città (copiato dalle integrazioni Frontendcloud)."""
        mapping = {
            # Premier League
            "liverpool": "Liverpool",
            "manchester city": "Manchester",
            "manchester united": "Manchester",
            "chelsea": "London",
            "arsenal": "London",
            "tottenham": "London",
            "west ham": "London",
            "crystal palace": "London",
            "fulham": "London",
            "newcastle": "Newcastle",
            "everton": "Liverpool",
            "aston villa": "Birmingham",
            "wolverhampton": "Wolverhampton",
            "wolves": "Wolverhampton",
            "leicester": "Leicester",
            "leeds": "Leeds",
            "southampton": "Southampton",
            "brighton": "Brighton",
            # Serie A
            "inter": "Milan",
            "internazionale": "Milan",
            "milan": "Milan",
            "ac milan": "Milan",
            "juventus": "Turin",
            "roma": "Rome",
            "lazio": "Rome",
            "napoli": "Naples",
            "ssc napoli": "Naples",
            "atalanta": "Bergamo",
            "fiorentina": "Florence",
            "torino": "Turin",
            "udinese": "Udine",
            # La Liga
            "barcelona": "Barcelona",
            "real madrid": "Madrid",
            "atletico": "Madrid",
            "sevilla": "Seville",
            "valencia": "Valencia",
            "real sociedad": "San Sebastian",
            "villarreal": "Villarreal",
            # Bundesliga
            "bayern": "Munich",
            "bayern munich": "Munich",
            "borussia dortmund": "Dortmund",
            "dortmund": "Dortmund",
            "rb leipzig": "Leipzig",
            "leipzig": "Leipzig",
            "bayer leverkusen": "Leverkusen",
            # Ligue 1
            "psg": "Paris",
            "paris saint germain": "Paris",
            "marseille": "Marseille",
            "lyon": "Lyon",
            "lille": "Lille",
        }

        key = team_name.lower()
        for token, city in mapping.items():
            if token in key:
                return city

        parts = team_name.split()
        return parts[0].title() if parts else "Unknown"

    def _calculate_weather_penalty(self, weather: Dict[str, Any]) -> Dict[str, Any]:
        """Replica la logica additiva: max -20% su probabilità Over."""
        penalty = 0.0
        flags = []

        rain = weather.get("rain_mm", 0.0)
        if rain > 5.0:
            penalty += 0.15
            flags.append(f"heavy_rain_{rain:.1f}mm")
        elif rain > 2.0:
            penalty += 0.08
            flags.append(f"moderate_rain_{rain:.1f}mm")

        wind = weather.get("wind_speed_kmh", 0.0)
        if wind > 30:
            penalty += 0.10
            flags.append(f"strong_wind_{wind:.1f}kmh")
        elif wind > 20:
            penalty += 0.05
            flags.append(f"wind_{wind:.1f}kmh")

        temp = weather.get("temperature", 20.0)
        if temp > 30:
            penalty += 0.08
            flags.append(f"hot_{temp:.1f}c")
        elif temp < 5:
            penalty += 0.05
            flags.append(f"cold_{temp:.1f}c")

        penalty = min(penalty, 0.20)
        recommendation = "neutral"
        if penalty >= 0.15:
            recommendation = "prefer_under_or_lower_line"
        elif penalty >= 0.08:
            recommendation = "monitor_under_trend"

        return {
            "total_penalty": penalty,
            "flags": flags,
            "recommendation": recommendation
        }

    def _get_match_identifier(self, match: Dict[str, Any], market: Optional[str]) -> str:
        home = (match.get("home") or "home").lower().replace(" ", "_")
        away = (match.get("away") or "away").lower().replace(" ", "_")
        date_val = match.get("date") or match.get("match_date") or "unknown"
        # Convert datetime to string if needed
        if isinstance(date_val, datetime):
            date = date_val.isoformat().replace(":", "-")
        else:
            date = str(date_val).replace(":", "-")
        market_key = (market or "h2h").lower()
        return f"{home}__{away}__{date}__{market_key}"

    def _load_odds_history_from_cache(self, match_key: str) -> List[Dict[str, Any]]:
        path = self.odds_history_dir / f"{match_key}.json"
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Unable to read odds history cache %s: %s", match_key, exc)
            return []

    def _save_odds_history_to_cache(self, match_key: str, history: List[Dict[str, Any]]) -> None:
        path = self.odds_history_dir / f"{match_key}.json"
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except OSError as exc:
            logger.debug("Unable to write odds history cache %s: %s", match_key, exc)

    def _collect_odds_history(
        self,
        match: Dict[str, Any],
        market: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        match_key = self._get_match_identifier(match, market)
        history = self._load_odds_history_from_cache(match_key)
        snapshot = None

        if self.theodds_client:
            snapshot = self.theodds_client.fetch_latest_snapshot(match, market or "h2h")
            if snapshot:
                entry = {
                    "timestamp": snapshot.get("timestamp"),
                    "market": snapshot.get("market"),
                    "prices": snapshot.get("prices"),
                    "source": "theoddsapi"
                }
                if not history or history[-1].get("timestamp") != entry["timestamp"]:
                    history.append(entry)
                    history = history[-50:]
                    self._save_odds_history_to_cache(match_key, history)

        return history, snapshot

    def _get_understat_context(self, match: Dict) -> Optional[Dict[str, Any]]:
        """Scarica xG gratuiti da Understat e li allinea al match."""
        home = match.get("home")
        away = match.get("away")
        if not home or not away:
            return None

        season = match.get("season") or self._infer_season_from_match(match)

        home_xg = self._fetch_understat_team_xg(home, season)
        away_xg = self._fetch_understat_team_xg(away, season)
        if not home_xg or not away_xg:
            return None

        return {
            "season": season,
            "home_team": home,
            "away_team": away,
            "home": home_xg,
            "away": away_xg,
            "home_advantage_xg": round(
                home_xg["xg_per_match"] - away_xg["xga_per_match"], 2
            ),
            "away_pressure_xg": round(
                away_xg["xg_per_match"] - home_xg["xga_per_match"], 2
            ),
            "source": "understat"
        }

    def _fetch_understat_team_xg(self, team_name: str, season: str) -> Optional[Dict[str, Any]]:
        """Replica integrazione Understat (scraping leggero)."""
        slug = self._normalize_team_slug(team_name)
        cache_key = f"understat:{slug}:{season}"
        cached = self._xg_cache.get(cache_key)
        if cached and (datetime.utcnow() - cached["timestamp"]) < timedelta(hours=12):
            return cached["data"]

        url = f"https://understat.com/team/{slug}/{season}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-System/1.0)"}

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                logger.debug(f"Understat returned {resp.status_code} for {team_name}")
                return None

            match_data = re.search(r"var teamsData\s*=\s*JSON\.parse\('(.+?)'\)", resp.text)
            if not match_data:
                logger.debug(f"Understat data not found for {team_name}")
                return None

            data_str = match_data.group(1).encode().decode('unicode_escape')
            teams_data = json.loads(data_str)
            if not isinstance(teams_data, dict):
                return None

            # teamsData keyed by team id; take first entry
            team_entries = next(iter(teams_data.values()), {})
            if not team_entries:
                return None

            total_xg = 0.0
            total_xga = 0.0
            matches = 0
            for match_info in team_entries:
                try:
                    total_xg += float(match_info.get("xG", 0.0))
                    total_xga += float(match_info.get("xGA", 0.0))
                    matches += 1
                except (TypeError, ValueError):
                    continue

            if matches == 0:
                return None

            result = {
                "matches_played": matches,
                "xg_total": round(total_xg, 2),
                "xga_total": round(total_xga, 2),
                "xg_per_match": round(total_xg / matches, 2),
                "xga_per_match": round(total_xga / matches, 2),
                "xg_diff_per_match": round((total_xg - total_xga) / matches, 2),
                "source": "understat.com"
            }

            self._xg_cache[cache_key] = {
                "timestamp": datetime.utcnow(),
                "data": result
            }

            return result

        except (requests.RequestException, json.JSONDecodeError) as exc:
            logger.warning(f"⚠️ Understat fetch failed for {team_name}: {exc}")
            return None

    def _normalize_team_slug(self, team_name: str) -> str:
        """Converte il nome squadra nel formato usato da Understat."""
        slug = team_name.lower().replace(" ", "_")
        manual = {
            "manchester_united": "Manchester_United",
            "manchester_city": "Manchester_City",
            "west_ham": "West_Ham",
            "newcastle_united": "Newcastle_United",
            "real_madrid": "Real_Madrid",
            "atletico_madrid": "Atletico_Madrid",
            "real_betis": "Betis",
            "bayern_munich": "Bayern_Munich",
            "borussia_dortmund": "Borussia_Dortmund",
            "rb_leipzig": "RB_Leipzig",
            "bayer_leverkusen": "Bayer_Leverkusen",
            "psg": "Paris_Saint_Germain",
            "paris_saint_germain": "Paris_Saint_Germain",
        }
        return manual.get(slug, slug.title().replace(" ", "_"))

    def _infer_season_from_match(self, match: Dict) -> str:
        """Ritorna stagione Understat (anno) basata sulla data match."""
        match_dt = self._parse_match_datetime(match)
        if not match_dt:
            return str(datetime.utcnow().year)
        year = match_dt.year
        # Stagioni europee partono in estate → se mese < luglio usa anno-1
        if match_dt.month < 7:
            year -= 1
        return str(year)

    def _calculate_data_quality(
        self,
        home_context: Dict,
        away_context: Dict,
        match_data: Dict
    ) -> float:
        """
        Calcola score di qualità dati (0-1).

        Fattori:
        - Source (API > cache > fallback)
        - Completezza dati
        - Freschezza cache
        - Coverage (injuries, form, etc)

        Returns:
            Score 0.0-1.0 (1.0 = dati perfetti)
        """
        quality = 0.5  # Base

        # Source quality
        sources_score = {
            "api": 0.25,
            "cache": 0.15,
            "fallback": 0.0
        }
        quality += sources_score.get(home_context.get("source", "fallback"), 0.0)
        quality += sources_score.get(away_context.get("source", "fallback"), 0.0)

        # Data completeness
        home_data = home_context.get("data", {})
        away_data = away_context.get("data", {})

        required_fields = ["style", "typical_position"]
        optional_fields = ["stadium", "description", "formed_year", "country"]

        # Count available fields
        home_fields = sum(1 for f in required_fields if f in home_data and home_data[f])
        away_fields = sum(1 for f in required_fields if f in away_data and away_data[f])
        home_optional = sum(1 for f in optional_fields if f in home_data and home_data[f])
        away_optional = sum(1 for f in optional_fields if f in away_data and away_data[f])

        completeness = (
            (home_fields + away_fields) / (len(required_fields) * 2) * 0.15 +
            (home_optional + away_optional) / (len(optional_fields) * 2) * 0.10
        )
        quality += completeness

        # Precision boosters
        if match_data.get("weather"):
            quality += 0.05
        if match_data.get("xg_metrics"):
            quality += 0.05
        if match_data.get("statsbomb_metrics"):
            quality += 0.07

        return min(quality, 1.0)  # Cap a 1.0

    def _get_sources_used(
        self,
        home_context: Dict,
        away_context: Dict
    ) -> List[str]:
        """Ritorna lista di sources utilizzate"""
        sources = set()

        if home_context.get("provider"):
            sources.add(home_context["provider"])
        if away_context.get("provider"):
            sources.add(away_context["provider"])

        return list(sources)

    def _get_fallback_context(self, match: Dict) -> Dict:
        """
        Context di fallback in caso di errore totale.

        Returns:
            Context minimo per permettere alla pipeline di continuare
        """
        self.stats["fallbacks"] += 1

        return {
            "home_context": {
                "source": "fallback",
                "provider": "fallback",
                "data": self._get_fallback_team_data(
                    match.get("home", "Unknown"),
                    match.get("league", "Unknown")
                ),
                "api_calls_used": 0
            },
            "away_context": {
                "source": "fallback",
                "provider": "fallback",
                "data": self._get_fallback_team_data(
                    match.get("away", "Unknown"),
                    match.get("league", "Unknown")
                ),
                "api_calls_used": 0
            },
            "match_data": {},
            "metadata": {
                "importance": 0.5,
                "data_quality": 0.1,  # Qualità molto bassa
                "api_calls_used": 0,
                "sources": ["fallback"],
                "timestamp": datetime.now().isoformat(),
                "cache_used": False,
                "fallback_reason": "complete_failure"
            }
        }

    def _get_fallback_team_data(self, team: str, league: str) -> Dict:
        """Dati di fallback per una squadra"""
        return {
            "team_name": team,
            "league": league,
            "style": "Possesso",  # Default safe
            "typical_position": "mid",
            "source": "fallback"
        }

    def get_odds_history(
        self,
        match: Dict,
        market: str = "1X2",
        lookback_hours: int = 24
    ) -> List[Dict]:
        """
        Ottiene storico movimenti quote per un match.

        Args:
            match: Info match
            market: Tipo mercato ("1X2", "Over/Under", etc)
            lookback_hours: Quante ore indietro guardare

        Returns:
            Lista di snapshots quote con timestamp
        """
        try:
            # TODO: Implementare raccolta odds history
            # Potrebbe venire da:
            # - API-Football odds endpoint
            # - Database locale se trackato
            # - Scraping (se permesso)

            logger.info(f"ℹ️ Odds history collection not yet implemented")
            return []

        except Exception as e:
            logger.error(f"❌ Error getting odds history: {e}")
            return []

    def monitor_odds_realtime(
        self,
        match: Dict,
        market: str,
        callback: Optional[callable] = None
    ) -> Dict:
        """
        Monitora quote in real-time fino a kickoff.

        Args:
            match: Info match
            market: Mercato da monitorare
            callback: Funzione da chiamare ad ogni update

        Returns:
            Status del monitoring
        """
        try:
            # TODO: Implementare monitoring real-time
            # - Polling periodico (ogni X minuti)
            # - Detection sharp movements
            # - Callback su eventi importanti

            logger.info(f"ℹ️ Real-time odds monitoring not yet implemented")
            return {"status": "not_implemented"}

        except Exception as e:
            logger.error(f"❌ Error monitoring odds: {e}")
            return {"status": "error", "error": str(e)}

    def get_statistics(self) -> Dict:
        """
        Ritorna statistiche utilizzo API Data Engine.

        Returns:
            Dizionario con statistiche
        """
        cache_hit_rate = 0.0
        if self.stats["total_requests"] > 0:
            cache_hit_rate = (
                self.stats["cache_hits"] / self.stats["total_requests"] * 100
            )

        api_usage = self.quota.get_all_usage()

        return {
            "requests": self.stats,
            "cache_hit_rate": cache_hit_rate,
            "api_usage": api_usage,
            "quota_remaining": {
                "api-football": (
                    self.config.api_daily_budget -
                    api_usage.get("api-football", 0)
                )
            }
        }

    def reset_statistics(self):
        """Reset statistiche"""
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "fallbacks": 0
        }
        logger.info("📊 Statistics reset")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def test_api_data_engine():
    """Test del API Data Engine"""
    print("=" * 70)
    print("TEST: API Data Engine")
    print("=" * 70)

    engine = APIDataEngine()

    # Test 1: Collect data for high importance match
    print("\nTest 1: High importance match (Serie A)")
    match = {
        "home": "Inter",
        "away": "Napoli",
        "league": "Serie A",
        "date": "2025-11-15",
        "importance": 0.95
    }

    context = engine.collect(match)
    print(f"  Data quality: {context['metadata']['data_quality']:.2f}")
    print(f"  API calls used: {context['metadata']['api_calls_used']}")
    print(f"  Sources: {context['metadata']['sources']}")
    print(f"  Cache used: {context['metadata']['cache_used']}")

    # Test 2: Same match again (should hit cache)
    print("\nTest 2: Same match (cache hit)")
    context2 = engine.collect(match)
    print(f"  Cache used: {context2['metadata']['cache_used']}")

    # Test 3: Low importance match
    print("\nTest 3: Low importance match (Serie B)")
    match_low = {
        "home": "Brescia",
        "away": "Cosenza",
        "league": "Serie B",
        "date": "2025-11-15"
    }

    context3 = engine.collect(match_low)
    print(f"  Importance: {context3['metadata']['importance']:.2f}")
    print(f"  Data quality: {context3['metadata']['data_quality']:.2f}")

    # Test 4: Statistics
    print("\nTest 4: Statistics")
    stats = engine.get_statistics()
    print(f"  Total requests: {stats['requests']['total_requests']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    print(f"  API calls: {stats['requests']['api_calls']}")
    print(f"  Quota remaining: {stats['quota_remaining']['api-football']}")

    print("\n" + "=" * 70)
    print("✅ API Data Engine tests completed")
    print("=" * 70)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_api_data_engine()
