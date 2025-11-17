"""
API MANAGER - LEVEL 2 Lite (Free Edition)

Gestisce chiamate API per auto-detection advanced features con:
- Multi-provider support (API-Football, Football-Data.org, TheSportsDB)
- Free tier optimization (100 calls/day)
- Cache intelligente SQLite (24h TTL)
- Quota tracking automatico
- Fallback cascade (API → Cache → LEVEL 1 DB)

Utilizzo:
    from api_manager import APIManager

    api = APIManager()
    context = api.get_team_context("Inter", "Serie A")
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import urllib.request
import urllib.parse
import urllib.error

logger = logging.getLogger(__name__)


@dataclass
class ProviderHealth:
    success: int = 0
    failures: int = 0
    last_error: Optional[str] = None
    last_success_ts: Optional[str] = None
    last_latency_ms: Optional[float] = None

    def mark_success(self, latency_ms: Optional[float] = None):
        self.success += 1
        self.last_success_ts = datetime.utcnow().isoformat()
        if latency_ms is not None:
            self.last_latency_ms = round(latency_ms, 2)
        self.last_error = None

    def mark_failure(self, error: str):
        self.failures += 1
        self.last_error = error

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ============================================================
# CONFIGURATION
# ============================================================

class APIConfig:
    """Configuration for API providers"""

    # API-Football (Free tier: 100 calls/day)
    API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "95c43f936816cd4389a747fd2cfe061a")
    API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
    API_FOOTBALL_QUOTA = 100  # calls/day

    # Football-Data.org (Free tier: 10 calls/minute)
    FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "")
    FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
    FOOTBALL_DATA_QUOTA = 10  # calls/minute

    # TheSportsDB (Free, unlimited)
    THESPORTSDB_KEY = "3"  # Free key
    THESPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json"

    # TheOddsAPI
    THEODDS_API_KEY = os.getenv("THEODDS_API_KEY", "")

    # OpenWeatherMap
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "01afa2183566fcf16d98b5a33c91eae1")

    # HuggingFace
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

    # Cache settings
    CACHE_TTL = 86400  # 24 hours in seconds
    CACHE_DB = "api_cache.db"


# ============================================================
# CACHE MANAGER
# ============================================================

class CacheManager:
    """SQLite-based cache for API responses"""

    def __init__(self, db_path: str = APIConfig.CACHE_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Team context cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS team_cache (
                    team TEXT NOT NULL,
                    league TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    PRIMARY KEY (team, league)
                )
            """)

            # API usage tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    date TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    calls INTEGER DEFAULT 0,
                    PRIMARY KEY (date, provider)
                )
            """)

            # Cache statistics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_stats (
                    date TEXT NOT NULL PRIMARY KEY,
                    hits INTEGER DEFAULT 0,
                    misses INTEGER DEFAULT 0
                )
            """)

            # Over markets cache (for caching calculated over/under probabilities)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS over_markets_cache (
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    match_date TEXT NOT NULL,
                    market_data TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    PRIMARY KEY (home_team, away_team, match_date)
                )
            """)

            # Complete predictions cache (for caching full calc_all_probabilities output)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions_cache (
                    cache_key TEXT NOT NULL PRIMARY KEY,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    match_date TEXT NOT NULL,
                    prediction_data TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            """)

            # Index for faster lookups by teams and date
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_teams_date
                ON predictions_cache(home_team, away_team, match_date)
            """)

            conn.commit()
            conn.close()

            # Verify tables were created
            self._verify_tables()

            logger.info(f"✅ Cache database initialized: {self.db_path}")

        except Exception as e:
            logger.error(f"❌ Error initializing cache DB: {e}")
            raise RuntimeError(f"Failed to initialize API cache database at {self.db_path}: {e}")

    def _verify_tables(self):
        """Verify that all required tables exist"""
        required_tables = ['team_cache', 'api_usage', 'cache_stats', 'over_markets_cache', 'predictions_cache']
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN (?, ?, ?, ?, ?)
            """, required_tables)

            existing_tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            missing = set(required_tables) - set(existing_tables)
            if missing:
                raise RuntimeError(f"Missing required tables: {missing}")

        except Exception as e:
            raise RuntimeError(f"Table verification failed: {e}")

    def get(self, team: str, league: str) -> Optional[Dict]:
        """Get cached data if not expired"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                team_key = team.lower()
                league_key = league.lower()

                cursor.execute("""
                    SELECT data, timestamp FROM team_cache
                    WHERE team = ? AND league = ?
                """, (team_key, league_key))

                result = cursor.fetchone()

                if not result:
                    self._log_cache_miss()
                    return None

                data_json, timestamp = result

                now = time.time()
                # Check if expired
                if now - timestamp > APIConfig.CACHE_TTL:
                    logger.info(f"⏰ Cache expired for {team} ({league})")
                    cursor.execute(
                        "DELETE FROM team_cache WHERE team = ? AND league = ?",
                        (team_key, league_key)
                    )
                    conn.commit()
                    self._log_cache_miss()
                    return None

                self._log_cache_hit()
                logger.info(f"✅ Cache HIT: {team} ({league})")
                return json.loads(data_json)

        except Exception as e:
            logger.error(f"❌ Cache get error: {e}")
            return None

    def set(self, team: str, league: str, data: Dict):
        """Store data in cache"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO team_cache (team, league, data, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (team.lower(), league.lower(), json.dumps(data), int(time.time())))

                # Auto-cleanup if cache too large (prevent unbounded growth)
                cursor.execute("SELECT COUNT(*) FROM team_cache")
                # FIX BUG #10.4: Safe array access on fetchone()
                result = cursor.fetchone()
                count = result[0] if result else 0
                if count > 1000:  # Max 1000 entries
                    # Delete 20% instead of 10% to provide buffer for race conditions
                    cleanup_count = max(100, int(count * 0.20))
                    logger.info(f"🧹 Cache size ({count}) exceeded limit, cleaning oldest {cleanup_count}")
                    cursor.execute("""
                        DELETE FROM team_cache
                        WHERE rowid IN (
                            SELECT rowid FROM team_cache
                            ORDER BY timestamp ASC
                            LIMIT ?
                        )
                    """, (cleanup_count,))

                logger.info(f"💾 Cached: {team} ({league})")

        except Exception as e:
            logger.error(f"❌ Cache set error: {e}")

    def _log_cache_hit(self):
        """Log cache hit for statistics"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()
                today = datetime.now().strftime("%Y-%m-%d")

                cursor.execute("""
                    INSERT INTO cache_stats (date, hits, misses)
                    VALUES (?, 1, 0)
                    ON CONFLICT(date) DO UPDATE SET hits = hits + 1
                """, (today,))

        except Exception as e:
            logger.error(f"❌ Error logging cache hit: {e}")

    def _log_cache_miss(self):
        """Log cache miss for statistics"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()
                today = datetime.now().strftime("%Y-%m-%d")

                cursor.execute("""
                    INSERT INTO cache_stats (date, hits, misses)
                    VALUES (?, 0, 1)
                    ON CONFLICT(date) DO UPDATE SET hits = hits, misses = misses + 1
                """, (today,))

        except Exception as e:
            logger.error(f"❌ Error logging cache miss: {e}")

    def get_over_markets(self, home_team: str, away_team: str, match_date: str) -> Optional[Dict]:
        """Get cached over markets data if not expired"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                home_key = home_team.lower()
                away_key = away_team.lower()

                cursor.execute("""
                    SELECT market_data, timestamp FROM over_markets_cache
                    WHERE home_team = ? AND away_team = ? AND match_date = ?
                """, (home_key, away_key, match_date))

                result = cursor.fetchone()

                if not result:
                    self._log_cache_miss()
                    return None

                data_json, timestamp = result

                # Check if expired (24h TTL)
                now = time.time()
                if now - timestamp > APIConfig.CACHE_TTL:
                    logger.info(f"⏰ Over markets cache expired for {home_team} vs {away_team}")
                    cursor.execute(
                        """
                        DELETE FROM over_markets_cache
                        WHERE home_team = ? AND away_team = ? AND match_date = ?
                        """,
                        (home_key, away_key, match_date)
                    )
                    conn.commit()
                    self._log_cache_miss()
                    return None

                self._log_cache_hit()
                logger.info(f"✅ Over markets cache HIT: {home_team} vs {away_team}")
                return json.loads(data_json)

        except Exception as e:
            logger.error(f"❌ Over markets cache get error: {e}")
            return None

    def set_over_markets(self, home_team: str, away_team: str, match_date: str, market_data: Dict):
        """Store over markets data in cache"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO over_markets_cache (home_team, away_team, match_date, market_data, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (home_team.lower(), away_team.lower(), match_date, json.dumps(market_data), int(time.time())))

                # Auto-cleanup if cache too large
                cursor.execute("SELECT COUNT(*) FROM over_markets_cache")
                result = cursor.fetchone()
                count = result[0] if result else 0
                if count > 5000:  # Max 5000 entries (matches can be many)
                    logger.info(f"🧹 Over markets cache size ({count}) exceeded limit, cleaning oldest 10%")
                    cursor.execute("""
                        DELETE FROM over_markets_cache
                        WHERE rowid IN (
                            SELECT rowid FROM over_markets_cache
                            ORDER BY timestamp ASC
                            LIMIT 500
                        )
                    """)

                logger.info(f"💾 Cached over markets: {home_team} vs {away_team}")

        except Exception as e:
            logger.error(f"❌ Over markets cache set error: {e}")

    def _build_prediction_cache_key(self, home_team: str, away_team: str, match_date: str,
                                     odds_1: Optional[float] = None, odds_x: Optional[float] = None,
                                     odds_2: Optional[float] = None) -> str:
        """Create a deterministic cache key (uses hashing for consistent length)."""
        cache_key_parts = [home_team.lower(), away_team.lower(), match_date]
        if odds_1 is not None and odds_x is not None and odds_2 is not None:
            cache_key_parts.extend([f"{odds_1:.2f}", f"{odds_x:.2f}", f"{odds_2:.2f}"])
        return hashlib.md5('|'.join(cache_key_parts).encode()).hexdigest()

    def get_prediction(self, home_team: str, away_team: str, match_date: str,
                       odds_1: float = None, odds_x: float = None, odds_2: float = None) -> Optional[Dict]:
        """Get cached complete prediction if not expired

        Args:
            home_team: Nome squadra casa
            away_team: Nome squadra trasferta
            match_date: Data partita (YYYY-MM-DD)
            odds_1: Quote casa (opzionale, per invalidare cache se cambiate)
            odds_x: Quote pareggio (opzionale)
            odds_2: Quote trasferta (opzionale)

        Returns:
            Prediction dict se trovato e valido, None altrimenti
        """
        try:
            # Genera cache key unica (hash MD5 per evitare collisioni e uniformare get/set)
            cache_key = self._build_prediction_cache_key(
                home_team, away_team, match_date, odds_1, odds_x, odds_2
            )

            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT prediction_data, timestamp FROM predictions_cache
                    WHERE cache_key = ?
                """, (cache_key,))

                result = cursor.fetchone()

                if not result:
                    self._log_cache_miss()
                    return None

                data_json, timestamp = result

                # Check if expired (24h TTL)
                if time.time() - timestamp > APIConfig.CACHE_TTL:
                    logger.info(f"⏰ Prediction cache expired for {home_team} vs {away_team}")
                    try:
                        cursor.execute(
                            "DELETE FROM predictions_cache WHERE cache_key = ?",
                            (cache_key,)
                        )
                        conn.commit()
                    except sqlite3.Error as delete_err:
                        logger.error(f"❌ Prediction cache delete error: {delete_err}")
                    self._log_cache_miss()
                    return None

                self._log_cache_hit()
                logger.info(f"✅ Prediction cache HIT: {home_team} vs {away_team}")
                return json.loads(data_json)

        except Exception as e:
            logger.error(f"❌ Prediction cache get error: {e}")
            return None

    def set_prediction(self, home_team: str, away_team: str, match_date: str,
                       prediction_data: Dict, odds_1: float = None, odds_x: float = None, odds_2: float = None):
        """Store complete prediction in cache

        Args:
            home_team: Nome squadra casa
            away_team: Nome squadra trasferta
            match_date: Data partita (YYYY-MM-DD)
            prediction_data: Dizionario completo con tutte le predizioni
            odds_1: Quote casa (opzionale)
            odds_x: Quote pareggio (opzionale)
            odds_2: Quote trasferta (opzionale)
        """
        try:
            # Genera cache key unica (stessa logica del get)
            cache_key = self._build_prediction_cache_key(
                home_team, away_team, match_date, odds_1, odds_x, odds_2
            )

            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO predictions_cache
                    (cache_key, home_team, away_team, match_date, prediction_data, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (cache_key, home_team.lower(), away_team.lower(), match_date,
                      json.dumps(prediction_data), int(time.time())))

                # Auto-cleanup if cache too large
                cursor.execute("SELECT COUNT(*) FROM predictions_cache")
                result = cursor.fetchone()
                count = result[0] if result else 0
                if count > 10000:  # Max 10000 predictions
                    logger.info(f"🧹 Predictions cache size ({count}) exceeded limit, cleaning oldest 10%")
                    cursor.execute("""
                        DELETE FROM predictions_cache
                        WHERE rowid IN (
                            SELECT rowid FROM predictions_cache
                            ORDER BY timestamp ASC
                            LIMIT 1000
                        )
                    """)

                logger.info(f"💾 Cached prediction: {home_team} vs {away_team}")

        except Exception as e:
            logger.error(f"❌ Prediction cache set error: {e}")

    def clear_prediction_cache(self, home_team: str = None, away_team: str = None):
        """Clear prediction cache, optionally filtered by teams

        Args:
            home_team: Se fornito, cancella solo cache per questa squadra casa
            away_team: Se fornito, cancella solo cache per questa squadra trasferta
        """
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                if home_team and away_team:
                    cursor.execute("""
                        DELETE FROM predictions_cache
                        WHERE home_team = ? AND away_team = ?
                    """, (home_team.lower(), away_team.lower()))
                    logger.info(f"🧹 Cleared prediction cache for {home_team} vs {away_team}")
                elif home_team:
                    cursor.execute("""
                        DELETE FROM predictions_cache
                        WHERE home_team = ? OR away_team = ?
                    """, (home_team.lower(), home_team.lower()))
                    logger.info(f"🧹 Cleared prediction cache for team: {home_team}")
                else:
                    cursor.execute("DELETE FROM predictions_cache")
                    logger.info("🧹 Cleared all prediction cache")

        except Exception as e:
            logger.error(f"❌ Clear prediction cache error: {e}")

    def get_stats(self) -> Dict:
        """Get cache statistics for today"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                today = datetime.now().strftime("%Y-%m-%d")

                cursor.execute("""
                    SELECT hits, misses FROM cache_stats WHERE date = ?
                """, (today,))

                result = cursor.fetchone()

                if not result:
                    return {"hits": 0, "misses": 0, "total": 0, "hit_rate": 0.0}

                hits, misses = result
                total = hits + misses
                hit_rate = (hits / total * 100) if total > 0 else 0.0

                return {
                    "hits": hits,
                    "misses": misses,
                    "total": total,
                    "hit_rate": hit_rate
                }

        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {e}")
            return {"hits": 0, "misses": 0, "total": 0, "hit_rate": 0.0}

    def cleanup_old(self, days: int = 7):
        """Remove cache entries older than N days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cutoff = int(time.time()) - (days * 86400)
                cursor.execute("DELETE FROM team_cache WHERE timestamp < ?", (cutoff,))
                deleted_team = cursor.rowcount

                cursor.execute("DELETE FROM over_markets_cache WHERE timestamp < ?", (cutoff,))
                deleted_over_markets = cursor.rowcount

                cursor.execute("DELETE FROM predictions_cache WHERE timestamp < ?", (cutoff,))
                deleted_predictions = cursor.rowcount

                conn.commit()

                logger.info(
                    f"🧹 Cleaned {deleted_team} team cache entries, "
                    f"{deleted_over_markets} over markets cache entries e "
                    f"{deleted_predictions} predictions cache entries "
                    f"older than {days} day(s)"
                )

        except Exception as e:
            logger.error(f"❌ Cache cleanup error: {e}")


# ============================================================
# QUOTA MANAGER
# ============================================================

class QuotaManager:
    """Track API usage and enforce quotas"""

    def __init__(self, cache_db: str = APIConfig.CACHE_DB):
        self.cache_db = cache_db

    def can_use(self, provider: str, calls: int = 1) -> bool:
        """Check if quota allows N calls"""
        used = self.get_usage(provider)

        if provider == "api-football":
            return (used + calls) <= APIConfig.API_FOOTBALL_QUOTA
        elif provider == "football-data":
            # Check per-minute quota (simplified: assume 1 call/6 seconds)
            return True  # Let it through, rate limiting handled by API

        return True  # Other providers unlimited

    def log_usage(self, provider: str, calls: int = 1):
        """Log API calls"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")

            cursor.execute("""
                INSERT INTO api_usage (date, provider, calls)
                VALUES (?, ?, ?)
                ON CONFLICT(date, provider) DO UPDATE SET calls = calls + ?
            """, (today, provider, calls, calls))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Error logging API usage: {e}")

    def get_usage(self, provider: str) -> int:
        """Get today's usage for provider"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")

            cursor.execute("""
                SELECT calls FROM api_usage WHERE date = ? AND provider = ?
            """, (today, provider))

            result = cursor.fetchone()
            conn.close()

            # FIX BUG #10.5: Safe array access - already has check but make explicit
            return result[0] if result and len(result) > 0 else 0

        except Exception as e:
            logger.error(f"❌ Error getting API usage: {e}")
            return 0

    def get_all_usage(self) -> Dict[str, int]:
        """Get today's usage for all providers"""
        providers = ["api-football", "football-data", "thesportsdb"]
        return {p: self.get_usage(p) for p in providers}


# ============================================================
# API PROVIDERS
# ============================================================

class APIFootballProvider:
    """API-Football.com provider (Free: 100 calls/day)"""

    def __init__(self, api_key: str = APIConfig.API_FOOTBALL_KEY):
        self.api_key = api_key
        self.base_url = APIConfig.API_FOOTBALL_BASE

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """Make HTTP request to API"""
        if not self.api_key:
            logger.warning("⚠️ API-Football key not configured")
            return None

        try:
            query = urllib.parse.urlencode(params or {})
            url = f"{self.base_url}/{endpoint}"
            if query:
                url = f"{url}?{query}"
            headers = {
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": "v3.football.api-sports.io"
            }

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data

        except Exception as e:
            logger.error(f"❌ API-Football request error: {e}")
            return None

    def search_team(self, team_name: str, league_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Search a team by name and optionally filter by league hint."""
        response = self._request("teams", {"search": team_name})
        if not response or not response.get("response"):
            return None

        entries = response["response"]
        if league_hint:
            league_hint = league_hint.lower()
            filtered = []
            for entry in entries:
                team_league = (entry.get("team") or {}).get("name", "")
                venue_league = (entry.get("venue") or {}).get("city", "")
                if league_hint in team_league.lower() or league_hint in venue_league.lower():
                    filtered.append(entry)
            if filtered:
                entries = filtered

        return entries[0]

    def get_team_info(self, team_name: str, league_name: str) -> Optional[Dict]:
        """Get team context (position, form, etc.)"""
        entry = self.search_team(team_name, league_name)
        if not entry:
            return None

        team = entry.get("team", {})
        venue = entry.get("venue", {})
        team_id = team.get("id")

        form = "DDDDD"
        if team_id:
            form = self.get_recent_form(team_id) or form

        context = {
            "team_name": team.get("name", team_name),
            "league": league_name or team.get("country"),
            "stadium": venue.get("name"),
            "description": team.get("national", False) and "National team" or None,
            "formed_year": team.get("founded"),
            "country": team.get("country"),
            "style": "Possesso" if not form else self._infer_style_from_form(form),
            "typical_position": "mid",
            "form": form,
            "api_football_team_id": team_id
        }

        return {
            "data": context,
            "api_calls_used": 2 if team_id else 1
        }

    def get_recent_form(self, team_id: int, last: int = 5) -> Optional[str]:
        """Return last `last` fixtures form string (W/D/L)."""
        fixtures = self._request("fixtures", {"team": team_id, "last": last})
        if not fixtures or not fixtures.get("response"):
            return None

        results = []
        for match in fixtures["response"]:
            teams = match.get("teams", {})
            if teams.get("home", {}).get("id") == team_id:
                winner = teams["home"].get("winner")
            elif teams.get("away", {}).get("id") == team_id:
                winner = teams["away"].get("winner")
            else:
                winner = None

            if winner is True:
                results.append("W")
            elif winner is False:
                results.append("L")
            else:
                results.append("D")

        return "".join(results) or None

    def get_team_match_data(self, team_name: str, season: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return injuries and basic match insights for a team."""
        entry = self.search_team(team_name)
        if not entry:
            return None

        team = entry.get("team", {})
        team_id = team.get("id")
        if not team_id:
            return None

        season = season or datetime.utcnow().year
        injuries = self.get_injuries(team_id, season)
        lineup = self.get_predicted_lineup(team_id, season)

        return {
            "team_id": team_id,
            "injuries": injuries,
            "suspensions": [inj for inj in injuries if inj.get("type", "").lower().startswith("susp")],
            "lineup": lineup,
            "form": self.get_recent_form(team_id) or "DDDDD"
        }

    def get_injuries(self, team_id: int, season: Optional[str]) -> List[Dict[str, Any]]:
        """Fetch injuries/suspensions list."""
        injuries_payload = self._request("injuries", {"team": team_id, "season": season})
        injuries: List[Dict[str, Any]] = []
        if not injuries_payload:
            return injuries

        for entry in injuries_payload.get("response", []):
            player = entry.get("player", {}) or {}
            fixture = entry.get("fixture", {}) or {}
            league = entry.get("league", {}) or {}

            injuries.append({
                "player": player.get("name"),
                "type": player.get("type"),
                "reason": player.get("reason"),
                "fixture_date": fixture.get("date"),
                "league": league.get("name"),
            })

        return injuries

    def get_predicted_lineup(self, team_id: int, season: Optional[str]) -> Dict[str, Any]:
        """Use the next fixture lineup if available."""
        lineup_payload = self._request("fixtures", {"team": team_id, "next": 1, "season": season})
        if not lineup_payload or not lineup_payload.get("response"):
            return {}

        fixture = lineup_payload["response"][0]
        lineups = fixture.get("lineups") or []
        if not lineups:
            return {}

        lineup = lineups[0]
        return {
            "formation": lineup.get("formation"),
            "coach": (lineup.get("coach") or {}).get("name"),
            "startXI": [
                {
                    "number": player.get("player", {}).get("number"),
                    "name": player.get("player", {}).get("name"),
                    "pos": player.get("position")
                }
                for player in lineup.get("startXI", [])
            ]
        }

    def get_head_to_head_stats(
        self,
        home_team: str,
        away_team: str,
        last: int = 10,
        season: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return summarized head-to-head stats between two teams."""
        home_entry = self.search_team(home_team)
        away_entry = self.search_team(away_team)
        if not home_entry or not away_entry:
            return None

        home_id = (home_entry.get("team") or {}).get("id")
        away_id = (away_entry.get("team") or {}).get("id")
        if not home_id or not away_id:
            return None

        params = {"h2h": f"{home_id}-{away_id}", "last": max(1, int(last))}
        if season:
            params["season"] = season

        fixtures = self._request("fixtures/headtohead", params)
        if not fixtures or not fixtures.get("response"):
            return None

        return self._summarize_h2h(fixtures["response"], home_id, away_id, last)

    def _summarize_h2h(
        self,
        fixtures: List[Dict[str, Any]],
        home_id: int,
        away_id: int,
        last: int
    ) -> Dict[str, Any]:
        total = len(fixtures)
        if total == 0:
            return {}

        home_wins = draws = away_wins = 0
        goals_home: List[int] = []
        goals_away: List[int] = []
        recent_matches: List[Dict[str, Any]] = []

        for fixture in fixtures[:last]:
            teams = fixture.get("teams", {}) or {}
            fixture_info = fixture.get("fixture") or {}
            goals = fixture.get("goals") or {}

            goals_h = goals.get("home")
            goals_a = goals.get("away")
            if goals_h is None or goals_a is None:
                continue

            record = {
                "date": fixture_info.get("date"),
                "home": (teams.get("home") or {}).get("name"),
                "away": (teams.get("away") or {}).get("name"),
                "score": f"{goals_h}-{goals_a}",
                "venue": (fixture_info.get("venue") or {}).get("name"),
            }

            if (teams.get("home") or {}).get("id") == home_id:
                goals_home.append(goals_h)
                goals_away.append(goals_a)
                if goals_h > goals_a:
                    home_wins += 1
                    record["result"] = "H"
                elif goals_h < goals_a:
                    away_wins += 1
                    record["result"] = "A"
                else:
                    draws += 1
                    record["result"] = "D"
            else:
                goals_home.append(goals_a)
                goals_away.append(goals_h)
                if goals_a > goals_h:
                    home_wins += 1
                    record["result"] = "H"
                elif goals_a < goals_h:
                    away_wins += 1
                    record["result"] = "A"
                else:
                    draws += 1
                    record["result"] = "D"

            recent_matches.append(record)

        avg_home = sum(goals_home) / len(goals_home) if goals_home else 0.0
        avg_away = sum(goals_away) / len(goals_away) if goals_away else 0.0

        return {
            "total_matches": total,
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
            "home_win_pct": round(home_wins / total * 100, 1) if total else 0.0,
            "draw_pct": round(draws / total * 100, 1) if total else 0.0,
            "away_win_pct": round(away_wins / total * 100, 1) if total else 0.0,
            "avg_goals_home": round(avg_home, 2),
            "avg_goals_away": round(avg_away, 2),
            "avg_total_goals": round(avg_home + avg_away, 2),
            "recent_matches": recent_matches[: min(len(recent_matches), 5)],
        }

    @staticmethod
    def _infer_style_from_form(form: str) -> str:
        wins = form.count("W")
        losses = form.count("L")
        if wins >= 3:
            return "Aggressiva"
        if losses >= 3:
            return "Difensiva"
        return "Possesso"


class FootballDataProvider:
    """Football-Data.org provider (Free: 10 calls/min)"""

    def __init__(self, api_key: str = APIConfig.FOOTBALL_DATA_KEY):
        self.api_key = api_key
        self.base_url = APIConfig.FOOTBALL_DATA_BASE

    def _request(self, endpoint: str) -> Optional[Dict]:
        """Make HTTP request to API"""
        if not self.api_key:
            logger.warning("⚠️ Football-Data.org key not configured")
            return None

        try:
            url = f"{self.base_url}/{endpoint}"
            headers = {"X-Auth-Token": self.api_key}

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data

        except Exception as e:
            logger.error(f"❌ Football-Data request error: {e}")
            return None

    def get_team_info(self, team_name: str, league_name: str) -> Optional[Dict]:
        """Get team context from standings"""
        if not self.api_key:
            return None

        response = self._request("teams", {"name": team_name})
        if not response or not response.get("teams"):
            return None

        team = response["teams"][0]
        context = {
            "team_name": team.get("name", team_name),
            "league": league_name or team.get("area", {}).get("name"),
            "stadium": team.get("venue"),
            "description": team.get("shortName"),
            "formed_year": team.get("founded"),
            "country": team.get("area", {}).get("name"),
            "style": "Possesso",
            "typical_position": "mid"
        }

        return {
            "data": context,
            "api_calls_used": 1
        }


class TheSportsDBProvider:
    """TheSportsDB provider (Free, unlimited)"""

    def __init__(self, api_key: str = APIConfig.THESPORTSDB_KEY):
        self.api_key = api_key
        self.base_url = f"{APIConfig.THESPORTSDB_BASE}/{api_key}"

    def _request(self, endpoint: str) -> Optional[Dict]:
        """Make HTTP request to API"""
        try:
            url = f"{self.base_url}/{endpoint}"

            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data

        except Exception as e:
            logger.error(f"❌ TheSportsDB request error: {e}")
            return None

    def search_team(self, team_name: str) -> Optional[Dict]:
        """Search team by name"""
        endpoint = f"searchteams.php?t={urllib.parse.quote(team_name)}"
        data = self._request(endpoint)

        if data and data.get("teams"):
            # FIX BUG #10.6: Safe array access on data["teams"]
            teams = data.get("teams", [])
            return teams[0] if teams and len(teams) > 0 else None

        return None


# ============================================================
# MAIN API MANAGER
# ============================================================

class APIManager:
    """
    Main API Manager with intelligent fallback cascade:
    1. Check cache (24h)
    2. Check quota
    3. Try providers in order
    4. Update cache
    5. Fallback to LEVEL 1 if all fail
    """

    def __init__(self):
        self.cache = CacheManager()
        self.quota = QuotaManager()

        # Initialize providers
        self.providers = {
            "thesportsdb": TheSportsDBProvider(),
            "api-football": APIFootballProvider(),
            "football-data": FootballDataProvider()
        }
        self.provider_health = {
            name: ProviderHealth() for name in self.providers.keys()
        }

        logger.info("✅ API Manager initialized")

    def get_team_context(
        self,
        team: str,
        league: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Get team context with intelligent fallback

        Returns:
            Dict with:
            - source: "cache", "api", "fallback"
            - data: team context data
            - provider: which provider was used
        """
        # 1. Check cache (unless force refresh)
        if not force_refresh:
            cached = self.cache.get(team, league)
            if cached:
                return {
                    "source": "cache",
                    "data": cached,
                    "provider": "cache",
                    "api_calls_used": 0
                }

        # 2. Try API providers (with quota check)
        api_data = self._fetch_from_apis(team, league)

        if api_data:
            # Cache the result
            self.cache.set(team, league, api_data["data"])
            return api_data

        # 3. Fallback to basic data
        logger.warning(f"⚠️ All APIs failed for {team}, using fallback")
        return {
            "source": "fallback",
            "data": self._fallback_data(team, league),
            "provider": "fallback",
            "api_calls_used": 0
        }

    def _mark_provider_success(self, provider: str, start_time: float):
        health = self.provider_health.get(provider)
        if not health:
            return
        latency_ms = (time.perf_counter() - start_time) * 1000
        health.mark_success(latency_ms=latency_ms)

    def _mark_provider_failure(self, provider: str, error: str):
        health = self.provider_health.get(provider)
        if not health:
            return
        health.mark_failure(error)

    def _fetch_from_apis(self, team: str, league: str) -> Optional[Dict]:
        """Try to fetch from API providers"""

        # Try TheSportsDB first (free, unlimited)
        if self.quota.can_use("thesportsdb", calls=1):
            logger.info(f"📡 Trying TheSportsDB for {team}...")

            try:
                start = time.perf_counter()
                provider = self.providers["thesportsdb"]
                team_data = provider.search_team(team)

                if team_data:
                    self.quota.log_usage("thesportsdb", calls=1)
                    self._mark_provider_success("thesportsdb", start)

                    # Extract useful info
                    context = {
                        "team_name": team_data.get("strTeam", team),
                        "league": team_data.get("strLeague", league),
                        "stadium": team_data.get("strStadium"),
                        "description": team_data.get("strDescriptionEN"),
                        "formed_year": team_data.get("intFormedYear"),
                        "country": team_data.get("strCountry"),
                        # For auto-detection
                        "style": self._infer_style_from_description(
                            team_data.get("strDescriptionEN", "")
                        ),
                        "typical_position": "mid"  # Default, would need standings
                    }

                    logger.info(f"✅ TheSportsDB success for {team}")
                    return {
                        "source": "api",
                        "data": context,
                        "provider": "thesportsdb",
                        "api_calls_used": 1
                    }
            except Exception as e:
                self._mark_provider_failure("thesportsdb", str(e))
                logger.error(f"❌ TheSportsDB error: {e}")

        # Try API-Football if key available
        api_football = self.providers.get("api-football")
        if api_football and api_football.api_key and self.quota.can_use("api-football", calls=1):
            logger.info(f"📡 Trying API-Football for {team}...")
            try:
                start = time.perf_counter()
                info = api_football.get_team_info(team, league)
                if info:
                    calls_used = info.get("api_calls_used", 1)
                    self.quota.log_usage("api-football", calls=calls_used)
                    self._mark_provider_success("api-football", start)
                    return {
                        "source": "api",
                        "data": info["data"],
                        "provider": "api-football",
                        "api_calls_used": calls_used
                    }
            except Exception as exc:
                self._mark_provider_failure("api-football", str(exc))
                logger.error(f"❌ API-Football error: {exc}")

        # Try Football-Data.org as fallback premium source
        football_data = self.providers.get("football-data")
        if football_data and football_data.api_key and self.quota.can_use("football-data", calls=1):
            logger.info(f"📡 Trying Football-Data.org for {team}...")
            try:
                start = time.perf_counter()
                info = football_data.get_team_info(team, league)
                if info:
                    self.quota.log_usage("football-data", calls=info.get("api_calls_used", 1))
                    self._mark_provider_success("football-data", start)
                    return {
                        "source": "api",
                        "data": info["data"],
                        "provider": "football-data",
                        "api_calls_used": info.get("api_calls_used", 1)
                    }
            except Exception as exc:
                self._mark_provider_failure("football-data", str(exc))
                logger.error(f"❌ Football-Data error: {exc}")

        # Other providers not implemented in free tier
        # Would add API-Football and Football-Data.org here

        return None

    def _infer_style_from_description(self, description: str) -> str:
        """Infer tactical style from team description"""
        if not description:
            return "Possesso"

        desc_lower = description.lower()

        # Keywords for styles
        if any(word in desc_lower for word in ["attacking", "offensive", "aggressive", "pressing"]):
            return "Pressing Alto"
        elif any(word in desc_lower for word in ["defensive", "solid", "disciplined"]):
            return "Difensiva"
        elif any(word in desc_lower for word in ["counter", "fast", "quick", "pace"]):
            return "Contropiede"
        else:
            return "Possesso"  # Default

    def _fallback_data(self, team: str, league: str) -> Dict:
        """Generate fallback data when APIs fail"""
        return {
            "team_name": team,
            "league": league,
            "style": "Possesso",  # Safe default
            "typical_position": "mid",
            "source": "fallback"
        }

    def get_status(self) -> Dict:
        """Get API manager status"""
        cache_stats = self.cache.get_stats()
        usage = self.quota.get_all_usage()

        return {
            "cache": cache_stats,
            "quota": {
                "api-football": {
                    "used": usage.get("api-football", 0),
                    "total": APIConfig.API_FOOTBALL_QUOTA,
                    "remaining": APIConfig.API_FOOTBALL_QUOTA - usage.get("api-football", 0)
                },
                "football-data": {
                    "used": usage.get("football-data", 0),
                    "note": "10 calls/minute limit"
                },
                "thesportsdb": {
                    "used": usage.get("thesportsdb", 0),
                    "note": "Unlimited (free)"
                }
            },
            "provider_health": {
                name: health.to_dict()
                for name, health in self.provider_health.items()
            }
        }

    def cleanup(self):
        """Cleanup old cache entries"""
        self.cache.cleanup_old(days=7)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def test_api_manager():
    """Test API Manager functionality"""
    print("=" * 70)
    print("TEST: API Manager")
    print("=" * 70)

    api = APIManager()

    # Test 1: Get team context
    print("\nTest 1: Get team context (Inter)")
    result = api.get_team_context("Inter", "Serie A")
    print(f"  Source: {result['source']}")
    print(f"  Provider: {result['provider']}")
    print(f"  Data: {result['data']}")

    # Test 2: Cache hit
    print("\nTest 2: Get same team (should hit cache)")
    result2 = api.get_team_context("Inter", "Serie A")
    print(f"  Source: {result2['source']}")
    print(f"  Cache hit rate: {api.cache.get_stats()['hit_rate']:.1f}%")

    # Test 3: Status
    print("\nTest 3: API Manager status")
    status = api.get_status()
    print(f"  Cache stats: {status['cache']}")
    print(f"  Quota: {status['quota']}")

    print("\n" + "=" * 70)
    print("✅ Tests completed")
    print("=" * 70)


if __name__ == "__main__":
    test_api_manager()
