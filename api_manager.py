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

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import urllib.request
import urllib.parse
import urllib.error

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

class APIConfig:
    """Configuration for API providers"""

    # API-Football (Free tier: 100 calls/day)
    API_FOOTBALL_KEY = "95c43f936816cd4389a747fd2cfe061a"
    API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
    API_FOOTBALL_QUOTA = 100  # calls/day

    # Football-Data.org (Free tier: 10 calls/minute)
    FOOTBALL_DATA_KEY = ""  # User will add this
    FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
    FOOTBALL_DATA_QUOTA = 10  # calls/minute

    # TheSportsDB (Free, unlimited)
    THESPORTSDB_KEY = "3"  # Free key
    THESPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json"

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
        required_tables = ['team_cache', 'api_usage', 'cache_stats', 'over_markets_cache']
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN (?, ?, ?, ?)
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

                cursor.execute("""
                    SELECT data, timestamp FROM team_cache
                    WHERE team = ? AND league = ?
                """, (team.lower(), league.lower()))

                result = cursor.fetchone()

                if not result:
                    self._log_cache_miss()
                    return None

                data_json, timestamp = result

                # Check if expired
                if time.time() - timestamp > APIConfig.CACHE_TTL:
                    logger.info(f"⏰ Cache expired for {team} ({league})")
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
                    logger.info(f"🧹 Cache size ({count}) exceeded limit, cleaning oldest 10%")
                    cursor.execute("""
                        DELETE FROM team_cache
                        WHERE rowid IN (
                            SELECT rowid FROM team_cache
                            ORDER BY timestamp ASC
                            LIMIT 100
                        )
                    """)

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

                cursor.execute("""
                    SELECT market_data, timestamp FROM over_markets_cache
                    WHERE home_team = ? AND away_team = ? AND match_date = ?
                """, (home_team.lower(), away_team.lower(), match_date))

                result = cursor.fetchone()

                if not result:
                    self._log_cache_miss()
                    return None

                data_json, timestamp = result

                # Check if expired (24h TTL)
                if time.time() - timestamp > APIConfig.CACHE_TTL:
                    logger.info(f"⏰ Over markets cache expired for {home_team} vs {away_team}")
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

    def get_stats(self) -> Dict:
        """Get cache statistics for today"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")

            cursor.execute("""
                SELECT hits, misses FROM cache_stats WHERE date = ?
            """, (today,))

            result = cursor.fetchone()
            conn.close()

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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff = int(time.time()) - (days * 86400)
            cursor.execute("DELETE FROM team_cache WHERE timestamp < ?", (cutoff,))

            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"🧹 Cleaned {deleted} old cache entries")

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

    def _request(self, endpoint: str) -> Optional[Dict]:
        """Make HTTP request to API"""
        if not self.api_key:
            logger.warning("⚠️ API-Football key not configured")
            return None

        try:
            url = f"{self.base_url}/{endpoint}"
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

    def get_team_info(self, team_name: str, league_name: str) -> Optional[Dict]:
        """Get team context (position, form, etc.)"""
        # For free tier, we'd need league ID first
        # Simplified: return None, use other providers
        logger.info(f"ℹ️ API-Football requires league ID (not implemented in free tier)")
        return None


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
        # Simplified implementation
        logger.info(f"ℹ️ Football-Data.org integration not fully implemented (free tier)")
        return None


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

    def _fetch_from_apis(self, team: str, league: str) -> Optional[Dict]:
        """Try to fetch from API providers"""

        # Try TheSportsDB first (free, unlimited)
        if self.quota.can_use("thesportsdb", calls=1):
            logger.info(f"📡 Trying TheSportsDB for {team}...")

            try:
                provider = self.providers["thesportsdb"]
                team_data = provider.search_team(team)

                if team_data:
                    self.quota.log_usage("thesportsdb", calls=1)

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
                logger.error(f"❌ TheSportsDB error: {e}")

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
