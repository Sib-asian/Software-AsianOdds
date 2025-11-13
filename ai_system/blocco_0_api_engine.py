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

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

# Imports from existing system
try:
    from api_manager import APIManager, CacheManager, QuotaManager
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from api_manager import APIManager, CacheManager, QuotaManager

from .config import AIConfig

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

        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "fallbacks": 0
        }

        logger.info("✅ API Data Engine initialized")

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

            logger.info(
                f"✅ Data collected for {match['home']} vs {match['away']}: "
                f"quality={quality_score:.2f}, importance={importance:.2f}"
            )

            return enriched

        except Exception as e:
            logger.error(f"❌ Error collecting data for match: {e}")
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
        try:
            # Check quota
            if not self.quota.can_use("api-football", calls=1):
                logger.warning("⚠️ API-Football quota exhausted")
                return None

            # TODO: Implementare chiamate specifiche API-Football
            # Per ora ritorna None (da implementare quando necessario)
            logger.info(f"ℹ️ API-Football enrichment not yet implemented for {team}")
            return None

        except Exception as e:
            logger.error(f"❌ Error enriching with API-Football: {e}")
            return None

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
            # TODO: Implementare raccolta dati match-specific
            # - H2H history da database
            # - Odds history da API
            # - Injuries/suspensions da API-Football
            # - Lineup prediction

            logger.info(f"ℹ️ Match-specific data collection not fully implemented")

        except Exception as e:
            logger.error(f"❌ Error getting match-specific data: {e}")

        return match_data

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
                "data": self._get_fallback_team_data(
                    match.get("home", "Unknown"),
                    match.get("league", "Unknown")
                ),
                "api_calls_used": 0
            },
            "away_context": {
                "source": "fallback",
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
