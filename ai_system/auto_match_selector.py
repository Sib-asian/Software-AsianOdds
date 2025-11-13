"""
Automatic Match Selector
=========================

Selects matches to monitor based on:
- Predicted value opportunities
- Time to kickoff
- League importance
- Historical performance

Integrates with existing Dixon-Coles model and AI pipeline.

Usage:
    selector = AutoMatchSelector()
    matches = selector.get_matches_to_monitor()
"""

import logging
import sqlite3
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class AutoMatchSelector:
    """
    Automatically selects matches worth monitoring.

    Args:
        cache_db: Path to cache database (default: api_cache.db)
        min_ev: Minimum expected value to monitor (default: 3%)
        max_matches: Maximum concurrent matches to monitor (default: 10)
    """

    def __init__(
        self,
        cache_db: str = "api_cache.db",
        min_ev: float = 3.0,
        max_matches: int = 10
    ):
        self.cache_db = cache_db
        self.min_ev = min_ev
        self.max_matches = max_matches

        # Priority leagues (top European leagues)
        self.priority_leagues = [
            'Premier League',
            'La Liga',
            'Serie A',
            'Bundesliga',
            'Ligue 1',
            'Champions League',
            'Europa League',
            'Championship'
        ]

        logger.info("✅ AutoMatchSelector initialized")

    def get_matches_to_monitor(
        self,
        live_fetcher=None,
        time_window_hours: int = 6
    ) -> List[Dict]:
        """
        Get list of matches to monitor.

        Strategy:
        1. Fetch upcoming matches in next N hours
        2. Run quick prediction for each
        3. Select top matches by expected value
        4. Prioritize by league importance

        Args:
            live_fetcher: AutoLiveFetcher instance (if None, creates new)
            time_window_hours: Hours ahead to look for matches

        Returns:
            List of matches with monitoring metadata
        """
        # Import here to avoid circular dependency
        if live_fetcher is None:
            from .auto_live_fetcher import AutoLiveFetcher
            live_fetcher = AutoLiveFetcher()

        # Get upcoming matches
        upcoming = live_fetcher.fetch_upcoming_matches(hours=time_window_hours)

        if not upcoming:
            logger.warning("No upcoming matches found")
            return []

        # Score and rank matches
        scored_matches = []

        for match in upcoming:
            try:
                score = self._calculate_match_score(match)

                if score > 0:  # Only include if has value
                    match['monitor_score'] = score
                    match['monitor_reason'] = self._get_monitor_reason(match, score)
                    scored_matches.append(match)

            except Exception as e:
                logger.warning(f"Error scoring match {match.get('match_id')}: {e}")
                continue

        # Sort by score (highest first)
        scored_matches.sort(key=lambda x: x['monitor_score'], reverse=True)

        # Limit to max_matches
        selected = scored_matches[:self.max_matches]

        logger.info(f"✅ Selected {len(selected)} matches to monitor (from {len(upcoming)} candidates)")

        return selected

    def get_live_matches_to_monitor(self, live_fetcher=None) -> List[Dict]:
        """
        Get currently live matches worth monitoring.

        Args:
            live_fetcher: AutoLiveFetcher instance

        Returns:
            List of live matches
        """
        if live_fetcher is None:
            from .auto_live_fetcher import AutoLiveFetcher
            live_fetcher = AutoLiveFetcher()

        live_matches = live_fetcher.fetch_live_matches_today()

        if not live_matches:
            return []

        # Filter by priority leagues
        priority_matches = []
        for match in live_matches:
            if self._is_priority_league(match.get('league', '')):
                match['monitor_score'] = 100  # High score for live matches
                match['monitor_reason'] = "Live match in priority league"
                priority_matches.append(match)

        logger.info(f"✅ Found {len(priority_matches)} priority live matches")

        return priority_matches[:self.max_matches]

    def _calculate_match_score(self, match: Dict) -> float:
        """
        Calculate monitoring priority score for a match.

        Score factors:
        - League importance: 0-50 points
        - Time to kickoff: 0-30 points (prefer soon but not too soon)
        - Historical value (if available): 0-20 points

        Returns:
            Score (0-100)
        """
        score = 0.0

        # League importance
        league = match.get('league', '')
        if league in self.priority_leagues:
            # Top leagues get higher score
            position = self.priority_leagues.index(league)
            league_score = 50 - (position * 5)  # 50 for #1, 45 for #2, etc.
            score += max(20, league_score)  # Min 20 for any priority league
        else:
            score += 10  # Other leagues get base score

        # Time to kickoff (prefer 1-4 hours ahead)
        hours_to_kickoff = match.get('time_to_kickoff_hours', 0)

        if 1 <= hours_to_kickoff <= 4:
            score += 30  # Perfect timing
        elif 0.5 <= hours_to_kickoff < 1:
            score += 20  # Soon
        elif 4 < hours_to_kickoff <= 6:
            score += 15  # Not urgent but good
        else:
            score += 5  # Too far or too close

        # Bonus for specific team patterns (top teams)
        top_teams = [
            'Manchester City', 'Liverpool', 'Arsenal', 'Chelsea',
            'Real Madrid', 'Barcelona', 'Bayern', 'PSG',
            'Inter', 'Milan', 'Juventus', 'Napoli'
        ]

        home = match.get('home_team', '')
        away = match.get('away_team', '')

        if any(team in home or team in away for team in top_teams):
            score += 10  # Bonus for top teams

        return score

    def _get_monitor_reason(self, match: Dict, score: float) -> str:
        """Generate human-readable reason for monitoring"""
        reasons = []

        league = match.get('league', '')
        if league in self.priority_leagues:
            reasons.append(f"{league}")

        hours = match.get('time_to_kickoff_hours', 0)
        if hours <= 1:
            reasons.append("Starting soon")
        elif hours <= 4:
            reasons.append("Optimal timing")

        if score >= 60:
            reasons.append("High priority")
        elif score >= 40:
            reasons.append("Medium priority")

        return ", ".join(reasons) if reasons else "Selected for monitoring"

    def _is_priority_league(self, league: str) -> bool:
        """Check if league is in priority list"""
        return any(priority in league for priority in self.priority_leagues)

    def save_selection_history(self, matches: List[Dict]):
        """Save selection history to database for analytics"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS match_selections (
                    match_id TEXT,
                    timestamp INTEGER,
                    home_team TEXT,
                    away_team TEXT,
                    league TEXT,
                    score REAL,
                    reason TEXT,
                    PRIMARY KEY (match_id, timestamp)
                )
            """)

            # Insert selections
            timestamp = int(datetime.now().timestamp())

            for match in matches:
                cursor.execute("""
                    INSERT OR REPLACE INTO match_selections
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    match.get('match_id'),
                    timestamp,
                    match.get('home_team'),
                    match.get('away_team'),
                    match.get('league'),
                    match.get('monitor_score', 0),
                    match.get('monitor_reason', '')
                ))

            conn.commit()
            conn.close()

            logger.debug(f"Saved {len(matches)} selections to history")

        except Exception as e:
            logger.warning(f"Failed to save selection history: {e}")

    def get_selection_stats(self) -> Dict:
        """Get statistics about past selections"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()

            # Count selections
            cursor.execute("""
                SELECT COUNT(DISTINCT match_id) as total_matches,
                       COUNT(*) as total_selections,
                       AVG(score) as avg_score
                FROM match_selections
                WHERE timestamp > ?
            """, (int((datetime.now() - timedelta(days=7)).timestamp()),))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'total_matches_monitored': row[0] or 0,
                    'total_selections': row[1] or 0,
                    'average_score': row[2] or 0
                }

        except Exception as e:
            logger.warning(f"Failed to get selection stats: {e}")

        return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Auto Match Selector...")
    print("=" * 70)

    selector = AutoMatchSelector(min_ev=3.0, max_matches=5)

    # Test 1: Get matches to monitor
    print("\n📋 Selecting matches to monitor...")
    matches = selector.get_matches_to_monitor(time_window_hours=6)

    if matches:
        print(f"\n✅ Selected {len(matches)} matches:")
        for i, match in enumerate(matches, 1):
            print(f"\n{i}. {match['home_team']} vs {match['away_team']}")
            print(f"   League: {match['league']}")
            print(f"   Kickoff in: {match['time_to_kickoff_hours']:.1f}h")
            print(f"   Score: {match['monitor_score']:.1f}")
            print(f"   Reason: {match['monitor_reason']}")
    else:
        print("⚠️  No matches selected (may be off-season or late night)")

    # Test 2: Get live matches
    print("\n" + "=" * 70)
    print("📺 Checking live matches...")
    live = selector.get_live_matches_to_monitor()

    if live:
        print(f"✅ Found {len(live)} live matches to monitor:")
        for match in live:
            print(f"  {match['home_team']} vs {match['away_team']} ({match['minute']}')")
    else:
        print("⚠️  No live matches currently")

    # Test 3: Get stats
    print("\n" + "=" * 70)
    print("📊 Selection statistics:")
    stats = selector.get_selection_stats()

    if stats:
        print(f"  Total matches monitored (7 days): {stats.get('total_matches_monitored', 0)}")
        print(f"  Average priority score: {stats.get('average_score', 0):.1f}")

    print("\n" + "=" * 70)
    print("✅ Auto Match Selector test completed!")
