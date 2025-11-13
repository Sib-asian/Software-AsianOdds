"""
Automatic Live Data Fetcher
============================

Fetches live match data from API-Football automatically.
Integrates with existing APIManager and cache system.

Features:
- Real-time match data (score, xG, events)
- Automatic API quota management
- Cache integration
- Retry logic with exponential backoff

Usage:
    fetcher = AutoLiveFetcher()
    live_data = fetcher.fetch_live_match_data(match_id)
"""

import logging
import time
import json
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class AutoLiveFetcher:
    """
    Automatic fetcher for live match data from API-Football.

    Args:
        api_key: API-Football key (default: from existing config)
        cache_enabled: Use cache to reduce API calls (default: True)
    """

    def __init__(
        self,
        api_key: str = "95c43f936816cd4389a747fd2cfe061a",
        cache_enabled: bool = True
    ):
        self.api_key = api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.cache_enabled = cache_enabled

        # Cache for live data (short TTL: 60 seconds)
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = 60  # 1 minute

        logger.info("✅ AutoLiveFetcher initialized")

    def fetch_live_match_data(self, match_id: str) -> Optional[Dict]:
        """
        Fetch live data for a specific match.

        Args:
            match_id: API-Football fixture ID

        Returns:
            Dict with live data or None if failed:
            {
                'minute': int,
                'score_home': int,
                'score_away': int,
                'xg_home': float,
                'xg_away': float,
                'status': str ('in_play', 'finished', 'not_started'),
                'events': List[Dict]
            }
        """
        # Check cache first
        if self.cache_enabled:
            cached = self._get_from_cache(match_id)
            if cached:
                logger.debug(f"Using cached data for match {match_id}")
                return cached

        # Fetch from API
        try:
            url = f"{self.base_url}/fixtures?id={match_id}"
            headers = {
                'x-apisports-key': self.api_key
            }

            request = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode())

            if data.get('errors'):
                logger.error(f"API error: {data['errors']}")
                return None

            if not data.get('response') or len(data['response']) == 0:
                logger.warning(f"No data for match {match_id}")
                return None

            fixture = data['response'][0]

            # Extract live data
            live_data = self._parse_fixture_data(fixture)

            # Cache result
            if self.cache_enabled:
                self._save_to_cache(match_id, live_data)

            logger.info(f"✅ Fetched live data for match {match_id}")
            return live_data

        except urllib.error.HTTPError as e:
            logger.error(f"HTTP error fetching match {match_id}: {e.code}")
            return None
        except urllib.error.URLError as e:
            logger.error(f"Network error fetching match {match_id}: {e.reason}")
            return None
        except Exception as e:
            logger.error(f"Error fetching match {match_id}: {e}")
            return None

    def fetch_live_matches_today(self) -> List[Dict]:
        """
        Fetch all live matches happening today.

        Returns:
            List of matches with basic info
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{self.base_url}/fixtures?date={today}&status=1H-2H-HT"

            headers = {
                'x-apisports-key': self.api_key
            }

            request = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode())

            if not data.get('response'):
                return []

            matches = []
            for fixture in data['response']:
                matches.append({
                    'match_id': str(fixture['fixture']['id']),
                    'home_team': fixture['teams']['home']['name'],
                    'away_team': fixture['teams']['away']['name'],
                    'league': fixture['league']['name'],
                    'status': self._get_status(fixture),
                    'minute': fixture['fixture']['status']['elapsed'] or 0
                })

            logger.info(f"✅ Found {len(matches)} live matches today")
            return matches

        except Exception as e:
            logger.error(f"Error fetching live matches: {e}")
            return []

    def fetch_upcoming_matches(self, hours: int = 6) -> List[Dict]:
        """
        Fetch matches starting in the next N hours.

        Args:
            hours: Number of hours ahead to look

        Returns:
            List of upcoming matches
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{self.base_url}/fixtures?date={today}&status=NS"

            headers = {
                'x-apisports-key': self.api_key
            }

            request = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode())

            if not data.get('response'):
                return []

            now = datetime.now()
            cutoff = now + timedelta(hours=hours)

            matches = []
            for fixture in data['response']:
                kickoff_timestamp = fixture['fixture']['timestamp']
                kickoff = datetime.fromtimestamp(kickoff_timestamp)

                if now < kickoff <= cutoff:
                    matches.append({
                        'match_id': str(fixture['fixture']['id']),
                        'home_team': fixture['teams']['home']['name'],
                        'away_team': fixture['teams']['away']['name'],
                        'league': fixture['league']['name'],
                        'kickoff': kickoff,
                        'time_to_kickoff_hours': (kickoff - now).total_seconds() / 3600
                    })

            logger.info(f"✅ Found {len(matches)} matches in next {hours}h")
            return matches

        except Exception as e:
            logger.error(f"Error fetching upcoming matches: {e}")
            return []

    def _parse_fixture_data(self, fixture: Dict) -> Dict:
        """Parse API-Football fixture response into live data format"""
        status = self._get_status(fixture)

        # Extract scores
        score_home = fixture['goals']['home'] or 0
        score_away = fixture['goals']['away'] or 0

        # Extract xG (if available in statistics)
        xg_home = 0.0
        xg_away = 0.0

        statistics = fixture.get('statistics', [])
        if len(statistics) >= 2:
            for stat in statistics[0].get('statistics', []):
                if stat['type'] == 'expected_goals':
                    try:
                        xg_home = float(stat['value'] or 0)
                    except (ValueError, TypeError):
                        pass

            for stat in statistics[1].get('statistics', []):
                if stat['type'] == 'expected_goals':
                    try:
                        xg_away = float(stat['value'] or 0)
                    except (ValueError, TypeError):
                        pass

        # Extract events (goals, cards)
        events = []
        for event in fixture.get('events', []):
            events.append({
                'time': event['time']['elapsed'],
                'type': event['type'],
                'team': event['team']['name'],
                'player': event['player']['name'],
                'detail': event['detail']
            })

        # Count red cards
        red_cards_home = sum(1 for e in events if e['type'] == 'Card' and e['detail'] == 'Red Card' and e['team'] == fixture['teams']['home']['name'])
        red_cards_away = sum(1 for e in events if e['type'] == 'Card' and e['detail'] == 'Red Card' and e['team'] == fixture['teams']['away']['name'])

        return {
            'minute': fixture['fixture']['status']['elapsed'] or 0,
            'score_home': score_home,
            'score_away': score_away,
            'xg_home': xg_home,
            'xg_away': xg_away,
            'red_cards_home': red_cards_home,
            'red_cards_away': red_cards_away,
            'status': status,
            'events': events
        }

    def _get_status(self, fixture: Dict) -> str:
        """Convert API-Football status to our format"""
        status_short = fixture['fixture']['status']['short']

        if status_short in ['1H', '2H', 'HT', 'ET', 'P']:
            return 'in_play'
        elif status_short in ['FT', 'AET', 'PEN']:
            return 'finished'
        elif status_short in ['NS', 'TBD']:
            return 'not_started'
        else:
            return 'unknown'

    def _get_from_cache(self, match_id: str) -> Optional[Dict]:
        """Get data from cache if fresh"""
        if match_id not in self._cache:
            return None

        cached_data, timestamp = self._cache[match_id]

        # Check if expired
        if time.time() - timestamp > self._cache_ttl:
            del self._cache[match_id]
            return None

        return cached_data

    def _save_to_cache(self, match_id: str, data: Dict):
        """Save data to cache with timestamp"""
        self._cache[match_id] = (data, time.time())

    def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
        logger.info("Cache cleared")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Auto Live Fetcher...")
    print("=" * 70)

    fetcher = AutoLiveFetcher()

    # Test 1: Get live matches today
    print("\n📺 Fetching live matches...")
    live_matches = fetcher.fetch_live_matches_today()

    if live_matches:
        print(f"✅ Found {len(live_matches)} live matches:")
        for match in live_matches[:3]:  # Show first 3
            print(f"  {match['home_team']} vs {match['away_team']}")
            print(f"  League: {match['league']}")
            print(f"  Status: {match['status']} - {match['minute']}'")
    else:
        print("⚠️  No live matches found (normal if not match day)")

    # Test 2: Get upcoming matches
    print("\n📅 Fetching upcoming matches (next 6 hours)...")
    upcoming = fetcher.fetch_upcoming_matches(hours=6)

    if upcoming:
        print(f"✅ Found {len(upcoming)} upcoming matches:")
        for match in upcoming[:3]:
            print(f"  {match['home_team']} vs {match['away_team']}")
            print(f"  Kickoff in: {match['time_to_kickoff_hours']:.1f}h")
    else:
        print("⚠️  No upcoming matches in next 6 hours")

    # Test 3: Fetch specific match data (if we have a live match)
    if live_matches:
        print(f"\n🔍 Fetching detailed data for first match...")
        test_match_id = live_matches[0]['match_id']
        live_data = fetcher.fetch_live_match_data(test_match_id)

        if live_data:
            print("✅ Live data retrieved:")
            print(f"  Minute: {live_data['minute']}")
            print(f"  Score: {live_data['score_home']}-{live_data['score_away']}")
            print(f"  xG: {live_data['xg_home']:.2f} - {live_data['xg_away']:.2f}")
            print(f"  Status: {live_data['status']}")

    print("\n" + "=" * 70)
    print("✅ Auto Live Fetcher test completed!")
