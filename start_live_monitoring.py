#!/usr/bin/env python3
"""
Live Betting Monitor Starter
=============================

Script per avviare il sistema di monitoring live con notifiche Telegram automatiche.

Usage:
    # Basic usage
    python start_live_monitoring.py

    # With custom interval
    python start_live_monitoring.py --interval 30

    # Dry run (no notifications)
    python start_live_monitoring.py --dry-run

Features:
- Auto-monitoring partite in corso
- Notifiche Telegram per opportunità di valore
- Alert per cambiamenti significativi
- Report giornaliero automatico

Setup:
1. Configura variabili d'ambiente:
   export TELEGRAM_BOT_TOKEN="your_bot_token"
   export TELEGRAM_CHAT_ID="your_chat_id"

2. Oppure modifica ai_system/config.py con i tuoi token

3. Run: python start_live_monitoring.py
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add ai_system to path
sys.path.insert(0, os.path.dirname(__file__))

from ai_system.telegram_notifier import TelegramNotifier
from ai_system.live_monitor import LiveMonitor
from ai_system.config import AIConfig
from ai_system.auto_live_fetcher import AutoLiveFetcher
from ai_system.auto_match_selector import AutoMatchSelector


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(level: str = "INFO"):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('live_monitoring.log')
        ]
    )

logger = logging.getLogger(__name__)


# ============================================================
# LIVE DATA FETCHER (REAL API INTEGRATION)
# ============================================================

# Global fetcher instance (reused across calls)
_live_fetcher = None

def get_live_fetcher():
    """Get or create live fetcher instance"""
    global _live_fetcher
    if _live_fetcher is None:
        _live_fetcher = AutoLiveFetcher()
    return _live_fetcher

def fetch_live_data_from_api(match_id: str) -> Optional[Dict]:
    """
    Fetch live match data from API-Football.

    Args:
        match_id: API-Football fixture ID

    Returns:
        Dict con dati live o None se non disponibili
    """
    try:
        fetcher = get_live_fetcher()
        live_data = fetcher.fetch_live_match_data(match_id)
        return live_data
    except Exception as e:
        logger.error(f"Error fetching live data for {match_id}: {e}")
        return None


# ============================================================
# MATCHES TO MONITOR (AUTOMATIC SELECTION)
# ============================================================

def get_matches_to_monitor(auto_select: bool = True, time_window_hours: int = 6) -> List[Dict]:
    """
    Ottieni lista di partite da monitorare.

    Strategy:
    1. Fetch live matches currently in progress
    2. Fetch upcoming matches in next N hours
    3. Auto-select based on priority scoring

    Args:
        auto_select: Use automatic selection (default: True)
        time_window_hours: Hours ahead to look (default: 6)

    Returns:
        Lista di dict con match data
    """
    if not auto_select:
        # Manual mode: return empty (user must add matches manually)
        logger.info("Manual mode: no automatic match selection")
        return []

    try:
        fetcher = get_live_fetcher()
        selector = AutoMatchSelector(min_ev=3.0, max_matches=10)

        matches = []

        # 1. Get live matches (highest priority)
        logger.info("🔍 Searching for live matches...")
        live_matches = selector.get_live_matches_to_monitor(live_fetcher=fetcher)

        for match in live_matches:
            matches.append({
                'match_id': match['match_id'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'league': match['league'],
                'pre_match_prob': 0.50,  # Default 50/50 for live (will be updated)
                'odds': 2.0,  # Default odds (will be updated from live data)
                'start_time': datetime.now(),  # Already started
                'is_live': True
            })

        # 2. Get upcoming matches
        logger.info(f"🔍 Searching for upcoming matches (next {time_window_hours}h)...")
        upcoming_matches = selector.get_matches_to_monitor(
            live_fetcher=fetcher,
            time_window_hours=time_window_hours
        )

        for match in upcoming_matches:
            # Avoid duplicates
            if match['match_id'] not in [m['match_id'] for m in matches]:
                matches.append({
                    'match_id': match['match_id'],
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'league': match['league'],
                    'pre_match_prob': 0.50,  # Default (would run Dixon-Coles for real prediction)
                    'odds': 2.0,  # Would fetch real odds from API
                    'start_time': match.get('kickoff', datetime.now()),
                    'is_live': False
                })

        # Save selection history
        if matches:
            selector.save_selection_history(matches)

        logger.info(f"✅ Total matches to monitor: {len(matches)}")
        return matches

    except Exception as e:
        logger.error(f"❌ Error selecting matches: {e}")
        logger.warning("Falling back to empty list")
        return []


# ============================================================
# MAIN MONITORING SYSTEM
# ============================================================

class LiveBettingSystem:
    """Sistema completo di live betting con notifiche"""

    def __init__(self, config: AIConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run

        # Initialize Telegram notifier
        if config.telegram_enabled and not dry_run:
            if not config.telegram_bot_token or not config.telegram_chat_id:
                logger.error("❌ Telegram credentials not configured!")
                logger.error("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
                sys.exit(1)

            self.notifier = TelegramNotifier(
                bot_token=config.telegram_bot_token,
                chat_id=config.telegram_chat_id,
                min_ev=config.telegram_min_ev,
                min_confidence=config.telegram_min_confidence,
                rate_limit_seconds=config.telegram_rate_limit_seconds
            )
            logger.info("✅ Telegram notifier initialized")
        else:
            logger.warning("⚠️  Running in DRY-RUN mode (no notifications)")
            self.notifier = None

        # Initialize live monitor
        self.monitor = LiveMonitor(
            telegram_notifier=self.notifier,
            fetch_live_data=fetch_live_data_from_api,
            update_interval=config.live_update_interval,
            min_ev_alert=config.live_min_ev_alert
        )

        logger.info("✅ Live monitoring system initialized")

    def load_matches(self):
        """Carica partite da monitorare"""
        matches = get_matches_to_monitor()

        if not matches:
            logger.warning("⚠️  No matches to monitor")
            return

        logger.info(f"📋 Loading {len(matches)} matches...")

        for match in matches:
            self.monitor.add_match(
                match_id=match['match_id'],
                home_team=match['home_team'],
                away_team=match['away_team'],
                league=match['league'],
                pre_match_prob=match['pre_match_prob'],
                odds=match['odds'],
                start_time=match.get('start_time')
            )

        logger.info(f"✅ Loaded {len(matches)} matches for monitoring")

    def start(self):
        """Avvia monitoring"""
        logger.info("🚀 Starting live monitoring system...")

        # Send startup notification
        if self.notifier:
            startup_msg = """
🤖 <b>LIVE MONITORING STARTED</b>

System is now monitoring live matches and will send notifications for:
• Value opportunities (EV > 5%)
• Significant probability shifts
• Optimal betting timing
• Important match events

Status: <b>ACTIVE ✅</b>
"""
            self.notifier._send_message(startup_msg.strip())

        # Start monitor
        self.monitor.start()
        logger.info("✅ Monitor running")

    def stop(self):
        """Ferma monitoring"""
        logger.info("🛑 Stopping live monitoring...")
        self.monitor.stop()

        # Send shutdown notification
        if self.notifier:
            shutdown_msg = """
🤖 <b>LIVE MONITORING STOPPED</b>

Status: <b>INACTIVE ⏸️</b>
"""
            self.notifier._send_message(shutdown_msg.strip())

        logger.info("✅ Monitor stopped")

    def print_status(self):
        """Stampa stato corrente"""
        status = self.monitor.get_status()

        print("\n" + "=" * 70)
        print(f"📊 LIVE MONITORING STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 70)
        print(f"Running: {'✅ YES' if status['running'] else '❌ NO'}")
        print(f"Matches monitored: {status['matches_monitored']}")

        if status['matches']:
            print("\n📋 Active matches:")
            for match in status['matches']:
                print(f"\n  {match['home']} vs {match['away']}")
                print(f"  Minute: {match['minute']} | Score: {match['score']}")
                print(f"  Win probability: {match['probability']:.1%}")
        else:
            print("\n⚠️  No active matches")

        print("=" * 70)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Live Betting Monitor with Telegram Notifications"
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Update interval in seconds (default: 60)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without sending Telegram notifications'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load config
    config = AIConfig()
    config.live_update_interval = args.interval

    # Create system
    system = LiveBettingSystem(config, dry_run=args.dry_run)

    # Load matches
    system.load_matches()

    # Start monitoring
    system.start()

    # Main loop
    try:
        logger.info("✅ System running. Press Ctrl+C to stop.")
        print("\n" + "=" * 70)
        print("🚀 LIVE MONITORING ACTIVE")
        print("=" * 70)
        print(f"Update interval: {args.interval}s")
        print(f"Dry run: {'YES (no notifications)' if args.dry_run else 'NO (notifications enabled)'}")
        print("\nPress Ctrl+C to stop...")
        print("=" * 70)

        # Status updates every 5 minutes
        status_interval = 300  # 5 minutes
        last_status = time.time()

        while True:
            time.sleep(10)

            # Print status update
            if time.time() - last_status >= status_interval:
                system.print_status()
                last_status = time.time()

    except KeyboardInterrupt:
        logger.info("\n👋 Shutdown requested by user")

    finally:
        system.stop()
        logger.info("✅ System shutdown complete")


if __name__ == "__main__":
    main()
