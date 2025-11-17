"""
Live Betting Monitor
====================

Worker che monitora partite live e invia notifiche automatiche.

Features:
- Monitoring continuo partite in corso
- Aggiornamento probabilità ogni minuto
- Rilevamento opportunità di valore
- Notifiche Telegram automatiche
- Gestione stato per evitare duplicati

Usage:
    monitor = LiveMonitor(telegram_notifier, api_client)
    monitor.add_match(match_id, pre_match_prob)
    monitor.start()  # Runs forever
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict

from .live_betting import LiveBettingEngine, LiveMatch
from .telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class LiveMonitor:
    """
    Monitor per partite live con notifiche automatiche.

    Args:
        telegram_notifier: Istanza TelegramNotifier per invio messaggi
        fetch_live_data: Function(match_id) -> dict con dati live
        update_interval: Secondi tra aggiornamenti (default: 60)
        min_ev_alert: EV minimo per alert valore (default: 8%)
    """

    def __init__(
        self,
        telegram_notifier: TelegramNotifier,
        fetch_live_data: Optional[Callable] = None,
        update_interval: int = 60,
        min_ev_alert: float = 8.0
    ):
        self.notifier = telegram_notifier
        self.fetch_live_data = fetch_live_data or self._mock_fetch_live_data
        self.update_interval = update_interval
        self.min_ev_alert = min_ev_alert

        # Live betting engine
        self.engine = LiveBettingEngine()

        # Tracked matches: {match_id: LiveMatch}
        self.monitored_matches: Dict[str, LiveMatch] = {}

        # Match metadata: {match_id: {home, away, league, odds, ...}}
        self.match_metadata: Dict[str, Dict] = {}

        # Previous probabilities (for change detection)
        self.previous_probs: Dict[str, float] = {}

        # Notification history (avoid spam)
        self.notified_events: Dict[str, List[str]] = defaultdict(list)

        # Control
        self.running = False
        self.monitor_thread = None

        logger.info("✅ Live Monitor initialized")

    def add_match(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        league: str,
        pre_match_prob: float,
        odds: float,
        start_time: Optional[datetime] = None
    ):
        """
        Aggiungi partita da monitorare.

        Args:
            match_id: ID univoco partita
            home_team: Nome squadra casa
            away_team: Nome squadra trasferta
            league: Nome lega
            pre_match_prob: Probabilità pre-match
            odds: Quote correnti
            start_time: Orario inizio (opzionale)
        """
        # Create LiveMatch
        live_match = self.engine.start_monitoring(match_id, pre_match_prob)

        # Store
        self.monitored_matches[match_id] = live_match
        self.match_metadata[match_id] = {
            'home': home_team,
            'away': away_team,
            'league': league,
            'odds': odds,
            'start_time': start_time or datetime.now(),
            'pre_match_prob': pre_match_prob
        }
        self.previous_probs[match_id] = pre_match_prob

        logger.info(f"📌 Monitoring: {home_team} vs {away_team} (ID: {match_id})")

    def remove_match(self, match_id: str):
        """Rimuovi partita dal monitoring"""
        if match_id in self.monitored_matches:
            meta = self.match_metadata.get(match_id, {})
            logger.info(f"🏁 Stopped monitoring: {meta.get('home', '?')} vs {meta.get('away', '?')}")

            del self.monitored_matches[match_id]
            del self.match_metadata[match_id]
            if match_id in self.previous_probs:
                del self.previous_probs[match_id]
            if match_id in self.notified_events:
                del self.notified_events[match_id]

    def start(self):
        """Avvia monitoring in background thread"""
        if self.running:
            logger.warning("Monitor already running")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("🚀 Live Monitor started")

    def stop(self):
        """Ferma monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("🛑 Live Monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._update_all_matches()
            except Exception as e:
                logger.error(f"❌ Monitor error: {e}", exc_info=True)

            # Sleep
            time.sleep(self.update_interval)

    def _update_all_matches(self):
        """Update tutte le partite monitorate"""
        if not self.monitored_matches:
            logger.debug("No matches to monitor")
            return

        logger.debug(f"🔄 Updating {len(self.monitored_matches)} matches...")

        for match_id in list(self.monitored_matches.keys()):
            try:
                self._update_match(match_id)
            except Exception as e:
                logger.error(f"Error updating match {match_id}: {e}")

    def _update_match(self, match_id: str):
        """Update singola partita"""
        live_match = self.monitored_matches.get(match_id)
        if not live_match:
            return

        # Fetch dati live
        live_data = self.fetch_live_data(match_id)

        if not live_data:
            logger.debug(f"No live data for match {match_id}")
            return

        # Check se partita finita
        minute = live_data.get('minute', 0)
        if minute >= 90 and live_data.get('status') == 'finished':
            logger.info(f"🏁 Match finished: {match_id}")
            self.remove_match(match_id)
            return

        # Update LiveMatch
        live_match.update(live_data)

        # Recalculate probability
        result = self.engine.recalculate_probability(live_match)

        new_prob = result['probability_home_win']
        old_prob = self.previous_probs.get(match_id, new_prob)

        # Detect events and send notifications
        self._check_and_notify(match_id, live_match, result, old_prob, new_prob)

        # Update previous
        self.previous_probs[match_id] = new_prob

    def _check_and_notify(
        self,
        match_id: str,
        live_match: LiveMatch,
        result: Dict,
        old_prob: float,
        new_prob: float
    ):
        """
        Check condizioni e invia notifiche se necessario.
        """
        metadata = self.match_metadata.get(match_id, {})
        odds = metadata.get('odds', 2.0)

        # Calculate EV
        ev = (new_prob * odds - 1) * 100

        # 1. VALUE SHIFT ALERT
        prob_change = abs(new_prob - old_prob)
        if prob_change >= 0.10 and ev > self.min_ev_alert:  # 10% shift
            event_key = f"value_shift_{live_match.minute}"
            if event_key not in self.notified_events[match_id]:
                logger.info(f"📈 Value shift detected in {match_id}: {old_prob:.1%} -> {new_prob:.1%}")

                self.notifier.send_live_alert(
                    match_data=self._build_match_data(match_id),
                    live_result=result,
                    alert_type="VALUE_SHIFT"
                )

                self.notified_events[match_id].append(event_key)

        # 2. BET NOW ALERT (optimal timing)
        timing = result.get('timing', 'WATCH')
        if timing == "NOW" and ev > self.min_ev_alert:
            event_key = f"bet_now_{live_match.minute}"
            if event_key not in self.notified_events[match_id]:
                logger.info(f"🚨 BET NOW alert for {match_id}")

                self.notifier.send_live_alert(
                    match_data=self._build_match_data(match_id),
                    live_result=result,
                    alert_type="BET_NOW"
                )

                self.notified_events[match_id].append(event_key)

        # 3. GOAL ALERT (quando cambia score)
        current_goals = live_match.score_home + live_match.score_away
        event_key_goal = f"goal_{current_goals}"
        if event_key_goal not in self.notified_events[match_id] and current_goals > 0:
            # Verifica se è un nuovo goal (controlla ultimo evento)
            if len(self.notified_events[match_id]) == 0 or not any('goal' in e for e in self.notified_events[match_id][-3:]):
                logger.info(f"⚽ Goal scored in {match_id}")

                self.notifier.send_live_alert(
                    match_data=self._build_match_data(match_id),
                    live_result=result,
                    alert_type="GOAL"
                )

                self.notified_events[match_id].append(event_key_goal)

        # 4. RED CARD ALERT
        total_red_cards = live_match.red_cards_home + live_match.red_cards_away
        event_key_red = f"red_card_{total_red_cards}"
        if total_red_cards > 0 and event_key_red not in self.notified_events[match_id]:
            logger.info(f"🟥 Red card in {match_id}")

            self.notifier.send_live_alert(
                match_data=self._build_match_data(match_id),
                live_result=result,
                alert_type="RED_CARD"
            )

            self.notified_events[match_id].append(event_key_red)

    def _build_match_data(self, match_id: str) -> Dict:
        """Build match data dict per notifiche"""
        metadata = self.match_metadata.get(match_id, {})
        live_match = self.monitored_matches.get(match_id)

        return {
            'match_id': match_id,
            'home': metadata.get('home', 'Home'),
            'away': metadata.get('away', 'Away'),
            'league': metadata.get('league', 'Unknown'),
            'odds': metadata.get('odds', 2.0),
            'minute': live_match.minute if live_match else 0,
            'score_home': live_match.score_home if live_match else 0,
            'score_away': live_match.score_away if live_match else 0
        }

    def _mock_fetch_live_data(self, match_id: str) -> Optional[Dict]:
        """
        Mock fetch function per testing.
        In produzione, sostituire con chiamata API reale.
        """
        # Simula progresso partita
        metadata = self.match_metadata.get(match_id, {})
        live_match = self.monitored_matches.get(match_id)

        if not live_match:
            return None

        # Incrementa minuto
        elapsed = (datetime.now() - metadata.get('start_time', datetime.now())).total_seconds()
        minute = min(int(elapsed / 60), 90)

        # Mock data
        return {
            'minute': minute,
            'score_home': 0 if minute < 30 else (1 if minute < 60 else 2),
            'score_away': 0 if minute < 45 else 1,
            'xg_home': minute * 0.03,
            'xg_away': minute * 0.02,
            'status': 'in_play' if minute < 90 else 'finished'
        }

    def get_status(self) -> Dict:
        """Ottieni stato corrente del monitor"""
        return {
            'running': self.running,
            'matches_monitored': len(self.monitored_matches),
            'matches': [
                {
                    'match_id': mid,
                    'home': meta.get('home'),
                    'away': meta.get('away'),
                    'minute': self.monitored_matches[mid].minute,
                    'score': f"{self.monitored_matches[mid].score_home}-{self.monitored_matches[mid].score_away}",
                    'probability': self.previous_probs.get(mid, 0)
                }
                for mid, meta in self.match_metadata.items()
            ]
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing Live Monitor...")
    print("=" * 70)

    # Initialize Telegram notifier (mock mode)
    from .telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier(
        bot_token="MOCK_TOKEN",
        chat_id="MOCK_CHAT",
        min_ev=5.0
    )

    # Initialize monitor
    monitor = LiveMonitor(
        telegram_notifier=notifier,
        update_interval=10,  # 10 seconds for testing
        min_ev_alert=5.0
    )

    # Add test match
    monitor.add_match(
        match_id="test_123",
        home_team="Manchester City",
        away_team="Arsenal",
        league="Premier League",
        pre_match_prob=0.65,
        odds=1.90,
        start_time=datetime.now()
    )

    # Start monitoring
    monitor.start()

    print("\n🔄 Monitoring started (will run for 60 seconds)...")
    print("Status updates:")

    # Run for 60 seconds
    try:
        for i in range(6):
            time.sleep(10)
            status = monitor.get_status()
            print(f"\nUpdate {i+1}/6:")
            for match in status['matches']:
                print(f"  {match['home']} vs {match['away']}")
                print(f"  Minute: {match['minute']} | Score: {match['score']}")
                print(f"  Probability: {match['probability']:.1%}")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Stop
    monitor.stop()

    print("\n" + "=" * 70)
    print("✅ Live Monitor test completed!")
