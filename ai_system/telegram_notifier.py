"""
Telegram Notifier
==================

Sistema di notifiche Telegram per opportunità betting live e pre-match.

Features:
- Formattazione messaggi con HTML
- Notifiche per opportunità di valore
- Alert live betting
- Report giornalieri
- Gestione rate limiting

Usage:
    notifier = TelegramNotifier(bot_token, chat_id)
    notifier.send_betting_opportunity(match_data, analysis_result)
"""

import logging
import requests
import time
import html
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Gestore notifiche Telegram per sistema betting.

    Args:
        bot_token: Token del bot Telegram
        chat_id: ID della chat destinataria
        min_ev: Expected Value minimo per inviare notifica (default: 5%)
        min_confidence: Confidence minima per inviare notifica (default: 60%)
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        min_ev: float = 5.0,
        min_confidence: float = 60.0,
        rate_limit_seconds: int = 3,
        live_alerts_enabled: bool = True
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.min_ev = min_ev
        self.min_confidence = min_confidence
        self.rate_limit_seconds = rate_limit_seconds
        self.live_alerts_enabled = live_alerts_enabled

        self.last_message_time = 0
        self.messages_sent = 0

        logger.info("✅ Telegram Notifier initialized")

    def send_betting_opportunity(
        self,
        match_data: Dict,
        analysis_result: Dict,
        opportunity_type: str = "PRE-MATCH"
    ) -> bool:
        """
        Invia notifica per opportunità betting.

        Args:
            match_data: Dati della partita
            analysis_result: Risultato analisi AI
            opportunity_type: Tipo ("PRE-MATCH", "LIVE", "ALERT")

        Returns:
            True se inviato con successo
        """
        # Check se vale la pena notificare
        if not self._should_notify(analysis_result):
            logger.debug("Opportunity below threshold, skipping notification")
            return False

        # Rate limiting
        self._apply_rate_limit()

        # Formatta messaggio
        message = self._format_betting_message(
            match_data,
            analysis_result,
            opportunity_type
        )

        # Invia
        success = self._send_message(message, parse_mode="HTML")

        if success:
            self.messages_sent += 1
            logger.info(f"📨 Sent Telegram notification #{self.messages_sent}")

        return success

    def send_live_alert(
        self,
        match_data: Dict,
        live_result: Dict,
        alert_type: str = "VALUE_SHIFT"
    ) -> bool:
        """
        Invia alert per evento live importante.

        Args:
            match_data: Dati partita
            live_result: Risultato live betting engine
            alert_type: Tipo alert ("VALUE_SHIFT", "GOAL", "RED_CARD", "BET_NOW")

        Returns:
            True se inviato
        """
        if not self.live_alerts_enabled:
            logger.debug("Live alerts disabilitati, skip invio per %s", match_data.get('match_id', 'unknown_match'))
            return False

        self._apply_rate_limit()

        message = self._format_live_alert(match_data, live_result, alert_type)
        return self._send_message(message, parse_mode="HTML")

    def send_daily_report(
        self,
        opportunities_found: int,
        bets_placed: int,
        total_stake: float,
        expected_profit: float,
        opportunities: List[Dict]
    ) -> bool:
        """
        Invia report giornaliero.

        Args:
            opportunities_found: Numero opportunità trovate
            bets_placed: Numero bet piazzate
            total_stake: Stake totale
            expected_profit: Profitto atteso
            opportunities: Lista opportunità dettagliate

        Returns:
            True se inviato
        """
        message = self._format_daily_report(
            opportunities_found,
            bets_placed,
            total_stake,
            expected_profit,
            opportunities
        )

        return self._send_message(message, parse_mode="HTML")

    def _should_notify(self, analysis_result: Dict) -> bool:
        """Check se opportunità merita notifica"""
        final_action = analysis_result.get('action')
        if not final_action and "final_decision" in analysis_result:
            final_action = analysis_result["final_decision"].get("action")
        if final_action != 'BET':
            return False

        ev = analysis_result.get('ev')
        if ev is None:
            ev = analysis_result.get('summary', {}).get('expected_value')
            if isinstance(ev, (int, float)):
                ev = ev * 100  # convert to %
        if ev is None:
            ev = 0
        if ev < self.min_ev:
            return False

        confidence = analysis_result.get('confidence_level')
        if confidence is None:
            confidence = analysis_result.get('summary', {}).get('confidence', 0)
        if confidence < self.min_confidence:
            return False

        return True

    def _format_betting_message(
        self,
        match_data: Dict,
        analysis_result: Dict,
        opportunity_type: str
    ) -> str:
        """Formatta messaggio HTML per opportunità betting (in ITALIANO)"""

        # Emoji per tipo
        emoji_map = {
            "PRE-MATCH": "⚽",
            "LIVE": "🔴",
            "ALERT": "🚨"
        }
        emoji = emoji_map.get(opportunity_type, "⚽")

        # Estrai dati
        home = match_data.get('home', 'Squadra Casa')
        away = match_data.get('away', 'Squadra Trasferta')
        league = match_data.get('league', 'Campionato Sconosciuto')

        final_decision = analysis_result.get('final_decision', {})
        summary = analysis_result.get('summary', {})

        action = analysis_result.get('action') or final_decision.get('action', 'BET')
        market_raw = analysis_result.get('market', final_decision.get('market', '1X2'))
        stake = analysis_result.get('stake_amount', final_decision.get('stake', 0))

        ev = analysis_result.get('ev')
        if ev is None:
            ev = summary.get('expected_value')
            if isinstance(ev, (int, float)):
                ev = ev * 100
        if ev is None:
            ev = 0.0

        probability = analysis_result.get('probability')
        if probability is None:
            probability = summary.get('probability')
        if probability is not None and probability <= 1:
            probability *= 100
        probability = probability or 0.0

        odds = analysis_result.get('odds') or summary.get('odds', 0)

        confidence = analysis_result.get('confidence_level')
        if confidence is None:
            confidence = summary.get('confidence', 0)

        # Traduci market in italiano
        market_translations = {
            '1X2_HOME': '1 (Vittoria Casa)',
            '1X2_DRAW': 'X (Pareggio)',
            '1X2_AWAY': '2 (Vittoria Trasferta)',
            '1X2': '1X2',
            'OVER_2.5': 'Over 2.5 Gol',
            'UNDER_2.5': 'Under 2.5 Gol',
            'OVER_1.5': 'Over 1.5 Gol',
            'UNDER_1.5': 'Under 1.5 Gol',
            'BTTS_YES': 'Goal (entrambe segnano)',
            'BTTS_NO': 'No Goal',
        }
        market = market_translations.get(market_raw, market_raw)

        # Formatta confidence in italiano
        if confidence >= 80:
            confidence_emoji = "🟢"
            confidence_text = "MOLTO ALTA"
        elif confidence >= 70:
            confidence_emoji = "🟡"
            confidence_text = "ALTA"
        elif confidence >= 60:
            confidence_emoji = "🟠"
            confidence_text = "MEDIA"
        else:
            confidence_emoji = "🔴"
            confidence_text = "BASSA"

        # Determina se è live o pre-match
        is_live = match_data.get('is_live', False)
        match_status = match_data.get('match_status', '')
        match_date = match_data.get('match_date')
        
        # Formatta data/ora partita
        time_info = ""
        if match_date:
            try:
                if isinstance(match_date, str):
                    from datetime import datetime
                    match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                if isinstance(match_date, datetime):
                    time_info = match_date.strftime("🕐 %H:%M")
            except:
                pass
        
        # Build message in ITALIANO
        status_emoji = "🔴 LIVE" if is_live else "⚽ PRE-PARTITA"
        status_text = f"{status_emoji} {time_info}" if time_info else status_emoji
        
        message = f"""
{emoji} <b>OPPORTUNITÀ DI SCOMMESSA</b> {emoji}

<b>📅 Partita</b>
{home} vs {away}
🏆 {league}
{status_text}

<b>💰 Raccomandazione</b>
Mercato: <b>{market}</b>
Puntata: <b>€{stake:.2f}</b>
Quota: <b>{odds:.2f}</b>

<b>📊 Analisi</b>
Valore Atteso (EV): <b>{ev:+.1f}%</b>
Probabilità Vittoria: <b>{probability:.1f}%</b>
Confidenza: {confidence_emoji} <b>{confidence_text} ({confidence:.0f}%)</b>
"""

        # 📈 NUOVO: Aggiungi statistiche dettagliate per verifica
        message += "\n<b>📈 Statistiche Estratte</b>\n"
        
        # Mostra quote di tutti i mercati
        odds_1 = match_data.get('odds_1', 0)
        odds_x = match_data.get('odds_x', 0)
        odds_2 = match_data.get('odds_2', 0)
        if odds_1 and odds_x and odds_2:
            message += f"Quote: 1={odds_1:.2f} X={odds_x:.2f} 2={odds_2:.2f}\n"
            
            # Calcola probabilità implicite
            impl_1 = (1.0 / odds_1 * 100) if odds_1 > 1 else 0
            impl_x = (1.0 / odds_x * 100) if odds_x > 1 else 0
            impl_2 = (1.0 / odds_2 * 100) if odds_2 > 1 else 0
            margin = impl_1 + impl_x + impl_2 - 100
            message += f"Prob. Implicite: 1={impl_1:.1f}% X={impl_x:.1f}% 2={impl_2:.1f}%\n"
            message += f"Margine Bookmaker: {margin:.1f}%\n"
        
        # Mostra dettagli calcolo EV
        implied_prob = (1.0 / odds * 100) if odds > 1.0 else 0
        value_edge = probability - implied_prob
        message += f"Vantaggio: {value_edge:+.1f}% (Nostra prob. vs Bookmaker)\n"

        # Aggiungi predizioni modelli se disponibili
        if 'ensemble' in analysis_result:
            ensemble = analysis_result['ensemble']
            predictions = ensemble.get('model_predictions', {})
            weights = ensemble.get('model_weights', {})

            message += "\n<b>🤖 Ensemble AI</b>\n"
            message += f"Dixon-Coles: {predictions.get('dixon_coles', 0)*100:.1f}% (peso: {weights.get('dixon_coles', 0)*100:.0f}%)\n"
            message += f"XGBoost: {predictions.get('xgboost', 0)*100:.1f}% (peso: {weights.get('xgboost', 0)*100:.0f}%)\n"
            message += f"LSTM: {predictions.get('lstm', 0)*100:.1f}% (peso: {weights.get('lstm', 0)*100:.0f}%)\n"

            uncertainty = ensemble.get('uncertainty', 0)
            message += f"Incertezza: {uncertainty*100:.1f}%"

        if 'bayesian_fusion' in analysis_result:
            fusion = analysis_result['bayesian_fusion']
            message += (
                "\n\n<b>🧠 Fusione Bayesiana</b>\n"
                f"Prob. Finale: {fusion.get('probability', 0)*100:.1f}%\n"
                f"Intervallo 95%: {fusion.get('ci_low', 0)*100:.1f}% – {fusion.get('ci_high', 0)*100:.1f}%\n"
                f"Affidabilità: <b>{fusion.get('confidence', 0):.0f}%</b>"
            )

        regime = final_decision.get("market_regime") or summary.get("market_regime")
        if regime:
            regime_it = regime.replace("STABLE", "STABILE").replace("VOLATILE", "VOLATILE").replace("SHARP", "MOVIMENTO ESPERTO")
            message += f"\n\n<b>Regime Mercato:</b> {regime_it}"

        llm_playbook = analysis_result.get("llm_playbook") or summary.get("llm_playbook")
        if isinstance(llm_playbook, dict) and llm_playbook.get("text"):
            message += (
                "\n\n🧠 <b>Analisi AI</b>\n"
                f"{html.escape(llm_playbook['text'])}"
            )

        # Timing
        now = datetime.now().strftime("%H:%M:%S")
        message += f"\n\n⏰ {now}"

        return message.strip()

    def _format_live_alert(
        self,
        match_data: Dict,
        live_result: Dict,
        alert_type: str
    ) -> str:
        """Formatta alert live"""

        home = match_data.get('home', 'Home')
        away = match_data.get('away', 'Away')

        minute = live_result.get('minute', 0)
        score = live_result.get('score', '0-0')
        prob = live_result.get('probability_home_win', 0) * 100
        timing = live_result.get('timing', 'WATCH')

        # Alert specifici
        if alert_type == "BET_NOW":
            emoji = "🚨"
            title = "BET NOW - OPTIMAL TIMING"
        elif alert_type == "VALUE_SHIFT":
            emoji = "📈"
            title = "VALUE OPPORTUNITY DETECTED"
        elif alert_type == "GOAL":
            emoji = "⚽"
            title = "GOAL SCORED"
        elif alert_type == "RED_CARD":
            emoji = "🟥"
            title = "RED CARD"
        else:
            emoji = "🔴"
            title = "LIVE UPDATE"

        message = f"""
{emoji} <b>{title}</b>

<b>🔴 LIVE</b>
{home} vs {away}
{minute}' - Score: <b>{score}</b>

<b>📊 Updated Probability</b>
Home Win: <b>{prob:.1f}%</b>

<b>⏱️ Timing</b>
Recommendation: <b>{timing}</b>
"""

        # Aggiungi adjustments se disponibili
        if 'adjustments' in live_result:
            adj = live_result['adjustments']
            message += f"\n<b>🔄 Live Adjustments</b>\n"
            message += f"Score impact: {adj.get('score', 1.0):.2f}x\n"
            message += f"xG impact: {adj.get('xg', 1.0):.2f}x\n"
            message += f"Momentum: {adj.get('momentum', 1.0):.2f}x\n"

        return message.strip()

    def _format_daily_report(
        self,
        opportunities_found: int,
        bets_placed: int,
        total_stake: float,
        expected_profit: float,
        opportunities: List[Dict]
    ) -> str:
        """Formatta report giornaliero"""

        today = datetime.now().strftime("%d/%m/%Y")

        message = f"""
📊 <b>DAILY REPORT - {today}</b>

<b>📈 Summary</b>
Opportunities Found: <b>{opportunities_found}</b>
Bets Placed: <b>{bets_placed}</b>
Total Stake: <b>€{total_stake:.2f}</b>
Expected Profit: <b>€{expected_profit:.2f}</b>

<b>🎯 Top Opportunities</b>
"""

        # Top 5 opportunità
        top_opps = sorted(
            opportunities,
            key=lambda x: x.get('ev', 0),
            reverse=True
        )[:5]

        for i, opp in enumerate(top_opps, 1):
            match = f"{opp.get('home', '?')} vs {opp.get('away', '?')}"
            ev = opp.get('ev', 0)
            stake = opp.get('stake', 0)
            message += f"\n{i}. {match}\n   EV: +{ev:.1f}% | Stake: €{stake:.2f}"

        return message.strip()

    def _send_message(
        self,
        message: str,
        parse_mode: str = "HTML"
    ) -> bool:
        """
        Invia messaggio via Telegram API.

        Args:
            message: Testo messaggio
            parse_mode: "HTML" o "Markdown"

        Returns:
            True se inviato con successo
        """
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                return True
            elif response.status_code == 401:
                logger.error("❌ Invalid Telegram bot token")
                return False
            elif response.status_code == 400:
                logger.error("❌ Invalid chat ID or message format")
                return False
            elif response.status_code == 429:
                logger.warning("⚠️  Rate limit reached, waiting...")
                time.sleep(30)
                return False
            else:
                logger.error(f"❌ Telegram API error: {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            logger.error("❌ Telegram API timeout")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return False

    def _apply_rate_limit(self):
        """Applica rate limiting tra messaggi"""
        elapsed = time.time() - self.last_message_time

        if elapsed < self.rate_limit_seconds:
            sleep_time = self.rate_limit_seconds - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

        self.last_message_time = time.time()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Telegram Notifier...")
    print("=" * 70)

    # Initialize (use mock credentials for testing)
    notifier = TelegramNotifier(
        bot_token="8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g",
        chat_id="-1003278011521",
        min_ev=5.0,
        min_confidence=60.0
    )

    # Test betting opportunity
    match_data = {
        'home': 'Manchester City',
        'away': 'Arsenal',
        'league': 'Premier League'
    }

    analysis_result = {
        'action': 'BET',
        'market': '1X2_HOME',
        'stake_amount': 133.68,
        'ev': 25.4,
        'probability': 0.66,
        'odds': 1.90,
        'confidence_level': 82,
        'ensemble': {
            'model_predictions': {
                'dixon_coles': 0.65,
                'xgboost': 0.71,
                'lstm': 0.68
            },
            'model_weights': {
                'dixon_coles': 0.30,
                'xgboost': 0.40,
                'lstm': 0.30
            },
            'uncertainty': 0.025
        }
    }

    print("\n📨 Sending test notification...")
    success = notifier.send_betting_opportunity(
        match_data,
        analysis_result,
        opportunity_type="PRE-MATCH"
    )

    if success:
        print("✅ Notification sent successfully!")
    else:
        print("❌ Failed to send notification (check credentials)")

    print("\n" + "=" * 70)
    print("✅ Telegram Notifier test completed!")
