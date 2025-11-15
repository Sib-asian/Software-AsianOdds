import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .config import AIConfig
from .blocco_0_api_engine import APIDataEngine
from .blocco_5_risk_manager import RiskManager
from .backtesting import Backtester
from .live_monitor import LiveMonitor
from .telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class BackgroundAutomationService:
    """
    Coordina le funzionalità avanzate (risk, live monitor, budget API, backtesting)
    senza esporre controlli nell'interfaccia Streamlit.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        self._lock = threading.Lock()
        self._config = config or AIConfig()
        self._config_signature = self._config_snapshot(self._config)

        self._api_engine = APIDataEngine(self._config)
        self._risk_shadow = RiskManager(self._config)
        self._notifier = self._build_notifier()

        self._live_monitor: Optional[LiveMonitor] = None
        self._live_monitor_started = False

        self._portfolio_log: List[Dict[str, Any]] = []
        self._usage_day = datetime.utcnow().date()
        self._api_usage_today = 0

        self._backtester: Optional[Backtester] = None
        self._last_backtest_run: Optional[datetime] = None
        self._last_backtest_summary: Optional[Dict[str, Any]] = None

        self._setup_live_monitor()

    def handle_analysis(
        self,
        match: Dict[str, Any],
        ai_result: Dict[str, Any],
        bankroll: float,
        ai_config: Optional[AIConfig] = None,
        time_to_kickoff_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processa i risultati della pipeline e attiva i servizi nascosti.
        """
        if ai_config is not None:
            self._apply_config(ai_config)

        summary = {
            "live_monitor_registered": False,
            "backtest_triggered": False,
            "shadow_decision": None,
            "api_usage_today": None
        }

        api_context = ai_result.get("api_context")
        self._update_api_usage(api_context)

        shadow_decision = self._shadow_risk_check(match, ai_result, bankroll)
        if shadow_decision:
            summary["shadow_decision"] = shadow_decision.get("decision")

        final_decision = ai_result.get("final_decision", {})
        if final_decision.get("action") == "BET":
            summary["live_monitor_registered"] = self._register_live_monitor(match, ai_result, time_to_kickoff_hours)
            self._log_portfolio_entry(match, final_decision, bankroll)

        summary["backtest_triggered"] = self._maybe_schedule_backtest(ai_result)
        summary["api_usage_today"] = self._api_usage_today

        if summary["live_monitor_registered"]:
            logger.info("📡 Live monitor attivo in background per %s vs %s", match.get("home"), match.get("away"))
        if summary["backtest_triggered"]:
            logger.info("📊 Backtest programmato in background")

        return summary

    def _apply_config(self, config: AIConfig):
        snapshot = self._config_snapshot(config)
        with self._lock:
            if self._config_signature == snapshot:
                self._config = config
                return

            self._config = config
            self._config_signature = snapshot
            self._risk_shadow = RiskManager(self._config)
            self._api_engine = APIDataEngine(self._config)
            self._setup_live_monitor(reset=True)

    def _config_snapshot(self, config: AIConfig):
        return (
            config.min_confidence_to_bet,
            config.kelly_default_fraction,
            config.max_active_bets,
            config.max_daily_bets,
            config.live_monitoring_enabled,
            config.live_update_interval,
            config.live_min_ev_alert,
            config.api_daily_budget,
            config.api_reserved_monitoring,
            config.api_reserved_enrichment,
            config.api_emergency_buffer,
        )

    def _build_notifier(self) -> Optional[TelegramNotifier]:
        if not self._config.telegram_enabled:
            return None

        token = self._config.telegram_bot_token
        chat_id = self._config.telegram_chat_id
        if not token or not chat_id:
            return None

        try:
            return TelegramNotifier(
                bot_token=token,
                chat_id=chat_id,
                min_ev=self._config.telegram_min_ev,
                min_confidence=self._config.telegram_min_confidence,
                rate_limit_seconds=self._config.telegram_rate_limit_seconds,
                live_alerts_enabled=self._config.telegram_live_alerts_enabled
            )
        except Exception as exc:
            logger.warning("⚠️  Telegram notifier non disponibile: %s", exc)
            return None

    def _setup_live_monitor(self, reset: bool = False):
        if reset and self._live_monitor:
            try:
                self._live_monitor.stop()
            except Exception:
                pass
            finally:
                self._live_monitor = None
                self._live_monitor_started = False

        if self._config.live_monitoring_enabled and self._notifier:
            self._live_monitor = LiveMonitor(
                telegram_notifier=self._notifier,
                update_interval=self._config.live_update_interval,
                min_ev_alert=self._config.live_min_ev_alert
            )
            self._live_monitor_started = False
        else:
            self._live_monitor = None
            self._live_monitor_started = False

    def _register_live_monitor(
        self,
        match: Dict[str, Any],
        ai_result: Dict[str, Any],
        time_to_kickoff_hours: Optional[float]
    ) -> bool:
        if not self._live_monitor:
            return False

        match_id = self._build_match_id(match)
        probability = ai_result.get("calibrated", {}).get("prob_calibrated", 0.5)
        odds = ai_result.get("summary", {}).get("odds", match.get("odds", 2.0))

        try:
            start_time = self._parse_datetime(match.get("match_datetime"))
            self._live_monitor.add_match(
                match_id=match_id,
                home_team=match.get("home", "Home"),
                away_team=match.get("away", "Away"),
                league=match.get("league", "Unknown"),
                pre_match_prob=probability,
                odds=odds,
                start_time=start_time
            )

            if not self._live_monitor_started:
                self._live_monitor.start()
                self._live_monitor_started = True
            return True
        except Exception as exc:
            logger.warning("⚠️  Impossibile registrare match per live monitor: %s", exc)
            return False

    def _log_portfolio_entry(self, match: Dict[str, Any], final_decision: Dict[str, Any], bankroll: float):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "home": match.get("home"),
            "away": match.get("away"),
            "league": match.get("league"),
            "stake": float(final_decision.get("stake", 0.0)),
            "bankroll": bankroll
        }
        with self._lock:
            self._portfolio_log.append(entry)
            self._portfolio_log = self._portfolio_log[-50:]

    def _build_portfolio_state(self, bankroll: float) -> Dict[str, Any]:
        with self._lock:
            active_bets = [
                {
                    "home": bet.get("home"),
                    "away": bet.get("away"),
                    "league": bet.get("league"),
                    "stake": bet.get("stake", 0.0)
                }
                for bet in self._portfolio_log[-10:]
            ]
        return {
            "bankroll": bankroll,
            "active_bets": active_bets
        }

    def _shadow_risk_check(self, match: Dict[str, Any], ai_result: Dict[str, Any], bankroll: float) -> Optional[Dict[str, Any]]:
        try:
            value = ai_result.get("value")
            confidence = ai_result.get("confidence")
            kelly = ai_result.get("kelly")
            if not (value and confidence and kelly):
                return None
            portfolio_state = self._build_portfolio_state(bankroll)
            return self._risk_shadow.decide(value, confidence, kelly, match, portfolio_state)
        except Exception as exc:
            logger.debug("Shadow risk check skipped: %s", exc)
            return None

    def _update_api_usage(self, api_context: Optional[Dict[str, Any]]):
        if not api_context:
            return

        metadata = api_context.get("metadata", {})
        calls = metadata.get("api_calls_used", 0)
        cache_used = metadata.get("cache_used", False)

        with self._lock:
            self._maybe_reset_daily_usage()
            self._api_usage_today += calls

            self._api_engine.stats["total_requests"] += 1
            if cache_used:
                self._api_engine.stats["cache_hits"] += 1
            else:
                self._api_engine.stats["cache_misses"] += 1
            self._api_engine.stats["api_calls"] += calls

    def _maybe_reset_daily_usage(self):
        today = datetime.utcnow().date()
        if today != self._usage_day:
            self._usage_day = today
            self._api_usage_today = 0
            self._api_engine.reset_statistics()

    def _maybe_schedule_backtest(self, ai_result: Dict[str, Any]) -> bool:
        if not self._config.track_model_performance:
            return False

        now = datetime.utcnow()
        if self._last_backtest_run and (now - self._last_backtest_run) < timedelta(hours=12):
            return False

        self._last_backtest_run = now
        thread = threading.Thread(
            target=self._run_backtest_safe,
            args=(
                ai_result.get("confidence", {}).get("confidence_score", 60.0),
                ai_result.get("value", {}).get("expected_value", 0.05)
            ),
            daemon=True
        )
        thread.start()
        return True

    def _run_backtest_safe(self, confidence_threshold: float, ev_threshold: float):
        try:
            if self._backtester is None:
                self._backtester = Backtester('data/historical.csv')

            strategy = self._build_strategy(confidence_threshold, ev_threshold)
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)

            report = self._backtester.run_backtest(
                strategy=strategy,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                initial_bankroll=10000
            )

            if 'error' not in report:
                self._last_backtest_summary = {
                    "roi": report['summary']['total_roi'],
                    "win_rate": report['summary']['win_rate'],
                    "bets": report['summary']['bets_placed'],
                    "max_drawdown": report['risk_metrics']['max_drawdown']
                }
                logger.info(
                    "📈 Backtest aggiornato: ROI %.1f%%, WinRate %.1f%%, Bets %s",
                    self._last_backtest_summary["roi"],
                    self._last_backtest_summary["win_rate"],
                    self._last_backtest_summary["bets"]
                )
        except Exception as exc:
            logger.warning("⚠️  Backtest in background fallito: %s", exc)

    def _build_strategy(self, confidence_threshold: float, ev_threshold: float):
        def strategy(match):
            odds = match.get('odds_1x2_home', 0)
            if odds <= 0:
                return None

            if confidence_threshold >= 70 and ev_threshold >= 0.05 and odds < 2.4:
                return {'market': '1x2_home', 'stake_amount': 75.0}
            if confidence_threshold >= 60 and odds < 2.0:
                return {'market': '1x2_home', 'stake_amount': 40.0}
            return None

        return strategy

    @staticmethod
    def _build_match_id(match: Dict[str, Any]) -> str:
        base = f"{match.get('home', 'home')}_{match.get('away', 'away')}"
        timestamp = match.get("match_datetime") or datetime.utcnow().isoformat()
        return f"{base}_{timestamp}"

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
