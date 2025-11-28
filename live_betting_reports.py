#!/usr/bin/env python3
"""
Sistema Report Automatici Live Betting
========================================

Genera report automatici per performance live betting:
- Report giornaliero
- Report settimanale
- Alert se win rate scende sotto soglia
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from live_betting_performance_tracker import LiveBettingPerformanceTracker

logger = logging.getLogger(__name__)


class LiveBettingReports:
    """Genera report automatici per live betting"""
    
    def __init__(self, tracker: LiveBettingPerformanceTracker, notifier=None):
        self.tracker = tracker
        self.notifier = notifier
        self.min_win_rate_alert = 45.0  # Soglia minima win rate per alert
    
    def generate_daily_report(self) -> str:
        """
        Genera report giornaliero
        
        Returns:
            Stringa con report formattato
        """
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        # Ottieni performance di ieri
        performance = self.tracker.get_all_market_performance(days=1)
        
        if not performance:
            return "📊 REPORT GIORNALIERO LIVE BETTING\n\nNessuna opportunità ieri."
        
        # Calcola totali (gestendo valori None)
        total_opps = sum((p.get('total') or 0) for p in performance.values())
        total_winners = sum((p.get('winners') or 0) for p in performance.values())
        total_losers = sum((p.get('losers') or 0) for p in performance.values())
        total_pl = sum((p.get('total_profit_loss') or 0) for p in performance.values())
        
        overall_win_rate = (total_winners / (total_winners + total_losers) * 100) if (total_winners + total_losers) > 0 else 0
        
        # Formatta report
        report = f"""📊 REPORT GIORNALIERO LIVE BETTING
📅 Data: {yesterday.strftime('%d/%m/%Y')}

📈 PERFORMANCE GENERALE:
   • Opportunità totali: {total_opps}
   • Vincite: {total_winners}
   • Perdite: {total_losers}
   • Win Rate: {overall_win_rate:.1f}%
   • Profit/Loss: {total_pl:+.2f} unità

📊 PERFORMANCE PER MERCATO:
"""
        
        # Top 5 mercati per volume (gestendo valori None)
        sorted_markets = sorted(
            performance.items(), 
            key=lambda x: (x[1].get('total') or 0), 
            reverse=True
        )[:5]
        
        for market, stats in sorted_markets:
            total = stats.get('total') or 0
            win_rate = stats.get('win_rate') or 0
            winners = stats.get('winners') or 0
            losers = stats.get('losers') or 0
            pl = stats.get('total_profit_loss') or 0
            avg_conf = stats.get('avg_confidence') or 0
            avg_ev = stats.get('avg_ev') or 0
            report += f"""
   {market.upper()}:
      • Totale: {total}
      • Win Rate: {win_rate:.1f}% ({winners}W/{losers}L)
      • P/L: {pl:+.2f}
      • Avg Conf: {avg_conf:.1f}%
      • Avg EV: {avg_ev:.1f}%
"""
        
        # Mercati migliori e peggiori (gestendo valori None)
        if len(performance) > 1:
            best_market = max(performance.items(), key=lambda x: (x[1].get('win_rate') or 0))
            worst_market = min(performance.items(), key=lambda x: (x[1].get('win_rate') or 0))
            
            best_wr = best_market[1].get('win_rate') or 0
            worst_wr = worst_market[1].get('win_rate') or 0
            report += f"""
🏆 MIGLIORE: {best_market[0].upper()} ({best_wr:.1f}%)
⚠️  PEGGIORE: {worst_market[0].upper()} ({worst_wr:.1f}%)
"""
        
        return report
    
    def generate_weekly_report(self) -> str:
        """
        Genera report settimanale
        
        Returns:
            Stringa con report formattato
        """
        # Ottieni performance ultimi 7 giorni
        performance = self.tracker.get_all_market_performance(days=7)
        
        if not performance:
            return "📊 REPORT SETTIMANALE LIVE BETTING\n\nNessuna opportunità questa settimana."
        
        # Calcola totali (gestendo valori None)
        total_opps = sum((p.get('total') or 0) for p in performance.values())
        total_winners = sum((p.get('winners') or 0) for p in performance.values())
        total_losers = sum((p.get('losers') or 0) for p in performance.values())
        total_pl = sum((p.get('total_profit_loss') or 0) for p in performance.values())
        
        overall_win_rate = (total_winners / (total_winners + total_losers) * 100) if (total_winners + total_losers) > 0 else 0
        
        # Formatta report
        report = f"""📊 REPORT SETTIMANALE LIVE BETTING
📅 Periodo: Ultimi 7 giorni

📈 PERFORMANCE GENERALE:
   • Opportunità totali: {total_opps}
   • Vincite: {total_winners}
   • Perdite: {total_losers}
   • Win Rate: {overall_win_rate:.1f}%
   • Profit/Loss: {total_pl:+.2f} unità

📊 TOP 5 MERCATI (per volume):
"""
        
        # Ordina per volume, gestendo valori None
        sorted_markets = sorted(
            performance.items(), 
            key=lambda x: (x[1].get('total') or 0), 
            reverse=True
        )[:5]
        
        for i, (market, stats) in enumerate(sorted_markets, 1):
            total = stats.get('total') or 0
            win_rate = stats.get('win_rate') or 0
            pl = stats.get('total_profit_loss') or 0
            report += f"""
   {i}. {market.upper()}
      • Volume: {total} | Win Rate: {win_rate:.1f}% | P/L: {pl:+.2f}
"""
        
        # Soglie dinamiche
        thresholds = self.tracker.calculate_dynamic_thresholds()
        if thresholds:
            report += "\n🎯 SOGLIE DINAMICHE AGGIORNATE:\n"
            for market, thresh in list(thresholds.items())[:5]:
                min_conf = thresh.get('min_confidence') or 0
                min_ev = thresh.get('min_ev') or 0
                reason = thresh.get('reason', 'N/A')
                report += f"   • {market.upper()}: Conf {min_conf:.1f}%, EV {min_ev:.1f}% ({reason})\n"
        
        return report
    
    def check_win_rate_alert(self) -> Optional[str]:
        """
        Verifica se win rate è sotto soglia e genera alert
        
        Returns:
            Stringa alert o None
        """
        # Performance ultimi 7 giorni
        performance = self.tracker.get_all_market_performance(days=7)
        
        if not performance:
            return None
        
        # Calcola win rate complessivo (gestendo valori None)
        total_winners = sum((p.get('winners') or 0) for p in performance.values())
        total_losers = sum((p.get('losers') or 0) for p in performance.values())
        
        if (total_winners + total_losers) == 0:
            return None
        
        overall_win_rate = (total_winners / (total_winners + total_losers) * 100)
        
        if overall_win_rate < self.min_win_rate_alert:
            # Trova mercati problematici (gestendo valori None)
            problematic_markets = [
                (market, stats.get('win_rate') or 0)
                for market, stats in performance.items()
                if (stats.get('win_rate') or 0) < self.min_win_rate_alert 
                and ((stats.get('winners') or 0) + (stats.get('losers') or 0)) >= 3
            ]
            
            alert = f"""⚠️  ALERT: WIN RATE BASSO

📉 Win Rate complessivo: {overall_win_rate:.1f}% (soglia: {self.min_win_rate_alert}%)

⚠️  Mercati problematici:
"""
            for market, wr in problematic_markets[:5]:
                alert += f"   • {market.upper()}: {wr:.1f}%\n"
            
            alert += "\n💡 Soglie dinamiche verranno aumentate automaticamente."
            
            return alert
        
        return None
    
    def send_daily_report(self):
        """Invia report giornaliero via Telegram"""
        if not self.notifier:
            logger.warning("⚠️  Notifier non disponibile per report giornaliero")
            return
        
        try:
            report = self.generate_daily_report()
            self.notifier._send_message(report, parse_mode="HTML")
            logger.info("✅ Report giornaliero inviato")
        except Exception as e:
            logger.error(f"❌ Errore invio report giornaliero: {e}")
    
    def send_weekly_report(self):
        """Invia report settimanale via Telegram"""
        if not self.notifier:
            logger.warning("⚠️  Notifier non disponibile per report settimanale")
            return
        
        try:
            report = self.generate_weekly_report()
            self.notifier._send_message(report, parse_mode="HTML")
            logger.info("✅ Report settimanale inviato")
        except Exception as e:
            logger.error(f"❌ Errore invio report settimanale: {e}")
    
    def check_and_send_alerts(self):
        """Verifica e invia alert se necessario"""
        if not self.notifier:
            return
        
        try:
            alert = self.check_win_rate_alert()
            if alert:
                self.notifier._send_message(alert, parse_mode="HTML")
                logger.warning("⚠️  Alert win rate basso inviato")
        except Exception as e:
            logger.error(f"❌ Errore verifica alert: {e}")

