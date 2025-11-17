#!/usr/bin/env python3
"""
Report Automatici Telegram
===========================

Genera e invia report automatici su Telegram.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from betting_results_tracker import BettingResultsTracker
from ai_system.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class AutomatedReports:
    """Genera e invia report automatici"""
    
    def __init__(self, telegram_notifier: TelegramNotifier, tracker: BettingResultsTracker):
        self.notifier = telegram_notifier
        self.tracker = tracker
    
    def send_daily_report(self):
        """Invia report giornaliero"""
        stats = self.tracker.get_statistics(days=1)
        
        message = "📊 REPORT GIORNALIERO AUTOMAZIONE 24/7\n\n"
        message += f"📅 Data: {datetime.now().strftime('%d/%m/%Y')}\n\n"
        
        message += "📈 STATISTICHE OGGI:\n"
        message += f"• Opportunità trovate: {stats['total_opportunities']}\n"
        message += f"• Vincite: {stats['winners']}\n"
        message += f"• Perdite: {stats['losers']}\n"
        message += f"• In attesa: {stats['pending']}\n"
        message += f"• Win Rate: {stats['win_rate_percent']:.1f}%\n\n"
        
        message += "💰 PERFORMANCE:\n"
        message += f"• Stake totale: €{stats['total_stake']:.2f}\n"
        message += f"• Profit/Loss: €{stats['total_profit_loss']:.2f}\n"
        message += f"• ROI: {stats['roi_percent']:.1f}%\n\n"
        
        if stats['by_market']:
            message += "🎯 PER MARKET:\n"
            for market, data in list(stats['by_market'].items())[:5]:
                message += f"• {market}: {data['winners']}/{data['count']} ({data['win_rate']:.1f}%) - €{data['profit_loss']:.2f}\n"
        
        try:
            self.notifier.send_message(message)
            logger.info("✅ Report giornaliero inviato")
        except Exception as e:
            logger.error(f"❌ Errore invio report: {e}")
    
    def send_weekly_report(self):
        """Invia report settimanale"""
        stats = self.tracker.get_statistics(days=7)
        
        message = "📊 REPORT SETTIMANALE AUTOMAZIONE 24/7\n\n"
        message += f"📅 Periodo: {datetime.now() - timedelta(days=7):%d/%m/%Y} - {datetime.now():%d/%m/%Y}\n\n"
        
        message += "📈 STATISTICHE SETTIMANALI:\n"
        message += f"• Opportunità totali: {stats['total_opportunities']}\n"
        message += f"• Vincite: {stats['winners']}\n"
        message += f"• Perdite: {stats['losers']}\n"
        message += f"• Win Rate: {stats['win_rate_percent']:.1f}%\n\n"
        
        message += "💰 PERFORMANCE:\n"
        message += f"• Stake totale: €{stats['total_stake']:.2f}\n"
        message += f"• Profit/Loss: €{stats['total_profit_loss']:.2f}\n"
        message += f"• ROI: {stats['roi_percent']:.1f}%\n"
        message += f"• P/L medio: €{stats['average_profit_loss']:.2f}\n\n"
        
        if stats['by_league']:
            message += "🏆 TOP LEGHE:\n"
            sorted_leagues = sorted(stats['by_league'].items(), 
                                  key=lambda x: x[1]['profit_loss'], reverse=True)
            for league, data in sorted_leagues[:5]:
                message += f"• {league}: €{data['profit_loss']:.2f} ({data['win_rate']:.1f}%)\n"
        
        try:
            self.notifier.send_message(message)
            logger.info("✅ Report settimanale inviato")
        except Exception as e:
            logger.error(f"❌ Errore invio report: {e}")
    
    def send_monthly_report(self):
        """Invia report mensile"""
        stats = self.tracker.get_statistics(days=30)
        
        message = "📊 REPORT MENSILE AUTOMAZIONE 24/7\n\n"
        message += f"📅 Mese: {datetime.now().strftime('%B %Y')}\n\n"
        
        message += "📈 STATISTICHE MENSILI:\n"
        message += f"• Opportunità totali: {stats['total_opportunities']}\n"
        message += f"• Vincite: {stats['winners']}\n"
        message += f"• Perdite: {stats['losers']}\n"
        message += f"• Win Rate: {stats['win_rate_percent']:.1f}%\n\n"
        
        message += "💰 PERFORMANCE:\n"
        message += f"• Stake totale: €{stats['total_stake']:.2f}\n"
        message += f"• Profit/Loss: €{stats['total_profit_loss']:.2f}\n"
        message += f"• ROI: {stats['roi_percent']:.1f}%\n\n"
        
        # Analisi dettagliata
        if stats['by_market']:
            message += "🎯 PERFORMANCE PER MARKET:\n"
            for market, data in stats['by_market'].items():
                message += f"• {market}: {data['count']} bets, {data['win_rate']:.1f}% WR, €{data['profit_loss']:.2f}\n"
            message += "\n"
        
        if stats['by_league']:
            message += "🏆 PERFORMANCE PER LEGA:\n"
            sorted_leagues = sorted(stats['by_league'].items(), 
                                  key=lambda x: x[1]['profit_loss'], reverse=True)
            for league, data in sorted_leagues[:10]:
                message += f"• {league}: €{data['profit_loss']:.2f} ({data['win_rate']:.1f}%)\n"
        
        try:
            self.notifier.send_message(message)
            logger.info("✅ Report mensile inviato")
        except Exception as e:
            logger.error(f"❌ Errore invio report: {e}")

