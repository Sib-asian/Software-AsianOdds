"""
Sistema Alert Pre-Partita Intelligente
=======================================

Analizza partite 1-2 ore prima e invia alert tempestivi.
Reminder 30 min prima e aggiorna quote fino a kickoff.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PreMatchAlert:
    """Alert pre-partita"""
    match_id: str
    match_data: Dict[str, Any]
    alert_type: str  # 'OPPORTUNITY', 'REMINDER', 'QUOTE_UPDATE'
    time_to_kickoff: timedelta
    message: str
    priority: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    timestamp: datetime


class PreMatchAlerter:
    """
    Sistema di alert pre-partita intelligente.
    """
    
    def __init__(self, notifier=None):
        self.notifier = notifier
        self.scheduled_alerts: Dict[str, List[PreMatchAlert]] = {}
        self.sent_alerts: set = set()
    
    def schedule_alert(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        match_date: datetime,
        alert_type: str = 'OPPORTUNITY',
        message: Optional[str] = None
    ):
        """
        Schedula alert per una partita.
        
        Args:
            match_id: ID partita
            match_data: Dati partita
            match_date: Data/ora kickoff
            alert_type: Tipo alert
            message: Messaggio personalizzato
        """
        try:
            now = datetime.now()
            time_to_kickoff = match_date - now
            
            # Non schedulare se partita già iniziata o troppo lontana (>24h)
            if time_to_kickoff.total_seconds() < 0:
                return
            if time_to_kickoff.total_seconds() > 86400:  # 24 ore
                return
            
            # Determina priorità basata su tempo rimanente
            hours_to_kickoff = time_to_kickoff.total_seconds() / 3600
            
            if hours_to_kickoff < 0.5:  # < 30 min
                priority = 'CRITICAL'
            elif hours_to_kickoff < 1:  # < 1 ora
                priority = 'HIGH'
            elif hours_to_kickoff < 2:  # < 2 ore
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
            
            # Genera messaggio se non fornito
            if not message:
                message = self._generate_alert_message(
                    match_data, time_to_kickoff, alert_type
                )
            
            alert = PreMatchAlert(
                match_id=match_id,
                match_data=match_data,
                alert_type=alert_type,
                time_to_kickoff=time_to_kickoff,
                message=message,
                priority=priority,
                timestamp=now
            )
            
            if match_id not in self.scheduled_alerts:
                self.scheduled_alerts[match_id] = []
            
            self.scheduled_alerts[match_id].append(alert)
            logger.debug(f"✅ Alert schedulato per {match_id}: {alert_type} ({priority})")
            
        except Exception as e:
            logger.error(f"❌ Errore schedulazione alert: {e}")
    
    def check_and_send_alerts(self) -> List[PreMatchAlert]:
        """
        Controlla e invia alert scaduti.
        
        Returns:
            Lista di alert inviati
        """
        sent = []
        now = datetime.now()
        
        for match_id, alerts in list(self.scheduled_alerts.items()):
            for alert in alerts[:]:  # Copia lista per modificarla
                try:
                    # Calcola tempo rimanente
                    time_remaining = alert.time_to_kickoff - (now - alert.timestamp)
                    
                    # Invia alert se è il momento giusto
                    should_send = False
                    
                    if alert.alert_type == 'OPPORTUNITY':
                        # Invia subito se opportunità trovata
                        should_send = True
                    elif alert.alert_type == 'REMINDER':
                        # Invia 30 min prima
                        if 0 <= time_remaining.total_seconds() <= 1800:  # 30 min
                            should_send = True
                    elif alert.alert_type == 'QUOTE_UPDATE':
                        # Invia se quote cambiate significativamente
                        should_send = True
                    
                    if should_send:
                        alert_key = f"{match_id}_{alert.alert_type}_{alert.timestamp.isoformat()}"
                        if alert_key not in self.sent_alerts:
                            if self.notifier:
                                self._send_alert(alert)
                            sent.append(alert)
                            self.sent_alerts.add(alert_key)
                            
                            # Rimuovi alert inviato
                            alerts.remove(alert)
                
                except Exception as e:
                    logger.error(f"❌ Errore invio alert: {e}")
                    continue
            
            # Rimuovi match senza alert
            if not alerts:
                del self.scheduled_alerts[match_id]
        
        return sent
    
    def _send_alert(self, alert: PreMatchAlert):
        """Invia alert via Telegram"""
        try:
            if not self.notifier:
                return
            
            # Formatta messaggio
            emoji = {
                'CRITICAL': '🚨',
                'HIGH': '🔥',
                'MEDIUM': '⚡',
                'LOW': '📊'
            }.get(alert.priority, 'ℹ️')
            
            hours = alert.time_to_kickoff.total_seconds() / 3600
            time_str = f"{int(hours)}h {int((hours % 1) * 60)}min" if hours >= 1 else f"{int(alert.time_to_kickoff.total_seconds() / 60)}min"
            
            full_message = f"{emoji} {alert.priority} - {alert.alert_type}\n\n"
            full_message += f"⏰ Kickoff tra: {time_str}\n\n"
            full_message += alert.message
            
            # Usa send_betting_opportunity se disponibile
            try:
                self.notifier.send_betting_opportunity(
                    alert.match_data,
                    {'summary': {'expected_value': 0, 'confidence': 0}},
                    opportunity_type=f"PRE-MATCH {alert.alert_type}"
                )
            except:
                # Fallback a messaggio semplice
                self.notifier._send_message(full_message)
            
            logger.info(f"✅ Alert inviato: {alert.match_id} ({alert.alert_type})")
            
        except Exception as e:
            logger.error(f"❌ Errore invio alert Telegram: {e}")
    
    def _generate_alert_message(
        self,
        match_data: Dict[str, Any],
        time_to_kickoff: timedelta,
        alert_type: str
    ) -> str:
        """Genera messaggio alert"""
        home = match_data.get('home', 'Home')
        away = match_data.get('away', 'Away')
        league = match_data.get('league', 'League')
        
        hours = time_to_kickoff.total_seconds() / 3600
        time_str = f"{int(hours)}h {int((hours % 1) * 60)}min" if hours >= 1 else f"{int(time_to_kickoff.total_seconds() / 60)}min"
        
        if alert_type == 'OPPORTUNITY':
            return (
                f"🎯 Opportunità trovata!\n\n"
                f"{home} vs {away}\n"
                f"Lega: {league}\n"
                f"Kickoff tra: {time_str}\n\n"
                f"Controlla le quote e valuta l'opportunità!"
            )
        elif alert_type == 'REMINDER':
            return (
                f"⏰ Reminder Partita\n\n"
                f"{home} vs {away}\n"
                f"Lega: {league}\n"
                f"Kickoff tra: {time_str}\n\n"
                f"Ultima possibilità per scommettere!"
            )
        elif alert_type == 'QUOTE_UPDATE':
            return (
                f"📊 Quote Aggiornate\n\n"
                f"{home} vs {away}\n"
                f"Lega: {league}\n"
                f"Kickoff tra: {time_str}\n\n"
                f"Le quote sono cambiate - controlla se c'è ancora valore!"
            )
        else:
            return f"Alert per {home} vs {away} - Kickoff tra {time_str}"

