"""
Sistema Alert Intelligente Multi-Livello
=========================================

Sistema di alerting intelligente che priorizza notifiche basandosi su:
- Consensus AI
- Quote movement
- Arbitrage detection
- Anomaly detection
- Sentiment analysis
- Risk assessment
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Livelli di alert"""
    INFO = 1          # Informazione base
    LOW = 2           # Opportunità normale
    MEDIUM = 3        # Opportunità buona
    HIGH = 4          # Opportunità molto buona
    CRITICAL = 5      # Opportunità critica - azione immediata


class IntelligentAlertSystem:
    """
    Sistema di alerting intelligente multi-livello.
    """
    
    def __init__(self):
        self.alert_history = []
    
    def calculate_alert_level(
        self,
        ai_result: Dict[str, Any],
        consensus_result: Dict[str, Any],
        odds_movement: Optional[Dict[str, Any]] = None,
        arbitrage: Optional[Dict[str, Any]] = None,
        anomaly: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calcola livello di alert basato su tutti i fattori.
        
        Args:
            ai_result: Risultato completo pipeline AI
            consensus_result: Risultato consensus multi-modello
            odds_movement: Dati movimento quote
            arbitrage: Opportunità arbitrage (se disponibile)
            anomaly: Risultato anomaly detection (se disponibile)
            
        Returns:
            Dict con alert_level, priority, urgency, message
        """
        try:
            score = 0.0
            factors = []
            
            # 1. Consensus AI (0-30 punti)
            consensus_score = consensus_result.get('consensus_score', 0.0)
            agreement_level = consensus_result.get('agreement_level', 'UNKNOWN')
            
            if agreement_level == 'VERY_HIGH':
                score += 30
                factors.append("✅ Consensus AI: VERY_HIGH (tutti d'accordo)")
            elif agreement_level == 'HIGH':
                score += 20
                factors.append("✅ Consensus AI: HIGH")
            elif agreement_level == 'MEDIUM':
                score += 10
                factors.append("⚠️ Consensus AI: MEDIUM")
            elif agreement_level in ['LOW', 'VERY_LOW']:
                score -= 10
                factors.append("❌ Consensus AI: Basso (disaccordo)")
            
            # 2. Confidence (0-25 punti)
            confidence = ai_result.get('confidence_level', 0) or \
                        ai_result.get('summary', {}).get('confidence', 0)
            
            if confidence >= 85:
                score += 25
                factors.append("✅ Confidence: VERY_HIGH")
            elif confidence >= 70:
                score += 18
                factors.append("✅ Confidence: HIGH")
            elif confidence >= 50:
                score += 10
                factors.append("⚠️ Confidence: MEDIUM")
            else:
                score -= 5
                factors.append("❌ Confidence: Bassa")
            
            # 3. Value Score (0-20 punti)
            value_score = ai_result.get('value_score', 0) or \
                         ai_result.get('summary', {}).get('value_score', 0)
            
            if value_score >= 80:
                score += 20
                factors.append("✅ Value Score: Eccellente")
            elif value_score >= 70:
                score += 15
                factors.append("✅ Value Score: Buono")
            elif value_score >= 60:
                score += 10
                factors.append("⚠️ Value Score: Medio")
            else:
                score -= 5
                factors.append("❌ Value Score: Basso")
            
            # 4. Arbitrage (0-15 punti) - BONUS
            if arbitrage and arbitrage.get('found', False):
                arb_type = arbitrage.get('type', '')
                if arb_type == 'SURE_BET':
                    score += 15
                    factors.append("💰 ARBITRAGE: Sure bet trovato!")
                elif arb_type == 'STATISTICAL_ARB':
                    score += 10
                    factors.append("💰 ARBITRAGE: Statistical arbitrage")
                else:
                    score += 5
                    factors.append("💰 ARBITRAGE: Opportunità")
            
            # 5. Quote Movement (0-15 punti)
            if odds_movement:
                movement_pct = odds_movement.get('movement_percent', 0)
                urgency = odds_movement.get('urgency', 'NONE')
                
                if urgency == 'HIGH' and movement_pct < -5:
                    score += 15
                    factors.append(f"⚡ Quote in calo: {movement_pct:.1f}% - AZIONE IMMEDIATA")
                elif urgency == 'MEDIUM' and movement_pct < -3:
                    score += 10
                    factors.append(f"⚡ Quote in calo: {movement_pct:.1f}%")
                elif movement_pct < -2:
                    score += 5
                    factors.append(f"📉 Quote in calo: {movement_pct:.1f}%")
            
            # 6. Anomaly Detection (-20 a +5 punti)
            if anomaly:
                is_anomaly = anomaly.get('is_anomaly', False)
                severity = anomaly.get('severity', 'LOW')
                
                if is_anomaly and severity in ['HIGH', 'CRITICAL']:
                    score -= 20
                    factors.append(f"🚨 ANOMALIA: {severity} - Possibile manipolazione")
                elif is_anomaly and severity == 'MEDIUM':
                    score -= 10
                    factors.append("⚠️ ANOMALIA: Media - Attenzione")
                elif not is_anomaly:
                    score += 5
                    factors.append("✅ Nessuna anomalia rilevata")
            
            # 7. Sentiment (0-10 punti)
            sentiment = ai_result.get('sentiment', {})
            if isinstance(sentiment, dict):
                sentiment_score = sentiment.get('overall_sentiment', 0)
                if sentiment_score > 0.7:
                    score += 10
                    factors.append("🟢 Sentiment: Molto positivo")
                elif sentiment_score > 0.5:
                    score += 5
                    factors.append("🟡 Sentiment: Positivo")
                elif sentiment_score < 0.3:
                    score -= 5
                    factors.append("🔴 Sentiment: Negativo")
            
            # 8. EV (0-10 punti)
            ev = ai_result.get('ev', 0) or \
                 ai_result.get('summary', {}).get('expected_value', 0)
            
            if isinstance(ev, float) and ev < 1.0:
                ev = ev * 100  # Convert to %
            
            if ev >= 15:
                score += 10
                factors.append(f"✅ EV: {ev:+.1f}% (Eccellente)")
            elif ev >= 10:
                score += 7
                factors.append(f"✅ EV: {ev:+.1f}% (Buono)")
            elif ev >= 5:
                score += 3
                factors.append(f"⚠️ EV: {ev:+.1f}% (Moderato)")
            
            # Determina livello alert
            if score >= 80:
                alert_level = AlertLevel.CRITICAL
                priority = "CRITICAL"
                urgency = "IMMEDIATE"
                emoji = "🚨"
            elif score >= 60:
                alert_level = AlertLevel.HIGH
                priority = "HIGH"
                urgency = "HIGH"
                emoji = "🔥"
            elif score >= 40:
                alert_level = AlertLevel.MEDIUM
                priority = "MEDIUM"
                urgency = "MEDIUM"
                emoji = "⚡"
            elif score >= 20:
                alert_level = AlertLevel.LOW
                priority = "LOW"
                urgency = "LOW"
                emoji = "📊"
            else:
                alert_level = AlertLevel.INFO
                priority = "INFO"
                urgency = "NONE"
                emoji = "ℹ️"
            
            # Genera messaggio
            message = self._generate_alert_message(
                alert_level, score, factors, ai_result
            )
            
            return {
                'alert_level': alert_level.name,
                'alert_score': score,
                'priority': priority,
                'urgency': urgency,
                'emoji': emoji,
                'factors': factors,
                'message': message,
                'should_notify': alert_level.value >= AlertLevel.MEDIUM.value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating alert level: {e}", exc_info=True)
            return {
                'alert_level': 'INFO',
                'alert_score': 0.0,
                'priority': 'LOW',
                'urgency': 'NONE',
                'error': str(e)
            }
    
    def _generate_alert_message(
        self,
        alert_level: AlertLevel,
        score: float,
        factors: List[str],
        ai_result: Dict[str, Any]
    ) -> str:
        """Genera messaggio alert personalizzato."""
        match_data = ai_result.get('match_data', {})
        home = match_data.get('home', 'Home')
        away = match_data.get('away', 'Away')
        
        if alert_level == AlertLevel.CRITICAL:
            return (
                f"🚨 ALERT CRITICO - Opportunità Premium\n"
                f"{home} vs {away}\n"
                f"Score: {score:.0f}/100\n\n"
                f"Fattori:\n" + "\n".join(f"  {f}" for f in factors[:5]) +
                f"\n\n→ AZIONE IMMEDIATA RICHIESTA"
            )
        elif alert_level == AlertLevel.HIGH:
            return (
                f"🔥 Opportunità Alta Priorità\n"
                f"{home} vs {away}\n"
                f"Score: {score:.0f}/100\n\n"
                f"Fattori chiave:\n" + "\n".join(f"  {f}" for f in factors[:4])
            )
        elif alert_level == AlertLevel.MEDIUM:
            return (
                f"⚡ Opportunità Buona\n"
                f"{home} vs {away}\n"
                f"Score: {score:.0f}/100"
            )
        else:
            return (
                f"📊 Opportunità Trovata\n"
                f"{home} vs {away}\n"
                f"Score: {score:.0f}/100"
            )

