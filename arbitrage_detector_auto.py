"""
Sistema Rilevamento Arbitraggi Multi-Bookmaker
===============================================

Confronta quote tra bookmaker diversi e trova arbitraggi (sure bets).
Alert immediato per arbitraggi con profitto garantito.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Opportunità di arbitraggio"""
    match_id: str
    match_data: Dict[str, Any]
    market: str  # '1X2', 'over_under', etc.
    bookmakers: Dict[str, Dict[str, float]]  # {bookmaker: {outcome: odds}}
    guaranteed_profit_pct: float
    total_stake: float
    profit: float
    stakes: Dict[str, Dict[str, float]]  # {bookmaker: {outcome: stake}}
    timestamp: datetime


class ArbitrageDetectorAuto:
    """
    Rileva opportunità di arbitraggio tra bookmaker.
    """
    
    def __init__(self, min_profit_pct: float = 1.0):
        """
        Args:
            min_profit_pct: Profitto minimo % per considerare arbitraggio
        """
        self.min_profit_pct = min_profit_pct
        self.detected_arbitrages: List[ArbitrageOpportunity] = []
    
    def detect_arbitrage(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        bookmaker_odds: Dict[str, Dict[str, float]]
    ) -> Optional[ArbitrageOpportunity]:
        """
        Rileva arbitraggio per una partita.
        
        Args:
            match_id: ID partita
            match_data: Dati partita
            bookmaker_odds: {bookmaker: {outcome: odds}}
                          Es: {'bet365': {'home': 2.1, 'draw': 3.4, 'away': 3.2},
                               'pinnacle': {'home': 2.0, 'draw': 3.5, 'away': 3.3}}
        
        Returns:
            ArbitrageOpportunity se trovata, None altrimenti
        """
        try:
            # Per market 1X2, cerca arbitraggio
            # Un arbitraggio esiste se: 1/odds_home + 1/odds_draw + 1/odds_away < 1
            
            # Trova migliori quote per ogni outcome
            best_odds = self._find_best_odds(bookmaker_odds)
            
            if not best_odds:
                return None
            
            # Calcola somma probabilità implicite
            total_implied_prob = sum(1.0 / odds for odds in best_odds.values() if odds > 0)
            
            # Se totale < 1, c'è arbitraggio
            if total_implied_prob < 1.0:
                profit_pct = ((1.0 / total_implied_prob) - 1.0) * 100
                
                if profit_pct >= self.min_profit_pct:
                    # Calcola stake ottimali
                    total_stake = 100.0  # Stake totale esempio
                    stakes = self._calculate_optimal_stakes(
                        best_odds, total_stake, total_implied_prob
                    )
                    
                    profit = total_stake * (profit_pct / 100)
                    
                    # Trova bookmaker per ogni outcome
                    bookmakers_used = self._find_bookmakers_for_odds(
                        bookmaker_odds, best_odds
                    )
                    
                    opportunity = ArbitrageOpportunity(
                        match_id=match_id,
                        match_data=match_data,
                        market='1X2',
                        bookmakers=bookmakers_used,
                        guaranteed_profit_pct=profit_pct,
                        total_stake=total_stake,
                        profit=profit,
                        stakes=stakes,
                        timestamp=datetime.now()
                    )
                    
                    logger.info(
                        f"💰 Arbitraggio trovato: {match_id} - "
                        f"Profitto: {profit_pct:.2f}%"
                    )
                    
                    return opportunity
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Errore rilevamento arbitraggio: {e}")
            return None
    
    def _find_best_odds(
        self,
        bookmaker_odds: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Trova migliori quote per ogni outcome"""
        best_odds = {}
        
        outcomes = ['home', 'draw', 'away']
        
        for outcome in outcomes:
            best_odd = 0.0
            for bookmaker, odds_dict in bookmaker_odds.items():
                if outcome in odds_dict:
                    if odds_dict[outcome] > best_odd:
                        best_odd = odds_dict[outcome]
            if best_odd > 0:
                best_odds[outcome] = best_odd
        
        return best_odds
    
    def _calculate_optimal_stakes(
        self,
        best_odds: Dict[str, float],
        total_stake: float,
        total_implied_prob: float
    ) -> Dict[str, Dict[str, float]]:
        """Calcola stake ottimali per garantire profitto"""
        stakes = {}
        
        for outcome, odds in best_odds.items():
            # Stake proporzionale alla probabilità implicita
            implied_prob = 1.0 / odds
            stake = (implied_prob / total_implied_prob) * total_stake
            stakes[outcome] = {'stake': stake, 'odds': odds}
        
        return stakes
    
    def _find_bookmakers_for_odds(
        self,
        bookmaker_odds: Dict[str, Dict[str, float]],
        best_odds: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Trova bookmaker che offrono le migliori quote"""
        bookmakers_used = {}
        
        for outcome, best_odd in best_odds.items():
            for bookmaker, odds_dict in bookmaker_odds.items():
                if outcome in odds_dict and odds_dict[outcome] == best_odd:
                    if bookmaker not in bookmakers_used:
                        bookmakers_used[bookmaker] = {}
                    bookmakers_used[bookmaker][outcome] = best_odd
                    break
        
        return bookmakers_used
    
    def format_arbitrage_message(self, opportunity: ArbitrageOpportunity) -> str:
        """Formatta messaggio per alert arbitraggio"""
        match_data = opportunity.match_data
        home = match_data.get('home', 'Home')
        away = match_data.get('away', 'Away')
        
        message = f"💰 ARBITRAGGIO TROVATO - Profitto Garantito!\n\n"
        message += f"⚽ {home} vs {away}\n"
        message += f"📊 Market: {opportunity.market}\n\n"
        message += f"💵 Profitto: {opportunity.guaranteed_profit_pct:.2f}%\n"
        message += f"💰 Stake Totale: €{opportunity.total_stake:.2f}\n"
        message += f"💶 Profitto: €{opportunity.profit:.2f}\n\n"
        message += f"📋 Bookmaker:\n"
        
        for bookmaker, outcomes in opportunity.bookmakers.items():
            message += f"  • {bookmaker}:\n"
            for outcome, odds in outcomes.items():
                stake = opportunity.stakes.get(outcome, {}).get('stake', 0)
                message += f"    - {outcome.upper()}: {odds:.2f} (stake: €{stake:.2f})\n"
        
        message += f"\n⏰ Azione immediata richiesta!"
        
        return message

