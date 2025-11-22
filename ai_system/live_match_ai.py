"""
Live Match AI - Sistema AI Dedicato ai Match Live
==================================================

Sistema AI specializzato esclusivamente per l'analisi di partite in corso.
Focus su:
- Analisi in tempo reale delle situazioni di gioco
- Rilevamento pattern specifici per match live
- Predizioni basate su dati live (score, statistiche, eventi)
- Ottimizzazione per velocità e precisione

Usage:
    live_ai = LiveMatchAI(ai_pipeline=None)
    analysis = live_ai.analyze_live_match(match_data, live_data, odds_data)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class LiveMatchAI:
    """
    Sistema AI dedicato esclusivamente all'analisi di match live.
    
    Caratteristiche:
    - Analisi in tempo reale basata su score, statistiche, eventi
    - Rilevamento pattern specifici (momentum, pressione, situazioni critiche)
    - Predizioni adattive che si aggiornano con il progredire della partita
    - Ottimizzato per velocità (analisi in < 1 secondo)
    """
    
    def __init__(
        self,
        ai_pipeline=None,
        min_confidence: float = 70.0,
        min_ev: float = 8.0
    ):
        """
        Inizializza Live Match AI.
        
        Args:
            ai_pipeline: Pipeline AI opzionale per analisi avanzate
            min_confidence: Confidence minima per segnali (default: 70%)
            min_ev: EV minimo per segnali (default: 8%)
        """
        self.ai_pipeline = ai_pipeline
        self.min_confidence = min_confidence
        self.min_ev = min_ev
        
        # Cache per analisi recenti (evita ricalcoli inutili)
        self._analysis_cache: Dict[str, Tuple[Dict, float]] = {}
        self._cache_ttl = 30  # 30 secondi TTL
        
        logger.info("✅ Live Match AI initialized")

    def _safe_get(self, data: Dict[str, Any], key: str, default: Any) -> Any:
        """
        Estrae valore da dizionario gestendo correttamente None.

        Il metodo dict.get() ritorna il default solo se la chiave non esiste.
        Se la chiave esiste ma il valore è None, ritorna None.
        Questa funzione ritorna il default anche quando il valore è None.

        Args:
            data: Dizionario da cui estrarre il valore
            key: Chiave da cercare
            default: Valore di default se chiave non esiste o valore è None

        Returns:
            Valore estratto o default se None/mancante
        """
        value = data.get(key)
        return value if value is not None else default

    def analyze_live_match(
        self,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any],
        odds_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analizza partita live e genera opportunità.
        
        Args:
            match_data: Dati partita (home, away, league, date, odds)
            live_data: Dati live (score, minute, stats, events)
            odds_data: Quote correnti (opzionale)
            
        Returns:
            Dict con analisi completa e opportunità trovate
        """
        try:
            match_id = match_data.get('id') or f"{match_data.get('home', '')}_{match_data.get('away', '')}"
            
            # Check cache
            cache_key = f"{match_id}_{live_data.get('minute', 0)}_{live_data.get('score_home', 0)}_{live_data.get('score_away', 0)}"
            current_time = datetime.now().timestamp()
            
            if cache_key in self._analysis_cache:
                cached_result, cache_time = self._analysis_cache[cache_key]
                if current_time - cache_time < self._cache_ttl:
                    logger.debug(f"✅ Using cached analysis for {match_id}")
                    return cached_result
            
            # 1. Analisi situazione corrente
            situation_analysis = self._analyze_current_situation(match_data, live_data)
            
            # 2. Rilevamento pattern
            patterns = self._detect_patterns(match_data, live_data)
            
            # 3. Calcolo probabilità aggiornate
            updated_probabilities = self._calculate_updated_probabilities(
                match_data, live_data, situation_analysis, patterns
            )
            
            # 4. Identificazione opportunità
            opportunities = self._identify_opportunities(
                match_data, live_data, updated_probabilities, odds_data, situation_analysis, patterns
            )
            
            # 5. Calcolo confidence e EV
            enhanced_opportunities = self._enhance_with_confidence_ev(
                opportunities, match_data, live_data, updated_probabilities
            )
            
            # 6. Filtraggio (solo opportunità valide)
            filtered_opportunities = [
                opp for opp in enhanced_opportunities
                if opp.get('confidence', 0) >= self.min_confidence
                and opp.get('ev', 0) >= self.min_ev
            ]
            
            # Costruisci risultato
            result = {
                'match_id': match_id,
                'match_data': match_data,
                'live_data': live_data,
                'situation_analysis': situation_analysis,
                'patterns': patterns,
                'updated_probabilities': updated_probabilities,
                'opportunities': filtered_opportunities,
                'timestamp': datetime.now().isoformat()
            }
            
            # Salva in cache
            self._analysis_cache[cache_key] = (result, current_time)
            
            # Cleanup cache vecchia (> 5 minuti)
            self._cleanup_cache(current_time)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing live match: {e}", exc_info=True)
            return {
                'match_id': match_data.get('id', 'unknown'),
                'error': str(e),
                'opportunities': []
            }
    
    def _analyze_current_situation(
        self,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analizza situazione corrente della partita.

        Returns:
            Dict con analisi situazione (momentum, pressione, equilibrio, etc.)
        """
        # Estrai dati con gestione sicura di None
        score_home = live_data.get('score_home') if live_data.get('score_home') is not None else 0
        score_away = live_data.get('score_away') if live_data.get('score_away') is not None else 0
        minute = live_data.get('minute') if live_data.get('minute') is not None else 0
        total_goals = score_home + score_away

        # Possesso palla
        possession_home = live_data.get('possession_home') if live_data.get('possession_home') is not None else 50
        possession_away = 100 - possession_home

        # Tiri
        shots_home = live_data.get('shots_home') if live_data.get('shots_home') is not None else 0
        shots_away = live_data.get('shots_away') if live_data.get('shots_away') is not None else 0
        shots_on_target_home = live_data.get('shots_on_target_home') if live_data.get('shots_on_target_home') is not None else 0
        shots_on_target_away = live_data.get('shots_on_target_away') if live_data.get('shots_on_target_away') is not None else 0
        
        # Calcoli
        goal_difference = score_home - score_away
        shot_difference = shots_home - shots_away
        possession_difference = possession_home - possession_away
        
        # Momentum (chi sta dominando)
        momentum_score = 0.0
        if shot_difference > 3:
            momentum_score += 0.3
        if possession_difference > 10:
            momentum_score += 0.2
        if shots_on_target_home > shots_on_target_away + 2:
            momentum_score += 0.2
        
        # Se away sta dominando, momentum negativo
        if shot_difference < -3:
            momentum_score -= 0.3
        if possession_difference < -10:
            momentum_score -= 0.2
        if shots_on_target_away > shots_on_target_home + 2:
            momentum_score -= 0.2
        
        # Pressione (quanto è probabile un gol)
        pressure_score = 0.0
        if shots_on_target_home > 3:
            pressure_score += 0.2
        if shots_on_target_away > 3:
            pressure_score += 0.2
        if minute > 70 and abs(goal_difference) <= 1:
            pressure_score += 0.3  # Finale partita, partita aperta
        if minute > 80 and goal_difference == 0:
            pressure_score += 0.3  # Finale, pareggio
        
        # Equilibrio partita
        is_balanced = abs(goal_difference) <= 1 and abs(shot_difference) <= 5
        
        # Situazione critica
        is_critical = False
        if minute > 75 and abs(goal_difference) == 1:
            is_critical = True  # Finale, un gol di differenza
        if minute > 80 and goal_difference == 0:
            is_critical = True  # Finale, pareggio
        
        return {
            'score_home': score_home,
            'score_away': score_away,
            'minute': minute,
            'goal_difference': goal_difference,
            'total_goals': total_goals,
            'possession_home': possession_home,
            'possession_away': possession_away,
            'shots_home': shots_home,
            'shots_away': shots_away,
            'shots_on_target_home': shots_on_target_home,
            'shots_on_target_away': shots_on_target_away,
            'momentum_score': momentum_score,  # -1 (away domina) a +1 (home domina)
            'pressure_score': pressure_score,  # 0 (bassa) a 1 (alta)
            'is_balanced': is_balanced,
            'is_critical': is_critical,
            'home_advantage': momentum_score > 0.2,
            'away_advantage': momentum_score < -0.2
        }
    
    def _detect_patterns(
        self,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Rileva pattern specifici nella partita.

        Returns:
            Dict con pattern rilevati
        """
        # Estrai dati con gestione sicura di None
        score_home = live_data.get('score_home') if live_data.get('score_home') is not None else 0
        score_away = live_data.get('score_away') if live_data.get('score_away') is not None else 0
        minute = live_data.get('minute') if live_data.get('minute') is not None else 0
        goal_difference = score_home - score_away
        
        patterns = {
            'comeback_possible': False,
            'defensive_mode': False,
            'attacking_mode': False,
            'under_pressure': False,
            'comfortable_lead': False,
            'late_goal_risk': False,
            'high_scoring': False,
            'low_scoring': False
        }
        
        total_goals = score_home + score_away
        
        # Pattern: Ribaltone possibile
        if minute > 60 and goal_difference == 1:
            patterns['comeback_possible'] = True
        
        # Pattern: Modalità difensiva (chi è in vantaggio)
        if goal_difference > 0 and minute > 70:
            patterns['defensive_mode'] = True  # Home in vantaggio, probabile difesa
        elif goal_difference < 0 and minute > 70:
            patterns['attacking_mode'] = True  # Away in svantaggio, probabile attacco
        
        # Pattern: Sotto pressione (chi è in svantaggio)
        if goal_difference < 0 and minute > 60:
            patterns['under_pressure'] = True  # Home in svantaggio
        
        # Pattern: Vantaggio comodo
        if abs(goal_difference) >= 2:
            patterns['comfortable_lead'] = True
        
        # Pattern: Rischio gol tardivo
        if minute > 80 and abs(goal_difference) <= 1:
            patterns['late_goal_risk'] = True
        
        # Pattern: Partita ad alto/basso scoring
        if total_goals >= 3 and minute < 60:
            patterns['high_scoring'] = True
        elif total_goals == 0 and minute > 60:
            patterns['low_scoring'] = True
        
        return patterns
    
    def _calculate_updated_probabilities(
        self,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any],
        situation: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calcola probabilità aggiornate basate su situazione live.
        
        Returns:
            Dict con probabilità aggiornate (home_win, draw, away_win, over_2_5, etc.)
        """
        # Probabilità base (da match_data o default)
        prob_home_base = 0.33
        prob_draw_base = 0.33
        prob_away_base = 0.34
        
        # Se disponibile, usa probabilità pre-match
        if 'pre_match_prob_home' in match_data:
            prob_home_base = match_data['pre_match_prob_home']
            prob_draw_base = match_data.get('pre_match_prob_draw', 0.30)
            prob_away_base = 1.0 - prob_home_base - prob_draw_base
        
        # Aggiustamenti basati su situazione live
        score_home = situation['score_home']
        score_away = situation['score_away']
        minute = situation['minute']
        goal_difference = situation['goal_difference']
        momentum = situation['momentum_score']
        
        # Aggiustamento per score corrente
        if goal_difference > 0:
            # Home in vantaggio
            prob_home_base += 0.15 * (minute / 90)  # Più probabile se avanti
            prob_draw_base -= 0.10 * (minute / 90)
            prob_away_base -= 0.05 * (minute / 90)
        elif goal_difference < 0:
            # Away in vantaggio
            prob_home_base -= 0.05 * (minute / 90)
            prob_draw_base -= 0.10 * (minute / 90)
            prob_away_base += 0.15 * (minute / 90)
        
        # Aggiustamento per momentum
        prob_home_base += momentum * 0.10
        prob_away_base -= momentum * 0.10
        
        # Normalizza
        total = prob_home_base + prob_draw_base + prob_away_base
        prob_home = max(0.05, min(0.95, prob_home_base / total))
        prob_draw = max(0.05, min(0.95, prob_draw_base / total))
        prob_away = max(0.05, min(0.95, prob_away_base / total))
        
        # Ricalcola total per normalizzazione finale
        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total
        
        # Probabilità Over/Under basate su gol attuali e minuto
        total_goals = score_home + score_away
        minutes_remaining = max(1, 90 - minute)
        
        # Stima gol attesi rimanenti (basata su media gol/minuto)
        avg_goals_per_minute = total_goals / max(1, minute) if minute > 0 else 0.02
        expected_remaining_goals = avg_goals_per_minute * minutes_remaining
        
        # Probabilità Over 2.5
        goals_needed_for_over_2_5 = max(0, 3 - total_goals)
        if goals_needed_for_over_2_5 == 0:
            prob_over_2_5 = 1.0
        elif goals_needed_for_over_2_5 > expected_remaining_goals * 2:
            prob_over_2_5 = 0.1
        else:
            # Stima probabilistica
            prob_over_2_5 = min(0.9, 0.3 + (expected_remaining_goals / goals_needed_for_over_2_5) * 0.4)
        
        prob_under_2_5 = 1.0 - prob_over_2_5
        
        return {
            'home_win': prob_home,
            'draw': prob_draw,
            'away_win': prob_away,
            'over_2_5': prob_over_2_5,
            'under_2_5': prob_under_2_5,
            'btts': 0.5,  # Default, può essere migliorato
            'home_or_draw': prob_home + prob_draw,
            'away_or_draw': prob_away + prob_draw
        }
    
    def _identify_opportunities(
        self,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any],
        probabilities: Dict[str, float],
        odds_data: Optional[Dict[str, Any]] = None,
        situation: Optional[Dict[str, Any]] = None,
        patterns: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identifica opportunità di valore basate su probabilità vs quote.
        
        Returns:
            Lista di opportunità trovate con reasoning dettagliato
        """
        opportunities = []
        
        if not odds_data:
            # Se non ci sono quote, usa quelle da match_data
            odds_data = {
                'home': match_data.get('odds_1', 2.0),
                'draw': match_data.get('odds_x', 3.0),
                'away': match_data.get('odds_2', 3.0)
            }
        
        # Opportunità 1X2
        if 'home' in odds_data and odds_data['home'] is not None and odds_data['home'] > 1.0:
            prob = probabilities['home_win']
            odds = odds_data['home']
            ev = (prob * odds - 1) * 100
            if ev > 0:
                reasoning = self._generate_reasoning(
                    market='1X2_HOME',
                    match_data=match_data,
                    live_data=live_data,
                    situation=situation,
                    patterns=patterns,
                    probability=prob,
                    odds=odds
                )
                opportunities.append({
                    'market': '1X2_HOME',
                    'probability': prob,
                    'odds': odds,
                    'ev': ev,
                    'recommendation': 'BET' if ev > self.min_ev else 'WATCH',
                    'reasoning': reasoning
                })
        
        if 'draw' in odds_data and odds_data['draw'] is not None and odds_data['draw'] > 1.0:
            prob = probabilities['draw']
            odds = odds_data['draw']
            ev = (prob * odds - 1) * 100
            if ev > 0:
                reasoning = self._generate_reasoning(
                    market='1X2_DRAW',
                    match_data=match_data,
                    live_data=live_data,
                    situation=situation,
                    patterns=patterns,
                    probability=prob,
                    odds=odds
                )
                opportunities.append({
                    'market': '1X2_DRAW',
                    'probability': prob,
                    'odds': odds,
                    'ev': ev,
                    'recommendation': 'BET' if ev > self.min_ev else 'WATCH',
                    'reasoning': reasoning
                })
        
        if 'away' in odds_data and odds_data['away'] is not None and odds_data['away'] > 1.0:
            prob = probabilities['away_win']
            odds = odds_data['away']
            ev = (prob * odds - 1) * 100
            if ev > 0:
                reasoning = self._generate_reasoning(
                    market='1X2_AWAY',
                    match_data=match_data,
                    live_data=live_data,
                    situation=situation,
                    patterns=patterns,
                    probability=prob,
                    odds=odds
                )
                opportunities.append({
                    'market': '1X2_AWAY',
                    'probability': prob,
                    'odds': odds,
                    'ev': ev,
                    'recommendation': 'BET' if ev > self.min_ev else 'WATCH',
                    'reasoning': reasoning
                })
        
        # Opportunità Over/Under (se disponibili)
        if 'over_2_5' in odds_data and odds_data['over_2_5'] is not None and odds_data['over_2_5'] > 1.0:
            prob = probabilities['over_2_5']
            odds = odds_data['over_2_5']
            ev = (prob * odds - 1) * 100
            if ev > 0:
                reasoning = self._generate_reasoning(
                    market='OVER_2_5',
                    match_data=match_data,
                    live_data=live_data,
                    situation=situation,
                    patterns=patterns,
                    probability=prob,
                    odds=odds
                )
                opportunities.append({
                    'market': 'OVER_2_5',
                    'probability': prob,
                    'odds': odds,
                    'ev': ev,
                    'recommendation': 'BET' if ev > self.min_ev else 'WATCH',
                    'reasoning': reasoning
                })
        
        if 'under_2_5' in odds_data and odds_data['under_2_5'] is not None and odds_data['under_2_5'] > 1.0:
            prob = probabilities['under_2_5']
            odds = odds_data['under_2_5']
            ev = (prob * odds - 1) * 100
            if ev > 0:
                reasoning = self._generate_reasoning(
                    market='UNDER_2_5',
                    match_data=match_data,
                    live_data=live_data,
                    situation=situation,
                    patterns=patterns,
                    probability=prob,
                    odds=odds
                )
                opportunities.append({
                    'market': 'UNDER_2_5',
                    'probability': prob,
                    'odds': odds,
                    'ev': ev,
                    'recommendation': 'BET' if ev > self.min_ev else 'WATCH',
                    'reasoning': reasoning
                })
        
        return opportunities
    
    def _enhance_with_confidence_ev(
        self,
        opportunities: List[Dict[str, Any]],
        match_data: Dict[str, Any],
        live_data: Dict[str, Any],
        probabilities: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Aggiunge confidence e migliora calcolo EV per ogni opportunità.
        """
        enhanced = []
        
        for opp in opportunities:
            market = opp['market']
            prob = opp['probability']
            odds = opp['odds']
            
            # Calcola confidence basata su:
            # 1. Quanto la probabilità è lontana da 0.5 (più estremo = più confidence)
            # 2. Minuto partita (più avanti = più confidence)
            # 3. Qualità dati live disponibili

            minute = live_data.get('minute') if live_data.get('minute') is not None else 0
            data_quality = self._assess_data_quality(live_data)

            # Confidence base
            prob_extremity = abs(prob - 0.5) * 2  # 0 (50%) a 1 (0% o 100%)
            minute_factor = min(1.0, minute / 90)  # Più avanti = più confidence
            
            confidence = 50 + (prob_extremity * 30) + (minute_factor * 15) + (data_quality * 5)
            confidence = max(50, min(95, confidence))
            
            # Ricalcola EV con confidence
            ev = (prob * odds - 1) * 100
            ev_adjusted = ev * (confidence / 100)  # EV aggiustato per confidence
            
            opp['confidence'] = confidence
            opp['ev'] = ev
            opp['ev_adjusted'] = ev_adjusted
            opp['data_quality'] = data_quality
            
            enhanced.append(opp)
        
        return enhanced
    
    def _generate_reasoning(
        self,
        market: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any],
        situation: Optional[Dict[str, Any]],
        patterns: Optional[Dict[str, Any]],
        probability: float,
        odds: float
    ) -> str:
        """
        Genera reasoning dettagliato per un mercato basato su statistiche live.
        
        Args:
            market: Nome del mercato
            match_data: Dati partita
            live_data: Dati live
            situation: Analisi situazione
            patterns: Pattern rilevati
            probability: Probabilità calcolata
            odds: Quote
            
        Returns:
            Stringa con reasoning dettagliato
        """
        home_team = match_data.get('home', 'Home')
        away_team = match_data.get('away', 'Away')

        # Estrai dati base con gestione sicura di None
        score_home = live_data.get('score_home') if live_data.get('score_home') is not None else 0
        score_away = live_data.get('score_away') if live_data.get('score_away') is not None else 0
        minute = live_data.get('minute') if live_data.get('minute') is not None else 0
        total_goals = score_home + score_away

        # Statistiche con gestione sicura di None
        possession_home = live_data.get('possession_home') if live_data.get('possession_home') is not None else 50
        possession_away = 100 - possession_home
        shots_home = live_data.get('shots_home') if live_data.get('shots_home') is not None else 0
        shots_away = live_data.get('shots_away') if live_data.get('shots_away') is not None else 0
        shots_on_target_home = live_data.get('shots_on_target_home') if live_data.get('shots_on_target_home') is not None else 0
        shots_on_target_away = live_data.get('shots_on_target_away') if live_data.get('shots_on_target_away') is not None else 0
        corners_home = live_data.get('corners_home') if live_data.get('corners_home') is not None else 0
        corners_away = live_data.get('corners_away') if live_data.get('corners_away') is not None else 0
        
        # Analisi situazione se disponibile
        goal_difference = score_home - score_away
        momentum = situation.get('momentum_score', 0) if situation else 0
        is_balanced = situation.get('is_balanced', False) if situation else False
        is_critical = situation.get('is_critical', False) if situation else False
        
        # Pattern se disponibili
        comeback_possible = patterns.get('comeback_possible', False) if patterns else False
        defensive_mode = patterns.get('defensive_mode', False) if patterns else False
        attacking_mode = patterns.get('attacking_mode', False) if patterns else False
        
        reasoning_parts = []
        
        # Reasoning specifico per mercato
        if market == '1X2_HOME':
            reasoning_parts.append(f"🎯 ANALISI VITTORIA {home_team}:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            if goal_difference > 0:
                reasoning_parts.append(f"• ✅ {home_team} è in vantaggio di {goal_difference} gol")
            elif goal_difference < 0:
                reasoning_parts.append(f"• ⚠️ {home_team} è in svantaggio di {abs(goal_difference)} gol")
            else:
                reasoning_parts.append(f"• ⚖️ Partita in pareggio")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Possesso: {possession_home}% vs {possession_away}%")
            reasoning_parts.append(f"• Tiri totali: {shots_home} vs {shots_away}")
            reasoning_parts.append(f"• Tiri in porta: {shots_on_target_home} vs {shots_on_target_away}")
            reasoning_parts.append(f"• Corner: {corners_home} vs {corners_away}")
            
            if momentum > 0.2:
                reasoning_parts.append(f"• ✅ Momentum favorevole a {home_team} (dominio campo)")
            elif momentum < -0.2:
                reasoning_parts.append(f"• ⚠️ Momentum sfavorevole a {home_team}")
            else:
                reasoning_parts.append(f"• ⚖️ Partita equilibrata")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
            
            if comeback_possible and goal_difference < 0:
                reasoning_parts.append(f"• 🔄 Pattern rilevato: Ribaltone possibile")
            if defensive_mode and goal_difference > 0:
                reasoning_parts.append(f"• 🛡️ Pattern rilevato: Modalità difensiva (mantenere vantaggio)")
            if is_critical:
                reasoning_parts.append(f"• ⚠️ Situazione critica: Partita in bilico")
        
        elif market == '1X2_AWAY':
            reasoning_parts.append(f"🎯 ANALISI VITTORIA {away_team}:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            if goal_difference < 0:
                reasoning_parts.append(f"• ✅ {away_team} è in vantaggio di {abs(goal_difference)} gol")
            elif goal_difference > 0:
                reasoning_parts.append(f"• ⚠️ {away_team} è in svantaggio di {goal_difference} gol")
            else:
                reasoning_parts.append(f"• ⚖️ Partita in pareggio")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Possesso: {possession_away}% vs {possession_home}%")
            reasoning_parts.append(f"• Tiri totali: {shots_away} vs {shots_home}")
            reasoning_parts.append(f"• Tiri in porta: {shots_on_target_away} vs {shots_on_target_home}")
            reasoning_parts.append(f"• Corner: {corners_away} vs {corners_home}")
            
            if momentum < -0.2:
                reasoning_parts.append(f"• ✅ Momentum favorevole a {away_team} (dominio campo)")
            elif momentum > 0.2:
                reasoning_parts.append(f"• ⚠️ Momentum sfavorevole a {away_team}")
            else:
                reasoning_parts.append(f"• ⚖️ Partita equilibrata")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
            
            if comeback_possible and goal_difference > 0:
                reasoning_parts.append(f"• 🔄 Pattern rilevato: Ribaltone possibile")
            if attacking_mode and goal_difference < 0:
                reasoning_parts.append(f"• ⚡ Pattern rilevato: Modalità attacco (recupero)")
        
        elif market == '1X2_DRAW':
            reasoning_parts.append(f"🎯 ANALISI PAREGGIO:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            if goal_difference == 0:
                reasoning_parts.append(f"• ✅ Partita in pareggio")
            else:
                reasoning_parts.append(f"• ⚠️ Partita non in pareggio (differenza: {abs(goal_difference)} gol)")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Possesso: {possession_home}% vs {possession_away}%")
            reasoning_parts.append(f"• Tiri totali: {shots_home} vs {shots_away}")
            reasoning_parts.append(f"• Tiri in porta: {shots_on_target_home} vs {shots_on_target_away}")
            
            if is_balanced:
                reasoning_parts.append(f"• ✅ Partita equilibrata (statistiche simili)")
            else:
                reasoning_parts.append(f"• ⚠️ Partita non equilibrata (una squadra domina)")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif market == 'OVER_2_5':
            reasoning_parts.append(f"🎯 ANALISI OVER 2.5 GOL:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Gol segnati: {total_goals} al {minute}'")
            goals_needed = max(0, 3 - total_goals)
            minutes_remaining = 90 - minute
            reasoning_parts.append(f"• Gol necessari: {goals_needed} nei prossimi {minutes_remaining} minuti")
            
            if total_goals >= 3:
                reasoning_parts.append(f"• ✅ Over 2.5 già raggiunto!")
            elif total_goals == 2:
                reasoning_parts.append(f"• ⚡ Serve 1 gol nei prossimi {minutes_remaining} minuti")
            elif total_goals == 1:
                reasoning_parts.append(f"• ⚠️ Servono 2 gol nei prossimi {minutes_remaining} minuti")
            else:
                reasoning_parts.append(f"• ❌ Servono 3 gol nei prossimi {minutes_remaining} minuti")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Tiri in porta: {shots_on_target_home + shots_on_target_away} totali")
            reasoning_parts.append(f"• Tiri totali: {shots_home + shots_away}")
            reasoning_parts.append(f"• Media gol/minuto: {total_goals/max(1, minute):.2f}")
            
            avg_goals_per_min = total_goals / max(1, minute) if minute > 0 else 0
            expected_remaining = avg_goals_per_min * minutes_remaining
            reasoning_parts.append(f"• Gol attesi rimanenti: ~{expected_remaining:.1f}")
            
            if shots_on_target_home + shots_on_target_away > 8:
                reasoning_parts.append(f"• ✅ Alta pressione offensiva (molti tiri in porta)")
            if attacking_mode:
                reasoning_parts.append(f"• ⚡ Modalità attacco attiva (entrambe le squadre cercano gol)")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif market == 'UNDER_2_5':
            reasoning_parts.append(f"🎯 ANALISI UNDER 2.5 GOL:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Gol segnati: {total_goals} al {minute}'")
            goals_allowed = 3 - total_goals
            minutes_remaining = 90 - minute
            reasoning_parts.append(f"• Gol massimi consentiti: {goals_allowed} nei prossimi {minutes_remaining} minuti")
            
            if total_goals == 0:
                reasoning_parts.append(f"• ✅ Partita molto chiusa (0 gol)")
            elif total_goals == 1:
                reasoning_parts.append(f"• ✅ Partita chiusa (1 gol, max 1 altro consentito)")
            elif total_goals == 2:
                reasoning_parts.append(f"• ⚠️ Partita aperta (2 gol, nessun altro gol consentito)")
            else:
                reasoning_parts.append(f"• ❌ Under 2.5 già fallito (3+ gol)")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Tiri in porta: {shots_on_target_home + shots_on_target_away} totali")
            reasoning_parts.append(f"• Tiri totali: {shots_home + shots_away}")
            
            if defensive_mode:
                reasoning_parts.append(f"• 🛡️ Modalità difensiva attiva (squadra in vantaggio si chiude)")
            if shots_on_target_home + shots_on_target_away < 4:
                reasoning_parts.append(f"• ✅ Bassa pressione offensiva (pochi tiri in porta)")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        # Mercati Over/Under aggiuntivi
        elif market.startswith('OVER_') or market.startswith('over_'):
            goal_line = self._extract_goal_line_from_market(market)
            is_ht = 'HT' in market.upper() or '_HT' in market.upper()
            
            if is_ht:
                reasoning_parts.append(f"🎯 ANALISI OVER {goal_line} PRIMO TEMPO:")
                reasoning_parts.append(f"")
                reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
                reasoning_parts.append(f"• Gol primo tempo: {total_goals if minute <= 45 else 'N/A'} al {min(minute, 45)}'")
                if minute > 45:
                    reasoning_parts.append(f"• ⚠️ Primo tempo già terminato")
                else:
                    goals_needed = max(0, goal_line - total_goals)
                    minutes_remaining_ht = 45 - minute
                    reasoning_parts.append(f"• Gol necessari: {goals_needed} nei prossimi {minutes_remaining_ht} minuti")
            else:
                reasoning_parts.append(f"🎯 ANALISI OVER {goal_line} GOL:")
                reasoning_parts.append(f"")
                reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
                reasoning_parts.append(f"• Gol segnati: {total_goals} al {minute}'")
                goals_needed = max(0, goal_line - total_goals)
                minutes_remaining = 90 - minute
                reasoning_parts.append(f"• Gol necessari: {goals_needed} nei prossimi {minutes_remaining} minuti")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Tiri in porta: {shots_on_target_home + shots_on_target_away} totali")
            reasoning_parts.append(f"• Tiri totali: {shots_home + shots_away}")
            
            if not is_ht:
                avg_goals_per_min = total_goals / max(1, minute) if minute > 0 else 0
                expected_remaining = avg_goals_per_min * (90 - minute)
                reasoning_parts.append(f"• Media gol/minuto: {avg_goals_per_min:.2f}")
                reasoning_parts.append(f"• Gol attesi rimanenti: ~{expected_remaining:.1f}")
            
            if shots_on_target_home + shots_on_target_away > 6:
                reasoning_parts.append(f"• ✅ Alta pressione offensiva")
            if attacking_mode:
                reasoning_parts.append(f"• ⚡ Modalità attacco attiva")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif market.startswith('UNDER_') or market.startswith('under_'):
            goal_line = self._extract_goal_line_from_market(market)
            is_ht = 'HT' in market.upper() or '_HT' in market.upper()
            
            if is_ht:
                reasoning_parts.append(f"🎯 ANALISI UNDER {goal_line} PRIMO TEMPO:")
                reasoning_parts.append(f"")
                reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
                if minute > 45:
                    reasoning_parts.append(f"• ⚠️ Primo tempo già terminato")
                else:
                    reasoning_parts.append(f"• Gol primo tempo: {total_goals} al {minute}'")
                    goals_allowed = goal_line - total_goals
                    minutes_remaining_ht = 45 - minute
                    reasoning_parts.append(f"• Gol massimi consentiti: {goals_allowed} nei prossimi {minutes_remaining_ht} minuti")
            else:
                reasoning_parts.append(f"🎯 ANALISI UNDER {goal_line} GOL:")
                reasoning_parts.append(f"")
                reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
                reasoning_parts.append(f"• Gol segnati: {total_goals} al {minute}'")
                goals_allowed = goal_line - total_goals
                minutes_remaining = 90 - minute
                reasoning_parts.append(f"• Gol massimi consentiti: {goals_allowed} nei prossimi {minutes_remaining} minuti")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Tiri in porta: {shots_on_target_home + shots_on_target_away} totali")
            reasoning_parts.append(f"• Tiri totali: {shots_home + shots_away}")
            
            if defensive_mode:
                reasoning_parts.append(f"• 🛡️ Modalità difensiva attiva")
            if shots_on_target_home + shots_on_target_away < 4:
                reasoning_parts.append(f"• ✅ Bassa pressione offensiva")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif market in ['1X', '1x', 'HOME_OR_DRAW']:
            reasoning_parts.append(f"🎯 ANALISI DOUBLE CHANCE 1X ({home_team} pareggio o vittoria):")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            if goal_difference >= 0:
                reasoning_parts.append(f"• ✅ {home_team} non perde (vantaggio o pareggio)")
            else:
                reasoning_parts.append(f"• ⚠️ {home_team} in svantaggio di {abs(goal_difference)} gol")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Possesso: {possession_home}% vs {possession_away}%")
            reasoning_parts.append(f"• Tiri: {shots_home} vs {shots_away}")
            reasoning_parts.append(f"• Tiri in porta: {shots_on_target_home} vs {shots_on_target_away}")
            
            if momentum > 0:
                reasoning_parts.append(f"• ✅ Momentum favorevole a {home_team}")
            if comeback_possible and goal_difference < 0:
                reasoning_parts.append(f"• 🔄 Ribaltone possibile")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif market in ['X2', 'x2', 'AWAY_OR_DRAW']:
            reasoning_parts.append(f"🎯 ANALISI DOUBLE CHANCE X2 ({away_team} pareggio o vittoria):")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            if goal_difference <= 0:
                reasoning_parts.append(f"• ✅ {away_team} non perde (vantaggio o pareggio)")
            else:
                reasoning_parts.append(f"• ⚠️ {away_team} in svantaggio di {goal_difference} gol")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Possesso: {possession_away}% vs {possession_home}%")
            reasoning_parts.append(f"• Tiri: {shots_away} vs {shots_home}")
            reasoning_parts.append(f"• Tiri in porta: {shots_on_target_away} vs {shots_on_target_home}")
            
            if momentum < 0:
                reasoning_parts.append(f"• ✅ Momentum favorevole a {away_team}")
            if comeback_possible and goal_difference > 0:
                reasoning_parts.append(f"• 🔄 Ribaltone possibile")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif market in ['DNB_HOME', 'dnb_home', 'DRAW_NO_BET_HOME']:
            reasoning_parts.append(f"🎯 ANALISI DRAW NO BET {home_team}:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            if goal_difference > 0:
                reasoning_parts.append(f"• ✅ {home_team} in vantaggio (ritorno stake se pareggio)")
            elif goal_difference == 0:
                reasoning_parts.append(f"• ⚖️ Pareggio (ritorno stake)")
            else:
                reasoning_parts.append(f"• ⚠️ {home_team} in svantaggio (serve ribaltone)")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Possesso: {possession_home}% vs {possession_away}%")
            reasoning_parts.append(f"• Tiri: {shots_home} vs {shots_away}")
            if momentum > 0.2:
                reasoning_parts.append(f"• ✅ Momentum favorevole a {home_team}")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif market in ['DNB_AWAY', 'dnb_away', 'DRAW_NO_BET_AWAY']:
            reasoning_parts.append(f"🎯 ANALISI DRAW NO BET {away_team}:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            if goal_difference < 0:
                reasoning_parts.append(f"• ✅ {away_team} in vantaggio (ritorno stake se pareggio)")
            elif goal_difference == 0:
                reasoning_parts.append(f"• ⚖️ Pareggio (ritorno stake)")
            else:
                reasoning_parts.append(f"• ⚠️ {away_team} in svantaggio (serve ribaltone)")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Possesso: {possession_away}% vs {possession_home}%")
            reasoning_parts.append(f"• Tiri: {shots_away} vs {shots_home}")
            if momentum < -0.2:
                reasoning_parts.append(f"• ✅ Momentum favorevole a {away_team}")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif market in ['BTTS_YES', 'btts_yes', 'BOTH_TEAMS_TO_SCORE']:
            reasoning_parts.append(f"🎯 ANALISI BTTS (Both Teams To Score) - SÌ:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            if score_home > 0 and score_away > 0:
                reasoning_parts.append(f"• ✅ BTTS già verificato! (entrambe hanno segnato)")
            elif score_home > 0:
                reasoning_parts.append(f"• ⚡ {home_team} ha segnato, serve gol a {away_team}")
            elif score_away > 0:
                reasoning_parts.append(f"• ⚡ {away_team} ha segnato, serve gol a {home_team}")
            else:
                reasoning_parts.append(f"• ❌ Nessuna squadra ha segnato (servono gol a entrambe)")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Tiri in porta {home_team}: {shots_on_target_home}")
            reasoning_parts.append(f"• Tiri in porta {away_team}: {shots_on_target_away}")
            if shots_on_target_home > 3 and shots_on_target_away > 3:
                reasoning_parts.append(f"• ✅ Alta pressione offensiva da entrambe le squadre")
            if attacking_mode:
                reasoning_parts.append(f"• ⚡ Entrambe le squadre in attacco")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif market in ['BTTS_NO', 'btts_no']:
            reasoning_parts.append(f"🎯 ANALISI BTTS (Both Teams To Score) - NO:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            if score_home == 0 or score_away == 0:
                reasoning_parts.append(f"• ✅ BTTS NO già verificato! (almeno una squadra non ha segnato)")
            else:
                reasoning_parts.append(f"• ⚠️ Entrambe hanno segnato (BTTS NO fallito)")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            if defensive_mode:
                reasoning_parts.append(f"• 🛡️ Modalità difensiva (squadra in vantaggio si chiude)")
            if shots_on_target_home < 2 or shots_on_target_away < 2:
                reasoning_parts.append(f"• ✅ Bassa pressione offensiva")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif 'WIN_TO_NIL' in market.upper() or 'CLEAN_SHEET' in market.upper():
            team_name = home_team if 'HOME' in market.upper() else away_team
            reasoning_parts.append(f"🎯 ANALISI WIN TO NIL {team_name}:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            if 'HOME' in market.upper():
                if score_home > 0 and score_away == 0:
                    reasoning_parts.append(f"• ✅ Condizione già verificata! ({home_team} vince senza subire)")
                elif score_away > 0:
                    reasoning_parts.append(f"• ❌ {home_team} ha già subito gol")
                else:
                    reasoning_parts.append(f"• ⚡ {home_team} non ha subito, serve vittoria")
            else:
                if score_away > 0 and score_home == 0:
                    reasoning_parts.append(f"• ✅ Condizione già verificata! ({away_team} vince senza subire)")
                elif score_home > 0:
                    reasoning_parts.append(f"• ❌ {away_team} ha già subito gol")
                else:
                    reasoning_parts.append(f"• ⚡ {away_team} non ha subito, serve vittoria")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            if defensive_mode:
                reasoning_parts.append(f"• 🛡️ Modalità difensiva attiva")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif 'CORNER' in market.upper():
            corner_line = self._extract_number_from_market(market)
            total_corners = corners_home + corners_away
            reasoning_parts.append(f"🎯 ANALISI CORNER:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Corner totali: {total_corners} al {minute}'")
            if 'OVER' in market.upper():
                corners_needed = max(0, corner_line - total_corners)
                reasoning_parts.append(f"• Corner necessari: {corners_needed} nei prossimi {90-minute} minuti")
            else:
                corners_allowed = corner_line - total_corners
                reasoning_parts.append(f"• Corner massimi consentiti: {corners_allowed}")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Corner {home_team}: {corners_home}")
            reasoning_parts.append(f"• Corner {away_team}: {corners_away}")
            reasoning_parts.append(f"• Media corner/minuto: {total_corners/max(1, minute):.2f}")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif 'CARD' in market.upper() or 'CARTELLIN' in market.upper():
            yellow_home = live_data.get('yellow_cards_home') if live_data.get('yellow_cards_home') is not None else 0
            yellow_away = live_data.get('yellow_cards_away') if live_data.get('yellow_cards_away') is not None else 0
            red_home = live_data.get('red_cards_home') if live_data.get('red_cards_home') is not None else 0
            red_away = live_data.get('red_cards_away') if live_data.get('red_cards_away') is not None else 0
            total_cards = yellow_home + yellow_away + red_home + red_away
            reasoning_parts.append(f"🎯 ANALISI CARTELLINI:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Cartellini totali: {total_cards} al {minute}'")
            reasoning_parts.append(f"• Gialli: {yellow_home + yellow_away} (Casa: {yellow_home}, Trasferta: {yellow_away})")
            reasoning_parts.append(f"• Rossi: {red_home + red_away} (Casa: {red_home}, Trasferta: {red_away})")
            
            if 'OVER' in market.upper():
                card_line = self._extract_number_from_market(market)
                cards_needed = max(0, card_line - total_cards)
                reasoning_parts.append(f"• Cartellini necessari: {cards_needed} nei prossimi {90-minute} minuti")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif 'NEXT_GOAL' in market.upper() or 'PROSSIMO_GOL' in market.upper():
            team_name = home_team if 'HOME' in market.upper() else away_team
            reasoning_parts.append(f"🎯 ANALISI PROSSIMO GOL {team_name}:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            if 'HOME' in market.upper():
                reasoning_parts.append(f"• Tiri in porta {home_team}: {shots_on_target_home}")
                reasoning_parts.append(f"• Possesso {home_team}: {possession_home}%")
                if momentum > 0.2:
                    reasoning_parts.append(f"• ✅ Momentum favorevole a {home_team}")
            else:
                reasoning_parts.append(f"• Tiri in porta {away_team}: {shots_on_target_away}")
                reasoning_parts.append(f"• Possesso {away_team}: {possession_away}%")
                if momentum < -0.2:
                    reasoning_parts.append(f"• ✅ Momentum favorevole a {away_team}")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif 'HANDICAP' in market.upper():
            reasoning_parts.append(f"🎯 ANALISI HANDICAP:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            reasoning_parts.append(f"• Handicap: {market}")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Possesso: {possession_home}% vs {possession_away}%")
            reasoning_parts.append(f"• Tiri: {shots_home} vs {shots_away}")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        elif 'ODD' in market.upper() or 'EVEN' in market.upper() or 'PARI' in market.upper() or 'DISPARI' in market.upper():
            is_odd = total_goals % 2 == 1
            reasoning_parts.append(f"🎯 ANALISI TOTAL GOALS {'DISPARI' if 'ODD' in market.upper() or 'DISPARI' in market.upper() else 'PARI'}:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Gol totali: {total_goals} al {minute}'")
            if 'ODD' in market.upper() or 'DISPARI' in market.upper():
                if is_odd:
                    reasoning_parts.append(f"• ✅ Totale già dispari!")
                else:
                    reasoning_parts.append(f"• ⚡ Totale pari, serve 1 gol per renderlo dispari")
            else:
                if not is_odd:
                    reasoning_parts.append(f"• ✅ Totale già pari!")
                else:
                    reasoning_parts.append(f"• ⚡ Totale dispari, serve 1 gol per renderlo pari")
            
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        else:
            # Reasoning generico per mercati non specificati
            reasoning_parts.append(f"🎯 ANALISI {market}:")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📊 SITUAZIONE ATTUALE:")
            reasoning_parts.append(f"• Score: {score_home}-{score_away} al {minute}'")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"📈 STATISTICHE LIVE:")
            reasoning_parts.append(f"• Possesso: {possession_home}% vs {possession_away}%")
            reasoning_parts.append(f"• Tiri: {shots_home} vs {shots_away}")
            reasoning_parts.append(f"")
            reasoning_parts.append(f"🧠 VALUTAZIONE AI:")
            reasoning_parts.append(f"• Probabilità reale: {probability*100:.1f}%")
            reasoning_parts.append(f"• Quote offerta: {odds:.2f} (probabilità implicita: {1/odds*100:.1f}%)")
            reasoning_parts.append(f"• Valore: {'✅ ALTO' if probability > 1/odds else '⚠️ BASSO'}")
        
        return "\n".join(reasoning_parts)
    
    def _extract_goal_line_from_market(self, market: str) -> float:
        """Estrae la goal line da un mercato (es. 'OVER_2_5' -> 2.5)"""
        import re
        match = re.search(r'(\d+\.?\d*)', market)
        if match:
            return float(match.group(1))
        return 2.5  # Default
    
    def _extract_number_from_market(self, market: str) -> float:
        """Estrae un numero da un mercato (es. 'OVER_8_5_CORNERS' -> 8.5)"""
        import re
        match = re.search(r'(\d+\.?\d*)', market)
        if match:
            return float(match.group(1))
        return 0.0
    
    def _assess_data_quality(self, live_data: Dict[str, Any]) -> float:
        """
        Valuta qualità dati live disponibili (0.0 = bassa, 1.0 = alta).
        """
        quality = 0.0
        
        # Score disponibile
        if 'score_home' in live_data and 'score_away' in live_data:
            quality += 0.3
        
        # Minuto disponibile
        if 'minute' in live_data:
            quality += 0.2
        
        # Statistiche disponibili
        if 'shots_home' in live_data and 'shots_away' in live_data:
            quality += 0.2
        
        if 'possession_home' in live_data:
            quality += 0.2
        
        if 'events' in live_data and len(live_data.get('events', [])) > 0:
            quality += 0.1
        
        return min(1.0, quality)
    
    def _cleanup_cache(self, current_time: float):
        """Rimuove entry cache vecchie (> 5 minuti)."""
        max_age = 300  # 5 minuti
        keys_to_remove = [
            key for key, (_, cache_time) in self._analysis_cache.items()
            if current_time - cache_time > max_age
        ]
        for key in keys_to_remove:
            del self._analysis_cache[key]
        
        if keys_to_remove:
            logger.debug(f"🧹 Cleaned up {len(keys_to_remove)} old cache entries")

