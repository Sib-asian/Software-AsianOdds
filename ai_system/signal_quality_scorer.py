"""
AI Signal Quality Scorer + Context-Aware Validator
===================================================

Sistema combinato che valuta la qualità di ogni segnale prima dell'invio:
1. Context-Aware Validator: Analizza contesto partita
2. AI Quality Scorer: Valuta qualità segnale
3. Consensus Check: Verifica coerenza
4. Final Decision: Invia solo se tutti approvano

Questo sistema riduce drasticamente falsi positivi e migliora precisione.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Punteggio qualità segnale"""
    total_score: float  # 0-100
    context_score: float  # 0-100
    data_quality_score: float  # 0-100
    logic_score: float  # 0-100
    timing_score: float  # 0-100
    is_approved: bool
    reasons: List[str]
    warnings: List[str]
    errors: List[str]


class ContextAwareValidator:
    """Valida se il segnale ha senso nel contesto della partita"""
    
    def __init__(self):
        self.min_minute_for_analysis = 10  # Minimo 10 minuti per analisi seria
        self.max_minute_for_opportunity = 85  # Dopo 85' troppo rischioso
    
    def validate_context(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Valida contesto partita.
        
        Returns:
            (score 0-100, reasons, warnings)
        """
        score = 100.0
        reasons = []
        warnings = []
        
        minute = live_data.get('minute', 0)
        score_home = live_data.get('score_home', 0)
        score_away = live_data.get('score_away', 0)
        total_goals = score_home + score_away
        market = getattr(opportunity.get('live_opportunity'), 'market', 'unknown')
        
        # 1. Validazione timing
        if minute < self.min_minute_for_analysis:
            score -= 30
            warnings.append(f"Partita troppo presto ({minute}'), pochi dati disponibili")
        elif minute > self.max_minute_for_opportunity:
            score -= 25
            warnings.append(f"Partita troppo avanzata ({minute}'), poco tempo rimanente")
        
        # 2. Validazione coerenza score-mercato
        if 'over_1.5' in market and total_goals >= 2:
            score -= 50
            reasons.append(f"Over 1.5 già superato (score: {total_goals})")
        elif 'over_2.5' in market and total_goals >= 3:
            score -= 50
            reasons.append(f"Over 2.5 già superato (score: {total_goals})")
        elif 'over_3.5' in market and total_goals >= 4:
            score -= 50
            reasons.append(f"Over 3.5 già superato (score: {total_goals})")
        elif 'under_1.5' in market and total_goals >= 2:
            score -= 50
            reasons.append(f"Under 1.5 già perso (score: {total_goals})")
        elif 'under_2.5' in market and total_goals >= 3:
            score -= 50
            reasons.append(f"Under 2.5 già perso (score: {total_goals})")
        
        # 🚫 FIX: Under 2.5 troppo presto dopo gol (rischioso)
        if 'under_2.5' in market and total_goals == 1 and minute < 30:
            score -= 40
            reasons.append(f"Under 2.5 troppo presto (minuto {minute}') dopo 1 gol - troppo rischioso")
        
        # 🚫 FILTRO CRITICO 1: Over 0.5 quando c'è già 1+ gol (BANALE!)
        if 'over_0.5' in market and total_goals >= 1:
            score -= 50
            reasons.append(f"Over 0.5 già superato (score: {total_goals})")
        
        # 🚫 FILTRO CRITICO 2: BTTS Yes quando entrambe hanno già segnato (BANALE!)
        if 'btts_yes' in market and score_home > 0 and score_away > 0:
            score -= 50
            reasons.append(f"BTTS Yes quando entrambe hanno già segnato ({score_home}-{score_away})")
        
        # 🚫 FILTRO CRITICO 3: BTTS Yes quando una squadra non ha segnato e siamo oltre 85' (ILLOGICO!)
        if 'btts_yes' in market and minute >= 85:
            if score_home == 0 or score_away == 0:
                score -= 40
                reasons.append(f"BTTS Yes quando è {score_home}-{score_away} al {minute}' - troppo tardi")
        
        # 🚫 FILTRO CRITICO 4: Clean Sheet quando risultato è 2-0+ al 75' (BANALE!)
        if 'clean_sheet' in market:
            goal_diff = abs(score_home - score_away)
            if goal_diff >= 2 and minute >= 75:
                score -= 40
                reasons.append(f"Clean Sheet quando risultato è {score_home}-{score_away} al {minute}' - troppo ovvio")
        
        # 🚫 FILTRO CRITICO 5: Exact Score quando suggerisce lo score attuale al 70'+ (BANALE!)
        if 'exact_score' in market and minute >= 70:
            # Estrai score dal market (es. "exact_score_2-0")
            score_match = re.search(r'exact_score_(\d+)-(\d+)', market)
            if score_match:
                market_score_home = int(score_match.group(1))
                market_score_away = int(score_match.group(2))
                if market_score_home == score_home and market_score_away == score_away:
                    score -= 50
                    reasons.append(f"Exact Score {market_score_home}-{market_score_away} quando è già {score_home}-{score_away} al {minute}' - banale")
        
        # 🚫 FILTRO CRITICO 6: Team to Score First quando NON è 0-0 (IMPOSSIBILE!)
        if 'team_to_score_first' in market or 'first_goal' in market:
            if score_home > 0 or score_away > 0:
                score -= 50
                reasons.append(f"Team to Score First quando è già {score_home}-{score_away} - impossibile")
        
        # 🚫 FILTRO CRITICO 7: Next Goal quando siamo oltre 85' (BANALE!)
        if 'next_goal' in market and minute >= 85:
            score -= 40
            reasons.append(f"Next Goal quando siamo al {minute}' - troppo tardi")
        
        # 🚫 FILTRO CRITICO 8: Goal Range 0-1 quando c'è già 1 gol al 60'+ (ILLOGICO!)
        if 'goal_range_0_1' in market:
            if total_goals == 1 and minute >= 60:
                score -= 40
                reasons.append(f"Goal Range 0-1 quando è già {score_home}-{score_away} (1 gol) al {minute}' - troppo rischioso")
            elif total_goals > 1:
                score -= 50
                reasons.append(f"Goal Range 0-1 quando ci sono già {total_goals} gol - impossibile")
        
        # 🚫 FILTRO CRITICO 9: Goal Range 2-3 quando ci sono già 4+ gol (ILLOGICO!)
        if 'goal_range_2_3' in market and total_goals >= 4:
            score -= 50
            reasons.append(f"Goal Range 2-3 quando ci sono già {total_goals} gol ({score_home}-{score_away}) - impossibile")
        
        # 🚫 FILTRO CRITICO 10: Win To Nil quando è 2-0+ al 75' (BANALE!)
        if 'win_to_nil' in market:
            goal_diff = abs(score_home - score_away)
            if goal_diff >= 2 and minute >= 75:
                score -= 40
                reasons.append(f"Win To Nil quando è {score_home}-{score_away} (diff: {goal_diff} gol) al {minute}' - troppo ovvio")
        
        # 🚫 FILTRO CRITICO 11: Odd/Even quando è troppo tardi (80'+) - BANALE!
        if 'total_goals_odd' in market or 'total_goals_even' in market:
            if minute >= 80:
                # Se è già dispari e suggerisce odd, è banale (è già dispari, poco tempo rimasto)
                if 'total_goals_odd' in market and total_goals % 2 == 1:
                    score -= 50
                    reasons.append(f"Total Goals Odd quando è già {score_home}-{score_away} (dispari) al {minute}' - banale, poco tempo rimasto")
                # Se è già pari e suggerisce even, è banale (è già pari, poco tempo rimasto)
                elif 'total_goals_even' in market and total_goals % 2 == 0:
                    score -= 50
                    reasons.append(f"Total Goals Even quando è già {score_home}-{score_away} (pari) al {minute}' - banale, poco tempo rimasto")
                # Se suggerisce odd quando è pari o viceversa oltre 80', è troppo rischioso
                elif minute >= 85:
                    score -= 40
                    reasons.append(f"Odd/Even al {minute}' - troppo tardi, poco tempo rimasto per cambiare")
        
        # 🚫 FILTRO CRITICO 12: Team Goal Anytime quando hanno già segnato (BANALE!)
        if 'home_goal_anytime' in market and score_home > 0:
            score -= 50
            reasons.append(f"Home Goal Anytime quando {score_home}-{score_away} - hanno già segnato, banale")
        if 'away_goal_anytime' in market and score_away > 0:
            score -= 50
            reasons.append(f"Away Goal Anytime quando {score_home}-{score_away} - hanno già segnato, banale")
        
        # 🚫 FILTRO CRITICO 13: Team to Score Last quando partita decisa o troppo tardi (BANALE!)
        if 'team_to_score_last' in market:
            goal_diff = abs(score_home - score_away)
            if goal_diff >= 3 and minute >= 70:
                score -= 40
                reasons.append(f"Team to Score Last quando partita decisa ({score_home}-{score_away}, diff: {goal_diff}) al {minute}' - banale")
            elif minute >= 88:
                score -= 50
                reasons.append(f"Team to Score Last al {minute}' - troppo tardi, impossibile")
        
        # 🚫 FILTRO CRITICO 14: Next Goal Before 75 quando siamo oltre 75' (IMPOSSIBILE!)
        if 'next_goal_before_75' in market and minute >= 75:
            score -= 50
            reasons.append(f"Next Goal Before 75 quando siamo al {minute}' - impossibile, 75' già passato")
        
        # 🚫 FILTRO CRITICO 15: Next Goal After 75 quando siamo oltre 85' (BANALE!)
        if 'next_goal_after_75' in market and minute >= 85:
            score -= 40
            reasons.append(f"Next Goal After 75 quando siamo al {minute}' - troppo tardi, poco tempo rimasto")
        
        # 🚫 FILTRO CRITICO 16: Half Time Result quando siamo oltre 45' (IMPOSSIBILE!)
        if 'half_time_result' in market and minute > 45:
            score -= 50
            reasons.append(f"Half Time Result quando siamo al {minute}' - impossibile, primo tempo già finito")
        
        # 🚫 FILTRO CRITICO 17: BTTS First Half quando siamo oltre 45' (IMPOSSIBILE!)
        if 'btts_first_half' in market and minute > 45:
            score -= 50
            reasons.append(f"BTTS First Half quando siamo al {minute}' - impossibile, primo tempo già finito")
        
        # 🚫 FILTRO CRITICO 18: Over/Under HT quando siamo oltre 45' (IMPOSSIBILE!)
        if ('over_' in market or 'under_' in market) and '_ht' in market and minute > 45:
            score -= 50
            reasons.append(f"Over/Under HT quando siamo al {minute}' - impossibile, primo tempo già finito")
        
        # 🚫 FILTRO CRITICO 19: Second Half Over quando siamo oltre 80' (BANALE!)
        if 'over_' in market and 'second_half' in market and minute >= 80:
            score -= 40
            reasons.append(f"Second Half Over quando siamo al {minute}' - troppo tardi, poco tempo rimasto")
        
        # 🚫 FILTRO CRITICO 20: Win Either Half quando partita decisa o troppo tardi (BANALE!)
        if 'win_either_half' in market:
            goal_diff = abs(score_home - score_away)
            if goal_diff >= 3 and minute >= 70:
                score -= 40
                reasons.append(f"Win Either Half quando partita decisa ({score_home}-{score_away}, diff: {goal_diff}) al {minute}' - banale")
            elif total_goals == 0 and minute > 60:
                score -= 30
                reasons.append(f"Win Either Half su 0-0 al {minute}' - troppo tardi")
        
        # 🚫 FILTRO CRITICO 21: Draw No Bet quando partita decisa (BANALE!)
        if 'dnb_' in market:
            goal_diff = abs(score_home - score_away)
            if goal_diff >= 3 and minute >= 75:
                score -= 40
                reasons.append(f"Draw No Bet quando partita decisa ({score_home}-{score_away}, diff: {goal_diff}) al {minute}' - banale")
        
        # 🚫 FILTRO CRITICO 22: Goal Range 4+ quando ci sono già 4+ gol e siamo oltre 70' (BANALE!)
        if 'goal_range_4_plus' in market and total_goals >= 4 and minute >= 70:
            score -= 40
            reasons.append(f"Goal Range 4+ quando ci sono già {total_goals} gol al {minute}' - banale")
        
        # 🚫 FILTRO CRITICO 23: Under 3.5 quando ci sono già 4+ gol (IMPOSSIBILE!)
        if 'under_3.5' in market and total_goals >= 4:
            score -= 50
            reasons.append(f"Under 3.5 quando ci sono già {total_goals} gol ({score_home}-{score_away}) - impossibile")
        
        # 🚫 FILTRO CRITICO 24: Over 4.5 quando ci sono già 5+ gol e siamo oltre 75' (BANALE!)
        if 'over_4.5' in market and total_goals >= 5 and minute >= 75:
            score -= 40
            reasons.append(f"Over 4.5 quando ci sono già {total_goals} gol al {minute}' - banale")
        
        # 3. Validazione situazione partita
        if '1x2_home' in market or 'home_win' in market:
            if score_home < score_away and (score_away - score_home) >= 2:
                score -= 40
                reasons.append(f"Casa in svantaggio di {score_away - score_home} gol, improbabile vittoria")
            elif score_home < score_away and minute >= 70:
                score -= 20
                warnings.append(f"Casa in svantaggio al {minute}', poco tempo rimanente")
        
        if '1x2_away' in market or 'away_win' in market:
            if score_away < score_home and (score_home - score_away) >= 2:
                score -= 40
                reasons.append(f"Ospite in svantaggio di {score_home - score_away} gol, improbabile vittoria")
            elif score_away < score_home and minute >= 70:
                score -= 20
                warnings.append(f"Ospite in svantaggio al {minute}', poco tempo rimanente")
        
        # 4. Validazione partita "morta"
        shots_home = live_data.get('shots_home', 0)
        shots_away = live_data.get('shots_away', 0)
        if minute >= 60 and shots_home + shots_away < 5:
            score -= 15
            warnings.append("Partita poco attiva (pochi tiri), potrebbe essere 'morta'")
        
        # 5. Validazione differenza gol eccessiva
        goal_diff = abs(score_home - score_away)
        if goal_diff >= 3:
            score -= 30
            warnings.append(f"Partita già decisa (differenza {goal_diff} gol), pochi eventi attesi")
        
        return max(0, score), reasons, warnings


class AISignalQualityScorer:
    """Valuta qualità complessiva del segnale"""
    
    def __init__(self, ai_pipeline=None, learned_weights=None):
        self.ai_pipeline = ai_pipeline
        self.context_validator = ContextAwareValidator()
        self.learned_weights = learned_weights  # Pesi appresi (opzionale)
    
    def score_signal(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Dict[str, Any],
        min_quality_score: float = 75.0
    ) -> QualityScore:
        """
        Valuta qualità segnale con sistema combinato.
        
        Returns:
            QualityScore con punteggio totale e dettagli
        """
        live_opp = opportunity.get('live_opportunity')
        if not live_opp:
            return QualityScore(
                total_score=0,
                context_score=0,
                data_quality_score=0,
                logic_score=0,
                timing_score=0,
                is_approved=False,
                reasons=["Opportunità senza live_opportunity"],
                warnings=[],
                errors=["live_opportunity mancante"]
            )
        
        # 1. Context-Aware Validation
        context_score, context_reasons, context_warnings = self.context_validator.validate_context(
            opportunity, match_data, live_data
        )
        
        # 2. Data Quality Score
        data_quality_score, data_reasons, data_warnings = self._score_data_quality(live_data, live_opp)
        
        # 3. Logic Score
        logic_score, logic_reasons, logic_warnings = self._score_logic(opportunity, match_data, live_data)
        
        # 4. Timing Score
        timing_score, timing_reasons, timing_warnings = self._score_timing(live_data, live_opp)
        
        # 5. Calcola score totale (media pesata)
        # Usa pesi appresi se disponibili, altrimenti default
        weights = self.learned_weights if self.learned_weights else {
            'context': 0.35,
            'data_quality': 0.25,
            'logic': 0.25,
            'timing': 0.15
        }
        
        total_score = (
            context_score * weights.get('context', 0.35) +
            data_quality_score * weights.get('data_quality', 0.25) +
            logic_score * weights.get('logic', 0.25) +
            timing_score * weights.get('timing', 0.15)
        )
        
        # 6. Raccogli tutti i motivi
        all_reasons = context_reasons + data_reasons + logic_reasons + timing_reasons
        all_warnings = context_warnings + data_warnings + logic_warnings + timing_warnings
        
        # 7. Decisione finale: approva solo se score >= min_quality_score E nessun errore critico
        # Se ci sono "reasons" (errori critici), blocca sempre
        has_critical_errors = len(all_reasons) > 0
        is_approved = total_score >= min_quality_score and not has_critical_errors
        
        return QualityScore(
            total_score=round(total_score, 1),
            context_score=round(context_score, 1),
            data_quality_score=round(data_quality_score, 1),
            logic_score=round(logic_score, 1),
            timing_score=round(timing_score, 1),
            is_approved=is_approved,
            reasons=all_reasons,
            warnings=all_warnings,
            errors=[]
        )
    
    def _score_data_quality(
        self,
        live_data: Dict[str, Any],
        live_opp: Any
    ) -> Tuple[float, List[str], List[str]]:
        """Valuta qualità dati statistici"""
        score = 100.0
        reasons = []
        warnings = []
        
        # Verifica statistiche disponibili
        shots_home = live_data.get('shots_home', 0)
        shots_away = live_data.get('shots_away', 0)
        shots_on_target_home = live_data.get('shots_on_target_home', 0)
        shots_on_target_away = live_data.get('shots_on_target_away', 0)
        possession_home = live_data.get('possession_home')
        xg_home = live_data.get('xg_home', 0.0)
        xg_away = live_data.get('xg_away', 0.0)
        
        # Conta statistiche disponibili
        stats_count = 0
        if shots_home > 0 or shots_away > 0:
            stats_count += 1
        if shots_on_target_home > 0 or shots_on_target_away > 0:
            stats_count += 1
        if possession_home is not None:
            stats_count += 1
        if xg_home > 0 or xg_away > 0:
            stats_count += 1
        
        # Penalizza se statistiche insufficienti
        if stats_count < 2:
            score -= 30
            reasons.append(f"Statistiche insufficienti (solo {stats_count} tipi disponibili)")
        elif stats_count < 3:
            score -= 15
            warnings.append(f"Statistiche limitate ({stats_count} tipi disponibili)")
        
        # Verifica coerenza statistiche
        if shots_on_target_home > shots_home:
            score -= 20
            reasons.append(f"Statistiche incoerenti: SOT ({shots_on_target_home}) > Shots ({shots_home})")
        
        if shots_on_target_away > shots_away:
            score -= 20
            reasons.append(f"Statistiche incoerenti: SOT ({shots_on_target_away}) > Shots ({shots_away})")
        
        # Verifica possesso
        if possession_home is not None:
            possession_away = live_data.get('possession_away', 0)
            total_possession = possession_home + possession_away
            if total_possession < 90 or total_possession > 110:
                score -= 10
                warnings.append(f"Possesso anomalo: {possession_home}% + {possession_away}% = {total_possession}%")
        
        # Verifica has_live_stats (errore critico)
        if hasattr(live_opp, 'has_live_stats') and not live_opp.has_live_stats:
            score -= 50
            reasons.append("Nessuna statistica live significativa disponibile")
        
        return max(0, score), reasons, warnings
    
    def _score_logic(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """Valuta coerenza logica del segnale"""
        score = 100.0
        reasons = []
        warnings = []
        
        live_opp = opportunity.get('live_opportunity')
        market = getattr(live_opp, 'market', 'unknown')
        confidence = getattr(live_opp, 'confidence', 0.0)
        ev = getattr(live_opp, 'ev', 0.0)
        
        # Verifica confidence ragionevole
        if confidence < 70:
            score -= 20
            warnings.append(f"Confidence bassa ({confidence:.1f}%)")
        elif confidence > 95:
            score -= 10
            warnings.append(f"Confidence molto alta ({confidence:.1f}%), potrebbe essere sovrastimata")
        
        # Verifica EV ragionevole
        if ev < 0:
            score -= 30
            reasons.append(f"EV negativo ({ev:.1f}%), segnale non profittevole")
        elif ev < 5:
            score -= 15
            warnings.append(f"EV basso ({ev:.1f}%), margine limitato")
        
        # Verifica coerenza confidence-EV
        if confidence > 80 and ev < 5:
            score -= 10
            warnings.append("Alta confidence ma EV basso, possibile incoerenza")
        
        return max(0, score), reasons, warnings
    
    def _score_timing(
        self,
        live_data: Dict[str, Any],
        live_opp: Any
    ) -> Tuple[float, List[str], List[str]]:
        """Valuta timing del segnale"""
        score = 100.0
        reasons = []
        warnings = []
        
        minute = live_data.get('minute', 0)
        market = getattr(live_opp, 'market', 'unknown')
        
        # Timing specifico per mercato
        if 'over_3.5' in market:
            if minute > 70:
                score -= 30
                reasons.append(f"Over 3.5 troppo tardi (minuto {minute})")
            elif minute > 60:
                score -= 15
                warnings.append(f"Over 3.5 avanzato (minuto {minuto})")
        
        if 'under_1.5' in market and 'ht' not in market:
            if minute < 65:
                score -= 20
                reasons.append(f"Under 1.5 troppo presto (minuto {minute})")
        
        if 'win_either_half' in market:
            total_goals = live_data.get('score_home', 0) + live_data.get('score_away', 0)
            if total_goals == 0 and minute > 60:
                score -= 25
                reasons.append(f"Win Either Half su 0-0 troppo tardi (minuto {minute})")
        
        # 🚫 FIX: Highest Scoring Half troppo tardi (banale)
        if 'highest_scoring_half' in market:
            total_goals = live_data.get('score_home', 0) + live_data.get('score_away', 0)
            if minute > 50:  # Dopo il primo tempo
                score -= 40
                reasons.append(f"Highest Scoring Half troppo tardi (minuto {minute}') - risultato già definito, banale")
            elif minute > 45 and total_goals > 0:
                # Se siamo dopo il primo tempo e ci sono gol, è banale
                score -= 30
                reasons.append(f"Highest Scoring Half dopo primo tempo (minuto {minute}') con {total_goals} gol - banale")
        
        # Timing generale
        if minute < 15:
            score -= 10
            warnings.append("Partita molto presto, pochi dati disponibili")
        elif minute > 85:
            score -= 20
            warnings.append("Partita molto avanzata, poco tempo rimanente")
        
        return max(0, score), reasons, warnings


class SignalQualityGate:
    """
    Gate finale che decide se inviare o bloccare un segnale.
    Combina Context-Aware Validator + AI Quality Scorer.
    
    Supporta apprendimento automatico tramite SignalQualityLearner.
    """

    def __init__(self, ai_pipeline=None, min_quality_score: float = 75.0, learner=None):
        """
        Args:
            ai_pipeline: Pipeline AI opzionale per analisi avanzata
            min_quality_score: Score minimo per approvare (default: 75)
            learner: SignalQualityLearner opzionale per apprendimento
        """
        # Carica pesi appresi se learner disponibile
        learned_weights = None
        if learner:
            learned_weights = learner.get_learned_weights()
            if learned_weights:
                min_quality_score = learner.get_learned_min_score()
                logger.info(f"✅ Signal Quality Gate initialized con apprendimento (min score: {min_quality_score:.1f})")
        
        self.quality_scorer = AISignalQualityScorer(ai_pipeline=ai_pipeline, learned_weights=learned_weights)
        self.min_quality_score = min_quality_score
        self.learner = learner
        
        if not learner or not learned_weights:
            logger.info(f"✅ Signal Quality Gate initialized (min score: {min_quality_score})")
    
    def should_send_signal(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> Tuple[bool, QualityScore]:
        """
        Decide se inviare o bloccare un segnale.
        
        Returns:
            (should_send, quality_score)
        """
        # Valuta qualità segnale (passa min_quality_score appreso)
        quality_score = self.quality_scorer.score_signal(
            opportunity, match_data, live_data, min_quality_score=self.min_quality_score
        )
        
        # Log dettagliato
        live_opp = opportunity.get('live_opportunity')
        market = getattr(live_opp, 'market', 'unknown') if live_opp else 'unknown'
        match_id = opportunity.get('match_id', 'unknown')
        
        if quality_score.is_approved:
            logger.info(
                f"✅ Segnale APPROVATO: {match_id} - {market} | "
                f"Score: {quality_score.total_score:.1f}/100 "
                f"(Context: {quality_score.context_score:.1f}, "
                f"Data: {quality_score.data_quality_score:.1f}, "
                f"Logic: {quality_score.logic_score:.1f}, "
                f"Timing: {quality_score.timing_score:.1f})"
            )
        else:
            logger.warning(
                f"❌ Segnale BLOCCATO: {match_id} - {market} | "
                f"Score: {quality_score.total_score:.1f}/100 "
                f"(min: {self.min_quality_score})"
            )
            if quality_score.reasons:
                logger.warning(f"   Motivi: {', '.join(quality_score.reasons)}")
            if quality_score.warnings:
                logger.info(f"   Avvisi: {', '.join(quality_score.warnings)}")
        
        # 🆕 Traccia segnale per apprendimento (se learner disponibile)
        if self.learner:
            try:
                live_opp = opportunity.get('live_opportunity')
                if live_opp:
                    record_id = self.learner.record_signal(
                        match_id=match_id,
                        market=market,
                        minute=live_data.get('minute', 0),
                        score_home=live_data.get('score_home', 0),
                        score_away=live_data.get('score_away', 0),
                        quality_score=quality_score.total_score,
                        context_score=quality_score.context_score,
                        data_quality_score=quality_score.data_quality_score,
                        logic_score=quality_score.logic_score,
                        timing_score=quality_score.timing_score,
                        was_approved=quality_score.is_approved,
                        block_reasons=quality_score.reasons,
                        confidence=getattr(live_opp, 'confidence', 0.0),
                        ev=getattr(live_opp, 'ev', 0.0)
                    )
                    logger.info(f"📝 Segnale registrato nel database (ID: {record_id}): {match_id}/{market} - {'APPROVATO' if quality_score.is_approved else 'BLOCCATO'}")
                else:
                    logger.debug(f"⚠️  live_opportunity non disponibile per registrazione: {match_id}")
            except Exception as e:
                logger.error(f"❌ Errore tracciamento segnale per apprendimento: {e}", exc_info=True)
        else:
            logger.debug(f"⚠️  Learner non disponibile per registrazione segnale: {match_id}")

        return quality_score.is_approved, quality_score

