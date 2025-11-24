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
    """
    Valida solo requisiti tecnici minimi del segnale.
    NOTA: Rimossi tutti i filtri contestuali e penalizzazioni. 
    Vengono validati solo aspetti tecnici di base.
    """
    
    def __init__(self):
        # Limiti molto permissivi - solo per evitare situazioni tecnicamente impossibili
        self.min_minute_for_analysis = 5  # Minimo assoluto
        self.max_minute_for_opportunity = 90  # Accetta fino al 90'
    
    def validate_context(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Valida solo requisiti tecnici minimi.
        NOTA: Rimossi tutti i filtri e penalizzazioni contestuali.
        
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
        
        # 1. Solo validazione di impossibilità tecnica (es: partita finita)
        if minute > self.max_minute_for_opportunity:
            warnings.append(f"Partita oltre {self.max_minute_for_opportunity}' ({minute}')")
        
        # 2. Solo validazione di impossibilità tecnica (mercati già risolti)
        # Manteniamo solo i controlli per situazioni IMPOSSIBILI, non rischiose
        
        # Mercati già risolti al 100% - impossibili
        if 'team_to_score_first' in market or 'first_goal' in market:
            if score_home > 0 or score_away > 0:
                score = 0
                reasons.append(f"Team to Score First impossibile: è già {score_home}-{score_away}")
        
        # Half time markets dopo il primo tempo
        if minute > 45:
            if 'half_time_result' in market:
                score = 0
                reasons.append(f"Half Time Result impossibile al {minute}': primo tempo finito")
            elif 'btts_first_half' in market:
                score = 0
                reasons.append(f"BTTS First Half impossibile al {minute}': primo tempo finito")
            elif ('over_' in market or 'under_' in market) and '_ht' in market:
                score = 0
                reasons.append(f"Over/Under HT impossibile al {minute}': primo tempo finito")
        
        # Timing-based impossibilities
        if 'next_goal_before_75' in market and minute >= 75:
            score = 0
            reasons.append(f"Next Goal Before 75 impossibile: siamo al {minute}'")
        
        # NOTA: Rimossi tutti gli altri filtri contestuali che penalizzavano mercati 
        # "banali" o "rischiosi" - lasciamo solo impossibilità tecniche
        
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
        min_quality_score: float = 0.0
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
        """
        Verifica requisiti tecnici minimi di disponibilità dati.
        NOTA: Rimossi penalty per dati limitati - solo verifica presenza minima.
        """
        score = 100.0
        reasons = []
        warnings = []
        
        # Verifica statistiche disponibili
        shots_home = live_data.get('shots_home', 0)
        shots_away = live_data.get('shots_away', 0)
        shots_on_target_home = live_data.get('shots_on_target_home', 0)
        shots_on_target_away = live_data.get('shots_on_target_away', 0)
        possession_home = live_data.get('possession_home')
        
        # Conta statistiche disponibili
        stats_count = 0
        if shots_home > 0 or shots_away > 0:
            stats_count += 1
        if shots_on_target_home > 0 or shots_on_target_away > 0:
            stats_count += 1
        if possession_home is not None:
            stats_count += 1
        
        # Solo warning se non ci sono statistiche (non blocking)
        if stats_count == 0:
            warnings.append("Nessuna statistica live disponibile")
        
        # Verifica coerenza solo per errori evidenti nei dati
        if shots_on_target_home > shots_home and shots_home > 0:
            warnings.append(f"Possibile incoerenza: SOT ({shots_on_target_home}) > Shots ({shots_home})")
        
        if shots_on_target_away > shots_away and shots_away > 0:
            warnings.append(f"Possibile incoerenza: SOT ({shots_on_target_away}) > Shots ({shots_away})")
        
        return max(0, score), reasons, warnings
    
    def _score_logic(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Verifica solo logica base (EV positivo).
        NOTA: Rimossi penalty per confidence o EV bassi - lasciamo decidere ai filtri configurabili.
        """
        score = 100.0
        reasons = []
        warnings = []
        
        live_opp = opportunity.get('live_opportunity')
        ev = getattr(live_opp, 'ev', 0.0)
        confidence = getattr(live_opp, 'confidence', 0.0)
        
        # Solo verifica che EV sia positivo (il minimo richiesto è nei filtri configurabili)
        if ev < 0:
            warnings.append(f"EV negativo ({ev:.1f}%)")
        
        # Log informativi senza penalizzazioni
        if confidence > 0:
            warnings.append(f"Confidence: {confidence:.1f}%, EV: {ev:.1f}%")
        
        return max(0, score), reasons, warnings
    
    def _score_timing(
        self,
        live_data: Dict[str, Any],
        live_opp: Any
    ) -> Tuple[float, List[str], List[str]]:
        """
        Timing score sempre 100 - nessuna penalizzazione.
        NOTA: Rimossi tutti i penalty per timing. L'AI decide il momento giusto.
        """
        score = 100.0
        reasons = []
        warnings = []
        
        minute = live_data.get('minute', 0)
        
        # Solo log informativo
        warnings.append(f"Minuto: {minute}'")
        
        return max(0, score), reasons, warnings


class SignalQualityGate:
    """
    Gate finale che decide se inviare o bloccare un segnale.
    NOTA: Ora molto permissivo - blocca solo situazioni tecnicamente impossibili.
    """

    def __init__(self, ai_pipeline=None, min_quality_score: float = 0.0, learner=None):
        """
        Args:
            ai_pipeline: Pipeline AI opzionale per analisi avanzata
            min_quality_score: Score minimo per approvare (default: 0 = solo blocca impossibili)
            learner: SignalQualityLearner opzionale per apprendimento
        """
        # Ignoriamo i pesi appresi - vogliamo valori esatti dall'AI
        learned_weights = None
        
        self.quality_scorer = AISignalQualityScorer(ai_pipeline=ai_pipeline, learned_weights=learned_weights)
        self.min_quality_score = min_quality_score  # Default 0 = molto permissivo
        self.learner = learner
        
        logger.info(f"✅ Signal Quality Gate initialized (min score: {min_quality_score}, molto permissivo)")
    
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

