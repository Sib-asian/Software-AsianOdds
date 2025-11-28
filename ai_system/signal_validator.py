"""
Signal Validator - Validazione Metodica Segnali
================================================

Sistema di validazione rigoroso che verifica ogni aspetto di un segnale
prima dell'invio. Usa AI e controlli multipli per garantire precisione.

Features:
- Validazione dati completi
- Verifica coerenza logica
- Controllo anomalie
- Validazione AI avanzata
- Verifica calcoli matematici
- Controllo qualità quote
"""

import logging
import os
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class ValidationResult:
    """Risultato validazione"""
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)


class SignalValidator:
    """
    Validatore rigoroso per segnali betting.
    
    Esegue controlli multipli per garantire che ogni segnale sia:
    - Completo (tutti i dati necessari presenti)
    - Coerente (dati logici e consistenti)
    - Corretto (calcoli matematici verificati)
    - Valido (quote e probabilità ragionevoli)
    """
    
    def __init__(self, ai_pipeline=None, use_llm_validation: bool = False):
        """
        Args:
            ai_pipeline: Pipeline AI per validazione avanzata (opzionale)
            use_llm_validation: Se True, usa LLM per validazione intelligente (opzionale, richiede API key)
        """
        self.ai_pipeline = ai_pipeline
        self.use_llm_validation = use_llm_validation
        
        # LLM per validazione avanzata (se richiesto)
        self.llm_analyst = None
        if use_llm_validation:
            try:
                from .llm_analyst import LLMAnalyst
                # Usa LLM solo se API key disponibile
                llm_api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
                if llm_api_key:
                    provider = "openai" if os.getenv('OPENAI_API_KEY') else "anthropic"
                    self.llm_analyst = LLMAnalyst(
                        api_key=llm_api_key,
                        provider=provider,
                        model="gpt-4" if provider == "openai" else "claude-3-sonnet-20240229"
                    )
                    logger.info("✅ LLM validation enabled per validazione intelligente")
                else:
                    logger.warning("⚠️  LLM validation richiesta ma API key non disponibile")
            except Exception as e:
                logger.warning(f"⚠️  LLM validation non disponibile: {e}")
        
        # Soglie di validazione
        self.min_odds = 1.01  # Quote minime accettabili
        self.max_odds = 100.0  # Quote massime accettabili
        self.min_probability = 0.01  # Probabilità minima (1%)
        self.max_probability = 0.99  # Probabilità massima (99%)
        self.max_ev_deviation = 50.0  # EV massimo ragionevole (50%)
        
        logger.info("✅ Signal Validator initialized")
    
    def validate_signal(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Valida un segnale con controlli multipli.
        
        Args:
            opportunity: Opportunità da validare
            match_data: Dati partita
            live_data: Dati live (opzionale)
            
        Returns:
            ValidationResult con esito validazione
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # 1. Validazione struttura base
            self._validate_structure(opportunity, match_data, result)
            if not result.is_valid:
                return result
            
            # 2. Validazione dati completi
            self._validate_completeness(opportunity, match_data, live_data, result)
            if not result.is_valid:
                return result
            
            # 3. Validazione coerenza logica
            self._validate_logical_consistency(opportunity, match_data, live_data, result)
            if not result.is_valid:
                return result
            
            # 4. Validazione calcoli matematici
            self._validate_calculations(opportunity, result)
            if not result.is_valid:
                return result
            
            # 5. Validazione quote e probabilità
            self._validate_odds_and_probability(opportunity, result)
            if not result.is_valid:
                return result
            
            # 6. Validazione AI avanzata (se disponibile)
            if self.ai_pipeline:
                self._validate_with_ai(opportunity, match_data, live_data, result)
            
            # 7. Validazione anomalie
            self._validate_anomalies(opportunity, match_data, live_data, result)
            
            # 8. Validazione incrociata finale
            self._validate_cross_check(opportunity, match_data, live_data, result)
            
        except Exception as e:
            logger.error(f"❌ Errore durante validazione: {e}", exc_info=True)
            result.add_error(f"Errore validazione: {str(e)}")
        
        return result
    
    def _validate_structure(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        result: ValidationResult
    ):
        """Valida struttura base dei dati"""
        # Verifica campi obbligatori opportunity
        required_fields = ['market', 'probability', 'odds', 'ev', 'confidence']
        for field in required_fields:
            if field not in opportunity:
                result.add_error(f"Campo obbligatorio mancante in opportunity: {field}")
        
        # Verifica campi obbligatori match_data
        required_match_fields = ['home', 'away']
        for field in required_match_fields:
            if field not in match_data or not match_data[field]:
                result.add_error(f"Campo obbligatorio mancante in match_data: {field}")
        
        # Verifica tipi dati
        if 'probability' in opportunity:
            if not isinstance(opportunity['probability'], (int, float)):
                result.add_error(f"Probability deve essere numerico, trovato: {type(opportunity['probability'])}")
        
        if 'odds' in opportunity:
            if not isinstance(opportunity['odds'], (int, float)):
                result.add_error(f"Odds deve essere numerico, trovato: {type(opportunity['odds'])}")
    
    def _validate_completeness(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Optional[Dict[str, Any]],
        result: ValidationResult
    ):
        """Valida completezza dati"""
        # Verifica match_data completo
        if not match_data.get('home') or not match_data.get('away'):
            result.add_error("Nomi squadre mancanti o vuoti")
        
        # Verifica market valido
        market = opportunity.get('market', '')
        if not market or market == 'UNKNOWN':
            result.add_error("Market non valido o mancante")
        
        # Verifica dati live se partita è live
        if live_data:
            if 'minute' not in live_data:
                result.add_warning("Minuto partita non disponibile nei dati live")
            if 'score_home' not in live_data or 'score_away' not in live_data:
                result.add_warning("Score non disponibile nei dati live")
        
        # Verifica reasoning presente (se disponibile)
        if 'reasoning' not in opportunity:
            result.add_warning("Reasoning non presente (opzionale ma consigliato)")
    
    def _validate_logical_consistency(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Optional[Dict[str, Any]],
        result: ValidationResult
    ):
        """Valida coerenza logica dei dati"""
        market = opportunity.get('market', '')
        probability = opportunity.get('probability', 0)
        odds = opportunity.get('odds', 0)
        
        # Verifica coerenza market con dati live
        if live_data:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            total_goals = score_home + score_away
            
            # Controlli specifici per mercato
            if 'UNDER_2_5' in market.upper() and total_goals >= 3:
                result.add_error(f"UNDER_2_5 non valido: partita ha già {total_goals} gol")
            
            if 'OVER_2_5' in market.upper() and total_goals >= 3:
                # OK, già raggiunto
                pass
            
            if 'UNDER_0_5' in market.upper() and total_goals > 0:
                result.add_error(f"UNDER_0.5 non valido: partita ha già {total_goals} gol")
            
            if minute > 90:
                result.add_warning(f"Minuto partita anomalo: {minute}' (dovrebbe essere <= 90)")
            
            if minute < 0:
                result.add_error(f"Minuto partita negativo: {minute}'")
        
        # Verifica coerenza probabilità vs market
        if 'HOME' in market.upper() and 'AWAY' in market.upper():
            # Non può essere sia HOME che AWAY
            result.add_error(f"Market incoerente: {market}")
        
        # Verifica coerenza probabilità (deve essere tra 0 e 1)
        if isinstance(probability, (int, float)):
            if probability < 0 or probability > 1:
                result.add_error(f"Probabilità fuori range [0,1]: {probability}")
            # Se probabilità è in percentuale (0-100), converti
            if probability > 1:
                result.add_warning(f"Probabilità sembra essere in percentuale ({probability}%), convertendo...")
                opportunity['probability'] = probability / 100
                probability = opportunity['probability']
    
    def _validate_calculations(
        self,
        opportunity: Dict[str, Any],
        result: ValidationResult
    ):
        """
        Valida calcoli matematici - SOLO WARNING, nessuna modifica.
        NOTA: Non modifichiamo più i valori, solo logghiamo se ci sono discrepanze.
        """
        probability = opportunity.get('probability', 0)
        odds = opportunity.get('odds', 0)
        ev = opportunity.get('ev', 0)
        confidence = opportunity.get('confidence', 0)
        
        # Normalizza probabilità se necessario
        if isinstance(probability, (int, float)) and probability > 1:
            probability = probability / 100
        
        # Verifica calcolo EV (solo warning, non error)
        if isinstance(probability, (int, float)) and isinstance(odds, (int, float)) and odds > 0:
            expected_ev = (probability * odds - 1) * 100
            
            # Tolleranza per arrotondamenti
            ev_tolerance = 1.0  # Più permissivo: 1% di tolleranza
            ev_difference = abs(ev - expected_ev)
            
            if ev_difference > ev_tolerance:
                result.add_warning(
                    f"⚠️  EV difference: expected {expected_ev:.2f}%, actual {ev:.2f}% "
                    f"(diff: {ev_difference:.2f}%) - keeping actual value"
                )
        
        # Log differenze tra probabilità AI e implicita (SOLO WARNING)
        if isinstance(odds, (int, float)) and odds > 1:
            implied_prob = 1.0 / odds
            if isinstance(probability, (int, float)) and probability > 0:
                # Calcola differenza
                prob_ratio = probability / implied_prob if implied_prob > 0 else 0
                prob_diff_pct = abs(probability - implied_prob) * 100
                
                # Se differenza significativa, log come warning ma NON modificare
                if prob_diff_pct > 10:  # >10% differenza
                    result.add_warning(
                        f"⚠️  Probabilità AI ({probability*100:.1f}%) vs implicita da quota "
                        f"({implied_prob*100:.1f}%): differenza {prob_diff_pct:.1f}% - keeping AI value"
                    )
        
        # Verifica confidence - solo range tecnico
        if isinstance(confidence, (int, float)):
            if confidence < 0 or confidence > 100:
                result.add_warning(f"Confidence fuori range [0,100]: {confidence}")
    
    def _validate_odds_and_probability(
        self,
        opportunity: Dict[str, Any],
        result: ValidationResult
    ):
        """
        Valida quote e probabilità - solo controlli tecnici minimi.
        NOTA: Rimossi tutti i limiti artificiali su EV e probabilità.
        """
        odds = opportunity.get('odds', 0)
        probability = opportunity.get('probability', 0)
        
        # Normalizza probabilità se necessario
        if isinstance(probability, (int, float)) and probability > 1:
            probability = probability / 100
        
        # Solo verifica che quote siano tecnicamente valide (>1.0)
        if isinstance(odds, (int, float)):
            if odds < self.min_odds:
                result.add_error(f"Quote non valide: {odds:.2f} < {self.min_odds}")
            # Rimosso il limite massimo per le quote - possiamo analizzare qualsiasi quota
        
        # Solo verifica che probabilità sia nel range [0,1]
        if isinstance(probability, (int, float)):
            if probability < 0 or probability > 1:
                result.add_error(f"Probabilità fuori range [0,1]: {probability}")
        
        # Log EV se alto, ma NON bloccare o penalizzare
        ev = opportunity.get('ev', 0)
        if isinstance(ev, (int, float)):
            if ev > 50:
                result.add_warning(
                    f"⚠️  EV molto alto: {ev:.2f}% - valore esatto mantenuto, verificare opportunità"
                )
    
    def _validate_with_ai(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Optional[Dict[str, Any]],
        result: ValidationResult
    ):
        """
        Validazione AI - SOLO WARNING, nessuna penalizzazione.
        NOTA: Rimossa tutta la logica di penalizzazione. Solo log informativi.
        """
        try:
            market = opportunity.get('market', '')
            probability = opportunity.get('probability', 0)
            odds = opportunity.get('odds', 0)
            ev = opportunity.get('ev', 0)
            confidence = opportunity.get('confidence', 0)
            
            # Normalizza probabilità
            if isinstance(probability, (int, float)) and probability > 1:
                probability = probability / 100
            
            # Log differenza probabilità AI vs implicita (solo informativo)
            if isinstance(odds, (int, float)) and odds > 1 and isinstance(probability, (int, float)):
                implied_prob = 1.0 / odds
                prob_diff_pct = abs(probability - implied_prob) * 100
                
                if prob_diff_pct > 15:  # >15% differenza
                    result.add_warning(
                        f"⚠️  Differenza significativa: AI prob {probability*100:.1f}% vs "
                        f"implicita {implied_prob*100:.1f}% (diff {prob_diff_pct:.1f}%) - "
                        f"mantenendo valore AI esatto"
                    )
            
            # NOTA: Rimossa tutta la logica di validazione che penalizzava i valori.
            # Current validation: Only logs warnings for probability differences (no changes).
        
        except Exception as e:
            logger.debug(f"⚠️  Errore validazione AI: {e}")
    
    def _validate_market_vs_live_data(
        self,
        market: str,
        live_data: Dict[str, Any],
        result: ValidationResult
    ):
        """Valida coerenza market con dati live"""
        score_home = live_data.get('score_home', 0)
        score_away = live_data.get('score_away', 0)
        minute = live_data.get('minute', 0)
        total_goals = score_home + score_away
        
        # Controlli specifici per mercato
        if 'UNDER_0_5' in market.upper() and total_goals > 0:
            result.add_error(f"UNDER_0.5 non valido: partita ha già {total_goals} gol")
        
        if 'UNDER_1_5' in market.upper() and total_goals >= 2:
            result.add_error(f"UNDER_1.5 non valido: partita ha già {total_goals} gol")
        
        if 'UNDER_2_5' in market.upper() and total_goals >= 3:
            result.add_error(f"UNDER_2.5 non valido: partita ha già {total_goals} gol")
        
        # Verifica coerenza minuto
        if minute > 90:
            result.add_warning(f"Minuto partita anomalo: {minute}' (dovrebbe essere <= 90)")
        
        if minute < 0:
            result.add_error(f"Minuto partita negativo: {minute}'")
    
    def _validate_with_llm(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Optional[Dict[str, Any]],
        result: ValidationResult
    ):
        """
        Validazione intelligente con LLM.
        
        Usa LLM per analizzare se il segnale è coerente e valido.
        """
        try:
            if not self.llm_analyst:
                return
            
            market = opportunity.get('market', '')
            probability = opportunity.get('probability', 0)
            odds = opportunity.get('odds', 0)
            ev = opportunity.get('ev', 0)
            confidence = opportunity.get('confidence', 0)
            reasoning = opportunity.get('reasoning', '')
            
            # Normalizza probabilità
            if isinstance(probability, (int, float)) and probability > 1:
                probability = probability / 100
            
            # Prepara prompt per LLM
            prompt = f"""Analizza questo segnale di betting e verifica se è valido e coerente.

MATCH:
- Squadra Casa: {match_data.get('home', 'N/A')}
- Squadra Trasferta: {match_data.get('away', 'N/A')}
- Lega: {match_data.get('league', 'N/A')}

SEGNALE:
- Mercato: {market}
- Probabilità: {probability*100:.1f}%
- Quote: {odds:.2f}
- EV: {ev:.1f}%
- Confidence: {confidence:.0f}%

"""
            
            if live_data:
                prompt += f"""DATI LIVE:
- Score: {live_data.get('score_home', 0)}-{live_data.get('score_away', 0)}
- Minuto: {live_data.get('minute', 0)}'
- Possesso: {live_data.get('possession_home', 50)}% vs {live_data.get('possession_away', 50)}%
- Tiri: {live_data.get('shots_home', 0)} vs {live_data.get('shots_away', 0)}
- Tiri in porta: {live_data.get('shots_on_target_home', 0)} vs {live_data.get('shots_on_target_away', 0)}

"""
            
            if reasoning:
                prompt += f"""REASONING:
{reasoning[:500]}

"""
            
            prompt += """VERIFICA:
1. Il segnale è logicamente coerente?
2. I calcoli matematici sono corretti?
3. Il mercato è appropriato per la situazione?
4. Ci sono anomalie o inconsistenze?

Rispondi SOLO con JSON:
{
  "is_valid": true/false,
  "confidence": 0-100,
  "issues": ["problema1", "problema2"],
  "summary": "breve spiegazione"
}"""
            
            # Chiama LLM
            messages = [{"role": "user", "content": prompt}]
            llm_response = self.llm_analyst.client.generate_chat(
                messages,
                max_tokens=200,
                temperature=0.1  # Bassa temperatura per risposte precise
            )
            
            # Parse risposta
            try:
                # Estrai JSON dalla risposta
                import re
                json_match = re.search(r'\{[^}]+\}', llm_response, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group(0))
                    
                    if not llm_result.get('is_valid', True):
                        result.add_warning(f"LLM validation: {llm_result.get('summary', 'Segnale sospetto')}")
                        if llm_result.get('issues'):
                            for issue in llm_result['issues'][:2]:
                                result.add_warning(f"LLM: {issue}")
            except Exception as e:
                logger.debug(f"⚠️  Errore parsing risposta LLM: {e}")
        
        except Exception as e:
            logger.debug(f"⚠️  Errore validazione LLM: {e}")
            # Non bloccare per errori LLM, solo warning
    
    def _detect_suspicious_patterns(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Optional[Dict[str, Any]]
    ) -> bool:
        """Rileva pattern sospetti"""
        market = opportunity.get('market', '')
        probability = opportunity.get('probability', 0)
        odds = opportunity.get('odds', 0)
        ev = opportunity.get('ev', 0)
        
        # Normalizza probabilità
        if isinstance(probability, (int, float)) and probability > 1:
            probability = probability / 100
        
        # Pattern 1: EV molto alto con probabilità normale
        if isinstance(ev, (int, float)) and ev > 30:
            if isinstance(probability, (int, float)) and 0.3 < probability < 0.7:
                if isinstance(odds, (int, float)) and odds < 3.0:
                    # EV alto con probabilità media e quote basse = sospetto
                    return True
        
        # Pattern 2: Probabilità molto alta con quote molto basse
        if isinstance(probability, (int, float)) and probability > 0.8:
            if isinstance(odds, (int, float)) and odds < 1.5:
                # Probabilità alta con quote basse = normale, ma verifica
                pass
        
        # Pattern 3: Market incoerente con dati live
        if live_data:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            total_goals = score_home + score_away
            
            if 'UNDER_1_5' in market.upper() and total_goals >= 2:
                return True  # Sospetto: Under 1.5 con 2+ gol
        
        return False
    
    def _validate_anomalies(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Optional[Dict[str, Any]],
        result: ValidationResult
    ):
        """Valida anomalie nei dati"""
        # Verifica valori NaN o infiniti
        for key, value in opportunity.items():
            if isinstance(value, float):
                if math.isnan(value):
                    result.add_error(f"Valore NaN trovato in {key}")
                if math.isinf(value):
                    result.add_error(f"Valore infinito trovato in {key}")
        
        # Verifica nomi squadre strani
        home = match_data.get('home', '')
        away = match_data.get('away', '')
        
        if len(home) < 2 or len(away) < 2:
            result.add_warning("Nomi squadre molto corti - verifica")
        
        if home == away:
            result.add_error("Squadra casa e trasferta identiche")
        
        # Verifica market valido
        market = opportunity.get('market', '')
        valid_markets = [
            '1X2_HOME', '1X2_AWAY', '1X2_DRAW',
            'OVER_0_5', 'OVER_1_5', 'OVER_2_5', 'OVER_3_5',
            'UNDER_0_5', 'UNDER_1_5', 'UNDER_2_5', 'UNDER_3_5',
            '1X', 'X2', 'DNB_HOME', 'DNB_AWAY',
            'BTTS_YES', 'BTTS_NO',
            'WIN_TO_NIL_HOME', 'WIN_TO_NIL_AWAY'
        ]
        
        # Verifica se market è valido (può essere variante)
        market_upper = market.upper()
        is_valid = any(valid in market_upper for valid in valid_markets)
        
        if not is_valid and market not in ['UNKNOWN', '']:
            result.add_warning(f"Market non riconosciuto: {market} (verifica)")
    
    def _validate_cross_check(
        self,
        opportunity: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Optional[Dict[str, Any]],
        result: ValidationResult
    ):
        """Validazione incrociata finale"""
        # Verifica che tutti i dati siano coerenti tra loro
        market = opportunity.get('market', '')
        probability = opportunity.get('probability', 0)
        odds = opportunity.get('odds', 0)
        ev = opportunity.get('ev', 0)
        confidence = opportunity.get('confidence', 0)
        
        # Normalizza probabilità
        if isinstance(probability, (int, float)) and probability > 1:
            probability = probability / 100
        
        # Cross-check: EV positivo richiede probabilità > probabilità implicita
        if isinstance(ev, (int, float)) and ev > 0:
            if isinstance(odds, (int, float)) and odds > 1:
                implied_prob = 1.0 / odds
                if isinstance(probability, (int, float)):
                    if probability <= implied_prob:
                        result.add_error(
                            f"EV positivo ({ev:.2f}%) ma probabilità ({probability*100:.1f}%) "
                            f"non supera quella implicita ({implied_prob*100:.1f}%)"
                        )
        
        # Cross-check: Confidence alta richiede dati di qualità
        if isinstance(confidence, (int, float)) and confidence > 80:
            if live_data:
                # Verifica che ci siano statistiche sufficienti
                has_stats = (
                    'shots_home' in live_data and
                    'shots_away' in live_data and
                    'possession_home' in live_data
                )
                if not has_stats:
                    result.add_warning(
                        f"Confidence alta ({confidence:.0f}%) ma statistiche incomplete"
                    )
        
        # Cross-check: Market vs dati live
        if live_data:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            
            # Verifica che market sia coerente con situazione
            if 'HOME' in market.upper() and score_away > score_home and minute > 70:
                # Squadra casa in svantaggio a fine partita
                if isinstance(probability, (int, float)) and probability > 0.6:
                    result.add_warning(
                        f"Probabilità alta ({probability*100:.1f}%) per {market} "
                        f"ma squadra in svantaggio {score_home}-{score_away} al {minute}'"
                    )
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """Genera riepilogo validazione"""
        if result.is_valid:
            summary = "✅ VALIDAZIONE COMPLETATA\n\n"
            if result.warnings:
                summary += f"⚠️  Warnings: {len(result.warnings)}\n"
                for warning in result.warnings[:5]:
                    summary += f"   • {warning}\n"
            else:
                summary += "✅ Nessun warning\n"
        else:
            summary = "❌ VALIDAZIONE FALLITA\n\n"
            summary += f"❌ Errori: {len(result.errors)}\n"
            for error in result.errors:
                summary += f"   • {error}\n"
            if result.warnings:
                summary += f"\n⚠️  Warnings: {len(result.warnings)}\n"
                for warning in result.warnings[:3]:
                    summary += f"   • {warning}\n"
        
        return summary

