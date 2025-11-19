"""
Sistema Live Betting Intelligente
==================================

Analizza partite in corso e suggerisce scommesse basate su:
- Situazione di gioco (favorita perde → ribaltone)
- Pattern di gioco (gol subito → under se partita chiusa)
- Eventi in campo (cartellini, possesso, ecc.)
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 🆕 Importa LiveMatchAI per analisi AI dedicata ai match live
try:
    from ai_system.live_match_ai import LiveMatchAI
    LIVE_MATCH_AI_AVAILABLE = True
except ImportError:
    LIVE_MATCH_AI_AVAILABLE = False
    logger.warning("⚠️  LiveMatchAI non disponibile - analisi AI base verrà utilizzata")


@dataclass
class LiveBettingOpportunity:
    """Opportunità di live betting"""
    match_id: str
    match_data: Dict[str, Any]
    situation: str  # Tipo di situazione (ribaltone, under_opportunity, etc.)
    recommendation: str  # Cosa puntare
    market: str  # Tipo di mercato (over_0.5, over_1.5, 1x, x2, etc.)
    reasoning: str  # Perché
    confidence: float  # 0-100
    odds: float
    stake_suggestion: float  # % del bankroll
    timestamp: datetime
    alternative_markets: List[Dict[str, Any]] = None  # Altri mercati suggeriti
    match_stats: Dict[str, Any] = None  # Statistiche partita dettagliate
    urgency_level: str = "NORMAL"  # URGENT, HIGH, NORMAL, LOW
    key_stats: Dict[str, Any] = field(default_factory=dict)  # Statistiche chiave mercato
    ev: float = 0.0  # Valore atteso (%)


class LiveBettingAdvisor:
    """
    Consulente per live betting basato su analisi situazione partita.
    """
    
    def __init__(
        self,
        notifier=None,
        min_confidence: float = 75.0,  # 🆕 AUMENTATO: 75% invece di 72% per ridurre segnali banali
        ai_pipeline=None,
        min_ev: float = 10.0,  # 🆕 AUMENTATO: 10% invece di 8% per ridurre segnali banali
        max_opportunities_per_match: int = 3
    ):
        """
        Args:
            notifier: TelegramNotifier per inviare alert
            min_confidence: Confidence minima % per considerare opportunità valida (default: 75% - aumentato per ridurre segnali banali)
            ai_pipeline: AI Pipeline per analisi avanzata (opzionale)
        """
        self.notifier = notifier
        self.min_confidence = min_confidence  # 75% aumentato: filtri più restrittivi per ridurre segnali banali
        self.ai_pipeline = ai_pipeline
        self.min_ev = max(0.0, min_ev)  # Soglia EV (default: 9% per partite live)
        self.max_opportunities_per_match = max(1, int(max_opportunities_per_match))
        self.market_translations = {
            'clean_sheet_home': 'Porta inviolata (Casa)',
            'clean_sheet_away': 'Porta inviolata (Trasferta)',
            'home_win': '1 (Vittoria Casa)',
            'away_win': '2 (Vittoria Trasferta)',
            'draw_no_bet_home': 'Draw No Bet Casa',
            'draw_no_bet_away': 'Draw No Bet Trasferta',
            'total_goals_odd': 'Totale gol dispari',
            'total_goals_even': 'Totale gol pari',
            'highest_scoring_half_1h': 'Tempo con più gol: 1° Tempo',
            'highest_scoring_half_2h': 'Tempo con più gol: 2° Tempo',
            'team_to_score_next_home': 'Segna prossimo gol: Casa',
            'team_to_score_next_away': 'Segna prossimo gol: Trasferta',
            'next_goal_home': 'Prossimo gol: Casa',
            'next_goal_away': 'Prossimo gol: Trasferta',
            'btts_first_half': 'Entrambe segnano 1° tempo',
            'win_either_half_home': 'Casa vince almeno un tempo',
            'win_either_half_away': 'Trasferta vince almeno un tempo',
            'goal_range_0_1': 'Range gol 0-1',
            'goal_range_2_3': 'Range gol 2-3',
            'goal_range_4_plus': 'Range gol 4+',
            'over_0.5': 'Over 0.5 gol',
            'over_1.5': 'Over 1.5 gol',
            'over_2.5': 'Over 2.5 gol',
            'over_3.5': 'Over 3.5 gol',
            'over_0.5_ht': 'Over 0.5 Primo Tempo',
            'over_1.5_ht': 'Over 1.5 Primo Tempo',
            'over_0.5_second_half': 'Over 0.5 Secondo Tempo',
            'over_1.5_second_half': 'Over 1.5 Secondo Tempo',
            'under_0.5': 'Under 0.5 gol',
            'under_1.5': 'Under 1.5 gol',
            'under_2.5': 'Under 2.5 gol',
            'under_3.5': 'Under 3.5 gol',
            'under_0.5_ht': 'Under 0.5 Primo Tempo',
            'under_1.5_ht': 'Under 1.5 Primo Tempo',
            'btts_yes': 'Entrambe segnano (BTTS)',
            'btts_no': 'Non entrambe segnano',
            'match_winner': 'Esito finale (1X2)',
            'ht_ft_home_home': 'HT/FT Casa-Casa',
            'ht_ft_away_away': 'HT/FT Trasferta-Trasferta',
            '1x': 'Doppia Chance 1X',
            'x2': 'Doppia Chance X2',
            '12': 'Doppia Chance 12',
            'dnb_home': 'Draw No Bet Casa',
            'dnb_away': 'Draw No Bet Trasferta',
            'next_goal_before_75': 'Prossimo gol prima del 75\'',
            'next_goal_after_75': 'Prossimo gol dopo il 75\'',
            'win_to_nil_home': 'Vittoria senza subire (Casa)',
            'win_to_nil_away': 'Vittoria senza subire (Trasferta)',
            'home_win_to_nil': 'Vittoria senza subire (Casa)',
            'away_win_to_nil': 'Vittoria senza subire (Trasferta)',
        }
        
        # 🆕 Inizializza LiveMatchAI dedicata esclusivamente ai match live
        self.live_match_ai = None
        if LIVE_MATCH_AI_AVAILABLE:
            try:
                self.live_match_ai = LiveMatchAI(
                    ai_pipeline=ai_pipeline,
                    min_confidence=min_confidence,
                    min_ev=5.0  # EV minimo per segnali
                )
                logger.info("✅ LiveMatchAI inizializzata - analisi AI dedicata ai match live attiva")
            except Exception as e:
                logger.warning(f"⚠️  Errore inizializzazione LiveMatchAI: {e} - utilizzerò analisi AI base")
        
        self.monitored_matches: Dict[str, Dict] = {}
        self.last_analysis: Dict[str, datetime] = {}
        
        # Leghe/categorie da escludere (SOLO giovanili e riserve - campionati minori OK se hanno dati validi)
        self.excluded_leagues_keywords = [
            'U17', 'U19', 'U21', 'U23', 'Youth', 'Junior', 'Giovanil',
            'Reserve', 'B Team', 'Second Team', 'Academy',
            # 'Women', 'Feminine', 'Femminile'  # RIMOSSO: Permettiamo Champions League femminile ed Europa Cup Women
        ]
        
        # Tornei femminili importanti da includere (eccezioni al filtro generale)
        self.allowed_women_tournaments = [
            'Champions League', 'UEFA Champions League', 'Champions League Women',
            'Europa Cup', 'Europa League', 'Europa Cup Women', 'Europa League Women',
            'UEFA Women', 'Women Champions', 'Women Europa'
        ]
        
        # NOTA: Campionati minori (Serie D, Division 3, ecc.) sono ACCETTATI se hanno dati live sufficienti
        # Il filtro _has_sufficient_live_data farà la scrematura basata sulla qualità dei dati
        # 🆕 AUMENTATE: Confidence minima specifica per mercato (aumentate per ridurre segnali banali)
        self.market_min_confidence: Dict[str, float] = {
            'over_0.5': 75.0,  # 🆕 AUMENTATO: 75% invece di 72%
            'over_0.5_ht': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'over_1.5': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'over_1.5_ht': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'over_2.5': 78.0,  # 🆕 AUMENTATO: 78% invece di 75%
            'over_3.5': 79.0,  # 🆕 AUMENTATO: 79% invece di 76%
            'under_0.5': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'under_0.5_ht': 78.0,  # 🆕 AUMENTATO: 78% invece di 75%
            'under_1.5': 78.0,  # 🆕 AUMENTATO: 78% invece di 75%
            'under_1.5_ht': 78.0,  # 🆕 AUMENTATO: 78% invece di 75%
            'under_2.5': 79.0,  # 🆕 AUMENTATO: 79% invece di 76%
            'under_3.5': 80.0,  # 🆕 AUMENTATO: 80% invece di 78%
            'exact_score': 82.0,  # 🆕 AUMENTATO: 82% invece di 80%
            'goal_range_': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'dnb_': 78.0,  # 🆕 AUMENTATO: 78% invece di 75%
            'clean_sheet': 80.0,  # 🆕 AUMENTATO: 80% invece di 78%
            'team_to_score_next': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'total_goals_odd': 78.0,  # 🆕 AUMENTATO: 78% invece di 75%
            'total_goals_even': 78.0,  # 🆕 AUMENTATO: 78% invece di 75%
            # 'asian_handicap': 75.0,  # 🆕 RIMOSSO: non interessano all'utente
            'match_winner': 78.0,  # 🆕 AUMENTATO: 78% invece di 75%
            'ht_ft': 78.0,  # 🆕 AUMENTATO: 78% invece di 75%
            'next_goal': 78.0,  # 🆕 AUMENTATO: 78% invece di 75%
            'btts': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'win_to_nil': 79.0,  # 🆕 AUMENTATO: 79% invece di 76%
            'corner': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'card': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            # 🆕 NUOVI MERCATI
            'team_to_score_first': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'team_to_score_last': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'highest_scoring_half': 78.0,  # 🆕 AUMENTATO: 78% invece di 75%
            'win_either_half': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'btts_first_half': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
            'half_time_result': 76.0,  # 🆕 AUMENTATO: 76% invece di 73%
        }
    
    def analyze_live_match(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Optional[Dict[str, Any]] = None
    ) -> List[LiveBettingOpportunity]:
        """
        Analizza partita live e suggerisce opportunità.
        
        Args:
            match_id: ID partita
            match_data: Dati partita base
            live_data: Dati live (score, minuto, eventi, ecc.)
        
        Returns:
            Lista di opportunità trovate
        """
        opportunities = []
        
        try:
            # 🆕 FILTRO PRELIMINARE: Escludi partite giovanili/minori/inutili
            if not self._is_match_worth_analyzing(match_data):
                logger.debug(f"⏭️  Partita saltata (giovanile/minore): {match_data.get('home', '?')} vs {match_data.get('away', '?')}")
                return opportunities
            
            # Se non abbiamo dati live, prova a ottenerli
            if not live_data:
                live_data = self._get_live_data(match_id, match_data)
            
            if not live_data:
                return opportunities
            
            # 🆕 FILTRO: Verifica qualità dati live
            if not self._has_sufficient_live_data(live_data):
                logger.debug(f"⏭️  Partita saltata (dati live insufficienti): {match_data.get('home', '?')} vs {match_data.get('away', '?')}")
                return opportunities
            
            # 🆕 OTTIMIZZATO: Verifica status partita (escludi sospese/interrotte)
            status = str(live_data.get('status', '')).lower()
            if any(keyword in status for keyword in ['suspended', 'interrupted', 'abandoned', 'postponed', 'cancelled']):
                logger.debug(f"⏭️  Partita saltata (status: {status}): {match_data.get('home', '?')} vs {match_data.get('away', '?')}")
                return opportunities
            
            # Analizza diverse situazioni con più mercati
            opportunities.extend(self._check_ribaltone_opportunity(match_id, match_data, live_data))
            opportunities.extend(self._check_under_over_opportunity(match_id, match_data, live_data))
            opportunities.extend(self._check_next_goal_opportunity(match_id, match_data, live_data))
            opportunities.extend(self._check_comeback_opportunity(match_id, match_data, live_data))
            opportunities.extend(self._check_ht_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_double_chance_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_over_under_markets(match_id, match_data, live_data))
            
            # 🆕 NUOVO: Mercati avanzati
            opportunities.extend(self._check_corner_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_card_markets(match_id, match_data, live_data))
            # 🆕 RIMOSSO: Asian Handicap markets (non interessano all'utente)
            # opportunities.extend(self._check_handicap_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_btts_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_win_to_nil_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_second_half_markets(match_id, match_data, live_data))
            
            # 🆕 NUOVO: Mercati aggiuntivi completi
            opportunities.extend(self._check_draw_no_bet_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_odd_even_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_exact_score_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_goal_range_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_team_to_score_next_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_clean_sheet_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_ht_ft_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_match_winner_markets(match_id, match_data, live_data))
            # 🆕 RIMOSSO: Asian Handicap markets (non interessano all'utente)
            # opportunities.extend(self._check_asian_handicap_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_time_of_next_goal_markets(match_id, match_data, live_data))
            
            # 🆕 NUOVI MERCATI: Aggiunti mercati utili con filtri anti-ovvietà
            opportunities.extend(self._check_team_to_score_first_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_team_to_score_last_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_highest_scoring_half_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_win_either_half_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_btts_first_half_markets(match_id, match_data, live_data))
            opportunities.extend(self._check_half_time_result_markets(match_id, match_data, live_data))
            
            # 🆕 NUOVO: Usa IA per analizzare e migliorare le opportunità (sempre attivo)
            opportunities = self._enhance_with_ai(opportunities, match_data, live_data)
            
            # 🆕 NUOVO: Aggiungi statistiche dettagliate a ogni opportunità
            for opp in opportunities:
                self._populate_opportunity_metadata(opp, live_data)
            
            # 🆕 FILTRI INTELLIGENTI: Rimuovi suggerimenti banali/ovvi
            opportunities = self._filter_obvious_opportunities(opportunities, live_data)
            opportunities = self._apply_market_specific_rules(opportunities, match_data, live_data)
            opportunities = self._apply_market_min_confidence(opportunities)
            
            # 🆕 OTTIMIZZATO: Filtra solo opportunità con EV molto negativo (non tutte quelle negative)
            opportunities = self._filter_by_expected_value(opportunities)
            
            # Filtra solo opportunità con alta confidence
            before_confidence_filter = len(opportunities)
            opportunities = [opp for opp in opportunities if opp.confidence >= self.min_confidence]
            after_confidence_filter = len(opportunities)
            if before_confidence_filter > 0:
                logger.info(f"📊 LiveBettingAdvisor: {before_confidence_filter} opportunità trovate, {after_confidence_filter} passano filtro confidence (min: {self.min_confidence}%)")
                if before_confidence_filter > after_confidence_filter:
                    # Log confidence delle opportunità filtrate
                    filtered_opps = [opp for opp in opportunities if opp.confidence < self.min_confidence] if before_confidence_filter > after_confidence_filter else []
                    if filtered_opps:
                        confidences = [f"{opp.confidence:.0f}%" for opp in filtered_opps[:5]]  # Prime 5
                        logger.debug(f"   Confidence filtrate: {', '.join(confidences)}")
            
            # 🆕 OTTIMIZZATO: Deduplica opportunità per match_id + market (PRIMA del limite)
            opportunities = self._deduplicate_opportunities(opportunities)
            
            # 🆕 OTTIMIZZATO: Filtra segnali contrastanti (es. Under + Ribaltone sulla stessa partita)
            opportunities = self._filter_contradictory_signals(opportunities, live_data)
            
            # 🆕 OTTIMIZZATO: Ordina per mix di Expected Value e Confidence (non solo EV)
            opportunities.sort(key=lambda x: self._calculate_combined_score(x), reverse=True)
            
            # 🆕 FIX CRITICO: Filtro finale di sicurezza - blocca TUTTI i segnali con confidence < min_confidence
            # Questo è un doppio controllo per essere sicuri che nessun segnale con confidence troppo bassa venga inviato
            before_final_filter = opportunities.copy()  # Salva copia prima del filtro
            opportunities = [opp for opp in opportunities if opp.confidence >= self.min_confidence]
            if len(before_final_filter) > len(opportunities):
                filtered_count = len(before_final_filter) - len(opportunities)
                logger.warning(f"⚠️  FILTRO FINALE: Bloccati {filtered_count} segnali con confidence < {self.min_confidence}% (BUG PREVENZIONE)")
                # Log dettagli dei segnali bloccati per debug
                for opp in before_final_filter:
                    if opp.confidence < self.min_confidence:
                        logger.warning(f"   ❌ Segnale bloccato: {opp.market} su {opp.match_id} con confidence {opp.confidence:.1f}% < {self.min_confidence}%")
            
            # 🆕 FILTRO: Limita numero di segnali per partita (max 2 migliori) E deduplica di nuovo
            # Raggruppa per match_id e mantieni solo i 2 migliori per partita
            opportunities = self._limit_and_deduplicate_per_match(
                opportunities,
                max_per_match=self.max_opportunities_per_match
            )
            
            # 🆕 FIX CRITICO: Filtro finale per bloccare home_win + away_win sulla stessa partita
            # Dopo tutti i filtri, verifica che non ci siano segnali contraddittori rimasti
            by_match_final = {}
            for opp in opportunities:
                match_id = opp.match_id
                if match_id not in by_match_final:
                    by_match_final[match_id] = []
                by_match_final[match_id].append(opp)
            
            final_opportunities = []
            for match_id, match_opps in by_match_final.items():
                # Se ci sono home_win e away_win, mantieni solo quello con confidence più alta
                home_wins = [o for o in match_opps if 'home_win' in o.market.lower() or '1x2_home' in o.market.lower()]
                away_wins = [o for o in match_opps if 'away_win' in o.market.lower() or '1x2_away' in o.market.lower()]
                
                if home_wins and away_wins:
                    # Mantieni solo il migliore tra i due
                    all_wins = home_wins + away_wins
                    all_wins.sort(key=lambda x: x.confidence, reverse=True)
                    best = all_wins[0]
                    logger.warning(f"⚠️  BLOCCATO segnale contraddittorio: {match_id} aveva sia home_win che away_win, mantenuto solo {best.market} (confidence: {best.confidence:.1f}%)")
                    # Rimuovi tutti i win signals e aggiungi solo il migliore
                    match_opps = [o for o in match_opps if not ('home_win' in o.market.lower() or 'away_win' in o.market.lower() or '1x2_home' in o.market.lower() or '1x2_away' in o.market.lower())]
                    match_opps.append(best)
                
                final_opportunities.extend(match_opps)
            
            opportunities = final_opportunities
            
        except Exception as e:
            logger.error(f"❌ Errore analisi live match {match_id}: {e}")
        
        return opportunities
    
    def _get_live_data(
        self,
        match_id: str,
        match_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Ottiene dati live per una partita.
        Per ora usa dati simulati, da integrare con API reali.
        """
        # TODO: Integrare con API-Football o altre API per dati live reali
        # Per ora ritorna None (da implementare)
        return None
    
    def _check_ribaltone_opportunity(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità "ribaltone":
        - Favorita perde → punta vittoria favorita
        """
        opportunities = []
        
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            
            # Determina favorita (basata su quote iniziali)
            odds_1 = match_data.get('odds_1', 2.0)
            odds_2 = match_data.get('odds_2', 2.0)
            
            is_home_favorite = odds_1 < odds_2
            
            # Situazione: favorita perde
            if is_home_favorite and score_home < score_away:
                # 🆕 FILTRO: Non generare ribaltone se differenza >= 2 gol (es. 0-2, 1-3, 3-0, etc.)
                goal_diff = score_away - score_home
                if goal_diff >= 2:
                    logger.debug(f"⏭️  Ribaltone saltato: differenza troppo alta ({score_home}-{score_away}, diff: {goal_diff} gol)")
                    return opportunities
                
                # Favorita in casa perde
                if minute >= 30 and minute <= 75:  # Tra 30' e 75'
                    # 🆕 OTTIMIZZATO: Aumentata confidence base per ribaltone (50% → 60%)
                    confidence = min(85, 60 + (minute - 30) * 0.5)  # Più tardi = più confidence
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id,
                        match_data=match_data,
                        situation='ribaltone_favorita',
                        market='1x2_home',
                        recommendation=f"Punta {match_data.get('home')} vince (ribaltone)",
                        reasoning=(
                            f"🎯 RIBALTONE OPPORTUNITY!\n\n"
                            f"• {match_data.get('home')} (favorita) perde {score_home}-{score_away}\n"
                            f"• Minuto: {minute}'\n"
                            f"• La favorita ha ancora tempo per ribaltare\n"
                            f"• Quote probabilmente aumentate → buon valore\n"
                            f"• Pattern storico: favorita in svantaggio spesso recupera"
                        ),
                        confidence=confidence,
                        odds=match_data.get('odds_1', 2.0),
                        stake_suggestion=3.0,  # 3% bankroll
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            elif not is_home_favorite and score_away < score_home:
                # 🆕 FILTRO: Non generare ribaltone se differenza >= 2 gol (es. 2-0, 3-0, 4-1, etc.)
                goal_diff = score_home - score_away
                if goal_diff >= 2:
                    logger.debug(f"⏭️  Ribaltone saltato: differenza troppo alta ({score_home}-{score_away}, diff: {goal_diff} gol)")
                    return opportunities
                
                # Favorita in trasferta perde
                if minute >= 30 and minute <= 75:
                    # 🆕 OTTIMIZZATO: Aumentata confidence base per ribaltone (50% → 60%)
                    confidence = min(85, 60 + (minute - 30) * 0.5)
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id,
                        match_data=match_data,
                        situation='ribaltone_favorita',
                        market='1x2_away',
                        recommendation=f"Punta {match_data.get('away')} vince (ribaltone)",
                        reasoning=(
                            f"🎯 RIBALTONE OPPORTUNITY!\n\n"
                            f"• {match_data.get('away')} (favorita) perde {score_away}-{score_home}\n"
                            f"• Minuto: {minute}'\n"
                            f"• La favorita ha ancora tempo per ribaltare\n"
                            f"• Quote probabilmente aumentate → buon valore"
                        ),
                        confidence=confidence,
                        odds=match_data.get('odds_2', 2.0),
                        stake_suggestion=3.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        
        except Exception as e:
            logger.debug(f"⚠️  Errore check ribaltone: {e}")
        
        return opportunities
    
    def _check_under_over_opportunity(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Under/Over:
        - Gol subito → Under se partita chiusa
        - Nessun gol → Over se partita aperta
        """
        opportunities = []
        
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            total_goals = score_home + score_away
            
            # Situazione: Gol subito (primi 15 minuti)
            if minute <= 15 and total_goals >= 1:
                # Gol subito → se partita sembra chiusa, punta Under
                # (da implementare analisi pattern partita)
                confidence = 70
                
                opportunity = LiveBettingOpportunity(
                    match_id=match_id,
                    match_data=match_data,
                    situation='under_early_goal',
                    market='under_2.5',
                    recommendation="Punta Under 2.5 (gol subito)",
                    reasoning=(
                        f"🎯 UNDER OPPORTUNITY!\n\n"
                        f"• Gol segnato nei primi {minute} minuti\n"
                        f"• Score: {score_home}-{score_away}\n"
                        f"• Pattern: gol precoci spesso portano a partite più chiuse\n"
                        f"• Quote Under probabilmente aumentate → buon valore"
                    ),
                    confidence=confidence,
                    odds=1.8,  # Stima, da ottenere da API
                    stake_suggestion=2.5,
                    timestamp=datetime.now()
                )
                opportunities.append(opportunity)
            
            # Situazione: Nessun gol (primi 30 minuti)
            elif minute >= 25 and minute <= 35 and total_goals == 0:
                # Nessun gol → se partita sembra aperta, punta Over
                confidence = 65
                
                opportunity = LiveBettingOpportunity(
                    match_id=match_id,
                    match_data=match_data,
                    situation='over_no_goals',
                    market='over_1.5',
                    recommendation="Punta Over 1.5 (nessun gol ma partita aperta)",
                    reasoning=(
                        f"🎯 OVER OPPORTUNITY!\n\n"
                        f"• Nessun gol dopo {minute} minuti\n"
                        f"• Partita sembra aperta\n"
                        f"• Pattern: partite senza gol iniziali spesso si aprono dopo\n"
                        f"• Quote Over probabilmente buone"
                    ),
                    confidence=confidence,
                    odds=1.6,  # Stima
                    stake_suggestion=2.0,
                    timestamp=datetime.now()
                )
                opportunities.append(opportunity)
        
        except Exception as e:
            logger.debug(f"⚠️  Errore check under/over: {e}")
        
        return opportunities
    
    def _check_next_goal_opportunity(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità "Prossimo Gol":
        - Squadra in svantaggio → probabile prossimo gol
        🆕 FIX: Non suggerire "2 della sfavorita" se la favorita sta vincendo
        """
        opportunities = []
        
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            
            # 🆕 FIX: Determina favorita basandosi sulle quote
            odds_1 = match_data.get('odds_1', 2.0)
            odds_2 = match_data.get('odds_2', 2.0)
            is_home_favorite = odds_1 < odds_2
            
            # Situazione: Una squadra in svantaggio
            if score_home != score_away and minute >= 20 and minute <= 70:
                if score_home < score_away:
                    # 🆕 FIX: Home in svantaggio - verifica se è la favorita o la sfavorita
                    # Se la favorita sta vincendo (away), NON suggerire "2 della sfavorita" (home)
                    if is_home_favorite:
                        # Home è favorita ma sta perdendo → OK suggerire prossimo gol home
                        confidence = 70
                        
                        opportunity = LiveBettingOpportunity(
                            match_id=match_id,
                            match_data=match_data,
                            situation='next_goal_underdog',
                            market='next_goal_home',
                            recommendation=f"Punta {match_data.get('home')} (favorita) segna prossimo gol",
                            reasoning=(
                                f"🎯 PROSSIMO GOL OPPORTUNITY!\n\n"
                                f"• {match_data.get('home')} (favorita) in svantaggio {score_home}-{score_away}\n"
                                f"• Minuto: {minute}'\n"
                                f"• La favorita in svantaggio spinge per pareggiare\n"
                                f"• Alta probabilità prossimo gol dalla favorita"
                            ),
                            confidence=confidence,
                            odds=2.2,  # Stima
                            stake_suggestion=2.5,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
                    # Se home NON è favorita (è la sfavorita), OK suggerire prossimo gol home
                    # perché la sfavorita in svantaggio spinge per pareggiare
                    elif not is_home_favorite:
                        confidence = 70
                        
                        opportunity = LiveBettingOpportunity(
                            match_id=match_id,
                            match_data=match_data,
                            situation='next_goal_underdog',
                            market='next_goal_home',
                            recommendation=f"Punta {match_data.get('home')} (sfavorita) segna prossimo gol",
                            reasoning=(
                                f"🎯 PROSSIMO GOL OPPORTUNITY!\n\n"
                                f"• {match_data.get('home')} (sfavorita) in svantaggio {score_home}-{score_away}\n"
                                f"• Minuto: {minute}'\n"
                                f"• Squadra in svantaggio spinge per pareggiare\n"
                                f"• Alta probabilità prossimo gol dalla squadra in svantaggio"
                            ),
                            confidence=confidence,
                            odds=2.2,  # Stima
                            stake_suggestion=2.5,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
                
                elif score_away < score_home:
                    # 🆕 FIX: Away in svantaggio - verifica se è la favorita o la sfavorita
                    # Se la favorita sta vincendo (home), NON suggerire "1 della sfavorita" (away)
                    if not is_home_favorite:
                        # Away è favorita ma sta perdendo → OK suggerire prossimo gol away
                        confidence = 70
                        
                        opportunity = LiveBettingOpportunity(
                            match_id=match_id,
                            match_data=match_data,
                            situation='next_goal_underdog',
                            market='next_goal_away',
                            recommendation=f"Punta {match_data.get('away')} (favorita) segna prossimo gol",
                            reasoning=(
                                f"🎯 PROSSIMO GOL OPPORTUNITY!\n\n"
                                f"• {match_data.get('away')} (favorita) in svantaggio {score_away}-{score_home}\n"
                                f"• Minuto: {minute}'\n"
                                f"• La favorita in svantaggio spinge per pareggiare\n"
                                f"• Alta probabilità prossimo gol dalla favorita"
                            ),
                            confidence=confidence,
                            odds=2.2,  # Stima
                            stake_suggestion=2.5,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
                    # Se away NON è favorita (è la sfavorita), OK suggerire prossimo gol away
                    elif is_home_favorite:
                        confidence = 70
                        
                        opportunity = LiveBettingOpportunity(
                            match_id=match_id,
                            match_data=match_data,
                            situation='next_goal_underdog',
                            market='next_goal_away',
                            recommendation=f"Punta {match_data.get('away')} (sfavorita) segna prossimo gol",
                            reasoning=(
                                f"🎯 PROSSIMO GOL OPPORTUNITY!\n\n"
                                f"• {match_data.get('away')} (sfavorita) in svantaggio {score_away}-{score_home}\n"
                                f"• Minuto: {minute}'\n"
                                f"• Squadra in svantaggio spinge per pareggiare\n"
                                f"• Alta probabilità prossimo gol dalla squadra in svantaggio"
                            ),
                            confidence=confidence,
                            odds=2.2,  # Stima
                            stake_suggestion=2.5,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
        
        except Exception as e:
            logger.debug(f"⚠️  Errore check next goal: {e}")
        
        return opportunities
    
    def _check_comeback_opportunity(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità "Comeback":
        - Squadra perde ma sta dominando → punta pareggio/vittoria
        """
        opportunities = []
        
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            possession_home = live_data.get('possession_home', 50)  # %
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            
            # Situazione: Home perde ma domina
            if score_home < score_away and minute >= 30 and minute <= 70:
                if possession_home > 60 and shots_home > shots_away * 1.5:
                    # Domina ma perde → probabile recupero
                    confidence = 75
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id,
                        match_data=match_data,
                        situation='comeback_dominance',
                        market='1x',
                        recommendation=f"Punta {match_data.get('home')} pareggio o vittoria",
                        reasoning=(
                            f"🎯 COMEBACK OPPORTUNITY!\n\n"
                            f"• {match_data.get('home')} perde {score_home}-{score_away}\n"
                            f"• Ma DOMINA: {possession_home}% possesso, {shots_home} vs {shots_away} tiri\n"
                            f"• Minuto: {minute}'\n"
                            f"• Pattern: squadra che domina spesso recupera\n"
                            f"• Buon valore sulle quote"
                        ),
                        confidence=confidence,
                        odds=2.5,  # Stima
                        stake_suggestion=3.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        
        except Exception as e:
            logger.debug(f"⚠️  Errore check comeback: {e}")
        
        return opportunities
    
    def _check_ht_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità mercati Primo Tempo (HT) - MIGLIORATO CON IA
        
        Mercati:
        - Over 0.5 HT
        - Over 1.5 HT
        - Under 0.5 HT
        - Under 1.5 HT
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            total_goals = score_home + score_away
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            total_shots = shots_home + shots_away
            total_shots_on_target = shots_on_target_home + shots_on_target_away
            
            # 🆕 FIX: Solo se siamo nel primo tempo (controllo rigoroso: minuto < 45 E non siamo nel secondo tempo)
            # Verifica anche che non siamo già nel secondo tempo inoltrato (minuto >= 45 significa secondo tempo)
            if minute < 45 and minute > 0:
                # OVER 0.5 HT: Nessun gol ma partita aperta
                if total_goals == 0 and minute >= 15 and minute <= 40:
                    # Analisi avanzata: partita aperta?
                    shots_per_minute = total_shots / minute if minute > 0 else 0
                    shots_on_target_per_minute = total_shots_on_target / minute if minute > 0 else 0
                    
                    # Se partita aperta (tiri frequenti)
                    if shots_per_minute > 0.3 and shots_on_target_per_minute > 0.1:
                        ai_boost = self._get_ai_market_confidence(match_data, live_data, 'over_0.5_ht') if self.ai_pipeline else 0
                        # Confidence aumenta con minuto e tiri
                        # 🆕 OTTIMIZZATO: Aumentata confidence base per mercato rischioso
                        base_confidence = 70 + (minute - 15) * 0.5 + min(10, total_shots_on_target * 2)
                        confidence = min(88, base_confidence + ai_boost)
                        
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='over_0.5_ht', market='over_0.5_ht',
                        recommendation="Punta Over 0.5 Primo Tempo",
                            reasoning=(
                                f"🎯 OVER 0.5 HT!\n\n"
                                f"• Score: {score_home}-{score_away} al {minute}'\n"
                                f"• Partita APERTA:\n"
                                f"  - Tiri: {total_shots} ({total_shots_on_target} in porta)\n"
                                f"  - Media: {shots_per_minute:.2f} tiri/min\n"
                                f"• Alta probabilità gol nel primo tempo\n"
                                f"• IA boost: +{ai_boost:.0f}%"
                            ),
                            confidence=confidence, odds=1.5, stake_suggestion=2.5,
                        timestamp=datetime.now(),
                            alternative_markets=[
                                {'market': 'over_1.5_ht', 'confidence': confidence - 15, 'odds': 2.2}
                            ]
                    )
                    opportunities.append(opportunity)
                
                # OVER 1.5 HT: Già 1 gol, probabile secondo
                elif total_goals == 1 and minute >= 20 and minute <= 40:
                    # Analisi: partita ancora aperta?
                    if total_shots >= 8 and total_shots_on_target >= 3:
                        ai_boost = self._get_ai_market_confidence(match_data, live_data, 'over_1.5_ht') if self.ai_pipeline else 0
                        # Confidence aumenta con tiri e minuto
                        base_confidence = 65 + (minute - 20) * 0.4 + min(10, total_shots_on_target * 2)
                        confidence = min(88, base_confidence + ai_boost)
                        
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='over_1.5_ht', market='over_1.5_ht',
                        recommendation="Punta Over 1.5 Primo Tempo",
                            reasoning=(
                                f"🎯 OVER 1.5 HT!\n\n"
                                f"• Score: {score_home}-{score_away} al {minute}'\n"
                                f"• Già 1 gol, partita ancora APERTA:\n"
                                f"  - Tiri: {total_shots} ({total_shots_on_target} in porta)\n"
                                f"• Alta probabilità secondo gol nel primo tempo\n"
                                f"• IA boost: +{ai_boost:.0f}%"
                            ),
                            confidence=confidence, odds=2.2, stake_suggestion=3.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                
                # UNDER 0.5 HT: Nessun gol e partita chiusa
                # 🆕 OTTIMIZZATO: Non generare se siamo oltre 40' (troppo banale al 44')
                elif total_goals == 0 and minute >= 30 and minute <= 40:  # Ridotto da 44' a 40'
                    # Analisi: partita chiusa?
                    shots_per_minute = total_shots / minute if minute > 0 else 0
                    shots_on_target_per_minute = total_shots_on_target / minute if minute > 0 else 0
                    
                    # Se partita chiusa (pochi tiri)
                    if shots_per_minute < 0.2 and shots_on_target_per_minute < 0.05:
                        ai_boost = self._get_ai_market_confidence(match_data, live_data, 'under_0.5_ht') if self.ai_pipeline else 0
                        # Confidence aumenta con minuto avanzato
                        base_confidence = 70 + (minute - 30) * 0.8
                        confidence = min(90, base_confidence + ai_boost)
                        
                        opportunity = LiveBettingOpportunity(
                            match_id=match_id, match_data=match_data,
                            situation='under_0.5_ht', market='under_0.5_ht',
                            recommendation="Punta Under 0.5 Primo Tempo",
                            reasoning=(
                                f"🎯 UNDER 0.5 HT!\n\n"
                                f"• Score: {score_home}-{score_away} al {minute}'\n"
                                f"• Partita CHIUSA:\n"
                                f"  - Tiri: {total_shots} ({total_shots_on_target} in porta)\n"
                                f"  - Media: {shots_per_minute:.2f} tiri/min (bassa)\n"
                                f"• Alta probabilità 0-0 al primo tempo\n"
                                f"• IA boost: +{ai_boost:.0f}%"
                            ),
                            confidence=confidence, odds=2.5, stake_suggestion=2.0,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
                
                # UNDER 1.5 HT: Massimo 1 gol
                # 🆕 OTTIMIZZATO: Non generare se siamo oltre 40' (troppo banale al 44')
                elif total_goals <= 1 and minute >= 35 and minute <= 40:  # Ridotto da 44' a 40'
                    # Analisi: partita chiusa?
                    shots_per_minute = total_shots / minute if minute > 0 else 0
                    if shots_per_minute < 0.25:
                        ai_boost = self._get_ai_market_confidence(match_data, live_data, 'under_1.5_ht') if self.ai_pipeline else 0
                        base_confidence = 75 + (minute - 35) * 0.5
                        confidence = min(92, base_confidence + ai_boost)
                        
                        opportunity = LiveBettingOpportunity(
                            match_id=match_id, match_data=match_data,
                            situation='under_1.5_ht', market='under_1.5_ht',
                            recommendation="Punta Under 1.5 Primo Tempo",
                            reasoning=(
                                f"🎯 UNDER 1.5 HT!\n\n"
                                f"• Score: {score_home}-{score_away} al {minute}'\n"
                                f"• Partita CHIUSA:\n"
                                f"  - Tiri: {total_shots} (media: {shots_per_minute:.2f}/min)\n"
                                f"• Alta probabilità max 1 gol al primo tempo\n"
                                f"• IA boost: +{ai_boost:.0f}%"
                            ),
                            confidence=confidence, odds=1.8, stake_suggestion=2.5,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
                        
        except Exception as e:
            logger.debug(f"⚠️  Errore check HT markets: {e}")
        return opportunities
    
    def _check_double_chance_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Doppia Chance (1X, X2) - SOLO SE C'È VALORE REALE
        
        NON suggerisce 1X se è già 1-0 (banale!)
        Suggerisce solo se:
        - Favorita perde ma domina (ribaltone)
        - Pareggio ma una squadra domina nettamente
        - Situazioni con valore reale, non ovvie
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            possession_home = live_data.get('possession_home', 50)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            
            # Determina favorita
            odds_1 = match_data.get('odds_1', 2.0)
            odds_2 = match_data.get('odds_2', 2.0)
            is_home_favorite = odds_1 < odds_2
            
            # SITUAZIONE 1: Favorita perde ma domina (ribaltone con 1X)
            if is_home_favorite and score_home < score_away and minute >= 40 and minute <= 70:
                # Favorita in casa perde ma domina
                if possession_home > 60 and shots_home > shots_away * 1.3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, '1x') if self.ai_pipeline else 0
                    confidence = 75 + ai_boost  # Alta confidence solo se domina
                    
                opportunity = LiveBettingOpportunity(
                    match_id=match_id, match_data=match_data,
                        situation='double_chance_1x_comeback', market='1x',
                        recommendation=f"Punta 1X - {match_data.get('home')} (favorita) perde ma DOMINA",
                        reasoning=(
                            f"🎯 1X CON VALORE!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('home')} (favorita) perde ma DOMINA:\n"
                            f"  - Possesso: {possession_home}%\n"
                            f"  - Tiri: {shots_home} vs {shots_away}\n"
                            f"• Alta probabilità recupero → 1X ha valore\n"
                            f"• NON banale: favorita in svantaggio ma domina"
                        ),
                        confidence=confidence, odds=1.6, stake_suggestion=3.0,
                    timestamp=datetime.now()
                )
                opportunities.append(opportunity)
            
            elif not is_home_favorite and score_away < score_home and minute >= 40 and minute <= 70:
                # Favorita in trasferta perde ma domina
                possession_away = 100 - possession_home
                if possession_away > 60 and shots_away > shots_home * 1.3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'x2') if self.ai_pipeline else 0
                    confidence = 75 + ai_boost
                    
                opportunity = LiveBettingOpportunity(
                    match_id=match_id, match_data=match_data,
                        situation='double_chance_x2_comeback', market='x2',
                        recommendation=f"Punta X2 - {match_data.get('away')} (favorita) perde ma DOMINA",
                        reasoning=(
                            f"🎯 X2 CON VALORE!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('away')} (favorita) perde ma DOMINA:\n"
                            f"  - Possesso: {possession_away}%\n"
                            f"  - Tiri: {shots_away} vs {shots_home}\n"
                            f"• Alta probabilità recupero → X2 ha valore\n"
                            f"• NON banale: favorita in svantaggio ma domina"
                        ),
                        confidence=confidence, odds=1.6, stake_suggestion=3.0,
                    timestamp=datetime.now()
                )
                opportunities.append(opportunity)
            
            # SITUAZIONE 2: Pareggio ma una squadra domina nettamente (solo se quote buone)
            elif score_home == score_away and minute >= 50 and minute <= 75:
                # Pareggio ma home domina nettamente
                if possession_home > 65 and shots_home > shots_away * 1.5:
                    # Solo se quote 1X sono buone (non troppo basse)
                    odds_1x = match_data.get('odds_1x', 1.3)  # Se disponibile
                    if odds_1x >= 1.4:  # Solo se quota decente
                        ai_boost = self._get_ai_market_confidence(match_data, live_data, '1x') if self.ai_pipeline else 0
                        confidence = 72 + ai_boost
                        
                        opportunity = LiveBettingOpportunity(
                            match_id=match_id, match_data=match_data,
                            situation='double_chance_1x_dominance', market='1x',
                            recommendation=f"Punta 1X - {match_data.get('home')} domina nettamente",
                            reasoning=(
                                f"🎯 1X CON VALORE!\n\n"
                                f"• Score: {score_home}-{score_away} (pareggio) al {minute}'\n"
                                f"• {match_data.get('home')} DOMINA nettamente:\n"
                                f"  - Possesso: {possession_home}%\n"
                                f"  - Tiri: {shots_home} vs {shots_away}\n"
                                f"• Alta probabilità che segni → 1X ha valore\n"
                                f"• NON banale: pareggio ma dominio netto"
                            ),
                            confidence=confidence, odds=odds_1x, stake_suggestion=2.5,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
                
                # Pareggio ma away domina nettamente
                elif possession_home < 35 and shots_away > shots_home * 1.5:
                    odds_x2 = match_data.get('odds_x2', 1.3)
                    if odds_x2 >= 1.4:
                        ai_boost = self._get_ai_market_confidence(match_data, live_data, 'x2') if self.ai_pipeline else 0
                        confidence = 72 + ai_boost
                        
                        opportunity = LiveBettingOpportunity(
                            match_id=match_id, match_data=match_data,
                            situation='double_chance_x2_dominance', market='x2',
                            recommendation=f"Punta X2 - {match_data.get('away')} domina nettamente",
                            reasoning=(
                                f"🎯 X2 CON VALORE!\n\n"
                                f"• Score: {score_home}-{score_away} (pareggio) al {minute}'\n"
                                f"• {match_data.get('away')} DOMINA nettamente:\n"
                                f"  - Possesso: {100 - possession_home}%\n"
                                f"  - Tiri: {shots_away} vs {shots_home}\n"
                                f"• Alta probabilità che segni → X2 ha valore\n"
                                f"• NON banale: pareggio ma dominio netto"
                            ),
                            confidence=confidence, odds=odds_x2, stake_suggestion=2.5,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
            
            # NON suggeriamo 1X se è già 1-0 (banale!)
            # NON suggeriamo X2 se è già 0-1 (banale!)
            # Solo situazioni con valore reale
            
        except Exception as e:
            logger.debug(f"⚠️  Errore check double chance: {e}")
        return opportunities
    
    def _check_over_under_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Over/Under multipli - MIGLIORATO CON IA
        
        Mercati:
        - Over 0.5, 1.5, 2.5, 3.5
        - Under 1.5, 2.5, 3.5
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            total_goals = score_home + score_away
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            total_shots = shots_home + shots_away
            total_shots_on_target = shots_on_target_home + shots_on_target_away
            
            # Calcola tasso gol atteso
            goals_per_minute = total_goals / minute if minute > 0 else 0
            expected_goals_final = goals_per_minute * 90 if minute > 0 else 0
            
            # OVER 0.5: Nessun gol ma partita aperta
            if total_goals == 0 and minute >= 20 and minute <= 70:
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute > 0.25 and total_shots_on_target >= 2:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'over_0.5') if self.ai_pipeline else 0
                    base_confidence = 70 + min(10, total_shots_on_target * 3)
                    confidence = min(88, base_confidence + ai_boost)
                    
                    opportunity = LiveBettingOpportunity(
                    match_id=match_id, match_data=match_data,
                    situation='over_0.5_general', market='over_0.5',
                    recommendation="Punta Over 0.5 Gol",
                        reasoning=(
                            f"🎯 OVER 0.5!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Partita APERTA:\n"
                            f"  - Tiri: {total_shots} ({total_shots_on_target} in porta)\n"
                            f"  - Media: {shots_per_minute:.2f} tiri/min\n"
                            f"• Alta probabilità almeno 1 gol\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                    confidence=confidence, odds=1.2, stake_suggestion=2.5,
                    timestamp=datetime.now(),
                    alternative_markets=[
                        {'market': 'over_1.5', 'confidence': confidence - 15, 'odds': 1.6},
                        {'market': 'over_2.5', 'confidence': confidence - 25, 'odds': 2.0}
                    ]
                )
                opportunities.append(opportunity)
            
            # OVER 1.5: Già 1 gol, probabile secondo
            elif total_goals == 1 and minute >= 25 and minute <= 75:
                if total_shots >= 10 and total_shots_on_target >= 3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'over_1.5') if self.ai_pipeline else 0
                    base_confidence = 72 + min(10, total_shots_on_target * 2)
                    confidence = min(90, base_confidence + ai_boost)
                    
                    opportunity = LiveBettingOpportunity(
                    match_id=match_id, match_data=match_data,
                    situation='over_1.5_general', market='over_1.5',
                    recommendation="Punta Over 1.5 Gol",
                        reasoning=(
                            f"🎯 OVER 1.5!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Già 1 gol, partita APERTA:\n"
                            f"  - Tiri: {total_shots} ({total_shots_on_target} in porta)\n"
                            f"• Alta probabilità secondo gol\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.6, stake_suggestion=3.0,
                    timestamp=datetime.now(),
                        alternative_markets=[{'market': 'over_2.5', 'confidence': confidence - 12, 'odds': 2.0}]
                )
                opportunities.append(opportunity)
            
            # OVER 2.5: Già 2 gol o partita molto aperta
            if total_goals == 2 and minute >= 30 and minute <= 75:
                # Già 2 gol, probabile terzo
                if total_shots >= 15:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'over_2.5') if self.ai_pipeline else 0
                    base_confidence = 75 + min(10, (total_shots - 15) * 0.5)
                    confidence = min(92, base_confidence + ai_boost)
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='over_2.5_general', market='over_2.5',
                        recommendation="Punta Over 2.5 Gol",
                        reasoning=(
                            f"🎯 OVER 2.5!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Già 2 gol, partita MOLTO APERTA:\n"
                            f"  - Tiri: {total_shots} ({total_shots_on_target} in porta)\n"
                            f"• Alta probabilità terzo gol\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.0, stake_suggestion=3.0,
                        timestamp=datetime.now(),
                        alternative_markets=[{'market': 'over_3.5', 'confidence': confidence - 20, 'odds': 3.0}]
                    )
                    opportunities.append(opportunity)
            elif total_goals == 1 and minute >= 40 and minute <= 70:
                # Solo 1 gol ma partita molto aperta
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute > 0.4 and total_shots >= 20:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'over_2.5') if self.ai_pipeline else 0
                    base_confidence = 68 + min(12, (total_shots - 20) * 0.3)
                    confidence = min(88, base_confidence + ai_boost)
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='over_2.5_high_tempo', market='over_2.5',
                        recommendation="Punta Over 2.5 Gol (partita molto aperta)",
                        reasoning=(
                            f"🎯 OVER 2.5!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Partita MOLTO APERTA:\n"
                            f"  - Tiri: {total_shots} (media: {shots_per_minute:.2f}/min)\n"
                            f"  - Tiri in porta: {total_shots_on_target}\n"
                            f"• Alta probabilità altri gol → Over 2.5\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.0, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            # OVER 3.5: Già 3 gol o partita estremamente aperta
            if total_goals == 3 and minute >= 40 and minute <= 80:
                ai_boost = self._get_ai_market_confidence(match_data, live_data, 'over_3.5') if self.ai_pipeline else 0
                base_confidence = 70 + min(15, (minute - 40) * 0.3)
                confidence = min(90, base_confidence + ai_boost)
                
                opportunity = LiveBettingOpportunity(
                    match_id=match_id, match_data=match_data,
                    situation='over_3.5_general', market='over_3.5',
                    recommendation="Punta Over 3.5 Gol",
                    reasoning=(
                        f"🎯 OVER 3.5!\n\n"
                        f"• Score: {score_home}-{score_away} al {minute}'\n"
                        f"• Già 3 gol, partita ESTREMAMENTE APERTA\n"
                        f"• Alta probabilità quarto gol\n"
                        f"• IA boost: +{ai_boost:.0f}%"
                    ),
                    confidence=confidence, odds=3.0, stake_suggestion=2.5,
                    timestamp=datetime.now()
                )
                opportunities.append(opportunity)
            
            # UNDER 1.5: Partita chiusa, max 1 gol
            # 🆕 FIX: NON generare Under 1.5 se c'è già 1 gol e siamo oltre 45' (illogico - se è 1-0 al 50', under 1.5 è già perso se segna un altro gol)
            if total_goals == 0 and minute >= 50 and minute <= 80:  # Solo se è 0-0, non se c'è già 1 gol
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute < 0.2 and total_shots < 15:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'under_1.5') if self.ai_pipeline else 0
                    base_confidence = 75 + (minute - 50) * 0.5
                    confidence = min(93, base_confidence + ai_boost)
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='under_1.5_general', market='under_1.5',
                        recommendation="Punta Under 1.5 Gol",
                        reasoning=(
                            f"🎯 UNDER 1.5!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Partita CHIUSA:\n"
                            f"  - Tiri: {total_shots} (media: {shots_per_minute:.2f}/min - bassa)\n"
                            f"• Alta probabilità max 1 gol totale\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.8, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            # UNDER 2.5: Partita chiusa, max 2 gol
            elif total_goals <= 2 and minute >= 60 and minute <= 85:
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute < 0.25 and total_shots < 20:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'under_2.5') if self.ai_pipeline else 0
                    base_confidence = 72 + (minute - 60) * 0.4
                    confidence = min(91, base_confidence + ai_boost)
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='under_2.5_general', market='under_2.5',
                        recommendation="Punta Under 2.5 Gol",
                        reasoning=(
                            f"🎯 UNDER 2.5!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Partita CHIUSA:\n"
                            f"  - Tiri: {total_shots} (media: {shots_per_minute:.2f}/min - bassa)\n"
                            f"• Alta probabilità max 2 gol totale\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.5, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            # UNDER 3.5: Partita chiusa, max 3 gol
            elif total_goals <= 3 and minute >= 70 and minute <= 85:
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute < 0.3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'under_3.5') if self.ai_pipeline else 0
                    base_confidence = 80 + (minute - 70) * 0.5
                    confidence = min(95, base_confidence + ai_boost)
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='under_3.5_general', market='under_3.5',
                        recommendation="Punta Under 3.5 Gol",
                        reasoning=(
                            f"🎯 UNDER 3.5!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Partita CHIUSA:\n"
                            f"  - Tiri: {total_shots} (media: {shots_per_minute:.2f}/min)\n"
                            f"• Alta probabilità max 3 gol totale\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.3, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                    
        except Exception as e:
            logger.debug(f"⚠️  Errore check over/under markets: {e}")
        return opportunities
    
    def _check_corner_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """Rileva opportunità mercati Corner"""
        opportunities = []
        try:
            minute = live_data.get('minute', 0)
            corners_home = live_data.get('corners_home', 0)
            corners_away = live_data.get('corners_away', 0)
            total_corners = corners_home + corners_away
            
            # Over Corner se partita aperta e pochi corner
            if minute >= 30 and minute <= 70 and total_corners < 5:
                # Calcola corner attesi basati su minuto
                expected_corners = (total_corners / minute) * 90 if minute > 0 else 0
                if expected_corners > 8:  # Se trend indica >8 corner
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'over_corners') if self.ai_pipeline else 0
                    confidence = 70 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='corner_over', market='over_8.5_corners',
                        recommendation="Punta Over 8.5 Corner",
                        reasoning=(
                            f"🎯 OVER CORNER OPPORTUNITY!\n\n"
                            f"• Corner attuali: {total_corners} al {minute}'\n"
                            f"• Trend: {expected_corners:.1f} corner attesi a fine partita\n"
                            f"• Partita aperta → più corner attesi"
                        ),
                        confidence=confidence, odds=1.8, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check corner markets: {e}")
        return opportunities
    
    def _check_card_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """Rileva opportunità mercati Cartellini"""
        opportunities = []
        try:
            minute = live_data.get('minute', 0)
            yellow_home = live_data.get('yellow_cards_home', 0)
            yellow_away = live_data.get('yellow_cards_away', 0)
            total_yellows = yellow_home + yellow_away
            
            # Over Cartellini se partita nervosa
            if minute >= 40 and minute <= 75 and total_yellows >= 3:
                # Trend indica molti cartellini
                expected_cards = (total_yellows / minute) * 90 if minute > 0 else 0
                if expected_cards > 5:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'over_cards') if self.ai_pipeline else 0
                    confidence = 65 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='card_over', market='over_5.5_cards',
                        recommendation="Punta Over 5.5 Cartellini",
                        reasoning=(
                            f"🎯 OVER CARTELLINI OPPORTUNITY!\n\n"
                            f"• Cartellini attuali: {total_yellows} gialli al {minute}'\n"
                            f"• Partita nervosa → più cartellini attesi"
                        ),
                        confidence=confidence, odds=1.7, stake_suggestion=1.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check card markets: {e}")
        return opportunities
    
    def _check_handicap_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """Rileva opportunità mercati Handicap"""
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            diff = score_home - score_away
            
            # Handicap se partita sbilanciata
            if minute >= 30 and minute <= 75:
                if diff >= 2:  # Home in vantaggio di 2+
                    # Handicap Away +1.5 o +2.5
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'handicap_away') if self.ai_pipeline else 0
                    confidence = 70 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='handicap_away', market='away_handicap_+1.5',
                        recommendation=f"Punta {match_data.get('away')} Handicap +1.5",
                        reasoning=(
                            f"🎯 HANDICAP OPPORTUNITY!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('away')} in svantaggio ma può recuperare\n"
                            f"• Handicap +1.5 offre buon valore"
                        ),
                        confidence=confidence, odds=1.6, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check handicap markets: {e}")
        return opportunities
    
    def _check_btts_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """Rileva opportunità Both Teams To Score (BTTS)"""
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            
            # BTTS Yes se entrambe hanno segnato o hanno tiri in porta
            # 🆕 FIX: NON generare se siamo oltre 80' (troppo tardi)
            if minute >= 25 and minute <= 80:  # Ridotto a 80' invece di 70'
                if (score_home > 0 and score_away == 0) or (score_home == 0 and score_away > 0):
                    # Una squadra ha segnato, l'altra ha tiri in porta
                    if (score_home > 0 and shots_on_target_away >= 2) or (score_away > 0 and shots_on_target_home >= 2):
                        ai_boost = self._get_ai_market_confidence(match_data, live_data, 'btts_yes') if self.ai_pipeline else 0
                        confidence = 75 + ai_boost
                        
                        opportunity = LiveBettingOpportunity(
                            match_id=match_id, match_data=match_data,
                            situation='btts_yes', market='btts_yes',
                            recommendation="Punta Both Teams To Score (BTTS) - Sì",
                            reasoning=(
                                f"🎯 BTTS OPPORTUNITY!\n\n"
                                f"• Score: {score_home}-{score_away} al {minute}'\n"
                                f"• Una squadra ha segnato, l'altra ha {shots_on_target_home if score_away > 0 else shots_on_target_away} tiri in porta\n"
                                f"• Alta probabilità che anche l'altra squadra segni"
                            ),
                            confidence=confidence, odds=1.9, stake_suggestion=2.5,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check BTTS markets: {e}")
        return opportunities
    
    def _check_win_to_nil_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """Rileva opportunità Win To Nil"""
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            
            # Win To Nil se una squadra vince e l'altra non ha tiri in porta
            # 🆕 FIX: Non generare se è già 2-0 avanzato (banale)
            if minute >= 50 and minute <= 80:
                goal_diff = score_home - score_away
                # Non generare se è già 2-0 o più avanzato (banale)
                if goal_diff >= 2 and minute >= 70:
                    logger.debug(f"⏭️  Win to nil home non generato: risultato {score_home}-{score_away} al {minute}' (troppo avanzato, banale)")
                elif score_home > 0 and score_away == 0 and shots_on_target_away == 0:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'win_to_nil_home') if self.ai_pipeline else 0
                    confidence = 70 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='win_to_nil_home', market='home_win_to_nil',
                        recommendation=f"Punta {match_data.get('home')} Win To Nil",
                        reasoning=(
                            f"🎯 WIN TO NIL OPPORTUNITY!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('home')} in vantaggio, {match_data.get('away')} senza tiri in porta\n"
                            f"• Alta probabilità che mantenga clean sheet"
                        ),
                        confidence=confidence, odds=2.2, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check win to nil markets: {e}")
        return opportunities
    
    def _check_second_half_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """Rileva opportunità Secondo Tempo"""
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            total_goals = score_home + score_away
            
            # 🆕 FIX: Solo se siamo nel secondo tempo (minuto >= 45)
            if minute >= 45 and minute <= 80:
                # 🆕 FIX: Calcola gol del primo tempo (assumendo che al 45' ci fossero X gol)
                # Per semplicità, se non abbiamo dati precisi, stimiamo che i gol del primo tempo
                # siano quelli segnati prima del 45'. Se siamo al 45'-50', probabilmente i gol totali
                # sono ancora quelli del primo tempo. Se siamo oltre 60', dobbiamo stimare.
                # Stima conservativa: se siamo oltre 60' e ci sono 2+ gol, probabilmente almeno 1 è nel secondo tempo
                goals_at_ht = live_data.get('score_home_ht', 0) + live_data.get('score_away_ht', 0)
                if goals_at_ht == 0:
                    # Se non abbiamo il risultato al primo tempo, stimiamo conservativamente
                    # Se siamo oltre 60' e ci sono 2+ gol, probabilmente almeno 1 è nel secondo tempo
                    if minute > 60 and total_goals >= 2:
                        goals_at_ht = total_goals - 1  # Stima conservativa
                    else:
                        goals_at_ht = total_goals  # Se siamo appena entrati nel secondo tempo
                
                goals_in_second_half = total_goals - goals_at_ht
                
                # 🆕 FIX: Over 0.5 Second Half solo se NON ci sono già gol nel secondo tempo
                # E solo se siamo all'inizio del secondo tempo (45'-60')
                if goals_in_second_half == 0 and minute >= 45 and minute <= 60:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'over_0.5_2h') if self.ai_pipeline else 0
                    confidence = 75 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='over_0.5_2h', market='over_0.5_second_half',
                        recommendation="Punta Over 0.5 Secondo Tempo",
                        reasoning=(
                            f"🎯 OVER 0.5 2H OPPORTUNITY!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Primo tempo chiuso, secondo tempo spesso più aperto\n"
                            f"• Alta probabilità almeno 1 gol nel secondo tempo"
                        ),
                        confidence=confidence, odds=1.4, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check second half markets: {e}")
        return opportunities
    
    def _check_draw_no_bet_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Draw No Bet (DNB)
        - Squadra favorita in svantaggio ma domina
        - Pareggio ma una squadra domina nettamente
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            possession_home = live_data.get('possession_home', 50)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            
            odds_1 = match_data.get('odds_1', 2.0)
            odds_2 = match_data.get('odds_2', 2.0)
            is_home_favorite = odds_1 < odds_2
            
            # DNB Home: Favorita in casa perde ma domina
            if is_home_favorite and score_home < score_away and minute >= 30 and minute <= 75:
                if possession_home > 60 and shots_home > shots_away * 1.5 and shots_on_target_home >= 3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'dnb_home') if self.ai_pipeline else 0
                    confidence = 72 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='dnb_home_comeback', market='dnb_home',
                        recommendation=f"Punta {match_data.get('home')} Draw No Bet (favorita perde ma domina)",
                        reasoning=(
                            f"🎯 DRAW NO BET HOME!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('home')} (favorita) perde ma DOMINA:\n"
                            f"  - Possesso: {possession_home}%\n"
                            f"  - Tiri: {shots_home} vs {shots_away}\n"
                            f"  - Tiri in porta: {shots_on_target_home} vs {shots_on_target_away}\n"
                            f"• Alta probabilità recupero → DNB sicuro\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.8, stake_suggestion=3.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            # DNB Away: Favorita in trasferta perde ma domina
            elif not is_home_favorite and score_away < score_home and minute >= 30 and minute <= 75:
                possession_away = 100 - possession_home
                if possession_away > 60 and shots_away > shots_home * 1.5 and shots_on_target_away >= 3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'dnb_away') if self.ai_pipeline else 0
                    confidence = 72 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='dnb_away_comeback', market='dnb_away',
                        recommendation=f"Punta {match_data.get('away')} Draw No Bet (favorita perde ma domina)",
                        reasoning=(
                            f"🎯 DRAW NO BET AWAY!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('away')} (favorita) perde ma DOMINA:\n"
                            f"  - Possesso: {possession_away}%\n"
                            f"  - Tiri: {shots_away} vs {shots_home}\n"
                            f"  - Tiri in porta: {shots_on_target_away} vs {shots_on_target_home}\n"
                            f"• Alta probabilità recupero → DNB sicuro\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.8, stake_suggestion=3.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                    
        except Exception as e:
            logger.debug(f"⚠️  Errore check draw no bet markets: {e}")
        return opportunities
    
    def _check_odd_even_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Total Goals Odd/Even
        - Analisi pattern partita
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            total_goals = score_home + score_away
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            total_shots = shots_home + shots_away
            
            # Odd: Se gol dispari e partita chiusa
            if total_goals % 2 == 1 and minute >= 60 and minute <= 85:
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute < 0.25:  # Partita chiusa
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'total_goals_odd') if self.ai_pipeline else 0
                    confidence = 75 + (minute - 60) * 0.3 + ai_boost
                    confidence = min(92, confidence)
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='total_goals_odd', market='total_goals_odd',
                        recommendation="Punta Total Goals Dispari",
                        reasoning=(
                            f"🎯 TOTALE GOL DISPARI!\n\n"
                            f"• Score: {score_home}-{score_away} ({total_goals} gol) al {minute}'\n"
                            f"• Partita CHIUSA (tiri/min: {shots_per_minute:.2f})\n"
                            f"• Alta probabilità rimanga dispari\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.9, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            # Even: Se gol pari e partita aperta
            elif total_goals % 2 == 0 and minute >= 40 and minute <= 75:
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute > 0.3 and total_shots >= 15:  # Partita aperta
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'total_goals_even') if self.ai_pipeline else 0
                    confidence = 70 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='total_goals_even', market='total_goals_even',
                        recommendation="Punta Total Goals Pari",
                        reasoning=(
                            f"🎯 TOTALE GOL PARI!\n\n"
                            f"• Score: {score_home}-{score_away} ({total_goals} gol) al {minute}'\n"
                            f"• Partita APERTA (tiri/min: {shots_per_minute:.2f})\n"
                            f"• Alta probabilità altro gol → pari\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.9, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                    
        except Exception as e:
            logger.debug(f"⚠️  Errore check odd/even markets: {e}")
        return opportunities
    
    def _check_exact_score_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Exact Score
        - Partita chiusa con score probabile
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            total_shots = shots_home + shots_away
            
            # Exact Score: Solo se partita molto chiusa e avanzata
            # 🆕 FIX: NON generare se suggerisce lo score attuale (banale)
            if minute >= 75 and minute <= 88:  # Aumentato a 75' per evitare troppo presto
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute < 0.2 and (shots_on_target_home + shots_on_target_away) < 5:
                    # Partita molto chiusa, probabile che rimanga così
                    # 🆕 FIX: NON generare exact score se è già 0-0 o 1-0 (troppo banale)
                    if (score_home == 0 and score_away == 0) or (score_home + score_away == 1):
                        logger.debug(f"⏭️  Non generare Exact Score: troppo banale per {score_home}-{score_away} al {minute}'")
                    else:
                        ai_boost = self._get_ai_market_confidence(match_data, live_data, 'exact_score') if self.ai_pipeline else 0
                        confidence = 70 + (minute - 75) * 0.5 + ai_boost
                        confidence = min(90, confidence)
                        
                        exact_score = f"{score_home}-{score_away}"
                        opportunity = LiveBettingOpportunity(
                            match_id=match_id, match_data=match_data,
                            situation='exact_score', market=f'exact_score_{exact_score}',
                            recommendation=f"Punta Exact Score {exact_score}",
                            reasoning=(
                                f"🎯 RISULTATO ESATTO!\n\n"
                                f"• Score attuale: {score_home}-{score_away} al {minute}'\n"
                                f"• Partita MOLTO CHIUSA:\n"
                                f"  - Tiri/min: {shots_per_minute:.2f} (bassa)\n"
                                f"  - Tiri in porta: {shots_on_target_home + shots_on_target_away}\n"
                                f"• Alta probabilità rimanga {exact_score}\n"
                                f"• IA boost: +{ai_boost:.0f}%"
                            ),
                            confidence=confidence, odds=3.5, stake_suggestion=1.5,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
                    
        except Exception as e:
            logger.debug(f"⚠️  Errore check exact score markets: {e}")
        return opportunities
    
    def _check_goal_range_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Goal Range (0-1, 2-3, 4+ gol)
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            total_goals = score_home + score_away
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            total_shots = shots_home + shots_away
            
            # Goal Range 0-1: Partita chiusa
            # 🆕 FIX: Solo se è 0-0, non se c'è già 1 gol (altrimenti è illogico)
            if total_goals == 0 and minute >= 60 and minute <= 85:  # Solo 0-0, non 1-0
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute < 0.2:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'goal_range_0_1') if self.ai_pipeline else 0
                    confidence = 75 + (minute - 60) * 0.4 + ai_boost
                    confidence = min(93, confidence)
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='goal_range_0_1', market='goal_range_0_1',
                        recommendation="Punta Goal Range 0-1",
                        reasoning=(
                            f"🎯 FASCIA GOL 0-1!\n\n"
                            f"• Score: {score_home}-{score_away} ({total_goals} gol) al {minute}'\n"
                            f"• Partita CHIUSA (tiri/min: {shots_per_minute:.2f})\n"
                            f"• Alta probabilità max 1 gol totale\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.0, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            # Goal Range 2-3: Partita aperta
            elif total_goals >= 2 and total_goals <= 3 and minute >= 50 and minute <= 80:
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute > 0.3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'goal_range_2_3') if self.ai_pipeline else 0
                    confidence = 72 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='goal_range_2_3', market='goal_range_2_3',
                        recommendation="Punta Goal Range 2-3",
                        reasoning=(
                            f"🎯 FASCIA GOL 2-3!\n\n"
                            f"• Score: {score_home}-{score_away} ({total_goals} gol) al {minute}'\n"
                            f"• Partita APERTA (tiri/min: {shots_per_minute:.2f})\n"
                            f"• Probabile rimanga 2-3 gol\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.2, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            # Goal Range 4+: Partita molto aperta - MA NON se ci sono già 4 gol oltre 80'
            # Goal Range 4+ significa "4 o più gol" - se ci sono già 4 gol, il range è raggiunto!
            elif total_goals == 4 and minute >= 40 and minute <= 75:
                # Solo se ci sono ESATTAMENTE 4 gol e siamo tra 40' e 75' (tempo per altri gol)
                # Se siamo oltre 75', è troppo tardi
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute > 0.35:  # Partita molto aperta
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'goal_range_4_plus') if self.ai_pipeline else 0
                    confidence = 75 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='goal_range_4_plus', market='goal_range_4_plus',
                        recommendation="Punta Goal Range 4+ (probabile 5+ gol)",
                        reasoning=(
                            f"🎯 FASCIA GOL 4+!\n\n"
                            f"• Score: {score_home}-{score_away} ({total_goals} gol) al {minute}'\n"
                            f"• Partita MOLTO APERTA (tiri/min: {shots_per_minute:.2f})\n"
                            f"• Alta probabilità altri gol → 5+ gol\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.5, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            elif total_goals >= 5 and minute >= 40 and minute <= 80:
                # Se ci sono già 5+ gol, Goal Range 4+ è già superato, ma possiamo suggerire se partita ancora aperta
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute > 0.4:  # Partita estremamente aperta
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'goal_range_4_plus') if self.ai_pipeline else 0
                    confidence = 85 + ai_boost  # Alta confidence perché già superato
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='goal_range_4_plus', market='goal_range_4_plus',
                        recommendation="Punta Goal Range 4+ (già superato, probabile altri gol)",
                        reasoning=(
                            f"🎯 FASCIA GOL 4+!\n\n"
                            f"• Score: {score_home}-{score_away} ({total_goals} gol) al {minute}'\n"
                            f"• Range già superato, partita ESTREMAMENTE APERTA\n"
                            f"• Alta probabilità altri gol\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.5, stake_suggestion=1.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                
        except Exception as e:
            logger.debug(f"⚠️  Errore check goal range markets: {e}")
        return opportunities
    
    def _check_team_to_score_next_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Team to Score Next
        - Squadra in svantaggio spinge
        - Squadra che domina
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            possession_home = live_data.get('possession_home', 50)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            
            # Home to Score Next: In svantaggio o domina
            # 🆕 FIX: NON generare se partita è già decisa (3+ gol di differenza) o troppo tardi (oltre 85')
            goal_diff = abs(score_home - score_away)
            if minute >= 20 and minute <= 85 and goal_diff < 3:  # Non generare se partita decisa
                if (score_home < score_away and possession_home > 55 and shots_on_target_home >= 2) or \
                   (score_home == score_away and possession_home > 60 and shots_home > shots_away * 1.3):
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'team_to_score_next_home') if self.ai_pipeline else 0
                    # 🆕 OTTIMIZZATO: Aumentata confidence base per mercato rischioso
                    confidence = 75 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='team_to_score_next_home', market='team_to_score_next_home',
                        recommendation=f"Punta {match_data.get('home')} segna prossimo gol",
                        reasoning=(
                            f"🎯 SQUADRA CHE SEGNA PROSSIMO GOL!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('home')}:\n"
                            f"  - Possesso: {possession_home}%\n"
                            f"  - Tiri in porta: {shots_on_target_home}\n"
                            f"  - {'In svantaggio, spinge' if score_home < score_away else 'Domina'}\n"
                            f"• Alta probabilità prossimo gol\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.2, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            # Away to Score Next
            # 🆕 FIX: NON generare se partita è già decisa (3+ gol di differenza) o troppo tardi (oltre 85')
            goal_diff = abs(score_home - score_away)
            if minute >= 20 and minute <= 85 and goal_diff < 3:  # Non generare se partita decisa
                possession_away = 100 - possession_home
                if (score_away < score_home and possession_away > 55 and shots_on_target_away >= 2) or \
                   (score_home == score_away and possession_away > 60 and shots_away > shots_home * 1.3):
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'team_to_score_next_away') if self.ai_pipeline else 0
                    # 🆕 OTTIMIZZATO: Aumentata confidence base per mercato rischioso
                    confidence = 75 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='team_to_score_next_away', market='team_to_score_next_away',
                        recommendation=f"Punta {match_data.get('away')} segna prossimo gol",
                        reasoning=(
                            f"🎯 SQUADRA CHE SEGNA PROSSIMO GOL!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('away')}:\n"
                            f"  - Possesso: {possession_away}%\n"
                            f"  - Tiri in porta: {shots_on_target_away}\n"
                            f"  - {'In svantaggio, spinge' if score_away < score_home else 'Domina'}\n"
                            f"• Alta probabilità prossimo gol\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.2, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                    
        except Exception as e:
            logger.debug(f"⚠️  Errore check team to score next markets: {e}")
        return opportunities
    
    def _check_clean_sheet_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Clean Sheet
        - Squadra in vantaggio, avversaria senza tiri in porta
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            dangerous_attacks_home = live_data.get('dangerous_attacks_home', 0)
            dangerous_attacks_away = live_data.get('dangerous_attacks_away', 0)
            max_clean_sheet_minute = 72
            
            # Home Clean Sheet: Home in vantaggio, away senza tiri in porta
            # 🆕 FILTRO: Non generare se risultato è già 3-0 o più al 75' (banale)
            goal_diff = abs(score_home - score_away)
            if score_home > 0 and score_away == 0 and minute >= 50:
                if minute > max_clean_sheet_minute:
                    logger.debug(f"⏭️  Clean sheet home non generato: minuto {minute}' oltre soglia {max_clean_sheet_minute}'")
                else:
                    # Non generare se risultato è già 3-0 o più al 75' (troppo ovvio)
                    if goal_diff >= 3 and minute >= 75:
                        logger.debug(f"⏭️  Clean sheet home non generato: risultato {score_home}-{score_away} al {minute}' (troppo ovvio)")
                    # 🆕 OTTIMIZZATO: Blocca anche 2-0 oltre 75' (non solo 80')
                    elif goal_diff >= 2 and minute >= 75:
                        logger.debug(f"⏭️  Clean sheet home non generato: risultato {score_home}-{score_away} al {minute}' (troppo tardi)")
                    elif minute >= 65 and (shots_on_target_away >= 2 or dangerous_attacks_away >= 18):
                        logger.debug(f"⏭️  Clean sheet home non generato: pressione avversaria alta (SoT: {shots_on_target_away}, attacchi pericolosi: {dangerous_attacks_away}) al {minute}'")
                    elif shots_on_target_away <= 1:
                        ai_boost = self._get_ai_market_confidence(match_data, live_data, 'clean_sheet_home') if self.ai_pipeline else 0
                        # 🆕 FIX: Calcola confidence base in modo più concreto
                        # Se partita già decisa (2-0 o più) al 70', confidence base più alta ma comunque sotto soglia
                        if goal_diff >= 2 and minute >= 70:
                            # Partita decisa: confidence base 78% (sotto soglia 80% ma alta)
                            base_confidence = 78
                        elif goal_diff == 1 and minute >= 60:
                            # 1-0 avanzato: confidence base 80%
                            base_confidence = 80
                        else:
                            # Altri casi: confidence base 75%
                            base_confidence = 75
                        
                        confidence = base_confidence + (minute - 50) * 0.3 + min(10, ai_boost)  # 🆕 Limita IA boost a +10% per clean sheet
                        confidence = min(92, confidence)
                        
                        # 🆕 FIX: Verifica che confidence sia almeno 80% (soglia minima per clean_sheet)
                        if confidence < 80:
                            logger.debug(f"⏭️  Clean sheet home non generato: confidence {confidence:.0f}% < 80% (soglia minima)")
                        else:
                            # Calcola statistiche concrete per il reasoning
                            total_shots_away = live_data.get('shots_away', 0)
                            dangerous_attacks_away = live_data.get('dangerous_attacks_away', 0)
                            xg_away = live_data.get('xg_away', 0)
                            
                            opportunity = LiveBettingOpportunity(
                                match_id=match_id, match_data=match_data,
                                situation='clean_sheet_home', market='clean_sheet_home',
                                recommendation=f"Punta {match_data.get('home')} Clean Sheet",
                                reasoning=(
                                    f"🎯 PORTA INVOLATA!\n\n"
                                    f"• Score: {score_home}-{score_away} al {minute}'\n"
                                    f"• {match_data.get('home')} in vantaggio di {goal_diff} gol\n"
                                    f"• {match_data.get('away')} OFFENSIVAMENTE INEFFICACE:\n"
                                    f"  - Tiri in porta: {shots_on_target_away} (massimo 1)\n"
                                    f"  - Tiri totali: {total_shots_away}\n"
                                    f"  - Attacchi pericolosi: {dangerous_attacks_away}\n"
                                    f"  - xG: {xg_away:.2f}\n"
                                    f"• Alta probabilità clean sheet basata su dati concreti\n"
                                    f"• IA boost: +{ai_boost:.0f}%"
                                ),
                                confidence=confidence, odds=2.0, stake_suggestion=2.0,
                                timestamp=datetime.now()
                            )
                            opportunities.append(opportunity)
            
            # Away Clean Sheet
            # 🆕 FILTRO: Non generare se risultato è già 3-0 o più al 75' (banale)
            if score_away > 0 and score_home == 0 and minute >= 50:
                if minute > max_clean_sheet_minute:
                    logger.debug(f"⏭️  Clean sheet away non generato: minuto {minute}' oltre soglia {max_clean_sheet_minute}'")
                else:
                    # Non generare se risultato è già 3-0 o più al 75' (troppo ovvio)
                    if goal_diff >= 3 and minute >= 75:
                        logger.debug(f"⏭️  Clean sheet away non generato: risultato {score_home}-{score_away} al {minute}' (troppo ovvio)")
                    # 🆕 OTTIMIZZATO: Blocca anche 2-0 oltre 75' (non solo 80')
                    elif goal_diff >= 2 and minute >= 75:
                        logger.debug(f"⏭️  Clean sheet away non generato: risultato {score_home}-{score_away} al {minute}' (troppo tardi)")
                    elif minute >= 65 and (shots_on_target_home >= 2 or dangerous_attacks_home >= 18):
                        logger.debug(f"⏭️  Clean sheet away non generato: pressione avversaria alta (SoT: {shots_on_target_home}, attacchi pericolosi: {dangerous_attacks_home}) al {minute}'")
                    elif shots_on_target_home <= 1:
                        ai_boost = self._get_ai_market_confidence(match_data, live_data, 'clean_sheet_away') if self.ai_pipeline else 0
                        # 🆕 FIX: Calcola confidence base in modo più concreto
                        # Se partita già decisa (2-0 o più) al 70', confidence base più alta ma comunque sotto soglia
                        if goal_diff >= 2 and minute >= 70:
                            # Partita decisa: confidence base 78% (sotto soglia 80% ma alta)
                            base_confidence = 78
                        elif goal_diff == 1 and minute >= 60:
                            # 0-1 avanzato: confidence base 80%
                            base_confidence = 80
                        else:
                            # Altri casi: confidence base 75%
                            base_confidence = 75
                        
                        confidence = base_confidence + (minute - 50) * 0.3 + min(10, ai_boost)  # 🆕 Limita IA boost a +10% per clean sheet
                        confidence = min(92, confidence)
                        
                        # 🆕 FIX: Verifica che confidence sia almeno 80% (soglia minima per clean_sheet)
                        if confidence < 80:
                            logger.debug(f"⏭️  Clean sheet away non generato: confidence {confidence:.0f}% < 80% (soglia minima)")
                        else:
                            # Calcola statistiche concrete per il reasoning
                            total_shots_home = live_data.get('shots_home', 0)
                            dangerous_attacks_home = live_data.get('dangerous_attacks_home', 0)
                            xg_home = live_data.get('xg_home', 0)
                            
                            opportunity = LiveBettingOpportunity(
                                match_id=match_id, match_data=match_data,
                                situation='clean_sheet_away', market='clean_sheet_away',
                                recommendation=f"Punta {match_data.get('away')} Clean Sheet",
                                reasoning=(
                                    f"🎯 PORTA INVOLATA!\n\n"
                                    f"• Score: {score_home}-{score_away} al {minute}'\n"
                                    f"• {match_data.get('away')} in vantaggio di {goal_diff} gol\n"
                                    f"• {match_data.get('home')} OFFENSIVAMENTE INEFFICACE:\n"
                                    f"  - Tiri in porta: {shots_on_target_home} (massimo 1)\n"
                                    f"  - Tiri totali: {total_shots_home}\n"
                                    f"  - Attacchi pericolosi: {dangerous_attacks_home}\n"
                                    f"  - xG: {xg_home:.2f}\n"
                                    f"• Alta probabilità clean sheet basata su dati concreti\n"
                                    f"• IA boost: +{ai_boost:.0f}%"
                                ),
                                confidence=confidence, odds=2.0, stake_suggestion=2.0,
                                timestamp=datetime.now()
                            )
                            opportunities.append(opportunity)
                    
        except Exception as e:
            logger.debug(f"⚠️  Errore check clean sheet markets: {e}")
        return opportunities
    
    def _check_ht_ft_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Half Time/Full Time
        - Solo se siamo nel secondo tempo
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            
            # Solo nel secondo tempo
            if minute >= 45 and minute <= 80:
                # HT/FT Home/Home: Home vinceva al primo tempo e ancora in vantaggio
                # (assumiamo che al primo tempo home vinceva se ora è in vantaggio)
                if score_home > score_away:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'ht_ft_home_home') if self.ai_pipeline else 0
                    confidence = 70 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='ht_ft_home_home', market='ht_ft_home_home',
                        recommendation=f"Punta HT/FT {match_data.get('home')}/{match_data.get('home')}",
                        reasoning=(
                            f"🎯 HALF TIME/FULL TIME!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('home')} in vantaggio\n"
                            f"• Probabile vittoria finale\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.5, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                    
        except Exception as e:
            logger.debug(f"⚠️  Errore check HT/FT markets: {e}")
        return opportunities
    
    def _check_match_winner_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Match Winner (1X2) migliorato
        - Solo se c'è valore reale, non banale
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            possession_home = live_data.get('possession_home', 50)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            
            odds_1 = match_data.get('odds_1', 2.0)
            odds_2 = match_data.get('odds_2', 2.0)
            
            # Home Win: Pareggio ma home domina nettamente
            # 🆕 FIX: Aumentata confidence base (70% troppo bassa, min richiesto 78% per match_winner)
            if score_home == score_away and minute >= 50 and minute <= 75:
                if possession_home > 65 and shots_home > shots_away * 1.5 and shots_on_target_home >= 4:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'home_win') if self.ai_pipeline else 0
                    # 🆕 FIX: Confidence base aumentata a 75% (minimo 78% richiesto, quindi serve almeno +3% da AI)
                    confidence = 75 + ai_boost
                    # 🆕 FIX: Se confidence finale < 78%, non generare (troppo rischioso)
                    if confidence < 78:
                        logger.debug(f"⏭️  Saltata opportunità: Home Win su 0-0 con confidence {confidence:.0f}% < 78% (troppo bassa)")
                        return opportunities
                    ev_pct = self._calculate_ev_from_values(confidence, odds_1)
                    if ev_pct < self.min_ev:
                        logger.debug(f"⏭️  Saltata opportunità: Home Win senza valore (EV {ev_pct:.1f}% < {self.min_ev:.1f}%)")
                        return opportunities
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='home_win_dominance', market='home_win',
                        recommendation=f"Punta {match_data.get('home')} vince",
                        reasoning=(
                            f"🎯 VITTORIA FINALE (1X2)!\n\n"
                            f"• Score: {score_home}-{score_away} (pareggio) al {minute}'\n"
                            f"• {match_data.get('home')} DOMINA nettamente:\n"
                            f"  - Possesso: {possession_home}%\n"
                            f"  - Tiri: {shots_home} vs {shots_away}\n"
                            f"  - Tiri in porta: {shots_on_target_home} vs {shots_on_target_away}\n"
                            f"• Alta probabilità vittoria\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=odds_1, stake_suggestion=3.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            # Away Win: Pareggio ma away domina nettamente
            # 🆕 FIX: Aumentata confidence base (70% troppo bassa, min richiesto 78% per match_winner)
            elif score_home == score_away and minute >= 50 and minute <= 75:
                possession_away = 100 - possession_home
                if possession_away > 65 and shots_away > shots_home * 1.5 and shots_on_target_away >= 4:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'away_win') if self.ai_pipeline else 0
                    # 🆕 FIX: Confidence base aumentata a 75% (minimo 78% richiesto, quindi serve almeno +3% da AI)
                    confidence = 75 + ai_boost
                    # 🆕 FIX: Se confidence finale < 78%, non generare (troppo rischioso)
                    if confidence < 78:
                        logger.debug(f"⏭️  Saltata opportunità: Away Win su 0-0 con confidence {confidence:.0f}% < 78% (troppo bassa)")
                        return opportunities
                    ev_pct = self._calculate_ev_from_values(confidence, odds_2)
                    if ev_pct < self.min_ev:
                        logger.debug(f"⏭️  Saltata opportunità: Away Win senza valore (EV {ev_pct:.1f}% < {self.min_ev:.1f}%)")
                        return opportunities
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='away_win_dominance', market='away_win',
                        recommendation=f"Punta {match_data.get('away')} vince",
                        reasoning=(
                            f"🎯 VITTORIA FINALE (1X2)!\n\n"
                            f"• Score: {score_home}-{score_away} (pareggio) al {minute}'\n"
                            f"• {match_data.get('away')} DOMINA nettamente:\n"
                            f"  - Possesso: {possession_away}%\n"
                            f"  - Tiri: {shots_away} vs {shots_home}\n"
                            f"  - Tiri in porta: {shots_on_target_away} vs {shots_on_target_home}\n"
                            f"• Alta probabilità vittoria\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=odds_2, stake_suggestion=3.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                    
        except Exception as e:
            logger.debug(f"⚠️  Errore check match winner markets: {e}")
        return opportunities
    
    def _check_asian_handicap_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Asian Handicap
        - Squadra in svantaggio ma domina
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            possession_home = live_data.get('possession_home', 50)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            diff = score_home - score_away
            
            # Asian Handicap Home +0.5 o +1.5: Se perde ma domina
            if score_home < score_away and minute >= 30 and minute <= 75:
                if possession_home > 60 and shots_home > shots_away * 1.3:
                    handicap = abs(diff) + 0.5
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'asian_handicap_home') if self.ai_pipeline else 0
                    confidence = 72 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='asian_handicap_home', market=f'asian_handicap_home_+{handicap}',
                        recommendation=f"Punta {match_data.get('home')} Asian Handicap +{handicap}",
                        reasoning=(
                            f"🎯 HANDICAP ASIATICO!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('home')} perde ma DOMINA:\n"
                            f"  - Possesso: {possession_home}%\n"
                            f"  - Tiri: {shots_home} vs {shots_away}\n"
                            f"• Handicap +{handicap} offre buon valore\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.7, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            # Asian Handicap Away
            elif score_away < score_home and minute >= 30 and minute <= 75:
                possession_away = 100 - possession_home
                if possession_away > 60 and shots_away > shots_home * 1.3:
                    handicap = abs(diff) + 0.5
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'asian_handicap_away') if self.ai_pipeline else 0
                    confidence = 72 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='asian_handicap_away', market=f'asian_handicap_away_+{handicap}',
                        recommendation=f"Punta {match_data.get('away')} Asian Handicap +{handicap}",
                        reasoning=(
                            f"🎯 HANDICAP ASIATICO!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('away')} perde ma DOMINA:\n"
                            f"  - Possesso: {possession_away}%\n"
                            f"  - Tiri: {shots_away} vs {shots_home}\n"
                            f"• Handicap +{handicap} offre buon valore\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.7, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                    
        except Exception as e:
            logger.debug(f"⚠️  Errore check asian handicap markets: {e}")
        return opportunities
    
    def _check_time_of_next_goal_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Rileva opportunità Time of Next Goal
        - Analisi pattern partita
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            total_shots = shots_home + shots_away
            
            # Next Goal Before 75': Se partita aperta
            # 🆕 FIX: NON generare se siamo oltre 75' (illogico)
            if minute >= 20 and minute <= 75:  # Ridotto a 75' invece di 70'
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute > 0.3 and total_shots >= 12:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'next_goal_before_75') if self.ai_pipeline else 0
                    # 🆕 OTTIMIZZATO: Aumentata confidence base per next goal (mercato rischioso)
                    confidence = 75 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='next_goal_before_75', market='next_goal_before_75',
                        recommendation="Punta Prossimo Gol Prima del 75'",
                        reasoning=(
                            f"🎯 TIME OF NEXT GOAL!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Partita APERTA:\n"
                            f"  - Tiri: {total_shots} (media: {shots_per_minute:.2f}/min)\n"
                            f"• Alta probabilità gol prima del 75'\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.8, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
            
            # Next Goal After 75': Se partita chiusa
            # 🆕 FIX: NON generare se siamo oltre 75' (illogico - il 75' è già passato)
            elif minute >= 60 and minute <= 75:
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute < 0.2:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'next_goal_after_75') if self.ai_pipeline else 0
                    # 🆕 OTTIMIZZATO: Aumentata confidence base per mercato rischioso
                    confidence = 75 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='next_goal_after_75', market='next_goal_after_75',
                        recommendation="Punta Prossimo Gol Dopo il 75'",
                        reasoning=(
                            f"🎯 TIME OF NEXT GOAL!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Partita CHIUSA (tiri/min: {shots_per_minute:.2f})\n"
                            f"• Probabile gol tardivo se c'è\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.2, stake_suggestion=1.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                    
        except Exception as e:
            logger.debug(f"⚠️  Errore check time of next goal markets: {e}")
        return opportunities
    
    def _check_team_to_score_first_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        🆕 NUOVO: Rileva opportunità Team to Score First
        - Solo se partita è 0-0 (altrimenti è banale!)
        - Analisi dominio partita
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            possession_home = live_data.get('possession_home', 50)
            
            # 🆕 FILTRO ANTI-OVVIETÀ: Solo se 0-0 (altrimenti è banale!)
            if score_home == 0 and score_away == 0 and minute >= 10 and minute <= 40:
                # Home domina nettamente
                if possession_home > 60 and shots_home > shots_away * 1.5 and shots_on_target_home >= 3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'team_to_score_first_home') if self.ai_pipeline else 0
                    confidence = 73 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='team_to_score_first_home', market='team_to_score_first_home',
                        recommendation=f"Punta {match_data.get('home')} segna per primo",
                        reasoning=(
                            f"🎯 SQUADRA CHE SEGNA PER PRIMA!\n\n"
                            f"• Score: 0-0 al {minute}'\n"
                            f"• {match_data.get('home')} DOMINA:\n"
                            f"  - Possesso: {possession_home}%\n"
                            f"  - Tiri: {shots_home} vs {shots_away}\n"
                            f"  - Tiri in porta: {shots_on_target_home} vs {shots_on_target_away}\n"
                            f"• Alta probabilità segna per primo\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.7, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                
                # Away domina nettamente
                elif possession_home < 40 and shots_away > shots_home * 1.5 and shots_on_target_away >= 3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'team_to_score_first_away') if self.ai_pipeline else 0
                    confidence = 73 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='team_to_score_first_away', market='team_to_score_first_away',
                        recommendation=f"Punta {match_data.get('away')} segna per primo",
                        reasoning=(
                            f"🎯 SQUADRA CHE SEGNA PER PRIMA!\n\n"
                            f"• Score: 0-0 al {minute}'\n"
                            f"• {match_data.get('away')} DOMINA:\n"
                            f"  - Possesso: {100-possession_home}%\n"
                            f"  - Tiri: {shots_away} vs {shots_home}\n"
                            f"  - Tiri in porta: {shots_on_target_away} vs {shots_on_target_home}\n"
                            f"• Alta probabilità segna per primo\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.7, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check team to score first markets: {e}")
        return opportunities
    
    def _check_team_to_score_last_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        🆕 NUOVO: Rileva opportunità Team to Score Last
        - Solo se partita è in corso e non è già decisa
        - Analisi momentum partita
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            possession_home = live_data.get('possession_home', 50)
            goal_diff = abs(score_home - score_away)
            
            # 🆕 FILTRO ANTI-OVVIETÀ: Solo se partita non decisa e non troppo tardi
            if goal_diff <= 2 and minute >= 50 and minute <= 85:
                # Home in vantaggio o pareggio ma domina
                if (score_home >= score_away) and possession_home > 55 and shots_home > shots_away * 1.3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'team_to_score_last_home') if self.ai_pipeline else 0
                    confidence = 73 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='team_to_score_last_home', market='team_to_score_last_home',
                        recommendation=f"Punta {match_data.get('home')} segna per ultimo",
                        reasoning=(
                            f"🎯 SQUADRA CHE SEGNA PER ULTIMA!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('home')} in momentum:\n"
                            f"  - Possesso: {possession_home}%\n"
                            f"  - Tiri: {shots_home} vs {shots_away}\n"
                            f"• Alta probabilità segna per ultimo\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.8, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                
                # Away in vantaggio o pareggio ma domina
                elif (score_away >= score_home) and possession_home < 45 and shots_away > shots_home * 1.3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'team_to_score_last_away') if self.ai_pipeline else 0
                    confidence = 73 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='team_to_score_last_away', market='team_to_score_last_away',
                        recommendation=f"Punta {match_data.get('away')} segna per ultimo",
                        reasoning=(
                            f"🎯 SQUADRA CHE SEGNA PER ULTIMA!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('away')} in momentum:\n"
                            f"  - Possesso: {100-possession_home}%\n"
                            f"  - Tiri: {shots_away} vs {shots_home}\n"
                            f"• Alta probabilità segna per ultimo\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.8, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check team to score last markets: {e}")
        return opportunities
    
    def _check_highest_scoring_half_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        🆕 NUOVO: Rileva opportunità Highest Scoring Half
        - Solo se siamo nel secondo tempo
        - Analisi gol per tempo
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            total_goals = score_home + score_away
            
            # 🆕 FILTRO ANTI-OVVIETÀ: Solo se siamo nel secondo tempo e partita non decisa
            if minute >= 50 and minute <= 80:
                # 🆕 MIGLIORATO: Usa eventi reali se disponibili (da API-Football)
                events = live_data.get('events', [])
                ht_goals = 0
                st_goals = 0
                
                # Calcola gol per tempo usando eventi reali
                if events:
                    for event in events:
                        event_type = event.get('type', '').lower()
                        event_minute = event.get('minute', 0)
                        if event_type in ['goal', 'goal penalty', 'goal own']:
                            if event_minute <= 45:
                                ht_goals += 1
                            elif event_minute > 45:
                                st_goals += 1
                
                # Se non abbiamo eventi, stima (fallback)
                if not events or (ht_goals == 0 and st_goals == 0):
                    # Stima gol primo tempo (assumendo distribuzione tipica)
                    # Se siamo a 50' e ci sono 2+ gol, probabilmente 1+ nel primo tempo
                    # Se siamo a 70' e ci sono 1 gol, probabilmente 0 nel primo tempo
                    estimated_ht_goals = max(0, total_goals - 1) if minute >= 60 else total_goals
                    estimated_st_goals = total_goals - estimated_ht_goals
                else:
                    # Usa dati reali dagli eventi
                    estimated_ht_goals = ht_goals
                    estimated_st_goals = st_goals
                
                # 🆕 BLOCCA se risultato già definito (es. 1-2 al 64' = primo tempo ha più gol, BANALE!)
                # Se abbiamo eventi reali e primo tempo ha già 2+ gol mentre secondo 0, è banale
                if events and ht_goals >= 2 and st_goals == 0:
                    logger.debug(f"⏭️  Saltata opportunità banale: Highest Scoring Half 1H su {score_home}-{score_away} al {minute}' (primo tempo ha {ht_goals} gol, secondo {st_goals} - OVVIO!)")
                # Se risultato è 1-2 o 2-1 al 64'+, è ovvio che primo tempo ha più gol
                elif total_goals >= 3 and minute >= 60:
                    logger.debug(f"⏭️  Saltata opportunità banale: Highest Scoring Half 1H su {score_home}-{score_away} al {minute}' (3+ gol totali, primo tempo probabilmente più prolifico - BANALE!)")
                # Se primo tempo ha più gol (solo se non banale)
                elif estimated_ht_goals >= 2 and total_goals <= 3 and not (total_goals >= 3 and minute >= 60):
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'highest_scoring_half_1h') if self.ai_pipeline else 0
                    confidence = 75 + ai_boost
                    
                    # Determina se dati sono reali o stimati
                    data_source = "reali" if events and (ht_goals > 0 or st_goals > 0) else "stimati"
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='highest_scoring_half_1h', market='highest_scoring_half_1h',
                        recommendation="Punta Primo Tempo con più gol",
                        reasoning=(
                            f"🎯 TEMPO CON PIÙ GOL: 1° TEMPO\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Primo tempo: {estimated_ht_goals} gol ({data_source})\n"
                            f"• Secondo tempo: {estimated_st_goals} gol ({data_source})\n"
                            f"• Primo tempo più prolifico\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.0, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                
                # Se secondo tempo sta avendo più gol
                elif minute >= 60 and total_goals >= 2:
                    if estimated_st_goals > estimated_ht_goals:
                        ai_boost = self._get_ai_market_confidence(match_data, live_data, 'highest_scoring_half_2h') if self.ai_pipeline else 0
                        confidence = 75 + ai_boost
                        
                        # Determina se dati sono reali o stimati
                        data_source = "reali" if events and (ht_goals > 0 or st_goals > 0) else "stimati"
                        
                        opportunity = LiveBettingOpportunity(
                            match_id=match_id, match_data=match_data,
                            situation='highest_scoring_half_2h', market='highest_scoring_half_2h',
                            recommendation="Punta Secondo Tempo con più gol",
                            reasoning=(
                                f"🎯 TEMPO CON PIÙ GOL!\n\n"
                                f"• Score: {score_home}-{score_away} al {minute}'\n"
                                f"• Primo tempo: {estimated_ht_goals} gol ({data_source})\n"
                                f"• Secondo tempo: {estimated_st_goals} gol ({data_source})\n"
                                f"• Secondo tempo più prolifico\n"
                                f"• IA boost: +{ai_boost:.0f}%"
                            ),
                            confidence=confidence, odds=2.0, stake_suggestion=2.0,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check highest scoring half markets: {e}")
        return opportunities
    
    def _check_win_either_half_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        🆕 NUOVO: Rileva opportunità To Win Either Half
        - Squadra che vince almeno un tempo
        - Solo se partita non decisa
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            possession_home = live_data.get('possession_home', 50)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            goal_diff = abs(score_home - score_away)
            
            # 🆕 FILTRO ANTI-OVVIETÀ: Solo se partita non decisa e non troppo tardi
            if goal_diff <= 2 and minute >= 20 and minute <= 75:
                # Home domina ma non vince nettamente
                if possession_home > 60 and shots_home > shots_away * 1.5 and score_home <= score_away + 1:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'win_either_half_home') if self.ai_pipeline else 0
                    confidence = 73 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='win_either_half_home', market='win_either_half_home',
                        recommendation=f"Punta {match_data.get('home')} vince almeno un tempo",
                        reasoning=(
                            f"🎯 WIN EITHER HALF!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('home')} DOMINA:\n"
                            f"  - Possesso: {possession_home}%\n"
                            f"  - Tiri: {shots_home} vs {shots_away}\n"
                            f"• Alta probabilità vince almeno un tempo\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.6, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                
                # Away domina ma non vince nettamente
                elif possession_home < 40 and shots_away > shots_home * 1.5 and score_away <= score_home + 1:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'win_either_half_away') if self.ai_pipeline else 0
                    confidence = 73 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='win_either_half_away', market='win_either_half_away',
                        recommendation=f"Punta {match_data.get('away')} vince almeno un tempo",
                        reasoning=(
                            f"🎯 WIN EITHER HALF!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('away')} DOMINA:\n"
                            f"  - Possesso: {100-possession_home}%\n"
                            f"  - Tiri: {shots_away} vs {shots_home}\n"
                            f"• Alta probabilità vince almeno un tempo\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=1.6, stake_suggestion=2.5,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check win either half markets: {e}")
        return opportunities
    
    def _check_btts_first_half_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        🆕 NUOVO: Rileva opportunità Both Teams to Score in First Half
        - Solo se siamo nel primo tempo
        - Analisi apertura partita
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            total_shots = live_data.get('shots_home', 0) + live_data.get('shots_away', 0)
            
            # 🆕 FILTRO ANTI-OVVIETÀ: Solo se primo tempo, partita aperta, e non già BTTS
            if minute >= 20 and minute <= 40:
                # Se una squadra ha già segnato e l'altra ha tiri in porta
                if (score_home > 0 and score_away == 0 and shots_on_target_away >= 2) or \
                   (score_away > 0 and score_home == 0 and shots_on_target_home >= 2):
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'btts_first_half') if self.ai_pipeline else 0
                    confidence = 73 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='btts_first_half', market='btts_first_half',
                        recommendation="Punta Both Teams To Score Primo Tempo",
                        reasoning=(
                            f"🎯 BTTS FIRST HALF!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• Una squadra ha segnato, l'altra ha {shots_on_target_home if score_away > 0 else shots_on_target_away} tiri in porta\n"
                            f"• Partita aperta: {total_shots} tiri totali\n"
                            f"• Alta probabilità BTTS nel primo tempo\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.5, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check BTTS first half markets: {e}")
        return opportunities
    
    def _check_half_time_result_markets(
        self,
        match_id: str,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        🆕 NUOVO: Rileva opportunità Half Time Result
        - Solo se siamo nel primo tempo
        - Analisi dominio primo tempo
        """
        opportunities = []
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            minute = live_data.get('minute', 0)
            possession_home = live_data.get('possession_home', 50)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            
            # 🆕 FILTRO ANTI-OVVIETÀ: Solo se primo tempo e non troppo tardi (al 44' è banale!)
            if minute >= 25 and minute <= 42:
                # Home domina nettamente
                if possession_home > 65 and shots_home > shots_away * 1.5 and shots_on_target_home >= 3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'half_time_result_home') if self.ai_pipeline else 0
                    confidence = 73 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='half_time_result_home', market='half_time_result_home',
                        recommendation=f"Punta {match_data.get('home')} vince Primo Tempo",
                        reasoning=(
                            f"🎯 HALF TIME RESULT!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('home')} DOMINA primo tempo:\n"
                            f"  - Possesso: {possession_home}%\n"
                            f"  - Tiri: {shots_home} vs {shots_away}\n"
                            f"  - Tiri in porta: {shots_on_target_home} vs {shots_on_target_away}\n"
                            f"• Alta probabilità vince primo tempo\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.2, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                
                # Away domina nettamente
                elif possession_home < 35 and shots_away > shots_home * 1.5 and shots_on_target_away >= 3:
                    ai_boost = self._get_ai_market_confidence(match_data, live_data, 'half_time_result_away') if self.ai_pipeline else 0
                    confidence = 73 + ai_boost
                    
                    opportunity = LiveBettingOpportunity(
                        match_id=match_id, match_data=match_data,
                        situation='half_time_result_away', market='half_time_result_away',
                        recommendation=f"Punta {match_data.get('away')} vince Primo Tempo",
                        reasoning=(
                            f"🎯 HALF TIME RESULT!\n\n"
                            f"• Score: {score_home}-{score_away} al {minute}'\n"
                            f"• {match_data.get('away')} DOMINA primo tempo:\n"
                            f"  - Possesso: {100-possession_home}%\n"
                            f"  - Tiri: {shots_away} vs {shots_home}\n"
                            f"  - Tiri in porta: {shots_on_target_away} vs {shots_on_target_home}\n"
                            f"• Alta probabilità vince primo tempo\n"
                            f"• IA boost: +{ai_boost:.0f}%"
                        ),
                        confidence=confidence, odds=2.2, stake_suggestion=2.0,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
        except Exception as e:
            logger.debug(f"⚠️  Errore check half time result markets: {e}")
        return opportunities
    
    def _extract_match_stats(self, live_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estrae statistiche partita da dati live"""
        return {
            'score_home': live_data.get('score_home', 0),
            'score_away': live_data.get('score_away', 0),
            'minute': live_data.get('minute', 0),
            'possession_home': live_data.get('possession_home', 50),
            'possession_away': live_data.get('possession_away', 50),
            'shots_home': live_data.get('shots_home', 0),
            'shots_away': live_data.get('shots_away', 0),
            'shots_on_target_home': live_data.get('shots_on_target_home', 0),
            'shots_on_target_away': live_data.get('shots_on_target_away', 0),
            'corners_home': live_data.get('corners_home', 0),
            'corners_away': live_data.get('corners_away', 0),
            'fouls_home': live_data.get('fouls_home', 0),
            'fouls_away': live_data.get('fouls_away', 0),
            'yellow_cards_home': live_data.get('yellow_cards_home', 0),
            'yellow_cards_away': live_data.get('yellow_cards_away', 0),
            'red_cards_home': live_data.get('red_cards_home', 0),
            'red_cards_away': live_data.get('red_cards_away', 0),
            'events': live_data.get('events', [])
        }
    
    def _extract_key_stats_for_market(
        self,
        opportunity: LiveBettingOpportunity,
        live_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Raccoglie le statistiche più rilevanti per il mercato specifico."""
        if not live_data:
            return {}
        
        market = (opportunity.market or '').lower()
        score_home = live_data.get('score_home', 0)
        score_away = live_data.get('score_away', 0)
        minute = live_data.get('minute', 0)
        possession_home = live_data.get('possession_home')
        shots_home = live_data.get('shots_home')
        shots_away = live_data.get('shots_away')
        shots_on_target_home = live_data.get('shots_on_target_home')
        shots_on_target_away = live_data.get('shots_on_target_away')
        xg_home = live_data.get('xg_home')
        xg_away = live_data.get('xg_away')
        dangerous_attacks_home = live_data.get('dangerous_attacks_home')
        dangerous_attacks_away = live_data.get('dangerous_attacks_away')
        attacks_home = live_data.get('attacks_home')
        attacks_away = live_data.get('attacks_away')
        
        stats: Dict[str, Any] = {
            "Score": f"{score_home}-{score_away} al {minute}'"
        }
        
        def _add(label: str, value: Optional[float], suffix: str = '') -> None:
            if value is None:
                return
            if isinstance(value, float):
                stats[label] = f"{value:.2f}{suffix}"
            else:
                stats[label] = f"{value}{suffix}" if suffix else value
        
        total_shots = (shots_home or 0) + (shots_away or 0)
        total_sot = (shots_on_target_home or 0) + (shots_on_target_away or 0)
        total_xg = None
        if xg_home is not None or xg_away is not None:
            total_xg = (xg_home or 0) + (xg_away or 0)
        total_dangerous = (dangerous_attacks_home or 0) + (dangerous_attacks_away or 0)
        
        if 'clean_sheet_home' in market:
            # 🆕 FIX: Statistiche concrete per clean sheet home
            _add("Tiri in porta ospiti", shots_on_target_away)
            _add("Tiri totali ospiti", shots_away)
            _add("xG ospiti", xg_away)
            _add("Attacchi pericolosi ospiti", dangerous_attacks_away)
            if minute > 0:
                shots_per_min_away = (shots_away or 0) / minute
                _add("Tiri/minuto ospiti", shots_per_min_away)
        elif 'clean_sheet_away' in market:
            # 🆕 FIX: Statistiche concrete per clean sheet away
            _add("Tiri in porta casa", shots_on_target_home)
            _add("Tiri totali casa", shots_home)
            _add("xG casa", xg_home)
            _add("Attacchi pericolosi casa", dangerous_attacks_home)
            if minute > 0:
                shots_per_min_home = (shots_home or 0) / minute
                _add("Tiri/minuto casa", shots_per_min_home)
        elif any(key in market for key in ['home_win', 'away_win', 'match_winner', '1x2']):
            if possession_home is not None:
                stats["Possesso"] = f"{possession_home:.0f}% - {100 - possession_home:.0f}%"
            _add("Tiri casa", shots_home)
            _add("Tiri trasferta", shots_away)
            _add("Tiri in porta casa", shots_on_target_home)
            _add("Tiri in porta trasferta", shots_on_target_away)
            _add("xG casa", xg_home)
            _add("xG trasferta", xg_away)
        elif any(key in market for key in ['over', 'goal_range', 'btts', 'team_to_score', 'next_goal']):
            _add("Tiri totali", total_shots)
            _add("Tiri in porta totali", total_sot)
            if total_xg is not None:
                _add("xG totali", total_xg)
            _add("Attacchi pericolosi totali", total_dangerous)
        elif 'highest_scoring_half' in market:
            _add("Tiri casa", shots_home)
            _add("Tiri trasferta", shots_away)
            _add("Tiri in porta totali", total_sot)
        else:
            # Default: mostra indicatori generali se disponibili
            if possession_home is not None:
                stats["Possesso"] = f"{possession_home:.0f}% - {100 - possession_home:.0f}%"
            _add("Tiri totali", total_shots)
            _add("Tiri in porta totali", total_sot)
        
        return {k: v for k, v in stats.items() if v is not None}
    
    def _translate_market_name(self, market: str) -> str:
        """Restituisce la traduzione italiana del mercato, se disponibile."""
        if not market:
            return ''
        key = market.lower().strip()
        return self.market_translations.get(key, market.upper().replace('_', ' '))
    
    def _populate_opportunity_metadata(
        self,
        opportunity: LiveBettingOpportunity,
        live_data: Dict[str, Any]
    ) -> None:
        """Arricchisce l'opportunità con stats, EV e urgenza."""
        opportunity.match_stats = self._extract_match_stats(live_data)
        opportunity.key_stats = self._extract_key_stats_for_market(opportunity, live_data)
        opportunity.urgency_level = self._calculate_urgency(opportunity, live_data)
        opportunity.ev = self._calculate_expected_value(opportunity)
    
    def _calculate_urgency(self, opportunity: LiveBettingOpportunity, live_data: Dict[str, Any]) -> str:
        """Calcola livello urgenza basato su confidence, minuto, situazione"""
        minute = live_data.get('minute', 0)
        confidence = opportunity.confidence
        
        # Urgenza basata su confidence e minuto
        if confidence >= 85 and minute >= 60:
            return 'URGENT'  # Alta confidence + partita avanzata
        elif confidence >= 75:
            return 'HIGH'
        elif confidence >= 65:
            return 'NORMAL'
        else:
            return 'LOW'
    
    def _get_ai_market_confidence(
        self,
        match_data: Dict[str, Any],
        live_data: Dict[str, Any],
        market: str
    ) -> float:
        """Usa analisi avanzata per calcolare confidence boost per un mercato (funziona anche senza AI pipeline)"""
        try:
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            total_goals = score_home + score_away
            minute = live_data.get('minute', 1)
            shots_home = live_data.get('shots_home', 0)
            shots_away = live_data.get('shots_away', 0)
            total_shots = shots_home + shots_away
            shots_on_target_home = live_data.get('shots_on_target_home', 0)
            shots_on_target_away = live_data.get('shots_on_target_away', 0)
            total_shots_on_target = shots_on_target_home + shots_on_target_away
            possession_home = live_data.get('possession_home', 50)
            boost = 0.0
            
            # 🆕 Se non abbiamo statistiche (tiri = 0), usa analisi basata su score e minuto
            if total_shots == 0 and minute > 0:
                # Analisi semplificata basata su pattern temporali e score
                if 'over' in market.lower():
                    # Se partita è aperta (score equilibrato) e avanzata, boost positivo
                    if abs(score_home - score_away) <= 1 and minute >= 30:
                        boost += 3
                    # Se ci sono già gol, probabilità altri gol
                    if total_goals >= 1 and minute < 70:
                        boost += 2
                elif 'under' in market.lower():
                    # Se partita è chiusa (0-0 o 1-0) e avanzata, boost positivo
                    if total_goals <= 1 and minute >= 50:
                        boost += 3
                    if total_goals == 0 and minute >= 60:
                        boost += 2
                return min(15, boost)  # Boost limitato se non abbiamo statistiche
            
            # Analisi avanzata per Over markets (con statistiche)
            if 'over' in market.lower():
                if total_shots > 10:
                    boost += 5
                if total_shots > 15:
                    boost += 5
                if total_shots > 20:
                    boost += 5  # Partita molto aperta
                if abs(score_home - score_away) <= 1:
                    boost += 3  # Partita equilibrata = più gol
                # Tiri in porta = più probabilità gol
                if total_shots_on_target > 5:
                    boost += 3
                if total_shots_on_target > 8:
                    boost += 3  # Molti tiri in porta
                # Calcola tasso gol
                if minute > 0:
                    goals_per_minute = total_goals / minute
                    if goals_per_minute > 0.03:  # >2.7 gol/90min
                        boost += 4
                    if goals_per_minute > 0.04:  # >3.6 gol/90min
                        boost += 3
                # Possesso alto = partita aperta
                if possession_home > 60 or possession_home < 40:
                    boost += 2  # Dominio di una squadra = più azione
            # Analisi avanzata per Under markets
            elif 'under' in market.lower():
                if total_shots < 8:
                    boost += 5
                if total_shots < 5:
                    boost += 5
                if total_shots < 3:
                    boost += 5  # Partita estremamente chiusa
                # Pochi tiri in porta = partita chiusa
                if total_shots_on_target < 3:
                    boost += 3
                if total_shots_on_target < 1:
                    boost += 5  # Nessun tiro in porta
                # Calcola tasso gol
                if minute > 0:
                    goals_per_minute = total_goals / minute
                    if goals_per_minute < 0.02:  # <1.8 gol/90min
                        boost += 4
                    if goals_per_minute < 0.015:  # <1.35 gol/90min
                        boost += 3
                # Possesso equilibrato = partita chiusa
                if 40 < possession_home < 60:
                    boost += 2
            # Analisi per Double Chance
            elif market in ['1x', 'x2']:
                if market == '1x' and score_home > score_away:
                    boost += 5
                elif market == 'x2' and score_away > score_home:
                    boost += 5
                # Se domina statisticamente
                if market == '1x' and shots_home > shots_away * 1.5:
                    boost += 3
                elif market == 'x2' and shots_away > shots_home * 1.5:
                    boost += 3
            # Analisi per Corner
            elif 'corner' in market.lower():
                corners = live_data.get('corners_home', 0) + live_data.get('corners_away', 0)
                minute = live_data.get('minute', 0)
                if minute > 0:
                    corners_per_minute = corners / minute
                    if corners_per_minute > 0.1:  # >9 corner/90min
                        boost += 5
            # Analisi per Cartellini
            elif 'card' in market.lower():
                yellows = live_data.get('yellow_cards_home', 0) + live_data.get('yellow_cards_away', 0)
                minute = live_data.get('minute', 0)
                if minute > 0 and yellows >= 3:
                    boost += 5
            # Analisi per BTTS
            elif 'btts' in market.lower():
                if shots_on_target_home >= 2 and shots_on_target_away >= 2:
                    boost += 5
                # Se una squadra ha già segnato e l'altra ha tiri in porta
                if (score_home > 0 and shots_on_target_away >= 2) or (score_away > 0 and shots_on_target_home >= 2):
                    boost += 3
            # Analisi per DNB
            elif 'dnb' in market.lower():
                # Se domina statisticamente ma perde
                if 'home' in market.lower() and shots_home > shots_away * 1.5 and score_home <= score_away:
                    boost += 5
                elif 'away' in market.lower() and shots_away > shots_home * 1.5 and score_away <= score_home:
                    boost += 5
            # Analisi per Goal Range
            elif 'goal_range' in market.lower():
                if '4_plus' in market.lower():
                    # Se partita è molto aperta
                    if total_shots > 20 and minute >= 40:
                        boost += 5
                    if minute > 0:
                        goals_per_minute = total_goals / minute
                        if goals_per_minute > 0.04:
                            boost += 3
                elif '2_3' in market.lower():
                    # Se partita è equilibrata
                    if 1 <= total_goals <= 2 and abs(score_home - score_away) <= 1:
                        boost += 3
            # Analisi per Clean Sheet
            elif 'clean_sheet' in market.lower():
                goal_diff = abs(score_home - score_away)
                # Boost positivo se squadra in vantaggio e avversaria senza tiri in porta
                if 'home' in market.lower() and score_home > 0 and score_away == 0:
                    if shots_on_target_away == 0:
                        boost += 5  # Nessun tiro in porta = alta probabilità clean sheet
                    if goal_diff == 1 and minute >= 60:  # 1-0 avanzato = buona probabilità
                        boost += 3
                    # NON dare boost se già 3-0 o più (banale)
                    if goal_diff >= 3:
                        boost = 0  # Reset boost se banale
                    # 🆕 OTTIMIZZATO: Limita boost se partita già decisa (2-0 o più)
                    elif goal_diff >= 2 and minute >= 70:
                        boost = min(5, boost)  # Max +5% se partita decisa
                elif 'away' in market.lower() and score_away > 0 and score_home == 0:
                    if shots_on_target_home == 0:
                        boost += 5  # Nessun tiro in porta = alta probabilità clean sheet
                    if goal_diff == 1 and minute >= 60:  # 0-1 avanzato = buona probabilità
                        boost += 3
                    # NON dare boost se già 0-3 o più (banale)
                    if goal_diff >= 3:
                        boost = 0  # Reset boost se banale
                    # 🆕 OTTIMIZZATO: Limita boost se partita già decisa (2-0 o più)
                    elif goal_diff >= 2 and minute >= 70:
                        boost = min(5, boost)  # Max +5% se partita decisa
                # 🆕 OTTIMIZZATO: Limita boost totale per clean sheet a +10%
                boost = min(10, boost)
            
            return min(20, boost)
        except Exception as e:
            logger.debug(f"⚠️  Errore AI market confidence: {e}")
            return 0.0
    
    def _filter_obvious_opportunities(
        self,
        opportunities: List[LiveBettingOpportunity],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        Filtra opportunità banali/ovvie.
        
        Rimuove suggerimenti come:
        - 1X quando è già 1-0 (ovvio!)
        - X2 quando è già 0-1 (ovvio!)
        - Over 0.5 quando è già 1-0 (ovvio!)
        - Segno 1 quando è già 1-0 (ovvio!)
        """
        filtered = []
        score_home = live_data.get('score_home', 0)
        score_away = live_data.get('score_away', 0)
        minute = live_data.get('minute', 0)
        total_goals = score_home + score_away  # 🆕 FIX: Definisci total_goals all'inizio
        
        for opp in opportunities:
            market = opp.market.lower()
            situation = opp.situation.lower()
            
            # FILTRO 1: 1X quando è già 1-0 o più (BANALE!)
            if market == '1x' and score_home > score_away:
                logger.debug(f"⏭️  Saltata opportunità banale: 1X quando è già {score_home}-{score_away}")
                continue
            
            # FILTRO 2: X2 quando è già 0-1 o più (BANALE!)
            if market == 'x2' and score_away > score_home:
                logger.debug(f"⏭️  Saltata opportunità banale: X2 quando è già {score_home}-{score_away}")
                continue
            
            # FILTRO 3: Over 0.5 quando c'è già almeno 1 gol (BANALE!)
            if 'over_0.5' in market and (score_home + score_away) >= 1:
                logger.debug(f"⏭️  Saltata opportunità banale: Over 0.5 quando ci sono già {score_home + score_away} gol")
                continue
            
            # FILTRO 4: Segno 1 quando è già 1-0 o più (BANALE!)
            # 🆕 FIX CRITICO: Blocca anche quando la casa è in SVANTAGGIO (es. 1-7, 0-5, etc.)
            if market in ['1x2_home', 'home_win']:
                # Caso 1: Casa in vantaggio
                if score_home > score_away:
                    goal_diff = score_home - score_away
                    # Se differenza >= 2 gol, è troppo sbilanciato per essere un ribaltone realistico
                    if goal_diff >= 2:
                        logger.debug(f"⏭️  Saltata opportunità banale: Segno 1 quando è già {score_home}-{score_away} (differenza {goal_diff} gol) - troppo sbilanciato")
                        continue
                    # Se differenza = 1 ma siamo oltre 60', è comunque banale
                    elif minute >= 60:
                        logger.debug(f"⏭️  Saltata opportunità banale: Segno 1 quando è già {score_home}-{score_away} al {minute}'")
                        continue
                # Caso 2: Casa in SVANTAGGIO (BUG FIX!)
                elif score_home < score_away:
                    goal_diff = score_away - score_home
                    # Se la casa è in svantaggio di 2+ gol, è IMPOSSIBILE che vinca (es. 1-7, 0-5)
                    if goal_diff >= 2:
                        logger.debug(f"⏭️  Saltata opportunità IMPOSSIBILE: Segno 1 quando è già {score_home}-{score_away} (casa in svantaggio di {goal_diff} gol) - IMPOSSIBILE!")
                        continue
                    # Se differenza = 1 ma siamo oltre 70', è praticamente impossibile
                    elif goal_diff == 1 and minute >= 70:
                        logger.debug(f"⏭️  Saltata opportunità banale: Segno 1 quando è già {score_home}-{score_away} al {minute}' (casa in svantaggio)")
                        continue
            
            # FILTRO 5: Segno 2 quando è già 0-1 o più (BANALE!)
            # 🆕 FIX CRITICO: Blocca anche quando l'ospite è in SVANTAGGIO (es. 7-1, 5-0, etc.)
            if market in ['1x2_away', 'away_win']:
                # Caso 1: Ospite in vantaggio
                if score_away > score_home:
                    goal_diff = score_away - score_home
                    # Se differenza >= 2 gol, è troppo sbilanciato per essere un ribaltone realistico
                    if goal_diff >= 2:
                        logger.debug(f"⏭️  Saltata opportunità banale: Segno 2 quando è già {score_home}-{score_away} (differenza {goal_diff} gol) - troppo sbilanciato")
                        continue
                    # Se differenza = 1 ma siamo oltre 60', è comunque banale
                    elif minute >= 60:
                        logger.debug(f"⏭️  Saltata opportunità banale: Segno 2 quando è già {score_home}-{score_away} al {minute}'")
                        continue
                # Caso 2: Ospite in SVANTAGGIO (BUG FIX!)
                elif score_away < score_home:
                    goal_diff = score_home - score_away
                    # Se l'ospite è in svantaggio di 2+ gol, è IMPOSSIBILE che vinca (es. 7-1, 5-0)
                    if goal_diff >= 2:
                        logger.debug(f"⏭️  Saltata opportunità IMPOSSIBILE: Segno 2 quando è già {score_home}-{score_away} (ospite in svantaggio di {goal_diff} gol) - IMPOSSIBILE!")
                        continue
                    # Se differenza = 1 ma siamo oltre 70', è praticamente impossibile
                    elif goal_diff == 1 and minute >= 70:
                        logger.debug(f"⏭️  Saltata opportunità banale: Segno 2 quando è già {score_home}-{score_away} al {minute}' (ospite in svantaggio)")
                        continue
            
            # FILTRO 6: Over 1.5 quando ci sono già 2+ gol (BANALE!)
            if 'over_1.5' in market and (score_home + score_away) >= 2:
                logger.debug(f"⏭️  Saltata opportunità banale: Over 1.5 quando ci sono già {score_home + score_away} gol")
                continue
            
            # FILTRO 7: Over 2.5 quando ci sono già 3+ gol (BANALE!)
            if 'over_2.5' in market and (score_home + score_away) >= 3:
                logger.debug(f"⏭️  Saltata opportunità banale: Over 2.5 quando ci sono già {score_home + score_away} gol")
                continue
            
            # FILTRO 8: Over 3.5 quando ci sono già 4+ gol (BANALE!)
            if 'over_3.5' in market and (score_home + score_away) >= 4:
                logger.debug(f"⏭️  Saltata opportunità banale: Over 3.5 quando ci sono già {score_home + score_away} gol")
                continue
            
            # FILTRO 9: Under 3.5 quando è 3-0 all'85' (BANALE - ESEMPIO UTENTE!)
            if 'under_3.5' in market and minute >= 80:
                if (score_home + score_away) == 3:
                    logger.debug(f"⏭️  Saltata opportunità banale: Under 3.5 quando è {score_home}-{score_away} all'{minute}'")
                    continue
            
            # FILTRO 10: Under 2.5 quando è 2-0 all'85' (BANALE!)
            if 'under_2.5' in market and minute >= 80:
                if (score_home + score_away) == 2:
                    logger.debug(f"⏭️  Saltata opportunità banale: Under 2.5 quando è {score_home}-{score_away} all'{minute}'")
                    continue
            
            # 🆕 FILTRO 10B: Under 1.5 quando c'è già 1 gol e siamo oltre 45' (ILLOGICO!)
            # Se è 1-0 al 50', under 1.5 significa che non ci saranno altri gol - troppo rischioso e illogico
            if 'under_1.5' in market and not 'ht' in market:  # Solo per under_1.5 generale, non HT
                total_goals = score_home + score_away
                if total_goals >= 1 and minute >= 45:
                    logger.debug(f"⏭️  Saltata opportunità illogica: Under 1.5 quando è già {score_home}-{score_away} (1+ gol) al {minute}' - troppo rischioso")
                    continue
                # Se è 1-0 o 0-1 e siamo oltre 50', è ancora più illogico
                if total_goals == 1 and minute >= 50:
                    logger.debug(f"⏭️  Saltata opportunità illogica: Under 1.5 quando è già {score_home}-{score_away} (1 gol) al {minute}' - partita già aperta")
                    continue
            
            # FILTRO 11: Partita già decisa (differenza >= 3 gol) - NO opportunità
            # 🆕 FIX CRITICO: Blocca a QUALSIASI minuto se differenza >= 3 gol (non solo >= 70')
            goal_diff = abs(score_home - score_away)
            if goal_diff >= 3:
                # Partita già decisa, NON suggerire NESSUN mercato su risultato
                # Blocca TUTTI i mercati che riguardano il risultato finale
                result_markets = ['home_win', 'away_win', 'match_winner', 'dnb_home', 'dnb_away', 
                                 '1x', 'x2', '1x2_home', '1x2_away', '1x2_draw',
                                 'exact_score', 'double_chance', 'ribaltone', 'comeback']
                if any(m in market for m in result_markets):
                    logger.debug(f"⏭️  Saltata opportunità: Partita già decisa ({score_home}-{score_away} al {minute}', diff: {goal_diff} gol) - BLOCCATO TUTTI I MERCATI RISULTATO")
                    continue
                # Se differenza >= 4 gol, blocca ANCHE altri mercati (partita completamente decisa)
                if goal_diff >= 4 and minute >= 50:
                    logger.debug(f"⏭️  Saltata opportunità: Partita completamente decisa ({score_home}-{score_away} al {minute}', diff: {goal_diff} gol) - BLOCCATO TUTTI I MERCATI")
                    continue
            
            # FILTRO 12: Minuto troppo avanzato per Over (oltre 85')
            if 'over' in market and minute >= 85:
                # Troppo tardi per Over, probabilità molto basse
                logger.debug(f"⏭️  Saltata opportunità: Over troppo tardi (minuto {minute}')")
                continue
            
            # 🆕 FILTRO 12B: BLOCCA TUTTI I MERCATI SU RISULTATO FINALE AL 90'+
            # FIX CRITICO: Al 90' è troppo tardi per suggerire vittorie (partita sta finendo!)
            if minute >= 88:
                result_final_markets = ['home_win', 'away_win', 'match_winner', '1x2_home', '1x2_away', 
                                       '1x2_draw', 'dnb_home', 'dnb_away', 'ribaltone', 'comeback',
                                       'double_chance', '1x', 'x2', '12']
                if any(m in market for m in result_final_markets):
                    logger.debug(f"⏭️  Saltata opportunità IMPOSSIBILE: Mercato risultato finale al {minute}' (partita sta finendo!) - {market}")
                    continue
            
            # 🆕 FILTRO 12C: BLOCCA MERCATI SU RISULTATO FINALE AL 85'+ SE PAREGGIO
            # FIX CRITICO: Al 85'+ su pareggio (es. 2-2), suggerire vittorie è troppo rischioso
            if minute >= 85 and score_home == score_away:
                result_final_markets = ['home_win', 'away_win', 'match_winner', '1x2_home', '1x2_away']
                if any(m in market for m in result_final_markets):
                    logger.debug(f"⏭️  Saltata opportunità RISCHIOSA: Mercato risultato finale al {minute}' su pareggio {score_home}-{score_away} - {market}")
                    continue
            
            # FILTRO 13: Quota troppo bassa (no valore) - DINAMICO per mercato
            if opp.odds:
                # 🆕 OTTIMIZZATO: Filtro quota dinamico basato su mercato
                min_odds = 1.3  # Default
                if 'clean_sheet' in market:
                    min_odds = 1.5  # Clean sheet richiede quota più alta
                elif 'exact_score' in market:
                    min_odds = 2.0  # Exact score richiede quota alta
                elif 'win_to_nil' in market:
                    min_odds = 1.5  # Win to nil richiede quota più alta
                elif 'under' in market and minute >= 80:
                    min_odds = 1.2  # Under avanzato può avere quota più bassa
                
                if opp.odds < min_odds:
                    logger.debug(f"⏭️  Saltata opportunità: Quota troppo bassa per {market} ({opp.odds:.2f} < {min_odds:.2f})")
                    continue
                
                # 🆕 OTTIMIZZATO: Filtro quota troppo alta (troppo rischiosa)
                max_odds = 8.0  # Quote >8.0 sono troppo rischiose
                if opp.odds > max_odds:
                    logger.debug(f"⏭️  Saltata opportunità: Quota troppo alta per {market} ({opp.odds:.2f} > {max_odds:.2f})")
                    continue
            
            # FILTRO 14: Double chance banali (già gestito in _check_double_chance_markets, ma doppio controllo)
            if 'double_chance' in situation and not ('comeback' in situation or 'dominance' in situation):
                # Se non è un comeback o dominance, potrebbe essere banale
                if (market == '1x' and score_home >= score_away) or (market == 'x2' and score_away >= score_home):
                    logger.debug(f"⏭️  Saltata opportunità banale: {market} senza valore reale")
                    continue
            
            # FILTRO 15: Exact Score quando partita non è chiusa
            if 'exact_score' in market and minute < 75:
                # Troppo presto per exact score
                logger.debug(f"⏭️  Saltata opportunità: Exact score troppo presto (minuto {minute}')")
                continue
            
            # FILTRO 16: Goal Range incoerente
            if 'goal_range_0_1' in market and (score_home + score_away) > 1:
                logger.debug(f"⏭️  Saltata opportunità banale: Goal range 0-1 quando ci sono già {score_home + score_away} gol")
                continue
            
            if 'goal_range_2_3' in market and ((score_home + score_away) < 2 or (score_home + score_away) > 3):
                logger.debug(f"⏭️  Saltata opportunità banale: Goal range 2-3 quando ci sono già {score_home + score_away} gol")
                continue
            
            # FILTRO 17: Goal Range 4+ quando ci sono già 4 gol oltre 80' (BANALE!)
            if 'goal_range_4_plus' in market and (score_home + score_away) == 4 and minute >= 80:
                logger.debug(f"⏭️  Saltata opportunità banale: Goal range 4+ quando ci sono già 4 gol all'{minute}' (range già raggiunto!)")
                continue
            
            # FILTRO 18: Goal Range 4+ quando ci sono già 4 gol e partita non è molto aperta
            if 'goal_range_4_plus' in market and (score_home + score_away) == 4:
                shots_home = live_data.get('shots_home', 0)
                shots_away = live_data.get('shots_away', 0)
                total_shots = shots_home + shots_away
                shots_per_minute = total_shots / minute if minute > 0 else 0
                if shots_per_minute < 0.3:  # Partita non molto aperta
                    logger.debug(f"⏭️  Saltata opportunità banale: Goal range 4+ quando ci sono già 4 gol ma partita non molto aperta (tiri/min: {shots_per_minute:.2f})")
                    continue
            
            # FILTRO 19: Clean Sheet quando risultato è già 3-0 o più al 75' (BANALE!)
            if 'clean_sheet' in market:
                goal_diff = abs(score_home - score_away)
                # Se risultato è 3-0 o più e siamo al 75' o oltre, clean sheet è troppo ovvio
                if goal_diff >= 3 and minute >= 75:
                    logger.debug(f"⏭️  Saltata opportunità banale: Clean sheet quando risultato è già {score_home}-{score_away} al {minute}' (troppo ovvio, partita decisa)")
                    continue
                # 🆕 OTTIMIZZATO: Se risultato è 2-0 o più e siamo oltre 75' (non solo 80'), clean sheet è molto probabile (banale)
                if goal_diff >= 2 and minute >= 75:
                    logger.debug(f"⏭️  Saltata opportunità banale: Clean sheet quando risultato è già {score_home}-{score_away} al {minute}' (troppo tardi, partita praticamente decisa)")
                    continue
            
            # FILTRO 20: Under HT banali quando siamo troppo avanti nel primo tempo (BANALE!)
            if 'ht' in market.lower() and 'under' in market.lower():
                total_goals = score_home + score_away
                # Under 0.5 HT al 44' quando è 0-0 è BANALE (troppo ovvio!)
                if 'under_0.5_ht' in market and minute >= 40 and total_goals == 0:
                    logger.debug(f"⏭️  Saltata opportunità banale: Under 0.5 HT al {minute}' quando è {score_home}-{score_away} (troppo ovvio, primo tempo quasi finito)")
                    continue
                # Under 0.5 HT al 42' o oltre quando è 0-0 è BANALE
                if 'under_0.5_ht' in market and minute >= 42 and total_goals == 0:
                    logger.debug(f"⏭️  Saltata opportunità banale: Under 0.5 HT al {minute}' quando è {score_home}-{score_away} (troppo tardi, primo tempo quasi finito)")
                    continue
                # Under 1.5 HT al 44' quando c'è 0 o 1 gol è BANALE
                if 'under_1.5_ht' in market and minute >= 42 and total_goals <= 1:
                    logger.debug(f"⏭️  Saltata opportunità banale: Under 1.5 HT al {minute}' quando ci sono {total_goals} gol (troppo tardi, primo tempo quasi finito)")
                    continue
            
            # FILTRO 21: Over HT banali quando siamo troppo avanti nel primo tempo (BANALE!)
            if 'ht' in market.lower() and 'over' in market.lower():
                total_goals = score_home + score_away
                # Over 0.5 HT al 44' quando c'è già almeno 1 gol è BANALE (già superato!)
                if 'over_0.5_ht' in market and minute >= 40 and total_goals >= 1:
                    logger.debug(f"⏭️  Saltata opportunità banale: Over 0.5 HT al {minute}' quando ci sono già {total_goals} gol (già superato!)")
                    continue
                # Over 1.5 HT al 44' quando ci sono già 2+ gol è BANALE (già superato!)
                if 'over_1.5_ht' in market and minute >= 40 and total_goals >= 2:
                    logger.debug(f"⏭️  Saltata opportunità banale: Over 1.5 HT al {minute}' quando ci sono già {total_goals} gol (già superato!)")
                    continue
            
            # 🆕 FILTRO 22: BTTS Yes quando è troppo tardi (oltre 85') - ILLOGICO!
            if 'btts_yes' in market and minute >= 85:
                # Se una squadra non ha ancora segnato e siamo oltre 85', BTTS è quasi impossibile
                if score_home == 0 or score_away == 0:
                    logger.debug(f"⏭️  Saltata opportunità illogica: BTTS Yes quando è {score_home}-{score_away} al {minute}' - troppo tardi")
                    continue
            
            # 🆕 FILTRO 22B: BTTS Yes quando entrambe hanno già segnato - BANALE!
            if 'btts_yes' in market and score_home > 0 and score_away > 0:
                logger.debug(f"⏭️  Saltata opportunità banale: BTTS Yes quando entrambe hanno già segnato ({score_home}-{score_away})")
                continue
            
            # 🆕 FILTRO 22C: BTTS No quando una squadra ha già segnato e siamo avanzati - ILLOGICO!
            if 'btts_no' in market and minute >= 80:
                # Se una squadra ha già segnato e siamo oltre 80', BTTS No è già perso
                if score_home > 0 or score_away > 0:
                    logger.debug(f"⏭️  Saltata opportunità illogica: BTTS No quando è {score_home}-{score_away} al {minute}' - già perso")
                    continue
            
            # 🆕 FILTRO 23: Exact Score quando suggerisce lo score attuale - BANALE!
            if 'exact_score' in market:
                # Estrai score dal market (es. "exact_score_2-0")
                import re
                score_match = re.search(r'exact_score_(\d+)-(\d+)', market)
                if score_match:
                    market_score_home = int(score_match.group(1))
                    market_score_away = int(score_match.group(2))
                    # Se suggerisce lo score attuale, è banale
                    if market_score_home == score_home and market_score_away == score_away and minute >= 70:
                        logger.debug(f"⏭️  Saltata opportunità banale: Exact Score {market_score_home}-{market_score_away} quando è già {score_home}-{score_away} al {minute}'")
                        continue
            
            # 🆕 FILTRO 24: Goal Range illogico (es. Goal Range 0-1 quando c'è già 1 gol al 60')
            if 'goal_range_0_1' in market:
                total_goals = score_home + score_away
                # Se c'è già 1 gol e siamo oltre 60', Goal Range 0-1 è illogico (è già 1 gol, quindi se segna un altro è perso)
                if total_goals == 1 and minute >= 60:
                    logger.debug(f"⏭️  Saltata opportunità illogica: Goal Range 0-1 quando è già {score_home}-{score_away} (1 gol) al {minute}' - troppo rischioso")
                    continue
            
            # 🆕 FILTRO 25: Goal Range 2-3 quando ci sono già 4+ gol - ILLOGICO!
            if 'goal_range_2_3' in market:
                total_goals = score_home + score_away
                if total_goals >= 4:
                    logger.debug(f"⏭️  Saltata opportunità illogica: Goal Range 2-3 quando ci sono già {total_goals} gol ({score_home}-{score_away})")
                    continue
            
            # 🆕 FILTRO 26: Goal Range 4+ quando ci sono già 4+ gol e siamo oltre 80' - BANALE!
            if 'goal_range_4_plus' in market or 'goal_range_4+' in market:
                total_goals = score_home + score_away
                if total_goals >= 4 and minute >= 80:
                    logger.debug(f"⏭️  Saltata opportunità banale: Goal Range 4+ quando ci sono già {total_goals} gol ({score_home}-{score_away}) al {minute}'")
                    continue
            
            # 🆕 FILTRO 27: Odd/Even banale quando è troppo tardi (oltre 85')
            if 'total_goals_odd' in market or 'total_goals_even' in market:
                total_goals = score_home + score_away
                if minute >= 85:
                    # Se è già dispari/pari e siamo oltre 85', suggerire lo stesso è banale
                    is_odd = total_goals % 2 == 1
                    if ('odd' in market and is_odd) or ('even' in market and not is_odd):
                        logger.debug(f"⏭️  Saltata opportunità banale: {'Odd' if 'odd' in market else 'Even'} quando è già {total_goals} gol ({score_home}-{score_away}) al {minute}'")
                        continue
            
            # 🆕 FILTRO 28: Time of Next Goal quando è troppo tardi (oltre 85')
            if 'next_goal' in market and minute >= 85:
                logger.debug(f"⏭️  Saltata opportunità banale: Time of Next Goal quando siamo al {minute}' - troppo tardi")
                continue
            
            # 🆕 FILTRO 29: Team to Score Next quando è troppo tardi (oltre 85') o partita decisa
            if 'team_to_score_next' in market:
                goal_diff = abs(score_home - score_away)
                if minute >= 85:
                    logger.debug(f"⏭️  Saltata opportunità banale: Team to Score Next quando siamo al {minute}' - troppo tardi")
                    continue
                # Se partita è già decisa (3+ gol di differenza), è banale
                if goal_diff >= 3 and minute >= 70:
                    logger.debug(f"⏭️  Saltata opportunità banale: Team to Score Next quando partita è già decisa ({score_home}-{score_away}) al {minute}'")
                    continue
            
            # 🆕 FILTRO 30: Win To Nil quando è già 2-0 o più al 75' - BANALE!
            if 'win_to_nil' in market:
                goal_diff = abs(score_home - score_away)
                if goal_diff >= 2 and minute >= 75:
                    logger.debug(f"⏭️  Saltata opportunità banale: Win To Nil quando è già {score_home}-{score_away} (diff: {goal_diff} gol) al {minute}' - troppo ovvio")
                    continue
            
            # 🆕 FILTRO 31: Second Half Over quando c'è già 1+ gol nel secondo tempo e siamo oltre 80'
            if 'second_half' in market and 'over' in market:
                # Stima gol nel secondo tempo (assumendo che al 45' ci fossero X gol)
                # Per semplicità, se siamo oltre 60' e ci sono già 2+ gol totali, probabilmente c'è già 1+ nel secondo tempo
                total_goals = score_home + score_away
                if minute >= 80 and total_goals >= 2:
                    logger.debug(f"⏭️  Saltata opportunità banale: Second Half Over quando è già {score_home}-{score_away} ({total_goals} gol) al {minute}' - probabilmente già superato")
                    continue
            
            # 🆕 FILTRO 32: DNB quando partita è già decisa (3+ gol di differenza)
            if 'dnb' in market:
                goal_diff = abs(score_home - score_away)
                if goal_diff >= 3 and minute >= 70:
                    logger.debug(f"⏭️  Saltata opportunità banale: DNB quando partita è già decisa ({score_home}-{score_away}, diff: {goal_diff} gol) al {minute}'")
                    continue
            
            # 🆕 FILTRO 33: Team to Score First quando NON è 0-0 (BANALE!)
            if 'team_to_score_first' in market:
                if score_home > 0 or score_away > 0:
                    logger.debug(f"⏭️  Saltata opportunità IMPOSSIBILE: Team to Score First quando è già {score_home}-{score_away} (BANALE!)")
                    continue
                # Se siamo oltre 40', è troppo tardi
                if minute >= 40:
                    logger.debug(f"⏭️  Saltata opportunità banale: Team to Score First al {minute}' (troppo tardi)")
                    continue
            
            # 🆕 FILTRO 34: Team to Score Last quando partita è già decisa o troppo tardi
            if 'team_to_score_last' in market:
                goal_diff = abs(score_home - score_away)
                if goal_diff >= 3:
                    logger.debug(f"⏭️  Saltata opportunità banale: Team to Score Last quando partita è già decisa ({score_home}-{score_away})")
                    continue
                if minute >= 88:
                    logger.debug(f"⏭️  Saltata opportunità banale: Team to Score Last al {minute}' (troppo tardi)")
                    continue
            
            # 🆕 FILTRO 35: Highest Scoring Half quando siamo ancora nel primo tempo (BANALE!)
            if 'highest_scoring_half' in market:
                if minute < 50:
                    logger.debug(f"⏭️  Saltata opportunità banale: Highest Scoring Half al {minute}' (troppo presto, primo tempo non finito)")
                    continue
                if minute >= 85:
                    logger.debug(f"⏭️  Saltata opportunità banale: Highest Scoring Half al {minute}' (troppo tardi)")
                    continue
                # 🆕 BLOCCA se risultato già definito (es. 1-2 al 64' = primo tempo ha più gol, BANALE!)
                events = live_data.get('events', [])
                ht_goals = 0
                st_goals = 0
                for event in events:
                    event_type = event.get('type', '').lower()
                    event_minute = event.get('minute', 0)
                    if event_type in ['goal', 'goal penalty', 'goal own']:
                        if event_minute <= 45:
                            ht_goals += 1
                        elif event_minute > 45:
                            st_goals += 1
                # Se abbiamo dati reali e la differenza è chiara, blocca
                if events and ht_goals > 0:
                    if '1h' in market and ht_goals >= 2 and st_goals == 0:
                        logger.debug(f"⏭️  Saltata opportunità banale: Highest Scoring Half 1H su {score_home}-{score_away} al {minute}' (primo tempo ha {ht_goals} gol, secondo {st_goals} - OVVIO!)")
                        continue
                    if '2h' in market and st_goals >= 2 and ht_goals == 0:
                        logger.debug(f"⏭️  Saltata opportunità banale: Highest Scoring Half 2H su {score_home}-{score_away} al {minute}' (secondo tempo ha {st_goals} gol, primo {ht_goals} - OVVIO!)")
                        continue
                # Se risultato è 1-2 o 2-1 al 64'+, è ovvio che primo tempo ha più gol
                if total_goals >= 3 and minute >= 60:
                    if '1h' in market:
                        logger.debug(f"⏭️  Saltata opportunità banale: Highest Scoring Half 1H su {score_home}-{score_away} al {minute}' (3+ gol totali, primo tempo probabilmente più prolifico - BANALE!)")
                        continue
            
            # 🆕 FILTRO 36: Win Either Half quando partita è già decisa
            if 'win_either_half' in market:
                goal_diff = abs(score_home - score_away)
                if goal_diff >= 3:
                    logger.debug(f"⏭️  Saltata opportunità banale: Win Either Half quando partita è già decisa ({score_home}-{score_away})")
                    continue
                if minute >= 80:
                    logger.debug(f"⏭️  Saltata opportunità banale: Win Either Half al {minute}' (troppo tardi)")
                    continue
            
            # 🆕 FILTRO 39: Match Winner (1X2) quando risultato è già definito (BANALE!)
            if any(x in market for x in ['home_win', 'away_win', 'match_winner']):
                goal_diff = abs(score_home - score_away)
                # Se risultato è 1-0 al 52' e suggerisce il 2, è BANALE (squadra in vantaggio)
                if score_home > score_away and goal_diff >= 1 and minute >= 50:
                    if 'away_win' in market or '2' in market:
                        logger.debug(f"⏭️  Saltata opportunità banale: Away Win su {score_home}-{score_away} al {minute}' (casa in vantaggio - BANALE!)")
                        continue
                if score_away > score_home and goal_diff >= 1 and minute >= 50:
                    if 'home_win' in market or '1' in market:
                        logger.debug(f"⏭️  Saltata opportunità banale: Home Win su {score_home}-{score_away} al {minute}' (trasferta in vantaggio - BANALE!)")
                        continue
                # Se risultato è 2-0 o più, non suggerire la squadra in svantaggio
                if goal_diff >= 2 and minute >= 60:
                    if (score_home > score_away and ('away_win' in market or '2' in market)) or \
                       (score_away > score_home and ('home_win' in market or '1' in market)):
                        logger.debug(f"⏭️  Saltata opportunità banale: Match Winner su {score_home}-{score_away} al {minute}' (partita già decisa - BANALE!)")
                        continue
            
            # 🆕 FILTRO 37: BTTS First Half quando NON siamo nel primo tempo (BANALE!)
            if 'btts_first_half' in market:
                if minute >= 45:
                    logger.debug(f"⏭️  Saltata opportunità IMPOSSIBILE: BTTS First Half quando primo tempo è già finito (al {minute}')")
                    continue
                # Se entrambe hanno già segnato, è banale
                if score_home > 0 and score_away > 0:
                    logger.debug(f"⏭️  Saltata opportunità banale: BTTS First Half quando è già {score_home}-{score_away} (già BTTS!)")
                    continue
            
            # 🆕 FILTRO 38: Half Time Result quando NON siamo nel primo tempo (BANALE!)
            if 'half_time_result' in market:
                if minute >= 45:
                    logger.debug(f"⏭️  Saltata opportunità IMPOSSIBILE: Half Time Result quando primo tempo è già finito (al {minute}')")
                    continue
                # Se siamo oltre 42', è troppo tardi
                if minute >= 42:
                    logger.debug(f"⏭️  Saltata opportunità banale: Half Time Result al {minute}' (troppo tardi, primo tempo quasi finito)")
                    continue
            
            # Se passa tutti i filtri, è un'opportunità seria
            filtered.append(opp)
        
        return filtered
    
    def _apply_market_specific_rules(
        self,
        opportunities: List[LiveBettingOpportunity],
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """Applica regole addizionali specifiche per mercato per evitare segnali banali."""
        filtered = []
        minute = live_data.get('minute', 0)
        score_home = live_data.get('score_home', 0)
        score_away = live_data.get('score_away', 0)
        total_goals = score_home + score_away
        
        for opp in opportunities:
            market = opp.market.lower()
            reason = None
            
            # Evita Under banali (es. Under 3.5 a 3-0 all'85')
            if 'under' in market:
                goal_line = self._extract_goal_line(market)
                if goal_line is not None:
                    if total_goals >= goal_line - 0.5 and minute >= 75:
                        reason = f"Under {goal_line} banale ({score_home}-{score_away} al {minute}')"
            
            # Evita Over quando già superato goal line
            if not reason and 'over' in market:
                goal_line = self._extract_goal_line(market)
                if goal_line is not None and total_goals >= goal_line:
                    reason = f"Over {goal_line} superato (score {score_home}-{score_away})"
            
            # Evita DNB/match winner se squadra già vince 2+ gol
            if not reason and market.startswith(('dnb_', 'home_win', 'away_win', 'match_winner')):
                if market.startswith(('dnb_home', 'home_win')) and score_home - score_away >= 1:
                    reason = "DNB/1 banale (home già in vantaggio)"
                elif market.startswith(('dnb_away', 'away_win')) and score_away - score_home >= 1:
                    reason = "DNB/2 banale (away già in vantaggio)"
            
            # Evita Goal Range banali (già determinato)
            if not reason and market.startswith('goal_range_'):
                if market == 'goal_range_0_1' and (total_goals > 1 or minute < 30):
                    reason = "Goal range 0-1 non coerente"
                if market == 'goal_range_4_plus' and total_goals == 4 and minute >= 80:
                    reason = f"Goal range 4+ banale (già 4 gol all'{minute}')"
                if market == 'goal_range_4_plus' and total_goals < 3:
                    reason = "Goal range 4+ prematuro"
            
            # Evita Clean Sheet banali (partita già decisa)
            if not reason and 'clean_sheet' in market:
                goal_diff = abs(score_home - score_away)
                # Se risultato è 3-0 o più al 75' o oltre, clean sheet è troppo ovvio
                if goal_diff >= 3 and minute >= 75:
                    reason = f"Clean sheet banale (risultato {score_home}-{score_away} al {minute}', partita decisa)"
                # 🆕 OTTIMIZZATO: Se risultato è 2-0 o più oltre 75' (non solo 80'), clean sheet è molto probabile (banale)
                elif goal_diff >= 2 and minute >= 75:
                    reason = f"Clean sheet banale (risultato {score_home}-{score_away} al {minute}', troppo tardi)"
            
            if reason:
                logger.debug(f"⏭️  Opportunità {market} filtrata (motivo: {reason})")
                continue
            
            filtered.append(opp)
        
        return filtered
    
    def _apply_market_min_confidence(
        self,
        opportunities: List[LiveBettingOpportunity]
    ) -> List[LiveBettingOpportunity]:
        """Applica confidence minima per mercato."""
        filtered = []
        for opp in opportunities:
            market = opp.market.lower()
            min_conf = self._get_market_specific_threshold(market)
            if min_conf and opp.confidence < min_conf:
                logger.debug(
                    f"⏭️  Opportunità {market} filtrata: confidence {opp.confidence:.0f}% < threshold {min_conf:.0f}%"
                )
                continue
            filtered.append(opp)
        return filtered
    
    def _get_market_specific_threshold(self, market: str) -> Optional[float]:
        """Ritorna confidence minima per mercato (match su prefisso)."""
        for key, value in self.market_min_confidence.items():
            if market.startswith(key):
                return value
        return None
    
    def _calculate_ev_from_values(self, confidence: float, odds: float) -> float:
        """Utility per calcolare l'EV (%) partendo da confidence e quota."""
        if not odds or odds <= 0:
            return 0.0
        ev_decimal = (confidence / 100.0) * odds - 1.0
        return ev_decimal * 100.0
    
    def _calculate_expected_value(self, opportunity: LiveBettingOpportunity) -> float:
        """
        🆕 Calcola Expected Value (EV) per un'opportunità.
        EV = (confidence/100) * odds - 1
        Valore positivo = opportunità con valore
        """
        return self._calculate_ev_from_values(opportunity.confidence, opportunity.odds)
    
    def _filter_by_expected_value(self, opportunities: List[LiveBettingOpportunity]) -> List[LiveBettingOpportunity]:
        """
        🆕 Filtra solo opportunità con Expected Value MOLTO negativo (non tutte quelle negative).
        Permette segnali con alta confidence anche se EV leggermente negativo.
        """
        filtered = []
        for opp in opportunities:
            ev = getattr(opp, 'ev', None)
            if ev is None:
                ev = self._calculate_expected_value(opp)
                opp.ev = ev
            if ev < self.min_ev:
                logger.info(
                    f"⏭️  Saltata opportunità: valore atteso {ev:.1f}% < soglia {self.min_ev:.1f}% per {opp.market} (confidence: {opp.confidence:.1f}%, odds: {opp.odds:.2f})"
                )
                continue
            filtered.append(opp)
        return filtered
    
    def _calculate_combined_score(self, opportunity: LiveBettingOpportunity) -> float:
        """
        🆕 Calcola score combinato di Expected Value e Confidence.
        Formula: (EV * 0.4) + (confidence/100 * 0.6)
        Questo bilancia valore e probabilità.
        """
        ev = getattr(opportunity, 'ev', None)
        if ev is None:
            ev = self._calculate_expected_value(opportunity)
            opportunity.ev = ev
        confidence_score = opportunity.confidence / 100.0
        
        # Normalizza EV (può essere negativo, quindi aggiungiamo 1 per renderlo positivo)
        ev_normalized = (ev / 100.0) + 1.0  # EV% da -100.. diventa 0..+
        
        # Combina: 40% peso su EV, 60% peso su confidence
        combined_score = (ev_normalized * 0.4) + (confidence_score * 0.6)
        
        return combined_score
    
    def _deduplicate_opportunities(self, opportunities: List[LiveBettingOpportunity]) -> List[LiveBettingOpportunity]:
        """
        🆕 OTTIMIZZATO: Deduplica opportunità per match_id + market (mantiene quella con confidence più alta).
        Elimina TUTTI i duplicati identici, anche se arrivano in momenti diversi.
        """
        seen = {}
        for opp in opportunities:
            # 🆕 Chiave esatta: match_id + market (case-insensitive per sicurezza)
            key = f"{opp.match_id}_{opp.market.lower().strip()}"
            
            if key not in seen:
                seen[key] = opp
            else:
                # 🆕 Mantieni quella con confidence più alta (o odds più alta se stessa confidence)
                if opp.confidence > seen[key].confidence:
                    seen[key] = opp
                elif opp.confidence == seen[key].confidence:
                    # Se stessa confidence, mantieni quella con odds più alta (migliore valore)
                    if opp.odds and seen[key].odds and opp.odds > seen[key].odds:
                        seen[key] = opp
                    # Se anche le odds sono uguali, mantieni la prima (evita duplicati identici)
        
        return list(seen.values())
    
    def _limit_and_deduplicate_per_match(
        self, 
        opportunities: List[LiveBettingOpportunity], 
        max_per_match: int = 2
    ) -> List[LiveBettingOpportunity]:
        """
        🆕 OTTIMIZZATO: Limita numero di segnali per partita E deduplica di nuovo per sicurezza.
        Assicura che non ci siano segnali identici o troppo simili sulla stessa partita.
        Mantiene solo mercati DIVERSI (non identici o simili).
        """
        # Raggruppa per match_id
        by_match = {}
        for opp in opportunities:
            match_id = opp.match_id
            if match_id not in by_match:
                by_match[match_id] = []
            by_match[match_id].append(opp)
        
        result = []
        for match_id, match_opps in by_match.items():
            # 🆕 Deduplica di nuovo per questa partita (per sicurezza)
            seen_markets = {}
            for opp in match_opps:
                market_key = opp.market.lower().strip()
                
                # 🆕 Controlla se è un mercato identico o troppo simile
                is_duplicate = False
                for seen_market in seen_markets.keys():
                    if self._are_markets_similar(market_key, seen_market):
                        is_duplicate = True
                        # Mantieni quello con confidence più alta
                        if opp.confidence > seen_markets[seen_market].confidence:
                            seen_markets[seen_market] = opp
                        elif opp.confidence == seen_markets[seen_market].confidence:
                            # Se stessa confidence, mantieni quello con odds più alta
                            if opp.odds and seen_markets[seen_market].odds and opp.odds > seen_markets[seen_market].odds:
                                seen_markets[seen_market] = opp
                        break
                
                if not is_duplicate:
                    seen_markets[market_key] = opp
            
            # Ordina per combined score e prendi solo i migliori
            deduplicated = list(seen_markets.values())
            deduplicated.sort(key=lambda x: self._calculate_combined_score(x), reverse=True)
            result.extend(deduplicated[:max_per_match])
        
        return result
    
    def _are_markets_similar(self, market1: str, market2: str) -> bool:
        """
        🆕 Verifica se due mercati sono identici o troppo simili (da considerare duplicati).
        Esempi di mercati SIMILI (duplicati):
        - "over_2.5" e "over_2.5" (identici)
        - "over_2.5" e "over_2.5 " (con spazio, identici)
        
        Esempi di mercati DIVERSI (non duplicati):
        - "over_2.5" e "over_2.5_ht" (diversi: uno è match, uno è primo tempo)
        - "over_2.5" e "under_2.5" (diversi: opposti)
        - "clean_sheet_home" e "clean_sheet_away" (diversi: squadre diverse)
        """
        # Normalizza: lowercase, strip
        m1 = market1.lower().strip()
        m2 = market2.lower().strip()
        
        # Se identici, sono duplicati
        if m1 == m2:
            return True
        
        # 🆕 Se sono molto simili (stesso tipo di mercato, stesso valore), sono duplicati
        # Esempio: "over_2.5" e "over_2.5" (già gestito sopra)
        # Esempio: "goal_range_4_plus" e "goal_range_4+" (potrebbero essere simili)
        
        # Rimuovi spazi e caratteri speciali per confronto
        m1_normalized = m1.replace('_', '').replace('-', '').replace(' ', '')
        m2_normalized = m2.replace('_', '').replace('-', '').replace(' ', '')
        
        if m1_normalized == m2_normalized:
            return True
        
        # 🆕 Se sono dello stesso tipo ma con varianti minime (es. "over_2.5" vs "over_2.5_match")
        # Considerali duplicati solo se la parte principale è identica
        # Ma NON considerare duplicati se uno è HT e l'altro no
        if '_ht' in m1 and '_ht' not in m2:
            return False  # Diversi: uno è HT, uno no
        if '_ht' in m2 and '_ht' not in m1:
            return False  # Diversi: uno è HT, uno no
        
        # Se sono identici tranne per suffissi come "_match", "_home", "_away" (ma solo se stesso tipo)
        # Esempio: "over_2.5_match" e "over_2.5" sono simili
        m1_base = m1.split('_')[0] if '_' in m1 else m1
        m2_base = m2.split('_')[0] if '_' in m2 else m2
        
        # Se la base è diversa, non sono simili
        if m1_base != m2_base:
            return False
        
        # Estrai valori numerici (es. "2.5" da "over_2.5")
        import re
        m1_numbers = re.findall(r'[0-9]+\.?[0-9]*', m1)
        m2_numbers = re.findall(r'[0-9]+\.?[0-9]*', m2)
        
        # Se hanno numeri diversi, non sono simili
        if m1_numbers != m2_numbers:
            return False
        
        # Se arriviamo qui, sono molto simili (stesso tipo, stesso valore)
        # Ma controlla ancora che non siano HT vs non-HT
        if ('ht' in m1) != ('ht' in m2):
            return False  # Uno è HT, uno no → diversi
        
        # Se sono home vs away per stesso mercato, sono diversi
        if ('home' in m1 and 'away' in m2) or ('away' in m1 and 'home' in m2):
            return False  # Squadre diverse → diversi
        
        # Altrimenti, sono simili (duplicati)
        return True
    
    def _filter_contradictory_signals(
        self,
        opportunities: List[LiveBettingOpportunity],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        🆕 Filtra segnali contrastanti sulla stessa partita.
        
        Esempi di segnali contrastanti:
        - Under 1.5 + Ribaltone (se Under, partita chiusa, no ribaltone)
        - Under 1.5 + Over 2.5 (contraddittori)
        - Under 0.5 HT + Over 1.5 HT (contraddittori)
        - Clean Sheet + BTTS (contraddittori)
        """
        if len(opportunities) <= 1:
            return opportunities
        
        filtered = []
        score_home = live_data.get('score_home', 0)
        score_away = live_data.get('score_away', 0)
        total_goals = score_home + score_away
        
        # Raggruppa per match_id
        by_match = {}
        for opp in opportunities:
            match_id = opp.match_id
            if match_id not in by_match:
                by_match[match_id] = []
            by_match[match_id].append(opp)
        
        # Per ogni partita, filtra segnali contrastanti
        for match_id, match_opps in by_match.items():
            if len(match_opps) == 1:
                filtered.append(match_opps[0])
                continue
            
            # Ordina per confidence (migliore prima)
            match_opps.sort(key=lambda x: x.confidence, reverse=True)
            
            # Lista segnali accettati per questa partita
            accepted = []
            
            for opp in match_opps:
                market = opp.market.lower()
                situation = opp.situation.lower()
                is_contradictory = False
                
                # Verifica contraddizioni con segnali già accettati
                for accepted_opp in accepted:
                    accepted_market = accepted_opp.market.lower()
                    accepted_situation = accepted_opp.situation.lower()
                    
                    # CONTRADDIZIONE 1: Under + Ribaltone
                    # Se c'è Under, la partita è chiusa → no ribaltone
                    if ('under' in market and 'ribaltone' in accepted_situation) or \
                       ('under' in accepted_market and 'ribaltone' in situation):
                        is_contradictory = True
                        logger.debug(f"⏭️  Segnale contrastante filtrato: {market} vs {accepted_market} (Under + Ribaltone)")
                        break
                    
                    # CONTRADDIZIONE 2: Under + Over (stesso goal line o simile)
                    if 'under' in market and 'over' in accepted_market:
                        # Estrai goal line
                        under_line = self._extract_goal_line(market)
                        over_line = self._extract_goal_line(accepted_market)
                        if under_line and over_line and abs(under_line - over_line) <= 1.0:
                            is_contradictory = True
                            logger.debug(f"⏭️  Segnale contrastante filtrato: {market} vs {accepted_market} (Under {under_line} + Over {over_line})")
                            break
                    
                    if 'over' in market and 'under' in accepted_market:
                        over_line = self._extract_goal_line(market)
                        under_line = self._extract_goal_line(accepted_market)
                        if over_line and under_line and abs(over_line - under_line) <= 1.0:
                            is_contradictory = True
                            logger.debug(f"⏭️  Segnale contrastante filtrato: {market} vs {accepted_market} (Over {over_line} + Under {under_line})")
                            break
                    
                    # CONTRADDIZIONE 3: Under HT + Over HT
                    if 'under' in market and 'ht' in market and 'over' in accepted_market and 'ht' in accepted_market:
                        is_contradictory = True
                        logger.debug(f"⏭️  Segnale contrastante filtrato: {market} vs {accepted_market} (Under HT + Over HT)")
                        break
                    
                    if 'over' in market and 'ht' in market and 'under' in accepted_market and 'ht' in accepted_market:
                        is_contradictory = True
                        logger.debug(f"⏭️  Segnale contrastante filtrato: {market} vs {accepted_market} (Over HT + Under HT)")
                        break
                    
                    # CONTRADDIZIONE 4: Clean Sheet + BTTS
                    if ('clean_sheet' in market and 'btts' in accepted_market) or \
                       ('btts' in market and 'clean_sheet' in accepted_market):
                        is_contradictory = True
                        logger.debug(f"⏭️  Segnale contrastante filtrato: {market} vs {accepted_market} (Clean Sheet + BTTS)")
                        break
                    
                    # 🆕 CONTRADDIZIONE 5B: Home Win + Away Win (IMPOSSIBILE!)
                    # FIX CRITICO: Non si può suggerire sia vittoria home che away sulla stessa partita!
                    if (('home_win' in market or '1x2_home' in market or '1x' in market) and 
                        ('away_win' in accepted_market or '1x2_away' in accepted_market or 'x2' in accepted_market)) or \
                       (('away_win' in market or '1x2_away' in market or 'x2' in market) and 
                        ('home_win' in accepted_market or '1x2_home' in accepted_market or '1x' in accepted_market)):
                        is_contradictory = True
                        logger.warning(f"⏭️  Segnale CONTRADDITTORIO BLOCCATO: {market} vs {accepted_market} (Home Win + Away Win sulla stessa partita - IMPOSSIBILE!)")
                        break
                    
                    # CONTRADDIZIONE 5: Under + Match Winner/Ribaltone (se Under, partita chiusa)
                    # Se c'è Under 1.5 o Under 0.5, la partita è chiusa → no ribaltone
                    if 'under' in market:
                        under_line = self._extract_goal_line(market)
                        # Se Under 1.5 o meno e ci sono già gol vicini al limite, partita chiusa
                        if under_line and under_line <= 1.5:
                            if ('match_winner' in accepted_market or 'ribaltone' in accepted_situation or 'comeback' in accepted_situation or '1x2' in accepted_market or 'dnb' in accepted_market):
                                # Under 1.5 su 1-0 significa partita chiusa → no ribaltone
                                if total_goals >= under_line - 0.5:  # Vicino al limite (es. Under 1.5 con 1 gol)
                                    is_contradictory = True
                                    logger.debug(f"⏭️  Segnale contrastante filtrato: {market} (Under {under_line}) vs {accepted_market} (Ribaltone) - partita chiusa ({total_goals} gol)")
                                    break
                    
                    if ('match_winner' in market or 'ribaltone' in situation or 'comeback' in situation or '1x2' in market or 'dnb' in market) and 'under' in accepted_market:
                        under_line = self._extract_goal_line(accepted_market)
                        if under_line and under_line <= 1.5:
                            if total_goals >= under_line - 0.5:  # Vicino al limite
                                is_contradictory = True
                                logger.debug(f"⏭️  Segnale contrastante filtrato: {market} (Ribaltone) vs {accepted_market} (Under {under_line}) - partita chiusa ({total_goals} gol)")
                                break
                
                # Se non è contraddittorio, aggiungilo
                if not is_contradictory:
                    accepted.append(opp)
                else:
                    logger.debug(f"⏭️  Segnale {market} filtrato per contraddizione logica con altri segnali della stessa partita")
            
            # Aggiungi segnali accettati
            filtered.extend(accepted)
        
        return filtered
    
    
    def _enhance_with_ai(
        self,
        opportunities: List[LiveBettingOpportunity],
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> List[LiveBettingOpportunity]:
        """
        🆕 OTTIMIZZATO: Usa IA per migliorare le opportunità.
        Utilizza sia l'analisi AI base (_get_ai_market_confidence) che LiveMatchAI dedicata ai match live.
        """
        enhanced = []
        
        # 🆕 Se LiveMatchAI è disponibile, usa analisi avanzata dedicata ai match live
        live_ai_analysis = None
        if self.live_match_ai:
            try:
                # Ottieni quote per analisi AI
                odds_data = {
                    'home': match_data.get('odds_1'),
                    'draw': match_data.get('odds_x'),
                    'away': match_data.get('odds_2'),
                    'over_2_5': match_data.get('odds_over_2_5'),
                    'under_2_5': match_data.get('odds_under_2_5')
                }
                
                # Analisi completa con LiveMatchAI
                live_ai_analysis = self.live_match_ai.analyze_live_match(
                    match_data=match_data,
                    live_data=live_data,
                    odds_data=odds_data
                )
                logger.debug(f"✅ LiveMatchAI analisi completata per {match_data.get('id', 'unknown')}")
            except Exception as e:
                logger.debug(f"⚠️  Errore analisi LiveMatchAI: {e} - utilizzerò analisi AI base")
        
        for opp in opportunities:
            try:
                # 🆕 Boost base da analisi statistica
                ai_boost = self._get_ai_market_confidence(match_data, live_data, opp.market)
                
                # 🆕 Se LiveMatchAI ha analizzato, aggiungi boost aggiuntivo basato su pattern e situazione
                if live_ai_analysis:
                    additional_boost = self._get_live_ai_boost(
                        opp, live_ai_analysis, match_data, live_data
                    )
                    ai_boost += additional_boost
                    logger.debug(f"✅ Boost AI totale: {ai_boost:.1f}% (base: {ai_boost - additional_boost:.1f}%, LiveMatchAI: {additional_boost:.1f}%)")
                
                opp.confidence = min(100, opp.confidence + ai_boost)
                enhanced.append(opp)
            except Exception as e:
                logger.debug(f"⚠️  Errore enhancement IA: {e}")
                enhanced.append(opp)
        return enhanced
    
    def _get_live_ai_boost(
        self,
        opportunity: LiveBettingOpportunity,
        live_ai_analysis: Dict[str, Any],
        match_data: Dict[str, Any],
        live_data: Dict[str, Any]
    ) -> float:
        """
        🆕 Calcola boost aggiuntivo basato su analisi LiveMatchAI.
        Utilizza pattern, situazione e probabilità aggiornate da LiveMatchAI.
        """
        boost = 0.0
        
        try:
            situation = live_ai_analysis.get('situation_analysis', {})
            patterns = live_ai_analysis.get('patterns', {})
            probabilities = live_ai_analysis.get('updated_probabilities', {})
            
            market = opportunity.market.lower()
            
            # 🆕 Boost basato su pattern rilevati
            if 'over' in market:
                if patterns.get('high_scoring', False):
                    boost += 3.0  # Partita ad alto scoring
                if patterns.get('attacking_mode', False):
                    boost += 2.0  # Modalità attacco
                if situation.get('pressure_score', 0) > 0.5:
                    boost += 2.0  # Alta pressione = più gol probabili
            
            elif 'under' in market:
                if patterns.get('low_scoring', False):
                    boost += 3.0  # Partita a basso scoring
                if patterns.get('defensive_mode', False):
                    boost += 2.0  # Modalità difensiva
                if situation.get('pressure_score', 0) < 0.3:
                    boost += 2.0  # Bassa pressione = meno gol probabili
            
            elif 'ribaltone' in opportunity.situation.lower() or 'comeback' in opportunity.situation.lower():
                if patterns.get('comeback_possible', False):
                    boost += 4.0  # Pattern ribaltone rilevato
                if situation.get('is_critical', False):
                    boost += 2.0  # Situazione critica
            
            elif 'clean_sheet' in market:
                if situation.get('pressure_score', 0) < 0.2:
                    boost += 2.0  # Bassa pressione = clean sheet più probabile
            
            # 🆕 Boost basato su probabilità aggiornate da LiveMatchAI
            # Se la probabilità aggiornata è molto diversa da quella base, boost positivo
            if 'over_2_5' in market and probabilities.get('over_2_5', 0) > 0.7:
                boost += 2.0
            elif 'under_2_5' in market and probabilities.get('under_2_5', 0) > 0.7:
                boost += 2.0
            
            # 🆕 Boost basato su momentum
            momentum = situation.get('momentum_score', 0)
            if 'home' in market and momentum > 0.3:
                boost += 2.0  # Home domina
            elif 'away' in market and momentum < -0.3:
                boost += 2.0  # Away domina
            
            # Limita boost totale da LiveMatchAI a +10%
            boost = min(10.0, boost)
            
        except Exception as e:
            logger.debug(f"⚠️  Errore calcolo LiveMatchAI boost: {e}")
        
        return boost
    
    def _is_match_worth_analyzing(self, match_data: Dict[str, Any]) -> bool:
        """
        Verifica se la partita vale la pena di essere analizzata.
        Esclude SOLO partite giovanili e riserve.
        Campionati minori sono ACCETTATI se hanno dati live sufficienti (verificato da _has_sufficient_live_data).
        Permette Champions League femminile ed Europa Cup Women.
        """
        try:
            home = match_data.get('home', '').upper()
            away = match_data.get('away', '').upper()
            league = match_data.get('league', '').upper()
            
            # 🆕 Verifica se è un torneo femminile importante (Champions League, Europa Cup)
            # Se sì, ACCETTA anche se contiene "Women", "Feminine", "Femminile"
            is_important_women_tournament = False
            for tournament in self.allowed_women_tournaments:
                if tournament.upper() in league:
                    is_important_women_tournament = True
                    break
            
            # Controlla se è una partita giovanile/riserva (ESCLUDI)
            for keyword in self.excluded_leagues_keywords:
                if keyword.upper() in home or keyword.upper() in away or keyword.upper() in league:
                    return False  # Escludi solo giovanili/riserve
            
            # 🆕 Se è un torneo femminile importante, ACCETTA anche se contiene "Women", "Feminine", "Femminile"
            if is_important_women_tournament:
                return True
            
            # 🆕 Filtro aggiuntivo: escludi altri campionati femminili (non Champions/Europa)
            # ma solo se NON è già un torneo importante
            women_keywords = ['Women', 'Feminine', 'Femminile']
            for keyword in women_keywords:
                if keyword.upper() in home or keyword.upper() in away or keyword.upper() in league:
                    # Se contiene keyword femminile ma NON è un torneo importante, escludi
                    if not is_important_women_tournament:
                        return False
            
            # Se passa il filtro giovanili, ACCETTA (anche campionati minori)
            # La qualità dei dati sarà verificata da _has_sufficient_live_data
            return True
            
        except Exception as e:
            logger.debug(f"⚠️  Errore verifica partita: {e}")
            return False  # In caso di errore, escludi per sicurezza
    
    def _has_sufficient_live_data(self, live_data: Dict[str, Any]) -> bool:
        """
        Verifica se i dati live sono sufficienti per analisi seria.
        Questo è il filtro principale per campionati minori: se hanno dati validi, sono accettati.
        """
        try:
            minute = live_data.get('minute', 0)
            score_home = live_data.get('score_home', 0)
            score_away = live_data.get('score_away', 0)
            
            # Deve avere almeno minuto valido (tra 1 e 90)
            if minute < 1 or minute > 90:
                return False
            
            # Deve avere score valido (anche 0-0 è valido se partita è iniziata)
            if score_home is None or score_away is None:
                return False
            
            # Se partita è oltre 10' e ancora 0-0, è OK (partita valida)
            # Se partita ha gol, è OK
            if minute >= 10:
                # Ha minuto valido e score valido (anche 0-0)
                # Verifica se ha almeno alcune statistiche base
                shots_home = live_data.get('shots_home', 0)
                shots_away = live_data.get('shots_away', 0)
                possession_home = live_data.get('possession_home', 0)
                
                # Se ha statistiche (tiri o possesso), è OK
                if shots_home > 0 or shots_away > 0 or possession_home > 0:
                    return True
                
                # Se non ha statistiche ma ha score (almeno 1 gol), è OK
                if score_home > 0 or score_away > 0:
                    return True
                
                # Se è oltre 20' senza statistiche e senza gol, potrebbe essere partita chiusa
                # Ma accettiamo comunque se ha minuto e score validi
                if minute >= 20:
                    return True  # Accetta anche partite chiuse se hanno dati base
            
            # Se è nei primi 10 minuti, accetta solo se ha almeno score o statistiche
            if minute < 10:
                shots_home = live_data.get('shots_home', 0)
                shots_away = live_data.get('shots_away', 0)
                if (score_home > 0 or score_away > 0) or (shots_home > 0 or shots_away > 0):
                    return True
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"⚠️  Errore verifica dati live: {e}")
            return False
    
    def _extract_goal_line(self, market: str) -> Optional[float]:
        """Estrae la linea goal (es. 2.5) dal nome mercato."""
        match = re.search(r'(\d+(?:\.\d)?)', market)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def format_live_betting_message(self, opportunity: LiveBettingOpportunity) -> str:
        """Formatta messaggio per alert live betting - VERSIONE MIGLIORATA"""
        match_data = opportunity.match_data
        home = match_data.get('home', 'Home')
        away = match_data.get('away', 'Away')
        league = match_data.get('league', '')
        
        # Emoji e livello urgenza
        urgency_emoji = {
            'URGENT': '🚨',
            'HIGH': '🔥',
            'NORMAL': '⚡',
            'LOW': '💡'
        }.get(opportunity.urgency_level, '⚡')
        
        situation_emoji = {
            'ribaltone_favorita': '🔄',
            'under_early_goal': '⬇️',
            'over_no_goals': '⬆️',
            'next_goal_underdog': '⚽',
            'comeback_dominance': '📈',
            'over_0.5_ht': '⏱️',
            'over_1.5_ht': '⏱️',
            'under_0.5_ht': '⏱️',
            'under_1.5_ht': '⏱️',
            'double_chance_1x': '🛡️',
            'double_chance_x2': '🛡️',
            'over_0.5_general': '⬆️',
            'over_1.5_general': '⬆️',
            'over_2.5_general': '⬆️',
            'over_2.5_high_tempo': '⬆️',
            'over_3.5_general': '⬆️',
            'under_1.5_general': '⬇️',
            'under_2.5_general': '⬇️',
            'under_3.5_general': '⬇️',
            'corner_over': '📐',
            'card_over': '🟨',
            'handicap_away': '⚖️',
            'btts_yes': '⚽⚽',
            'win_to_nil_home': '🏆',
            'over_0.5_2h': '⏰',
            'dnb_home_comeback': '🔄',
            'dnb_away_comeback': '🔄',
            'total_goals_odd': '🔢',
            'total_goals_even': '🔢',
            'exact_score': '🎯',
            'goal_range_0_1': '📊',
            'goal_range_2_3': '📊',
            'goal_range_4_plus': '📊',
            'team_to_score_next_home': '⚽',
            'team_to_score_next_away': '⚽',
            'clean_sheet_home': '🛡️',
            'clean_sheet_away': '🛡️',
            'ht_ft_home_home': '⏱️',
            'home_win_dominance': '🏆',
            'away_win_dominance': '🏆',
            'asian_handicap_home': '⚖️',
            'asian_handicap_away': '⚖️',
            'next_goal_before_75': '⏰',
            'next_goal_after_75': '⏰'
        }.get(opportunity.situation, '🎯')
        
        # Header con urgenza
        message = f"{urgency_emoji} {situation_emoji} OPPORTUNITÀ LIVE BETTING!\n"
        message += f"{'='*50}\n\n"
        
        # Info partita
        message += f"⚽ {home} vs {away}\n"
        if league:
            message += f"🏆 {league}\n"
        
        # Statistiche partita se disponibili - MIGLIORATO: mostra solo dati disponibili
        if opportunity.match_stats:
            stats = opportunity.match_stats
            message += f"\n📊 STATISTICHE LIVE:\n"
            message += f"   Score: {stats.get('score_home', 0)}-{stats.get('score_away', 0)} al {stats.get('minute', 0)}'\n"
            
            # Possesso (solo se disponibile e non default 50%)
            possession_home = stats.get('possession_home', 0)
            if possession_home and possession_home != 50:
                possession_away = stats.get('possession_away', 100 - possession_home)
                message += f"   Possesso: {possession_home:.0f}% - {possession_away:.0f}%\n"
            
            # Tiri (solo se disponibili e > 0)
            shots_home = stats.get('shots_home', 0)
            shots_away = stats.get('shots_away', 0)
            shots_on_target_home = stats.get('shots_on_target_home', 0)
            shots_on_target_away = stats.get('shots_on_target_away', 0)
            
            if shots_home > 0 or shots_away > 0:
                # Mostra tiri in porta solo se disponibili
                if shots_on_target_home > 0 or shots_on_target_away > 0:
                    message += f"   Tiri: {shots_home} ({shots_on_target_home} in porta) vs {shots_away} ({shots_on_target_away} in porta)\n"
                else:
                    message += f"   Tiri: {shots_home} vs {shots_away}\n"
            
            # Corner (solo se disponibili e > 0)
            corners_home = stats.get('corners_home', 0)
            corners_away = stats.get('corners_away', 0)
            if corners_home > 0 or corners_away > 0:
                message += f"   Corner: {corners_home} - {corners_away}\n"
            
            # Cartellini (solo se disponibili e > 0)
            yellow_home = stats.get('yellow_cards_home', 0)
            yellow_away = stats.get('yellow_cards_away', 0)
            red_home = stats.get('red_cards_home', 0)
            red_away = stats.get('red_cards_away', 0)
            if yellow_home > 0 or yellow_away > 0 or red_home > 0 or red_away > 0:
                message += f"   Cartellini: 🟨{yellow_home}/{yellow_away}"
                if red_home > 0 or red_away > 0:
                    message += f" 🔴{red_home}/{red_away}"
                    message += "\n"
        
        message += f"\n{'='*50}\n"
        message += f"💡 RACCOMANDAZIONE PRINCIPALE:\n"
        message += f"{'='*50}\n"
        message += f"   {opportunity.recommendation}\n\n"
        
        # Mercato e dettagli
        market_display = self._translate_market_name(opportunity.market)
        message += f"📊 MERCATO: {market_display}\n"
        message += f"📈 CONFIDENCE: {opportunity.confidence:.0f}% "
        
        # Barra confidence visiva
        confidence_bars = int(opportunity.confidence / 10)
        message += "[" + "█" * confidence_bars + "░" * (10 - confidence_bars) + "]\n"
        
        message += f"💰 QUOTA: {opportunity.odds:.2f}\n"
        message += f"💵 STAKE SUGGERITO: {opportunity.stake_suggestion}% bankroll\n"
        if hasattr(opportunity, 'ev') and opportunity.ev is not None:
            message += f"📈 VALORE ATTESO: {opportunity.ev:+.1f}%\n"
        
        # Traduci urgenza in italiano
        urgency_it = {
            'URGENT': 'URGENTE',
            'HIGH': 'ALTA',
            'NORMAL': 'NORMALE',
            'LOW': 'BASSA'
        }.get(opportunity.urgency_level, opportunity.urgency_level)
        message += f"⏱️ URGENZA: {urgency_it}\n\n"
        
        # Mercati alternativi
        if opportunity.alternative_markets:
            message += f"🔄 MERCATI ALTERNATIVI CONSIGLIATI:\n"
            message += f"{'-'*50}\n"
            for i, alt_market in enumerate(opportunity.alternative_markets, 1):
                market_name = alt_market.get('market', '').upper().replace('_', ' ')
                alt_confidence = alt_market.get('confidence', 0)
                alt_odds = alt_market.get('odds', 0)
                message += f"   {i}. {market_name}\n"
                message += f"      Confidence: {alt_confidence:.0f}% | Quota: {alt_odds:.2f}\n"
            message += "\n"
        
        # 🆕 FIX: Ragionamento dettagliato - sempre presente con statistiche chiave
        message += f"🧠 ANALISI DETTAGLIATA:\n"
        message += f"{'-'*50}\n"
        
        # Usa reasoning base se disponibile
        base_reasoning = opportunity.reasoning if opportunity.reasoning and len(opportunity.reasoning.strip()) >= 30 else ""
        
        # Aggiungi statistiche chiave al reasoning
        stats = opportunity.match_stats if opportunity.match_stats else {}
        key_stats = opportunity.key_stats if hasattr(opportunity, 'key_stats') and opportunity.key_stats else {}
        
        if base_reasoning:
            message += f"{base_reasoning}\n\n"
        
        # 🆕 FIX: Statistiche chiave estratte per l'analisi - sempre mostrate per verificare correttezza
        stats_shown = False
        if key_stats and len(key_stats) > 0:
            message += "📊 STATISTICHE ESTRATTE PER L'ANALISI:\n"
            message += f"{'-'*50}\n"
            for label, value in key_stats.items():
                if value is not None and str(value).strip():
                    message += f"• {label}: {value}\n"
                    stats_shown = True
            if stats_shown:
                message += "\n"
        
        # Se key_stats è vuoto o insufficiente, mostra statistiche generali da match_stats
        if not stats_shown and stats:
            message += "📊 STATISTICHE ESTRATTE PER L'ANALISI:\n"
            message += f"{'-'*50}\n"
            # Mostra sempre score e minuto
            message += f"• Score: {stats.get('score_home', 0)}-{stats.get('score_away', 0)}\n"
            message += f"• Minuto: {stats.get('minute', 0)}'\n"
            
            # Possesso
            if stats.get('possession_home') and stats.get('possession_home') != 50:
                possession_away = stats.get('possession_away', 100 - stats.get('possession_home', 50))
                message += f"• Possesso: {stats.get('possession_home', 0):.0f}% - {possession_away:.0f}%\n"
                stats_shown = True
            
            # Tiri
            shots_home = stats.get('shots_home', 0)
            shots_away = stats.get('shots_away', 0)
            shots_on_target_home = stats.get('shots_on_target_home', 0)
            shots_on_target_away = stats.get('shots_on_target_away', 0)
            if shots_home > 0 or shots_away > 0:
                if shots_on_target_home > 0 or shots_on_target_away > 0:
                    message += f"• Tiri totali: {shots_home + shots_away} ({shots_on_target_home + shots_on_target_away} in porta)\n"
                    message += f"  - Casa: {shots_home} ({shots_on_target_home} in porta)\n"
                    message += f"  - Trasferta: {shots_away} ({shots_on_target_away} in porta)\n"
                else:
                    message += f"• Tiri totali: {shots_home + shots_away}\n"
                    message += f"  - Casa: {shots_home}, Trasferta: {shots_away}\n"
                stats_shown = True
            
            # Calcola tiri/minuto se disponibile
            minute = stats.get('minute', 0)
            if minute > 0 and (shots_home > 0 or shots_away > 0):
                shots_per_minute = (shots_home + shots_away) / minute
                message += f"• Tiri/minuto: {shots_per_minute:.2f}\n"
            
            # Corner
            corners_home = stats.get('corners_home', 0)
            corners_away = stats.get('corners_away', 0)
            if corners_home > 0 or corners_away > 0:
                message += f"• Corner: {corners_home + corners_away} totali ({corners_home}-{corners_away})\n"
            
            # Cartellini
            yellow_home = stats.get('yellow_cards_home', 0)
            yellow_away = stats.get('yellow_cards_away', 0)
            red_home = stats.get('red_cards_home', 0)
            red_away = stats.get('red_cards_away', 0)
            if yellow_home > 0 or yellow_away > 0 or red_home > 0 or red_away > 0:
                message += f"• Cartellini: {yellow_home + yellow_away} gialli, {red_home + red_away} rossi\n"
            
            message += "\n"
        
        # Aggiungi analisi contestuale
        score_home = stats.get('score_home', 0)
        score_away = stats.get('score_away', 0)
        minute = stats.get('minute', 0)
        total_goals = score_home + score_away
        goal_diff = abs(score_home - score_away)
        ev_value = getattr(opportunity, 'ev', 0) if hasattr(opportunity, 'ev') else 0
        
        if not base_reasoning or len(base_reasoning.strip()) < 50:
            message += f"• Analisi basata su score attuale ({score_home}-{score_away}) al {minute}' minuto\n"
            message += f"• Mercato {market_display} con confidence {opportunity.confidence:.0f}% e valore atteso {ev_value:+.1f}%\n"
            if total_goals > 0:
                message += f"• Partita con {total_goals} gol totali e differenza di {goal_diff} gol\n"
            if minute >= 60:
                message += f"• Fase avanzata della partita ({minute}'), situazione più definita\n"
            else:
                message += f"• Fase iniziale ({minute}'), ancora tempo per sviluppi\n"
        
        
        # Footer
        message += f"{'='*50}\n"
        message += f"⏰ {datetime.now().strftime('%H:%M:%S')} | Minuto: {opportunity.match_stats.get('minute', 'N/A') if opportunity.match_stats else 'N/A'}'"
        
        return message

