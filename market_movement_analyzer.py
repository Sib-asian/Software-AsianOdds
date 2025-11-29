#!/usr/bin/env python3
"""
Market Movement Analyzer
========================

Tool per analizzare i movimenti di Spread e Total e generare
interpretazioni e giocate consigliate basate su pattern di mercato.

Usage:
    python market_movement_analyzer.py
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import math


class MovementDirection(Enum):
    """Direzione del movimento"""
    HARDEN = "🔽"  # Si indurisce (aumenta fiducia)
    SOFTEN = "🔼"  # Si ammorbidisce (cala fiducia)
    STABLE = "⚖️"  # Stabile (nessun cambiamento)


class MovementIntensity(Enum):
    """Intensità del movimento"""
    LIGHT = "Leggero"
    MEDIUM = "Medio"
    STRONG = "Forte"
    NONE = "Nessun movimento"


class ConfidenceLevel(Enum):
    """Livello di confidenza"""
    HIGH = "Alta"
    MEDIUM = "Media"
    LOW = "Bassa"


@dataclass
class MovementAnalysis:
    """Analisi di un movimento"""
    direction: MovementDirection
    intensity: MovementIntensity
    opening_value: float
    closing_value: float
    movement_steps: float
    interpretation: str


@dataclass
class MarketRecommendation:
    """Raccomandazione per un mercato"""
    market_name: str
    recommendation: str
    confidence: ConfidenceLevel
    explanation: str


@dataclass
class ExpectedGoals:
    """Expected Goals (xG) calcolati da spread e total"""
    home_xg: float  # lambda casa
    away_xg: float  # lambda trasferta
    home_clean_sheet_prob: float  # P(casa clean sheet)
    away_clean_sheet_prob: float  # P(trasferta clean sheet)
    btts_prob: float  # P(Both Teams To Score)
    home_win_prob: float  # P(casa vince) via Poisson
    draw_prob: float  # P(pareggio) via Poisson
    away_win_prob: float  # P(trasferta vince) via Poisson


@dataclass
class AnalysisResult:
    """Risultato completo dell'analisi"""
    spread_analysis: MovementAnalysis
    total_analysis: MovementAnalysis
    combination_interpretation: str
    core_recommendations: List[MarketRecommendation]  # Consigli principali (HIGH/MEDIUM confidence)
    alternative_recommendations: List[MarketRecommendation]  # Opzioni alternative (MEDIUM confidence)
    value_recommendations: List[MarketRecommendation]  # Value bets (LOW confidence, high value)
    exchange_recommendations: List[MarketRecommendation]  # Consigli Exchange (Punta/Banca)
    overall_confidence: ConfidenceLevel
    expected_goals: ExpectedGoals  # xG e probabilità calcolate


def calculate_expected_goals(spread: float, total: float) -> ExpectedGoals:
    """
    Calcola Expected Goals (xG) da spread e total usando Asian Handicap.

    Formula:
    - spread = lambda_casa - lambda_trasferta
    - total = lambda_casa + lambda_trasferta

    Quindi:
    - lambda_casa = (total - spread) / 2
    - lambda_trasferta = (total + spread) / 2

    Args:
        spread: Spread di chiusura (negativo = casa favorita)
        total: Total di chiusura

    Returns:
        ExpectedGoals con xG e probabilità calcolate
    """
    # Calcola xG (lambda) per casa e trasferta
    home_xg = (total - spread) / 2
    away_xg = (total + spread) / 2

    # Assicura che xG siano positivi (non può essere negativo)
    home_xg = max(0.1, home_xg)
    away_xg = max(0.1, away_xg)

    # Calcola probabilità clean sheet: P(0 gol) = e^(-lambda)
    home_clean_sheet_prob = math.exp(-away_xg)
    away_clean_sheet_prob = math.exp(-home_xg)

    # Calcola BTTS: P(entrambe segnano) = 1 - P(almeno una clean sheet)
    btts_prob = (1 - home_clean_sheet_prob) * (1 - away_clean_sheet_prob)

    # Calcola probabilità 1X2 usando Poisson
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0

    # Calcola per i primi 8 gol (coprono >99% dei casi)
    for home_goals in range(8):
        for away_goals in range(8):
            prob = poisson_probability(home_goals, home_xg) * poisson_probability(away_goals, away_xg)

            if home_goals > away_goals:
                home_win_prob += prob
            elif home_goals == away_goals:
                draw_prob += prob
            else:
                away_win_prob += prob

    return ExpectedGoals(
        home_xg=home_xg,
        away_xg=away_xg,
        home_clean_sheet_prob=home_clean_sheet_prob,
        away_clean_sheet_prob=away_clean_sheet_prob,
        btts_prob=btts_prob,
        home_win_prob=home_win_prob,
        draw_prob=draw_prob,
        away_win_prob=away_win_prob
    )


def poisson_probability(k: int, lambda_: float) -> float:
    """
    Calcola probabilità Poisson: P(X = k) = (λ^k * e^(-λ)) / k!

    Args:
        k: Numero di gol
        lambda_: Expected goals (xG)

    Returns:
        Probabilità che vengano segnati esattamente k gol
    """
    return (lambda_ ** k) * math.exp(-lambda_) / math.factorial(k)


def get_most_likely_score(home_xg: float, away_xg: float, top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Calcola i punteggi più probabili usando distribuzione Poisson.

    Args:
        home_xg: Expected goals casa
        away_xg: Expected goals trasferta
        top_n: Numero di punteggi da restituire

    Returns:
        Lista di tuple (punteggio, probabilità) ordinate per probabilità
    """
    scores = []

    # Calcola probabilità per tutti i punteggi realistici (0-6 gol per squadra)
    for home_goals in range(7):
        for away_goals in range(7):
            prob = poisson_probability(home_goals, home_xg) * poisson_probability(away_goals, away_xg)
            scores.append((f"{home_goals}-{away_goals}", prob))

    # Ordina per probabilità decrescente
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_n]


def calculate_halftime_probabilities(home_xg: float, away_xg: float) -> Dict:
    """
    Calcola probabilità per il Primo Tempo (HT) usando xG.

    Assunzione: circa 45% dei gol totali avvengono nel primo tempo.

    Args:
        home_xg: Expected goals casa (full time)
        away_xg: Expected goals trasferta (full time)

    Returns:
        Dict con probabilità HT: home_win, draw, away_win, btts, total_goals
    """
    # xG primo tempo (tipicamente 45% del totale)
    home_xg_ht = home_xg * 0.45
    away_xg_ht = away_xg * 0.45

    # Calcola probabilità 1X2 HT usando Poisson
    home_win_ht = 0.0
    draw_ht = 0.0
    away_win_ht = 0.0

    for home_goals in range(5):
        for away_goals in range(5):
            prob = poisson_probability(home_goals, home_xg_ht) * poisson_probability(away_goals, away_xg_ht)

            if home_goals > away_goals:
                home_win_ht += prob
            elif home_goals == away_goals:
                draw_ht += prob
            else:
                away_win_ht += prob

    # Calcola P(BTTS HT)
    home_cs_ht = math.exp(-away_xg_ht)
    away_cs_ht = math.exp(-home_xg_ht)
    btts_ht = (1 - home_cs_ht) * (1 - away_cs_ht)

    # Calcola P(Over/Under HT)
    total_xg_ht = home_xg_ht + away_xg_ht

    # P(Over 0.5 HT) = 1 - P(0 gol totali)
    over_05_ht = 1 - (math.exp(-home_xg_ht) * math.exp(-away_xg_ht))

    # P(Over 1.5 HT) = somma probabilità >= 2 gol
    over_15_ht = 0.0
    for total_goals in range(2, 8):
        for home_goals in range(total_goals + 1):
            away_goals = total_goals - home_goals
            over_15_ht += poisson_probability(home_goals, home_xg_ht) * poisson_probability(away_goals, away_xg_ht)

    return {
        "home_xg_ht": home_xg_ht,
        "away_xg_ht": away_xg_ht,
        "total_xg_ht": total_xg_ht,
        "home_win_ht": home_win_ht,
        "draw_ht": draw_ht,
        "away_win_ht": away_win_ht,
        "btts_ht": btts_ht,
        "over_05_ht": over_05_ht,
        "over_15_ht": over_15_ht
    }


def calculate_total_goals_distribution(home_xg: float, away_xg: float) -> Dict[int, float]:
    """
    Calcola distribuzione probabilità per numero totale di gol usando Poisson.

    Returns:
        Dict {numero_gol: probabilità}
    """
    distribution = {}
    total_xg = home_xg + away_xg

    for total_goals in range(10):
        # Somma tutte le combinazioni che danno questo totale
        prob = 0.0
        for home_goals in range(total_goals + 1):
            away_goals = total_goals - home_goals
            prob += poisson_probability(home_goals, home_xg) * poisson_probability(away_goals, away_xg)
        distribution[total_goals] = prob

    return distribution


def calculate_team_totals(home_xg: float, away_xg: float) -> Dict:
    """
    Calcola probabilità Over/Under per singole squadre.

    Returns:
        Dict con probabilità per team totals
    """
    # P(Casa Over 0.5) = 1 - P(Casa 0 gol)
    home_over_05 = 1 - poisson_probability(0, home_xg)
    home_over_15 = 1 - poisson_probability(0, home_xg) - poisson_probability(1, home_xg)
    home_over_25 = sum(poisson_probability(k, home_xg) for k in range(3, 8))

    # P(Trasferta Over 0.5) = 1 - P(Trasferta 0 gol)
    away_over_05 = 1 - poisson_probability(0, away_xg)
    away_over_15 = 1 - poisson_probability(0, away_xg) - poisson_probability(1, away_xg)
    away_over_25 = sum(poisson_probability(k, away_xg) for k in range(3, 8))

    return {
        "home_over_05": home_over_05,
        "home_over_15": home_over_15,
        "home_over_25": home_over_25,
        "away_over_05": away_over_05,
        "away_over_15": away_over_15,
        "away_over_25": away_over_25
    }


def calculate_winning_margin(home_xg: float, away_xg: float) -> Dict[str, float]:
    """
    Calcola probabilità margini di vittoria usando Poisson.

    Returns:
        Dict con probabilità per ogni margine
    """
    margins = {
        "draw": 0.0,
        "home_1": 0.0,  # Casa vince per 1 gol
        "home_2": 0.0,  # Casa vince per 2 gol
        "home_3+": 0.0,  # Casa vince per 3+ gol
        "away_1": 0.0,
        "away_2": 0.0,
        "away_3+": 0.0
    }

    for home_goals in range(8):
        for away_goals in range(8):
            prob = poisson_probability(home_goals, home_xg) * poisson_probability(away_goals, away_xg)
            diff = home_goals - away_goals

            if diff == 0:
                margins["draw"] += prob
            elif diff == 1:
                margins["home_1"] += prob
            elif diff == 2:
                margins["home_2"] += prob
            elif diff >= 3:
                margins["home_3+"] += prob
            elif diff == -1:
                margins["away_1"] += prob
            elif diff == -2:
                margins["away_2"] += prob
            elif diff <= -3:
                margins["away_3+"] += prob

    return margins


def calculate_first_to_score(home_xg: float, away_xg: float) -> Dict[str, float]:
    """
    Calcola probabilità chi segna per primo.

    Formula approssimata:
    P(Casa prima) ≈ home_xg / (home_xg + away_xg) * (1 - P(0-0))

    Returns:
        Dict con probabilità first scorer
    """
    total_xg = home_xg + away_xg

    # P(0-0)
    prob_00 = poisson_probability(0, home_xg) * poisson_probability(0, away_xg)

    # P(qualcuno segna)
    prob_goals = 1 - prob_00

    # Distribuzione basata su xG ratio
    if total_xg > 0:
        home_ratio = home_xg / total_xg
        away_ratio = away_xg / total_xg
    else:
        home_ratio = 0.5
        away_ratio = 0.5

    return {
        "home_first": home_ratio * prob_goals,
        "away_first": away_ratio * prob_goals,
        "no_goal": prob_00
    }


class SpreadAnalyzer:
    """Analizzatore per movimenti Spread"""
    
    # Tabelle interpretazione movimenti spread
    SPREAD_MOVEMENTS_DOWN = {
        -1.75: (-1.5, "Leggero calo di fiducia nel favorito, mercato più prudente"),
        -1.5: (-1.25, "Favorito meno dominante, mercato vede 1–0 o 2–1"),
        -1.25: (-1.0, "Fiducia in calo evidente, l'underdog 'resiste' meglio"),
        -1.0: (-0.75, "Mercato molto più equilibrato, favorito vulnerabile"),
        -0.75: (-0.5, "Fiducia nel favorito in netto calo"),
        -0.5: (-0.25, "Favorito quasi al pari"),
        -0.25: (0.0, "Favorita perde fiducia → equilibrio totale"),
    }
    
    SPREAD_MOVEMENTS_UP = {
        -0.25: (-0.5, "Cresce la fiducia nel favorito, mercato lo vede superiore"),
        -0.5: (-0.75, "Favorito in netto rafforzamento, l'1 diventa più stabile"),
        -0.75: (-1.0, "Vittoria più larga attesa, possibile 2–0"),
        -1.0: (-1.25, "Grande fiducia nel favorito, mercato sbilanciato su di lui"),
        -1.25: (-1.5, "Favorito dominante, l'underdog viene svalutato"),
        -1.5: (-1.75, "Massima fiducia, probabile goleada"),
    }
    
    def analyze(self, opening: float, closing: float) -> MovementAnalysis:
        """Analizza il movimento dello spread"""
        # Per lo spread, confrontiamo i valori assoluti per determinare se si indurisce o ammorbidisce
        abs_opening = abs(opening)
        abs_closing = abs(closing)
        movement = abs_closing - abs_opening  # Se negativo = si ammorbidisce, positivo = si indurisce

        # Determina direzione
        if abs(movement) < 0.01:
            direction = MovementDirection.STABLE
            intensity = MovementIntensity.NONE
            interpretation = "Spread stabile, nessun cambiamento significativo"
        elif movement < 0:
            direction = MovementDirection.SOFTEN
            interpretation = self._get_soften_interpretation(opening, closing)
        else:
            direction = MovementDirection.HARDEN
            interpretation = self._get_harden_interpretation(opening, closing)
        
        # Calcola intensità
        abs_movement = abs(movement)
        if abs_movement < 0.3:
            intensity = MovementIntensity.LIGHT
        elif abs_movement < 0.6:
            intensity = MovementIntensity.MEDIUM
        else:
            intensity = MovementIntensity.STRONG
        
        # Calcola step (multipli di 0.25)
        steps = abs_movement / 0.25
        
        if direction == MovementDirection.STABLE:
            intensity = MovementIntensity.NONE
        
        return MovementAnalysis(
            direction=direction,
            intensity=intensity,
            opening_value=opening,
            closing_value=closing,
            movement_steps=steps,
            interpretation=interpretation
        )
    
    def _get_soften_interpretation(self, opening: float, closing: float) -> str:
        """Ottiene interpretazione per spread che si ammorbidisce"""
        # Cerca nella tabella il range più vicino
        for start, (end, interpretation) in self.SPREAD_MOVEMENTS_DOWN.items():
            if opening >= start and closing <= end:
                return interpretation
        
        # Fallback generico
        if closing > opening:
            return f"Fiducia nel favorito cala da {opening} a {closing}"
        return "Spread si ammorbidisce, mercato più equilibrato"
    
    def _get_harden_interpretation(self, opening: float, closing: float) -> str:
        """Ottiene interpretazione per spread che si indurisce"""
        for start, (end, interpretation) in self.SPREAD_MOVEMENTS_UP.items():
            if opening >= start and closing <= end:
                return interpretation
        
        # Fallback generico
        if closing < opening:
            return f"Fiducia nel favorito aumenta da {opening} a {closing}"
        return "Spread si indurisce, favorito più forte"


class TotalAnalyzer:
    """Analizzatore per movimenti Total"""
    
    TOTAL_MOVEMENTS_DOWN = {
        3.25: (3.0, "Goleada meno probabile, favorito controlla"),
        3.0: (2.75, "Match più tattico, meno spazio e ritmo"),
        2.75: (2.5, "Squadre attente, equilibrio o controllo"),
        2.5: (2.25, "Calo di ritmo, difese più stabili"),
        2.25: (2.0, "Pochi gol previsti, rischio 0–0"),
        2.0: (1.75, "Partita chiusissima o maltempo"),
    }
    
    TOTAL_MOVEMENTS_UP = {
        1.75: (2.0, "Più fiducia nei gol, squadre non attendiste"),
        2.0: (2.25, "Attacchi in forma, difese normali"),
        2.25: (2.5, "Partita aperta, occasioni da entrambe"),
        2.5: (2.75, "Over spinto, ritmo alto"),
        2.75: (3.0, "Attesa di 3–4 gol, favorito dominante"),
        3.0: (3.25, "Fiducia in goleada, squadre ultra offensive"),
    }
    
    def analyze(self, opening: float, closing: float) -> MovementAnalysis:
        """Analizza il movimento del total"""
        movement = closing - opening  # Positivo = sale, negativo = scende
        
        # Determina direzione
        if abs(movement) < 0.01:
            direction = MovementDirection.STABLE
            intensity = MovementIntensity.NONE
            interpretation = "Total stabile, ritmo atteso invariato"
        elif movement > 0:
            direction = MovementDirection.HARDEN  # Sale = più gol
            interpretation = self._get_up_interpretation(opening, closing)
        else:
            direction = MovementDirection.SOFTEN  # Scende = meno gol
            interpretation = self._get_down_interpretation(opening, closing)
        
        # Calcola intensità
        abs_movement = abs(movement)
        if abs_movement < 0.3:
            intensity = MovementIntensity.LIGHT
        elif abs_movement < 0.6:
            intensity = MovementIntensity.MEDIUM
        else:
            intensity = MovementIntensity.STRONG
        
        steps = abs_movement / 0.25
        
        if direction == MovementDirection.STABLE:
            intensity = MovementIntensity.NONE
        
        return MovementAnalysis(
            direction=direction,
            intensity=intensity,
            opening_value=opening,
            closing_value=closing,
            movement_steps=steps,
            interpretation=interpretation
        )
    
    def _get_up_interpretation(self, opening: float, closing: float) -> str:
        """Interpretazione per total che sale"""
        for start, (end, interpretation) in self.TOTAL_MOVEMENTS_UP.items():
            if opening <= start and closing >= end:
                return interpretation
        return f"Total sale da {opening} a {closing}, più gol attesi"
    
    def _get_down_interpretation(self, opening: float, closing: float) -> str:
        """Interpretazione per total che scende"""
        for start, (end, interpretation) in self.TOTAL_MOVEMENTS_DOWN.items():
            if opening >= start and closing <= end:
                return interpretation
        return f"Total scende da {opening} a {closing}, meno gol attesi"


class MarketMovementAnalyzer:
    """Analizzatore principale dei movimenti di mercato"""
    
    # Matrice 4 combinazioni
    COMBINATION_MATRIX = {
        ("HARDEN", "HARDEN"): {
            "meaning": "Favorito più forte e partita viva",
            "goals_tendency": "🔼 GOAL",
            "recommendations": ["GOAL", "Over", "1 + Over"]
        },
        ("HARDEN", "SOFTEN"): {
            "meaning": "Favorito solido ma tattico",
            "goals_tendency": "🔽 NOGOAL",
            "recommendations": ["1", "Under", "NOGOAL"]
        },
        ("SOFTEN", "HARDEN"): {
            "meaning": "Match più equilibrato e aperto",
            "goals_tendency": "🔼 GOAL",
            "recommendations": ["GOAL", "Over", "X2"]
        },
        ("SOFTEN", "SOFTEN"): {
            "meaning": "Fiducia calante + ritmo basso",
            "goals_tendency": "🔽 NOGOAL",
            "recommendations": ["Under", "X", "NOGOAL"]
        }
    }
    
    def __init__(self):
        self.spread_analyzer = SpreadAnalyzer()
        self.total_analyzer = TotalAnalyzer()
    
    def analyze(self, spread_open: float, spread_close: float,
                total_open: float, total_close: float) -> AnalysisResult:
        """Esegue analisi completa"""

        # Analizza spread e total
        spread_analysis = self.spread_analyzer.analyze(spread_open, spread_close)
        total_analysis = self.total_analyzer.analyze(total_open, total_close)

        # Calcola Expected Goals (xG) dai valori di chiusura
        expected_goals = calculate_expected_goals(spread_close, total_close)

        # Gestisci casi stabili
        spread_dir_key = spread_analysis.direction.name if spread_analysis.direction != MovementDirection.STABLE else "STABLE"
        total_dir_key = total_analysis.direction.name if total_analysis.direction != MovementDirection.STABLE else "STABLE"

        # Ottieni combinazione
        combination = self._get_combination_interpretation(
            spread_analysis, total_analysis, spread_dir_key, total_dir_key
        )

        # Calcola mercati nelle 4 categorie (ora con xG)
        core_recs = self._calculate_core_markets(spread_analysis, total_analysis, combination, expected_goals)
        alternative_recs = self._calculate_alternative_markets(spread_analysis, total_analysis, combination, expected_goals)
        value_recs = self._calculate_value_markets(spread_analysis, total_analysis, combination, expected_goals)
        exchange_recs = self._calculate_exchange_recommendations(spread_analysis, total_analysis)

        # Valida coerenza tra tutte le raccomandazioni
        all_recs = core_recs + alternative_recs + value_recs
        validated_recs = self._validate_market_coherence(all_recs, expected_goals, spread_analysis, total_analysis)

        # Riassegna alle categorie (mantieni ordine originale)
        core_recs = [r for r in validated_recs if r in core_recs]
        alternative_recs = [r for r in validated_recs if r in alternative_recs]
        value_recs = [r for r in validated_recs if r in value_recs]

        # Calcola confidenza generale (migliorata con xG)
        overall_confidence = self._calculate_confidence(spread_analysis, total_analysis, expected_goals)

        return AnalysisResult(
            spread_analysis=spread_analysis,
            total_analysis=total_analysis,
            combination_interpretation=combination["interpretation"],
            core_recommendations=core_recs,
            alternative_recommendations=alternative_recs,
            value_recommendations=value_recs,
            exchange_recommendations=exchange_recs,
            overall_confidence=overall_confidence,
            expected_goals=expected_goals
        )
    
    def _get_combination_interpretation(self, spread: MovementAnalysis, 
                                       total: MovementAnalysis,
                                       spread_key: str, total_key: str) -> Dict:
        """Ottiene interpretazione della combinazione"""
        
        # Se uno dei due è stabile, usa solo l'altro
        if spread_key == "STABLE" and total_key != "STABLE":
            return {
                "interpretation": f"Spread stabile. {total.interpretation}",
                "tendency": "Neutra" if total.direction == MovementDirection.HARDEN else "Tendenza " + total.direction.value
            }
        
        if total_key == "STABLE" and spread_key != "STABLE":
            return {
                "interpretation": f"Total stabile. {spread.interpretation}",
                "tendency": "Neutra"
            }
        
        if spread_key == "STABLE" and total_key == "STABLE":
            return {
                "interpretation": "Mercato stabile, nessun movimento significativo",
                "tendency": "Neutra"
            }
        
        # Cerca nella matrice
        key = (spread_key, total_key)
        if key in self.COMBINATION_MATRIX:
            combo = self.COMBINATION_MATRIX[key]
            return {
                "interpretation": combo["meaning"],
                "tendency": combo["goals_tendency"],
                "recommendations": combo["recommendations"]
            }
        
        # Fallback
        return {
            "interpretation": f"{spread.interpretation}. {total.interpretation}",
            "tendency": "Neutra"
        }
    
    def _calculate_core_markets(self, spread: MovementAnalysis,
                                total: MovementAnalysis, combination: Dict,
                                xg: ExpectedGoals) -> List[MarketRecommendation]:
        """Calcola raccomandazioni principali (CORE) - Alta/Media confidenza"""
        recommendations = []

        abs_spread = abs(spread.closing_value)
        
        # Determina chi è favorito in base al segno dello spread
        # spread < 0 (negativo): Casa favorita (1)
        # spread > 0 (positivo): Trasferta favorita (2)
        favorito = "1" if spread.closing_value < 0 else "2" if spread.closing_value > 0 else "X"

        # Determina underdog (l'opposto del favorito)
        underdog = "2" if favorito == "1" else "1" if favorito == "2" else "X"
        underdog_x = f"X{underdog}" if underdog != "X" else "X"

        # EDGE CASE: Spread quasi pick'em (< 0.25) - Match 50-50
        if abs_spread < 0.25:
            recommendations.append(MarketRecommendation(
                market_name="1X2",
                recommendation="X o 12 (Risultato incerto)",
                confidence=ConfidenceLevel.LOW,
                explanation=f"Spread quasi pick'em ({spread.closing_value}), match 50-50, evita favorito"
            ))
        # EDGE CASE: Spread molto alto (> 2.0) - Favorito schiacciante
        elif abs_spread > 2.0:
            recommendations.append(MarketRecommendation(
                market_name="1X2",
                recommendation=f"{favorito} (Favorito dominante)",
                confidence=ConfidenceLevel.HIGH,
                explanation=f"Spread schiacciante ({format_spread_display(spread.closing_value)}), favorito nettamente superiore"
            ))
        # 1X2 - LOGICA NORMALE: considera movimento + closing value
        elif spread.direction == MovementDirection.HARDEN:
            # Spread si indurisce verso favorito
            conf = ConfidenceLevel.HIGH if spread.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="1X2",
                recommendation=f"{favorito} (Favorito)",
                confidence=conf,
                explanation=f"Spread si indurisce: {spread.interpretation}"
            ))
        elif spread.direction == MovementDirection.SOFTEN:
            # Spread si ammorbidisce → guarda closing value con THRESHOLDS GRANULARI
            if abs_spread < 0.5:
                # Match molto equilibrato
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation=f"X (Pareggio) o {underdog_x}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Spread equilibrato ({spread.closing_value}), favorito molto debole"
                ))
            elif abs_spread < 0.75:
                # Match abbastanza equilibrato - threshold più granulare
                favorito_x = f"{favorito}X" if favorito != "X" else "X"
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation=f"X o {favorito_x} (Incertezza, evita underdog)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Spread si ammorbidisce a {format_spread_display(spread.closing_value)}, match equilibrato"
                ))
            elif abs_spread < 1.0:
                # Leggero favorito - threshold più granulare
                favorito_x = f"{favorito}X" if favorito != "X" else "X"
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation=f"{favorito_x} o {favorito} (Leggero favorito)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Spread {format_spread_display(spread.closing_value)}, leggero favorito"
                ))
            elif abs_spread < 1.5:
                # Favorito medio - threshold più granulare
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation=f"{favorito} (Favorito)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Spread {format_spread_display(spread.closing_value)}, favorito medio nonostante ammorbidimento"
                ))
            else:
                # Favorito forte - threshold più granulare
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation=f"{favorito} (Favorito forte)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Spread {format_spread_display(spread.closing_value)}, favorito ancora forte nonostante ammorbidimento"
                ))
        else:  # STABLE
            # Spread stabile → guarda solo closing value con THRESHOLDS GRANULARI
            if abs_spread < 0.5:
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation="X (Pareggio)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Match equilibrato, spread {spread.closing_value}"
                ))
            elif abs_spread < 0.75:
                # Threshold granulare
                favorito_x = f"{favorito}X" if favorito != "X" else "X"
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation=f"X o {favorito_x}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Spread equilibrato {format_spread_display(spread.closing_value)}"
                ))
            elif abs_spread < 1.0:
                # Threshold granulare
                favorito_x = f"{favorito}X" if favorito != "X" else "X"
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation=f"{favorito_x} o {favorito} (Leggero favorito)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Spread {format_spread_display(spread.closing_value)}, leggero favorito"
                ))
            elif abs_spread < 1.5:
                # Threshold granulare
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation=f"{favorito} (Favorito)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Favorito medio, spread {format_spread_display(spread.closing_value)}"
                ))
            else:
                # Threshold granulare
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation=f"{favorito} (Favorito forte)",
                    confidence=ConfidenceLevel.HIGH,
                    explanation=f"Favorito forte, spread {format_spread_display(spread.closing_value)}"
                ))

        # Over/Under - LOGICA MIGLIORATA: considera movimento + closing value
        if total.direction == MovementDirection.HARDEN:
            # Total sale → Over
            conf = ConfidenceLevel.HIGH if total.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="Over/Under",
                recommendation=f"Over {total.closing_value}",
                confidence=conf,
                explanation=f"Total sale: {total.interpretation}"
            ))
        elif total.direction == MovementDirection.SOFTEN:
            # Total scende → guarda closing value
            if total.closing_value >= 2.75:
                # Scende ma ancora alto → Over con MEDIUM confidence
                recommendations.append(MarketRecommendation(
                    market_name="Over/Under",
                    recommendation=f"Over {total.closing_value}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Total scende ma closing {total.closing_value} ancora alto"
                ))
            else:
                # Scende e basso → Under
                conf = ConfidenceLevel.HIGH if total.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
                recommendations.append(MarketRecommendation(
                    market_name="Over/Under",
                    recommendation=f"Under {total.closing_value}",
                    confidence=conf,
                    explanation=f"Total scende: {total.interpretation}"
                ))
        else:  # STABLE
            # Total stabile → guarda closing value
            if total.closing_value >= 2.75:
                recommendations.append(MarketRecommendation(
                    market_name="Over/Under",
                    recommendation=f"Over {total.closing_value}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Total alto e stabile, partita aperta attesa"
                ))
            elif total.closing_value <= 2.25:
                recommendations.append(MarketRecommendation(
                    market_name="Over/Under",
                    recommendation=f"Under {total.closing_value}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Total basso e stabile, partita tattica attesa"
                ))
            else:
                # Total medio stabile (2.25-2.75) → raccomandazione neutra
                recommendations.append(MarketRecommendation(
                    market_name="Over/Under",
                    recommendation=f"Over {total.closing_value} o Under {total.closing_value}",
                    confidence=ConfidenceLevel.LOW,
                    explanation=f"Total {total.closing_value} neutro, valuta quote exchange"
                ))

        # GOAL/NOGOAL - LOGICA MIGLIORATA con xG e probabilità Poisson
        # Usa probabilità BTTS calcolate da xG per decisioni più precise
        btts_prob = xg.btts_prob
        home_cs_prob = xg.home_clean_sheet_prob
        away_cs_prob = xg.away_clean_sheet_prob

        # EDGE CASE: Total molto basso (< 1.75) - Partita chiusissima
        if total.closing_value < 1.75:
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation="NOGOAL + Under (Partita chiusissima)",
                confidence=ConfidenceLevel.HIGH,
                explanation=f"Total molto basso ({total.closing_value}), P(BTTS)={btts_prob:.1%} molto bassa"
            ))
        # EDGE CASE: Total molto alto (> 3.5) - Goleada
        elif total.closing_value > 3.5:
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation="GOAL + Over (Goleada attesa)",
                confidence=ConfidenceLevel.HIGH,
                explanation=f"Total molto alto ({total.closing_value}), P(BTTS)={btts_prob:.1%} molto alta"
            ))
        # LOGICA CON PROBABILITÀ xG
        elif btts_prob >= 0.65:
            # Alta probabilità BTTS
            conf = ConfidenceLevel.HIGH if btts_prob >= 0.75 else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation="GOAL (Entrambe segnano)",
                confidence=conf,
                explanation=f"Partita viva, P(BTTS)={btts_prob:.1%} (xG: {xg.home_xg:.2f} vs {xg.away_xg:.2f})"
            ))
        elif btts_prob <= 0.40:
            # Bassa probabilità BTTS
            conf = ConfidenceLevel.HIGH if btts_prob <= 0.30 else ConfidenceLevel.MEDIUM
            # Determina quale squadra ha più probabilità clean sheet
            if home_cs_prob > away_cs_prob:
                clean_sheet_team = "1" if spread.closing_value < 0 else "2"
            else:
                clean_sheet_team = "2" if spread.closing_value < 0 else "1"
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation=f"NOGOAL (Almeno una non segna)",
                confidence=conf,
                explanation=f"Partita chiusa, P(BTTS)={btts_prob:.1%}, squadra {clean_sheet_team} potrebbe clean sheet"
            ))
        else:
            # Zona grigia (0.40 - 0.65)
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation=f"GOAL (lievemente favorito)",
                confidence=ConfidenceLevel.LOW,
                explanation=f"Incertezza, P(BTTS)={btts_prob:.1%} in zona media, valuta quote"
            ))

        # Handicap Asiatico - LOGICA MIGLIORATA: bilancia intensità movimento e spread
        # Raccomanda solo se:
        # - Intensità STRONG: abs_spread >= 0.75
        # - Intensità MEDIUM: abs_spread >= 1.0
        # - Intensità LIGHT: abs_spread >= 1.5 (solo spread molto alti)
        should_recommend_handicap = False
        if spread.direction != MovementDirection.STABLE:
            if spread.intensity == MovementIntensity.STRONG and abs_spread >= 0.75:
                should_recommend_handicap = True
            elif spread.intensity == MovementIntensity.MEDIUM and abs_spread >= 1.0:
                should_recommend_handicap = True
            elif spread.intensity == MovementIntensity.LIGHT and abs_spread >= 1.5:
                should_recommend_handicap = True

        if should_recommend_handicap:

            handicap_value = abs_spread
            if spread.direction == MovementDirection.HARDEN:
                # Favorito si rafforza → gioca favorito con handicap
                # Se spread < 0, favorito è casa (1), quindi handicap negativo
                # Se spread > 0, favorito è trasferta (2), quindi handicap positivo
                handicap_sign = "-" if spread.closing_value < 0 else "+"
                conf = ConfidenceLevel.HIGH if spread.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
                recommendations.append(MarketRecommendation(
                    market_name="Handicap Asiatico",
                    recommendation=f"{favorito} {handicap_sign}{handicap_value}",
                    confidence=conf,
                    explanation=f"Favorito copre spread {format_spread_display(spread.closing_value)}"
                ))
            else:  # SOFTEN
                # Spread si ammorbidisce
                if abs_spread >= 1.0:
                    # Ancora favorito forte → Favorito con handicap ridotto
                    handicap_sign = "-" if spread.closing_value < 0 else "+"
                    recommendations.append(MarketRecommendation(
                        market_name="Handicap Asiatico",
                        recommendation=f"{favorito} {handicap_sign}{handicap_value} (valore)",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Spread si ammorbidisce ma {format_spread_display(spread.closing_value)} ancora vantaggioso"
                    ))
                else:
                    # Match equilibrato → Underdog con handicap
                    # Se favorito è casa (1), underdog è trasferta (2) e viceversa
                    recommendations.append(MarketRecommendation(
                        market_name="Handicap Asiatico",
                        recommendation=f"{underdog} +{handicap_value}",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Match equilibrato, underdog copre {format_spread_display(spread.closing_value)}"
                    ))

        # HT Winner - LOGICA MIGLIORATA: considera closing value
        if spread.direction == MovementDirection.HARDEN and spread.intensity in [MovementIntensity.MEDIUM, MovementIntensity.STRONG]:
            # Favorito si rafforza → favorito HT
            recommendations.append(MarketRecommendation(
                market_name="1X2 HT",
                recommendation=f"{favorito} HT (Favorito vince primo tempo)",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Favorito forte parte bene"
            ))
        elif spread.direction == MovementDirection.SOFTEN or (spread.direction == MovementDirection.STABLE and abs_spread < 0.75):
            # Spread si ammorbidisce o stabile equilibrato
            if abs_spread < 1.0:
                # Match equilibrato → X HT
                recommendations.append(MarketRecommendation(
                    market_name="1X2 HT",
                    recommendation="X HT (Pareggio primo tempo)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Match equilibrato ({spread.closing_value}), primo tempo incerto"
                ))
            # Se >= 1.0 ma SOFTEN, non consiglia HT (troppo incerto)

        return recommendations[:5]  # Max 5 core recommendations

    def _calculate_alternative_markets(self, spread: MovementAnalysis,
                                       total: MovementAnalysis, combination: Dict,
                                       xg: ExpectedGoals) -> List[MarketRecommendation]:
        """
        Calcola raccomandazioni alternative - MIGLIORATO con xG e Poisson.

        Usa probabilità precise per HT/FT, Over/Under HT, GOAL HT, Multigol.
        """
        recommendations = []

        abs_spread = abs(spread.closing_value)

        # Determina chi è favorito
        favorito = "1" if spread.closing_value < 0 else "2" if spread.closing_value > 0 else "X"

        # Calcola probabilità HT usando xG
        ht_probs = calculate_halftime_probabilities(xg.home_xg, xg.away_xg)

        # === 1. HT/FT COMBINATIONS - Basato su probabilità Poisson HT e FT ===
        ht_home_prob = ht_probs["home_win_ht"]
        ht_draw_prob = ht_probs["draw_ht"]
        ht_away_prob = ht_probs["away_win_ht"]

        ft_home_prob = xg.home_win_prob
        ft_draw_prob = xg.draw_prob
        ft_away_prob = xg.away_win_prob

        # Trova combinazione HT/FT più probabile
        ht_ft_combinations = [
            ("1/1", ht_home_prob * ft_home_prob),
            ("X/1", ht_draw_prob * ft_home_prob),
            ("2/1", ht_away_prob * ft_home_prob),
            ("1/X", ht_home_prob * ft_draw_prob),
            ("X/X", ht_draw_prob * ft_draw_prob),
            ("2/X", ht_away_prob * ft_draw_prob),
            ("1/2", ht_home_prob * ft_away_prob),
            ("X/2", ht_draw_prob * ft_away_prob),
            ("2/2", ht_away_prob * ft_away_prob)
        ]
        ht_ft_combinations.sort(key=lambda x: x[1], reverse=True)

        # Raccomanda top 1-2 combinazioni HT/FT
        top_ht_ft = ht_ft_combinations[0]
        if top_ht_ft[1] >= 0.15:  # Almeno 15% probabilità
            conf = ConfidenceLevel.HIGH if top_ht_ft[1] >= 0.25 else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="HT/FT",
                recommendation=f"{top_ht_ft[0]}",
                confidence=conf,
                explanation=f"Probabilità Poisson: {top_ht_ft[1]:.1%} (HT xG: {ht_probs['home_xg_ht']:.2f} vs {ht_probs['away_xg_ht']:.2f})"
            ))

        # === 2. OVER/UNDER HT - Basato su xG HT ===
        total_xg_ht = ht_probs["total_xg_ht"]
        over_05_ht_prob = ht_probs["over_05_ht"]
        over_15_ht_prob = ht_probs["over_15_ht"]

        if over_15_ht_prob >= 0.55:
            # Alta prob Over 1.5 HT
            recommendations.append(MarketRecommendation(
                market_name="Over/Under HT",
                recommendation="Over 1.5 HT",
                confidence=ConfidenceLevel.MEDIUM,
                explanation=f"P(Over 1.5 HT)={over_15_ht_prob:.1%}, xG HT totale={total_xg_ht:.2f}"
            ))
        elif over_05_ht_prob >= 0.70:
            # Alta prob Over 0.5 HT
            recommendations.append(MarketRecommendation(
                market_name="Over/Under HT",
                recommendation="Over 0.5 HT",
                confidence=ConfidenceLevel.MEDIUM,
                explanation=f"P(Over 0.5 HT)={over_05_ht_prob:.1%}, almeno 1 gol 1T probabile"
            ))
        else:
            # Under HT
            recommendations.append(MarketRecommendation(
                market_name="Over/Under HT",
                recommendation="Under 0.5 HT",
                confidence=ConfidenceLevel.MEDIUM,
                explanation=f"P(0 gol HT)={(1-over_05_ht_prob):.1%}, primo tempo tattico"
            ))

        # === 3. GOAL/NOGOAL HT - Basato su P(BTTS HT) ===
        btts_ht_prob = ht_probs["btts_ht"]

        if btts_ht_prob >= 0.40:
            conf = ConfidenceLevel.HIGH if btts_ht_prob >= 0.55 else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL HT",
                recommendation="GOAL 1T (Entrambe segnano 1T)",
                confidence=conf,
                explanation=f"P(BTTS HT)={btts_ht_prob:.1%}, entrambe offensive nel 1T"
            ))
        else:
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL HT",
                recommendation="NOGOAL 1T",
                confidence=ConfidenceLevel.MEDIUM,
                explanation=f"P(BTTS HT)={btts_ht_prob:.1%}, almeno una squadra non segna 1T"
            ))

        # === 4. MULTIGOL - Basato su distribuzione Poisson ===
        goals_dist = calculate_total_goals_distribution(xg.home_xg, xg.away_xg)

        # Calcola probabilità per range multigol
        prob_0_1 = goals_dist.get(0, 0) + goals_dist.get(1, 0)
        prob_1_2 = goals_dist.get(1, 0) + goals_dist.get(2, 0)
        prob_1_3 = sum(goals_dist.get(i, 0) for i in range(1, 4))
        prob_2_3 = goals_dist.get(2, 0) + goals_dist.get(3, 0)
        prob_2_4 = sum(goals_dist.get(i, 0) for i in range(2, 5))
        prob_3_5 = sum(goals_dist.get(i, 0) for i in range(3, 6))

        # Trova range multigol più probabile
        multigol_ranges = [
            ("0-1 gol", prob_0_1),
            ("1-2 gol", prob_1_2),
            ("1-3 gol", prob_1_3),
            ("2-3 gol", prob_2_3),
            ("2-4 gol", prob_2_4),
            ("3-5 gol", prob_3_5)
        ]
        multigol_ranges.sort(key=lambda x: x[1], reverse=True)

        top_multigol = multigol_ranges[0]
        if top_multigol[1] >= 0.30:  # Almeno 30% probabilità
            recommendations.append(MarketRecommendation(
                market_name="Multigol",
                recommendation=top_multigol[0],
                confidence=ConfidenceLevel.MEDIUM,
                explanation=f"Probabilità {top_multigol[1]:.1%} (xG totale={xg.home_xg + xg.away_xg:.2f})"
            ))

        # === 5. COMBO - Basato su probabilità 1X2 e O/U ===
        # Trova miglior combo favorito + O/U
        if favorito in ["1", "2"]:
            fav_prob = ft_home_prob if favorito == "1" else ft_away_prob

            # Determina Over o Under più probabile
            total_xg_ft = xg.home_xg + xg.away_xg
            ou_rec = f"Over {total.closing_value}" if total_xg_ft > total.closing_value else f"Under {total.closing_value}"

            if fav_prob >= 0.50:
                recommendations.append(MarketRecommendation(
                    market_name="Combo",
                    recommendation=f"{favorito} + {ou_rec}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"P({favorito})={fav_prob:.1%}, xG totale={total_xg_ft:.2f}"
                ))

        return recommendations[:6]  # Max 6 alternative recommendations

    def _calculate_value_markets(self, spread: MovementAnalysis,
                                 total: MovementAnalysis, combination: Dict,
                                 xg: ExpectedGoals) -> List[MarketRecommendation]:
        """
        Calcola value bets - Mercati avanzati con xG e Poisson.

        Include: Risultati Esatti, First to Score, Winning Margin, Team Totals
        """
        recommendations = []

        abs_spread = abs(spread.closing_value)
        favorito = "1" if spread.closing_value < 0 else "2" if spread.closing_value > 0 else "X"
        underdog = "2" if favorito == "1" else "1" if favorito == "2" else "X"

        # === 1. RISULTATI ESATTI - Top 3 usando Poisson ===
        exact_scores_poisson = get_most_likely_score(xg.home_xg, xg.away_xg, top_n=5)
        for score, prob in exact_scores_poisson[:3]:  # Max 3
            recommendations.append(MarketRecommendation(
                market_name="Risultato Esatto",
                recommendation=score,
                confidence=ConfidenceLevel.LOW,
                explanation=f"Probabilità Poisson: {prob:.1%} (xG {xg.home_xg:.2f} vs {xg.away_xg:.2f})"
            ))

        # === 2. FIRST TO SCORE - Chi segna per primo ===
        first_scorer = calculate_first_to_score(xg.home_xg, xg.away_xg)

        if first_scorer["home_first"] > first_scorer["away_first"] * 1.3:
            # Casa nettamente favorita a segnare prima
            recommendations.append(MarketRecommendation(
                market_name="First to Score",
                recommendation="Casa segna per prima",
                confidence=ConfidenceLevel.LOW,
                explanation=f"P(Casa prima)={first_scorer['home_first']:.1%}, più offensiva"
            ))
        elif first_scorer["away_first"] > first_scorer["home_first"] * 1.3:
            # Trasferta favorita a segnare prima
            recommendations.append(MarketRecommendation(
                market_name="First to Score",
                recommendation="Trasferta segna per prima",
                confidence=ConfidenceLevel.LOW,
                explanation=f"P(Trasferta prima)={first_scorer['away_first']:.1%}, più pericolosa"
            ))
        elif first_scorer["no_goal"] >= 0.15:
            # Alta probabilità 0-0
            recommendations.append(MarketRecommendation(
                market_name="First to Score",
                recommendation="Nessun gol (0-0)",
                confidence=ConfidenceLevel.LOW,
                explanation=f"P(0-0)={first_scorer['no_goal']:.1%}, partita bloccata"
            ))

        # === 3. WINNING MARGIN - Margine di vittoria ===
        margins = calculate_winning_margin(xg.home_xg, xg.away_xg)

        # Trova margine più probabile
        margins_sorted = sorted(margins.items(), key=lambda x: x[1], reverse=True)
        top_margin = margins_sorted[0]

        if top_margin[0] != "draw" and top_margin[1] >= 0.20:
            margin_names = {
                "home_1": "Casa vince per 1 gol",
                "home_2": "Casa vince per 2 gol",
                "home_3+": "Casa vince per 3+ gol",
                "away_1": "Trasferta vince per 1 gol",
                "away_2": "Trasferta vince per 2 gol",
                "away_3+": "Trasferta vince per 3+ gol"
            }
            if top_margin[0] in margin_names:
                recommendations.append(MarketRecommendation(
                    market_name="Winning Margin",
                    recommendation=margin_names[top_margin[0]],
                    confidence=ConfidenceLevel.LOW,
                    explanation=f"Probabilità {top_margin[1]:.1%} (xG gap={abs(xg.home_xg - xg.away_xg):.2f})"
                ))

        # === 4. TEAM TOTALS - Over/Under per singola squadra ===
        team_totals = calculate_team_totals(xg.home_xg, xg.away_xg)

        # Casa Over/Under
        if team_totals["home_over_15"] >= 0.55:
            recommendations.append(MarketRecommendation(
                market_name="Team Total Casa",
                recommendation="Casa Over 1.5 gol",
                confidence=ConfidenceLevel.LOW,
                explanation=f"P(Casa >1.5)={team_totals['home_over_15']:.1%}, xG casa={xg.home_xg:.2f}"
            ))
        elif team_totals["home_over_05"] >= 0.70:
            recommendations.append(MarketRecommendation(
                market_name="Team Total Casa",
                recommendation="Casa Over 0.5 gol",
                confidence=ConfidenceLevel.LOW,
                explanation=f"P(Casa >0.5)={team_totals['home_over_05']:.1%}"
            ))

        # Trasferta Over/Under
        if team_totals["away_over_15"] >= 0.55:
            recommendations.append(MarketRecommendation(
                market_name="Team Total Trasferta",
                recommendation="Trasferta Over 1.5 gol",
                confidence=ConfidenceLevel.LOW,
                explanation=f"P(Trasferta >1.5)={team_totals['away_over_15']:.1%}, xG trasferta={xg.away_xg:.2f}"
            ))
        elif team_totals["away_over_05"] >= 0.70:
            recommendations.append(MarketRecommendation(
                market_name="Team Total Trasferta",
                recommendation="Trasferta Over 0.5 gol",
                confidence=ConfidenceLevel.LOW,
                explanation=f"P(Trasferta >0.5)={team_totals['away_over_05']:.1%}"
            ))

        return recommendations[:8]  # Max 8 value recommendations (aumentato per nuovi mercati)

    def _calculate_exchange_recommendations(self, spread: MovementAnalysis,
                                           total: MovementAnalysis) -> List[MarketRecommendation]:
        """Calcola consigli Exchange - Punta (Back) vs Banca (Lay)"""
        recommendations = []

        abs_spread = abs(spread.closing_value)

        # Determina chi è favorito
        favorito = "1" if spread.closing_value < 0 else "2" if spread.closing_value > 0 else "X"
        underdog = "2" if favorito == "1" else "1" if favorito == "2" else "X"

        # === SPREAD: Punta/Banca sul favorito ===
        if spread.direction == MovementDirection.HARDEN and spread.intensity in [MovementIntensity.MEDIUM, MovementIntensity.STRONG]:
            # Spread si indurisce verso favorito → Punta favorito, Banca underdog
            conf = ConfidenceLevel.HIGH if spread.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="Exchange 1X2",
                recommendation=f"✅ PUNTA {favorito} / ❌ BANCA {underdog}",
                confidence=conf,
                explanation=f"Favorito {favorito} si rafforza ({spread.intensity.value.lower()})"
            ))
        elif spread.direction == MovementDirection.SOFTEN and abs_spread < 1.0:
            # Spread si ammorbidisce e basso → Banca favorito, Punta underdog
            recommendations.append(MarketRecommendation(
                market_name="Exchange 1X2",
                recommendation=f"✅ PUNTA {underdog} o X / ❌ BANCA {favorito}",
                confidence=ConfidenceLevel.MEDIUM,
                explanation=f"Favorito perde fiducia, spread {format_spread_display(spread.closing_value)}"
            ))
        elif abs_spread >= 1.5 and spread.direction != MovementDirection.SOFTEN:
            # Spread molto alto e non in calo → Punta favorito
            recommendations.append(MarketRecommendation(
                market_name="Exchange 1X2",
                recommendation=f"✅ PUNTA {favorito}",
                confidence=ConfidenceLevel.MEDIUM,
                explanation=f"Spread alto {format_spread_display(spread.closing_value)}, favorito forte"
            ))

        # === TOTAL: Punta/Banca Over/Under ===
        if total.direction == MovementDirection.HARDEN:
            # Total sale → Punta Over, Banca Under
            conf = ConfidenceLevel.HIGH if total.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="Exchange Over/Under",
                recommendation=f"✅ PUNTA Over {total.closing_value} / ❌ BANCA Under {total.closing_value}",
                confidence=conf,
                explanation=f"Total sale a {total.closing_value} ({total.intensity.value.lower()})"
            ))
        elif total.direction == MovementDirection.SOFTEN:
            # Total scende → Punta Under, Banca Over
            conf = ConfidenceLevel.HIGH if total.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="Exchange Over/Under",
                recommendation=f"✅ PUNTA Under {total.closing_value} / ❌ BANCA Over {total.closing_value}",
                confidence=conf,
                explanation=f"Total scende a {total.closing_value} ({total.intensity.value.lower()})"
            ))
        elif total.closing_value >= 3.0:
            # Total molto alto → Punta Over
            recommendations.append(MarketRecommendation(
                market_name="Exchange Over/Under",
                recommendation=f"✅ PUNTA Over {total.closing_value}",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Total molto alto, molti gol attesi"
            ))
        elif total.closing_value <= 2.0:
            # Total molto basso → Punta Under
            recommendations.append(MarketRecommendation(
                market_name="Exchange Over/Under",
                recommendation=f"✅ PUNTA Under {total.closing_value}",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Total molto basso, pochi gol attesi"
            ))

        # === GOAL/NOGOAL Exchange ===
        if total.closing_value >= 2.75 and total.direction != MovementDirection.SOFTEN:
            recommendations.append(MarketRecommendation(
                market_name="Exchange GOAL",
                recommendation="✅ PUNTA GOAL / ❌ BANCA NOGOAL",
                confidence=ConfidenceLevel.MEDIUM,
                explanation=f"Total {total.closing_value}, partita viva"
            ))
        elif total.closing_value <= 2.0 and total.direction != MovementDirection.HARDEN:
            recommendations.append(MarketRecommendation(
                market_name="Exchange GOAL",
                recommendation="✅ PUNTA NOGOAL / ❌ BANCA GOAL",
                confidence=ConfidenceLevel.MEDIUM,
                explanation=f"Total {total.closing_value}, partita chiusa"
            ))

        return recommendations[:4]  # Max 4 exchange recommendations

    def _get_likely_exact_scores(self, spread: MovementAnalysis, total: MovementAnalysis) -> List[Tuple[str, str]]:
        """Restituisce i risultati esatti più probabili basati su spread e total"""
        scores = []

        # Determina chi è favorito
        # spread < 0 (negativo): Casa favorita (1)
        # spread > 0 (positivo): Trasferta favorita (2)
        favorito = "1" if spread.closing_value < 0 else "2" if spread.closing_value > 0 else "X"

        # Favorito forte + molti gol
        if spread.direction == MovementDirection.HARDEN and total.closing_value >= 2.75:
            if spread.intensity == MovementIntensity.STRONG:
                scores.append(("3-0", "Favorito dominante con molti gol"))
                scores.append(("3-1", "Favorito vince largamente"))
                scores.append(("2-0", "Favorito controlla"))
            else:
                scores.append(("2-0", "Favorito vince con gol"))
                scores.append(("2-1", "Favorito vince, underdog segna"))
                scores.append(("3-1", "Vittoria larga"))

        # Favorito forte + pochi gol
        elif spread.direction == MovementDirection.HARDEN and total.closing_value <= 2.25:
            scores.append(("1-0", "Favorito vince di corto"))
            scores.append(("2-0", "Favorito controlla, pochi gol"))
            scores.append(("2-1", "Vittoria solida"))

        # Favorito debole + molti gol
        elif spread.direction == MovementDirection.SOFTEN and total.closing_value >= 2.75:
            scores.append(("2-2", "Match aperto e equilibrato"))
            scores.append(("1-2", "Underdog sorprende"))
            scores.append(("2-1", "Match combattuto"))

        # Favorito debole + pochi gol
        elif spread.direction == MovementDirection.SOFTEN and total.closing_value <= 2.25:
            scores.append(("1-1", "Pareggio equilibrato"))
            scores.append(("0-0", "Nessun gol, match chiuso"))
            scores.append(("1-0", "Vittoria risicata"))

        # Spread stabile
        elif spread.direction == MovementDirection.STABLE:
            if total.closing_value >= 2.75:
                scores.append(("2-1", "Match normale con gol"))
                scores.append(("2-2", "Pareggio con gol"))
                scores.append(("1-2", "Match combattuto"))
            elif total.closing_value <= 2.25:
                scores.append(("1-0", "Vittoria di misura"))
                scores.append(("0-1", "Singolo gol decide"))
                scores.append(("1-1", "Pareggio"))
            else:  # Total medio
                scores.append(("1-1", "Pareggio standard"))
                scores.append(("2-1", "Vittoria normale"))
                scores.append(("1-0", "Gol decisivo"))

        # Default
        if not scores:
            scores.append(("1-1", "Pareggio possibile"))
            scores.append(("2-1", "Risultato comune"))
            scores.append(("1-0", "Vittoria di misura"))

        # Se favorito è trasferta (2), inverti tutti i punteggi
        # Es: "3-0" diventa "0-3", "2-1" diventa "1-2"
        if favorito == "2":
            inverted_scores = []
            for score, explanation in scores:
                # Inverti il punteggio
                parts = score.split('-')
                inverted_score = f"{parts[1]}-{parts[0]}"
                inverted_scores.append((inverted_score, explanation))
            scores = inverted_scores

        return scores
    
    def _calculate_confidence(self, spread: MovementAnalysis,
                             total: MovementAnalysis,
                             xg: ExpectedGoals) -> ConfidenceLevel:
        """Calcola confidenza generale - MIGLIORATO: considera intensità relativa, coerenza e xG"""

        # Calcola score di confidence (0-100)
        confidence_score = 50  # Base: MEDIUM

        # 1. DIREZIONE MOVIMENTI (±20 punti)
        if spread.direction == total.direction and spread.direction != MovementDirection.STABLE:
            # Movimenti concordi (stessa direzione) → +20 confidence
            confidence_score += 20
        elif (spread.direction == MovementDirection.HARDEN and total.direction == MovementDirection.SOFTEN) or \
             (spread.direction == MovementDirection.SOFTEN and total.direction == MovementDirection.HARDEN):
            # Segnali contrastanti (direzioni opposte) → -20 confidence
            confidence_score -= 20
        # STABLE non cambia score

        # 2. INTENSITÀ RELATIVA (±15 punti)
        strong_movements = 0
        if spread.intensity == MovementIntensity.STRONG:
            strong_movements += 1
        if total.intensity == MovementIntensity.STRONG:
            strong_movements += 1

        if strong_movements == 2:
            # Entrambi STRONG → +15
            confidence_score += 15
        elif strong_movements == 1:
            # Uno STRONG → +5
            confidence_score += 5
        elif spread.intensity == MovementIntensity.LIGHT and total.intensity == MovementIntensity.LIGHT:
            # Entrambi LIGHT → -10
            confidence_score -= 10

        # 3. COERENZA xG con Spread (±15 punti)
        # Verifica che le probabilità 1X2 siano coerenti con spread
        abs_spread = abs(spread.closing_value)
        home_prob = xg.home_win_prob
        away_prob = xg.away_win_prob
        draw_prob = xg.draw_prob

        # Determina favorito da xG
        if home_prob > away_prob + 0.15:
            xg_favorite = "1"  # Casa favorita
        elif away_prob > home_prob + 0.15:
            xg_favorite = "2"  # Trasferta favorita
        else:
            xg_favorite = "X"  # Equilibrio

        # Determina favorito da spread
        if spread.closing_value < -0.5:
            spread_favorite = "1"
        elif spread.closing_value > 0.5:
            spread_favorite = "2"
        else:
            spread_favorite = "X"

        # Verifica coerenza
        if xg_favorite == spread_favorite:
            # xG e spread concordano → +15
            confidence_score += 15
        elif (xg_favorite == "X" or spread_favorite == "X"):
            # Uno dice equilibrio, l'altro no → neutra (0)
            pass
        else:
            # xG e spread in disaccordo → -15
            confidence_score -= 15

        # 4. VALORE ASSOLUTO MOVIMENTI (±10 punti)
        # Movimenti più grandi = più fiducia
        total_movement = abs(spread.movement_steps) + abs(total.movement_steps)
        if total_movement >= 3.0:
            # Movimento totale >= 0.75 (3 steps) → +10
            confidence_score += 10
        elif total_movement <= 1.0:
            # Movimento totale <= 0.25 (1 step) → -10
            confidence_score -= 10

        # Converti score in ConfidenceLevel
        if confidence_score >= 70:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 40:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


    def _validate_market_coherence(self, all_recommendations: List[MarketRecommendation],
                                   xg: ExpectedGoals, spread: MovementAnalysis,
                                   total: MovementAnalysis) -> List[MarketRecommendation]:
        """
        Valida coerenza tra raccomandazioni e rimuove/modifica quelle contraddittorie.

        Regole di coerenza:
        1. Over/Under: se raccomando "Over X", non raccomandare "Under Y" se Y > X
        2. GOAL/NOGOAL: GOAL dovrebbe allinearsi con Over, NOGOAL con Under
        3. Favorito forte (spread > 1.5) non dovrebbe avere "X" come raccomandazione primaria
        4. BTTS alto (>70%) incompatibile con NOGOAL HIGH confidence
        """
        coherent_recs = []

        # Raccogli informazioni dalle raccomandazioni
        has_over = False
        has_under = False
        has_goal = False
        has_nogoal = False
        over_value = None
        under_value = None

        for rec in all_recommendations:
            if "Over" in rec.recommendation and "Over/Under" in rec.market_name:
                has_over = True
                # Estrai valore (es. "Over 2.5" → 2.5)
                try:
                    over_value = float(rec.recommendation.split()[-1])
                except:
                    pass

            if "Under" in rec.recommendation and "Over/Under" in rec.market_name:
                has_under = True
                try:
                    under_value = float(rec.recommendation.split()[-1])
                except:
                    pass

            if rec.market_name == "GOAL/NOGOAL":
                if "GOAL" in rec.recommendation and "NOGOAL" not in rec.recommendation:
                    has_goal = True
                elif "NOGOAL" in rec.recommendation:
                    has_nogoal = True

        # Validazione 1: Over/Under contraddittori
        if has_over and has_under and over_value and under_value:
            if over_value == under_value:
                # Raccomando sia Over che Under dello stesso valore → rimuovi quello con confidence più bassa
                # Filtra e mantieni solo quello con confidence più alta
                over_rec = None
                under_rec = None
                for rec in all_recommendations:
                    if "Over" in rec.recommendation and "Over/Under" in rec.market_name:
                        over_rec = rec
                    if "Under" in rec.recommendation and "Over/Under" in rec.market_name:
                        under_rec = rec

                if over_rec and under_rec:
                    # Usa xG per decidere quale mantenere
                    total_xg = xg.home_xg + xg.away_xg
                    if total_xg > under_value:
                        # xG suggerisce Over → rimuovi Under
                        all_recommendations = [r for r in all_recommendations if r != under_rec]
                    else:
                        # xG suggerisce Under → rimuovi Over
                        all_recommendations = [r for r in all_recommendations if r != over_rec]

        # Validazione 2: GOAL/NOGOAL vs Over/Under
        if has_goal and has_under:
            # GOAL incompatibile con Under (specialmente se total basso)
            if total.closing_value <= 2.25:
                # Contraddizione: mantengo quello con confidence più alta
                goal_rec = next((r for r in all_recommendations if r.market_name == "GOAL/NOGOAL" and "GOAL" in r.recommendation), None)
                under_rec = next((r for r in all_recommendations if "Under" in r.recommendation and "Over/Under" in r.market_name), None)

                if goal_rec and under_rec:
                    # Usa xG BTTS per decidere
                    if xg.btts_prob > 0.5:
                        # Favorisci GOAL → rimuovi Under se ha confidence <= MEDIUM
                        if under_rec.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]:
                            all_recommendations = [r for r in all_recommendations if r != under_rec]
                    else:
                        # Favorisci Under → rimuovi GOAL se ha confidence <= MEDIUM
                        if goal_rec.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]:
                            all_recommendations = [r for r in all_recommendations if r != goal_rec]

        if has_nogoal and has_over:
            # NOGOAL incompatibile con Over (specialmente se total alto)
            if total.closing_value >= 2.75:
                nogoal_rec = next((r for r in all_recommendations if r.market_name == "GOAL/NOGOAL" and "NOGOAL" in r.recommendation), None)
                over_rec = next((r for r in all_recommendations if "Over" in r.recommendation and "Over/Under" in r.market_name), None)

                if nogoal_rec and over_rec:
                    # Usa xG BTTS per decidere
                    if xg.btts_prob > 0.5:
                        # Favorisci Over → rimuovi NOGOAL se ha confidence <= MEDIUM
                        if nogoal_rec.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]:
                            all_recommendations = [r for r in all_recommendations if r != nogoal_rec]
                    else:
                        # Favorisci NOGOAL → rimuovi Over se ha confidence <= MEDIUM
                        if over_rec.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]:
                            all_recommendations = [r for r in all_recommendations if r != over_rec]

        # Validazione 3: Favorito forte non dovrebbe avere "X" primario
        abs_spread = abs(spread.closing_value)
        if abs_spread >= 1.5:
            # Favorito molto forte → rimuovi raccomandazioni "X" (Pareggio) con HIGH confidence
            all_recommendations = [
                r for r in all_recommendations
                if not (r.recommendation.strip() == "X (Pareggio)" and r.confidence == ConfidenceLevel.HIGH)
            ]

        return all_recommendations


def format_spread_display(spread_value: float) -> str:
    """
    Formatta lo spread per la visualizzazione corretta.
    - (-) indica casa (home) favorita
    - (+) indica trasferta (away) favorita

    Args:
        spread_value: Valore spread
                      (negativo = casa favorita, positivo = trasferta favorita)

    Returns:
        Stringa formattata con segno corretto
    """
    if spread_value is None:
        return "N/A"

    # Visualizza spread senza inversione
    # spread < 0 → casa favorita → mostra con -
    # spread > 0 → trasferta favorita → mostra con +
    if spread_value < 0:
        return f"-{abs(spread_value):.2f}"
    elif spread_value > 0:
        return f"+{abs(spread_value):.2f}"
    else:
        return "0.00"


def format_output(result: AnalysisResult) -> str:
    """Formatta l'output in modo leggibile"""
    
    output = []
    output.append("=" * 60)
    output.append("📊 ANALISI MOVIMENTO MERCATO")
    output.append("=" * 60)
    output.append("")
    
    # Movimenti
    output.append("📈 MOVIMENTI:")
    output.append(f"   Spread: {format_spread_display(result.spread_analysis.opening_value)} → {format_spread_display(result.spread_analysis.closing_value)}")
    output.append(f"           {result.spread_analysis.direction.value} {result.spread_analysis.intensity.value}")
    output.append(f"   Total:  {result.total_analysis.opening_value:.2f} → {result.total_analysis.closing_value:.2f}")
    output.append(f"           {result.total_analysis.direction.value} {result.total_analysis.intensity.value}")
    output.append("")
    
    # Interpretazione
    output.append("🎯 INTERPRETAZIONE:")
    output.append(f"   {result.combination_interpretation}")
    output.append("")
    
    # Confidenza
    output.append(f"✅ CONFIDENZA GENERALE: {result.overall_confidence.value}")
    output.append("")
    
    # CORE RECOMMENDATIONS
    output.append("🎯 RACCOMANDAZIONI CORE (Alta/Media confidenza):")
    for i, market in enumerate(result.core_recommendations, 1):
        conf_icon = "🟢" if market.confidence == ConfidenceLevel.HIGH else "🟡" if market.confidence == ConfidenceLevel.MEDIUM else "🔴"
        output.append(f"   {i}  {conf_icon} {market.market_name}: {market.recommendation}")
        output.append(f"      └─ {market.explanation}")
    output.append("")

    # ALTERNATIVE RECOMMENDATIONS
    if result.alternative_recommendations:
        output.append("💼 OPZIONI ALTERNATIVE (Media confidenza):")
        for i, market in enumerate(result.alternative_recommendations, 1):
            conf_icon = "🟢" if market.confidence == ConfidenceLevel.HIGH else "🟡" if market.confidence == ConfidenceLevel.MEDIUM else "🔴"
            output.append(f"   {i}  {conf_icon} {market.market_name}: {market.recommendation}")
            output.append(f"      └─ {market.explanation}")
        output.append("")

    # VALUE BETS
    if result.value_recommendations:
        output.append("💎 VALUE BETS (Bassa confidenza, potenziale valore):")
        for i, market in enumerate(result.value_recommendations, 1):
            conf_icon = "🟢" if market.confidence == ConfidenceLevel.HIGH else "🟡" if market.confidence == ConfidenceLevel.MEDIUM else "🔴"
            output.append(f"   {i}  {conf_icon} {market.market_name}: {market.recommendation}")
            output.append(f"      └─ {market.explanation}")
        output.append("")

    # EXCHANGE RECOMMENDATIONS (PUNTA/BANCA)
    if result.exchange_recommendations:
        output.append("🔄 CONSIGLI EXCHANGE (Punta/Banca):")
        for i, market in enumerate(result.exchange_recommendations, 1):
            conf_icon = "🟢" if market.confidence == ConfidenceLevel.HIGH else "🟡" if market.confidence == ConfidenceLevel.MEDIUM else "🔴"
            output.append(f"   {i}  {conf_icon} {market.market_name}: {market.recommendation}")
            output.append(f"      └─ {market.explanation}")
        output.append("")

    output.append("=" * 60)
    
    return "\n".join(output)


def get_user_input() -> Tuple[float, float, float, float]:
    """Raccoglie input dall'utente"""
    print("\n" + "=" * 60)
    print("📥 INSERISCI I VALORI DEL MERCATO")
    print("=" * 60)
    print()
    
    try:
        spread_open = float(input("Spread Apertura (es. -1.5): ").strip())
        spread_close = float(input("Spread Chiusura (es. -1.0): ").strip())
        total_open = float(input("Total Apertura (es. 2.5): ").strip())
        total_close = float(input("Total Chiusura (es. 2.75): ").strip())
        
        return spread_open, spread_close, total_open, total_close
    except ValueError:
        print("\n❌ Errore: inserisci valori numerici validi!")
        return get_user_input()


def main():
    """Funzione principale"""
    print("\n" + "=" * 60)
    print("🏆 MARKET MOVEMENT ANALYZER")
    print("=" * 60)
    print("\nTool per analizzare movimenti Spread e Total")
    print("e generare interpretazioni e giocate consigliate.\n")
    
    analyzer = MarketMovementAnalyzer()
    
    while True:
        try:
            # Input
            spread_open, spread_close, total_open, total_close = get_user_input()
            
            # Analisi
            print("\n🔄 Analisi in corso...\n")
            result = analyzer.analyze(spread_open, spread_close, total_open, total_close)
            
            # Output
            output = format_output(result)
            print(output)
            
            # Continua?
            print()
            continue_choice = input("Vuoi analizzare un altro movimento? (s/n): ").strip().lower()
            if continue_choice not in ['s', 'si', 'y', 'yes']:
                break
            
            print("\n" * 2)
            
        except KeyboardInterrupt:
            print("\n\n👋 Arrivederci!")
            break
        except Exception as e:
            print(f"\n❌ Errore: {e}")
            print("Riprova...\n")


if __name__ == "__main__":
    main()

