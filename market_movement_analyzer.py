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

    # ===== OPZIONE C: ADVANCED PREDICTIONS =====
    market_adjusted_1x2: Optional[Dict] = None  # 1X2 aggiustato con movimenti
    bayesian_btts: Optional[Dict] = None  # BTTS con Bayesian update
    confidence_score: Optional[float] = None  # Market confidence (0-100)
    prediction_method: Optional[str] = None  # "bayesian_ensemble", "xg_only", ecc.
    ensemble_weights: Optional[Dict] = None  # Pesi usati (xg, spread, movement)


@dataclass
class MarketIntelligence:
    """Advanced Market Intelligence indicators"""
    sharp_money_detected: bool
    sharp_spread_velocity: float  # % velocity spread
    sharp_total_velocity: float  # % velocity total
    contrarian_signal: bool
    sharp_confidence_boost: float

    steam_move_detected: bool
    steam_magnitude: float
    reverse_steam: bool
    steam_direction: str  # "favorito" o "underdog"

    correlation_score: float  # -1 to +1
    correlation_interpretation: str
    market_coherent: bool

    on_key_spread: bool
    on_key_total: bool
    spread_key_number: Optional[float]
    total_key_number: Optional[float]
    key_confidence_boost: float

    efficiency_score: float  # 0-100
    efficiency_status: str  # "Efficient", "Normal", "Inefficient"
    value_opportunity: bool


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
    market_intelligence: MarketIntelligence  # Advanced market indicators


# ============================================================================
# ADVANCED MARKET INTELLIGENCE FUNCTIONS
# ============================================================================

def detect_sharp_money(spread_open: float, spread_close: float,
                      total_open: float, total_close: float) -> Dict:
    """
    Rileva movimento 'sharp' (professionisti) vs 'public' (amatori).

    Sharp money indicators:
    - Movimento rapido (>15% spread, >10% total)
    - Movimento contrastante (spread e total in direzioni opposte)

    Args:
        spread_open: Spread apertura
        spread_close: Spread chiusura
        total_open: Total apertura
        total_close: Total chiusura

    Returns:
        Dict con sharp money indicators
    """
    # Calcola velocità movimento in %
    spread_change = abs(spread_close - spread_open)
    total_change = abs(total_close - total_open)

    # Evita divisione per zero
    spread_base = max(abs(spread_open), 0.1)
    total_base = max(abs(total_open), 0.1)

    spread_velocity = (spread_change / spread_base) * 100
    total_velocity = (total_change / total_base) * 100

    # Sharp thresholds
    is_sharp_spread = spread_velocity > 15  # >15% = sharp
    is_sharp_total = total_velocity > 10    # >10% = sharp

    # Contrarian signal: direzioni opposte
    spread_direction = spread_close - spread_open
    total_direction = total_close - total_open
    contrarian = (spread_direction * total_direction) < 0

    # Sharp money detected se almeno uno è sharp
    sharp_detected = is_sharp_spread or is_sharp_total

    # Confidence boost
    confidence_boost = 0.0
    if contrarian:
        confidence_boost += 0.15
    if is_sharp_spread and is_sharp_total:
        confidence_boost += 0.10

    return {
        "sharp_detected": sharp_detected,
        "spread_velocity": spread_velocity,
        "total_velocity": total_velocity,
        "contrarian": contrarian,
        "confidence_boost": confidence_boost
    }


def detect_steam_move(spread_open: float, spread_close: float) -> Dict:
    """
    Steam Move = movimento improvviso >0.5 punti in spread.
    Indica denaro istituzionale massiccio.

    Args:
        spread_open: Spread apertura
        spread_close: Spread chiusura

    Returns:
        Dict con steam move indicators
    """
    movement = abs(spread_close - spread_open)

    # Steam threshold: >=0.5 punti
    is_steam = movement >= 0.5

    # Reverse steam: cambia segno (favorito diventa underdog o viceversa)
    reverse_steam = (spread_open * spread_close) < 0

    # Direzione steam
    if spread_close < spread_open:
        direction = "favorito"  # Spread scende = favorito si rafforza
    elif spread_close > spread_open:
        direction = "underdog"  # Spread sale = underdog si rafforza
    else:
        direction = "neutro"

    return {
        "is_steam": is_steam,
        "magnitude": movement,
        "reverse_steam": reverse_steam,
        "direction": direction
    }


def calculate_market_correlation(spread_open: float, spread_close: float,
                                 total_open: float, total_close: float) -> Dict:
    """
    Calcola correlazione tra movimenti spread e total.

    Correlation:
    - Positiva (+1): si muovono insieme = mercato coerente
    - Negativa (-1): direzioni opposte = segnali contrastanti
    - Zero: indipendenti

    Args:
        spread_open: Spread apertura
        spread_close: Spread chiusura
        total_open: Total apertura
        total_close: Total chiusura

    Returns:
        Dict con correlation indicators
    """
    spread_movement = spread_close - spread_open
    total_movement = total_close - total_open

    # Normalizza movimenti
    spread_norm = spread_movement / max(abs(spread_movement), 0.01)
    total_norm = total_movement / max(abs(total_movement), 0.01)

    # Correlazione semplice (-1 a +1)
    correlation = spread_norm * total_norm

    # Interpreta
    if correlation > 0.5:
        interpretation = "Alta correlazione positiva - Mercato COERENTE"
        coherent = True
    elif correlation < -0.5:
        interpretation = "Correlazione negativa - Segnali CONTRASTANTI"
        coherent = False
    else:
        interpretation = "Mercati INDIPENDENTI"
        coherent = True  # Non è incoerente, sono solo indipendenti

    return {
        "score": correlation,
        "interpretation": interpretation,
        "coherent": coherent
    }


def analyze_key_numbers(spread_value: float, total_value: float) -> Dict:
    """
    Identifica se spread/total sono su 'key numbers' statisticamente cruciali.

    Key numbers per Asian Handicap:
    - Spread: -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3
    - Total: 1.5, 2.0, 2.5, 2.75, 3.0, 3.5

    Args:
        spread_value: Valore spread attuale
        total_value: Valore total attuale

    Returns:
        Dict con key numbers indicators
    """
    # Key numbers per soccer/asian handicap
    spread_key_numbers = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    total_key_numbers = [1.5, 2.0, 2.5, 2.75, 3.0, 3.5]

    # Trova key number più vicino
    closest_spread_key = min(spread_key_numbers, key=lambda x: abs(x - spread_value))
    closest_total_key = min(total_key_numbers, key=lambda x: abs(x - total_value))

    # Distanza da key number
    spread_distance = abs(spread_value - closest_spread_key)
    total_distance = abs(total_value - closest_total_key)

    # Su key number se distanza < 0.1
    on_key_spread = spread_distance < 0.1
    on_key_total = total_distance < 0.1

    # Confidence boost
    confidence_boost = 0.0
    if on_key_spread or on_key_total:
        confidence_boost = 0.10

    return {
        "on_key_spread": on_key_spread,
        "on_key_total": on_key_total,
        "spread_key": closest_spread_key if on_key_spread else None,
        "total_key": closest_total_key if on_key_total else None,
        "spread_distance": spread_distance,
        "total_distance": total_distance,
        "confidence_boost": confidence_boost
    }


def calculate_market_efficiency(spread_open: float, spread_close: float,
                                total_open: float, total_close: float) -> Dict:
    """
    Calcola quanto è efficiente il mercato.

    Efficienza:
    - Alta (90-100): piccoli movimenti = prezzi accurati
    - Normale (70-89): movimenti moderati
    - Bassa (<70): grandi movimenti = possibili value bets

    Args:
        spread_open: Spread apertura
        spread_close: Spread chiusura
        total_open: Total apertura
        total_close: Total chiusura

    Returns:
        Dict con efficiency indicators
    """
    # Calcola % di cambiamento
    spread_change_pct = abs((spread_close - spread_open) / max(abs(spread_open), 0.1))
    total_change_pct = abs((total_close - total_open) / total_open)

    # Media movimenti
    avg_movement = (spread_change_pct + total_change_pct) / 2

    # Score 0-100 (100 = massima efficienza = pochi movimenti)
    efficiency = max(0, min(100, 100 - (avg_movement * 200)))

    # Interpreta
    if efficiency >= 90:
        status = "Efficient"
        value_opp = False
    elif efficiency >= 70:
        status = "Normal"
        value_opp = False
    else:
        status = "Inefficient"
        value_opp = True

    return {
        "score": efficiency,
        "status": status,
        "value_opportunity": value_opp
    }


# ============================================================================
# OPZIONE C: ADVANCED PROBABILITY CALCULATION FUNCTIONS
# ============================================================================

def spread_to_implied_probability(spread: float) -> Dict:
    """
    Converte Asian Handicap spread in probabilità 1X2 implicite.

    Formula empirica basata su analisi dati reali:
    - Spread -1.5 ≈ 65% home win, 25% draw, 10% away win
    - Spread 0.0 ≈ 33% home, 33% draw, 33% away

    Args:
        spread: Spread value (negativo = casa favorita)

    Returns:
        Dict con probabilità implicite {home, draw, away}
    """
    if spread < 0:
        # Casa favorita
        home_base = 0.50  # 50% base per spread 0
        home_boost = abs(spread) * 0.16  # 16% per punto
        home_prob = min(0.85, home_base + home_boost)

        # Draw diminuisce con spread
        draw_prob = max(0.10, 0.33 - abs(spread) * 0.08)

        # Away è il resto
        away_prob = 1.0 - home_prob - draw_prob

    elif spread > 0:
        # Trasferta favorita (speculare)
        away_base = 0.50
        away_boost = abs(spread) * 0.16
        away_prob = min(0.85, away_base + away_boost)

        draw_prob = max(0.10, 0.33 - abs(spread) * 0.08)
        home_prob = 1.0 - away_prob - draw_prob

    else:
        # Spread 0 = equilibrio perfetto
        home_prob = draw_prob = away_prob = 0.333

    # Normalizza
    total = home_prob + draw_prob + away_prob
    return {
        'home': home_prob / total,
        'draw': draw_prob / total,
        'away': away_prob / total
    }


def calculate_dynamic_home_advantage(spread_open: float, spread_close: float) -> float:
    """
    Home advantage dinamico basato su movimenti di mercato.

    LOGICA:
    - Spread si muove verso casa = HA aumenta (mercato crede di più in casa)
    - Spread si muove verso trasferta = HA diminuisce

    Args:
        spread_open: Spread apertura
        spread_close: Spread chiusura

    Returns:
        Home advantage factor (1.05 - 1.25)
    """
    # Base home advantage
    base_ha = 1.15  # +15% standard

    # Calcola movimento spread (negativo = verso casa)
    movement = spread_close - spread_open

    # Aggiustamento: ogni 0.25 punti di movimento = ±2% HA
    ha_adjustment = -movement * 0.08  # -0.25 → +2%, +0.25 → -2%

    # Clamp tra 1.05 e 1.25 (min +5%, max +25%)
    dynamic_ha = max(1.05, min(1.25, base_ha + ha_adjustment))

    return dynamic_ha


def bayesian_probability_update(
    prior_probs: Dict,
    market_signal: float,
    signal_confidence: float
) -> Dict:
    """
    Aggiornamento Bayesiano: Prior (xG) + Evidence (movimenti).

    FORMULA BAYESIANA:
    P(H|E) = P(E|H) * P(H) / P(E)

    Dove:
    - H = hypothesis (es. "casa vince")
    - E = evidence (movimento di mercato)
    - P(H) = prior (da xG)
    - P(E|H) = likelihood (quanto è probabile questo movimento se casa vince)
    - P(H|E) = posterior (probabilità aggiornata)

    Args:
        prior_probs: Probabilità prior da xG {home, draw, away}
        market_signal: Segnale mercato (-1 a +1, negativo = pro-casa)
        signal_confidence: Quanto fidarsi del segnale (0-1)

    Returns:
        Probabilità aggiornate {home, draw, away}
    """
    # Calcola likelihood ratios basati sul segnale
    if market_signal < -0.2:  # Segnale pro-casa
        strength = abs(market_signal)
        home_likelihood = 1.0 + signal_confidence * strength
        away_likelihood = 1.0 - signal_confidence * strength * 0.5
        draw_likelihood = 1.0 - signal_confidence * strength * 0.3

    elif market_signal > 0.2:  # Segnale pro-trasferta
        strength = abs(market_signal)
        away_likelihood = 1.0 + signal_confidence * strength
        home_likelihood = 1.0 - signal_confidence * strength * 0.5
        draw_likelihood = 1.0 - signal_confidence * strength * 0.3

    else:  # Segnale neutro
        home_likelihood = draw_likelihood = away_likelihood = 1.0

    # Posterior = Prior * Likelihood
    home_posterior = prior_probs['home'] * home_likelihood
    draw_posterior = prior_probs['draw'] * draw_likelihood
    away_posterior = prior_probs['away'] * away_likelihood

    # Normalizza
    total = home_posterior + draw_posterior + away_posterior

    return {
        'home': home_posterior / total,
        'draw': draw_posterior / total,
        'away': away_posterior / total
    }


def monte_carlo_btts_simulation(
    home_xg: float,
    away_xg: float,
    n_simulations: int = 5000
) -> Dict:
    """
    Monte Carlo Simulation per BTTS usando distribuzione Poisson.

    Simula N partite e calcola % di partite con Both Teams To Score.

    Args:
        home_xg: Expected goals casa
        away_xg: Expected goals trasferta
        n_simulations: Numero simulazioni (default 5000)

    Returns:
        Dict con statistiche BTTS
    """
    import random

    btts_count = 0
    home_scores = []
    away_scores = []

    for _ in range(n_simulations):
        # Simula gol usando distribuzione Poisson
        # P(k goals) = (λ^k * e^-λ) / k!
        # Usiamo metodo inverso: genera random e trova k corrispondente

        # Simula gol casa
        home_goals = 0
        prob_sum = math.exp(-home_xg)  # P(0)
        rand = random.random()

        while rand > prob_sum and home_goals < 10:
            home_goals += 1
            prob_sum += poisson_probability(home_goals, home_xg)

        # Simula gol trasferta
        away_goals = 0
        prob_sum = math.exp(-away_xg)  # P(0)
        rand = random.random()

        while rand > prob_sum and away_goals < 10:
            away_goals += 1
            prob_sum += poisson_probability(away_goals, away_xg)

        home_scores.append(home_goals)
        away_scores.append(away_goals)

        # BTTS se entrambe > 0
        if home_goals > 0 and away_goals > 0:
            btts_count += 1

    btts_prob = btts_count / n_simulations

    return {
        'btts_prob': btts_prob,
        'nobtts_prob': 1 - btts_prob,
        'avg_home_goals': sum(home_scores) / n_simulations,
        'avg_away_goals': sum(away_scores) / n_simulations,
        'simulations': n_simulations
    }


def calculate_smart_btts(
    home_xg: float,
    away_xg: float,
    total_open: float,
    total_close: float,
    spread_close: float,
    use_monte_carlo: bool = True
) -> Dict:
    """
    BTTS intelligente con logica di mercato + Monte Carlo.

    LOGICA:
    1. Total alto + Spread basso = partita aperta = GG più probabile
    2. Total basso + Spread alto = dominio difensivo = NOGG
    3. Total sale = mercato si aspetta più gol = boost GG

    Args:
        home_xg: Expected goals casa
        away_xg: Expected goals trasferta
        total_open: Total apertura
        total_close: Total chiusura
        spread_close: Spread chiusura
        use_monte_carlo: Se True usa Monte Carlo (default)

    Returns:
        Dict con BTTS probability e fattori
    """
    # 1. BTTS base (Monte Carlo o formula standard)
    if use_monte_carlo:
        mc_result = monte_carlo_btts_simulation(home_xg, away_xg, n_simulations=5000)
        base_btts = mc_result['btts_prob']
    else:
        # Formula standard
        base_btts = (1 - math.exp(-home_xg)) * (1 - math.exp(-away_xg))

    # 2. Fattore "Partita Aperta" (spread basso + total alto = aperta)
    abs_spread = abs(spread_close)
    openness_factor = total_close / max(abs_spread, 0.5)
    # Es: Total 3.0 / Spread 0.5 = 6.0 (molto aperta)
    # Es: Total 2.0 / Spread 2.0 = 1.0 (chiusa)

    # Normalizza 0-1 (6.0 → 1.0, 1.0 → 0.2)
    openness_normalized = min(1.0, max(0.2, openness_factor / 6.0))

    # 3. Fattore "Total Movement" (se total sale = più gol attesi)
    total_movement = total_close - total_open
    total_boost = total_movement * 0.1  # Ogni +0.25 → +2.5% boost BTTS

    # 4. Fattore "Balance" (partita equilibrata = più BTTS)
    balance_factor = max(0.5, 1.0 - (abs_spread / 3.0))  # Spread 0 = 1.0, Spread 3 = 0.5

    # 5. BTTS aggiustato - Usa boost additivi invece di moltiplicativi
    # Base = base_btts
    # +total_boost per movimento total
    # +openness_boost per partita aperta
    # +balance_boost per equilibrio
    openness_boost = (openness_normalized - 0.5) * 0.15  # Max +7.5% se molto aperta
    balance_boost = (balance_factor - 0.75) * 0.10  # Max +2.5% se equilibrata

    btts_adjusted = base_btts * (1 + total_boost + openness_boost + balance_boost)
    btts_adjusted = min(0.95, max(0.05, btts_adjusted))  # Clamp 5-95%

    return {
        'btts_prob': btts_adjusted,
        'nobtts_prob': 1 - btts_adjusted,
        'base_btts': base_btts,
        'openness_score': openness_normalized,
        'balance_score': balance_factor,
        'total_boost': total_boost,
        'method': 'monte_carlo' if use_monte_carlo else 'formula'
    }


def calculate_market_adjusted_probabilities(
    home_xg: float,
    away_xg: float,
    spread_open: float,
    spread_close: float,
    sharp_money_detected: bool,
    steam_move_detected: bool
) -> Dict:
    """
    1X2 aggiustato usando movimenti di mercato come segnale.

    WEIGHTED ENSEMBLE:
    - Probabilità base da xG (Dixon-Coles)
    - Probabilità implied da spread
    - Peso dinamico basato su sharp money signals

    Args:
        home_xg: Expected goals casa
        away_xg: Expected goals trasferta
        spread_open: Spread apertura
        spread_close: Spread chiusura
        sharp_money_detected: Se rilevato sharp money
        steam_move_detected: Se rilevato steam move

    Returns:
        Dict con probabilità aggiustate e metadati
    """
    # 1. Calcola probabilità BASE da xG (Dixon-Coles)
    base_home = 0.0
    base_draw = 0.0
    base_away = 0.0

    for h in range(10):
        for a in range(10):
            prob = dixon_coles_probability(h, a, home_xg, away_xg)
            if h > a:
                base_home += prob
            elif h == a:
                base_draw += prob
            else:
                base_away += prob

    base_probs = {'home': base_home, 'draw': base_draw, 'away': base_away}

    # 2. Calcola IMPLIED probabilities dallo SPREAD
    spread_implied = spread_to_implied_probability(spread_close)

    # 3. Calcola MOVEMENT STRENGTH e market signal
    movement = spread_close - spread_open
    movement_strength = abs(movement) / max(abs(spread_open), 0.1)
    movement_strength = min(movement_strength, 0.5)  # Cap al 50%

    # Market signal: -1 (pro-casa) a +1 (pro-trasferta)
    market_signal = movement / max(abs(spread_open), 0.5)
    market_signal = max(-1.0, min(1.0, market_signal))

    # 4. Calcola CONFIDENCE nel segnale
    signal_confidence = 0.3  # Base 30%

    if sharp_money_detected:
        signal_confidence += 0.3  # +30% se sharp
    if steam_move_detected:
        signal_confidence += 0.2  # +20% se steam

    signal_confidence = min(0.8, signal_confidence)  # Max 80%

    # 5. BAYESIAN UPDATE
    bayesian_probs = bayesian_probability_update(
        base_probs,
        market_signal,
        signal_confidence
    )

    # 6. WEIGHTED ENSEMBLE: xG + Spread implied
    # Peso movimento dipende da sharp signals
    alpha = signal_confidence * 0.5  # Max 40% peso a spread

    final_home = (1 - alpha) * bayesian_probs['home'] + alpha * spread_implied['home']
    final_draw = (1 - alpha) * bayesian_probs['draw'] + alpha * spread_implied['draw']
    final_away = (1 - alpha) * bayesian_probs['away'] + alpha * spread_implied['away']

    # Normalizza
    total = final_home + final_draw + final_away

    return {
        'home_win': final_home / total,
        'draw': final_draw / total,
        'away_win': final_away / total,
        'method': 'bayesian_ensemble',
        'base_probs': base_probs,
        'spread_implied': spread_implied,
        'bayesian_probs': bayesian_probs,
        'ensemble_weight': {
            'xg': 1 - alpha,
            'spread': alpha
        },
        'market_signal': market_signal,
        'signal_confidence': signal_confidence
    }


def calculate_market_confidence_score(
    spread_analysis: 'MovementAnalysis',
    total_analysis: 'MovementAnalysis',
    market_intel: 'MarketIntelligence',
    prediction_variance: float
) -> float:
    """
    Calcola confidence score (0-100) per le predizioni.

    FATTORI:
    1. Market coherence (correlation tra spread/total)
    2. Sharp money signals (professionalità movimento)
    3. Prediction variance (quanto sono diverse le stime)
    4. Market efficiency (quanto il mercato è stabile)

    Args:
        spread_analysis: Analisi movimento spread
        total_analysis: Analisi movimento total
        market_intel: Market intelligence data
        prediction_variance: Varianza tra xG e spread implied

    Returns:
        Confidence score 0-100
    """
    score = 50.0  # Base 50

    # 1. Market Coherence (+20 pts se coerente)
    if market_intel.market_coherent:
        score += 20
    elif market_intel.correlation_score < -0.5:
        score -= 10  # Penalità segnali contrastanti

    # 2. Sharp Money (+15 pts se sharp)
    if market_intel.sharp_money_detected:
        score += 15
        if market_intel.contrarian_signal:
            score += 5  # Bonus contrarian

    # 3. Steam Move (+10 pts se steam)
    if market_intel.steam_move_detected:
        score += 10

    # 4. Key Numbers (+5 pts)
    if market_intel.on_key_spread or market_intel.on_key_total:
        score += 5

    # 5. Market Efficiency (+10 pts se efficiente)
    if market_intel.efficiency_score >= 85:
        score += 10
    elif market_intel.efficiency_score < 60:
        score -= 10

    # 6. Prediction Variance (penalità se troppo diverse)
    # Variance bassa = stime concordano = +confidence
    if prediction_variance < 0.1:
        score += 10
    elif prediction_variance > 0.3:
        score -= 15

    # 7. Movement Intensity (moderato è meglio)
    avg_intensity = (spread_analysis.movement_steps + total_analysis.movement_steps) / 2
    if 0.25 <= avg_intensity <= 0.75:
        score += 5  # Movimento moderato = buono
    elif avg_intensity > 1.5:
        score -= 10  # Movimento eccessivo = caos

    # Clamp 0-100
    return max(0, min(100, score))


def calculate_expected_goals(spread: float, total: float, use_advanced_formulas: bool = True) -> ExpectedGoals:
    """
    Calcola Expected Goals (xG) da spread e total usando Asian Handicap.

    VERSIONE AVANZATA con:
    - Home Advantage adjustment (+15% casa, -12% trasferta)
    - Dixon-Coles Bivariate Poisson (correlazione risultati bassi)
    - Massima precisione per betting professionale

    Formula base:
    - spread = lambda_casa - lambda_trasferta
    - total = lambda_casa + lambda_trasferta

    Quindi:
    - lambda_casa = (total - spread) / 2
    - lambda_trasferta = (total + spread) / 2

    Args:
        spread: Spread di chiusura (negativo = casa favorita)
        total: Total di chiusura
        use_advanced_formulas: Se True usa Home Advantage + Dixon-Coles (default True)

    Returns:
        ExpectedGoals con xG e probabilità calcolate
    """
    # Calcola xG (lambda) base per casa e trasferta
    home_xg_base = (total - spread) / 2
    away_xg_base = (total + spread) / 2

    # Assicura che xG siano positivi
    home_xg_base = max(0.1, home_xg_base)
    away_xg_base = max(0.1, away_xg_base)

    # FORMULA AVANZATA 1: Home Advantage Adjustment
    if use_advanced_formulas:
        home_xg, away_xg = adjust_for_home_advantage(home_xg_base, away_xg_base)
    else:
        home_xg, away_xg = home_xg_base, away_xg_base

    # Calcola probabilità clean sheet: P(0 gol) = e^(-lambda)
    # Usa Dixon-Coles per maggiore precisione
    if use_advanced_formulas:
        home_clean_sheet_prob = dixon_coles_probability(0, 0, home_xg, away_xg) / dixon_coles_probability(0, 0, home_xg, 0)
        away_clean_sheet_prob = dixon_coles_probability(0, 0, home_xg, away_xg) / dixon_coles_probability(0, 0, 0, away_xg)
        # Fallback a formula standard se Dixon-Coles da valori strani
        if home_clean_sheet_prob > 1 or home_clean_sheet_prob < 0:
            home_clean_sheet_prob = math.exp(-away_xg)
        if away_clean_sheet_prob > 1 or away_clean_sheet_prob < 0:
            away_clean_sheet_prob = math.exp(-home_xg)
    else:
        home_clean_sheet_prob = math.exp(-away_xg)
        away_clean_sheet_prob = math.exp(-home_xg)

    # FORMULA AVANZATA 2: BTTS con Dixon-Coles
    if use_advanced_formulas:
        # Calcola P(BTTS) accurata con Dixon-Coles
        btts_prob = 0.0
        for h in range(1, 8):
            for a in range(1, 8):
                btts_prob += dixon_coles_probability(h, a, home_xg, away_xg)
    else:
        btts_prob = (1 - home_clean_sheet_prob) * (1 - away_clean_sheet_prob)

    # FORMULA AVANZATA 3: 1X2 con Dixon-Coles
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0

    for home_goals in range(10):
        for away_goals in range(10):
            if use_advanced_formulas:
                prob = dixon_coles_probability(home_goals, away_goals, home_xg, away_xg)
            else:
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


def calculate_expected_goals_advanced(
    spread_open: float,
    spread_close: float,
    total_open: float,
    total_close: float,
    market_intel: MarketIntelligence,
    spread_analysis: MovementAnalysis,
    total_analysis: MovementAnalysis
) -> ExpectedGoals:
    """
    OPZIONE C: Expected Goals con TUTTE le funzioni avanzate.

    Usa:
    - Bayesian Market Update
    - Monte Carlo BTTS Simulation
    - Market-Adjusted Probabilities
    - Dynamic Home Advantage
    - Confidence Score

    Args:
        spread_open: Spread apertura
        spread_close: Spread chiusura
        total_open: Total apertura
        total_close: Total chiusura
        market_intel: Market intelligence data
        spread_analysis: Analisi movimento spread
        total_analysis: Analisi movimento total

    Returns:
        ExpectedGoals con tutti i campi avanzati popolati
    """
    # 1. Calcola xG base
    home_xg_base = (total_close - spread_close) / 2
    away_xg_base = (total_close + spread_close) / 2

    home_xg_base = max(0.1, home_xg_base)
    away_xg_base = max(0.1, away_xg_base)

    # 2. DYNAMIC HOME ADVANTAGE (usa movimenti)
    dynamic_ha = calculate_dynamic_home_advantage(spread_open, spread_close)
    home_xg = home_xg_base * dynamic_ha
    away_xg = away_xg_base * (2.0 - dynamic_ha)  # Inverso per trasferta

    # 3. Calcola probabilità BASE con Dixon-Coles
    base_home = 0.0
    base_draw = 0.0
    base_away = 0.0

    for h in range(10):
        for a in range(10):
            prob = dixon_coles_probability(h, a, home_xg, away_xg)
            if h > a:
                base_home += prob
            elif h == a:
                base_draw += prob
            else:
                base_away += prob

    # 4. MARKET-ADJUSTED 1X2 (usa movimenti + sharp signals)
    market_adj_1x2 = calculate_market_adjusted_probabilities(
        home_xg,
        away_xg,
        spread_open,
        spread_close,
        market_intel.sharp_money_detected,
        market_intel.steam_move_detected
    )

    # 5. SMART BTTS con Monte Carlo
    smart_btts = calculate_smart_btts(
        home_xg,
        away_xg,
        total_open,
        total_close,
        spread_close,
        use_monte_carlo=True
    )

    # 6. Calcola Clean Sheet probabilities
    home_clean_sheet_prob = math.exp(-away_xg)
    away_clean_sheet_prob = math.exp(-home_xg)

    # 7. Calcola PREDICTION VARIANCE (quanto sono diverse le stime)
    variance = abs(market_adj_1x2['home_win'] - base_home)
    variance += abs(market_adj_1x2['draw'] - base_draw)
    variance += abs(market_adj_1x2['away_win'] - base_away)
    variance /= 3  # Media

    # 8. MARKET CONFIDENCE SCORE
    confidence = calculate_market_confidence_score(
        spread_analysis,
        total_analysis,
        market_intel,
        variance
    )

    # 9. Return ExpectedGoals con TUTTI i campi
    return ExpectedGoals(
        # Standard fields
        home_xg=home_xg,
        away_xg=away_xg,
        home_clean_sheet_prob=home_clean_sheet_prob,
        away_clean_sheet_prob=away_clean_sheet_prob,
        btts_prob=smart_btts['btts_prob'],
        home_win_prob=market_adj_1x2['home_win'],
        draw_prob=market_adj_1x2['draw'],
        away_win_prob=market_adj_1x2['away_win'],

        # Advanced fields (Opzione C)
        market_adjusted_1x2=market_adj_1x2,
        bayesian_btts=smart_btts,
        confidence_score=confidence,
        prediction_method="bayesian_ensemble_monte_carlo",
        ensemble_weights=market_adj_1x2['ensemble_weight']
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


def get_most_likely_score(home_xg: float, away_xg: float, top_n: int = 5,
                          use_dixon_coles: bool = True) -> List[Tuple[str, float]]:
    """
    Calcola i punteggi più probabili usando Dixon-Coles Bivariate Poisson.

    VERSIONE AVANZATA: Usa Dixon-Coles per massima precisione su risultati bassi
    (0-0, 1-0, 0-1, 1-1 sono più accurati con correlazione)

    Args:
        home_xg: Expected goals casa
        away_xg: Expected goals trasferta
        top_n: Numero di punteggi da restituire
        use_dixon_coles: Se True usa Dixon-Coles (default), altrimenti Poisson standard

    Returns:
        Lista di tuple (punteggio, probabilità) ordinate per probabilità
    """
    scores = []

    # Calcola probabilità per tutti i punteggi realistici (0-7 gol per squadra)
    for home_goals in range(8):
        for away_goals in range(8):
            if use_dixon_coles:
                prob = dixon_coles_probability(home_goals, away_goals, home_xg, away_xg)
            else:
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


# ============================================================================
# FORMULE AVANZATE PER MASSIMA PRECISIONE
# ============================================================================

# Costanti statistiche
HOME_ADVANTAGE_BOOST = 1.15  # +15% xG casa (dato statistico)
AWAY_DISADVANTAGE = 0.88     # -12% xG trasferta
DIXON_COLES_RHO = -0.13      # Parametro correlazione Dixon-Coles


def dixon_coles_probability(h: int, a: int, lambda_h: float, lambda_a: float,
                            rho: float = DIXON_COLES_RHO) -> float:
    """
    Dixon-Coles Bivariate Poisson - Gold Standard per betting professionale.

    Corregge l'assunzione di indipendenza tra gol casa/trasferta.
    Risultati bassi (0-0, 1-0, 0-1, 1-1) sono correlati.

    Args:
        h: Gol casa
        a: Gol trasferta
        lambda_h: xG casa
        lambda_a: xG trasferta
        rho: Parametro correlazione (tipicamente -0.10 a -0.15)

    Returns:
        Probabilità corretta per correlazione
    """
    # Probabilità Poisson base
    base_prob = poisson_probability(h, lambda_h) * poisson_probability(a, lambda_a)

    # Fattore di correlazione tau (solo per risultati bassi)
    # Formula Dixon-Coles: con rho negativo, 0-0 e 1-1 diminuiscono, 1-0 e 0-1 aumentano
    if h == 0 and a == 0:
        tau = 1 + lambda_h * lambda_a * rho
    elif h == 1 and a == 0:
        tau = 1 - lambda_a * rho
    elif h == 0 and a == 1:
        tau = 1 - lambda_h * rho
    elif h == 1 and a == 1:
        tau = 1 + rho
    else:
        tau = 1  # Nessuna correlazione per risultati alti

    return base_prob * tau


def adjust_for_home_advantage(home_xg: float, away_xg: float) -> Tuple[float, float]:
    """
    Aggiusta xG per vantaggio casa statisticamente provato.

    Studi statistici mostrano:
    - Casa segna ~1.46x più gol
    - Trasferta segna ~0.85x meno gol

    IMPORTANTE: Preserva il total originale (somma xG) mentre redistribuisce i gol.

    Args:
        home_xg: xG casa (da spread e total)
        away_xg: xG trasferta (da spread e total)

    Returns:
        (home_xg_adjusted, away_xg_adjusted) - somma preservata
    """
    original_total = home_xg + away_xg

    # Applica boost/penalità
    adjusted_home = home_xg * HOME_ADVANTAGE_BOOST
    adjusted_away = away_xg * AWAY_DISADVANTAGE

    # Renormalizza per preservare il total
    new_total = adjusted_home + adjusted_away
    adjusted_home = (adjusted_home / new_total) * original_total
    adjusted_away = (adjusted_away / new_total) * original_total

    return adjusted_home, adjusted_away


def remove_vig(odds_home: float, odds_draw: float, odds_away: float) -> Tuple[float, float, float]:
    """
    Rimuove overround (margine bookmaker) per ottenere true probabilities.

    Quote bookmaker includono margine di profitto (overround).
    Esempio: 1.85 / 3.40 / 4.50 = 105.7% (5.7% overround)

    Args:
        odds_home: Quota casa
        odds_draw: Quota pareggio
        odds_away: Quota trasferta

    Returns:
        (true_prob_home, true_prob_draw, true_prob_away) normalizzate a 100%
    """
    # Probabilità implicite dalle quote
    prob_home = 1 / odds_home
    prob_draw = 1 / odds_draw
    prob_away = 1 / odds_away

    # Overround totale
    overround = prob_home + prob_draw + prob_away

    # Normalizza a 100%
    true_home = prob_home / overround
    true_draw = prob_draw / overround
    true_away = prob_away / overround

    return true_home, true_draw, true_away


def calculate_edge(our_probability: float, bookmaker_odds: float) -> float:
    """
    Calcola edge (vantaggio) reale su una scommessa.

    Edge = Nostra probabilità - Probabilità implicita dalle quote
    Edge positivo = Value bet!

    Args:
        our_probability: Nostra stima probabilità (0-1)
        bookmaker_odds: Quota bookmaker

    Returns:
        Edge in percentuale (es. 0.06 = +6% edge)
    """
    implied_prob = 1 / bookmaker_odds
    edge = our_probability - implied_prob
    return edge


# ============================================================================
# OPZIONE B: ALTERNATIVE MARKETS REFINEMENT
# ============================================================================

def calculate_dynamic_ht_percentage(spread: float, total: float) -> float:
    """
    Calcola la percentuale dinamica di gol che avvengono nel primo tempo.

    MOTIVAZIONE:
    - Il fisso 45% è troppo grezzo
    - Partite sbilanciate (spread alto) hanno più gol nel 2T (inseguimento)
    - Partite ad alto punteggio (total alto) hanno più gol nel 1T (ritmi alti)

    Args:
        spread: Asian Handicap spread (es. -1.5)
        total: Over/Under total (es. 2.5)

    Returns:
        Percentuale HT dinamica (range: 35% - 55%)
    """
    abs_spread = abs(spread)

    # Base: 45% (media storica)
    ht_percentage = 0.45

    # Aggiustamento per spread (match sbilanciati)
    # Spread alto → underdog insegue nel 2T → meno gol 1T
    # IMPORTANTE: Ordine dal più alto al più basso per elif
    if abs_spread >= 2.5:
        ht_percentage -= 0.10  # 35%
    elif abs_spread >= 2.0:
        ht_percentage -= 0.08  # 37%
    elif abs_spread >= 1.5:
        ht_percentage -= 0.05  # 40%
    elif abs_spread <= 0.5:
        # Match equilibrato → distribuzione uniforme
        ht_percentage += 0.02  # 47%

    # Aggiustamento per total (ritmo della partita)
    # Total alto → ritmi alti → più gol 1T
    if total >= 3.5:
        ht_percentage += 0.08  # Boost verso 50-53%
    elif total >= 3.0:
        ht_percentage += 0.05  # Boost verso 47-50%
    elif total <= 2.0:
        # Partita tattica → meno gol 1T
        ht_percentage -= 0.05  # Penalità verso 40%

    # Clamp al range 35%-55%
    ht_percentage = max(0.35, min(0.55, ht_percentage))

    return ht_percentage


def calculate_ht_ft_with_correlation(home_xg: float, away_xg: float, spread: float) -> Dict:
    """
    Calcola probabilità HT/FT con correlazione e momentum.

    PROBLEMA RISOLTO:
    - Assumere indipendenza HT/FT è irrealistico
    - Chi vince HT ha momentum nel 2T
    - Chi perde HT spesso reagisce (contropiede)

    CORREZIONI:
    1. Se vinci HT → +20% probabilità vinci FT (momentum)
    2. Se pareggi HT → possibilità aperta
    3. Se perdi HT → underdog boost (disperazione)

    Args:
        home_xg: Expected goals casa (full time)
        away_xg: Expected goals trasferta (full time)
        spread: Asian Handicap spread

    Returns:
        Dict con probabilità HT/FT corrette per correlazione
    """
    # Calcola probabilità base HT e FT
    ht_probs = calculate_halftime_probabilities(home_xg, away_xg)

    # FT probabilities
    ft_home = 0.0
    ft_draw = 0.0
    ft_away = 0.0

    for home_goals in range(6):
        for away_goals in range(6):
            prob = poisson_probability(home_goals, home_xg) * poisson_probability(away_goals, away_xg)
            if home_goals > away_goals:
                ft_home += prob
            elif home_goals == away_goals:
                ft_draw += prob
            else:
                ft_away += prob

    # ===== CORRELAZIONE HT/FT CON MOMENTUM =====

    # Fattori di correlazione
    momentum_boost = 0.20  # Chi vince HT ha +20% prob vincere FT
    underdog_desperation = 0.15  # Chi perde HT reagisce con +15%

    ht_ft_probs = {}

    # --- 1/1: Casa vince HT e FT ---
    # Momentum boost: se casa vince HT, molto più probabile vinca FT
    ht_ft_probs["1/1"] = ht_probs["home_win_ht"] * ft_home * (1 + momentum_boost)

    # --- 1/X: Casa vince HT ma pareggia FT ---
    # Penalità: difficile perdere vantaggio HT
    ht_ft_probs["1/X"] = ht_probs["home_win_ht"] * ft_draw * 0.80

    # --- 1/2: Casa vince HT ma perde FT ---
    # Raro ma possibile: trasferta rimonta (underdog desperation)
    ht_ft_probs["1/2"] = ht_probs["home_win_ht"] * ft_away * (0.50 + underdog_desperation)

    # --- X/1: Pareggio HT, casa vince FT ---
    # Normale: nessuna correlazione forte
    ht_ft_probs["X/1"] = ht_probs["draw_ht"] * ft_home

    # --- X/X: Pareggio HT e FT ---
    # Boost: se pareggi HT, più probabile pareggio FT (squadre equivalenti)
    ht_ft_probs["X/X"] = ht_probs["draw_ht"] * ft_draw * 1.30

    # --- X/2: Pareggio HT, trasferta vince FT ---
    # Normale
    ht_ft_probs["X/2"] = ht_probs["draw_ht"] * ft_away

    # --- 2/1: Trasferta vince HT ma casa vince FT ---
    # Rimonta casa (underdog desperation + fattore campo)
    ht_ft_probs["2/1"] = ht_probs["away_win_ht"] * ft_home * (0.50 + underdog_desperation + 0.10)

    # --- 2/X: Trasferta vince HT ma pareggia FT ---
    # Penalità: difficile perdere vantaggio HT
    ht_ft_probs["2/X"] = ht_probs["away_win_ht"] * ft_draw * 0.80

    # --- 2/2: Trasferta vince HT e FT ---
    # Momentum boost
    ht_ft_probs["2/2"] = ht_probs["away_win_ht"] * ft_away * (1 + momentum_boost)

    # Normalizza a 100%
    total_prob = sum(ht_ft_probs.values())
    for key in ht_ft_probs:
        ht_ft_probs[key] /= total_prob

    return ht_ft_probs


def apply_sticky_scores_adjustment(goals_dist: Dict[int, float],
                                   home_xg: float, away_xg: float) -> Dict[int, float]:
    """
    Applica aggiustamenti per "sticky scores" (risultati comuni).

    PROBLEMA RISOLTO:
    - Poisson puro sottostima risultati comuni (1-1, 2-1, 1-0)
    - Sopravvaluta risultati rari (5-4, 6-3)
    - Statistiche storiche mostrano clustering su certi punteggi

    STICKY SCORES (boost):
    - 1-0, 0-1: +15% (risultato più comune nel calcio)
    - 1-1: +20% (pareggio più frequente)
    - 2-1, 1-2: +15% (secondo risultato più comune)
    - 2-0, 0-2: +10%

    RARE SCORES (penalità):
    - 5+ gol: -20%
    - 7+ gol: -50% (estremamente raro)

    Args:
        goals_dist: Distribuzione Poisson {total_gol: prob}
        home_xg: Expected goals casa
        away_xg: Expected goals trasferta

    Returns:
        Distribuzione aggiustata per sticky scores
    """
    # Calcola distribuzione esatta per ogni score
    score_dist = {}

    for total_goals in range(10):
        for home_goals in range(total_goals + 1):
            away_goals = total_goals - home_goals
            score = f"{home_goals}-{away_goals}"

            prob = poisson_probability(home_goals, home_xg) * poisson_probability(away_goals, away_xg)
            score_dist[score] = prob

    # ===== APPLICA STICKY SCORES ADJUSTMENTS =====

    # Boost per risultati comuni
    sticky_boosts = {
        "1-0": 1.15,
        "0-1": 1.15,
        "1-1": 1.20,
        "2-1": 1.15,
        "1-2": 1.15,
        "2-0": 1.10,
        "0-2": 1.10,
        "2-2": 1.08,
        "0-0": 1.05,  # Leggermente più comune del previsto
    }

    for score, boost in sticky_boosts.items():
        if score in score_dist:
            score_dist[score] *= boost

    # Penalità per risultati rari (5+ gol totali)
    for score in score_dist:
        home_g, away_g = map(int, score.split("-"))
        total_g = home_g + away_g

        if total_g >= 7:
            score_dist[score] *= 0.50  # -50% per 7+ gol
        elif total_g >= 5:
            score_dist[score] *= 0.80  # -20% per 5-6 gol

    # Normalizza a 100%
    total_prob = sum(score_dist.values())
    for score in score_dist:
        score_dist[score] /= total_prob

    # Riconverti a distribuzione per total goals
    adjusted_goals_dist = {}
    for total_goals in range(10):
        adjusted_goals_dist[total_goals] = 0.0
        for home_goals in range(total_goals + 1):
            away_goals = total_goals - home_goals
            score = f"{home_goals}-{away_goals}"
            if score in score_dist:
                adjusted_goals_dist[total_goals] += score_dist[score]

    return adjusted_goals_dist


def calculate_time_weighted_xg_ht(home_xg: float, away_xg: float,
                                  total: float, spread: float) -> Dict:
    """
    Calcola xG HT con time-weighting (gol non uniformi nel tempo).

    PROBLEMA RISOLTO:
    - Gol non distribuiti uniformemente nei 90 minuti
    - Più gol tra 60-75 minuti (stanchezza)
    - Primi 15 minuti cauti (studio avversario)
    - Ultimi 15 minuti disperazione o gestione

    TIME DISTRIBUTION REALE:
    - 0-15 min: 15% dei gol totali
    - 15-30 min: 25%
    - 30-45 min: 20%
    - TOTALE HT: 60% → ma varia con spread/total

    Args:
        home_xg: Expected goals casa (FT)
        away_xg: Expected goals trasferta (FT)
        total: Over/Under total
        spread: Asian Handicap spread

    Returns:
        Dict con xG HT time-weighted e probabilità
    """
    # Usa percentuale dinamica HT
    ht_percentage = calculate_dynamic_ht_percentage(spread, total)

    # Distribuzione temporale all'interno del HT
    # Non uniforme: inizio cauto, crescendo verso fine 1T
    time_weights = {
        "0-15": 0.15,   # Primi 15 min: cauti
        "15-30": 0.45,  # Minuti centrali: picco
        "30-45": 0.40,  # Finale 1T: altro picco (prima dell'intervallo)
    }

    # Aggiusta in base allo spread (favorito parte forte o aspetta?)
    abs_spread = abs(spread)
    if abs_spread >= 1.5:
        # Favorito forte: parte aggressivo
        time_weights["0-15"] += 0.05
        time_weights["30-45"] -= 0.05

    # xG per periodo temporale
    home_xg_ht_total = home_xg * ht_percentage
    away_xg_ht_total = away_xg * ht_percentage

    home_xg_by_period = {
        period: home_xg_ht_total * weight
        for period, weight in time_weights.items()
    }
    away_xg_by_period = {
        period: away_xg_ht_total * weight
        for period, weight in time_weights.items()
    }

    # Calcola probabilità HT usando xG time-weighted
    home_xg_ht = home_xg_ht_total
    away_xg_ht = away_xg_ht_total

    # Probabilità 1X2 HT
    home_win_ht = 0.0
    draw_ht = 0.0
    away_win_ht = 0.0

    # Use range(6) instead of range(5) for better precision
    for home_goals in range(6):
        for away_goals in range(6):
            prob = poisson_probability(home_goals, home_xg_ht) * poisson_probability(away_goals, away_xg_ht)

            if home_goals > away_goals:
                home_win_ht += prob
            elif home_goals == away_goals:
                draw_ht += prob
            else:
                away_win_ht += prob

    # BTTS HT
    home_cs_ht = math.exp(-away_xg_ht)
    away_cs_ht = math.exp(-home_xg_ht)
    btts_ht = (1 - home_cs_ht) * (1 - away_cs_ht)

    # Over/Under HT
    over_05_ht = 1 - (math.exp(-home_xg_ht) * math.exp(-away_xg_ht))

    over_15_ht = 0.0
    for total_goals in range(2, 8):
        for home_goals in range(total_goals + 1):
            away_goals = total_goals - home_goals
            over_15_ht += poisson_probability(home_goals, home_xg_ht) * poisson_probability(away_goals, away_xg_ht)

    return {
        "home_xg_ht": home_xg_ht,
        "away_xg_ht": away_xg_ht,
        "total_xg_ht": home_xg_ht + away_xg_ht,
        "ht_percentage": ht_percentage,
        "time_weights": time_weights,
        "home_xg_by_period": home_xg_by_period,
        "away_xg_by_period": away_xg_by_period,
        "home_win_ht": home_win_ht,
        "draw_ht": draw_ht,
        "away_win_ht": away_win_ht,
        "btts_ht": btts_ht,
        "over_05_ht": over_05_ht,
        "over_15_ht": over_15_ht
    }


def regression_to_mean(observed_value: float, prior_mean: float,
                       confidence: float = 0.75) -> float:
    """
    James-Stein Shrinkage - Corregge over-reactions del mercato.

    Movimenti estremi tendono a ritornare verso la media.
    Esempio: Spread -1.0 → -2.5 (movimento -1.5) è probabilmente eccessivo.

    Args:
        observed_value: Valore osservato (es. spread closing)
        prior_mean: Media prior (es. spread opening)
        confidence: Peso al valore osservato (0-1), default 0.75 = 75%

    Returns:
        Valore aggiustato (meno estremo)
    """
    adjusted = confidence * observed_value + (1 - confidence) * prior_mean
    return adjusted


def btts_and_total_joint(home_xg: float, away_xg: float,
                         threshold: float = 2.5,
                         use_dixon_coles: bool = True) -> Dict[str, float]:
    """
    Probabilità congiunte per mercati correlati (BTTS + Over/Under).

    BTTS e Total sono correlati, non indipendenti!
    P(BTTS=Yes AND Over 2.5) ≠ P(BTTS=Yes) × P(Over 2.5)

    Args:
        home_xg: xG casa
        away_xg: xG trasferta
        threshold: Soglia Over/Under (default 2.5)
        use_dixon_coles: Usa Dixon-Coles invece di Poisson standard

    Returns:
        Dict con probabilità congiunte
    """
    prob_btts_and_over = 0.0
    prob_btts_and_under = 0.0
    prob_nobtts_and_over = 0.0
    prob_nobtts_and_under = 0.0

    for h in range(10):
        for a in range(10):
            # Usa Dixon-Coles o Poisson
            if use_dixon_coles:
                prob = dixon_coles_probability(h, a, home_xg, away_xg)
            else:
                prob = poisson_probability(h, home_xg) * poisson_probability(a, away_xg)

            total_goals = h + a
            btts = (h > 0 and a > 0)
            over = (total_goals > threshold)

            if btts and over:
                prob_btts_and_over += prob
            elif btts and not over:
                prob_btts_and_under += prob
            elif not btts and over:
                prob_nobtts_and_over += prob
            else:
                prob_nobtts_and_under += prob

    return {
        "btts_and_over": prob_btts_and_over,
        "btts_and_under": prob_btts_and_under,
        "nobtts_and_over": prob_nobtts_and_over,
        "nobtts_and_under": prob_nobtts_and_under,
        "btts_total": prob_btts_and_over + prob_btts_and_under,
        "over_total": prob_btts_and_over + prob_nobtts_and_over
    }


def monte_carlo_validation(home_xg: float, away_xg: float,
                           n_simulations: int = 10000,
                           use_dixon_coles: bool = True) -> Dict:
    """
    Monte Carlo Simulation per validazione robustezza.

    Simula partita N volte per:
    - Verificare accuratezza modello
    - Calcolare confidence intervals
    - Catturare varianza e tail risks

    Args:
        home_xg: xG casa
        away_xg: xG trasferta
        n_simulations: Numero simulazioni (default 10000)
        use_dixon_coles: Usa Dixon-Coles per probabilità

    Returns:
        Dict con statistiche simulazione
    """
    import random

    results = {"1": 0, "X": 0, "2": 0}
    total_goals_list = []
    btts_count = 0

    # Precalcola probabilità per velocità
    probs_matrix = []
    for h in range(10):
        for a in range(10):
            if use_dixon_coles:
                prob = dixon_coles_probability(h, a, home_xg, away_xg)
            else:
                prob = poisson_probability(h, home_xg) * poisson_probability(a, away_xg)
            probs_matrix.append(((h, a), prob))

    # Normalizza probabilità
    total_prob = sum(p for _, p in probs_matrix)
    probs_matrix = [((h, a), p/total_prob) for (h, a), p in probs_matrix]

    # Simula usando distribuzione
    outcomes = [outcome for outcome, _ in probs_matrix]
    probs = [prob for _, prob in probs_matrix]

    for _ in range(n_simulations):
        # Campiona risultato dalla distribuzione usando random.choices
        home_goals, away_goals = random.choices(outcomes, weights=probs, k=1)[0]

        # Registra statistiche
        if home_goals > away_goals:
            results["1"] += 1
        elif home_goals == away_goals:
            results["X"] += 1
        else:
            results["2"] += 1

        total_goals_list.append(home_goals + away_goals)

        if home_goals > 0 and away_goals > 0:
            btts_count += 1

    # Calcola statistiche
    prob_1 = results["1"] / n_simulations
    prob_x = results["X"] / n_simulations
    prob_2 = results["2"] / n_simulations
    prob_btts = btts_count / n_simulations

    # Confidence intervals (95%)
    ci_1 = 1.96 * math.sqrt(prob_1 * (1 - prob_1) / n_simulations)
    ci_x = 1.96 * math.sqrt(prob_x * (1 - prob_x) / n_simulations)
    ci_2 = 1.96 * math.sqrt(prob_2 * (1 - prob_2) / n_simulations)

    return {
        "prob_1": prob_1,
        "prob_x": prob_x,
        "prob_2": prob_2,
        "prob_btts": prob_btts,
        "ci_1": ci_1,
        "ci_x": ci_x,
        "ci_2": ci_2,
        "avg_total_goals": sum(total_goals_list) / len(total_goals_list),
        "std_total_goals": math.sqrt(sum((x - sum(total_goals_list)/len(total_goals_list))**2 for x in total_goals_list) / len(total_goals_list))
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

        # ============== ADVANCED MARKET INTELLIGENCE ==============
        # 1. Sharp Money Detection
        sharp = detect_sharp_money(spread_open, spread_close, total_open, total_close)

        # 2. Steam Move Detection
        steam = detect_steam_move(spread_open, spread_close)

        # 3. Market Correlation
        correlation = calculate_market_correlation(spread_open, spread_close, total_open, total_close)

        # 4. Key Numbers Analysis
        key_numbers = analyze_key_numbers(spread_close, total_close)

        # 5. Market Efficiency
        efficiency = calculate_market_efficiency(spread_open, spread_close, total_open, total_close)

        # Crea oggetto MarketIntelligence
        market_intelligence = MarketIntelligence(
            sharp_money_detected=sharp["sharp_detected"],
            sharp_spread_velocity=sharp["spread_velocity"],
            sharp_total_velocity=sharp["total_velocity"],
            contrarian_signal=sharp["contrarian"],
            sharp_confidence_boost=sharp["confidence_boost"],
            steam_move_detected=steam["is_steam"],
            steam_magnitude=steam["magnitude"],
            reverse_steam=steam["reverse_steam"],
            steam_direction=steam["direction"],
            correlation_score=correlation["score"],
            correlation_interpretation=correlation["interpretation"],
            market_coherent=correlation["coherent"],
            on_key_spread=key_numbers["on_key_spread"],
            on_key_total=key_numbers["on_key_total"],
            spread_key_number=key_numbers["spread_key"],
            total_key_number=key_numbers["total_key"],
            key_confidence_boost=key_numbers["confidence_boost"],
            efficiency_score=efficiency["score"],
            efficiency_status=efficiency["status"],
            value_opportunity=efficiency["value_opportunity"]
        )
        # ==========================================================

        # ============== OPZIONE C: ADVANCED EXPECTED GOALS ==============
        # Usa TUTTE le funzioni avanzate: Bayesian, Monte Carlo, Market-Adjusted, ecc.
        expected_goals = calculate_expected_goals_advanced(
            spread_open=spread_open,
            spread_close=spread_close,
            total_open=total_open,
            total_close=total_close,
            market_intel=market_intelligence,
            spread_analysis=spread_analysis,
            total_analysis=total_analysis
        )
        # ================================================================

        # Gestisci casi stabili
        spread_dir_key = spread_analysis.direction.name if spread_analysis.direction != MovementDirection.STABLE else "STABLE"
        total_dir_key = total_analysis.direction.name if total_analysis.direction != MovementDirection.STABLE else "STABLE"

        # Ottieni combinazione
        combination = self._get_combination_interpretation(
            spread_analysis, total_analysis, spread_dir_key, total_dir_key
        )

        # Calcola mercati nelle 4 categorie (ora con xG e market intelligence)
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

        # Calcola confidenza generale (migliorata con xG e market intelligence)
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
            expected_goals=expected_goals,
            market_intelligence=market_intelligence
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
    
    def _validate_ou_with_xg(self, movement_rec: str, movement_conf: ConfidenceLevel,
                             movement_explanation: str, total_value: float,
                             xg: ExpectedGoals) -> tuple:
        """
        OPZIONE B: Valida raccomandazione O/U movimento-based con xG totale.

        Se movimento e xG discordano, riduce confidence e aggiunge warning.

        Args:
            movement_rec: Raccomandazione basata su movimento (es. "Over 2.5")
            movement_conf: Confidence della raccomandazione movimento
            movement_explanation: Spiegazione movimento-based
            total_value: Valore total closing
            xg: ExpectedGoals con home_xg e away_xg

        Returns:
            (recommendation, confidence, explanation) aggiustati
        """
        # Calcola xG totale
        total_xg = xg.home_xg + xg.away_xg

        # Determina raccomandazione xG-based
        xg_rec = "Over" if total_xg > total_value else "Under"

        # Estrai raccomandazione movimento (Over o Under)
        movement_type = "Over" if "Over" in movement_rec else "Under" if "Under" in movement_rec else None

        if movement_type is None:
            # Neutrale o altro → ritorna originale
            return movement_rec, movement_conf, movement_explanation

        # Verifica concordanza
        if movement_type == xg_rec:
            # CONCORDANTI → tutto ok, boost confidence se entrambi d'accordo
            diff = abs(total_xg - total_value)
            if diff >= 0.3:
                # xG molto lontano da total → forte segnale
                boosted_expl = movement_explanation + f" ✓ xG conferma ({total_xg:.1f})"
                # Non aumentare confidence oltre HIGH
                boosted_conf = ConfidenceLevel.HIGH if movement_conf != ConfidenceLevel.HIGH else movement_conf
                return movement_rec, boosted_conf, boosted_expl
            else:
                # Concordanti ma xG vicino a total
                return movement_rec, movement_conf, movement_explanation

        else:
            # DISCORDANTI → riduce confidence e aggiunge warning
            diff = abs(total_xg - total_value)

            if diff >= 0.5:
                # xG molto lontano → forte discordanza
                warning = f" ⚠️ xG suggerisce {xg_rec} ({total_xg:.1f} vs {total_value})"
                return movement_rec, ConfidenceLevel.LOW, movement_explanation + warning
            else:
                # xG vicino a total → discordanza lieve
                warning = f" (xG {total_xg:.1f} vicino a {total_value})"
                # Riduci di un livello
                reduced_conf = (ConfidenceLevel.MEDIUM if movement_conf == ConfidenceLevel.HIGH
                                else ConfidenceLevel.LOW)
                return movement_rec, reduced_conf, movement_explanation + warning

    def _validate_1x2_with_bayesian(self, spread_rec: str, spread_conf: ConfidenceLevel,
                                     spread_explanation: str, xg: ExpectedGoals,
                                     favorito: str) -> tuple:
        """
        OPZIONE C: Valida raccomandazione 1X2 spread-based con probabilità Market-Adjusted Bayesian.

        Se discrepanza > 20%, aggiunge warning.
        Se discrepanza > 30%, override con Bayesian.

        Args:
            spread_rec: Raccomandazione basata su spread (es. "1", "X", "1X")
            spread_conf: Confidence della raccomandazione spread
            spread_explanation: Spiegazione spread-based
            xg: ExpectedGoals con market_adjusted_1x2
            favorito: Chi è il favorito ("1", "2", "X")

        Returns:
            (recommendation, confidence, explanation) aggiustati
        """
        # Se non abbiamo market-adjusted, ritorna originale
        if xg.market_adjusted_1x2 is None:
            return spread_rec, spread_conf, spread_explanation

        # Estrai probabilità Bayesian
        bayesian_home = xg.market_adjusted_1x2['home_win']
        bayesian_draw = xg.market_adjusted_1x2['draw']
        bayesian_away = xg.market_adjusted_1x2['away_win']

        # Determina raccomandazione Bayesian
        max_prob = max(bayesian_home, bayesian_draw, bayesian_away)

        if max_prob == bayesian_home:
            bayesian_rec = "1"
            bayesian_prob = bayesian_home
        elif max_prob == bayesian_draw:
            bayesian_rec = "X"
            bayesian_prob = bayesian_draw
        else:
            bayesian_rec = "2"
            bayesian_prob = bayesian_away

        # Verifica se spread_rec è compatibile con bayesian_rec
        # "1" compatibile con "1", "1X", "12"
        # "X" compatibile con "X", "1X", "X2", "12"
        # "2" compatibile con "2", "X2", "12"

        def is_compatible(spread_r: str, bayesian_r: str) -> bool:
            # Normalizza spread_rec (rimuovi spazi e parentesi)
            spread_r_clean = spread_r.split()[0] if ' ' in spread_r else spread_r

            if bayesian_r == "1":
                return "1" in spread_r_clean
            elif bayesian_r == "X":
                return "X" in spread_r_clean
            elif bayesian_r == "2":
                return "2" in spread_r_clean
            return False

        # Calcola discrepanza
        # Se spread raccomanda favorito, confronta con probabilità favorito
        if is_compatible(spread_rec, bayesian_rec):
            # Compatibile → tutto ok
            discrepancy = 0.0
        else:
            # Incompatibile → calcola discrepanza
            discrepancy = abs(bayesian_prob - 0.5)  # Quanto è sicuro Bayesian?

        # DECISIONE BASATA SU DISCREPANZA

        if discrepancy <= 0.20:
            # Discrepanza bassa (≤ 20%) → mantieni spread-based
            return spread_rec, spread_conf, spread_explanation

        elif discrepancy <= 0.30:
            # Discrepanza media (20-30%) → warning ma mantieni spread
            warning = f" ⚠️ Market-Adjusted suggerisce {bayesian_rec} ({bayesian_prob:.0%})"
            return spread_rec, ConfidenceLevel.LOW, spread_explanation + warning

        else:
            # Discrepanza alta (> 30%) → override con Bayesian
            override_rec = bayesian_rec
            override_conf = ConfidenceLevel.MEDIUM if bayesian_prob >= 0.60 else ConfidenceLevel.LOW
            override_explanation = (
                f"Market-Adjusted Bayesian: {bayesian_rec} {bayesian_prob:.0%} "
                f"(Home {bayesian_home:.0%}, Draw {bayesian_draw:.0%}, Away {bayesian_away:.0%}). "
                f"Override spread-based ({spread_rec}) per alta discrepanza."
            )
            return override_rec, override_conf, override_explanation

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
        # Usa probabilità BTTS AVANZATA (Bayesian) se disponibile, altrimenti base Poisson
        # IMPORTANTE: Bayesian include movimenti mercato → più accurato!
        if xg.bayesian_btts is not None:
            btts_prob = xg.bayesian_btts['btts_prob']
        else:
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
            # Zona grigia (0.40 - 0.65): usa soglia 50% per decidere
            if btts_prob >= 0.50:
                # BTTS >= 50% → GOAL lievemente favorito
                recommendations.append(MarketRecommendation(
                    market_name="GOAL/NOGOAL",
                    recommendation=f"GOAL (lievemente favorito)",
                    confidence=ConfidenceLevel.LOW,
                    explanation=f"P(BTTS)={btts_prob:.1%} > 50%, entrambe probabilmente segnano"
                ))
            else:
                # BTTS < 50% → NOGOAL lievemente favorito
                recommendations.append(MarketRecommendation(
                    market_name="GOAL/NOGOAL",
                    recommendation=f"NOGOAL (lievemente favorito)",
                    confidence=ConfidenceLevel.LOW,
                    explanation=f"P(BTTS)={btts_prob:.1%} < 50%, almeno una probabilmente non segna"
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

        # OPZIONE B: Calcola probabilità HT usando xG TIME-WEIGHTED (non più fisso 45%)
        ht_probs = calculate_time_weighted_xg_ht(
            xg.home_xg, xg.away_xg,
            total.closing_value, spread.closing_value
        )

        # === 1. HT/FT COMBINATIONS - OPZIONE B: CON CORRELAZIONE E MOMENTUM ===
        # Non più indipendenza: chi vince HT ha +20% prob vincere FT (momentum)
        ht_ft_probs = calculate_ht_ft_with_correlation(xg.home_xg, xg.away_xg, spread.closing_value)

        # FT probabilities (needed for COMBO section later)
        ft_home_prob = xg.home_win_prob
        ft_draw_prob = xg.draw_prob
        ft_away_prob = xg.away_win_prob

        # Trova combinazione HT/FT più probabile
        ht_ft_combinations = sorted(ht_ft_probs.items(), key=lambda x: x[1], reverse=True)

        # Raccomanda top 1-2 combinazioni HT/FT
        top_ht_ft = ht_ft_combinations[0]
        if top_ht_ft[1] >= 0.15:  # Almeno 15% probabilità
            conf = ConfidenceLevel.HIGH if top_ht_ft[1] >= 0.25 else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="HT/FT",
                recommendation=f"{top_ht_ft[0]}",
                confidence=conf,
                explanation=f"Probabilità con correlazione: {top_ht_ft[1]:.1%} (HT xG: {ht_probs['home_xg_ht']:.2f} vs {ht_probs['away_xg_ht']:.2f}, HT%={ht_probs['ht_percentage']:.0%})"
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

        # === 4. MULTIGOL - OPZIONE B: CON STICKY SCORES (1-1, 2-1 boost) ===
        # Poisson puro sottostima risultati comuni, sovrastima risultati rari
        goals_dist = calculate_total_goals_distribution(xg.home_xg, xg.away_xg)

        # Applica aggiustamenti sticky scores (1-0, 1-1, 2-1 boost; 5+ gol penalità)
        goals_dist_adjusted = apply_sticky_scores_adjustment(goals_dist, xg.home_xg, xg.away_xg)

        # Calcola probabilità per range multigol (con sticky scores)
        prob_0_1 = goals_dist_adjusted.get(0, 0) + goals_dist_adjusted.get(1, 0)
        prob_1_2 = goals_dist_adjusted.get(1, 0) + goals_dist_adjusted.get(2, 0)
        prob_1_3 = sum(goals_dist_adjusted.get(i, 0) for i in range(1, 4))
        prob_2_3 = goals_dist_adjusted.get(2, 0) + goals_dist_adjusted.get(3, 0)
        prob_2_4 = sum(goals_dist_adjusted.get(i, 0) for i in range(2, 5))
        prob_3_5 = sum(goals_dist_adjusted.get(i, 0) for i in range(3, 6))

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
                explanation=f"Probabilità {top_multigol[1]:.1%} con sticky scores (xG totale={xg.home_xg + xg.away_xg:.2f})"
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

        # ===== OPZIONE C: Validazione 1X2 con Market-Adjusted Bayesian =====
        # Trova raccomandazione 1X2
        x2_rec_idx = next((i for i, r in enumerate(all_recommendations) if r.market_name == "1X2"), None)

        if x2_rec_idx is not None and xg.market_adjusted_1x2 is not None:
            x2_rec = all_recommendations[x2_rec_idx]

            # Usa la funzione di validazione
            validated_rec, validated_conf, validated_expl = self._validate_1x2_with_bayesian(
                x2_rec.recommendation,
                x2_rec.confidence,
                x2_rec.explanation,
                xg,
                "1" if spread.closing_value < 0 else "2" if spread.closing_value > 0 else "X"
            )

            # Se cambiato, sostituisci
            if (validated_rec != x2_rec.recommendation or
                validated_conf != x2_rec.confidence or
                validated_expl != x2_rec.explanation):
                all_recommendations[x2_rec_idx] = MarketRecommendation(
                    market_name="1X2",
                    recommendation=validated_rec,
                    confidence=validated_conf,
                    explanation=validated_expl
                )

        # ===== OPZIONE B: Validazione Over/Under con xG =====
        # Trova raccomandazione Over/Under
        ou_rec_idx = next((i for i, r in enumerate(all_recommendations) if r.market_name == "Over/Under"), None)

        if ou_rec_idx is not None:
            ou_rec = all_recommendations[ou_rec_idx]

            # Usa la funzione di validazione
            validated_rec, validated_conf, validated_expl = self._validate_ou_with_xg(
                ou_rec.recommendation,
                ou_rec.confidence,
                ou_rec.explanation,
                total.closing_value,
                xg
            )

            # Se cambiato, sostituisci
            if (validated_rec != ou_rec.recommendation or
                validated_conf != ou_rec.confidence or
                validated_expl != ou_rec.explanation):
                all_recommendations[ou_rec_idx] = MarketRecommendation(
                    market_name="Over/Under",
                    recommendation=validated_rec,
                    confidence=validated_conf,
                    explanation=validated_expl
                )

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

