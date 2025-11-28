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
class AnalysisResult:
    """Risultato completo dell'analisi"""
    spread_analysis: MovementAnalysis
    total_analysis: MovementAnalysis
    combination_interpretation: str
    core_recommendations: List[MarketRecommendation]  # Consigli principali (HIGH/MEDIUM confidence)
    alternative_recommendations: List[MarketRecommendation]  # Opzioni alternative (MEDIUM confidence)
    value_recommendations: List[MarketRecommendation]  # Value bets (LOW confidence, high value)
    overall_confidence: ConfidenceLevel


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
        
        # Gestisci casi stabili
        spread_dir_key = spread_analysis.direction.name if spread_analysis.direction != MovementDirection.STABLE else "STABLE"
        total_dir_key = total_analysis.direction.name if total_analysis.direction != MovementDirection.STABLE else "STABLE"
        
        # Ottieni combinazione
        combination = self._get_combination_interpretation(
            spread_analysis, total_analysis, spread_dir_key, total_dir_key
        )
        
        # Calcola mercati nelle 3 categorie
        core_recs = self._calculate_core_markets(spread_analysis, total_analysis, combination)
        alternative_recs = self._calculate_alternative_markets(spread_analysis, total_analysis, combination)
        value_recs = self._calculate_value_markets(spread_analysis, total_analysis, combination)

        # Calcola confidenza generale
        overall_confidence = self._calculate_confidence(spread_analysis, total_analysis)

        return AnalysisResult(
            spread_analysis=spread_analysis,
            total_analysis=total_analysis,
            combination_interpretation=combination["interpretation"],
            core_recommendations=core_recs,
            alternative_recommendations=alternative_recs,
            value_recommendations=value_recs,
            overall_confidence=overall_confidence
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
                                total: MovementAnalysis, combination: Dict) -> List[MarketRecommendation]:
        """Calcola raccomandazioni principali (CORE) - Alta/Media confidenza"""
        recommendations = []

        abs_spread = abs(spread.closing_value)

        # 1X2 - Sempre presente - LOGICA FISSATA: considera movimento + closing value
        if spread.direction == MovementDirection.HARDEN:
            # Spread si indurisce verso favorito → sempre "1"
            conf = ConfidenceLevel.HIGH if spread.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="1X2",
                recommendation="1 (Favorito)",
                confidence=conf,
                explanation=f"Spread si indurisce: {spread.interpretation}"
            ))
        elif spread.direction == MovementDirection.SOFTEN:
            # Spread si ammorbidisce → guarda closing value
            if abs_spread < 0.5:
                # Match molto equilibrato
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation="X (Pareggio) o X2",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Spread equilibrato ({spread.closing_value}), favorito molto debole"
                ))
            elif abs_spread < 1.0:
                # Match abbastanza equilibrato
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation="1X o X (Favorito o Pareggio)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Spread si ammorbidisce a {spread.closing_value}, incertezza"
                ))
            else:
                # Favorito ancora forte nonostante ammorbidimento
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation="1 (Favorito)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Spread si ammorbidisce ma closing {spread.closing_value} indica favorito ancora forte"
                ))
        else:  # STABLE
            # Spread stabile → guarda solo closing value
            if abs_spread < 0.5:
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation="X (Pareggio)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Match equilibrato, spread {spread.closing_value}"
                ))
            elif abs_spread < 1.0:
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation="1X (Favorito o Pareggio)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Spread moderato {spread.closing_value}"
                ))
            else:
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation="1 (Favorito)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Favorito forte, spread {spread.closing_value}"
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

        # GOAL/NOGOAL - LOGICA MIGLIORATA
        if total.direction == MovementDirection.HARDEN or total.closing_value >= 2.75:
            conf = ConfidenceLevel.HIGH if total.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation="GOAL (Entrambe segnano)",
                confidence=conf,
                explanation=f"Partita viva, total {total.closing_value}"
            ))
        elif total.direction == MovementDirection.SOFTEN or total.closing_value <= 2.0:
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation="NOGOAL (Almeno una non segna)",
                confidence=ConfidenceLevel.MEDIUM,
                explanation=f"Partita chiusa, total {total.closing_value}"
            ))

        # Handicap Asiatico - LOGICA MIGLIORATA: considera closing value
        if spread.direction != MovementDirection.STABLE:
            handicap_value = abs_spread
            if spread.direction == MovementDirection.HARDEN:
                # Favorito si rafforza → gioca favorito con handicap
                recommendations.append(MarketRecommendation(
                    market_name="Handicap Asiatico",
                    recommendation=f"Favorito -{handicap_value}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Favorito copre spread {spread.closing_value}"
                ))
            else:  # SOFTEN
                # Spread si ammorbidisce
                if abs_spread >= 1.0:
                    # Ancora favorito forte → Favorito con handicap ridotto
                    recommendations.append(MarketRecommendation(
                        market_name="Handicap Asiatico",
                        recommendation=f"Favorito -{handicap_value} (valore)",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Spread si ammorbidisce ma {spread.closing_value} ancora vantaggioso"
                    ))
                else:
                    # Match equilibrato → Underdog con handicap
                    recommendations.append(MarketRecommendation(
                        market_name="Handicap Asiatico",
                        recommendation=f"Underdog +{handicap_value}",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Match equilibrato, underdog copre {spread.closing_value}"
                    ))

        # HT Winner - LOGICA MIGLIORATA: considera closing value
        if spread.direction == MovementDirection.HARDEN and spread.intensity in [MovementIntensity.MEDIUM, MovementIntensity.STRONG]:
            # Favorito si rafforza → 1 HT
            recommendations.append(MarketRecommendation(
                market_name="1X2 HT",
                recommendation="1 HT (Favorito vince primo tempo)",
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
                                       total: MovementAnalysis, combination: Dict) -> List[MarketRecommendation]:
        """Calcola raccomandazioni alternative - Media confidenza"""
        recommendations = []

        abs_spread = abs(spread.closing_value)

        # Combo 1 + Over/Under - LOGICA MIGLIORATA: considera closing value
        if spread.direction == MovementDirection.HARDEN:
            # Favorito si rafforza
            if total.direction == MovementDirection.HARDEN:
                recommendations.append(MarketRecommendation(
                    market_name="Combo",
                    recommendation=f"1 + Over {total.closing_value}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Favorito vince con gol"
                ))
            elif total.closing_value >= 2.75:
                # Total scende ma ancora alto
                recommendations.append(MarketRecommendation(
                    market_name="Combo",
                    recommendation=f"1 + Over {total.closing_value}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Favorito vince, total ancora alto"
                ))
            else:
                # Total basso
                recommendations.append(MarketRecommendation(
                    market_name="Combo",
                    recommendation=f"1 + Under {total.closing_value}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Favorito vince di corto"
                ))
        elif spread.direction == MovementDirection.SOFTEN:
            # Spread si ammorbidisce → guarda closing value
            if abs_spread >= 1.0:
                # Favorito ancora forte
                if total.closing_value <= 2.25:
                    recommendations.append(MarketRecommendation(
                        market_name="Combo",
                        recommendation=f"1 + Under {total.closing_value}",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Favorito vince corto ({spread.closing_value})"
                    ))
            else:
                # Match equilibrato
                if total.direction == MovementDirection.SOFTEN or total.closing_value <= 2.25:
                    recommendations.append(MarketRecommendation(
                        market_name="Combo",
                        recommendation=f"X + Under {total.closing_value}",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation="Match equilibrato e chiuso"
                    ))

        # HT/FT Combinations - LOGICA MIGLIORATA: considera closing value
        if spread.direction == MovementDirection.HARDEN:
            # Favorito si rafforza
            if spread.intensity == MovementIntensity.STRONG:
                recommendations.append(MarketRecommendation(
                    market_name="HT/FT",
                    recommendation="1/1 (Favorito HT e FT)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Favorito domina dall'inizio"
                ))
            else:
                recommendations.append(MarketRecommendation(
                    market_name="HT/FT",
                    recommendation="X/1 (Pareggio HT, Favorito FT)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Favorito decide nella ripresa"
                ))
        elif spread.direction == MovementDirection.SOFTEN:
            # Spread si ammorbidisce → guarda closing value
            if abs_spread < 0.75:
                # Match molto equilibrato
                recommendations.append(MarketRecommendation(
                    market_name="HT/FT",
                    recommendation="X/X (Pareggio HT e FT)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Match equilibrato ({spread.closing_value})"
                ))
            elif abs_spread >= 1.0:
                # Favorito ancora presente
                recommendations.append(MarketRecommendation(
                    market_name="HT/FT",
                    recommendation="X/1 (Pareggio HT, Favorito FT)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Favorito decide nella ripresa ({spread.closing_value})"
                ))
            else:
                # Incertezza
                recommendations.append(MarketRecommendation(
                    market_name="HT/FT",
                    recommendation="X/X o X/1",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Match incerto ({spread.closing_value})"
                ))
        else:  # STABLE
            # Spread stabile → guarda closing value
            if abs_spread >= 1.0:
                recommendations.append(MarketRecommendation(
                    market_name="HT/FT",
                    recommendation="X/1 (Pareggio HT, Favorito FT)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Favorito vince ({spread.closing_value})"
                ))
            else:
                recommendations.append(MarketRecommendation(
                    market_name="HT/FT",
                    recommendation="X/X (Pareggio HT e FT)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Match equilibrato ({spread.closing_value})"
                ))

        # Over/Under HT
        ht_total_estimate = total.closing_value * 0.5
        if total.direction == MovementDirection.HARDEN:
            if ht_total_estimate >= 1.0:
                recommendations.append(MarketRecommendation(
                    market_name="Over/Under HT",
                    recommendation="Over 1.0 HT",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Total alto ({total.closing_value}), partenza aggressiva"
                ))
            else:
                recommendations.append(MarketRecommendation(
                    market_name="Over/Under HT",
                    recommendation="Over 0.5 HT",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Almeno 1 gol nel primo tempo"
                ))
        elif total.direction == MovementDirection.SOFTEN or total.closing_value <= 2.25:
            recommendations.append(MarketRecommendation(
                market_name="Over/Under HT",
                recommendation="Under 1.0 HT",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Primo tempo tattico, massimo 1 gol"
            ))

        # GOAL HT
        if total.direction == MovementDirection.HARDEN and ht_total_estimate >= 1.0:
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL HT",
                recommendation="GOAL 1T (Entrambe segnano 1T)",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Gol da entrambe già nel primo tempo"
            ))
        elif total.direction == MovementDirection.SOFTEN:
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL HT",
                recommendation="NOGOAL 1T",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Almeno una squadra non segna nel 1T"
            ))

        # Multigol (se rilevante)
        if total.closing_value >= 2.5:
            if total.direction == MovementDirection.HARDEN:
                recommendations.append(MarketRecommendation(
                    market_name="Multigol",
                    recommendation="2-4 gol o 3-5 gol",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Partita con molti gol attesi"
                ))
            else:
                recommendations.append(MarketRecommendation(
                    market_name="Multigol",
                    recommendation="1-3 gol",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Gol moderati attesi"
                ))
        elif total.closing_value <= 2.0:
            recommendations.append(MarketRecommendation(
                market_name="Multigol",
                recommendation="1-2 gol",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Pochi gol attesi"
            ))

        return recommendations[:5]  # Max 5 alternative recommendations

    def _calculate_value_markets(self, spread: MovementAnalysis,
                                 total: MovementAnalysis, combination: Dict) -> List[MarketRecommendation]:
        """Calcola value bets - Bassa confidenza ma potenziale valore"""
        recommendations = []

        abs_spread = abs(spread.closing_value)

        # Risultati esatti - Top 2-3 più probabili
        exact_scores = self._get_likely_exact_scores(spread, total)
        for score, explanation in exact_scores[:3]:  # Max 3
            recommendations.append(MarketRecommendation(
                market_name="Risultato Esatto",
                recommendation=score,
                confidence=ConfidenceLevel.LOW,
                explanation=explanation
            ))

        # Double Chance - LOGICA FISSATA: considera closing value
        if spread.direction == MovementDirection.HARDEN or abs_spread >= 1.0:
            # Favorito si rafforza O è comunque forte
            recommendations.append(MarketRecommendation(
                market_name="Double Chance",
                recommendation="1X (Favorito o Pareggio)",
                confidence=ConfidenceLevel.LOW,
                explanation=f"Opzione sicura, favorito non perde ({spread.closing_value})"
            ))
        elif spread.direction == MovementDirection.SOFTEN and abs_spread < 0.75:
            # Spread si ammorbidisce E match molto equilibrato
            recommendations.append(MarketRecommendation(
                market_name="Double Chance",
                recommendation="X2 (Pareggio o Underdog)",
                confidence=ConfidenceLevel.LOW,
                explanation="Opzione sicura contro favorito debole"
            ))

        # Primo gol timing (se rilevante)
        if total.direction == MovementDirection.HARDEN and total.intensity == MovementIntensity.STRONG:
            recommendations.append(MarketRecommendation(
                market_name="Timing",
                recommendation="Primo gol prima del 30' minuto",
                confidence=ConfidenceLevel.LOW,
                explanation="Partita viva, inizio aggressivo"
            ))
        elif total.direction == MovementDirection.SOFTEN:
            recommendations.append(MarketRecommendation(
                market_name="Timing",
                recommendation="Primo gol dopo il 30' minuto",
                confidence=ConfidenceLevel.LOW,
                explanation="Partita tattica, fase di studio"
            ))

        return recommendations[:3]  # Max 3 value recommendations

    def _get_likely_exact_scores(self, spread: MovementAnalysis, total: MovementAnalysis) -> List[Tuple[str, str]]:
        """Restituisce i risultati esatti più probabili basati su spread e total"""
        scores = []

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

        return scores
    
    def _calculate_confidence(self, spread: MovementAnalysis,
                             total: MovementAnalysis) -> ConfidenceLevel:
        """Calcola confidenza generale"""
        
        # Se entrambi concordi e forti → alta
        if (spread.direction == total.direction and 
            spread.intensity in [MovementIntensity.MEDIUM, MovementIntensity.STRONG] and
            total.intensity in [MovementIntensity.MEDIUM, MovementIntensity.STRONG]):
            return ConfidenceLevel.HIGH
        
        # Se concordi ma uno leggero → media
        if spread.direction == total.direction:
            return ConfidenceLevel.MEDIUM
        
        # Se discordi → bassa
        if spread.direction != MovementDirection.STABLE and total.direction != MovementDirection.STABLE:
            if spread.direction != total.direction:
                return ConfidenceLevel.LOW
        
        # Default: media
        return ConfidenceLevel.MEDIUM


def format_output(result: AnalysisResult) -> str:
    """Formatta l'output in modo leggibile"""
    
    output = []
    output.append("=" * 60)
    output.append("📊 ANALISI MOVIMENTO MERCATO")
    output.append("=" * 60)
    output.append("")
    
    # Movimenti
    output.append("📈 MOVIMENTI:")
    output.append(f"   Spread: {result.spread_analysis.opening_value:.2f} → {result.spread_analysis.closing_value:.2f}")
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

