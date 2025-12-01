#!/usr/bin/env python3
"""
Market Movement Analyzer (Sib10)
=================================

Tool per analizzare i movimenti di Spread e Total e generare
interpretazioni e giocate consigliate basate su pattern di mercato.

Version: Sib10 - Calcoli migliorati con logica avanzata per tutti i mercati

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
    
    # Tabelle interpretazione movimenti spread (gestisce valori positivi e negativi)
    SPREAD_MOVEMENTS_DOWN = {
        # Valori negativi (casa favorita)
        -1.75: (-1.5, "Leggero calo di fiducia nel favorito, mercato più prudente"),
        -1.5: (-1.25, "Favorito meno dominante, mercato vede 1–0 o 2–1"),
        -1.25: (-1.0, "Fiducia in calo evidente, l'underdog 'resiste' meglio"),
        -1.0: (-0.75, "Mercato molto più equilibrato, favorito vulnerabile"),
        -0.75: (-0.5, "Fiducia nel favorito in netto calo"),
        -0.5: (-0.25, "Favorito quasi al pari"),
        -0.25: (0.0, "Favorito perde fiducia → equilibrio totale"),
        # Valori positivi (trasferta favorita)
        1.75: (1.5, "Leggero calo di fiducia nella trasferta favorita, mercato più prudente"),
        1.5: (1.25, "Trasferta meno dominante, mercato vede 0–1 o 1–2"),
        1.25: (1.0, "Fiducia in calo evidente, la casa 'resiste' meglio"),
        1.0: (0.75, "Mercato molto più equilibrato, trasferta vulnerabile"),
        0.75: (0.5, "Fiducia nella trasferta in netto calo"),
        0.5: (0.25, "Trasferta quasi al pari"),
        0.25: (0.0, "Trasferta perde fiducia → equilibrio totale"),
    }
    
    SPREAD_MOVEMENTS_UP = {
        # Valori negativi (casa favorita)
        -0.25: (-0.5, "Cresce la fiducia nella casa favorita, mercato la vede superiore"),
        -0.5: (-0.75, "Casa in netto rafforzamento, l'1 diventa più stabile"),
        -0.75: (-1.0, "Vittoria più larga attesa, possibile 2–0"),
        -1.0: (-1.25, "Grande fiducia nella casa, mercato sbilanciato su di lei"),
        -1.25: (-1.5, "Casa dominante, l'underdog viene svalutato"),
        -1.5: (-1.75, "Massima fiducia, probabile goleada"),
        # Valori positivi (trasferta favorita)
        0.25: (0.5, "Cresce la fiducia nella trasferta favorita, mercato la vede superiore"),
        0.5: (0.75, "Trasferta in netto rafforzamento, il 2 diventa più stabile"),
        0.75: (1.0, "Vittoria più larga attesa, possibile 0–2"),
        1.0: (1.25, "Grande fiducia nella trasferta, mercato sbilanciato su di lei"),
        1.25: (1.5, "Trasferta dominante, la casa viene svalutata"),
        1.5: (1.75, "Massima fiducia, probabile goleada trasferta"),
    }
    
    def analyze(self, opening: float, closing: float) -> MovementAnalysis:
        """Analizza il movimento dello spread con gestione migliorata"""
        # Gestione cambio segno (pick'em o cambio favorito)
        sign_changed = (opening > 0 and closing < 0) or (opening < 0 and closing > 0)
        
        if sign_changed:
            # Cambio di favorito: caso speciale
            total_movement = abs(opening) + abs(closing)
            direction = MovementDirection.SOFTEN  # Sempre ammorbidisce verso equilibrio
            interpretation = f"Cambio favorito: da {format_spread_display(opening)} a {format_spread_display(closing)}"
            steps = self._calculate_discrete_steps(opening, closing, is_spread=True)
            intensity = self._calculate_intensity(total_movement, opening, closing, is_spread=True)
        else:
            # Movimento normale: usa valore assoluto
            abs_opening = abs(opening)
            abs_closing = abs(closing)
            movement = abs_closing - abs_opening  # Se negativo = si ammorbidisce, positivo = si indurisce
            
            # Soglia stabilità migliorata (0.125 = mezzo step discreto)
            if abs(movement) < 0.125:
                direction = MovementDirection.STABLE
                intensity = MovementIntensity.NONE
                interpretation = "Spread stabile, nessun cambiamento significativo"
                steps = 0.0
            elif movement < 0:
                direction = MovementDirection.SOFTEN
                interpretation = self._get_soften_interpretation(opening, closing)
                steps = self._calculate_discrete_steps(opening, closing, is_spread=True)
                intensity = self._calculate_intensity(abs(movement), opening, closing, is_spread=True)
            else:
                direction = MovementDirection.HARDEN
                interpretation = self._get_harden_interpretation(opening, closing)
                steps = self._calculate_discrete_steps(opening, closing, is_spread=True)
                intensity = self._calculate_intensity(abs(movement), opening, closing, is_spread=True)
        
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
    
    def _calculate_discrete_steps(self, opening: float, closing: float, is_spread: bool) -> float:
        """Calcola step discreti effettivi tra valori di linea (multipli di 0.25)"""
        # Normalizza ai valori di linea (multipli di 0.25)
        def normalize_to_line(value: float) -> float:
            return round(value * 4) / 4
        
        norm_open = normalize_to_line(opening)
        norm_close = normalize_to_line(closing)
        
        if is_spread:
            # Per spread: usa valore assoluto
            abs_open = abs(norm_open)
            abs_close = abs(norm_close)
            diff = abs_close - abs_open
        else:
            # Per total: differenza diretta
            diff = norm_close - norm_open
        
        # Converti in step discreti (ogni 0.25 = 1 step)
        steps = diff / 0.25
        
        return round(steps, 2)  # Arrotonda a 2 decimali per precisione
    
    def _calculate_intensity(self, movement: float, opening_value: float, 
                            closing_value: float, is_spread: bool) -> MovementIntensity:
        """Calcola intensità normalizzata e contestualizzata"""
        abs_movement = abs(movement)
        
        # Normalizza per valore base (movimento relativo)
        if abs(opening_value) > 0.01:
            relative_movement = abs_movement / abs(opening_value)
        else:
            relative_movement = abs_movement
        
        # Arrotonda a step discreti (0.25)
        discrete_steps = round(abs_movement / 0.25)
        
        # Intensità basata su step discreti + movimento relativo
        if discrete_steps == 0:
            return MovementIntensity.NONE
        elif discrete_steps == 1:
            # Leggero: 1 step (0.25) o movimento relativo < 15%
            if relative_movement < 0.15:
                return MovementIntensity.LIGHT
            else:
                return MovementIntensity.MEDIUM
        elif discrete_steps == 2:
            # Medio: 2 step (0.5) o movimento relativo 15-35%
            if relative_movement < 0.35:
                return MovementIntensity.MEDIUM
            else:
                return MovementIntensity.STRONG
        elif discrete_steps >= 3:
            # Forte: 3+ step (0.75+) o movimento relativo > 35%
            return MovementIntensity.STRONG
        else:
            # Default per valori intermedi
            if relative_movement < 0.20:
                return MovementIntensity.LIGHT
            elif relative_movement < 0.40:
                return MovementIntensity.MEDIUM
            else:
                return MovementIntensity.STRONG
    
    def _get_soften_interpretation(self, opening: float, closing: float) -> str:
        """Ottiene interpretazione per spread che si ammorbidisce con interpolazione"""
        # Normalizza per gestire valori assoluti
        abs_opening = abs(opening)
        abs_closing = abs(closing)
        opening_sign = -1 if opening < 0 else 1
        closing_sign = -1 if closing < 0 else 1
        
        # Cerca nella tabella il range più vicino (considera segno)
        best_match = None
        min_distance = float('inf')
        
        for start, (end, interpretation) in self.SPREAD_MOVEMENTS_DOWN.items():
            # Controlla se il segno corrisponde
            if (start < 0 and opening < 0) or (start > 0 and opening > 0) or (start == 0):
                # Match esatto
                if (opening_sign * abs_opening >= opening_sign * start and 
                    closing_sign * abs_closing <= closing_sign * end):
                    return interpretation
                
                # Calcola distanza per interpolazione
                if opening_sign * abs_opening >= opening_sign * start:
                    distance = abs(abs_closing - abs(end)) if closing_sign * abs_closing > closing_sign * end else 0
                else:
                    distance = abs(abs_opening - abs(start))
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = (start, end, interpretation)
        
        # Se trovato un match vicino, modifica l'interpretazione
        if best_match and min_distance < 0.5:
            start, end, base_interpretation = best_match
            movement_completion = abs(abs_closing - abs(end)) / abs(abs(start) - abs(end)) if abs(abs(start) - abs(end)) > 0 else 1.0
            
            if movement_completion < 0.3:
                modifier = "parzialmente "
            elif movement_completion > 0.7:
                modifier = ""
            else:
                modifier = "progressivamente "
            
            return modifier + base_interpretation.lower()
        
        # Fallback migliorato
        if abs_closing < abs_opening:
            return f"Fiducia nel favorito cala da {format_spread_display(opening)} a {format_spread_display(closing)}"
        return "Spread si ammorbidisce, mercato più equilibrato"
    
    def _get_harden_interpretation(self, opening: float, closing: float) -> str:
        """Ottiene interpretazione per spread che si indurisce con interpolazione"""
        abs_opening = abs(opening)
        abs_closing = abs(closing)
        opening_sign = -1 if opening < 0 else 1
        closing_sign = -1 if closing < 0 else 1
        
        best_match = None
        min_distance = float('inf')
        
        for start, (end, interpretation) in self.SPREAD_MOVEMENTS_UP.items():
            # Controlla se il segno corrisponde
            if (start < 0 and opening < 0) or (start > 0 and opening > 0) or (start == 0):
                # Match esatto
                if (opening_sign * abs_opening <= opening_sign * start and 
                    closing_sign * abs_closing >= closing_sign * end):
                    return interpretation
                
                # Calcola distanza per interpolazione
                if opening_sign * abs_opening <= opening_sign * start:
                    distance = abs(abs_closing - abs(end)) if closing_sign * abs_closing < closing_sign * end else 0
                else:
                    distance = abs(abs_opening - abs(start))
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = (start, end, interpretation)
        
        # Se trovato un match vicino, modifica l'interpretazione
        if best_match and min_distance < 0.5:
            start, end, base_interpretation = best_match
            movement_completion = abs(abs_closing - abs(end)) / abs(abs(start) - abs(end)) if abs(abs(start) - abs(end)) > 0 else 1.0
            
            if movement_completion < 0.3:
                modifier = "parzialmente "
            elif movement_completion > 0.7:
                modifier = ""
            else:
                modifier = "progressivamente "
            
            return modifier + base_interpretation.lower()
        
        # Fallback migliorato
        if abs_closing > abs_opening:
            return f"Fiducia nel favorito aumenta da {format_spread_display(opening)} a {format_spread_display(closing)}"
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
        """Analizza il movimento del total con calcoli migliorati"""
        movement = closing - opening  # Positivo = sale, negativo = scende
        abs_movement = abs(movement)
        
        # Soglia stabilità migliorata (0.125 = mezzo step discreto)
        if abs_movement < 0.125:
            direction = MovementDirection.STABLE
            intensity = MovementIntensity.NONE
            interpretation = "Total stabile, ritmo atteso invariato"
            steps = 0.0
        elif movement > 0:
            direction = MovementDirection.HARDEN  # Sale = più gol
            interpretation = self._get_up_interpretation(opening, closing)
            steps = self._calculate_discrete_steps(opening, closing, is_spread=False)
            intensity = self._calculate_intensity(abs_movement, opening, closing, is_spread=False)
        else:
            direction = MovementDirection.SOFTEN  # Scende = meno gol
            interpretation = self._get_down_interpretation(opening, closing)
            steps = self._calculate_discrete_steps(opening, closing, is_spread=False)
            intensity = self._calculate_intensity(abs_movement, opening, closing, is_spread=False)
        
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
    
    def _calculate_discrete_steps(self, opening: float, closing: float, is_spread: bool) -> float:
        """Calcola step discreti effettivi tra valori di linea (multipli di 0.25)"""
        # Normalizza ai valori di linea (multipli di 0.25)
        def normalize_to_line(value: float) -> float:
            return round(value * 4) / 4
        
        norm_open = normalize_to_line(opening)
        norm_close = normalize_to_line(closing)
        
        if is_spread:
            # Per spread: usa valore assoluto
            abs_open = abs(norm_open)
            abs_close = abs(norm_close)
            diff = abs_close - abs_open
        else:
            # Per total: differenza diretta
            diff = norm_close - norm_open
        
        # Converti in step discreti (ogni 0.25 = 1 step)
        steps = diff / 0.25
        
        return round(steps, 2)  # Arrotonda a 2 decimali per precisione
    
    def _calculate_intensity(self, movement: float, opening_value: float, 
                            closing_value: float, is_spread: bool) -> MovementIntensity:
        """Calcola intensità normalizzata e contestualizzata"""
        abs_movement = abs(movement)
        
        # Normalizza per valore base (movimento relativo)
        if abs(opening_value) > 0.01:
            relative_movement = abs_movement / abs(opening_value)
        else:
            relative_movement = abs_movement
        
        # Arrotonda a step discreti (0.25)
        discrete_steps = round(abs_movement / 0.25)
        
        # Intensità basata su step discreti + movimento relativo
        if discrete_steps == 0:
            return MovementIntensity.NONE
        elif discrete_steps == 1:
            # Leggero: 1 step (0.25) o movimento relativo < 10%
            if relative_movement < 0.10:
                return MovementIntensity.LIGHT
            else:
                return MovementIntensity.MEDIUM
        elif discrete_steps == 2:
            # Medio: 2 step (0.5) o movimento relativo 10-25%
            if relative_movement < 0.25:
                return MovementIntensity.MEDIUM
            else:
                return MovementIntensity.STRONG
        elif discrete_steps >= 3:
            # Forte: 3+ step (0.75+) o movimento relativo > 25%
            return MovementIntensity.STRONG
        else:
            # Default per valori intermedi
            if relative_movement < 0.15:
                return MovementIntensity.LIGHT
            elif relative_movement < 0.30:
                return MovementIntensity.MEDIUM
            else:
                return MovementIntensity.STRONG
    
    def _get_up_interpretation(self, opening: float, closing: float) -> str:
        """Interpretazione per total che sale con interpolazione"""
        best_match = None
        min_distance = float('inf')
        
        for start, (end, interpretation) in self.TOTAL_MOVEMENTS_UP.items():
            # Match esatto
            if opening <= start and closing >= end:
                return interpretation
            
            # Calcola distanza per interpolazione
            if opening <= start:
                distance = abs(closing - end) if closing < end else 0
            else:
                distance = abs(opening - start)
            
            if distance < min_distance:
                min_distance = distance
                best_match = (start, end, interpretation)
        
        # Se trovato un match vicino, modifica l'interpretazione
        if best_match and min_distance < 0.5:
            start, end, base_interpretation = best_match
            movement_completion = abs(closing - end) / abs(start - end) if abs(start - end) > 0 else 1.0
            
            if movement_completion < 0.3:
                modifier = "parzialmente "
            elif movement_completion > 0.7:
                modifier = ""
            else:
                modifier = "progressivamente "
            
            return modifier + base_interpretation.lower()
        
        return f"Total sale da {opening:.2f} a {closing:.2f}, più gol attesi"
    
    def _get_down_interpretation(self, opening: float, closing: float) -> str:
        """Interpretazione per total che scende con interpolazione"""
        best_match = None
        min_distance = float('inf')
        
        for start, (end, interpretation) in self.TOTAL_MOVEMENTS_DOWN.items():
            # Match esatto
            if opening >= start and closing <= end:
                return interpretation
            
            # Calcola distanza per interpolazione
            if opening >= start:
                distance = abs(closing - end) if closing > end else 0
            else:
                distance = abs(opening - start)
            
            if distance < min_distance:
                min_distance = distance
                best_match = (start, end, interpretation)
        
        # Se trovato un match vicino, modifica l'interpretazione
        if best_match and min_distance < 0.5:
            start, end, base_interpretation = best_match
            movement_completion = abs(closing - end) / abs(start - end) if abs(start - end) > 0 else 1.0
            
            if movement_completion < 0.3:
                modifier = "parzialmente "
            elif movement_completion > 0.7:
                modifier = ""
            else:
                modifier = "progressivamente "
            
            return modifier + base_interpretation.lower()
        
        return f"Total scende da {opening:.2f} a {closing:.2f}, meno gol attesi"


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
        
        # Determina chi è favorito in base al segno dello spread
        # spread = lambda_a - lambda_h (come in Frontendcloud.py)
        # spread > 0: Trasferta favorita (2) - lambda_a > lambda_h
        # spread < 0: Casa favorita (1) - lambda_a < lambda_h
        favorito = "2" if spread.closing_value > 0 else "1" if spread.closing_value < 0 else "X"

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
            # Spread si ammorbidisce → guarda closing value + movimento total con THRESHOLDS GRANULARI
            if abs_spread < 0.5:
                # Match molto equilibrato → considera movimento total
                if total.direction == MovementDirection.HARDEN:
                    # Total sale → match più aperto → meno probabile pareggio
                    recommendations.append(MarketRecommendation(
                        market_name="1X2",
                        recommendation="12 (Risultato incerto, evita X)",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Spread equilibrato ({spread.closing_value:.2f}) ma total sale ({total.closing_value:.2f}), match aperto"
                    ))
                else:
                    # Total stabile o scende → pareggio più probabile
                    recommendations.append(MarketRecommendation(
                        market_name="1X2",
                        recommendation="X (Pareggio) o X2",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Spread equilibrato ({spread.closing_value:.2f}), favorito molto debole"
                    ))
            elif abs_spread < 0.75:
                # Match abbastanza equilibrato - threshold più granulare
                favorito_x = f"{favorito}X" if favorito != "X" else "X"
                # Se total sale molto, match più aperto → meno probabile X
                if total.direction == MovementDirection.HARDEN and total.intensity == MovementIntensity.STRONG:
                    recommendation_text = f"{favorito_x} (Evita X, match aperto)"
                else:
                    recommendation_text = f"X o {favorito_x} (Incertezza, evita underdog)"
                recommendations.append(MarketRecommendation(
                    market_name="1X2",
                    recommendation=recommendation_text,
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

        # Over/Under - LOGICA MIGLIORATA: considera movimento, opening vs closing, e soglie granulari
        # Normalizza il total closing a linea discreta per la raccomandazione
        total_line = round(total.closing_value * 4) / 4
        
        if total.direction == MovementDirection.HARDEN:
            # Total sale → Over (movimento verso più gol)
            # Calcola quanto è aumentato rispetto all'opening
            movement_magnitude = total.closing_value - total.opening_value
            
            # Se aumento significativo (> 0.5) → HIGH confidence
            if total.intensity == MovementIntensity.STRONG or movement_magnitude >= 0.5:
                conf = ConfidenceLevel.HIGH
            else:
                conf = ConfidenceLevel.MEDIUM
            
            recommendations.append(MarketRecommendation(
                market_name="Over/Under",
                recommendation=f"Over {total_line:.2f}",
                confidence=conf,
                explanation=f"Total sale da {total.opening_value:.2f} a {total.closing_value:.2f}: {total.interpretation}"
            ))
        elif total.direction == MovementDirection.SOFTEN:
            # Total scende → decisione basata su closing value e intensità movimento
            movement_magnitude = total.opening_value - total.closing_value
            
            # ZONA ALTA (>= 2.75): Anche se scende, è ancora alto
            if total.closing_value >= 2.75:
                # Se scende poco → Over, se scende molto → Under con cautela
                if movement_magnitude < 0.5:
                    recommendations.append(MarketRecommendation(
                        market_name="Over/Under",
                        recommendation=f"Over {total_line:.2f}",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Total scende ma {total.closing_value:.2f} ancora alto, partita aperta"
                    ))
                else:
                    # Scende molto → Under con MEDIUM confidence
                    recommendations.append(MarketRecommendation(
                        market_name="Over/Under",
                        recommendation=f"Under {total_line:.2f}",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Total scende significativamente da {total.opening_value:.2f} a {total.closing_value:.2f}"
                    ))
            # ZONA INTERMEDIA (2.25-2.75): Dipende da intensità movimento
            elif total.closing_value >= 2.25:
                if total.intensity == MovementIntensity.STRONG or movement_magnitude >= 0.5:
                    # Scende molto → Under
                    conf = ConfidenceLevel.HIGH if total.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
                    recommendations.append(MarketRecommendation(
                        market_name="Over/Under",
                        recommendation=f"Under {total_line:.2f}",
                        confidence=conf,
                        explanation=f"Total scende: {total.interpretation}"
                    ))
                else:
                    # Scende poco → Over ancora possibile
                    recommendations.append(MarketRecommendation(
                        market_name="Over/Under",
                        recommendation=f"Over {total_line:.2f}",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Total scende poco, ancora sopra media ({total.closing_value:.2f})"
                    ))
            # ZONA BASSA (< 2.25): Under
            else:
                conf = ConfidenceLevel.HIGH if total.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
                recommendations.append(MarketRecommendation(
                    market_name="Over/Under",
                    recommendation=f"Under {total_line:.2f}",
                    confidence=conf,
                    explanation=f"Total basso e scende: {total.interpretation}"
                ))
        else:  # STABLE
            # Total stabile → guarda closing value con soglie granulari
            if total.closing_value >= 2.75:
                recommendations.append(MarketRecommendation(
                    market_name="Over/Under",
                    recommendation=f"Over {total_line:.2f}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Total alto e stabile ({total.closing_value:.2f}), partita aperta attesa"
                ))
            elif total.closing_value >= 2.5:
                # Zona intermedia alta: Over con MEDIUM confidence
                recommendations.append(MarketRecommendation(
                    market_name="Over/Under",
                    recommendation=f"Over {total_line:.2f}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Total medio-alto e stabile ({total.closing_value:.2f}), partita abbastanza aperta"
                ))
            elif total.closing_value <= 2.0:
                # Zona bassa: Under
                recommendations.append(MarketRecommendation(
                    market_name="Over/Under",
                    recommendation=f"Under {total_line:.2f}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Total basso e stabile ({total.closing_value:.2f}), partita tattica attesa"
                ))
            else:
                # Zona intermedia (2.0-2.5): Neutrale, dipende da spread
                # Se spread si ammorbidisce → match più equilibrato → Over più probabile
                if spread.direction == MovementDirection.SOFTEN and abs_spread < 1.0:
                    recommendations.append(MarketRecommendation(
                        market_name="Over/Under",
                        recommendation=f"Over {total_line:.2f}",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Total medio ({total.closing_value:.2f}), match equilibrato favorisce gol"
                    ))
                else:
                    recommendations.append(MarketRecommendation(
                        market_name="Over/Under",
                        recommendation=f"Under {total_line:.2f}",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Total medio ({total.closing_value:.2f}), partita più controllata"
                    ))

        # GOAL/NOGOAL - LOGICA MIGLIORATA: considera spread, total e zone intermedie
        # EDGE CASE: Total molto basso (< 1.75) - Partita chiusissima
        if total.closing_value < 1.75:
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation="NOGOAL + Under (Partita chiusissima)",
                confidence=ConfidenceLevel.HIGH,
                explanation=f"Total molto basso ({total.closing_value:.2f}), pochi/nessun gol atteso"
            ))
        # EDGE CASE: Total molto alto (> 3.5) - Goleada
        elif total.closing_value > 3.5:
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation="GOAL + Over (Goleada attesa)",
                confidence=ConfidenceLevel.HIGH,
                explanation=f"Total molto alto ({total.closing_value:.2f}), molti gol e GOAL sicuro"
            ))
        # ZONA ALTA (2.75-3.5): Partita aperta, probabile GOAL
        elif total.closing_value >= 2.75:
            # Considera anche spread: se si ammorbidisce molto, match più equilibrato = più probabile GOAL
            spread_contribution = 0.15 if spread.direction == MovementDirection.SOFTEN and abs_spread < 0.75 else 0.0
            conf = ConfidenceLevel.HIGH if (total.intensity == MovementIntensity.STRONG or total.direction == MovementDirection.HARDEN) else ConfidenceLevel.MEDIUM
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation="GOAL (Entrambe segnano)",
                confidence=conf,
                explanation=f"Partita aperta (total {total.closing_value:.2f}), entrambe le squadre segnano"
            ))
        # ZONA INTERMEDIA ALTA (2.5-2.75): Dipende da movimento e spread
        elif total.closing_value >= 2.5:
            # Se total sale o è stabile alto + spread equilibrato → GOAL
            if total.direction == MovementDirection.HARDEN or (total.direction == MovementDirection.STABLE and abs_spread < 1.0):
                conf = ConfidenceLevel.MEDIUM if total.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM
                recommendations.append(MarketRecommendation(
                    market_name="GOAL/NOGOAL",
                    recommendation="GOAL (Entrambe segnano)",
                    confidence=conf,
                    explanation=f"Total medio-alto ({total.closing_value:.2f}), partita abbastanza aperta"
                ))
            else:
                # Total scende o spread sbilanciato → NOGOAL possibile
                recommendations.append(MarketRecommendation(
                    market_name="GOAL/NOGOAL",
                    recommendation="NOGOAL (Almeno una non segna)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Total medio ({total.closing_value:.2f}), partita più chiusa"
                ))
        # ZONA INTERMEDIA BASSA (2.0-2.5): Dipende molto da movimento
        elif total.closing_value >= 2.0:
            # Se total scende significativamente o spread si indurisce molto → NOGOAL
            if (total.direction == MovementDirection.SOFTEN and total.intensity != MovementIntensity.LIGHT) or \
               (spread.direction == MovementDirection.HARDEN and spread.intensity == MovementIntensity.STRONG):
                recommendations.append(MarketRecommendation(
                    market_name="GOAL/NOGOAL",
                    recommendation="NOGOAL (Almeno una non segna)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Total medio-basso ({total.closing_value:.2f}), partita più tattica"
                ))
            else:
                # Total stabile o sale leggermente → GOAL possibile
                recommendations.append(MarketRecommendation(
                    market_name="GOAL/NOGOAL",
                    recommendation="GOAL (Entrambe segnano)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Total medio ({total.closing_value:.2f}), entrambe possono segnare"
                ))
        # ZONA BASSA (< 2.0): NOGOAL probabile
        else:
            recommendations.append(MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation="NOGOAL (Almeno una non segna)",
                confidence=ConfidenceLevel.HIGH if total.direction == MovementDirection.SOFTEN else ConfidenceLevel.MEDIUM,
                explanation=f"Total basso ({total.closing_value:.2f}), partita chiusa"
            ))

        # Handicap Asiatico - LOGICA MIGLIORATA: considera closing value e normalizza a linee discrete
        if spread.direction != MovementDirection.STABLE:
            # Normalizza handicap a linea discreta (multipli di 0.25)
            handicap_value_raw = abs_spread
            handicap_value = round(handicap_value_raw * 4) / 4
            
            # Se valore normalizzato è 0, usa 0.25 come minimo
            if handicap_value < 0.25 and abs_spread > 0.05:
                handicap_value = 0.25
            
            if spread.direction == MovementDirection.HARDEN:
                # Favorito si rafforza → gioca favorito con handicap
                # Se spread > 0, favorito è trasferta (2), quindi handicap positivo
                # Se spread < 0, favorito è casa (1), quindi handicap negativo
                handicap_sign = "+" if spread.closing_value > 0 else "-"
                recommendations.append(MarketRecommendation(
                    market_name="Handicap Asiatico",
                    recommendation=f"{favorito} {handicap_sign}{handicap_value}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Favorito copre spread {format_spread_display(spread.closing_value)}"
                ))
            else:  # SOFTEN
                # Spread si ammorbidisce
                if abs_spread >= 1.0:
                    # Ancora favorito forte → Favorito con handicap ridotto
                    handicap_sign = "+" if spread.closing_value > 0 else "-"
                    recommendations.append(MarketRecommendation(
                        market_name="Handicap Asiatico",
                        recommendation=f"{favorito} {handicap_sign}{handicap_value} (valore)",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Spread si ammorbidisce ma {format_spread_display(spread.closing_value)} ancora vantaggioso"
                    ))
                else:
                    # Match equilibrato → Underdog con handicap
                    # Se favorito è casa (1), underdog è trasferta (2) e viceversa
                    underdog = "2" if favorito == "1" else "1" if favorito == "2" else "X"
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
                                       total: MovementAnalysis, combination: Dict) -> List[MarketRecommendation]:
        """Calcola raccomandazioni alternative - Media confidenza"""
        recommendations = []

        abs_spread = abs(spread.closing_value)
        
        # Determina chi è favorito in base al segno dello spread
        # spread = lambda_a - lambda_h
        # spread > 0: Trasferta favorita (2)
        # spread < 0: Casa favorita (1)
        favorito = "2" if spread.closing_value > 0 else "1" if spread.closing_value < 0 else "X"

        # Combo favorito + Over/Under - LOGICA MIGLIORATA: considera closing value
        if spread.direction == MovementDirection.HARDEN:
            # Favorito si rafforza
            if total.direction == MovementDirection.HARDEN:
                recommendations.append(MarketRecommendation(
                    market_name="Combo",
                    recommendation=f"{favorito} + Over {total.closing_value}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Favorito vince con gol"
                ))
            elif total.closing_value >= 2.75:
                # Total scende ma ancora alto
                recommendations.append(MarketRecommendation(
                    market_name="Combo",
                    recommendation=f"{favorito} + Over {total.closing_value}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Favorito vince, total ancora alto"
                ))
            else:
                # Total basso
                recommendations.append(MarketRecommendation(
                    market_name="Combo",
                    recommendation=f"{favorito} + Under {total.closing_value}",
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
                        recommendation=f"{favorito} + Under {total.closing_value}",
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation=f"Favorito vince corto ({format_spread_display(spread.closing_value)})"
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

        # HT/FT Combinations - LOGICA MIGLIORATA: considera closing value e favorito reale
        if spread.direction == MovementDirection.HARDEN:
            # Favorito si rafforza
            if spread.intensity == MovementIntensity.STRONG:
                recommendations.append(MarketRecommendation(
                    market_name="HT/FT",
                    recommendation=f"{favorito}/{favorito} (Favorito HT e FT)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Favorito domina dall'inizio"
                ))
            else:
                recommendations.append(MarketRecommendation(
                    market_name="HT/FT",
                    recommendation=f"X/{favorito} (Pareggio HT, Favorito FT)",
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
                    recommendation=f"X/{favorito} (Pareggio HT, Favorito FT)",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Favorito decide nella ripresa ({spread.closing_value})"
                ))
            else:
                # Incertezza
                recommendations.append(MarketRecommendation(
                    market_name="HT/FT",
                    recommendation=f"X/X o X/{favorito}",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"Match incerto ({spread.closing_value})"
                ))
        else:  # STABLE
            # Spread stabile → guarda closing value
            if abs_spread >= 1.0:
                recommendations.append(MarketRecommendation(
                    market_name="HT/FT",
                    recommendation=f"X/{favorito} (Pareggio HT, Favorito FT)",
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

        # Over/Under HT - Stima migliorata
        ht_total_estimate = self._estimate_ht_total(total.closing_value, spread_analysis=spread, total_analysis=total)
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
        
        # Determina chi è favorito in base al segno dello spread
        # spread = lambda_a - lambda_h
        # spread > 0: Trasferta favorita (2)
        # spread < 0: Casa favorita (1)
        favorito = "2" if spread.closing_value > 0 else "1" if spread.closing_value < 0 else "X"

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
            favorito_x = f"{favorito}X" if favorito != "X" else "X"
            recommendations.append(MarketRecommendation(
                market_name="Double Chance",
                recommendation=f"{favorito_x} (Favorito o Pareggio)",
                confidence=ConfidenceLevel.LOW,
                explanation=f"Opzione sicura, favorito non perde ({format_spread_display(spread.closing_value)})"
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
    
    def _estimate_ht_total(self, total_value: float, spread_analysis: MovementAnalysis,
                          total_analysis: MovementAnalysis) -> float:
        """Stima più accurata del Total per primo tempo"""
        
        base_ht = total_value * 0.45  # Base: ~45% dei gol nel 1T (media storica)
        
        # Fattore spread: favorito forte → più gol 1T
        if spread_analysis.direction == MovementDirection.HARDEN:
            if spread_analysis.intensity == MovementIntensity.STRONG:
                spread_factor = 1.15  # +15% se favorito molto forte
            else:
                spread_factor = 1.08  # +8% se favorito medio
        elif spread_analysis.direction == MovementDirection.SOFTEN:
            spread_factor = 0.92  # -8% se favorito debole (partita più equilibrata)
        else:
            spread_factor = 1.0
        
        # Fattore total movement: se total sale → partita più viva → più gol 1T
        if total_analysis.direction == MovementDirection.HARDEN:
            total_factor = 1.10 if total_analysis.intensity == MovementIntensity.STRONG else 1.05
        elif total_analysis.direction == MovementDirection.SOFTEN:
            total_factor = 0.95  # Partita più tattica
        else:
            total_factor = 1.0
        
        # Valore base totale
        adjusted_ht = base_ht * spread_factor * total_factor
        
        # Normalizza ai valori di linea
        return round(adjusted_ht * 4) / 4
    
    def _calculate_confidence(self, spread: MovementAnalysis,
                             total: MovementAnalysis) -> ConfidenceLevel:
        """Calcola confidenza generale pesata per intensità e ampiezza"""

        # Pesi per intensità
        intensity_weights = {
            MovementIntensity.NONE: 0.0,
            MovementIntensity.LIGHT: 0.3,
            MovementIntensity.MEDIUM: 0.6,
            MovementIntensity.STRONG: 1.0
        }
        
        spread_weight = intensity_weights.get(spread.intensity, 0.5)
        total_weight = intensity_weights.get(total.intensity, 0.5)
        
        # Fattore ampiezza: movimento più ampio = più confidenza
        spread_amplitude = min(abs(spread.movement_steps) / 4.0, 1.0)  # Normalizza a max 4 step
        total_amplitude = min(abs(total.movement_steps) / 4.0, 1.0)
        
        # Peso combinato (spread più importante)
        combined_weight = (spread_weight * 0.6 + total_weight * 0.4)
        amplitude_factor = (spread_amplitude * 0.6 + total_amplitude * 0.4)
        
        # Score finale (0-1)
        confidence_score = combined_weight * 0.7 + amplitude_factor * 0.3
        
        # Penalità per segnali contrastanti
        if (spread.direction == MovementDirection.HARDEN and total.direction == MovementDirection.SOFTEN) or \
           (spread.direction == MovementDirection.SOFTEN and total.direction == MovementDirection.HARDEN):
            confidence_score *= 0.65  # Riduce confidenza del 35%
        
        # Bonus per concordi e forti
        if (spread.direction == total.direction and
            spread.direction != MovementDirection.STABLE and
            spread.intensity in [MovementIntensity.MEDIUM, MovementIntensity.STRONG] and
            total.intensity in [MovementIntensity.MEDIUM, MovementIntensity.STRONG]):
            confidence_score = min(confidence_score * 1.15, 1.0)  # Boost del 15% (max 1.0)
        
        # Penalità per entrambi STABLE
        if (spread.direction == MovementDirection.STABLE and 
            total.direction == MovementDirection.STABLE):
            confidence_score *= 0.5  # Riduce confidenza del 50%
        
        # Conversione a ConfidenceLevel
        if confidence_score >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.45:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


def format_spread_display(spread_value: float) -> str:
    """
    Formatta lo spread per la visualizzazione corretta.
    - (-) indica casa (home)
    - (+) indica trasferta (away)
    
    Args:
        spread_value: Valore spread (lambda_h - lambda_a)
    
    Returns:
        Stringa formattata con segno corretto
    """
    if spread_value is None:
        return "N/A"
    
    # Spread viene calcolato come lambda_a - lambda_h (come in Frontendcloud.py)
    # - spread > 0: Trasferta favorita → mostra (+)
    # - spread < 0: Casa favorita → mostra (-)
    # Quindi il segno è già corretto per la visualizzazione
    if spread_value > 0:
        return f"+{abs(spread_value):.2f}"
    elif spread_value < 0:
        return f"-{abs(spread_value):.2f}"
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

