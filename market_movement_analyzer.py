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
    ft_markets: List[MarketRecommendation]
    ht_markets: List[MarketRecommendation]
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
        
        # Calcola mercati FT
        ft_markets = self._calculate_ft_markets(spread_analysis, total_analysis, combination)
        
        # Calcola mercati HT (derivati da FT)
        ht_markets = self._calculate_ht_markets(spread_analysis, total_analysis, total_close)
        
        # Calcola confidenza generale
        overall_confidence = self._calculate_confidence(spread_analysis, total_analysis)
        
        return AnalysisResult(
            spread_analysis=spread_analysis,
            total_analysis=total_analysis,
            combination_interpretation=combination["interpretation"],
            ft_markets=ft_markets,
            ht_markets=ht_markets,
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
    
    def _calculate_ft_markets(self, spread: MovementAnalysis, 
                             total: MovementAnalysis, combination: Dict) -> List[MarketRecommendation]:
        """Calcola mercati FT consigliati"""
        recommendations = []
        
        # Mercati 1X2 basati su spread
        if spread.direction == MovementDirection.HARDEN:
            rec = MarketRecommendation(
                market_name="1X2",
                recommendation="1 (Favorito)",
                confidence=ConfidenceLevel.MEDIUM if spread.intensity == MovementIntensity.MEDIUM else ConfidenceLevel.HIGH,
                explanation=spread.interpretation
            )
            recommendations.append(rec)
        elif spread.direction == MovementDirection.SOFTEN and spread.closing_value < -0.5:
            rec = MarketRecommendation(
                market_name="1X2",
                recommendation="X2 o Underdog + Handicap",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Favorito vulnerabile"
            )
            recommendations.append(rec)
        
        # Over/Under basato su total
        if total.direction == MovementDirection.HARDEN:
            rec = MarketRecommendation(
                market_name=f"Over/Under",
                recommendation=f"Over {total.closing_value}",
                confidence=ConfidenceLevel.HIGH if total.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM,
                explanation=total.interpretation
            )
            recommendations.append(rec)
        elif total.direction == MovementDirection.SOFTEN:
            rec = MarketRecommendation(
                market_name=f"Over/Under",
                recommendation=f"Under {total.closing_value}",
                confidence=ConfidenceLevel.HIGH if total.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM,
                explanation=total.interpretation
            )
            recommendations.append(rec)
        
        # GOAL/NOGOAL
        if total.direction == MovementDirection.HARDEN or (combination.get("tendency", "").find("GOAL") != -1):
            rec = MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation="GOAL",
                confidence=ConfidenceLevel.HIGH if total.intensity == MovementIntensity.STRONG else ConfidenceLevel.MEDIUM,
                explanation="Partita viva, gol da entrambe"
            )
            recommendations.append(rec)
        elif total.direction == MovementDirection.SOFTEN or (combination.get("tendency", "").find("NOGOAL") != -1):
            rec = MarketRecommendation(
                market_name="GOAL/NOGOAL",
                recommendation="NOGOAL",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Ritmo basso, partita chiusa"
            )
            recommendations.append(rec)
        
        # Aggiungi raccomandazioni dalla matrice
        if "recommendations" in combination:
            for rec_name in combination["recommendations"]:
                # Evita duplicati
                if not any(r.market_name == rec_name or rec_name in r.recommendation for r in recommendations):
                    recommendations.append(MarketRecommendation(
                        market_name="Alternativa",
                        recommendation=rec_name,
                        confidence=ConfidenceLevel.MEDIUM,
                        explanation="Dalla matrice combinazione"
                    ))
        
        return recommendations
    
    def _calculate_ht_markets(self, spread: MovementAnalysis,
                              total: MovementAnalysis, total_close: float) -> List[MarketRecommendation]:
        """Calcola mercati HT derivati da FT"""
        recommendations = []
        
        # Formula base: HT_Total ≈ FT_Total × 0.5
        ht_total_estimate = total_close * 0.5
        
        # Spread HT ≈ FT_Spread × 0.5, ma aggiungi logica
        ht_spread_estimate = spread.closing_value * 0.5
        
        # Se spread si ammorbidisce, HT tende più al pareggio
        if spread.direction == MovementDirection.SOFTEN:
            ht_spread_estimate = ht_spread_estimate * 0.8  # Riduci l'handicap
        
        # Over/Under HT
        if total.direction == MovementDirection.HARDEN:
            # Total sale → HT più gol
            if ht_total_estimate >= 1.0:
                rec = MarketRecommendation(
                    market_name="Over/Under HT",
                    recommendation=f"Over 1.0 HT o Over 1.25 HT",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=f"FT Total {total_close} → HT stimato {ht_total_estimate:.2f}, partenza aggressiva"
                )
            else:
                rec = MarketRecommendation(
                    market_name="Over/Under HT",
                    recommendation="Over 0.75 HT",
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation="Total FT in aumento, ritmo crescente nel 1°T"
                )
            recommendations.append(rec)
        elif total.direction == MovementDirection.SOFTEN:
            rec = MarketRecommendation(
                market_name="Over/Under HT",
                recommendation="Under 1.0 HT",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Total FT in calo, fase di studio probabile"
            )
            recommendations.append(rec)
        
        # GOAL/NOGOAL HT
        if total.direction == MovementDirection.HARDEN and ht_total_estimate >= 1.0:
            rec = MarketRecommendation(
                market_name="GOAL HT",
                recommendation="GOAL 1°T",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Almeno 1 gol atteso nel primo tempo"
            )
            recommendations.append(rec)
        elif total.direction == MovementDirection.SOFTEN:
            rec = MarketRecommendation(
                market_name="GOAL HT",
                recommendation="NOGOAL 1°T o X HT",
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Ritmo basso, partita tattica"
            )
            recommendations.append(rec)
        
        return recommendations
    
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
    
    # Mercati FT
    output.append("💡 MERCATI FT CONSIGLIATI:")
    for i, market in enumerate(result.ft_markets, 1):
        conf_icon = "🟢" if market.confidence == ConfidenceLevel.HIGH else "🟡" if market.confidence == ConfidenceLevel.MEDIUM else "🔴"
        output.append(f"   {i}️⃣  {conf_icon} {market.market_name}: {market.recommendation}")
        output.append(f"      └─ {market.explanation}")
    output.append("")
    
    # Mercati HT
    if result.ht_markets:
        output.append("💡 MERCATI HT CONSIGLIATI:")
        for i, market in enumerate(result.ht_markets, 1):
            conf_icon = "🟢" if market.confidence == ConfidenceLevel.HIGH else "🟡" if market.confidence == ConfidenceLevel.MEDIUM else "🔴"
            output.append(f"   {i}️⃣  {conf_icon} {market.market_name}: {market.recommendation}")
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

