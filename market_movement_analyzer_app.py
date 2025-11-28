#!/usr/bin/env python3
"""
Market Movement Analyzer - Streamlit App
=========================================

App Streamlit per analizzare i movimenti di Spread e Total
e generare interpretazioni e giocate consigliate.

Usage:
    streamlit run market_movement_analyzer_app.py
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


# Configurazione pagina Streamlit
st.set_page_config(
    page_title="Market Movement Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-high {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-medium {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-low {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .movement-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


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
    
    def _get_soften_interpretation(self, opening: float, closing: float) -> str:
        """Ottiene interpretazione per spread che si ammorbidisce"""
        for start, (end, interpretation) in self.SPREAD_MOVEMENTS_DOWN.items():
            if opening >= start and closing <= end:
                return interpretation
        
        if closing > opening:
            return f"Fiducia nel favorito cala da {opening} a {closing}"
        return "Spread si ammorbidisce, mercato più equilibrato"
    
    def _get_harden_interpretation(self, opening: float, closing: float) -> str:
        """Ottiene interpretazione per spread che si indurisce"""
        for start, (end, interpretation) in self.SPREAD_MOVEMENTS_UP.items():
            if opening >= start and closing <= end:
                return interpretation
        
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
        movement = closing - opening
        
        if abs(movement) < 0.01:
            direction = MovementDirection.STABLE
            intensity = MovementIntensity.NONE
            interpretation = "Total stabile, ritmo atteso invariato"
        elif movement > 0:
            direction = MovementDirection.HARDEN
            interpretation = self._get_up_interpretation(opening, closing)
        else:
            direction = MovementDirection.SOFTEN
            interpretation = self._get_down_interpretation(opening, closing)
        
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
        
        spread_analysis = self.spread_analyzer.analyze(spread_open, spread_close)
        total_analysis = self.total_analyzer.analyze(total_open, total_close)
        
        spread_dir_key = spread_analysis.direction.name if spread_analysis.direction != MovementDirection.STABLE else "STABLE"
        total_dir_key = total_analysis.direction.name if total_analysis.direction != MovementDirection.STABLE else "STABLE"
        
        combination = self._get_combination_interpretation(
            spread_analysis, total_analysis, spread_dir_key, total_dir_key
        )
        
        ft_markets = self._calculate_ft_markets(spread_analysis, total_analysis, combination)
        ht_markets = self._calculate_ht_markets(spread_analysis, total_analysis, total_close)
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
        
        key = (spread_key, total_key)
        if key in self.COMBINATION_MATRIX:
            combo = self.COMBINATION_MATRIX[key]
            return {
                "interpretation": combo["meaning"],
                "tendency": combo["goals_tendency"],
                "recommendations": combo["recommendations"]
            }
        
        return {
            "interpretation": f"{spread.interpretation}. {total.interpretation}",
            "tendency": "Neutra"
        }
    
    def _calculate_ft_markets(self, spread: MovementAnalysis, 
                             total: MovementAnalysis, combination: Dict) -> List[MarketRecommendation]:
        """Calcola mercati FT consigliati"""
        recommendations = []
        
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
        
        if "recommendations" in combination:
            for rec_name in combination["recommendations"]:
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
        
        ht_total_estimate = total_close * 0.5
        ht_spread_estimate = spread.closing_value * 0.5
        
        if spread.direction == MovementDirection.SOFTEN:
            ht_spread_estimate = ht_spread_estimate * 0.8
        
        if total.direction == MovementDirection.HARDEN:
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
        
        if (spread.direction == total.direction and 
            spread.intensity in [MovementIntensity.MEDIUM, MovementIntensity.STRONG] and
            total.intensity in [MovementIntensity.MEDIUM, MovementIntensity.STRONG]):
            return ConfidenceLevel.HIGH
        
        if spread.direction == total.direction:
            return ConfidenceLevel.MEDIUM
        
        if spread.direction != MovementDirection.STABLE and total.direction != MovementDirection.STABLE:
            if spread.direction != total.direction:
                return ConfidenceLevel.LOW
        
        return ConfidenceLevel.MEDIUM


def render_movement_box(label: str, analysis: MovementAnalysis):
    """Renderizza una box per mostrare movimento"""
    direction_icon = analysis.direction.value
    intensity_text = analysis.intensity.value
    
    st.markdown(f"""
    <div class="movement-box">
        <strong>{label}</strong><br>
        {analysis.opening_value:.2f} → {analysis.closing_value:.2f} {direction_icon}<br>
        <em>{intensity_text}</em>
    </div>
    """, unsafe_allow_html=True)


def render_recommendation(rec: MarketRecommendation, index: int):
    """Renderizza una raccomandazione"""
    conf_class = {
        ConfidenceLevel.HIGH: "recommendation-high",
        ConfidenceLevel.MEDIUM: "recommendation-medium",
        ConfidenceLevel.LOW: "recommendation-low"
    }.get(rec.confidence, "recommendation-medium")
    
    conf_icon = {
        ConfidenceLevel.HIGH: "🟢",
        ConfidenceLevel.MEDIUM: "🟡",
        ConfidenceLevel.LOW: "🔴"
    }.get(rec.confidence, "🟡")
    
    st.markdown(f"""
    <div class="{conf_class}">
        <strong>{conf_icon} {rec.market_name}:</strong> {rec.recommendation}<br>
        <small>{rec.explanation}</small>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Funzione principale Streamlit"""
    
    # Header
    st.markdown('<div class="main-header">📊 Market Movement Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analizza movimenti Spread e Total per generare giocate consigliate</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar con info
    with st.sidebar:
        st.header("ℹ️ Info")
        st.markdown("""
        Questo tool analizza i movimenti di **Spread** e **Total** 
        per fornire interpretazioni e giocate consigliate basate 
        su pattern di mercato consolidati.
        
        ### Come usare:
        1. Inserisci Spread apertura e chiusura
        2. Inserisci Total apertura e chiusura
        3. Clicca "Analizza" per vedere i risultati
        
        ### Valori tipici:
        - **Spread**: da -2.0 a +2.0
        - **Total**: da 1.5 a 4.0
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Esempi")
        
        if st.button("Esempio 1: Favorito forte + Partita viva"):
            st.session_state.spread_open = -1.5
            st.session_state.spread_close = -1.75
            st.session_state.total_open = 2.5
            st.session_state.total_close = 2.75
        
        if st.button("Esempio 2: Favorito cala + Partita chiusa"):
            st.session_state.spread_open = -1.5
            st.session_state.spread_close = -1.0
            st.session_state.total_open = 2.75
            st.session_state.total_close = 2.5
    
    # Form input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Spread")
        spread_open = st.number_input(
            "Spread Apertura",
            min_value=-3.0,
            max_value=3.0,
            value=st.session_state.get('spread_open', -1.5),
            step=0.25,
            key='spread_open_input'
        )
        spread_close = st.number_input(
            "Spread Chiusura",
            min_value=-3.0,
            max_value=3.0,
            value=st.session_state.get('spread_close', -1.0),
            step=0.25,
            key='spread_close_input'
        )
    
    with col2:
        st.subheader("⚽ Total")
        total_open = st.number_input(
            "Total Apertura",
            min_value=1.0,
            max_value=5.0,
            value=st.session_state.get('total_open', 2.5),
            step=0.25,
            key='total_open_input'
        )
        total_close = st.number_input(
            "Total Chiusura",
            min_value=1.0,
            max_value=5.0,
            value=st.session_state.get('total_close', 2.75),
            step=0.25,
            key='total_close_input'
        )
    
    st.markdown("---")
    
    # Bottone analisi
    if st.button("🔍 Analizza Movimenti", type="primary", use_container_width=True):
        
        analyzer = MarketMovementAnalyzer()
        
        with st.spinner("🔄 Analisi in corso..."):
            result = analyzer.analyze(spread_open, spread_close, total_open, total_close)
        
        # Sezione Movimenti
        st.header("📈 Analisi Movimenti")
        
        col1, col2 = st.columns(2)
        with col1:
            render_movement_box("Spread", result.spread_analysis)
            st.caption(f"*{result.spread_analysis.interpretation}*")
        
        with col2:
            render_movement_box("Total", result.total_analysis)
            st.caption(f"*{result.total_analysis.interpretation}*")
        
        st.markdown("---")
        
        # Interpretazione combinata
        st.header("🎯 Interpretazione Combinata")
        
        conf_color = {
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🔴"
        }.get(result.overall_confidence, "🟡")
        
        st.info(f"**{result.combination_interpretation}**")
        st.metric("Confidenza", f"{conf_color} {result.overall_confidence.value}")
        
        st.markdown("---")
        
        # Mercati FT
        st.header("💡 Mercati FT Consigliati")
        
        if result.ft_markets:
            for i, rec in enumerate(result.ft_markets, 1):
                render_recommendation(rec, i)
        else:
            st.warning("Nessuna raccomandazione FT disponibile")
        
        st.markdown("---")
        
        # Mercati HT
        if result.ht_markets:
            st.header("💡 Mercati HT Consigliati")
            for i, rec in enumerate(result.ht_markets, 1):
                render_recommendation(rec, i)
        
        st.markdown("---")
        
        # Dettagli tecnici (espandibile)
        with st.expander("🔧 Dettagli Tecnici"):
            st.write("**Spread Analysis:**")
            st.json({
                "direction": result.spread_analysis.direction.name,
                "intensity": result.spread_analysis.intensity.value,
                "movement_steps": round(result.spread_analysis.movement_steps, 2)
            })
            
            st.write("**Total Analysis:**")
            st.json({
                "direction": result.total_analysis.direction.name,
                "intensity": result.total_analysis.intensity.value,
                "movement_steps": round(result.total_analysis.movement_steps, 2)
            })


if __name__ == "__main__":
    main()

