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

# Import classi dal modulo principale (CORRETTE E AGGIORNATE)
from market_movement_analyzer import (
    MovementDirection,
    MovementIntensity,
    ConfidenceLevel,
    MovementAnalysis,
    MarketRecommendation,
    AnalysisResult,
    MarketMovementAnalyzer,
    format_spread_display
)


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
        
        # CORE RECOMMENDATIONS
        st.header("🎯 Raccomandazioni CORE")
        st.caption("Alta/Media confidenza - I consigli più solidi")

        if result.core_recommendations:
            for i, rec in enumerate(result.core_recommendations, 1):
                render_recommendation(rec, i)
        else:
            st.warning("Nessuna raccomandazione core disponibile")

        st.markdown("---")

        # ALTERNATIVE RECOMMENDATIONS
        if result.alternative_recommendations:
            st.header("💼 Opzioni Alternative")
            st.caption("Media confidenza - Opzioni tattiche")
            for i, rec in enumerate(result.alternative_recommendations, 1):
                render_recommendation(rec, i)
            st.markdown("---")

        # VALUE BETS
        if result.value_recommendations:
            st.header("💎 Value Bets")
            st.caption("Bassa confidenza ma potenziale valore - Per chi vuole rischiare")
            for i, rec in enumerate(result.value_recommendations, 1):
                render_recommendation(rec, i)
            st.markdown("---")

        # EXCHANGE RECOMMENDATIONS (Punta/Banca)
        if result.exchange_recommendations:
            st.header("💱 Exchange - Punta/Banca")
            st.caption("Consigli per mercato Exchange (Back/Lay)")
            for i, rec in enumerate(result.exchange_recommendations, 1):
                render_recommendation(rec, i)

        st.markdown("---")

        # Expected Goals (xG) e Probabilità
        st.header("📊 Expected Goals (xG) & Probabilità")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "xG Casa",
                f"{result.expected_goals.home_xg:.2f}",
                help="Expected Goals squadra casa (calcolati da spread e total)"
            )
            st.metric(
                "P(Casa vince)",
                f"{result.expected_goals.home_win_prob:.1%}",
                help="Probabilità vittoria casa (Poisson)"
            )

        with col2:
            st.metric(
                "xG Trasferta",
                f"{result.expected_goals.away_xg:.2f}",
                help="Expected Goals squadra trasferta (calcolati da spread e total)"
            )
            st.metric(
                "P(Trasferta vince)",
                f"{result.expected_goals.away_win_prob:.1%}",
                help="Probabilità vittoria trasferta (Poisson)"
            )

        with col3:
            st.metric(
                "P(Pareggio)",
                f"{result.expected_goals.draw_prob:.1%}",
                help="Probabilità pareggio (Poisson)"
            )
            st.metric(
                "P(BTTS)",
                f"{result.expected_goals.btts_prob:.1%}",
                help="Probabilità Both Teams To Score"
            )

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

