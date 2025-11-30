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

        # ============== ADVANCED MARKET INTELLIGENCE ==============
        st.header("🔬 Advanced Market Intelligence")
        st.caption("Analisi avanzata basata su pattern professionali")

        intel = result.market_intelligence

        # Summary box
        summary_signals = []
        if intel.sharp_money_detected:
            summary_signals.append("🟢 Sharp Money")
        if intel.steam_move_detected:
            summary_signals.append("🔥 Steam Move")
        if intel.contrarian_signal:
            summary_signals.append("⚡ Contrarian")
        if intel.on_key_spread or intel.on_key_total:
            summary_signals.append("🎯 Key Number")

        if summary_signals:
            st.info(f"**Segnali rilevati:** {' • '.join(summary_signals)}")

        # Tabs per organizzare meglio
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💰 Sharp Money",
            "🚂 Steam Move",
            "📊 Correlation",
            "🔢 Key Numbers",
            "💯 Efficiency"
        ])

        with tab1:
            # Sharp Money
            if intel.sharp_money_detected:
                st.success("🟢 **SHARP MONEY DETECTED**")
            else:
                st.info("⚪ Movimento normale (public money)")

            col1, col2 = st.columns(2)
            with col1:
                velocity_color = "🟢" if intel.sharp_spread_velocity > 15 else "🟡" if intel.sharp_spread_velocity > 8 else "⚪"
                st.metric("Spread Velocity", f"{velocity_color} {intel.sharp_spread_velocity:.1f}%")
            with col2:
                velocity_color = "🟢" if intel.sharp_total_velocity > 10 else "🟡" if intel.sharp_total_velocity > 5 else "⚪"
                st.metric("Total Velocity", f"{velocity_color} {intel.sharp_total_velocity:.1f}%")

            if intel.contrarian_signal:
                st.warning("⚡ **CONTRARIAN SIGNAL**: Spread e Total si muovono in direzioni opposte - Forte segnale professionale!")

            if intel.sharp_confidence_boost > 0:
                st.success(f"✅ Confidence Boost: +{intel.sharp_confidence_boost*100:.0f}%")

        with tab2:
            # Steam Move
            if intel.steam_move_detected:
                st.error("🚨 **STEAM MOVE DETECTED!**")
                st.markdown(f"**Magnitude:** {intel.steam_magnitude:.2f} punti")
                st.markdown(f"**Direction:** {intel.steam_direction.upper()}")

                if intel.reverse_steam:
                    st.warning("🔄 **REVERSE STEAM**: Il favorito è cambiato!")

                st.info("""
                💡 **Azione consigliata:**
                Movimento massiccio di denaro istituzionale. Considera di seguire la direzione
                prima che la quota peggiori ulteriormente.
                """)
            else:
                st.success(f"✅ Nessun Steam Move rilevato (movimento: {intel.steam_magnitude:.2f} punti)")
                st.caption("Il movimento è nella norma")

        with tab3:
            # Correlation
            score_color = "🟢" if intel.correlation_score > 0.5 else "🔴" if intel.correlation_score < -0.5 else "🟡"
            st.metric("Correlation Score", f"{score_color} {intel.correlation_score:+.2f}")

            if intel.market_coherent:
                st.success("✅ **MERCATO COERENTE**")
            else:
                st.warning("⚠️ **SEGNALI CONTRASTANTI**")

            st.info(f"**Interpretazione:** {intel.correlation_interpretation}")

            # Visual gauge
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=intel.correlation_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "lightcoral"},
                        {'range': [-0.5, 0.5], 'color': "lightyellow"},
                        {'range': [0.5, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            # Key Numbers
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Spread")
                if intel.on_key_spread:
                    st.success(f"🟢 **ON KEY NUMBER**: {intel.spread_key_number}")
                    st.caption("Linea più affidabile possibile")
                else:
                    st.info(f"📍 Nearest: {intel.spread_key_number if intel.spread_key_number else 'N/A'}")

            with col2:
                st.subheader("Total")
                if intel.on_key_total:
                    st.success(f"🟢 **ON KEY NUMBER**: {intel.total_key_number}")
                    st.caption("Linea più affidabile possibile")
                else:
                    st.info(f"📍 Nearest: {intel.total_key_number if intel.total_key_number else 'N/A'}")

            if intel.key_confidence_boost > 0:
                st.success(f"✅ Confidence Boost: +{intel.key_confidence_boost*100:.0f}%")

        with tab5:
            # Market Efficiency
            efficiency_color = "🟢" if intel.efficiency_score >= 90 else "🟡" if intel.efficiency_score >= 70 else "🔴"
            st.metric("Efficiency Score", f"{efficiency_color} {intel.efficiency_score:.0f}/100")

            # Progress bar
            st.progress(intel.efficiency_score / 100)

            status_color = {
                "Efficient": "success",
                "Normal": "info",
                "Inefficient": "warning"
            }.get(intel.efficiency_status, "info")

            getattr(st, status_color)(f"**Status:** {intel.efficiency_status.upper()}")

            if intel.value_opportunity:
                st.warning("💎 **VALUE OPPORTUNITY DETECTED**: Mercato inefficiente - possibili value bets!")
            else:
                st.info("Mercato efficiente - prezzi accurati")

        st.markdown("---")
        # ==========================================================

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

        # ============== OPZIONE C: ADVANCED PREDICTIONS ==============
        st.header("🎯 Opzione C: Advanced Predictions")
        st.caption("Predizioni avanzate con Bayesian Inference + Monte Carlo")

        xg = result.expected_goals

        # CONFIDENCE SCORE GAUGE
        if xg.confidence_score is not None:
            st.subheader("📊 Market Confidence Score")

            col1, col2 = st.columns([2, 1])

            with col1:
                # Gauge visuale
                import plotly.graph_objects as go

                conf_score = xg.confidence_score

                # Colore basato su score
                if conf_score >= 80:
                    bar_color = "green"
                elif conf_score >= 60:
                    bar_color = "orange"
                else:
                    bar_color = "red"

                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=conf_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Affidabilità Predizione"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': bar_color},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 70], 'color': "lightyellow"},
                            {'range': [70, 85], 'color': "lightgreen"},
                            {'range': [85, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "darkblue", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("Score", f"{conf_score:.0f}/100")

                if conf_score >= 80:
                    st.success("🟢 **ALTA AFFIDABILITÀ**")
                    st.caption("Predizione molto affidabile")
                elif conf_score >= 60:
                    st.warning("🟡 **MEDIA AFFIDABILITÀ**")
                    st.caption("Predizione moderatamente affidabile")
                else:
                    st.error("🔴 **BASSA AFFIDABILITÀ**")
                    st.caption("Usare con cautela")

                st.caption(f"Metodo: {xg.prediction_method or 'N/A'}")

        # 1X2 MARKET-ADJUSTED vs BASE
        if xg.market_adjusted_1x2 is not None:
            st.subheader("📈 1X2 Market-Adjusted (Bayesian Ensemble)")

            ma = xg.market_adjusted_1x2

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Home Win",
                    f"{ma['home_win']:.1%}",
                    help="Probabilità vittoria casa (Bayesian + Ensemble)"
                )
                if ma.get('base_probs'):
                    delta = ma['home_win'] - ma['base_probs']['home']
                    st.caption(f"Base xG: {ma['base_probs']['home']:.1%} ({delta:+.1%})")

            with col2:
                st.metric(
                    "Draw",
                    f"{ma['draw']:.1%}",
                    help="Probabilità pareggio"
                )
                if ma.get('base_probs'):
                    delta = ma['draw'] - ma['base_probs']['draw']
                    st.caption(f"Base xG: {ma['base_probs']['draw']:.1%} ({delta:+.1%})")

            with col3:
                st.metric(
                    "Away Win",
                    f"{ma['away_win']:.1%}",
                    help="Probabilità vittoria trasferta"
                )
                if ma.get('base_probs'):
                    delta = ma['away_win'] - ma['base_probs']['away']
                    st.caption(f"Base xG: {ma['base_probs']['away']:.1%} ({delta:+.1%})")

            # Ensemble Weights
            if xg.ensemble_weights:
                st.caption(f"**Ensemble Weights:** xG {xg.ensemble_weights['xg']:.0%} • Spread {xg.ensemble_weights['spread']:.0%}")

                # Spiega il peso
                if xg.ensemble_weights['spread'] > 0.30:
                    st.info("🟢 **Sharp money detected**: Spread ha peso maggiore (più affidabile)")
                else:
                    st.caption("⚪ Peso standard: xG ha più influenza")

        # BTTS BAYESIAN
        if xg.bayesian_btts is not None:
            st.subheader("⚽⚽ BTTS Bayesian (Monte Carlo + Market Factors)")

            bb = xg.bayesian_btts

            col1, col2 = st.columns([2, 1])

            with col1:
                # Progress bar visuale
                st.metric("BTTS Probability", f"{bb['btts_prob']:.1%}")
                st.progress(bb['btts_prob'])

                if bb.get('base_btts'):
                    delta = bb['btts_prob'] - bb['base_btts']
                    st.caption(f"Base (Monte Carlo): {bb['base_btts']:.1%} ({delta:+.1%})")

                st.caption(f"Method: {bb.get('method', 'N/A')}")

            with col2:
                st.metric("NO BTTS", f"{bb['nobtts_prob']:.1%}")

                # Mostra fattori
                if bb.get('openness_score') is not None:
                    openness_pct = bb['openness_score'] * 100
                    st.caption(f"🎯 Openness: {openness_pct:.0f}%")

                if bb.get('balance_score') is not None:
                    balance_pct = bb['balance_score'] * 100
                    st.caption(f"⚖️ Balance: {balance_pct:.0f}%")

                if bb.get('total_boost') is not None:
                    st.caption(f"📈 Total Boost: {bb['total_boost']:+.1%}")

            # Interpretazione
            if bb['btts_prob'] > 0.60:
                st.success("🟢 **Alta probabilità BTTS**: Entrambe le squadre segnano molto probabilmente")
            elif bb['btts_prob'] > 0.40:
                st.info("🟡 **Media probabilità BTTS**: Situazione equilibrata")
            else:
                st.warning("🔴 **Bassa probabilità BTTS**: Almeno una squadra potrebbe non segnare")

        # Dettagli tecnici espandibili
        with st.expander("🔧 Dettagli Tecnici Opzione C"):
            if xg.market_adjusted_1x2:
                st.write("**Market-Adjusted 1X2:**")
                st.json({
                    "method": xg.market_adjusted_1x2.get('method'),
                    "market_signal": round(xg.market_adjusted_1x2.get('market_signal', 0), 3),
                    "signal_confidence": round(xg.market_adjusted_1x2.get('signal_confidence', 0), 3),
                    "spread_implied": {
                        "home": round(xg.market_adjusted_1x2.get('spread_implied', {}).get('home', 0), 3),
                        "draw": round(xg.market_adjusted_1x2.get('spread_implied', {}).get('draw', 0), 3),
                        "away": round(xg.market_adjusted_1x2.get('spread_implied', {}).get('away', 0), 3)
                    }
                })

            if xg.bayesian_btts:
                st.write("**Bayesian BTTS:**")
                st.json({
                    "base_btts": round(xg.bayesian_btts.get('base_btts', 0), 3),
                    "openness_score": round(xg.bayesian_btts.get('openness_score', 0), 3),
                    "balance_score": round(xg.bayesian_btts.get('balance_score', 0), 3),
                    "total_boost": round(xg.bayesian_btts.get('total_boost', 0), 3),
                    "method": xg.bayesian_btts.get('method')
                })

        st.markdown("---")
        # ==========================================================

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

