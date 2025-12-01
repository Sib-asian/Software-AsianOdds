#!/usr/bin/env python3
"""
Dashboard Monitoraggio Automazione 24/7
========================================

Dashboard Streamlit per monitorare l'automazione in tempo reale.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
import time

# Import tracker
from betting_results_tracker import BettingResultsTracker

# Configurazione pagina
st.set_page_config(
    page_title="Automazione 24/7 Monitor",
    page_icon="🤖",
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
        margin-bottom: 2rem;
    }
    .status-active {
        color: #28a745;
        font-weight: bold;
    }
    .status-inactive {
        color: #dc3545;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Inizializza tracker
@st.cache_resource
def get_tracker():
    return BettingResultsTracker()

tracker = get_tracker()

# Header
st.markdown('<div class="main-header">🤖 Automazione 24/7 - Monitor</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configurazione")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
refresh_interval = st.sidebar.slider("Intervallo refresh (secondi)", 10, 60, 30)

# Verifica stato servizio
def check_service_status():
    """Verifica se il servizio è attivo"""
    import psutil
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'automation_service_wrapper.py' in ' '.join(cmdline):
                return True, proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False, None

try:
    is_running, pid = check_service_status()
    status_text = "🟢 ATTIVO" if is_running else "🔴 FERMO"
    status_class = "status-active" if is_running else "status-inactive"
    st.markdown(f'<p class="{status_class}">Stato Servizio: {status_text} (PID: {pid if pid else "N/A"})</p>', unsafe_allow_html=True)
except:
    st.warning("⚠️ Impossibile verificare stato servizio (installa psutil: pip install psutil)")

# Statistiche principali
st.header("📊 Statistiche Oggi")

col1, col2, col3, col4 = st.columns(4)

stats = tracker.get_statistics(days=1)

with col1:
    st.metric("Opportunità Trovate", stats['total_opportunities'])
with col2:
    st.metric("Vincite", stats['winners'], delta=f"{stats['win_rate_percent']:.1f}%")
with col3:
    st.metric("Perdite", stats['losers'])
with col4:
    profit_color = "normal" if stats['total_profit_loss'] >= 0 else "inverse"
    st.metric("Profit/Loss", f"€{stats['total_profit_loss']:.2f}", 
              delta=f"ROI: {stats['roi_percent']:.1f}%", delta_color=profit_color)

# Statistiche periodo
st.header("📈 Statistiche Periodo")

period_days = st.selectbox("Periodo", [7, 14, 30, 60, 90], index=2)
stats_period = tracker.get_statistics(days=period_days)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Performance Generale")
    st.metric("Totale Opportunità", stats_period['total_opportunities'])
    st.metric("Win Rate", f"{stats_period['win_rate_percent']:.1f}%")
    st.metric("ROI", f"{stats_period['roi_percent']:.1f}%")
    st.metric("Profit/Loss Totale", f"€{stats_period['total_profit_loss']:.2f}")

with col2:
    st.subheader("Distribuzione Risultati")
    if stats_period['total_opportunities'] > 0:
        fig = px.pie(
            values=[stats_period['winners'], stats_period['losers'], stats_period['pending']],
            names=['Vincite', 'Perdite', 'Pending'],
            title="Distribuzione Risultati"
        )
        st.plotly_chart(fig, use_container_width=True)

# Grafico performance nel tempo
st.header("📊 Performance nel Tempo")

try:
    conn = sqlite3.connect(tracker.db_path)
    df = pd.read_sql_query("""
        SELECT date, opportunities_count, winners_count, losers_count,
               total_profit_loss, roi, win_rate
        FROM daily_stats
        ORDER BY date DESC
        LIMIT 30
    """, conn)
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['total_profit_loss'].cumsum(),
            mode='lines+markers',
            name='Cumulative P/L',
            line=dict(color='green' if df['total_profit_loss'].cumsum().iloc[-1] >= 0 else 'red')
        ))
        fig.update_layout(
            title="Profit/Loss Cumulativo",
            xaxis_title="Data",
            yaxis_title="€",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Grafico ROI
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df['date'],
            y=df['roi'],
            name='ROI %',
            marker_color=['green' if x >= 0 else 'red' for x in df['roi']]
        ))
        fig2.update_layout(
            title="ROI Giornaliero",
            xaxis_title="Data",
            yaxis_title="ROI %"
        )
        st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.warning(f"Nessun dato disponibile: {e}")

# Performance per Market
if stats_period['by_market']:
    st.header("🎯 Performance per Market")
    
    market_data = []
    for market, data in stats_period['by_market'].items():
        market_data.append({
            'Market': market,
            'Count': data['count'],
            'Winners': data['winners'],
            'Win Rate %': data['win_rate'],
            'Profit/Loss': data['profit_loss']
        })
    
    df_markets = pd.DataFrame(market_data)
    st.dataframe(df_markets, use_container_width=True)
    
    # Grafico
    fig = px.bar(
        df_markets,
        x='Market',
        y='Profit/Loss',
        title="Profit/Loss per Market",
        color='Profit/Loss',
        color_continuous_scale=['red', 'green']
    )
    st.plotly_chart(fig, use_container_width=True)

# Performance per Lega
if stats_period['by_league']:
    st.header("🏆 Performance per Lega")
    
    league_data = []
    for league, data in stats_period['by_league'].items():
        league_data.append({
            'Lega': league,
            'Count': data['count'],
            'Winners': data['winners'],
            'Win Rate %': data['win_rate'],
            'Profit/Loss': data['profit_loss']
        })
    
    df_leagues = pd.DataFrame(league_data)
    st.dataframe(df_leagues, use_container_width=True)

# Opportunità Recenti
st.header("📋 Opportunità Recenti")

recent = tracker.get_recent_opportunities(limit=20)
if recent:
    df_recent = pd.DataFrame(recent)
    df_recent['notified_at'] = pd.to_datetime(df_recent['notified_at'])
    df_recent = df_recent.sort_values('notified_at', ascending=False)
    
    # Formatta colonne
    display_cols = ['home_team', 'away_team', 'league', 'market', 'odds', 
                    'expected_value', 'confidence', 'result', 'profit_loss', 'notified_at']
    available_cols = [col for col in display_cols if col in df_recent.columns]
    
    st.dataframe(df_recent[available_cols], use_container_width=True)
else:
    st.info("Nessuna opportunità ancora registrata")

# Log in tempo reale
st.header("📝 Log Recenti")

log_file = Path("logs/automation_service_20251117.log")
if log_file.exists():
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        recent_logs = lines[-50:] if len(lines) > 50 else lines
        st.text_area("Ultimi log", '\n'.join(recent_logs), height=300)
else:
    st.info("File log non trovato")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

