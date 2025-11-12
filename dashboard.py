#!/usr/bin/env python3
"""
BETTING PERFORMANCE DASHBOARD
Dashboard Streamlit per visualizzare metriche e performance del modello di betting

Utilizzo:
    streamlit run dashboard.py

Features:
- Performance Overview (ROI, Win Rate, Sharpe Ratio)
- Grafici profit/loss nel tempo
- Analisi per mercato (1X2, Over/Under, BTTS)
- Migliori/Peggiori scommesse
- Scommesse pending
- Statistiche avanzate
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Import dal file principale
sys.path.append(str(Path(__file__).parent))
from Frontendcloud import (
    initialize_database,
    get_performance_summary,
    get_best_worst_bets,
    get_performance_by_market,
    get_pending_bets,
    get_db_connection,
    get_advanced_team_data,
)

# Configurazione pagina
st.set_page_config(
    page_title="Betting Performance Dashboard",
    page_icon="⚽",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Inizializza database
try:
    initialize_database()
except Exception as e:
    st.error(f"Errore inizializzazione database: {e}")

# Header
st.markdown('<div class="main-header">⚽ Betting Performance Dashboard</div>', unsafe_allow_html=True)

# Sidebar - Filtri
st.sidebar.header("📊 Filtri")
period_days = st.sidebar.selectbox(
    "Periodo di analisi",
    [7, 14, 30, 60, 90, 365],
    index=2
)

LEAGUE_OPTIONS = {
    "Serie A (Italia)": "serie_a",
    "Premier League (Inghilterra)": "premier_league",
    "La Liga (Spagna)": "la_liga",
    "Bundesliga (Germania)": "bundesliga",
    "Ligue 1 (Francia)": "ligue_1",
}

if "api_insight" not in st.session_state:
    st.session_state["api_insight"] = None
    st.session_state["api_error"] = None

st.sidebar.markdown("---")
st.sidebar.header("🌐 Dati API gratuiti")
league_label = st.sidebar.selectbox("Competizione", list(LEAGUE_OPTIONS.keys()))
home_team_input = st.sidebar.text_input("Squadra di casa", "")
away_team_input = st.sidebar.text_input("Squadra ospite", "")
api_match_date = st.sidebar.date_input("Data partita", datetime.now().date())

if st.sidebar.button("Recupera dati API"):
    if not home_team_input.strip() or not away_team_input.strip():
        st.sidebar.warning("Inserisci nome di entrambe le squadre.")
    else:
        with st.spinner("Recupero dati dalle API gratuite..."):
            try:
                match_date_str = datetime.combine(api_match_date, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%SZ")
                advanced = get_advanced_team_data(
                    home_team_input.strip(),
                    away_team_input.strip(),
                    LEAGUE_OPTIONS[league_label],
                    match_date_str,
                )
                st.session_state["api_insight"] = {
                    "home_team": home_team_input.strip(),
                    "away_team": away_team_input.strip(),
                    "league_label": league_label,
                    "data": advanced,
                }
                st.session_state["api_error"] = None
            except Exception as api_exc:
                st.session_state["api_insight"] = None
                st.session_state["api_error"] = str(api_exc)

# Carica dati
try:
    summary = get_performance_summary(days=period_days)
    best_worst = get_best_worst_bets(limit=10)
    by_market = get_performance_by_market()
    pending = get_pending_bets()
except Exception as e:
    st.error(f"Errore caricamento dati: {e}")
    st.stop()

# === SEZIONE 1: METRICHE PRINCIPALI ===
st.header("📈 Performance Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Bets",
        value=summary['total_bets'],
        delta=None
    )

with col2:
    win_rate = summary['win_rate']
    color = "normal" if win_rate >= 55 else "inverse"
    st.metric(
        label="Win Rate",
        value=f"{win_rate}%",
        delta=f"{win_rate - 54:.1f}% vs target" if summary['total_bets'] > 0 else None,
        delta_color=color
    )

with col3:
    roi = summary['roi']
    color = "normal" if roi > 0 else "inverse"
    st.metric(
        label="ROI",
        value=f"{roi}%",
        delta=f"€{summary['total_profit']}" if summary['total_bets'] > 0 else None,
        delta_color=color
    )

with col4:
    sharpe = summary['sharpe_ratio']
    if sharpe:
        color = "normal" if sharpe > 1.0 else "inverse"
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe:.2f}",
            delta="Good" if sharpe > 1.0 else "Poor",
            delta_color=color
        )
    else:
        st.metric(label="Sharpe Ratio", value="N/A")

with col5:
    brier = summary['brier_score']
    if brier:
        color = "inverse" if brier < 0.20 else "normal"  # Lower is better
        st.metric(
            label="Brier Score",
            value=f"{brier:.4f}",
            delta="Excellent" if brier < 0.20 else "Average",
            delta_color=color
        )
    else:
        st.metric(label="Brier Score", value="N/A")

st.divider()

# === SEZIONE 2: GRAFICO PROFIT NEL TEMPO ===
st.header("💰 Profit/Loss nel Tempo")

# Query per ottenere profit giornaliero
try:
    with get_db_connection() as conn:
        date_limit = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')

        query = """
            SELECT
                DATE(bet_time) as date,
                SUM(COALESCE(profit, 0)) as daily_profit,
                COUNT(*) as daily_bets
            FROM bets
            WHERE bet_time >= ? AND result != 'pending'
            GROUP BY DATE(bet_time)
            ORDER BY date ASC
        """

        df_profit = pd.read_sql_query(query, conn, params=(date_limit,))

        if not df_profit.empty:
            # Calcola profit cumulativo
            df_profit['cumulative_profit'] = df_profit['daily_profit'].cumsum()

            # Grafico con Plotly
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_profit['date'],
                y=df_profit['cumulative_profit'],
                mode='lines+markers',
                name='Cumulative Profit',
                line=dict(color='#1f77b4', width=3),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))

            fig.update_layout(
                title="Profit Cumulativo",
                xaxis_title="Data",
                yaxis_title="Profit (€)",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Tabella riepilogo
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(
                    df_profit[['date', 'daily_bets', 'daily_profit', 'cumulative_profit']].tail(10),
                    use_container_width=True
                )
            with col2:
                # Grafico a barre daily profit
                fig_bars = px.bar(
                    df_profit.tail(30),
                    x='date',
                    y='daily_profit',
                    title="Profit Giornaliero (ultimi 30 giorni)",
                    color='daily_profit',
                    color_continuous_scale=['red', 'gray', 'green']
                )
                st.plotly_chart(fig_bars, use_container_width=True)
        else:
            st.info("Nessun dato disponibile per il periodo selezionato")

except Exception as e:
    st.error(f"Errore caricamento grafico profit: {e}")

st.divider()

# === SEZIONE 3: PERFORMANCE PER MERCATO ===
st.header("🎯 Performance per Mercato")

if by_market:
    df_markets = pd.DataFrame(by_market)

    col1, col2 = st.columns(2)

    with col1:
        # Grafico ROI per mercato
        fig_roi = px.bar(
            df_markets,
            x='market',
            y='roi',
            title="ROI per Mercato",
            color='roi',
            color_continuous_scale=['red', 'yellow', 'green'],
            text='roi'
        )
        fig_roi.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_roi, use_container_width=True)

    with col2:
        # Grafico Win Rate per mercato
        fig_wr = px.bar(
            df_markets,
            x='market',
            y='win_rate',
            title="Win Rate per Mercato",
            color='win_rate',
            color_continuous_scale=['red', 'yellow', 'green'],
            text='win_rate'
        )
        fig_wr.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_wr, use_container_width=True)

    # Tabella dettagliata
    st.dataframe(
        df_markets.sort_values('total_profit', ascending=False),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Nessuna scommessa trovata")

st.divider()

# === SEZIONE 4: MIGLIORI E PEGGIORI BETS ===
st.header("🏆 Best & Worst Bets")

col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ Top 10 Best Bets")
    if best_worst['best_bets']:
        df_best = pd.DataFrame(best_worst['best_bets'])
        df_best['match'] = df_best['home_team'] + ' vs ' + df_best['away_team']
        st.dataframe(
            df_best[['match', 'date', 'market', 'selection', 'odds', 'stake', 'profit', 'roi_bet']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Nessuna scommessa completata")

with col2:
    st.subheader("❌ Top 10 Worst Bets")
    if best_worst['worst_bets']:
        df_worst = pd.DataFrame(best_worst['worst_bets'])
        df_worst['match'] = df_worst['home_team'] + ' vs ' + df_worst['away_team']
        st.dataframe(
            df_worst[['match', 'date', 'market', 'selection', 'odds', 'stake', 'profit', 'roi_bet']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Nessuna scommessa completata")

st.divider()

# === SEZIONE 5: PENDING BETS ===
st.header("⏳ Scommesse Pending")

if pending:
    df_pending = pd.DataFrame(pending)
    df_pending['match'] = df_pending['home_team'] + ' vs ' + df_pending['away_team']

    # Calcola potential profit
    df_pending['potential_profit'] = df_pending.apply(
        lambda row: row['stake'] * (row['odds'] - 1) if row['odds'] else 0,
        axis=1
    )

    # Mostra tabella
    st.dataframe(
        df_pending[[
            'match', 'date', 'market', 'selection',
            'probability', 'odds', 'edge', 'stake', 'potential_profit'
        ]],
        use_container_width=True,
        hide_index=True
    )

    # Riepilogo
    total_stake_pending = df_pending['stake'].sum()
    total_potential = df_pending['potential_profit'].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pending Bets", len(df_pending))
    col2.metric("Total Stake at Risk", f"€{total_stake_pending:.2f}")
    col3.metric("Potential Win", f"€{total_potential:.2f}")
else:
    st.info("Nessuna scommessa pending")

st.divider()

# === SEZIONE 6: STATISTICHE AVANZATE ===
with st.expander("📊 Statistiche Avanzate"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Staked", f"€{summary['total_staked']}")
        st.metric("Avg Odds", f"{summary['avg_odds']}" if summary['avg_odds'] else "N/A")

    with col2:
        st.metric("Wins", summary['wins'])
        st.metric("Losses", summary['losses'])
        st.metric("Pushes", summary['pushes'])

    with col3:
        st.metric("Avg Edge", f"{summary['avg_edge']}%" if summary['avg_edge'] else "N/A")

        # Calcola expected value
        if summary['total_bets'] > 0:
            ev_per_bet = summary['total_profit'] / summary['total_bets']
            st.metric("EV per Bet", f"€{ev_per_bet:.2f}")

# === SEZIONE 7: ANALISI API GRATUITE ===
st.divider()
st.header("🌐 Analisi Squadre (API gratuite)")

# FIX BUG: Avoid race condition - use single .get() call instead of checking then accessing
api_error = st.session_state.get("api_error")
if api_error:
    st.error(f"Errore recupero dati API: {api_error}")

api_insight = st.session_state.get("api_insight")
if api_insight and api_insight.get("data"):
    advanced = api_insight["data"] or {}
    st.caption(
        f"{api_insight['home_team']} vs {api_insight['away_team']} – {api_insight['league_label']}"
    )

    statsbomb_home = advanced.get("statsbomb_home")
    statsbomb_away = advanced.get("statsbomb_away")
    fd_home = advanced.get("football_data_home_metrics")
    fd_away = advanced.get("football_data_away_metrics")
    form_home = advanced.get("home_team_stats")
    form_away = advanced.get("away_team_stats")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("StatsBomb xG (ultime partite)")
        if (
            statsbomb_home
            and statsbomb_home.get("available")
            and statsbomb_away
            and statsbomb_away.get("available")
        ):
            statsbomb_df = pd.DataFrame(
                {
                    "Casa": [
                        statsbomb_home.get("xg_for_avg"),
                        statsbomb_home.get("xg_against_avg"),
                        statsbomb_home.get("shots_for_avg"),
                        statsbomb_home.get("shots_against_avg"),
                    ],
                    "Ospiti": [
                        statsbomb_away.get("xg_for_avg"),
                        statsbomb_away.get("xg_against_avg"),
                        statsbomb_away.get("shots_for_avg"),
                        statsbomb_away.get("shots_against_avg"),
                    ],
                },
                index=["xG fatti", "xG concessi", "Tiri", "Tiri concessi"],
            )
            st.dataframe(statsbomb_df.round(3), use_container_width=True)
            st.caption(
                "Metriche calcolate sui dataset open data StatsBomb (max 12 match recenti)."
            )
        else:
            st.info("Metriche StatsBomb non disponibili per entrambe le squadre.")

    with col2:
        st.subheader("Forma recente (Football-Data.org)")
        if fd_home and fd_home.get("available") and fd_away and fd_away.get("available"):
            fd_df = pd.DataFrame(
                {
                    "Casa": [
                        fd_home.get("avg_goals_for"),
                        fd_home.get("avg_goals_against"),
                        fd_home.get("points_per_game"),
                        fd_home.get("clean_sheet_rate"),
                        fd_home.get("recent_form"),
                    ],
                    "Ospiti": [
                        fd_away.get("avg_goals_for"),
                        fd_away.get("avg_goals_against"),
                        fd_away.get("points_per_game"),
                        fd_away.get("clean_sheet_rate"),
                        fd_away.get("recent_form"),
                    ],
                },
                index=[
                    "Gol fatti",
                    "Gol subiti",
                    "Punti/partita",
                    "Clean sheet %",
                    "Forma (ultimi match)",
                ],
            )
            st.dataframe(fd_df, use_container_width=True)
            st.caption("Dati ufficiali Football-Data.org (ultime 6 partite finite).")
        else:
            st.info("Statistiche Football-Data.org non disponibili per entrambe le squadre.")

    st.subheader("Fattori modello (blend API)")
    form_cols = st.columns(2)
    form_df_cols = [
        ("Indice attacco", "form_attack"),
        ("Indice difesa", "form_defense"),
        ("Punti normalizzati", "form_points"),
        ("Confidence", "confidence"),
    ]

    def build_form_df(label: str, data: dict) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()
        values = []
        for title, key in form_df_cols:
            values.append((title, data.get(key)))
        return pd.DataFrame(values, columns=["Metrica", label]).set_index("Metrica")

    with form_cols[0]:
        if form_home:
            st.dataframe(build_form_df("Casa", form_home).round(3), use_container_width=True)
        else:
            st.info("Nessun dato di forma casa disponibile.")

    with form_cols[1]:
        if form_away:
            st.dataframe(build_form_df("Ospiti", form_away).round(3), use_container_width=True)
        else:
            st.info("Nessun dato di forma ospite disponibile.")
else:
    st.info("Usa la sezione laterale per recuperare dati dalle API gratuite su una partita specifica.")

# Footer
st.divider()
st.caption(f"Dashboard aggiornata: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("Dati provenienti da: betting_database.db")
