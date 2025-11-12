"""
INTEGRATION PATCH - Istruzioni per integrare Sprint 1 & 2

Questo file contiene le modifiche da fare a Frontendcloud.py

STEP 1: Aggiungere import all'inizio del file (dopo riga 18)
STEP 2: Aggiungere UI controls (dopo riga 14863, dentro la sezione input)
STEP 3: Modificare la chiamata al calcolo lambda per includere constraints
STEP 4: Applicare calibrazione alle probabilità finali
"""

# ============================================================
# STEP 1: IMPORT (Aggiungere dopo riga 18, warnings.filterwarnings)
# ============================================================

IMPORT_CODE = """
# ============================================================
#   ADVANCED FEATURES (Sprint 1 & 2)
# ============================================================
from advanced_features import (
    # Sprint 1
    apply_physical_constraints_to_lambda,
    neumaier_sum,
    precise_probability_sum,
    load_calibration_map,
    apply_calibration,
    # Sprint 2
    apply_motivation_factor,
    apply_fixture_congestion,
    apply_tactical_matchup,
    apply_all_advanced_features,
    MOTIVATION_FACTORS,
    TACTICAL_STYLES
)

# Carica calibration map al startup (una volta sola)
CALIBRATION_MAP = load_calibration_map("storico_analisi.csv")
if CALIBRATION_MAP:
    logger.info(f"✅ Calibration map caricata: {len(CALIBRATION_MAP)} outcomes")
else:
    logger.warning("⚠️ Calibration map non disponibile (serve storico con risultati)")
"""

# ============================================================
# STEP 2: UI CONTROLS (Aggiungere dopo sezione xG/xA, riga ~14863)
# ============================================================

UI_CODE = """
# === FUNZIONALITÀ AVANZATE (Sprint 1 & 2) ===
with st.expander("🚀 Funzionalità Avanzate (Precisione Migliorata)", expanded=False):
    st.markdown(\"\"\"
    **Nuove funzionalità per massimizzare la precisione:**

    ✅ **Constraints Fisici**: Impedisce predizioni irrealistiche
    ✅ **Precision Math**: Elimina errori di arrotondamento
    ✅ **Calibrazione**: Usa storico per rendere probabilità "oneste"
    ✅ **Motivation Index**: Considera importanza match
    ✅ **Fixture Congestion**: Penalità per calendario fitto
    ✅ **Tactical Matchup**: Analizza stili di gioco
    \"\"\")

    # Motivation
    st.markdown("**🎯 Motivation Index**")
    col_mot1, col_mot2 = st.columns(2)

    with col_mot1:
        motivation_home = st.selectbox(
            "Motivazione Casa",
            list(MOTIVATION_FACTORS.keys()),
            index=0,
            key="motivation_home",
            help="Lotta Champions/Salvezza aumentano intensità (+10-20%). Fine stagione senza obiettivi riduce (-8%)"
        )

    with col_mot2:
        motivation_away = st.selectbox(
            "Motivazione Trasferta",
            list(MOTIVATION_FACTORS.keys()),
            index=0,
            key="motivation_away"
        )

    # Fixture Congestion
    st.markdown("**📅 Fixture Congestion (Calendario)**")
    col_fix1, col_fix2 = st.columns(2)

    with col_fix1:
        days_since_home = st.number_input(
            "Giorni dall'ultimo match (Casa)",
            min_value=2,
            max_value=21,
            value=7,
            step=1,
            key="days_since_home",
            help="≤3 giorni = stanchezza (-5%). ≥10 giorni = riposati (+3%)"
        )

        days_until_home = st.number_input(
            "Giorni al prossimo match importante (Casa)",
            min_value=2,
            max_value=14,
            value=7,
            step=1,
            key="days_until_home",
            help="Se match importante fra 3gg + giocato 3gg fa = -8% (rotation risk)"
        )

    with col_fix2:
        days_since_away = st.number_input(
            "Giorni dall'ultimo match (Trasferta)",
            min_value=2,
            max_value=21,
            value=7,
            step=1,
            key="days_since_away"
        )

        days_until_away = st.number_input(
            "Giorni al prossimo match importante (Trasferta)",
            min_value=2,
            max_value=14,
            value=7,
            step=1,
            key="days_until_away"
        )

    # Tactical Styles
    st.markdown("**⚔️ Tactical Matchup (Stili di Gioco)**")
    col_tac1, col_tac2 = st.columns(2)

    with col_tac1:
        style_home = st.selectbox(
            "Stile tattico Casa",
            TACTICAL_STYLES,
            index=0,
            key="style_home",
            help=\\"\\"\\"
            **Possesso**: Dominio palla, manovra lenta (es. Man City, Barcellona)
            **Contropiede**: Difesa compatta + ripartenze veloci (es. Atalanta, Leicester)
            **Pressing Alto**: Aggressivi, recupero alto (es. Liverpool, Napoli)
            **Difensiva**: Blocco basso, pochi rischi (es. Atletico, Burnley)
            \\"\\"\\"
        )

    with col_tac2:
        style_away = st.selectbox(
            "Stile tattico Trasferta",
            TACTICAL_STYLES,
            index=0,
            key="style_away"
        )

    # Preview fattori
    st.markdown("**📊 Preview Adjustments**")
    preview_factor_home = MOTIVATION_FACTORS[motivation_home]
    preview_factor_away = MOTIVATION_FACTORS[motivation_away]

    col_prev1, col_prev2 = st.columns(2)
    with col_prev1:
        st.metric("Fattore Motivation Casa", f"{preview_factor_home:.2f}x")
    with col_prev2:
        st.metric("Fattore Motivation Trasferta", f"{preview_factor_away:.2f}x")

    # Opzioni constraints
    st.markdown("**⚙️ Opzioni Avanzate**")
    apply_constraints = st.checkbox(
        "Applica Constraints Fisici",
        value=True,
        key="apply_constraints",
        help="Forza il modello a rispettare limiti realistici: total 0.5-6.0 gol, P(0-0) ≥ 5%, ecc."
    )

    apply_calibration_enabled = st.checkbox(
        "Applica Calibrazione Probabilità",
        value=True if CALIBRATION_MAP else False,
        key="apply_calibration_enabled",
        help=f"Usa storico per correggere bias. {'✅ Attiva (' + str(len(CALIBRATION_MAP)) + ' outcomes)' if CALIBRATION_MAP else '⚠️ Non disponibile (serve storico)'}"
    )

    use_precision_math = st.checkbox(
        "Usa Precision Math (Neumaier sum)",
        value=True,
        key="use_precision_math",
        help="Elimina errori di arrotondamento nelle somme. Consigliato sempre."
    )
"""

# ============================================================
# STEP 3: MODIFICA CALCOLO LAMBDA (Trova estimate_lambda_from_market_optimized)
# ============================================================

LAMBDA_INTEGRATION_CODE = """
# PRIMA (codice originale):
lambda_h, lambda_a = estimate_lambda_from_market_optimized(
    odds_1, odds_x, odds_2, total_line,
    odds_over25, odds_under25,
    odds_dnb_home, odds_dnb_away,
    home_advantage, rho_initial=0.0
)

# DOPO (con advanced features):
# Calcolo base
lambda_h_base, lambda_a_base = estimate_lambda_from_market_optimized(
    odds_1, odds_x, odds_2, total_line,
    odds_over25, odds_under25,
    odds_dnb_home, odds_dnb_away,
    home_advantage, rho_initial=0.0
)

# Stima rho
rho_base = estimate_rho_optimized(
    lambda_h_base, lambda_a_base,
    p_draw_target,
    odds_btts if odds_btts > 0 else None
)

# Applica advanced features
advanced_result = apply_all_advanced_features(
    lambda_h=lambda_h_base,
    lambda_a=lambda_a_base,
    rho=rho_base,
    total_target=total_line,
    motivation_home=motivation_home,
    motivation_away=motivation_away,
    days_since_home=days_since_home,
    days_since_away=days_since_away,
    days_until_home=days_until_home,
    days_until_away=days_until_away,
    style_home=style_home,
    style_away=style_away,
    apply_constraints=apply_constraints
)

# Usa valori adjustati
lambda_h = advanced_result['lambda_h']
lambda_a = advanced_result['lambda_a']
rho = advanced_result['rho']

# Mostra summary adjustments
if advanced_result['lambda_h_change_pct'] != 0 or advanced_result['lambda_a_change_pct'] != 0:
    st.info(f\\"\\"\\"
    **🚀 Advanced Adjustments Applied:**
    - λ_home: {lambda_h_base:.2f} → {lambda_h:.2f} ({advanced_result['lambda_h_change_pct']:+.1f}%)
    - λ_away: {lambda_a_base:.2f} → {lambda_a:.2f} ({advanced_result['lambda_a_change_pct']:+.1f}%)
    - rho: {rho_base:.3f} → {rho:.3f} ({advanced_result['rho_change']:+.3f})
    \\"\\"\\")
"""

# ============================================================
# STEP 4: APPLICA CALIBRAZIONE PROBABILITÀ (Dopo calc_match_result_from_matrix)
# ============================================================

CALIBRATION_INTEGRATION_CODE = """
# PRIMA:
p_home, p_draw, p_away = calc_match_result_from_matrix(score_matrix)

# DOPO (con calibrazione):
p_home_raw, p_draw_raw, p_away_raw = calc_match_result_from_matrix(score_matrix)

# Applica calibrazione se abilitata
if apply_calibration_enabled and CALIBRATION_MAP:
    p_home, p_draw, p_away = apply_calibration(
        p_home_raw, p_draw_raw, p_away_raw, CALIBRATION_MAP
    )

    # Mostra differenze se significative
    diff_1 = abs(p_home - p_home_raw)
    diff_x = abs(p_draw - p_draw_raw)
    diff_2 = abs(p_away - p_away_raw)

    if max(diff_1, diff_x, diff_2) > 0.02:  # >2% differenza
        st.info(f\\"\\"\\"
        **📊 Calibrazione Applicata:**
        - Prob. Casa: {p_home_raw:.1%} → {p_home:.1%} ({(p_home-p_home_raw)*100:+.1f}pp)
        - Prob. Pareggio: {p_draw_raw:.1%} → {p_draw:.1%} ({(p_draw-p_draw_raw)*100:+.1f}pp)
        - Prob. Trasferta: {p_away_raw:.1%} → {p_away:.1%} ({(p_away-p_away_raw)*100:+.1f}pp)
        \\"\\"\\")
else:
    p_home, p_draw, p_away = p_home_raw, p_draw_raw, p_away_raw
"""

# ============================================================
# STEP 5: PRECISION MATH PER SOMME (Opzionale ma consigliato)
# ============================================================

PRECISION_MATH_CODE = """
# In calc_match_result_from_matrix e altre funzioni che sommano probabilità,
# sostituisci:
#   total = probs.sum()
# con:
#   if use_precision_math:
#       total = neumaier_sum(probs)
#   else:
#       total = probs.sum()

# Oppure usa precise_probability_sum per normalizzare:
#   probs_normalized = precise_probability_sum(probs, expected_total=1.0)
"""

print("=" * 70)
print("INTEGRATION PATCH - Sprint 1 & 2")
print("=" * 70)
print()
print("File creati:")
print("  ✅ advanced_features.py (modulo con tutte le funzionalità)")
print("  ✅ integration_patch.py (questo file con istruzioni)")
print()
print("Per integrare in Frontendcloud.py:")
print()
print("1. Aggiungi IMPORT_CODE dopo riga 18")
print("2. Aggiungi UI_CODE dopo riga 14863 (sezione xG/xA)")
print("3. Modifica calcolo lambda come in LAMBDA_INTEGRATION_CODE")
print("4. Applica calibrazione come in CALIBRATION_INTEGRATION_CODE")
print("5. (Opzionale) Usa Precision Math come in PRECISION_MATH_CODE")
print()
print("NOTA: Sto per creare una versione integrata automaticamente!")
print("=" * 70)
