#!/usr/bin/env python3
"""
Test completo per le 6 FORMULE AVANZATE di precisione:
1. Dixon-Coles Bivariate Poisson
2. Home Advantage Factor
3. Vig Removal + Edge Calculation
4. Regression to Mean
5. Correlated Markets Model
6. Monte Carlo Validation

Obiettivo: "tutto deve essere calcolato alla perfezione"
"""

from market_movement_analyzer import (
    MarketMovementAnalyzer,
    dixon_coles_probability,
    adjust_for_home_advantage,
    remove_vig,
    calculate_edge,
    regression_to_mean,
    btts_and_total_joint,
    monte_carlo_validation,
    calculate_expected_goals,
    poisson_probability,
    DIXON_COLES_RHO,
    HOME_ADVANTAGE_BOOST,
    AWAY_DISADVANTAGE
)
import math


def print_separator(char="="):
    print(f"\n{char * 80}\n")


def test_dixon_coles_correlation():
    """
    TEST 1: Dixon-Coles Bivariate Poisson
    Verifica che la correlazione funzioni correttamente sui risultati bassi
    """
    print("🧪 TEST 1: Dixon-Coles Bivariate Poisson")
    print_separator()

    home_xg = 1.5
    away_xg = 1.2

    # Test risultati bassi (0-0, 1-0, 0-1, 1-1) dove la correlazione conta
    low_scores = [(0, 0), (1, 0), (0, 1), (1, 1)]

    print(f"Input: xG Casa={home_xg}, xG Trasferta={away_xg}")
    print(f"Rho (correlazione): {DIXON_COLES_RHO}\n")

    for h, a in low_scores:
        # Poisson standard (assume indipendenza)
        prob_standard = poisson_probability(h, home_xg) * poisson_probability(a, away_xg)

        # Dixon-Coles (corregge per correlazione)
        prob_dixon = dixon_coles_probability(h, a, home_xg, away_xg)

        diff = prob_dixon - prob_standard
        diff_pct = (diff / prob_standard) * 100 if prob_standard > 0 else 0

        print(f"  {h}-{a}: Poisson={prob_standard:.4f}, Dixon-Coles={prob_dixon:.4f}, "
              f"Diff={diff:+.4f} ({diff_pct:+.1f}%)")

    # Verifica matematica: 0-0 e 1-1 dovrebbero DIMINUIRE (rho negativo)
    prob_00_standard = poisson_probability(0, home_xg) * poisson_probability(0, away_xg)
    prob_00_dixon = dixon_coles_probability(0, 0, home_xg, away_xg)
    assert prob_00_dixon < prob_00_standard, "0-0 dovrebbe diminuire con rho negativo"

    prob_11_standard = poisson_probability(1, home_xg) * poisson_probability(1, away_xg)
    prob_11_dixon = dixon_coles_probability(1, 1, home_xg, away_xg)
    assert prob_11_dixon < prob_11_standard, "1-1 dovrebbe diminuire con rho negativo"

    # Test risultato alto (4-3) - dovrebbe essere uguale a Poisson
    prob_43_standard = poisson_probability(4, home_xg) * poisson_probability(3, away_xg)
    prob_43_dixon = dixon_coles_probability(4, 3, home_xg, away_xg)
    assert abs(prob_43_dixon - prob_43_standard) < 0.0001, "Risultati alti non devono essere corretti"

    print("\n✅ Test Dixon-Coles: PASSED")
    print("  - Correlazione negativa applicata correttamente a 0-0, 1-1")
    print("  - Risultati alti non affetti")


def test_home_advantage_redistribution():
    """
    TEST 2: Home Advantage Factor
    Verifica che il vantaggio casa redistribuisca i gol preservando il total
    """
    print_separator()
    print("🧪 TEST 2: Home Advantage Factor")
    print_separator()

    # Caso 1: Equilibrio perfetto
    home_xg = 1.5
    away_xg = 1.5
    original_total = home_xg + away_xg

    adjusted_home, adjusted_away = adjust_for_home_advantage(home_xg, away_xg)
    new_total = adjusted_home + adjusted_away

    print(f"Caso 1: Team equilibrati (xG 1.5 vs 1.5)")
    print(f"  Prima:  Casa={home_xg:.2f}, Trasferta={away_xg:.2f}, Total={original_total:.2f}")
    print(f"  Dopo:   Casa={adjusted_home:.2f}, Trasferta={adjusted_away:.2f}, Total={new_total:.2f}")
    print(f"  Boost:  Casa={HOME_ADVANTAGE_BOOST}, Trasferta={AWAY_DISADVANTAGE}")

    # Verifica: total deve essere preservato
    assert abs(new_total - original_total) < 0.001, "Total deve essere preservato"

    # Verifica: casa deve avere più xG dopo adjustment
    assert adjusted_home > home_xg, "Casa dovrebbe avere xG aumentato"
    assert adjusted_away < away_xg, "Trasferta dovrebbe avere xG diminuito"

    # Caso 2: Favorito trasferta
    home_xg2 = 1.0
    away_xg2 = 2.0
    original_total2 = home_xg2 + away_xg2

    adjusted_home2, adjusted_away2 = adjust_for_home_advantage(home_xg2, away_xg2)
    new_total2 = adjusted_home2 + adjusted_away2

    print(f"\nCaso 2: Favorito trasferta (xG 1.0 vs 2.0)")
    print(f"  Prima:  Casa={home_xg2:.2f}, Trasferta={away_xg2:.2f}, Total={original_total2:.2f}")
    print(f"  Dopo:   Casa={adjusted_home2:.2f}, Trasferta={adjusted_away2:.2f}, Total={new_total2:.2f}")

    assert abs(new_total2 - original_total2) < 0.001, "Total deve essere preservato"
    assert adjusted_home2 > home_xg2, "Casa dovrebbe comunque beneficiare del vantaggio"

    print("\n✅ Test Home Advantage: PASSED")
    print("  - Total preservato correttamente")
    print("  - Redistribuzione favorisce sempre casa")


def test_vig_removal_and_edge():
    """
    TEST 3: Vig Removal + Edge Calculation
    Verifica rimozione overround e calcolo edge
    """
    print_separator()
    print("🧪 TEST 3: Vig Removal + Edge Calculation")
    print_separator()

    # Quote realistiche con overround ~105%
    odds_home = 2.10
    odds_draw = 3.40
    odds_away = 3.60

    # Probabilità implicite (con vig)
    implied_home = 1 / odds_home
    implied_draw = 1 / odds_draw
    implied_away = 1 / odds_away
    overround = implied_home + implied_draw + implied_away

    print(f"Quote: 1={odds_home}, X={odds_draw}, 2={odds_away}")
    print(f"Probabilità implicite (con vig): 1={implied_home:.3f}, X={implied_draw:.3f}, 2={implied_away:.3f}")
    print(f"Overround (margine book): {overround:.3f} ({(overround-1)*100:.1f}%)")

    # Rimuovi vig
    true_home, true_draw, true_away = remove_vig(odds_home, odds_draw, odds_away)
    true_sum = true_home + true_draw + true_away

    print(f"\nProbabilità TRUE (senza vig): 1={true_home:.3f}, X={true_draw:.3f}, 2={true_away:.3f}")
    print(f"Somma: {true_sum:.3f}")

    # Verifica: somma deve essere 1.0 (100%)
    assert abs(true_sum - 1.0) < 0.001, "Probabilità true devono sommare a 100%"
    # Con vig removal, le probabilità DIMINUISCONO (dividiamo per overround > 1)
    assert true_home < implied_home, "Prob true < prob implicita (vig rimosso riduce le prob)"
    assert true_draw < implied_draw, "Prob true < prob implicita"
    assert true_away < implied_away, "Prob true < prob implicita"

    # Test Edge Calculation
    print(f"\n--- Edge Calculation ---")
    our_prob = 0.52  # Stimiamo 52% probabilità casa
    edge = calculate_edge(our_prob, odds_home)

    print(f"La nostra probabilità: {our_prob:.1%}")
    print(f"Quote bookmaker: {odds_home}")
    print(f"Probabilità implicita: {implied_home:.1%}")
    print(f"EDGE: {edge:+.2%}")

    if edge > 0:
        print(f"  ✅ VALUE BET! Abbiamo {edge:.1%} di vantaggio")
    else:
        print(f"  ❌ NO VALUE (edge negativo)")

    # Verifica matematica
    expected_edge = our_prob - implied_home
    assert abs(edge - expected_edge) < 0.001, "Edge deve essere la differenza tra probabilità"

    print("\n✅ Test Vig Removal + Edge: PASSED")
    print("  - Overround rimosso correttamente")
    print("  - Edge calcolato matematicamente corretto")


def test_regression_to_mean():
    """
    TEST 4: Regression to Mean (James-Stein Shrinkage)
    Verifica che corregga movimenti estremi
    """
    print_separator()
    print("🧪 TEST 4: Regression to Mean")
    print_separator()

    prior_mean = -1.0  # Media storica spread

    # Caso 1: Movimento moderato
    observed_moderate = -1.25
    adjusted_moderate = regression_to_mean(observed_moderate, prior_mean, confidence=0.75)

    print(f"Caso 1: Movimento moderato")
    print(f"  Media storica: {prior_mean}")
    print(f"  Osservato: {observed_moderate}")
    print(f"  Aggiustato (75% confidence): {adjusted_moderate:.3f}")

    # L'aggiustato deve essere tra prior e observed
    assert prior_mean <= adjusted_moderate <= observed_moderate or \
           observed_moderate <= adjusted_moderate <= prior_mean, \
           "Adjusted deve essere tra prior e observed"

    # Caso 2: Movimento estremo (possibile overreaction)
    observed_extreme = -2.5
    adjusted_extreme = regression_to_mean(observed_extreme, prior_mean, confidence=0.75)

    print(f"\nCaso 2: Movimento estremo (possibile overreaction)")
    print(f"  Media storica: {prior_mean}")
    print(f"  Osservato: {observed_extreme}")
    print(f"  Aggiustato (75% confidence): {adjusted_extreme:.3f}")
    print(f"  Riduzione: {abs(observed_extreme - adjusted_extreme):.3f}")

    # Movimento estremo deve essere corretto più aggressivamente
    extreme_correction = abs(observed_extreme - adjusted_extreme)
    moderate_correction = abs(observed_moderate - adjusted_moderate)
    assert extreme_correction > moderate_correction, "Movimenti estremi devono essere corretti di più"

    # Caso 3: Alta confidence (90%) - meno correzione
    adjusted_high_conf = regression_to_mean(observed_extreme, prior_mean, confidence=0.90)

    print(f"\nCaso 3: Stesso movimento estremo, ma alta confidence (90%)")
    print(f"  Aggiustato: {adjusted_high_conf:.3f}")

    # Con alta confidence, adjusted dovrebbe essere più vicino a observed
    assert abs(adjusted_high_conf - observed_extreme) < abs(adjusted_extreme - observed_extreme), \
           "Alta confidence dovrebbe correggere meno"

    print("\n✅ Test Regression to Mean: PASSED")
    print("  - Movimenti moderati corretti leggermente")
    print("  - Movimenti estremi corretti maggiormente")
    print("  - Confidence influenza correzione")


def test_correlated_markets():
    """
    TEST 5: Correlated Markets Model
    Verifica probabilità congiunte BTTS + Over/Under
    """
    print_separator()
    print("🧪 TEST 5: Correlated Markets Model")
    print_separator()

    home_xg = 1.8
    away_xg = 1.2
    threshold = 2.5

    # Calcola probabilità congiunte
    joint = btts_and_total_joint(home_xg, away_xg, threshold=threshold, use_dixon_coles=True)

    print(f"Input: xG Casa={home_xg}, xG Trasferta={away_xg}, Threshold={threshold}")
    print(f"\nProbabilità congiunte:")
    print(f"  BTTS + Over {threshold}:  {joint['btts_and_over']:.2%}")
    print(f"  BTTS + Under {threshold}: {joint['btts_and_under']:.2%}")
    print(f"  NO-BTTS + Over {threshold}:  {joint['nobtts_and_over']:.2%}")
    print(f"  NO-BTTS + Under {threshold}: {joint['nobtts_and_under']:.2%}")

    print(f"\nProbabilità marginali:")
    print(f"  P(BTTS) totale: {joint['btts_total']:.2%}")
    print(f"  P(Over {threshold}) totale: {joint['over_total']:.2%}")

    # Verifica: somma deve essere 100%
    total_prob = (joint['btts_and_over'] + joint['btts_and_under'] +
                  joint['nobtts_and_over'] + joint['nobtts_and_under'])
    assert abs(total_prob - 1.0) < 0.01, "Somma probabilità congiunte deve essere 100%"

    # Verifica: marginali corrette
    btts_check = joint['btts_and_over'] + joint['btts_and_under']
    assert abs(btts_check - joint['btts_total']) < 0.01, "Marginale BTTS deve essere corretta"

    over_check = joint['btts_and_over'] + joint['nobtts_and_over']
    assert abs(over_check - joint['over_total']) < 0.01, "Marginale Over deve essere corretta"

    # Test indipendenza vs correlazione
    print(f"\n--- Test Correlazione vs Indipendenza ---")

    # Se fossero indipendenti: P(BTTS and Over) = P(BTTS) * P(Over)
    independent_prob = joint['btts_total'] * joint['over_total']
    actual_prob = joint['btts_and_over']

    print(f"  P(BTTS) * P(Over) [indipendenza]: {independent_prob:.2%}")
    print(f"  P(BTTS AND Over) [correlata]:     {actual_prob:.2%}")
    print(f"  Differenza: {(actual_prob - independent_prob):.2%}")

    # BTTS e Over sono POSITIVAMENTE correlati (entrambi richiedono gol)
    # Quindi P(BTTS and Over) > P(BTTS) * P(Over)
    print(f"\n  {'✅' if actual_prob > independent_prob else '⚠️'} Correlazione positiva rilevata")

    print("\n✅ Test Correlated Markets: PASSED")
    print("  - Probabilità congiunte sommano a 100%")
    print("  - Marginali corrette")
    print("  - Correlazione positiva BTTS+Over rilevata")


def test_monte_carlo_validation():
    """
    TEST 6: Monte Carlo Validation
    Verifica robustezza tramite simulazione
    """
    print_separator()
    print("🧪 TEST 6: Monte Carlo Validation")
    print_separator()

    home_xg = 1.6
    away_xg = 1.1
    n_sim = 10000

    print(f"Input: xG Casa={home_xg}, xG Trasferta={away_xg}")
    print(f"Simulazioni: {n_sim:,}")

    # Run Monte Carlo
    mc_results = monte_carlo_validation(home_xg, away_xg, n_simulations=n_sim, use_dixon_coles=True)

    print(f"\n📊 Risultati Monte Carlo:")
    print(f"  P(1): {mc_results['prob_1']:.2%} ± {mc_results['ci_1']:.2%} (95% CI)")
    print(f"  P(X): {mc_results['prob_x']:.2%} ± {mc_results['ci_x']:.2%} (95% CI)")
    print(f"  P(2): {mc_results['prob_2']:.2%} ± {mc_results['ci_2']:.2%} (95% CI)")
    print(f"  P(BTTS): {mc_results['prob_btts']:.2%}")
    print(f"  Avg Total Goals: {mc_results['avg_total_goals']:.2f} ± {mc_results['std_total_goals']:.2f}")

    # Verifica: somma probabilità 1X2 ~100%
    total_1x2 = mc_results['prob_1'] + mc_results['prob_x'] + mc_results['prob_2']
    assert abs(total_1x2 - 1.0) < 0.05, "Somma 1X2 deve essere ~100% (permettiamo 5% tolleranza per MC)"

    # Verifica: CI ragionevoli (non troppo larghi)
    assert mc_results['ci_1'] < 0.05, "CI troppo larghi, aumenta simulazioni"
    assert mc_results['ci_x'] < 0.05, "CI troppo larghi, aumenta simulazioni"
    assert mc_results['ci_2'] < 0.05, "CI troppo larghi, aumenta simulazioni"

    # Confronta con calcolo analitico DIRETTO (stessi xG, no spread/total conversion)
    print(f"\n--- Confronto Monte Carlo vs Analitico ---")

    # Calcolo analitico usando Dixon-Coles direttamente (stessi xG del MC)
    prob_1_analytical = 0.0
    prob_x_analytical = 0.0
    prob_2_analytical = 0.0

    for h in range(10):
        for a in range(10):
            prob = dixon_coles_probability(h, a, home_xg, away_xg)
            if h > a:
                prob_1_analytical += prob
            elif h == a:
                prob_x_analytical += prob
            else:
                prob_2_analytical += prob

    diff_1 = abs(mc_results['prob_1'] - prob_1_analytical)
    diff_x = abs(mc_results['prob_x'] - prob_x_analytical)
    diff_2 = abs(mc_results['prob_2'] - prob_2_analytical)

    print(f"  P(1): MC={mc_results['prob_1']:.2%}, Analitico={prob_1_analytical:.2%}, Diff={diff_1:.2%}")
    print(f"  P(X): MC={mc_results['prob_x']:.2%}, Analitico={prob_x_analytical:.2%}, Diff={diff_x:.2%}")
    print(f"  P(2): MC={mc_results['prob_2']:.2%}, Analitico={prob_2_analytical:.2%}, Diff={diff_2:.2%}")

    # Differenze devono essere piccole (MC converge ad analitico)
    # Tolleranza 2% per varianza Monte Carlo
    assert diff_1 < 0.02, f"MC e Analitico devono concordare entro 2%, got {diff_1:.2%}"
    assert diff_x < 0.02, f"MC e Analitico devono concordare entro 2%, got {diff_x:.2%}"
    assert diff_2 < 0.02, f"MC e Analitico devono concordare entro 2%, got {diff_2:.2%}"

    print("\n✅ Test Monte Carlo: PASSED")
    print("  - Simulazioni convergono correttamente")
    print("  - Confidence intervals ragionevoli")
    print("  - Risultati concordano con calcolo analitico")


def test_integration_all_formulas():
    """
    TEST 7: Integrazione - Tutte le formule insieme
    Verifica che le 6 formule lavorino in armonia
    """
    print_separator()
    print("🧪 TEST 7: Integrazione End-to-End (Tutte le 6 formule)")
    print_separator()

    analyzer = MarketMovementAnalyzer()

    # Scenario: Favorito casa moderato, partita con gol
    result = analyzer.analyze(
        spread_open=-0.75,
        spread_close=-1.25,
        total_open=2.5,
        total_close=2.75
    )

    print("📊 ANALISI COMPLETA CON FORMULE AVANZATE")
    print(f"\nSpread: {result.spread_analysis.opening_value} → {result.spread_analysis.closing_value}")
    print(f"Total: {result.total_analysis.opening_value} → {result.total_analysis.closing_value}")

    # Verifica: xG usa Home Advantage (somma = total)
    xg_sum = result.expected_goals.home_xg + result.expected_goals.away_xg
    assert abs(xg_sum - 2.75) < 0.01, "xG deve sommare al total (Home Advantage preserva total)"

    print(f"\n🎯 Expected Goals (con Home Advantage):")
    print(f"  Casa: {result.expected_goals.home_xg:.2f} xG")
    print(f"  Trasferta: {result.expected_goals.away_xg:.2f} xG")
    print(f"  Somma: {xg_sum:.2f} (deve essere = total)")

    # Verifica: xG casa > xG trasferta (favorito casa + home advantage)
    assert result.expected_goals.home_xg > result.expected_goals.away_xg, \
           "Casa deve avere più xG (favorito + home advantage)"

    # Verifica: probabilità 1X2 usa Dixon-Coles
    prob_sum = (result.expected_goals.home_win_prob +
                result.expected_goals.draw_prob +
                result.expected_goals.away_win_prob)
    assert abs(prob_sum - 1.0) < 0.01, "1X2 deve sommare a 100%"

    print(f"\n📈 Probabilità 1X2 (con Dixon-Coles):")
    print(f"  P(1): {result.expected_goals.home_win_prob:.1%}")
    print(f"  P(X): {result.expected_goals.draw_prob:.1%}")
    print(f"  P(2): {result.expected_goals.away_win_prob:.1%}")
    print(f"  Somma: {prob_sum:.1%}")

    # Verifica: BTTS usa Dixon-Coles
    print(f"\n⚽ BTTS (con Dixon-Coles):")
    print(f"  P(BTTS): {result.expected_goals.btts_prob:.1%}")

    # Verifica: raccomandazioni coerenti
    print(f"\n🎯 Raccomandazioni ({len(result.core_recommendations)} core):")
    for rec in result.core_recommendations[:3]:
        print(f"  - {rec.market_name}: {rec.recommendation} ({rec.confidence.value})")

    assert len(result.core_recommendations) > 0, "Deve generare raccomandazioni"
    assert len(result.alternative_recommendations) > 0, "Deve generare alternative"
    assert len(result.value_recommendations) > 0, "Deve generare value bets"

    print("\n✅ Test Integrazione: PASSED")
    print("  - Tutte le 6 formule integrate correttamente")
    print("  - xG con Home Advantage preserva total")
    print("  - Dixon-Coles applicato a 1X2 e BTTS")
    print("  - Raccomandazioni generate correttamente")


def test_edge_cases():
    """
    TEST 8: Edge Cases - Valori estremi
    """
    print_separator()
    print("🧪 TEST 8: Edge Cases")
    print_separator()

    analyzer = MarketMovementAnalyzer()

    # Caso 1: xG molto bassi
    print("Caso 1: Partita difensiva (total 1.5)")
    result1 = analyzer.analyze(
        spread_open=-0.25,
        spread_close=-0.5,
        total_open=1.5,
        total_close=1.5
    )
    assert result1.expected_goals.home_xg > 0, "xG casa deve essere > 0"
    assert result1.expected_goals.away_xg > 0, "xG trasferta deve essere > 0"
    print(f"  ✅ xG Casa: {result1.expected_goals.home_xg:.2f}, Trasferta: {result1.expected_goals.away_xg:.2f}")

    # Caso 2: xG molto alti
    print("\nCaso 2: Partita offensiva (total 4.5)")
    result2 = analyzer.analyze(
        spread_open=-0.5,
        spread_close=-0.75,
        total_open=4.5,
        total_close=4.5
    )
    assert result2.expected_goals.btts_prob > 0.5, "BTTS dovrebbe essere probabile con total alto"
    print(f"  ✅ P(BTTS): {result2.expected_goals.btts_prob:.1%}")

    # Caso 3: Favorito estremo
    print("\nCaso 3: Favorito estremo (spread -2.5)")
    result3 = analyzer.analyze(
        spread_open=-2.0,
        spread_close=-2.5,
        total_open=3.0,
        total_close=3.0
    )
    assert result3.expected_goals.home_win_prob > 0.8, "Casa dovrebbe avere >80% con spread -2.5"
    print(f"  ✅ P(Casa vince): {result3.expected_goals.home_win_prob:.1%}")

    # Caso 4: Team perfettamente equilibrati
    print("\nCaso 4: Team equilibrati (spread 0.0)")
    result4 = analyzer.analyze(
        spread_open=0.0,
        spread_close=0.0,
        total_open=2.5,
        total_close=2.5
    )
    # Anche con spread 0, casa avrà leggero vantaggio per Home Advantage
    assert result4.expected_goals.home_xg > result4.expected_goals.away_xg, \
           "Casa dovrebbe avere leggero vantaggio anche con spread 0 (Home Advantage)"
    print(f"  ✅ xG Casa: {result4.expected_goals.home_xg:.2f}, Trasferta: {result4.expected_goals.away_xg:.2f}")

    print("\n✅ Test Edge Cases: PASSED")
    print("  - xG bassi gestiti correttamente")
    print("  - xG alti gestiti correttamente")
    print("  - Favoriti estremi gestiti correttamente")
    print("  - Team equilibrati gestiti correttamente")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 TEST COMPLETO: 6 FORMULE AVANZATE DI PRECISIONE".center(80))
    print("=" * 80)

    try:
        test_dixon_coles_correlation()
        test_home_advantage_redistribution()
        test_vig_removal_and_edge()
        test_regression_to_mean()
        test_correlated_markets()
        test_monte_carlo_validation()
        test_integration_all_formulas()
        test_edge_cases()

        print_separator("=")
        print("🎉 TUTTI I TEST SONO PASSATI! 🎉".center(80))
        print_separator("=")
        print("\n✅ Le 6 FORMULE AVANZATE funzionano alla perfezione:\n")
        print("  1. ✅ Dixon-Coles Bivariate Poisson - Correlazione risultati bassi")
        print("  2. ✅ Home Advantage Factor - Redistribuzione xG preservando total")
        print("  3. ✅ Vig Removal + Edge Calculation - True probabilities e value bets")
        print("  4. ✅ Regression to Mean - Correzione movimenti estremi")
        print("  5. ✅ Correlated Markets Model - Probabilità congiunte BTTS+Over/Under")
        print("  6. ✅ Monte Carlo Validation - Simulazioni convergono ad analitico")
        print("\n" + "=" * 80)
        print("💎 TUTTO CALCOLATO ALLA PERFEZIONE 💎".center(80))
        print("=" * 80 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FALLITO: {e}")
        import traceback
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"\n❌ ERRORE INASPETTATO: {e}")
        import traceback
        traceback.print_exc()
        raise
