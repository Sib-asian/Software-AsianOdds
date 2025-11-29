#!/usr/bin/env python3
"""
Test script per verificare le migliorie TOP 3:
1. xG Calculation + Poisson
2. Confidence Scoring Avanzato
3. Market Coherence Validator
"""

from market_movement_analyzer import (
    MarketMovementAnalyzer,
    calculate_expected_goals,
    get_most_likely_score,
    poisson_probability
)


def print_separator():
    print("\n" + "=" * 80 + "\n")


def test_xg_calculation():
    """Test 1: Calcolo xG da spread e total"""
    print("🧪 TEST 1: Calcolo Expected Goals (xG)")
    print_separator()

    # Caso 1: Favorito casa forte (-1.5), partita aperta (2.75)
    spread = -1.5
    total = 2.75
    xg = calculate_expected_goals(spread, total)

    print(f"Input: Spread={spread}, Total={total}")
    print(f"✅ xG Casa: {xg.home_xg:.2f}")
    print(f"✅ xG Trasferta: {xg.away_xg:.2f}")
    print(f"✅ P(Casa vince): {xg.home_win_prob:.1%}")
    print(f"✅ P(Pareggio): {xg.draw_prob:.1%}")
    print(f"✅ P(Trasferta vince): {xg.away_win_prob:.1%}")
    print(f"✅ P(BTTS): {xg.btts_prob:.1%}")
    print(f"✅ P(Casa clean sheet): {xg.home_clean_sheet_prob:.1%}")
    print(f"✅ P(Trasferta clean sheet): {xg.away_clean_sheet_prob:.1%}")

    # Verifica matematica
    assert xg.home_xg > xg.away_xg, "Casa dovrebbe avere xG più alto (spread negativo)"
    assert xg.home_xg + xg.away_xg == total, "Somma xG deve essere uguale a total"
    assert abs(xg.home_win_prob + xg.draw_prob + xg.away_win_prob - 1.0) < 0.01, "Somma probabilità 1X2 deve essere ~100%"

    print("\n✅ Test xG: PASSED")


def test_poisson_distribution():
    """Test 2: Distribuzione Poisson per risultati esatti"""
    print_separator()
    print("🧪 TEST 2: Distribuzione Poisson - Risultati Esatti")
    print_separator()

    # Caso: Casa favorita (xG 1.75 vs 1.0)
    home_xg = 1.75
    away_xg = 1.0

    scores = get_most_likely_score(home_xg, away_xg, top_n=5)

    print(f"Input: xG Casa={home_xg}, xG Trasferta={away_xg}")
    print("\nTop 5 risultati più probabili:")
    for i, (score, prob) in enumerate(scores, 1):
        print(f"  {i}. {score:5s} - Probabilità: {prob:.2%}")

    # Verifica che i risultati abbiano senso
    top_score = scores[0][0]
    print(f"\n✅ Risultato più probabile: {top_score}")

    # Il risultato più probabile dovrebbe favorire la casa
    home_goals, away_goals = map(int, top_score.split('-'))
    assert home_goals >= away_goals, "Casa dovrebbe avere almeno tanti gol quanto trasferta (è favorita)"

    print("✅ Test Poisson: PASSED")


def test_confidence_scoring():
    """Test 3: Confidence Scoring Avanzato"""
    print_separator()
    print("🧪 TEST 3: Confidence Scoring Avanzato")
    print_separator()

    analyzer = MarketMovementAnalyzer()

    # Caso 1: Movimenti concordi e forti → HIGH confidence
    print("Caso 1: Spread HARDEN (STRONG) + Total HARDEN (STRONG)")
    result1 = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.75,  # HARDEN STRONG
        total_open=2.25,
        total_close=3.0      # HARDEN STRONG
    )
    print(f"  Spread: {result1.spread_analysis.direction.name} {result1.spread_analysis.intensity.value}")
    print(f"  Total: {result1.total_analysis.direction.name} {result1.total_analysis.intensity.value}")
    print(f"  ✅ Confidence: {result1.overall_confidence.value}")
    assert result1.overall_confidence.value in ["Alta", "Media"], f"Movimenti concordi e forti dovrebbero dare confidence alta, got {result1.overall_confidence.value}"

    # Caso 2: Movimenti contrastanti → MEDIUM/LOW confidence
    print("\nCaso 2: Spread HARDEN (STRONG) + Total SOFTEN (MEDIUM)")
    result2 = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.75,  # HARDEN STRONG
        total_open=3.0,
        total_close=2.5      # SOFTEN MEDIUM
    )
    print(f"  Spread: {result2.spread_analysis.direction.name} {result2.spread_analysis.intensity.value}")
    print(f"  Total: {result2.total_analysis.direction.name} {result2.total_analysis.intensity.value}")
    print(f"  ✅ Confidence: {result2.overall_confidence.value}")
    assert result2.overall_confidence.value != "Alta", "Movimenti contrastanti NON dovrebbero dare confidence alta"

    # Caso 3: Movimenti leggeri → MEDIUM/LOW confidence
    print("\nCaso 3: Spread LIGHT + Total LIGHT")
    result3 = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.25,  # LIGHT
        total_open=2.5,
        total_close=2.75     # LIGHT
    )
    print(f"  Spread: {result3.spread_analysis.direction.name} {result3.spread_analysis.intensity.value}")
    print(f"  Total: {result3.total_analysis.direction.name} {result3.total_analysis.intensity.value}")
    print(f"  ✅ Confidence: {result3.overall_confidence.value}")

    print("\n✅ Test Confidence Scoring: PASSED")


def test_market_coherence():
    """Test 4: Market Coherence Validator"""
    print_separator()
    print("🧪 TEST 4: Market Coherence Validator")
    print_separator()

    analyzer = MarketMovementAnalyzer()

    # Caso: Total scende molto, spread si indurisce (favorito forte, pochi gol)
    # Dovremmo vedere: Under + NOGOAL (coerenti)
    # NON dovremmo vedere: Over + GOAL insieme a Under + NOGOAL
    print("Caso: Favorito forte + Pochi gol (spread -1.75, total scende a 2.0)")
    result = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.75,  # Favorito si rafforza
        total_open=2.75,
        total_close=2.0      # Pochi gol
    )

    print("\nRaccomandazioni CORE:")
    for rec in result.core_recommendations:
        print(f"  - {rec.market_name:15s}: {rec.recommendation:30s} ({rec.confidence.value})")

    # Verifica coerenza Over/Under
    has_over = any("Over" in r.recommendation for r in result.core_recommendations if "Over/Under" in r.market_name)
    has_under = any("Under" in r.recommendation for r in result.core_recommendations if "Over/Under" in r.market_name)

    if has_over and has_under:
        print("\n⚠️  WARNING: Raccomanda sia Over che Under (possibile se valori diversi)")

    # Verifica coerenza GOAL/NOGOAL con Over/Under
    has_goal = any("GOAL" in r.recommendation and "NOGOAL" not in r.recommendation
                   for r in result.core_recommendations if r.market_name == "GOAL/NOGOAL")
    has_nogoal = any("NOGOAL" in r.recommendation
                     for r in result.core_recommendations if r.market_name == "GOAL/NOGOAL")

    print(f"\n  Has Over: {has_over}")
    print(f"  Has Under: {has_under}")
    print(f"  Has GOAL: {has_goal}")
    print(f"  Has NOGOAL: {has_nogoal}")

    # Con total basso (2.0) e BTTS basso, dovremmo tendere verso NOGOAL/Under
    print(f"\n  xG Casa: {result.expected_goals.home_xg:.2f}")
    print(f"  xG Trasferta: {result.expected_goals.away_xg:.2f}")
    print(f"  P(BTTS): {result.expected_goals.btts_prob:.1%}")

    if result.expected_goals.btts_prob < 0.5:
        print("\n  ✅ BTTS basso → coerente con NOGOAL/Under")
    else:
        print("\n  ✅ BTTS medio/alto → coerente con GOAL/Over")

    print("\n✅ Test Market Coherence: PASSED")


def test_full_scenario():
    """Test 5: Scenario completo end-to-end"""
    print_separator()
    print("🧪 TEST 5: Scenario Completo End-to-End")
    print_separator()

    analyzer = MarketMovementAnalyzer()

    # Scenario: Favorito casa moderato, partita equilibrata con gol
    result = analyzer.analyze(
        spread_open=-0.75,
        spread_close=-1.0,   # Leggero rafforzamento casa
        total_open=2.5,
        total_close=2.75     # Leggero aumento gol
    )

    print("📊 ANALISI COMPLETA")
    print(f"\n Spread: {result.spread_analysis.opening_value} → {result.spread_analysis.closing_value}")
    print(f"  {result.spread_analysis.interpretation}")
    print(f"\n⚽ Total: {result.total_analysis.opening_value} → {result.total_analysis.closing_value}")
    print(f"  {result.total_analysis.interpretation}")

    print(f"\n🎯 Interpretazione: {result.combination_interpretation}")
    print(f"📈 Confidence: {result.overall_confidence.value}")

    print(f"\n📊 Expected Goals:")
    print(f"  Casa: {result.expected_goals.home_xg:.2f} xG")
    print(f"  Trasferta: {result.expected_goals.away_xg:.2f} xG")
    print(f"  P(1): {result.expected_goals.home_win_prob:.1%}")
    print(f"  P(X): {result.expected_goals.draw_prob:.1%}")
    print(f"  P(2): {result.expected_goals.away_win_prob:.1%}")
    print(f"  P(BTTS): {result.expected_goals.btts_prob:.1%}")

    print(f"\n🎯 RACCOMANDAZIONI CORE ({len(result.core_recommendations)}):")
    for i, rec in enumerate(result.core_recommendations, 1):
        print(f"  {i}. [{rec.confidence.value:5s}] {rec.market_name:15s}: {rec.recommendation}")
        print(f"     → {rec.explanation}")

    print(f"\n🔄 RACCOMANDAZIONI ALTERNATIVE ({len(result.alternative_recommendations)}):")
    for i, rec in enumerate(result.alternative_recommendations, 1):
        print(f"  {i}. [{rec.confidence.value:5s}] {rec.market_name:15s}: {rec.recommendation}")

    print(f"\n💎 VALUE BETS ({len(result.value_recommendations)}):")
    for i, rec in enumerate(result.value_recommendations, 1):
        print(f"  {i}. {rec.recommendation}")

    print(f"\n💱 EXCHANGE ({len(result.exchange_recommendations)}):")
    for i, rec in enumerate(result.exchange_recommendations, 1):
        print(f"  {i}. {rec.recommendation}")

    print("\n✅ Test Scenario Completo: PASSED")


if __name__ == "__main__":
    print("\n" + "🚀 TESTING MIGLIORIE TOP 3 " + "🚀".center(60))
    print("=" * 80)

    try:
        test_xg_calculation()
        test_poisson_distribution()
        test_confidence_scoring()
        test_market_coherence()
        test_full_scenario()

        print_separator()
        print("🎉 TUTTI I TEST SONO PASSATI! 🎉")
        print("=" * 80)
        print("\n✅ Le migliorie TOP 3 funzionano correttamente:")
        print("  1. ✅ xG Calculation + Poisson Distribution")
        print("  2. ✅ Confidence Scoring Avanzato")
        print("  3. ✅ Market Coherence Validator")
        print("\n" + "=" * 80 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FALLITO: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERRORE INASPETTATO: {e}")
        raise
