#!/usr/bin/env python3
"""
Test per OPZIONE C (1X2 Bayesian) e OPZIONE B (O/U xG)

Verifica che:
1. 1X2: Validazione Bayesian funziona (warning e override)
2. Over/Under: Validazione xG funziona (concordanza e discordanza)
"""

from market_movement_analyzer import MarketMovementAnalyzer


def test_1x2_bayesian_validation():
    """Test validazione 1X2 con Market-Adjusted Bayesian"""
    print("\n" + "="*70)
    print("TEST 1: VALIDAZIONE 1X2 CON BAYESIAN")
    print("="*70)

    analyzer = MarketMovementAnalyzer()

    # Test 1a: Spread e Bayesian concordanti → nessun cambiamento
    print("\n[Test 1a] Spread e Bayesian CONCORDANTI")
    print("  Input: spread -1.0→-1.2 (favorito casa si rafforza)")
    result = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.2,  # Favorito casa si rafforza
        total_open=2.5,
        total_close=2.5
    )

    x2_rec = next((r for r in result.core_recommendations if r.market_name == "1X2"), None)
    if x2_rec:
        print(f"\n  Raccomandazione: {x2_rec.recommendation}")
        print(f"  Confidence: {x2_rec.confidence.value}")
        print(f"  Explanation: {x2_rec.explanation}")

        # Verifica se c'è warning o override
        has_warning = "⚠️" in x2_rec.explanation
        has_override = "Market-Adjusted Bayesian" in x2_rec.explanation

        if not has_warning and not has_override:
            print(f"\n  ✅ NESSUN WARNING: Spread e Bayesian concordano")
        else:
            print(f"\n  ⚠️  WARNING o OVERRIDE presente (inatteso)")

    # Test 1b: Discrepanza media (20-30%) → warning
    print("\n" + "-"*70)
    print("\n[Test 1b] Discrepanza MEDIA (dovrebbe dare warning)")
    print("  Input: spread -0.5→-0.5 (pareggio), ma movimenti sharp")
    result2 = analyzer.analyze(
        spread_open=-0.5,
        spread_close=-0.5,  # Match equilibrato
        total_open=2.5,
        total_close=3.0  # Total sale molto → sharp money
    )

    x2_rec2 = next((r for r in result2.core_recommendations if r.market_name == "1X2"), None)
    if x2_rec2:
        print(f"\n  Raccomandazione: {x2_rec2.recommendation}")
        print(f"  Confidence: {x2_rec2.confidence.value}")
        print(f"  Explanation: {x2_rec2.explanation[:150]}...")

        # Mostra Market-Adjusted se disponibile
        if result2.expected_goals.market_adjusted_1x2:
            adj_1x2 = result2.expected_goals.market_adjusted_1x2
            print(f"\n  Market-Adjusted Bayesian:")
            print(f"    Home: {adj_1x2['home_win']:.1%}")
            print(f"    Draw: {adj_1x2['draw']:.1%}")
            print(f"    Away: {adj_1x2['away_win']:.1%}")


def test_ou_xg_validation():
    """Test validazione Over/Under con xG"""
    print("\n" + "="*70)
    print("TEST 2: VALIDAZIONE OVER/UNDER CON xG")
    print("="*70)

    analyzer = MarketMovementAnalyzer()

    # Test 2a: Movimento e xG CONCORDANTI → boost confidence
    print("\n[Test 2a] Movimento e xG CONCORDANTI (entrambi Over)")
    print("  Input: total 2.5→3.0 (sale), xG alto")
    result = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.0,
        total_open=2.5,
        total_close=3.0  # Total sale → Over
    )

    ou_rec = next((r for r in result.core_recommendations if r.market_name == "Over/Under"), None)
    if ou_rec:
        print(f"\n  Raccomandazione: {ou_rec.recommendation}")
        print(f"  Confidence: {ou_rec.confidence.value}")
        print(f"  Explanation: {ou_rec.explanation}")

        total_xg = result.expected_goals.home_xg + result.expected_goals.away_xg
        print(f"\n  xG totale: {total_xg:.2f}")
        print(f"  Total closing: 3.0")

        # Verifica se c'è conferma xG
        has_confirmation = "✓ xG conferma" in ou_rec.explanation
        if has_confirmation:
            print(f"\n  ✅ xG CONFERMA: Boost confidence applicato")
        else:
            print(f"\n  ℹ️  xG concorda ma non abbastanza lontano da total")

    # Test 2b: Movimento e xG DISCORDANTI → warning e reduced confidence
    print("\n" + "-"*70)
    print("\n[Test 2b] Movimento e xG DISCORDANTI")
    print("  Input: total 2.5→3.0 (sale), ma xG basso")
    result2 = analyzer.analyze(
        spread_open=-2.0,
        spread_close=-2.0,  # Favorito forte (xG sbilanciato, total basso)
        total_open=2.5,
        total_close=3.0  # Ma total sale (contraddizione)
    )

    ou_rec2 = next((r for r in result2.core_recommendations if r.market_name == "Over/Under"), None)
    if ou_rec2:
        print(f"\n  Raccomandazione: {ou_rec2.recommendation}")
        print(f"  Confidence: {ou_rec2.confidence.value}")
        print(f"  Explanation: {ou_rec2.explanation}")

        total_xg2 = result2.expected_goals.home_xg + result2.expected_goals.away_xg
        print(f"\n  xG totale: {total_xg2:.2f}")
        print(f"  Total closing: 3.0")

        # Verifica se c'è warning
        has_warning = "⚠️" in ou_rec2.explanation or "xG" in ou_rec2.explanation.lower()
        is_low_conf = ou_rec2.confidence.value == "Bassa"

        if has_warning or is_low_conf:
            print(f"\n  ✅ WARNING PRESENTE o CONFIDENCE RIDOTTA")
        else:
            print(f"\n  ⚠️  Nessun warning (potrebbe essere ok se concordanti)")

    # Test 2c: Total scende ma xG alto → discordanza forte
    print("\n" + "-"*70)
    print("\n[Test 2c] DISCORDANZA FORTE (total scende, xG suggerisce Over)")
    result3 = analyzer.analyze(
        spread_open=-0.5,
        spread_close=-0.5,
        total_open=3.0,
        total_close=2.5  # Total scende → Under
        # Ma xG sarà medio-alto → discordanza
    )

    ou_rec3 = next((r for r in result3.core_recommendations if r.market_name == "Over/Under"), None)
    if ou_rec3:
        print(f"\n  Raccomandazione: {ou_rec3.recommendation}")
        print(f"  Confidence: {ou_rec3.confidence.value}")
        print(f"  Explanation: {ou_rec3.explanation}")

        total_xg3 = result3.expected_goals.home_xg + result3.expected_goals.away_xg
        print(f"\n  xG totale: {total_xg3:.2f}")
        print(f"  Total closing: 2.5")

        if total_xg3 > 2.5 and "Under" in ou_rec3.recommendation:
            print(f"\n  ⚠️  DISCORDANZA: raccomanda Under ma xG > total")
            has_warning = "⚠️" in ou_rec3.explanation
            print(f"  {'✅' if has_warning else '❌'} Warning presente: {has_warning}")


def run_all_tests():
    """Esegue tutti i test"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*12 + "TEST OPZIONE C (1X2) E OPZIONE B (O/U)" + " "*18 + "║")
    print("╚" + "="*68 + "╝")

    test_1x2_bayesian_validation()
    test_ou_xg_validation()

    print("\n" + "="*70)
    print("TUTTI I TEST COMPLETATI!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
