#!/usr/bin/env python3
"""
Test script per verificare la logica corretta delle raccomandazioni
"""

from market_movement_analyzer import MarketMovementAnalyzer

def test_spread_soften_strong_favorite():
    """Test: spread -1.5 → -1.25 (SOFTEN ma favorito ancora forte)"""
    print("=" * 60)
    print("TEST: Spread -1.5 → -1.25 (SOFTEN, favorito ancora forte)")
    print("=" * 60)

    analyzer = MarketMovementAnalyzer()
    result = analyzer.analyze(
        spread_open=-1.5,
        spread_close=-1.25,
        total_open=2.5,
        total_close=2.5
    )

    print(f"\n📊 SPREAD: {result.spread_analysis.direction.value}")
    print(f"   Opening: -1.5 → Closing: -1.25")
    print(f"   {result.spread_analysis.interpretation}")

    print(f"\n📈 TOTAL: {result.total_analysis.direction.value}")
    print(f"   Opening: 2.5 → Closing: 2.5")
    print(f"   {result.total_analysis.interpretation}")

    print("\n🎯 RACCOMANDAZIONI CORE:")
    for rec in result.core_recommendations:
        print(f"   [{rec.confidence.value}] {rec.market_name}: {rec.recommendation}")
        print(f"       → {rec.explanation}")

    print("\n💼 RACCOMANDAZIONI ALTERNATIVE:")
    for rec in result.alternative_recommendations:
        print(f"   [{rec.confidence.value}] {rec.market_name}: {rec.recommendation}")
        print(f"       → {rec.explanation}")

    print("\n💎 VALUE BETS:")
    for rec in result.value_recommendations:
        print(f"   [{rec.confidence.value}] {rec.market_name}: {rec.recommendation}")
        print(f"       → {rec.explanation}")

    # Verifica che 1X2 raccomandi "1 (Favorito)"
    x2_rec = [r for r in result.core_recommendations if r.market_name == "1X2"][0]
    if "1 (Favorito)" in x2_rec.recommendation and "X2" not in x2_rec.recommendation:
        print("\n✅ SUCCESSO: Raccomanda '1 (Favorito)' come previsto!")
        print(f"   Raccomandazione: {x2_rec.recommendation}")
        print(f"   Spiegazione: {x2_rec.explanation}")
    else:
        print("\n❌ ERRORE: Non raccomanda '1 (Favorito)'")
        print(f"   Raccomandazione attuale: {x2_rec.recommendation}")


def test_spread_soften_balanced():
    """Test: spread -1.0 → -0.75 (SOFTEN, match abbastanza equilibrato)"""
    print("\n\n" + "=" * 60)
    print("TEST: Spread -1.0 → -0.75 (SOFTEN, abbastanza equilibrato)")
    print("=" * 60)

    analyzer = MarketMovementAnalyzer()
    result = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-0.75,
        total_open=2.5,
        total_close=2.5
    )

    print(f"\n📊 SPREAD: {result.spread_analysis.direction.value}")
    print(f"   Opening: -1.0 → Closing: -0.75")
    print(f"   {result.spread_analysis.interpretation}")

    print("\n🎯 Raccomandazione 1X2:")
    x2_rec = [r for r in result.core_recommendations if r.market_name == "1X2"][0]
    print(f"   [{x2_rec.confidence.value}] {x2_rec.recommendation}")
    print(f"   → {x2_rec.explanation}")

    if "1X" in x2_rec.recommendation or "X" in x2_rec.recommendation:
        print("\n✅ SUCCESSO: Raccomanda pareggio/equilibrio come previsto!")
    else:
        print("\n❌ ERRORE: Non raccomanda equilibrio")


def test_spread_harden():
    """Test: spread -1.0 → -1.5 (HARDEN, favorito si rafforza)"""
    print("\n\n" + "=" * 60)
    print("TEST: Spread -1.0 → -1.5 (HARDEN, favorito si rafforza)")
    print("=" * 60)

    analyzer = MarketMovementAnalyzer()
    result = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.5,
        total_open=2.5,
        total_close=2.5
    )

    print(f"\n📊 SPREAD: {result.spread_analysis.direction.value}")
    print(f"   Opening: -1.0 → Closing: -1.5")
    print(f"   {result.spread_analysis.interpretation}")

    print("\n🎯 Raccomandazione 1X2:")
    x2_rec = [r for r in result.core_recommendations if r.market_name == "1X2"][0]
    print(f"   [{x2_rec.confidence.value}] {x2_rec.recommendation}")
    print(f"   → {x2_rec.explanation}")

    if "1 (Favorito)" in x2_rec.recommendation:
        print("\n✅ SUCCESSO: Raccomanda '1 (Favorito)' come previsto!")
    else:
        print("\n❌ ERRORE: Non raccomanda favorito")


def test_edge_case_pick_em():
    """Test: spread -0.15 → -0.20 (EDGE CASE: quasi pick'em)"""
    print("\n\n" + "=" * 60)
    print("TEST EDGE CASE: Spread -0.15 → -0.20 (quasi pick'em)")
    print("=" * 60)

    analyzer = MarketMovementAnalyzer()
    result = analyzer.analyze(
        spread_open=-0.15,
        spread_close=-0.20,
        total_open=2.5,
        total_close=2.5
    )

    print(f"\n📊 SPREAD: {result.spread_analysis.direction.value}")
    print(f"   Opening: -0.15 → Closing: -0.20")

    print("\n🎯 Raccomandazione 1X2:")
    x2_rec = [r for r in result.core_recommendations if r.market_name == "1X2"][0]
    print(f"   [{x2_rec.confidence.value}] {x2_rec.recommendation}")
    print(f"   → {x2_rec.explanation}")

    if "X o 12" in x2_rec.recommendation:
        print("\n✅ SUCCESSO: Edge case pick'em riconosciuto!")
    else:
        print("\n❌ ERRORE: Edge case pick'em NON riconosciuto")


def test_edge_case_dominant_favorite():
    """Test: spread -2.25 (EDGE CASE: favorito schiacciante)"""
    print("\n\n" + "=" * 60)
    print("TEST EDGE CASE: Spread -2.25 (favorito schiacciante)")
    print("=" * 60)

    analyzer = MarketMovementAnalyzer()
    result = analyzer.analyze(
        spread_open=-2.0,
        spread_close=-2.25,
        total_open=2.5,
        total_close=2.5
    )

    print(f"\n📊 SPREAD: {result.spread_analysis.direction.value}")
    print(f"   Opening: -2.0 → Closing: -2.25")

    print("\n🎯 Raccomandazione 1X2:")
    x2_rec = [r for r in result.core_recommendations if r.market_name == "1X2"][0]
    print(f"   [{x2_rec.confidence.value}] {x2_rec.recommendation}")
    print(f"   → {x2_rec.explanation}")

    if "dominante" in x2_rec.recommendation or "schiacciante" in x2_rec.explanation:
        print("\n✅ SUCCESSO: Edge case favorito schiacciante riconosciuto!")
    else:
        print("\n❌ ERRORE: Edge case favorito schiacciante NON riconosciuto")


def test_edge_case_very_low_total():
    """Test: total 1.5 (EDGE CASE: partita chiusissima)"""
    print("\n\n" + "=" * 60)
    print("TEST EDGE CASE: Total 1.5 (partita chiusissima)")
    print("=" * 60)

    analyzer = MarketMovementAnalyzer()
    result = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.0,
        total_open=1.75,
        total_close=1.5
    )

    print(f"\n📈 TOTAL: {result.total_analysis.direction.value}")
    print(f"   Opening: 1.75 → Closing: 1.5")

    print("\n🎯 Raccomandazione GOAL/NOGOAL:")
    goal_rec = [r for r in result.core_recommendations if r.market_name == "GOAL/NOGOAL"][0]
    print(f"   [{goal_rec.confidence.value}] {goal_rec.recommendation}")
    print(f"   → {goal_rec.explanation}")

    if ("chiusissima" in goal_rec.recommendation or "molto basso" in goal_rec.explanation) and goal_rec.confidence.value == "Alta":
        print("\n✅ SUCCESSO: Edge case total basso riconosciuto!")
    else:
        print("\n❌ ERRORE: Edge case total basso NON riconosciuto")


def test_contrasting_signals():
    """Test: spread HARDEN ma total SOFTEN (segnali contrastanti)"""
    print("\n\n" + "=" * 60)
    print("TEST: Segnali contrastanti (spread HARDEN, total SOFTEN)")
    print("=" * 60)

    analyzer = MarketMovementAnalyzer()
    result = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.5,  # HARDEN
        total_open=3.0,
        total_close=2.0     # SOFTEN
    )

    print(f"\n📊 SPREAD: {result.spread_analysis.direction.value}")
    print(f"📈 TOTAL: {result.total_analysis.direction.value}")
    print(f"\n✅ CONFIDENZA GENERALE: {result.overall_confidence.value}")

    if result.overall_confidence.value == "Media":
        print("\n✅ SUCCESSO: Segnali contrastanti → confidence MEDIA (mai HIGH)!")
    else:
        print(f"\n❌ ERRORE: Confidence dovrebbe essere MEDIA ma è {result.overall_confidence.value}")


if __name__ == "__main__":
    # Test originali
    test_spread_soften_strong_favorite()
    test_spread_soften_balanced()
    test_spread_harden()

    # Nuovi test edge cases
    test_edge_case_pick_em()
    test_edge_case_dominant_favorite()
    test_edge_case_very_low_total()
    test_contrasting_signals()

    print("\n\n" + "=" * 60)
    print("TUTTI I TEST COMPLETATI!")
    print("=" * 60)
