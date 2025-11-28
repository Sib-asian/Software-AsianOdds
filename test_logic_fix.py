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


if __name__ == "__main__":
    test_spread_soften_strong_favorite()
    test_spread_soften_balanced()
    test_spread_harden()

    print("\n\n" + "=" * 60)
    print("TUTTI I TEST COMPLETATI!")
    print("=" * 60)
