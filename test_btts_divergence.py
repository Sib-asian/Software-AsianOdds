#!/usr/bin/env python3
"""
Test per verificare che BTTS Bayesian diverga dal base quando ci sono
movimenti di mercato significativi
"""

from market_movement_analyzer import MarketMovementAnalyzer


def test_btts_divergence():
    """
    Test con movimenti di mercato forti per vedere se BTTS Bayesian
    si discosta dal base Poisson
    """
    print("\n" + "="*70)
    print("TEST DIVERGENZA: BTTS Bayesian vs Base con movimenti forti")
    print("="*70)

    analyzer = MarketMovementAnalyzer()

    # Test 1: Total sale molto (mercato si aspetta più gol → GG boost)
    print("\n[Test 1] TOTAL SALE MOLTO (2.0 → 3.0)")
    print("  Mercato si aspetta molti più gol → BTTS Bayesian dovrebbe salire")

    result = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.0,  # Spread stabile
        total_open=2.0,
        total_close=3.0     # +1.0 movimento (FORTE)
    )

    btts_base = result.expected_goals.btts_prob
    btts_bayesian = result.expected_goals.bayesian_btts['btts_prob'] if result.expected_goals.bayesian_btts else None

    print(f"\n  BTTS Base (Poisson):      {btts_base:.1%}")
    print(f"  BTTS Bayesian (Advanced): {btts_bayesian:.1%}" if btts_bayesian else "  N/A")

    if btts_bayesian:
        diff = btts_bayesian - btts_base
        print(f"  Differenza:               {diff:+.1%}")
        if diff > 0.05:
            print(f"  ✅ BOOST APPLICATO: Bayesian > Base (+{diff:.1%})")
        elif diff < -0.05:
            print(f"  ⚠️  PENALITÀ: Bayesian < Base ({diff:.1%})")
        else:
            print(f"  ⚠️  NESSUN CAMBIO SIGNIFICATIVO")

    # Test 2: Total scende molto (mercato si aspetta meno gol → GG penalità)
    print("\n" + "-"*70)
    print("\n[Test 2] TOTAL SCENDE MOLTO (3.5 → 2.5)")
    print("  Mercato si aspetta molti meno gol → BTTS Bayesian dovrebbe scendere")

    result2 = analyzer.analyze(
        spread_open=-0.5,
        spread_close=-0.5,  # Spread stabile
        total_open=3.5,
        total_close=2.5     # -1.0 movimento (FORTE)
    )

    btts_base2 = result2.expected_goals.btts_prob
    btts_bayesian2 = result2.expected_goals.bayesian_btts['btts_prob'] if result2.expected_goals.bayesian_btts else None

    print(f"\n  BTTS Base (Poisson):      {btts_base2:.1%}")
    print(f"  BTTS Bayesian (Advanced): {btts_bayesian2:.1%}" if btts_bayesian2 else "  N/A")

    if btts_bayesian2:
        diff2 = btts_bayesian2 - btts_base2
        print(f"  Differenza:               {diff2:+.1%}")
        if diff2 < -0.05:
            print(f"  ✅ PENALITÀ APPLICATA: Bayesian < Base ({diff2:.1%})")
        elif diff2 > 0.05:
            print(f"  ⚠️  BOOST: Bayesian > Base (+{diff2:.1%})")
        else:
            print(f"  ⚠️  NESSUN CAMBIO SIGNIFICATIVO")

    # Test 3: Spread chiude molto + Total alto (partita sbilanciata ma aperta)
    print("\n" + "-"*70)
    print("\n[Test 3] PARTITA SBILANCIATA MA APERTA")
    print("  Spread alto (-2.0) + Total alto (3.5) → BTTS complesso")

    result3 = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-2.0,  # Favorito si rafforza
        total_open=3.0,
        total_close=3.5     # Total sale
    )

    btts_base3 = result3.expected_goals.btts_prob
    btts_bayesian3 = result3.expected_goals.bayesian_btts['btts_prob'] if result3.expected_goals.bayesian_btts else None

    print(f"\n  BTTS Base (Poisson):      {btts_base3:.1%}")
    print(f"  BTTS Bayesian (Advanced): {btts_bayesian3:.1%}" if btts_bayesian3 else "  N/A")

    if btts_bayesian3:
        diff3 = btts_bayesian3 - btts_base3
        print(f"  Differenza:               {diff3:+.1%}")

        # Mostra i fattori Bayesian
        if result3.expected_goals.bayesian_btts:
            print(f"\n  Fattori Bayesian:")
            print(f"    Base BTTS:        {result3.expected_goals.bayesian_btts['base_btts']:.1%}")
            print(f"    Openness score:   {result3.expected_goals.bayesian_btts['openness_score']:.2f}")
            print(f"    Balance score:    {result3.expected_goals.bayesian_btts['balance_score']:.2f}")
            print(f"    Total boost:      {result3.expected_goals.bayesian_btts['total_boost']:+.1%}")

    # Test 4: Raccomandazione GOAL/NOGOAL coerente?
    print("\n" + "-"*70)
    print("\n[Test 4] VERIFICA RACCOMANDAZIONE")

    gg_rec = next((r for r in result.core_recommendations if r.market_name == "GOAL/NOGOAL"), None)
    if gg_rec and btts_bayesian:
        print(f"\n  Caso 1 (Total sale 2.0→3.0):")
        print(f"    BTTS Bayesian: {btts_bayesian:.1%}")
        print(f"    Raccomandazione: {gg_rec.recommendation}")
        print(f"    Explanation: {gg_rec.explanation}")

    gg_rec2 = next((r for r in result2.core_recommendations if r.market_name == "GOAL/NOGOAL"), None)
    if gg_rec2 and btts_bayesian2:
        print(f"\n  Caso 2 (Total scende 3.5→2.5):")
        print(f"    BTTS Bayesian: {btts_bayesian2:.1%}")
        print(f"    Raccomandazione: {gg_rec2.recommendation}")
        print(f"    Explanation: {gg_rec2.explanation}")

    print("\n" + "="*70)
    print("TEST COMPLETATO")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_btts_divergence()
