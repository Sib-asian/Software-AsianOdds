#!/usr/bin/env python3
"""
Test per verificare la CONSISTENZA tra BTTS Bayesian e raccomandazioni GOAL/NOGOAL
"""

from market_movement_analyzer import MarketMovementAnalyzer


def test_btts_consistency():
    """
    Verifica che le raccomandazioni GOAL/NOGOAL usino il BTTS Bayesian
    e non il BTTS base da Poisson
    """
    print("\n" + "="*70)
    print("TEST CONSISTENZA: BTTS Bayesian vs Raccomandazioni GOAL/NOGOAL")
    print("="*70)

    analyzer = MarketMovementAnalyzer()

    # Test caso 1: Spread forte, total medio → BTTS potrebbe essere basso
    print("\n[Test 1] Favorito forte, total medio")
    print("  Input: spread -1.5→-2.0, total 2.5→2.6")

    result = analyzer.analyze(
        spread_open=-1.5,
        spread_close=-2.0,
        total_open=2.5,
        total_close=2.6
    )

    # Estrai valori BTTS
    btts_base = result.expected_goals.btts_prob
    btts_bayesian = None
    if result.expected_goals.bayesian_btts:
        btts_bayesian = result.expected_goals.bayesian_btts['btts_prob']

    print(f"\n  BTTS Base (Poisson):    {btts_base:.1%}")
    print(f"  BTTS Bayesian (Advanced): {btts_bayesian:.1%}" if btts_bayesian else "  BTTS Bayesian: NON DISPONIBILE")

    # Trova raccomandazione GOAL/NOGOAL
    gg_rec = next((r for r in result.core_recommendations if r.market_name == "GOAL/NOGOAL"), None)

    if gg_rec:
        print(f"\n  Raccomandazione: {gg_rec.recommendation}")
        print(f"  Confidence: {gg_rec.confidence.value}")
        print(f"  Explanation: {gg_rec.explanation}")

        # Verifica consistenza
        if btts_bayesian:
            # Estrai P(BTTS) dalla explanation
            import re
            match = re.search(r'P\(BTTS\)=(\d+\.\d+)%', gg_rec.explanation)
            if match:
                btts_in_explanation = float(match.group(1)) / 100

                print(f"\n  P(BTTS) in explanation: {btts_in_explanation:.1%}")
                print(f"  P(BTTS) Bayesian:       {btts_bayesian:.1%}")

                # Verifica che usino lo stesso valore (tolleranza 1%)
                if abs(btts_in_explanation - btts_bayesian) < 0.01:
                    print(f"\n  ✅ CONSISTENZA OK: Usa BTTS Bayesian!")
                elif abs(btts_in_explanation - btts_base) < 0.01:
                    print(f"\n  ❌ ERRORE: Usa ancora BTTS base invece di Bayesian!")
                else:
                    print(f"\n  ⚠️  WARNING: Valore BTTS non corrisponde a nessuno dei due")

                # Verifica logica della raccomandazione
                if btts_bayesian >= 0.65:
                    expected = "GOAL"
                elif btts_bayesian <= 0.40:
                    expected = "NOGOAL"
                else:
                    expected = "GOAL" # Zona grigia favorisce GOAL

                if expected in gg_rec.recommendation:
                    print(f"  ✅ LOGICA OK: Con P(BTTS)={btts_bayesian:.1%} → raccomanda {expected}")
                else:
                    print(f"  ❌ LOGICA ERRATA: Con P(BTTS)={btts_bayesian:.1%} dovrebbe raccomandare {expected}")
    else:
        print("\n  ⚠️  GOAL/NOGOAL non presente nelle raccomandazioni")

    # Test caso 2: Partita aperta (total alto)
    print("\n" + "-"*70)
    print("\n[Test 2] Partita aperta, total alto")
    print("  Input: spread -0.5→-0.25, total 3.0→3.25")

    result2 = analyzer.analyze(
        spread_open=-0.5,
        spread_close=-0.25,
        total_open=3.0,
        total_close=3.25
    )

    btts_base2 = result2.expected_goals.btts_prob
    btts_bayesian2 = None
    if result2.expected_goals.bayesian_btts:
        btts_bayesian2 = result2.expected_goals.bayesian_btts['btts_prob']

    print(f"\n  BTTS Base (Poisson):    {btts_base2:.1%}")
    print(f"  BTTS Bayesian (Advanced): {btts_bayesian2:.1%}" if btts_bayesian2 else "  BTTS Bayesian: NON DISPONIBILE")

    gg_rec2 = next((r for r in result2.core_recommendations if r.market_name == "GOAL/NOGOAL"), None)

    if gg_rec2:
        print(f"\n  Raccomandazione: {gg_rec2.recommendation}")
        print(f"  Explanation: {gg_rec2.explanation}")

        if btts_bayesian2:
            import re
            match = re.search(r'P\(BTTS\)=(\d+\.\d+)%', gg_rec2.explanation)
            if match:
                btts_in_explanation2 = float(match.group(1)) / 100

                if abs(btts_in_explanation2 - btts_bayesian2) < 0.01:
                    print(f"\n  ✅ CONSISTENZA OK: Usa BTTS Bayesian ({btts_bayesian2:.1%})")
                else:
                    print(f"\n  ❌ ERRORE: Non usa BTTS Bayesian")

    print("\n" + "="*70)
    print("TEST COMPLETATO")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_btts_consistency()
