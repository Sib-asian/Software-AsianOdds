#!/usr/bin/env python3
"""
AUDIT COMPLETO: Cerca tutte le incoerenze e problemi nei calcoli
"""

import math
from market_movement_analyzer import (
    MarketMovementAnalyzer,
    calculate_expected_goals,
    poisson_probability,
    dixon_coles_probability
)


def test_probability_normalization():
    """Verifica che tutte le probabilità sommino a 1.0"""
    print("\n" + "="*70)
    print("TEST 1: NORMALIZZAZIONE PROBABILITÀ")
    print("="*70)

    analyzer = MarketMovementAnalyzer()

    test_cases = [
        {"spread": -0.5, "total": 2.5, "desc": "Match equilibrato"},
        {"spread": -2.0, "total": 2.5, "desc": "Favorito forte"},
        {"spread": -1.0, "total": 3.5, "desc": "Total alto"},
        {"spread": -0.25, "total": 2.0, "desc": "Total basso"},
    ]

    all_ok = True

    for case in test_cases:
        print(f"\n[{case['desc']}] spread={case['spread']}, total={case['total']}")

        result = analyzer.analyze(
            spread_open=case['spread'],
            spread_close=case['spread'],
            total_open=case['total'],
            total_close=case['total']
        )

        # Check 1: 1X2 probabilities sum to 1.0
        xg = result.expected_goals
        prob_sum_1x2 = xg.home_win_prob + xg.draw_prob + xg.away_win_prob

        print(f"  1X2 sum: {prob_sum_1x2:.6f} ", end="")
        if abs(prob_sum_1x2 - 1.0) < 0.01:
            print("✅")
        else:
            print(f"❌ (errore: {prob_sum_1x2 - 1.0:+.6f})")
            all_ok = False

        # Check 2: Market-Adjusted 1X2 (if available)
        if xg.market_adjusted_1x2:
            adj = xg.market_adjusted_1x2
            adj_sum = adj['home_win'] + adj['draw'] + adj['away_win']

            print(f"  Market-Adjusted 1X2 sum: {adj_sum:.6f} ", end="")
            if abs(adj_sum - 1.0) < 0.01:
                print("✅")
            else:
                print(f"❌ (errore: {adj_sum - 1.0:+.6f})")
                all_ok = False

        # Check 3: BTTS probability in valid range
        btts = xg.btts_prob
        print(f"  BTTS prob: {btts:.1%} ", end="")
        if 0 <= btts <= 1:
            print("✅")
        else:
            print(f"❌ (fuori range [0, 1])")
            all_ok = False

        # Check 4: Bayesian BTTS (if available)
        if xg.bayesian_btts:
            bay_btts = xg.bayesian_btts['btts_prob']
            print(f"  Bayesian BTTS: {bay_btts:.1%} ", end="")
            if 0 <= bay_btts <= 1:
                print("✅")
            else:
                print(f"❌ (fuori range [0, 1])")
                all_ok = False

    print(f"\n{'✅ TUTTI I TEST OK' if all_ok else '❌ ERRORI RILEVATI'}")


def test_logical_coherence():
    """Verifica coerenza logica: spread aumenta → home_win_prob aumenta"""
    print("\n" + "="*70)
    print("TEST 2: COERENZA LOGICA (MONOTONIA)")
    print("="*70)

    analyzer = MarketMovementAnalyzer()

    print("\n[Test] Spread aumenta (favorito si rafforza) → Home win prob aumenta")

    spreads = [-0.5, -1.0, -1.5, -2.0, -2.5]
    total = 2.5

    prev_home_prob = 0
    all_ok = True

    for spread in spreads:
        result = analyzer.analyze(
            spread_open=spread,
            spread_close=spread,
            total_open=total,
            total_close=total
        )

        home_prob = result.expected_goals.home_win_prob

        print(f"  Spread {spread:+.1f} → Home win: {home_prob:.1%} ", end="")

        if home_prob > prev_home_prob:
            print("✅")
        else:
            print(f"❌ (non aumenta, prev={prev_home_prob:.1%})")
            all_ok = False

        prev_home_prob = home_prob

    print(f"\n[Test] Total aumenta → BTTS prob aumenta (spread fisso)")

    totals = [2.0, 2.5, 3.0, 3.5, 4.0]
    spread = -1.0

    prev_btts = 0

    for tot in totals:
        result = analyzer.analyze(
            spread_open=spread,
            spread_close=spread,
            total_open=tot,
            total_close=tot
        )

        btts = result.expected_goals.btts_prob

        print(f"  Total {tot:.1f} → BTTS: {btts:.1%} ", end="")

        if btts > prev_btts:
            print("✅")
        else:
            print(f"❌ (non aumenta, prev={prev_btts:.1%})")
            all_ok = False

        prev_btts = btts

    print(f"\n{'✅ TUTTI I TEST OK' if all_ok else '❌ ERRORI RILEVATI'}")


def test_edge_cases():
    """Testa valori estremi"""
    print("\n" + "="*70)
    print("TEST 3: EDGE CASES (VALORI ESTREMI)")
    print("="*70)

    analyzer = MarketMovementAnalyzer()

    edge_cases = [
        {"spread": -5.0, "total": 5.0, "desc": "Favorito fortissimo"},
        {"spread": -0.1, "total": 2.0, "desc": "Quasi pareggio, total basso"},
        {"spread": -1.0, "total": 1.5, "desc": "Total molto basso"},
        {"spread": -1.0, "total": 5.0, "desc": "Total molto alto"},
    ]

    all_ok = True

    for case in edge_cases:
        print(f"\n[{case['desc']}] spread={case['spread']}, total={case['total']}")

        try:
            result = analyzer.analyze(
                spread_open=case['spread'],
                spread_close=case['spread'],
                total_open=case['total'],
                total_close=case['total']
            )

            xg = result.expected_goals

            # Verifica xG positivi
            if xg.home_xg < 0 or xg.away_xg < 0:
                print(f"  ❌ xG negativi: home={xg.home_xg:.2f}, away={xg.away_xg:.2f}")
                all_ok = False
            else:
                print(f"  ✅ xG positivi: home={xg.home_xg:.2f}, away={xg.away_xg:.2f}")

            # Verifica probabilità valide
            if not (0 <= xg.home_win_prob <= 1 and 0 <= xg.draw_prob <= 1 and 0 <= xg.away_win_prob <= 1):
                print(f"  ❌ Probabilità fuori range")
                all_ok = False
            else:
                print(f"  ✅ Probabilità in range [0, 1]")

            # Verifica raccomandazioni presenti
            if len(result.core_recommendations) > 0:
                print(f"  ✅ Raccomandazioni generate ({len(result.core_recommendations)})")
            else:
                print(f"  ❌ Nessuna raccomandazione generata")
                all_ok = False

        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            all_ok = False

    print(f"\n{'✅ TUTTI I TEST OK' if all_ok else '❌ ERRORI RILEVATI'}")


def test_btts_consistency():
    """Verifica che BTTS = 1 - P(0-0) - P(X-0) - P(0-X)"""
    print("\n" + "="*70)
    print("TEST 4: CONSISTENZA BTTS (BTTS vs Clean Sheet)")
    print("="*70)

    test_cases = [
        {"spread": -1.0, "total": 2.5},
        {"spread": -2.0, "total": 2.5},
        {"spread": -0.5, "total": 3.0},
    ]

    all_ok = True

    for case in test_cases:
        print(f"\nSpread={case['spread']}, Total={case['total']}")

        xg_result = calculate_expected_goals(case['spread'], case['total'], use_advanced_formulas=True)

        home_xg = xg_result.home_xg
        away_xg = xg_result.away_xg

        # Calcola P(almeno una squadra non segna) manualmente
        home_cs = xg_result.home_clean_sheet_prob
        away_cs = xg_result.away_clean_sheet_prob

        # BTTS calcolato
        btts_calculated = xg_result.btts_prob

        # BTTS atteso (da clean sheets)
        # P(BTTS) = 1 - P(home non segna) - P(away non segna) + P(entrambi non segnano)
        # P(BTTS) = (1 - away_cs) * (1 - home_cs)
        btts_from_cs = (1 - home_cs) * (1 - away_cs)

        print(f"  Home clean sheet: {home_cs:.1%}")
        print(f"  Away clean sheet: {away_cs:.1%}")
        print(f"  BTTS (calcolato):      {btts_calculated:.1%}")
        print(f"  BTTS (da clean sheet): {btts_from_cs:.1%}")

        diff = abs(btts_calculated - btts_from_cs)
        print(f"  Differenza: {diff:.1%} ", end="")

        # Tolleranza maggiore per Dixon-Coles (considera correlazione)
        if diff < 0.15:
            print("✅")
        else:
            print(f"❌ (troppo grande)")
            all_ok = False

    print(f"\n{'✅ TUTTI I TEST OK' if all_ok else '❌ ERRORI RILEVATI'}")


def test_poisson_dixon_coles_consistency():
    """Verifica che Dixon-Coles converga a Poisson per xG alti"""
    print("\n" + "="*70)
    print("TEST 5: DIXON-COLES vs POISSON (alti xG)")
    print("="*70)

    print("\nPer xG alti (>2.0), Dixon-Coles dovrebbe convergere a Poisson")

    test_scores = [(2, 2), (3, 1), (1, 3), (4, 2)]
    home_xg = 2.5
    away_xg = 2.0

    all_ok = True

    for h, a in test_scores:
        poiss = poisson_probability(h, home_xg) * poisson_probability(a, away_xg)
        dixon = dixon_coles_probability(h, a, home_xg, away_xg)

        diff = abs(poiss - dixon)
        diff_pct = (diff / poiss * 100) if poiss > 0 else 0

        print(f"  P({h}-{a}): Poisson={poiss:.4f}, Dixon-Coles={dixon:.4f}, diff={diff_pct:.1f}% ", end="")

        if diff_pct < 20:  # Tolleranza 20%
            print("✅")
        else:
            print(f"❌ (differenza troppo grande)")
            all_ok = False

    print("\nPer risultati bassi (0-0, 1-0, 0-1, 1-1), Dixon-Coles applica correzione")

    test_low_scores = [(0, 0), (1, 0), (0, 1), (1, 1)]

    for h, a in test_low_scores:
        poiss = poisson_probability(h, home_xg) * poisson_probability(a, away_xg)
        dixon = dixon_coles_probability(h, a, home_xg, away_xg)

        diff = abs(poiss - dixon)
        diff_pct = (diff / poiss * 100) if poiss > 0 else 0

        print(f"  P({h}-{a}): Poisson={poiss:.4f}, Dixon-Coles={dixon:.4f}, diff={diff_pct:.1f}% ", end="")

        # Per risultati bassi, ci aspettiamo differenze
        if diff > 0:
            print("✅ (correzione applicata)")
        else:
            print(f"⚠️ (nessuna correzione, potrebbe essere ok)")

    print(f"\n{'✅ TUTTI I TEST OK' if all_ok else '❌ ERRORI RILEVATI'}")


def test_movement_detection():
    """Verifica che i movimenti siano rilevati correttamente"""
    print("\n" + "="*70)
    print("TEST 6: RILEVAMENTO MOVIMENTI")
    print("="*70)

    analyzer = MarketMovementAnalyzer()

    print("\n[Test] Movimento spread significativo")
    result = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-2.0,  # Movimento -1.0 (significativo)
        total_open=2.5,
        total_close=2.5
    )

    spread_mov = result.spread_analysis
    movement_size = spread_mov.closing_value - spread_mov.opening_value
    print(f"  Movimento: {movement_size:+.2f}")
    print(f"  Direzione: {spread_mov.direction.value}")
    print(f"  Steps: {spread_mov.movement_steps:.2f}")
    print(f"  Intensità: {spread_mov.intensity.value}")

    if abs(movement_size) > 0.5:
        print("  ✅ Movimento rilevato correttamente")
    else:
        print("  ❌ Movimento non rilevato")

    print("\n[Test] Movimento total significativo")
    result2 = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.0,
        total_open=2.5,
        total_close=3.5  # Movimento +1.0 (significativo)
    )

    total_mov = result2.total_analysis
    movement_size2 = total_mov.closing_value - total_mov.opening_value
    print(f"  Movimento: {movement_size2:+.2f}")
    print(f"  Direzione: {total_mov.direction.value}")
    print(f"  Steps: {total_mov.movement_steps:.2f}")

    if abs(movement_size2) > 0.5:
        print("  ✅ Movimento rilevato correttamente")
    else:
        print("  ❌ Movimento non rilevato")


def run_all_audits():
    """Esegue tutti gli audit"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "AUDIT COMPLETO SISTEMA" + " "*30 + "║")
    print("╚" + "="*68 + "╝")

    test_probability_normalization()
    test_logical_coherence()
    test_edge_cases()
    test_btts_consistency()
    test_poisson_dixon_coles_consistency()
    test_movement_detection()

    print("\n" + "="*70)
    print("AUDIT COMPLETATO!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_audits()
