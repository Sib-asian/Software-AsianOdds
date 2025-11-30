#!/usr/bin/env python3
"""
AUDIT: Verifica coerenza tra spread, total e probabilità derivate

Controlla:
1. xG derivato da spread/total → riconverso a spread/total → match originale?
2. Probabilità 1X2 da spread → match probabilità xG?
3. Total implicito da xG → match total di mercato?
4. Arbitraggi o incoerenze tra mercati
"""

import math
from market_movement_analyzer import (
    MarketMovementAnalyzer,
    calculate_expected_goals,
    spread_to_implied_probability,
    poisson_probability
)


def reverse_engineer_spread_from_xg(home_xg: float, away_xg: float) -> float:
    """
    Dato xG, calcola quale spread implicherebbe.

    Logic (MATEMATICAMENTE CORRETTO):
    - Poiché: home_xg = (total - spread) / 2
    -         away_xg = (total + spread) / 2
    - Allora: home_xg - away_xg = -spread
    - Quindi: spread = away_xg - home_xg
    """
    return away_xg - home_xg


def reverse_engineer_total_from_xg(home_xg: float, away_xg: float) -> float:
    """
    Dato xG, calcola quale total implicherebbe.

    Logic:
    - xG totale ≈ total (dovrebbero essere molto vicini)
    """
    return home_xg + away_xg


def calculate_1x2_prob_from_xg(home_xg: float, away_xg: float) -> tuple:
    """Calcola probabilità 1X2 da xG usando Poisson"""
    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for h in range(10):
        for a in range(10):
            prob = poisson_probability(h, home_xg) * poisson_probability(a, away_xg)
            if h > a:
                home_win += prob
            elif h == a:
                draw += prob
            else:
                away_win += prob

    return home_win, draw, away_win


def test_coherence(spread_open: float, spread_close: float,
                   total_open: float, total_close: float):
    """
    Testa coerenza completa di un match
    """
    print("\n" + "="*80)
    print(f"TEST COERENZA: spread {spread_open}→{spread_close}, total {total_open}→{total_close}")
    print("="*80)

    # ===== STEP 1: OPENING VALUES =====
    print("\n[OPENING VALUES]")
    print(f"  Spread open: {spread_open}")
    print(f"  Total open:  {total_open}")

    # Calcola xG da spread/total apertura
    xg_open = calculate_expected_goals(spread_open, total_open, use_advanced_formulas=False)
    print(f"\n  xG derivato:")
    print(f"    Home xG: {xg_open.home_xg:.2f}")
    print(f"    Away xG: {xg_open.away_xg:.2f}")
    print(f"    Total xG: {xg_open.home_xg + xg_open.away_xg:.2f}")

    # REVERSE ENGINEER: xG → spread/total
    implied_spread_open = reverse_engineer_spread_from_xg(xg_open.home_xg, xg_open.away_xg)
    implied_total_open = reverse_engineer_total_from_xg(xg_open.home_xg, xg_open.away_xg)

    print(f"\n  Reverse engineered:")
    print(f"    Implied spread: {implied_spread_open:.2f}")
    print(f"    Implied total:  {implied_total_open:.2f}")

    # COERENZA CHECK
    spread_diff_open = abs(spread_open - implied_spread_open)
    total_diff_open = abs(total_open - implied_total_open)

    print(f"\n  📊 COERENZA OPENING:")
    print(f"    Spread diff: {spread_diff_open:.3f} {'✅' if spread_diff_open < 0.2 else '⚠️' if spread_diff_open < 0.5 else '❌'}")
    print(f"    Total diff:  {total_diff_open:.3f} {'✅' if total_diff_open < 0.2 else '⚠️' if total_diff_open < 0.5 else '❌'}")

    # ===== STEP 2: CLOSING VALUES =====
    print("\n[CLOSING VALUES]")
    print(f"  Spread close: {spread_close}")
    print(f"  Total close:  {total_close}")

    xg_close = calculate_expected_goals(spread_close, total_close, use_advanced_formulas=False)
    print(f"\n  xG derivato:")
    print(f"    Home xG: {xg_close.home_xg:.2f}")
    print(f"    Away xG: {xg_close.away_xg:.2f}")
    print(f"    Total xG: {xg_close.home_xg + xg_close.away_xg:.2f}")

    implied_spread_close = reverse_engineer_spread_from_xg(xg_close.home_xg, xg_close.away_xg)
    implied_total_close = reverse_engineer_total_from_xg(xg_close.home_xg, xg_close.away_xg)

    print(f"\n  Reverse engineered:")
    print(f"    Implied spread: {implied_spread_close:.2f}")
    print(f"    Implied total:  {implied_total_close:.2f}")

    spread_diff_close = abs(spread_close - implied_spread_close)
    total_diff_close = abs(total_close - implied_total_close)

    print(f"\n  📊 COERENZA CLOSING:")
    print(f"    Spread diff: {spread_diff_close:.3f} {'✅' if spread_diff_close < 0.2 else '⚠️' if spread_diff_close < 0.5 else '❌'}")
    print(f"    Total diff:  {total_diff_close:.3f} {'✅' if total_diff_close < 0.2 else '⚠️' if total_diff_close < 0.5 else '❌'}")

    # ===== STEP 3: 1X2 PROBABILITIES COHERENCE =====
    print("\n[1X2 PROBABILITIES COHERENCE]")

    # Da spread
    prob_1x2_from_spread = spread_to_implied_probability(spread_close)
    print(f"  Da spread {spread_close}:")
    print(f"    Home: {prob_1x2_from_spread['home']:.1%}")
    print(f"    Draw: {prob_1x2_from_spread['draw']:.1%}")
    print(f"    Away: {prob_1x2_from_spread['away']:.1%}")

    # Da xG
    prob_1x2_from_xg = calculate_1x2_prob_from_xg(xg_close.home_xg, xg_close.away_xg)
    print(f"\n  Da xG ({xg_close.home_xg:.2f} vs {xg_close.away_xg:.2f}):")
    print(f"    Home: {prob_1x2_from_xg[0]:.1%}")
    print(f"    Draw: {prob_1x2_from_xg[1]:.1%}")
    print(f"    Away: {prob_1x2_from_xg[2]:.1%}")

    # Differenze
    home_diff = abs(prob_1x2_from_spread['home'] - prob_1x2_from_xg[0])
    draw_diff = abs(prob_1x2_from_spread['draw'] - prob_1x2_from_xg[1])
    away_diff = abs(prob_1x2_from_spread['away'] - prob_1x2_from_xg[2])

    print(f"\n  📊 DIFFERENZE 1X2:")
    print(f"    Home: {home_diff:.1%} {'✅' if home_diff < 0.05 else '⚠️' if home_diff < 0.10 else '❌'}")
    print(f"    Draw: {draw_diff:.1%} {'✅' if draw_diff < 0.05 else '⚠️' if draw_diff < 0.10 else '❌'}")
    print(f"    Away: {away_diff:.1%} {'✅' if away_diff < 0.05 else '⚠️' if away_diff < 0.10 else '❌'}")

    # ===== STEP 4: MOVIMENTO COERENTE? =====
    print("\n[MOVIMENTO COERENTE?]")

    spread_movement = spread_close - spread_open
    total_movement = total_close - total_open

    print(f"  Spread movimento: {spread_movement:+.2f}")
    print(f"  Total movimento:  {total_movement:+.2f}")

    # Logic check: se spread aumenta (favorito più forte), total dovrebbe salire o scendere?
    # Dipende: se favorito più forte ma total scende → partita più chiusa (difensiva)
    #          se favorito più forte e total sale → favorito vincerà largo

    xg_diff_open = xg_open.home_xg - xg_open.away_xg
    xg_diff_close = xg_close.home_xg - xg_close.away_xg
    xg_total_open = xg_open.home_xg + xg_open.away_xg
    xg_total_close = xg_close.home_xg + xg_close.away_xg

    print(f"\n  xG diff open→close: {xg_diff_open:.2f} → {xg_diff_close:.2f} ({xg_diff_close - xg_diff_open:+.2f})")
    print(f"  xG total open→close: {xg_total_open:.2f} → {xg_total_close:.2f} ({xg_total_close - xg_total_open:+.2f})")

    # ===== SUMMARY =====
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)

    all_good = (spread_diff_open < 0.3 and spread_diff_close < 0.3 and
                total_diff_open < 0.3 and total_diff_close < 0.3 and
                home_diff < 0.10 and draw_diff < 0.10 and away_diff < 0.10)

    if all_good:
        print("✅ COERENZA OK: Spread e total rispettano le formule matematiche")
    else:
        print("⚠️ INCOERENZE RILEVATE:")
        if spread_diff_open >= 0.3 or spread_diff_close >= 0.3:
            print("  - Spread non coerente con xG derivato")
        if total_diff_open >= 0.3 or total_diff_close >= 0.3:
            print("  - Total non coerente con xG derivato")
        if home_diff >= 0.10 or draw_diff >= 0.10 or away_diff >= 0.10:
            print("  - Probabilità 1X2 da spread vs xG discordanti")


def run_comprehensive_audit():
    """Esegue audit su vari scenari"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "AUDIT COERENZA SPREAD/TOTAL/xG" + " "*28 + "║")
    print("╚" + "="*78 + "╝")

    # Test 1: Match equilibrato
    test_coherence(
        spread_open=-0.5, spread_close=-0.5,
        total_open=2.5, total_close=2.5
    )

    # Test 2: Favorito forte
    test_coherence(
        spread_open=-1.5, spread_close=-2.0,
        total_open=2.5, total_close=2.75
    )

    # Test 3: Total alto
    test_coherence(
        spread_open=-1.0, spread_close=-1.0,
        total_open=3.0, total_close=3.5
    )

    # Test 4: Movimenti contrastanti (spread sale, total scende)
    test_coherence(
        spread_open=-1.0, spread_close=-1.5,
        total_open=3.0, total_close=2.5
    )

    print("\n" + "="*80)
    print("AUDIT COMPLETATO")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_comprehensive_audit()
