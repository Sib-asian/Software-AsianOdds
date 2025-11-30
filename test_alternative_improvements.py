#!/usr/bin/env python3
"""
Test per OPZIONE B: Alternative Markets Refinement

Testa le 4 nuove funzioni avanzate:
1. Dynamic HT Percentage (non più fisso 45%)
2. HT/FT with Correlation (momentum e conditional probabilities)
3. Sticky Scores Adjustment (boost 1-1, 2-1; penalità 5+ gol)
4. Time-Weighted xG HT (distribuzione non uniforme nei 90 min)
"""

from market_movement_analyzer import (
    calculate_dynamic_ht_percentage,
    calculate_ht_ft_with_correlation,
    apply_sticky_scores_adjustment,
    calculate_time_weighted_xg_ht,
    calculate_total_goals_distribution,
    MarketMovementAnalyzer
)


def test_dynamic_ht_percentage():
    """Test Dynamic HT Percentage"""
    print("\n" + "="*60)
    print("TEST 1: DYNAMIC HT PERCENTAGE")
    print("="*60)

    # Test 1a: Match equilibrato, total medio
    print("\n[Test 1a] Match equilibrato (-0.5), total medio (2.5)")
    ht_pct = calculate_dynamic_ht_percentage(spread=-0.5, total=2.5)
    print(f"  HT Percentage: {ht_pct:.1%}")
    print(f"  Expected: ~47% (equilibrato boost)")
    print(f"  ✓ PASS" if 0.45 <= ht_pct <= 0.48 else "  ✗ FAIL")

    # Test 1b: Spread alto (favorito forte), total basso
    print("\n[Test 1b] Favorito forte (-2.0), total basso (2.0)")
    ht_pct = calculate_dynamic_ht_percentage(spread=-2.0, total=2.0)
    print(f"  HT Percentage: {ht_pct:.1%}")
    print(f"  Expected: ~32% (spread alto -8%, total basso -5%)")
    print(f"  ✓ PASS" if 0.30 <= ht_pct <= 0.35 else "  ✗ FAIL")

    # Test 1c: Spread basso, total alto (partita aperta)
    print("\n[Test 1c] Match equilibrato (-0.5), total alto (3.5)")
    ht_pct = calculate_dynamic_ht_percentage(spread=-0.5, total=3.5)
    print(f"  HT Percentage: {ht_pct:.1%}")
    print(f"  Expected: ~55% (total alto +8%, equilibrato +2%)")
    print(f"  ✓ PASS" if 0.53 <= ht_pct <= 0.55 else "  ✗ FAIL")

    # Test 1d: Spread molto alto
    print("\n[Test 1d] Favorito dominante (-2.5), total basso (2.0)")
    ht_pct = calculate_dynamic_ht_percentage(spread=-2.5, total=2.0)
    print(f"  HT Percentage: {ht_pct:.1%}")
    print(f"  Expected: ~30% (spread -10%, total -5% = floor 35%)")
    print(f"  ✓ PASS" if 0.30 <= ht_pct <= 0.36 else "  ✗ FAIL")


def test_ht_ft_with_correlation():
    """Test HT/FT with Correlation"""
    print("\n" + "="*60)
    print("TEST 2: HT/FT WITH CORRELATION")
    print("="*60)

    # Test 2a: Favorito forte (momentum)
    print("\n[Test 2a] Favorito forte casa (home_xg=2.0, away_xg=0.8)")
    ht_ft_probs = calculate_ht_ft_with_correlation(home_xg=2.0, away_xg=0.8, spread=-1.5)

    print(f"\n  Top 3 HT/FT probabilities:")
    sorted_probs = sorted(ht_ft_probs.items(), key=lambda x: x[1], reverse=True)
    for combo, prob in sorted_probs[:3]:
        print(f"    {combo}: {prob:.1%}")

    # Con momentum, 1/1 dovrebbe essere molto più alta del vecchio metodo
    print(f"\n  P(1/1) = {ht_ft_probs['1/1']:.1%}")
    print(f"  Expected: >20% (momentum boost su favorito)")
    print(f"  ✓ PASS" if ht_ft_probs['1/1'] >= 0.20 else "  ✗ FAIL")

    # Test 2b: Match equilibrato (X/X boost)
    print("\n[Test 2b] Match equilibrato (home_xg=1.2, away_xg=1.2)")
    ht_ft_probs = calculate_ht_ft_with_correlation(home_xg=1.2, away_xg=1.2, spread=0.0)

    print(f"\n  P(X/X) = {ht_ft_probs['X/X']:.1%}")
    print(f"  Expected: Alta (1.30 boost per pareggio HT→FT)")
    print(f"  ✓ PASS" if ht_ft_probs['X/X'] >= 0.10 else "  ✗ FAIL")

    # Test 2c: Rimonta (underdog desperation)
    print("\n[Test 2c] Rimonta possibile (2/1 - trasferta vince HT, casa vince FT)")
    ht_ft_probs = calculate_ht_ft_with_correlation(home_xg=1.8, away_xg=1.0, spread=-1.0)

    print(f"\n  P(2/1) = {ht_ft_probs['2/1']:.1%}")
    print(f"  Expected: >2% (underdog desperation + home advantage)")
    print(f"  ✓ PASS" if ht_ft_probs['2/1'] >= 0.02 else "  ✗ FAIL")

    # Verifica somma = 100%
    total_prob = sum(ht_ft_probs.values())
    print(f"\n  Somma probabilità: {total_prob:.4f}")
    print(f"  ✓ NORMALIZZAZIONE OK" if abs(total_prob - 1.0) < 0.001 else "  ✗ FAIL")


def test_sticky_scores_adjustment():
    """Test Sticky Scores Adjustment"""
    print("\n" + "="*60)
    print("TEST 3: STICKY SCORES ADJUSTMENT")
    print("="*60)

    # Test 3a: Match equilibrato (1-1 boost)
    print("\n[Test 3a] Match equilibrato (home_xg=1.3, away_xg=1.3)")
    goals_dist = calculate_total_goals_distribution(home_xg=1.3, away_xg=1.3)
    adjusted_dist = apply_sticky_scores_adjustment(goals_dist, home_xg=1.3, away_xg=1.3)

    # Calcola P(2 gol totali) prima e dopo
    prob_2goals_before = goals_dist.get(2, 0)
    prob_2goals_after = adjusted_dist.get(2, 0)

    print(f"  P(2 gol totali) prima: {prob_2goals_before:.1%}")
    print(f"  P(2 gol totali) dopo sticky: {prob_2goals_after:.1%}")
    print(f"  Boost: {(prob_2goals_after/prob_2goals_before - 1)*100:.1f}%")
    print(f"  Expected: +10-20% (1-1 è sticky score)")
    print(f"  ✓ PASS" if prob_2goals_after > prob_2goals_before * 1.05 else "  ✗ FAIL")

    # Test 3b: High scoring (penalità 5+ gol)
    print("\n[Test 3b] High scoring (home_xg=2.5, away_xg=2.5)")
    goals_dist = calculate_total_goals_distribution(home_xg=2.5, away_xg=2.5)
    adjusted_dist = apply_sticky_scores_adjustment(goals_dist, home_xg=2.5, away_xg=2.5)

    # Calcola P(5+ gol) prima e dopo
    prob_5plus_before = sum(goals_dist.get(i, 0) for i in range(5, 10))
    prob_5plus_after = sum(adjusted_dist.get(i, 0) for i in range(5, 10))

    print(f"  P(5+ gol) prima: {prob_5plus_before:.1%}")
    print(f"  P(5+ gol) dopo sticky: {prob_5plus_after:.1%}")
    print(f"  Penalità: {(1 - prob_5plus_after/prob_5plus_before)*100:.1f}%")
    print(f"  Expected: -10-30% (5+ gol penalizzati)")
    print(f"  ✓ PASS" if prob_5plus_after < prob_5plus_before * 0.95 else "  ✗ FAIL")

    # Verifica somma = 100%
    total_prob = sum(adjusted_dist.values())
    print(f"\n  Somma probabilità: {total_prob:.4f}")
    print(f"  ✓ NORMALIZZAZIONE OK" if abs(total_prob - 1.0) < 0.001 else "  ✗ FAIL")


def test_time_weighted_xg_ht():
    """Test Time-Weighted xG HT"""
    print("\n" + "="*60)
    print("TEST 4: TIME-WEIGHTED xG HT")
    print("="*60)

    # Test 4a: Match equilibrato, total medio
    print("\n[Test 4a] Match equilibrato (home_xg=1.5, away_xg=1.5, spread=-0.5, total=2.5)")
    result = calculate_time_weighted_xg_ht(
        home_xg=1.5, away_xg=1.5,
        total=2.5, spread=-0.5
    )

    print(f"\n  HT Percentage: {result['ht_percentage']:.1%}")
    print(f"  Home xG HT: {result['home_xg_ht']:.2f}")
    print(f"  Away xG HT: {result['away_xg_ht']:.2f}")
    print(f"  Total xG HT: {result['total_xg_ht']:.2f}")

    print(f"\n  Time weights:")
    for period, weight in result['time_weights'].items():
        print(f"    {period}: {weight:.1%}")

    # Verifica time weights sommano a 100%
    total_weight = sum(result['time_weights'].values())
    print(f"\n  Somma time weights: {total_weight:.4f}")
    print(f"  ✓ PASS" if abs(total_weight - 1.0) < 0.001 else "  ✗ FAIL")

    # Test 4b: Favorito forte (parte aggressivo)
    print("\n[Test 4b] Favorito forte (spread=-2.0)")
    result = calculate_time_weighted_xg_ht(
        home_xg=2.0, away_xg=0.8,
        total=2.5, spread=-2.0
    )

    print(f"\n  Time weights (favorito inizia forte):")
    for period, weight in result['time_weights'].items():
        print(f"    {period}: {weight:.1%}")

    print(f"\n  Expected: 0-15 min boost (favorito aggressivo)")
    print(f"  ✓ PASS" if result['time_weights']['0-15'] >= 0.18 else "  ✗ FAIL")

    # Test 4c: Probabilità HT
    print("\n[Test 4c] Probabilità HT calcolate correttamente")
    print(f"  P(Home Win HT): {result['home_win_ht']:.1%}")
    print(f"  P(Draw HT): {result['draw_ht']:.1%}")
    print(f"  P(Away Win HT): {result['away_win_ht']:.1%}")
    print(f"  P(BTTS HT): {result['btts_ht']:.1%}")
    print(f"  P(Over 0.5 HT): {result['over_05_ht']:.1%}")
    print(f"  P(Over 1.5 HT): {result['over_15_ht']:.1%}")

    # Verifica somma 1X2 HT = 100% (tolleranza 0.5% per troncamento Poisson)
    total_1x2 = result['home_win_ht'] + result['draw_ht'] + result['away_win_ht']
    print(f"\n  Somma 1X2 HT: {total_1x2:.4f}")
    print(f"  ✓ NORMALIZZAZIONE OK" if abs(total_1x2 - 1.0) < 0.005 else "  ✗ FAIL")


def test_full_integration():
    """Test integrazione completa nell'analyzer"""
    print("\n" + "="*60)
    print("TEST 5: INTEGRAZIONE COMPLETA")
    print("="*60)

    analyzer = MarketMovementAnalyzer()

    # Test con favorito forte, total medio
    print("\n[Test 5] Analisi completa con OPZIONE B improvements")
    result = analyzer.analyze(
        spread_open=-1.5,
        spread_close=-1.75,
        total_open=2.5,
        total_close=2.6
    )

    # Verifica alternative recommendations esistono
    print(f"\n  Alternative Recommendations: {len(result.alternative_recommendations)}")

    for rec in result.alternative_recommendations:
        print(f"\n  [{rec.market_name}] {rec.recommendation}")
        print(f"    Confidence: {rec.confidence.value}")
        print(f"    Explanation: {rec.explanation}")

    # Verifica che almeno HT/FT, Multigol esistano
    market_names = {rec.market_name for rec in result.alternative_recommendations}

    print(f"\n  Mercati presenti: {market_names}")
    print(f"  ✓ HT/FT presente" if "HT/FT" in market_names else "  ⚠ HT/FT mancante")
    print(f"  ✓ Multigol presente" if "Multigol" in market_names else "  ⚠ Multigol mancante")

    # Verifica che le spiegazioni menzionino le nuove features
    ht_ft_rec = next((r for r in result.alternative_recommendations if r.market_name == "HT/FT"), None)
    if ht_ft_rec:
        has_correlation = "correlazione" in ht_ft_rec.explanation.lower() or "HT%" in ht_ft_rec.explanation
        print(f"\n  HT/FT usa correlation: {has_correlation}")
        print(f"  ✓ PASS" if has_correlation else "  ✗ FAIL")

    multigol_rec = next((r for r in result.alternative_recommendations if r.market_name == "Multigol"), None)
    if multigol_rec:
        has_sticky = "sticky" in multigol_rec.explanation.lower()
        print(f"  Multigol usa sticky scores: {has_sticky}")
        print(f"  ✓ PASS" if has_sticky else "  ✗ FAIL")

    print(f"\n  ✓ TEST INTEGRAZIONE COMPLETO")


def run_all_tests():
    """Esegue tutti i test"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*8 + "OPZIONE B: ALTERNATIVE MARKETS REFINEMENT" + " "*9 + "║")
    print("╚" + "="*58 + "╝")

    test_dynamic_ht_percentage()
    test_ht_ft_with_correlation()
    test_sticky_scores_adjustment()
    test_time_weighted_xg_ht()
    test_full_integration()

    print("\n" + "="*60)
    print("TUTTI I TEST COMPLETATI!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
