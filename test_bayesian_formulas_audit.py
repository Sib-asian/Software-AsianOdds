#!/usr/bin/env python3
"""
AUDIT BAYESIAN & MARKET-ADJUSTED FORMULAS
Verifica correttezza matematica delle formule bayesiane e market-adjusted
"""

import math
from market_movement_analyzer import (
    MarketMovementAnalyzer,
    calculate_expected_goals_advanced,
    MarketIntelligence,
    MovementAnalysis,
    MovementDirection,
    MovementIntensity
)


def test_bayesian_btts_formula():
    """Verifica la formula Bayesian BTTS"""
    print("\n" + "="*70)
    print("TEST 1: BAYESIAN BTTS FORMULA")
    print("="*70)

    analyzer = MarketMovementAnalyzer()

    # Test con movimento significativo
    print("\n[Test 1] Total sale 2.5→3.0 (sharp money su Over)")
    print("Aspettativa: BTTS Bayesian dovrebbe aumentare vs base")

    result_no_movement = analyzer.analyze(
        spread_open=-1.0, spread_close=-1.0,
        total_open=2.5, total_close=2.5
    )

    result_total_up = analyzer.analyze(
        spread_open=-1.0, spread_close=-1.0,
        total_open=2.5, total_close=3.0  # +0.5 movimento
    )

    btts_base = result_no_movement.expected_goals.btts_prob
    btts_bayesian_no_mov = result_no_movement.expected_goals.bayesian_btts

    btts_base_mov = result_total_up.expected_goals.btts_prob
    btts_bayesian_mov = result_total_up.expected_goals.bayesian_btts

    print(f"\n  Nessun movimento:")
    print(f"    Base BTTS: {btts_base:.1%}")
    if btts_bayesian_no_mov:
        print(f"    Bayesian BTTS: {btts_bayesian_no_mov['btts_prob']:.1%}")

    print(f"\n  Con movimento total +0.5:")
    print(f"    Base BTTS: {btts_base_mov:.1%}")
    if btts_bayesian_mov:
        bay_prob = btts_bayesian_mov['btts_prob']
        print(f"    Bayesian BTTS: {bay_prob:.1%}")

        # Verifica che Bayesian sia aumentato
        if btts_bayesian_no_mov:
            diff = bay_prob - btts_bayesian_no_mov['btts_prob']
            print(f"    Delta: {diff:+.1%} ", end="")
            if diff > 0:
                print("✅ (Bayesian aumenta con total up)")
            else:
                print("❌ (dovrebbe aumentare)")

    # Test 2: Spread movimento
    print("\n[Test 2] Spread scende -1.0→-1.5 (favorito si rafforza)")
    print("Aspettativa: BTTS Bayesian dovrebbe diminuire (favorito domina)")

    result_spread_down = analyzer.analyze(
        spread_open=-1.0, spread_close=-1.5,  # -0.5 movimento
        total_open=2.5, total_close=2.5
    )

    btts_spread = result_spread_down.expected_goals.bayesian_btts
    if btts_spread and btts_bayesian_no_mov:
        diff = btts_spread['btts_prob'] - btts_bayesian_no_mov['btts_prob']
        print(f"    Bayesian BTTS: {btts_spread['btts_prob']:.1%}")
        print(f"    Delta vs base: {diff:+.1%} ", end="")
        if diff < 0:
            print("✅ (BTTS diminuisce correttamente)")
        else:
            print("⚠️ (dovrebbe diminuire)")


def test_market_adjusted_1x2():
    """Verifica Market-Adjusted 1X2 Bayesian"""
    print("\n" + "="*70)
    print("TEST 2: MARKET-ADJUSTED 1X2 BAYESIAN")
    print("="*70)

    analyzer = MarketMovementAnalyzer()

    print("\n[Test 1] Spread scende -1.0→-2.0 (sharp money su casa)")
    print("Aspettativa: Market-Adjusted Home Win dovrebbe essere > Base")

    result = analyzer.analyze(
        spread_open=-1.0, spread_close=-2.0,
        total_open=2.5, total_close=2.5
    )

    xg = result.expected_goals
    base_home = xg.home_win_prob
    base_draw = xg.draw_prob
    base_away = xg.away_win_prob

    print(f"\n  Base 1X2:")
    print(f"    Home: {base_home:.1%}")
    print(f"    Draw: {base_draw:.1%}")
    print(f"    Away: {base_away:.1%}")

    if xg.market_adjusted_1x2:
        adj_home = xg.market_adjusted_1x2['home_win']
        adj_draw = xg.market_adjusted_1x2['draw']
        adj_away = xg.market_adjusted_1x2['away_win']

        print(f"\n  Market-Adjusted 1X2:")
        print(f"    Home: {adj_home:.1%} ({adj_home - base_home:+.1%})")
        print(f"    Draw: {adj_draw:.1%} ({adj_draw - base_draw:+.1%})")
        print(f"    Away: {adj_away:.1%} ({adj_away - base_away:+.1%})")

        # Verifica normalizzazione
        total = adj_home + adj_draw + adj_away
        print(f"\n  Normalizzazione: {total:.6f} ", end="")
        if abs(total - 1.0) < 0.01:
            print("✅")
        else:
            print(f"❌ (dovrebbe essere 1.0)")

        # Verifica che home sia aumentato
        print(f"\n  Home win boost: {adj_home - base_home:+.1%} ", end="")
        if adj_home > base_home:
            print("✅ (sharp money su casa aumenta home win)")
        else:
            print("⚠️ (dovrebbe aumentare)")
    else:
        print("\n  ❌ Market-Adjusted 1X2 non calcolato!")


def test_bayesian_update_math():
    """Verifica matematica dell'update Bayesiano"""
    print("\n" + "="*70)
    print("TEST 3: MATEMATICA BAYESIAN UPDATE")
    print("="*70)

    print("\nVerifica Teorema di Bayes: P(H|E) = P(E|H) * P(H) / P(E)")
    print("Dove: H = ipotesi (es. Home vince), E = evidence (movimento)")

    analyzer = MarketMovementAnalyzer()

    # Test con movimento forte
    result = analyzer.analyze(
        spread_open=-1.0, spread_close=-2.5,  # Movimento MOLTO forte
        total_open=2.5, total_close=2.5
    )

    xg = result.expected_goals

    if xg.market_adjusted_1x2:
        prior_home = xg.home_win_prob  # P(H)
        posterior_home = xg.market_adjusted_1x2['home_win']  # P(H|E)

        # Likelihood ratio (quanto forte è l'evidenza)
        # Per sharp money su home, likelihood dovrebbe essere > 1
        if prior_home > 0:
            likelihood_ratio = posterior_home / prior_home

            print(f"\n  Prior P(Home): {prior_home:.1%}")
            print(f"  Posterior P(Home|Movement): {posterior_home:.1%}")
            print(f"  Likelihood Ratio: {likelihood_ratio:.2f}")

            # Verifica che sia ragionevole (1.0 - 2.0 range tipico)
            print(f"\n  Likelihood ratio check: ", end="")
            if 1.0 <= likelihood_ratio <= 3.0:
                print(f"✅ ({likelihood_ratio:.2f} è ragionevole)")
            elif likelihood_ratio < 1.0:
                print(f"❌ ({likelihood_ratio:.2f} < 1.0, dovrebbe aumentare!)")
            else:
                print(f"⚠️ ({likelihood_ratio:.2f} > 3.0, molto alto)")


def test_home_advantage_adjustment():
    """Verifica Home Advantage Adjustment"""
    print("\n" + "="*70)
    print("TEST 4: HOME ADVANTAGE ADJUSTMENT")
    print("="*70)

    print("\nVerifica che home advantage sia applicato correttamente")
    print("Tipicamente: +10-15% home, -10-15% away")

    from market_movement_analyzer import adjust_for_home_advantage

    test_cases = [
        (1.5, 1.5, "Equilibrio perfetto"),
        (2.0, 1.0, "Home già favorito"),
        (1.0, 2.0, "Away favorito"),
    ]

    all_ok = True

    for home_base, away_base, desc in test_cases:
        home_adj, away_adj = adjust_for_home_advantage(home_base, away_base)

        home_boost = (home_adj / home_base - 1) * 100
        away_penalty = (away_adj / away_base - 1) * 100

        print(f"\n  [{desc}]")
        print(f"    Base: Home {home_base:.2f}, Away {away_base:.2f}")
        print(f"    Adjusted: Home {home_adj:.2f}, Away {away_adj:.2f}")
        print(f"    Home boost: {home_boost:+.1f}%")
        print(f"    Away penalty: {away_penalty:+.1f}%")

        # Verifica che home aumenti e away diminuisca
        if home_adj > home_base and away_adj < away_base:
            print(f"    ✅ Home advantage corretto")
        else:
            print(f"    ❌ Home advantage non applicato correttamente")
            all_ok = False

        # Verifica range ragionevole (5-25%)
        if 5 <= home_boost <= 25 and -25 <= away_penalty <= -5:
            print(f"    ✅ Boost/penalty in range ragionevole")
        else:
            print(f"    ⚠️ Boost/penalty fuori range tipico (5-25%)")

    return all_ok


def test_dixon_coles_tau():
    """Verifica parametro tau Dixon-Coles"""
    print("\n" + "="*70)
    print("TEST 5: DIXON-COLES TAU PARAMETER")
    print("="*70)

    print("\nVerifica che tau sia nel range corretto (-0.5 a 0.5)")
    print("Tau negativo = risultati bassi meno probabili di Poisson")

    from market_movement_analyzer import dixon_coles_probability

    # Test con diversi xG
    test_xgs = [
        (1.5, 1.0, "Equilibrato"),
        (2.5, 0.5, "Favorito forte"),
        (3.0, 3.0, "Alto scoring"),
    ]

    for home_xg, away_xg, desc in test_xgs:
        print(f"\n  [{desc}] Home xG: {home_xg}, Away xG: {away_xg}")

        # Per 0-0, Dixon-Coles dovrebbe ridurre la probabilità
        p_00_poisson = math.exp(-home_xg) * math.exp(-away_xg)
        p_00_dixon = dixon_coles_probability(0, 0, home_xg, away_xg)

        reduction = (1 - p_00_dixon / p_00_poisson) * 100 if p_00_poisson > 0 else 0

        print(f"    P(0-0) Poisson: {p_00_poisson:.4f}")
        print(f"    P(0-0) Dixon-Coles: {p_00_dixon:.4f}")
        print(f"    Riduzione: {reduction:.1f}% ", end="")

        # Dixon-Coles tipicamente riduce 0-0 del 30-70%
        if 20 <= reduction <= 80:
            print("✅")
        else:
            print(f"⚠️ (riduzione tipica 30-70%)")


def test_numerical_stability():
    """Verifica stabilità numerica"""
    print("\n" + "="*70)
    print("TEST 6: STABILITÀ NUMERICA")
    print("="*70)

    print("\nVerifica overflow/underflow con valori estremi")

    analyzer = MarketMovementAnalyzer()

    extreme_cases = [
        {"spread": -10.0, "total": 10.0, "desc": "Spread estremo"},
        {"spread": -0.01, "total": 0.5, "desc": "Total bassissimo"},
        {"spread": -1.0, "total": 10.0, "desc": "Total altissimo"},
    ]

    all_ok = True

    for case in extreme_cases:
        print(f"\n  [{case['desc']}] spread={case['spread']}, total={case['total']}")

        try:
            result = analyzer.analyze(
                spread_open=case['spread'],
                spread_close=case['spread'],
                total_open=case['total'],
                total_close=case['total']
            )

            xg = result.expected_goals

            # Verifica valori non NaN/Inf
            checks = [
                ("home_xg", xg.home_xg),
                ("away_xg", xg.away_xg),
                ("home_win_prob", xg.home_win_prob),
                ("draw_prob", xg.draw_prob),
                ("away_win_prob", xg.away_win_prob),
                ("btts_prob", xg.btts_prob),
            ]

            has_issues = False
            for name, value in checks:
                if math.isnan(value) or math.isinf(value):
                    print(f"    ❌ {name} = {value} (NaN/Inf)")
                    has_issues = True
                    all_ok = False

            if not has_issues:
                print(f"    ✅ Tutti i valori validi")
                print(f"       Home xG: {xg.home_xg:.2f}, Away xG: {xg.away_xg:.2f}")
                print(f"       1X2: {xg.home_win_prob:.1%} / {xg.draw_prob:.1%} / {xg.away_win_prob:.1%}")

        except Exception as e:
            print(f"    ❌ EXCEPTION: {e}")
            all_ok = False

    return all_ok


def test_confidence_score():
    """Verifica Market Confidence Score"""
    print("\n" + "="*70)
    print("TEST 7: MARKET CONFIDENCE SCORE")
    print("="*70)

    print("\nVerifica che confidence score sia in range [0, 100]")

    analyzer = MarketMovementAnalyzer()

    test_scenarios = [
        {
            "spread_open": -1.0, "spread_close": -2.0,
            "total_open": 2.5, "total_close": 3.0,
            "desc": "Movimento concordante (sharp money)"
        },
        {
            "spread_open": -1.0, "spread_close": -0.5,
            "total_open": 2.5, "total_close": 3.5,
            "desc": "Movimento discordante (confusione)"
        },
        {
            "spread_open": -1.0, "spread_close": -1.0,
            "total_open": 2.5, "total_close": 2.5,
            "desc": "Nessun movimento (stabile)"
        },
    ]

    all_ok = True

    for scenario in test_scenarios:
        print(f"\n  [{scenario['desc']}]")

        result = analyzer.analyze(
            spread_open=scenario['spread_open'],
            spread_close=scenario['spread_close'],
            total_open=scenario['total_open'],
            total_close=scenario['total_close']
        )

        if hasattr(result.expected_goals, 'market_confidence_score'):
            score = result.expected_goals.market_confidence_score
            print(f"    Confidence Score: {score}")

            if 0 <= score <= 100:
                print(f"    ✅ Score in range [0, 100]")
            else:
                print(f"    ❌ Score fuori range: {score}")
                all_ok = False
        else:
            print(f"    ⚠️ market_confidence_score non presente")

    return all_ok


def run_bayesian_audit():
    """Esegue tutti gli audit bayesiani"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*12 + "AUDIT FORMULE BAYESIANE & MARKET-ADJUSTED" + " "*14 + "║")
    print("╚" + "="*68 + "╝")

    test_bayesian_btts_formula()
    test_market_adjusted_1x2()
    test_bayesian_update_math()
    test_home_advantage_adjustment()
    test_dixon_coles_tau()
    test_numerical_stability()
    test_confidence_score()

    print("\n" + "="*70)
    print("AUDIT BAYESIAN COMPLETATO!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_bayesian_audit()
