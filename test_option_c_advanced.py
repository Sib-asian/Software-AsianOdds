#!/usr/bin/env python3
"""
Test completi per OPZIONE C - Advanced Probability Calculations

Testa:
1. Spread to Implied Probability
2. Dynamic Home Advantage
3. Bayesian Probability Update
4. Monte Carlo BTTS Simulation
5. Smart BTTS Calculator
6. Market-Adjusted Probabilities
7. Market Confidence Score
8. calculate_expected_goals_advanced (integrazione completa)
"""

from market_movement_analyzer import (
    spread_to_implied_probability,
    calculate_dynamic_home_advantage,
    bayesian_probability_update,
    monte_carlo_btts_simulation,
    calculate_smart_btts,
    calculate_market_adjusted_probabilities,
    calculate_market_confidence_score,
    calculate_expected_goals_advanced,
    MarketMovementAnalyzer,
    MovementDirection,
    MovementIntensity,
    MovementAnalysis,
    MarketIntelligence
)


def test_spread_to_implied_probability():
    """Test conversione spread → probabilità"""
    print("\n" + "="*70)
    print("TEST 1: SPREAD TO IMPLIED PROBABILITY")
    print("="*70)

    # Test 1: Casa favorita (-1.5)
    print("\n[Test 1a] Spread -1.5 (casa favorita)")
    result = spread_to_implied_probability(-1.5)
    print(f"  Home: {result['home']:.1%}")
    print(f"  Draw: {result['draw']:.1%}")
    print(f"  Away: {result['away']:.1%}")
    print(f"  Sum: {sum(result.values()):.3f}")
    assert 0.99 < sum(result.values()) < 1.01, "Le probabilità devono sommare a 1"
    assert result['home'] > 0.60, "Casa deve avere >60% con spread -1.5"
    print(f"  ✓ PASS")

    # Test 2: Trasferta favorita (+1.0)
    print("\n[Test 1b] Spread +1.0 (trasferta favorita)")
    result = spread_to_implied_probability(1.0)
    print(f"  Home: {result['home']:.1%}")
    print(f"  Draw: {result['draw']:.1%}")
    print(f"  Away: {result['away']:.1%}")
    assert result['away'] > result['home'], "Trasferta deve avere prob maggiore"
    print(f"  ✓ PASS")

    # Test 3: Equilibrio (spread 0)
    print("\n[Test 1c] Spread 0.0 (equilibrio)")
    result = spread_to_implied_probability(0.0)
    print(f"  Home: {result['home']:.1%}")
    print(f"  Draw: {result['draw']:.1%}")
    print(f"  Away: {result['away']:.1%}")
    assert abs(result['home'] - result['away']) < 0.01, "Home e Away devono essere uguali"
    print(f"  ✓ PASS")


def test_dynamic_home_advantage():
    """Test home advantage dinamico"""
    print("\n" + "="*70)
    print("TEST 2: DYNAMIC HOME ADVANTAGE")
    print("="*70)

    # Test 1: Movimento verso casa
    print("\n[Test 2a] Spread si muove verso casa (-1.5 → -2.0)")
    ha = calculate_dynamic_home_advantage(-1.5, -2.0)
    print(f"  Dynamic HA: {ha:.3f} (base 1.15)")
    assert ha > 1.15, "HA deve aumentare se spread si muove verso casa"
    print(f"  ✓ PASS")

    # Test 2: Movimento verso trasferta
    print("\n[Test 2b] Spread si muove verso trasferta (-2.0 → -1.5)")
    ha = calculate_dynamic_home_advantage(-2.0, -1.5)
    print(f"  Dynamic HA: {ha:.3f} (base 1.15)")
    assert ha < 1.15, "HA deve diminuire se spread si muove verso trasferta"
    print(f"  ✓ PASS")

    # Test 3: Nessun movimento
    print("\n[Test 2c] Nessun movimento (-1.5 → -1.5)")
    ha = calculate_dynamic_home_advantage(-1.5, -1.5)
    print(f"  Dynamic HA: {ha:.3f} (base 1.15)")
    assert abs(ha - 1.15) < 0.01, "HA deve rimanere base se nessun movimento"
    print(f"  ✓ PASS")


def test_bayesian_probability_update():
    """Test aggiornamento bayesiano"""
    print("\n" + "="*70)
    print("TEST 3: BAYESIAN PROBABILITY UPDATE")
    print("="*70)

    prior = {'home': 0.50, 'draw': 0.30, 'away': 0.20}

    # Test 1: Segnale pro-casa
    print("\n[Test 3a] Segnale pro-casa (signal -0.5, confidence 0.6)")
    posterior = bayesian_probability_update(prior, market_signal=-0.5, signal_confidence=0.6)
    print(f"  Prior Home: {prior['home']:.1%}")
    print(f"  Posterior Home: {posterior['home']:.1%}")
    print(f"  Change: +{(posterior['home'] - prior['home'])*100:.1f}%")
    assert posterior['home'] > prior['home'], "Home prob deve aumentare con segnale pro-casa"
    assert 0.99 < sum(posterior.values()) < 1.01, "Le probabilità devono sommare a 1"
    print(f"  ✓ PASS")

    # Test 2: Segnale pro-trasferta
    print("\n[Test 3b] Segnale pro-trasferta (signal +0.5, confidence 0.6)")
    posterior = bayesian_probability_update(prior, market_signal=0.5, signal_confidence=0.6)
    print(f"  Prior Away: {prior['away']:.1%}")
    print(f"  Posterior Away: {posterior['away']:.1%}")
    print(f"  Change: +{(posterior['away'] - prior['away'])*100:.1f}%")
    assert posterior['away'] > prior['away'], "Away prob deve aumentare con segnale pro-trasferta"
    print(f"  ✓ PASS")


def test_monte_carlo_btts():
    """Test Monte Carlo BTTS simulation"""
    print("\n" + "="*70)
    print("TEST 4: MONTE CARLO BTTS SIMULATION")
    print("="*70)

    # Test 1: xG alte (dovrebbe dare alta prob BTTS)
    print("\n[Test 4a] xG alte (1.8 vs 1.5) - 5000 simulazioni")
    result = monte_carlo_btts_simulation(1.8, 1.5, n_simulations=5000)
    print(f"  BTTS Probability: {result['btts_prob']:.1%}")
    print(f"  Avg Home Goals: {result['avg_home_goals']:.2f}")
    print(f"  Avg Away Goals: {result['avg_away_goals']:.2f}")
    assert result['btts_prob'] > 0.50, "BTTS deve essere >50% con xG alte"
    assert 0.99 < result['btts_prob'] + result['nobtts_prob'] < 1.01, "Deve sommare a 1"
    print(f"  ✓ PASS")

    # Test 2: xG basse (dovrebbe dare bassa prob BTTS)
    print("\n[Test 4b] xG basse (0.8 vs 0.6) - 5000 simulazioni")
    result = monte_carlo_btts_simulation(0.8, 0.6, n_simulations=5000)
    print(f"  BTTS Probability: {result['btts_prob']:.1%}")
    assert result['btts_prob'] < 0.40, "BTTS deve essere <40% con xG basse"
    print(f"  ✓ PASS")


def test_smart_btts():
    """Test Smart BTTS con fattori di mercato"""
    print("\n" + "="*70)
    print("TEST 5: SMART BTTS CALCULATOR")
    print("="*70)

    # Test 1: Partita aperta (total alto, spread basso)
    print("\n[Test 5a] Partita aperta (total 3.0, spread -0.5)")
    result = calculate_smart_btts(
        home_xg=1.75,
        away_xg=1.25,
        total_open=2.75,
        total_close=3.0,
        spread_close=-0.5,
        use_monte_carlo=True
    )
    print(f"  Base BTTS: {result['base_btts']:.1%}")
    print(f"  Adjusted BTTS: {result['btts_prob']:.1%}")
    print(f"  Openness Score: {result['openness_score']:.2f}")
    print(f"  Balance Score: {result['balance_score']:.2f}")
    print(f"  Total Boost: {result['total_boost']:.2%}")
    assert result['btts_prob'] >= result['base_btts'], "BTTS adjusted deve essere >= base per partita aperta"
    print(f"  ✓ PASS")

    # Test 2: Partita chiusa (total basso, spread alto)
    print("\n[Test 5b] Partita chiusa (total 2.0, spread -2.5)")
    result = calculate_smart_btts(
        home_xg=1.5,
        away_xg=0.5,
        total_open=2.0,
        total_close=2.0,
        spread_close=-2.5,
        use_monte_carlo=False  # Usa formula per velocità
    )
    print(f"  Base BTTS: {result['base_btts']:.1%}")
    print(f"  Adjusted BTTS: {result['btts_prob']:.1%}")
    print(f"  Openness Score: {result['openness_score']:.2f}")
    assert result['btts_prob'] < 0.60, "BTTS deve essere bassa per partita chiusa"
    print(f"  ✓ PASS")


def test_market_adjusted_probabilities():
    """Test probabilità aggiustate con movimenti"""
    print("\n" + "="*70)
    print("TEST 6: MARKET-ADJUSTED PROBABILITIES")
    print("="*70)

    # Test 1: Con sharp money
    print("\n[Test 6a] Con sharp money detected")
    result = calculate_market_adjusted_probabilities(
        home_xg=1.5,
        away_xg=1.0,
        spread_open=-1.0,
        spread_close=-1.5,
        sharp_money_detected=True,
        steam_move_detected=False
    )
    print(f"  Method: {result['method']}")
    print(f"  Home Win: {result['home_win']:.1%}")
    print(f"  Draw: {result['draw']:.1%}")
    print(f"  Away Win: {result['away_win']:.1%}")
    print(f"  Ensemble Weight xG: {result['ensemble_weight']['xg']:.1%}")
    print(f"  Ensemble Weight Spread: {result['ensemble_weight']['spread']:.1%}")
    print(f"  Signal Confidence: {result['signal_confidence']:.1%}")

    # Verifica che dia più peso allo spread con sharp money
    assert result['ensemble_weight']['spread'] > 0.20, "Deve dare >20% peso a spread con sharp money"
    assert 0.99 < result['home_win'] + result['draw'] + result['away_win'] < 1.01, "Deve sommare a 1"
    print(f"  ✓ PASS")

    # Test 2: Senza sharp money
    print("\n[Test 6b] Senza sharp money")
    result = calculate_market_adjusted_probabilities(
        home_xg=1.5,
        away_xg=1.0,
        spread_open=-1.0,
        spread_close=-1.1,
        sharp_money_detected=False,
        steam_move_detected=False
    )
    print(f"  Signal Confidence: {result['signal_confidence']:.1%}")
    print(f"  Ensemble Weight Spread: {result['ensemble_weight']['spread']:.1%}")
    assert result['ensemble_weight']['spread'] < 0.20, "Deve dare <20% peso a spread senza sharp"
    print(f"  ✓ PASS")


def test_market_confidence_score():
    """Test confidence score"""
    print("\n" + "="*70)
    print("TEST 7: MARKET CONFIDENCE SCORE")
    print("="*70)

    # Crea dati mock
    spread_analysis = MovementAnalysis(
        direction=MovementDirection.HARDEN,
        intensity=MovementIntensity.MEDIUM,
        opening_value=-1.0,
        closing_value=-1.5,
        movement_steps=0.5,
        interpretation="Test"
    )

    total_analysis = MovementAnalysis(
        direction=MovementDirection.HARDEN,
        intensity=MovementIntensity.MEDIUM,
        opening_value=2.5,
        closing_value=2.75,
        movement_steps=0.25,
        interpretation="Test"
    )

    market_intel = MarketIntelligence(
        sharp_money_detected=True,
        sharp_spread_velocity=20.0,
        sharp_total_velocity=12.0,
        contrarian_signal=False,
        sharp_confidence_boost=0.15,
        steam_move_detected=True,
        steam_magnitude=0.5,
        reverse_steam=False,
        steam_direction="favorito",
        correlation_score=0.8,
        correlation_interpretation="Coerente",
        market_coherent=True,
        on_key_spread=False,
        on_key_total=True,
        spread_key_number=None,
        total_key_number=2.75,
        key_confidence_boost=0.10,
        efficiency_score=88.0,
        efficiency_status="Efficient",
        value_opportunity=False
    )

    # Test: Alta confidenza (sharp + steam + coherent + efficient)
    print("\n[Test 7] Alta confidenza (tutti i segnali positivi)")
    score = calculate_market_confidence_score(
        spread_analysis,
        total_analysis,
        market_intel,
        prediction_variance=0.08
    )
    print(f"  Confidence Score: {score:.0f}/100")
    print(f"  Expected: >80 (sharp + steam + coherent + efficient + low variance)")
    assert score > 80, f"Score deve essere >80 con tutti i segnali positivi, got {score}"
    print(f"  ✓ PASS")


def test_full_integration():
    """Test integrazione completa con analyzer"""
    print("\n" + "="*70)
    print("TEST 8: INTEGRAZIONE COMPLETA (OPZIONE C)")
    print("="*70)

    analyzer = MarketMovementAnalyzer()

    # Test scenario: Sharp money + steam move
    print("\n[Test 8] Scenario completo: Sharp money + Steam move")
    print("  Input: Spread -1.0→-1.75, Total 2.5→2.75")

    result = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.75,
        total_open=2.5,
        total_close=2.75
    )

    xg = result.expected_goals
    intel = result.market_intelligence

    print(f"\n  === EXPECTED GOALS (Opzione C) ===")
    print(f"  Home xG: {xg.home_xg:.2f}")
    print(f"  Away xG: {xg.away_xg:.2f}")
    print(f"  Home Win: {xg.home_win_prob:.1%}")
    print(f"  Draw: {xg.draw_prob:.1%}")
    print(f"  Away Win: {xg.away_win_prob:.1%}")
    print(f"  BTTS: {xg.btts_prob:.1%}")

    print(f"\n  === ADVANCED FIELDS ===")
    print(f"  Prediction Method: {xg.prediction_method}")
    print(f"  Confidence Score: {xg.confidence_score:.0f}/100")

    if xg.ensemble_weights:
        print(f"  Ensemble Weight xG: {xg.ensemble_weights['xg']:.1%}")
        print(f"  Ensemble Weight Spread: {xg.ensemble_weights['spread']:.1%}")

    if xg.market_adjusted_1x2:
        print(f"\n  Market-Adjusted 1X2:")
        print(f"    Home: {xg.market_adjusted_1x2['home_win']:.1%}")
        print(f"    Draw: {xg.market_adjusted_1x2['draw']:.1%}")
        print(f"    Away: {xg.market_adjusted_1x2['away_win']:.1%}")

    if xg.bayesian_btts:
        print(f"\n  Bayesian BTTS:")
        print(f"    BTTS: {xg.bayesian_btts['btts_prob']:.1%}")
        print(f"    Method: {xg.bayesian_btts['method']}")

    print(f"\n  === MARKET INTELLIGENCE ===")
    print(f"  Sharp Money: {intel.sharp_money_detected}")
    print(f"  Steam Move: {intel.steam_move_detected}")
    print(f"  Market Coherent: {intel.market_coherent}")

    # Verifica che tutti i campi advanced siano popolati
    assert xg.confidence_score is not None, "Confidence score deve essere calcolato"
    assert xg.market_adjusted_1x2 is not None, "Market adjusted 1x2 deve essere calcolato"
    assert xg.bayesian_btts is not None, "Bayesian BTTS deve essere calcolato"
    assert xg.prediction_method is not None, "Prediction method deve essere specificato"
    assert xg.ensemble_weights is not None, "Ensemble weights deve essere specificato"

    print(f"\n  ✓ TEST INTEGRAZIONE COMPLETA PASSATO!")


def run_all_tests():
    """Esegue tutti i test"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "OPZIONE C - ADVANCED TESTS" + " "*27 + "║")
    print("╚" + "="*68 + "╝")

    try:
        test_spread_to_implied_probability()
        test_dynamic_home_advantage()
        test_bayesian_probability_update()
        test_monte_carlo_btts()
        test_smart_btts()
        test_market_adjusted_probabilities()
        test_market_confidence_score()
        test_full_integration()

        print("\n" + "="*70)
        print("✅ TUTTI I TEST PASSATI! Opzione C funziona perfettamente!")
        print("="*70 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST FALLITO: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
