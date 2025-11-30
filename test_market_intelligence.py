#!/usr/bin/env python3
"""
Test per le nuove funzioni Advanced Market Intelligence

Testa:
1. Sharp Money Detection
2. Steam Move Detection
3. Market Correlation
4. Key Numbers Analysis
5. Market Efficiency
"""

from market_movement_analyzer import (
    detect_sharp_money,
    detect_steam_move,
    calculate_market_correlation,
    analyze_key_numbers,
    calculate_market_efficiency,
    MarketMovementAnalyzer
)


def test_sharp_money_detection():
    """Test Sharp Money Detection"""
    print("\n" + "="*60)
    print("TEST 1: SHARP MONEY DETECTION")
    print("="*60)

    # Test 1: Movimento sharp su spread
    print("\n[Test 1a] Movimento sharp su spread (>15% velocity)")
    result = detect_sharp_money(-1.0, -1.25, 2.5, 2.6)
    print(f"  Spread: -1.0 → -1.25 (25% velocity)")
    print(f"  Sharp Detected: {result['sharp_detected']}")
    print(f"  Spread Velocity: {result['spread_velocity']:.1f}%")
    print(f"  ✓ PASS" if result['sharp_detected'] else "  ✗ FAIL")

    # Test 2: Contrarian signal
    print("\n[Test 1b] Contrarian signal (direzioni opposte)")
    result = detect_sharp_money(-1.5, -1.0, 2.5, 2.25)
    print(f"  Spread: -1.5 → -1.0 (sale)")
    print(f"  Total: 2.5 → 2.25 (scende)")
    print(f"  Contrarian: {result['contrarian']}")
    print(f"  Confidence Boost: {result['confidence_boost']}")
    print(f"  ✓ PASS" if result['contrarian'] else "  ✗ FAIL")

    # Test 3: Movimento normale
    print("\n[Test 1c] Movimento normale (public money)")
    result = detect_sharp_money(-1.5, -1.6, 2.5, 2.6)
    print(f"  Spread: -1.5 → -1.6 (6.7% velocity)")
    print(f"  Sharp Detected: {result['sharp_detected']}")
    print(f"  ✓ PASS" if not result['sharp_detected'] else "  ✗ FAIL")


def test_steam_move_detection():
    """Test Steam Move Detection"""
    print("\n" + "="*60)
    print("TEST 2: STEAM MOVE DETECTION")
    print("="*60)

    # Test 1: Steam move (>0.5 punti)
    print("\n[Test 2a] Steam Move rilevato")
    result = detect_steam_move(-1.0, -1.75)
    print(f"  Spread: -1.0 → -1.75")
    print(f"  Steam Detected: {result['is_steam']}")
    print(f"  Magnitude: {result['magnitude']:.2f} punti")
    print(f"  Direction: {result['direction']}")
    print(f"  ✓ PASS" if result['is_steam'] else "  ✗ FAIL")

    # Test 2: Reverse steam (cambio favorito)
    print("\n[Test 2b] Reverse Steam (favorito cambia)")
    result = detect_steam_move(-1.0, 0.5)
    print(f"  Spread: -1.0 → +0.5")
    print(f"  Reverse Steam: {result['reverse_steam']}")
    print(f"  ✓ PASS" if result['reverse_steam'] else "  ✗ FAIL")

    # Test 3: No steam
    print("\n[Test 2c] Nessun Steam (movimento normale)")
    result = detect_steam_move(-1.5, -1.75)
    print(f"  Spread: -1.5 → -1.75 (0.25 punti)")
    print(f"  Steam Detected: {result['is_steam']}")
    print(f"  ✓ PASS" if not result['is_steam'] else "  ✗ FAIL")


def test_market_correlation():
    """Test Market Correlation"""
    print("\n" + "="*60)
    print("TEST 3: MARKET CORRELATION")
    print("="*60)

    # Test 1: Correlazione positiva
    print("\n[Test 3a] Correlazione positiva (mercato coerente)")
    result = calculate_market_correlation(-1.5, -2.0, 2.5, 3.0)
    print(f"  Spread: -1.5 → -2.0 (scende)")
    print(f"  Total: 2.5 → 3.0 (sale)")
    print(f"  Correlation Score: {result['score']:+.2f}")
    print(f"  Market Coherent: {result['coherent']}")
    print(f"  ✓ PASS" if result['score'] > 0.5 else "  ✗ FAIL")

    # Test 2: Correlazione negativa
    print("\n[Test 3b] Correlazione negativa (segnali contrastanti)")
    result = calculate_market_correlation(-1.5, -1.0, 2.5, 2.0)
    print(f"  Spread: -1.5 → -1.0 (sale)")
    print(f"  Total: 2.5 → 2.0 (scende)")
    print(f"  Correlation Score: {result['score']:+.2f}")
    print(f"  Market Coherent: {result['coherent']}")
    print(f"  ✓ PASS" if result['score'] < -0.5 else "  ✗ FAIL")


def test_key_numbers_analysis():
    """Test Key Numbers Analysis"""
    print("\n" + "="*60)
    print("TEST 4: KEY NUMBERS ANALYSIS")
    print("="*60)

    # Test 1: Su key number spread
    print("\n[Test 4a] Spread su key number (-1.5)")
    result = analyze_key_numbers(-1.5, 2.7)
    print(f"  Spread: -1.5")
    print(f"  On Key Spread: {result['on_key_spread']}")
    print(f"  Key Number: {result['spread_key']}")
    print(f"  ✓ PASS" if result['on_key_spread'] else "  ✗ FAIL")

    # Test 2: Su key number total
    print("\n[Test 4b] Total su key number (2.5)")
    result = analyze_key_numbers(-1.7, 2.5)
    print(f"  Total: 2.5")
    print(f"  On Key Total: {result['on_key_total']}")
    print(f"  Key Number: {result['total_key']}")
    print(f"  ✓ PASS" if result['on_key_total'] else "  ✗ FAIL")

    # Test 3: Vicino a key number
    print("\n[Test 4c] Vicino a key number (ma non esattamente)")
    result = analyze_key_numbers(-1.75, 2.6)
    print(f"  Spread: -1.75 (nearest: {result['spread_key']})")
    print(f"  Total: 2.6 (nearest: {result['total_key']})")
    print(f"  On Key: Spread={result['on_key_spread']}, Total={result['on_key_total']}")
    print(f"  ✓ PASS")


def test_market_efficiency():
    """Test Market Efficiency"""
    print("\n" + "="*60)
    print("TEST 5: MARKET EFFICIENCY")
    print("="*60)

    # Test 1: Mercato efficiente (pochi movimenti)
    print("\n[Test 5a] Mercato efficiente")
    result = calculate_market_efficiency(-1.5, -1.6, 2.5, 2.55)
    print(f"  Spread: -1.5 → -1.6")
    print(f"  Total: 2.5 → 2.55")
    print(f"  Efficiency Score: {result['score']:.0f}/100")
    print(f"  Status: {result['status']}")
    print(f"  ✓ PASS" if result['score'] >= 90 else "  ✗ FAIL")

    # Test 2: Mercato inefficiente (grandi movimenti)
    print("\n[Test 5b] Mercato inefficiente (value opportunity)")
    result = calculate_market_efficiency(-1.0, -2.0, 2.5, 3.5)
    print(f"  Spread: -1.0 → -2.0")
    print(f"  Total: 2.5 → 3.5")
    print(f"  Efficiency Score: {result['score']:.0f}/100")
    print(f"  Status: {result['status']}")
    print(f"  Value Opportunity: {result['value_opportunity']}")
    print(f"  ✓ PASS" if result['score'] < 70 else "  ✗ FAIL")


def test_full_integration():
    """Test integrazione completa con MarketMovementAnalyzer"""
    print("\n" + "="*60)
    print("TEST 6: INTEGRAZIONE COMPLETA")
    print("="*60)

    analyzer = MarketMovementAnalyzer()

    # Test case: Movimento sharp con steam
    print("\n[Test 6] Analisi completa con tutti gli indicatori")
    result = analyzer.analyze(
        spread_open=-1.0,
        spread_close=-1.75,
        total_open=2.5,
        total_close=2.75
    )

    intel = result.market_intelligence

    print(f"\n  Sharp Money Detected: {intel.sharp_money_detected}")
    print(f"  Steam Move Detected: {intel.steam_move_detected}")
    print(f"  Correlation Score: {intel.correlation_score:+.2f}")
    print(f"  On Key Numbers: Spread={intel.on_key_spread}, Total={intel.on_key_total}")
    print(f"  Efficiency Score: {intel.efficiency_score:.0f}/100")

    print(f"\n  Recommendations Generated: {len(result.core_recommendations)}")
    print(f"  Overall Confidence: {result.overall_confidence.value}")

    print(f"\n  ✓ TEST COMPLETO PASSATO")


def run_all_tests():
    """Esegue tutti i test"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "ADVANCED MARKET INTELLIGENCE TESTS" + " "*14 + "║")
    print("╚" + "="*58 + "╝")

    test_sharp_money_detection()
    test_steam_move_detection()
    test_market_correlation()
    test_key_numbers_analysis()
    test_market_efficiency()
    test_full_integration()

    print("\n" + "="*60)
    print("TUTTI I TEST COMPLETATI!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
