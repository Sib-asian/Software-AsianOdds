#!/usr/bin/env python3
"""
Test per verificare che lo spread sia interpretato correttamente
"""

from market_movement_analyzer import MarketMovementAnalyzer, format_output

print("=" * 70)
print("TEST: Verifica interpretazione spread")
print("=" * 70)
print()

analyzer = MarketMovementAnalyzer()

# Test 1: Spread positivo (favorito trasferta)
print("📊 TEST 1: Spread 0 → 0.25 (positivo)")
print("-" * 70)
result1 = analyzer.analyze(0, 0.25, 2.5, 2.5)
print(f"Spread closing: {result1.spread_analysis.closing_value}")
print(f"Chi è il favorito? ", end="")
for rec in result1.core_recommendations:
    if rec.market_name == "1X2":
        if "2" in rec.recommendation:
            print("✅ CORRETTO - Favorito 2 (trasferta)")
        else:
            print("❌ SBAGLIATO - Favorito 1 (casa)")
        break
print()

# Test 2: Spread negativo (favorito casa)
print("📊 TEST 2: Spread 0 → -0.25 (negativo)")
print("-" * 70)
result2 = analyzer.analyze(0, -0.25, 2.5, 2.5)
print(f"Spread closing: {result2.spread_analysis.closing_value}")
print(f"Chi è il favorito? ", end="")
for rec in result2.core_recommendations:
    if rec.market_name == "1X2":
        if "1" in rec.recommendation:
            print("✅ CORRETTO - Favorito 1 (casa)")
        else:
            print("❌ SBAGLIATO - Favorito 2 (trasferta)")
        break
print()

# Mostra raccomandazioni complete Test 1
print("=" * 70)
print("RACCOMANDAZIONI COMPLETE - Test 1 (spread 0.25, favorito 2)")
print("=" * 70)
print()
print("CORE:")
for i, rec in enumerate(result1.core_recommendations[:5], 1):
    print(f"{i}. {rec.market_name}: {rec.recommendation}")

print()
print("ALTERNATIVE:")
for i, rec in enumerate(result1.alternative_recommendations[:3], 1):
    print(f"{i}. {rec.market_name}: {rec.recommendation}")

print()
print("VALUE:")
for i, rec in enumerate(result1.value_recommendations[:3], 1):
    print(f"{i}. {rec.market_name}: {rec.recommendation}")

print()
print("=" * 70)
print("✅ Se vedi 'Favorito 2' nel Test 1, le modifiche funzionano!")
print("=" * 70)
