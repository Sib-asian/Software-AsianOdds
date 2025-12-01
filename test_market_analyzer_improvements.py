#!/usr/bin/env python3
"""
Test completo per verificare tutti i miglioramenti del Market Movement Analyzer
"""

import sys
from market_movement_analyzer import MarketMovementAnalyzer, format_output

def test_case(name: str, spread_open: float, spread_close: float, 
              total_open: float, total_close: float, expected_checks: list = None):
    """Test un caso specifico"""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    
    analyzer = MarketMovementAnalyzer()
    result = analyzer.analyze(spread_open, spread_close, total_open, total_close)
    
    # Output formattato
    output = format_output(result)
    print(output)
    
    # Verifiche automatiche
    if expected_checks:
        print("\n📋 Verifiche:")
        for check in expected_checks:
            if check['type'] == 'intensity':
                actual = result.spread_analysis.intensity if check['source'] == 'spread' else result.total_analysis.intensity
                expected = check.get('value')
                if expected:
                    status = "✅" if actual == expected else "❌"
                    print(f"  {status} {check['description']}: {actual.value} (atteso: {expected.value})")
                else:
                    print(f"  ℹ️  {check['description']}: {actual.value}")
            elif check['type'] == 'confidence':
                actual = result.overall_confidence
                expected = check.get('value')
                if expected:
                    status = "✅" if actual == expected else "❌"
                    print(f"  {status} {check['description']}: {actual.value} (atteso: {expected.value})")
                else:
                    print(f"  ℹ️  {check['description']}: {actual.value}")
            elif check['type'] == 'contains':
                all_recs = [r.recommendation for r in result.core_recommendations]
                all_recs.extend([r.recommendation for r in result.alternative_recommendations])
                all_recs_text = ' '.join(all_recs)
                contains = check['value'].lower() in all_recs_text.lower()
                status = "✅" if contains else "❌"
                print(f"  {status} {check['description']}: contiene '{check['value']}'")
    
    return result

def main():
    """Esegue tutti i test"""
    print("🧪 TEST COMPLETO MARKET MOVEMENT ANALYZER - MIGLIORAMENTI")
    print("="*70)
    
    # Test 1: Cambio segno spread
    test_case(
        "Cambio Favorito (Casa → Trasferta)",
        spread_open=-0.5,
        spread_close=+0.5,
        total_open=2.5,
        total_close=2.5,
        expected_checks=[
            {'type': 'intensity', 'source': 'spread', 'description': 'Intensità spread dopo cambio', 'value': None},
            {'type': 'contains', 'description': 'Rileva cambio favorito', 'value': 'Cambio favorito'}
        ]
    )
    
    # Test 2: Spread si indurisce forte
    test_case(
        "Spread Si Indurisce Forte + Total Sale",
        spread_open=-0.5,
        spread_close=-1.25,
        total_open=2.25,
        total_close=2.75,
        expected_checks=[
            {'type': 'confidence', 'description': 'Confidenza alta per segnali concordi', 'value': None},
            {'type': 'contains', 'description': 'Raccomanda Over', 'value': 'Over'},
            {'type': 'contains', 'description': 'Raccomanda GOAL', 'value': 'GOAL'}
        ]
    )
    
    # Test 3: GOAL/NOGOAL - Zona intermedia
    test_case(
        "GOAL/NOGOAL Zona Intermedia (2.0-2.5)",
        spread_open=-1.0,
        spread_close=-0.75,
        total_open=2.5,
        total_close=2.3,
        expected_checks=[
            {'type': 'contains', 'description': 'Gestisce zona intermedia', 'value': 'GOAL'}
        ]
    )
    
    # Test 4: Over/Under - Total scende ma ancora alto
    test_case(
        "Over/Under - Total Scende da Alto",
        spread_open=-1.0,
        spread_close=-1.0,
        total_open=3.0,
        total_close=2.8,
        expected_checks=[
            {'type': 'contains', 'description': 'Raccomanda Over (ancora alto)', 'value': 'Over'}
        ]
    )
    
    # Test 5: 1X2 - Spread ammorbidisce + Total sale
    test_case(
        "1X2 - Spread Ammorbidisce + Total Sale (Match Aperto)",
        spread_open=-0.75,
        spread_close=-0.5,
        total_open=2.5,
        total_close=2.75,
        expected_checks=[
            {'type': 'contains', 'description': 'Evita X se match aperto', 'value': '12'}
        ]
    )
    
    # Test 6: Handicap normalizzato
    test_case(
        "Handicap Asiatico - Valore Non Discreto",
        spread_open=-1.67,
        spread_close=-1.67,
        total_open=2.5,
        total_close=2.5,
        expected_checks=[
            {'type': 'contains', 'description': 'Handicap normalizzato', 'value': '-1.75'}
        ]
    )
    
    # Test 7: Spread molto alto
    test_case(
        "Spread Schiacciante (> 2.0)",
        spread_open=-2.25,
        spread_close=-2.5,
        total_open=3.0,
        total_close=3.0,
        expected_checks=[
            {'type': 'contains', 'description': 'Riconosce favorito dominante', 'value': 'dominante'}
        ]
    )
    
    # Test 8: Total molto basso
    test_case(
        "Total Molto Basso (< 1.75) - Partita Chiusissima",
        spread_open=-0.5,
        spread_close=-0.5,
        total_open=1.75,
        total_close=1.5,
        expected_checks=[
            {'type': 'contains', 'description': 'Raccomanda NOGOAL', 'value': 'NOGOAL'}
        ]
    )
    
    # Test 9: Segnali contrastanti
    test_case(
        "Segnali Contrastanti (Spread HARDEN + Total SOFTEN)",
        spread_open=-0.5,
        spread_close=-1.0,
        total_open=2.75,
        total_close=2.25,
        expected_checks=[
            {'type': 'confidence', 'description': 'Confidenza ridotta per contrasti', 'value': None}
        ]
    )
    
    # Test 10: Spread pick'em
    test_case(
        "Spread Pick'em (< 0.25)",
        spread_open=-0.25,
        spread_close=-0.25,
        total_open=2.5,
        total_close=2.5,
        expected_checks=[
            {'type': 'contains', 'description': 'Riconosce match 50-50', 'value': '50-50'}
        ]
    )
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETATI")
    print("="*70)
    print("\nVerificare manualmente i risultati sopra per confermare i miglioramenti.")
    print("Tutti i calcoli dovrebbero essere più precisi e dettagliati.")

if __name__ == "__main__":
    main()

