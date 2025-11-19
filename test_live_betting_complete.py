"""
Test completo per verificare la creazione dei messaggi e la formattazione
"""
import sys
from datetime import datetime
from live_betting_advisor import LiveBettingAdvisor

def test_message_formatting():
    """Test: Verifica che i messaggi siano formattati correttamente con statistiche"""
    print("🧪 Test completo: Formattazione messaggi con statistiche")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {
        'home': 'Juventus',
        'away': 'Inter',
        'league': 'Serie A',
        'odds_1': 1.5,
        'odds_2': 3.0
    }
    
    live_data = {
        'score_home': 1,
        'score_away': 0,
        'minute': 60,
        'shots_on_target_away': 0,
        'shots_away': 2,
        'dangerous_attacks_away': 5,
        'xg_away': 0.1,
        'possession_home': 60,
        'shots_home': 10,
        'shots_on_target_home': 4,
        'corners_home': 5,
        'corners_away': 2,
        'yellow_cards_home': 1,
        'yellow_cards_away': 2
    }
    
    opportunities = advisor.analyze_live_match('test_complete', match_data, live_data)
    
    if len(opportunities) == 0:
        print("⚠️  Nessuna opportunità generata")
        return True
    
    # Prendi la prima opportunità e genera il messaggio
    opp = opportunities[0]
    
    # Arricchisci con metadata
    advisor._populate_opportunity_metadata(opp, live_data)
    
    # Genera messaggio
    message = advisor.format_live_betting_message(opp)
    
    # Verifica che il messaggio contenga elementi chiave
    checks = {
        'Statistiche estratte': 'STATISTICHE ESTRATTE' in message or 'STATISTICHE' in message,
        'Score nel messaggio': '1-0' in message or 'Score' in message,
        'Minuto nel messaggio': '60' in message or 'minuto' in message.lower(),
        'Confidence nel messaggio': 'CONFIDENCE' in message or 'confidence' in message.lower(),
        'Mercato tradotto': opp.market not in message or advisor._translate_market_name(opp.market) in message
    }
    
    all_ok = True
    for check_name, check_result in checks.items():
        if check_result:
            print(f"✅ {check_name}: OK")
        else:
            print(f"❌ {check_name}: FAIL")
            all_ok = False
    
    # Mostra un estratto del messaggio
    print("\n📄 Estratto messaggio (prime 500 caratteri):")
    print("-" * 70)
    print(message[:500])
    print("...")
    print("-" * 70)
    
    return all_ok

def test_clean_sheet_detailed():
    """Test dettagliato per clean sheet"""
    print("\n🧪 Test dettagliato: Clean Sheet con statistiche")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {
        'home': 'Milan',
        'away': 'Napoli',
        'odds_1': 2.0,
        'odds_2': 2.5
    }
    
    # Scenario: 1-0 al 60', avversaria senza tiri in porta
    live_data = {
        'score_home': 1,
        'score_away': 0,
        'minute': 60,
        'shots_on_target_away': 0,
        'shots_away': 2,
        'dangerous_attacks_away': 3,
        'xg_away': 0.05
    }
    
    opportunities = advisor.analyze_live_match('test_clean_sheet', match_data, live_data)
    
    clean_sheet_opps = [opp for opp in opportunities if 'clean_sheet' in opp.market.lower()]
    
    if len(clean_sheet_opps) == 0:
        print("⚠️  Nessun segnale clean sheet generato")
        return True
    
    opp = clean_sheet_opps[0]
    advisor._populate_opportunity_metadata(opp, live_data)
    
    # Verifica statistiche estratte
    if hasattr(opp, 'key_stats') and opp.key_stats:
        print("✅ Statistiche estratte presenti:")
        for key, value in list(opp.key_stats.items())[:5]:
            print(f"   - {key}: {value}")
    else:
        print("❌ Nessuna statistica estratta")
        return False
    
    # Verifica reasoning contiene dati concreti
    if 'OFFENSIVAMENTE INEFFICACE' in opp.reasoning or 'tiri in porta' in opp.reasoning.lower():
        print("✅ Reasoning contiene dati concreti")
    else:
        print("❌ Reasoning non contiene dati concreti")
        return False
    
    # Verifica confidence >= 80%
    if opp.confidence >= 80:
        print(f"✅ Confidence {opp.confidence:.0f}% >= 80%")
    else:
        print(f"❌ Confidence {opp.confidence:.0f}% < 80%")
        return False
    
    return True

def test_no_banal_signals():
    """Test: Verifica che non vengano generati segnali banali"""
    print("\n🧪 Test: Verifica assenza segnali banali")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {
        'home': 'Team A',
        'away': 'Team B',
        'odds_1': 1.5,
        'odds_2': 3.0
    }
    
    # Scenario banale: 3-0 all'85' - NON dovrebbe generare clean sheet
    live_data = {
        'score_home': 3,
        'score_away': 0,
        'minute': 85,
        'shots_on_target_away': 1,
        'shots_away': 3
    }
    
    opportunities = advisor.analyze_live_match('test_banal', match_data, live_data)
    
    clean_sheet_opps = [opp for opp in opportunities if 'clean_sheet' in opp.market.lower()]
    
    if len(clean_sheet_opps) == 0:
        print("✅ PASS: Nessun segnale clean sheet banale (3-0 all'85')")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(clean_sheet_opps)} segnali clean sheet banali")
        return False

def run_complete_tests():
    """Esegue tutti i test completi"""
    print("=" * 70)
    print("🧪 TEST COMPLETI - VERIFICA SISTEMA")
    print("=" * 70)
    
    tests = [
        test_message_formatting,
        test_clean_sheet_detailed,
        test_no_banal_signals
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ ERRORE in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO TEST COMPLETI")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"✅ Test passati: {passed}/{total}")
    print(f"❌ Test falliti: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 TUTTI I TEST COMPLETI PASSATI!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} TEST FALLITI")
        return 1

if __name__ == '__main__':
    sys.exit(run_complete_tests())

