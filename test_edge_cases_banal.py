"""
Test edge cases per verificare situazioni limite banali
"""
import sys
from live_betting_advisor import LiveBettingAdvisor

def test_edge_cases():
    """Test situazioni limite"""
    print("=" * 70)
    print("🧪 TEST EDGE CASES - SITUAZIONI LIMITE")
    print("=" * 70)
    
    advisor = LiveBettingAdvisor()
    results = []
    
    # Test 1: Partita 5-0 all'80' - nessun mercato
    print("\n🧪 Test 1: Partita 5-0 all'80'")
    match_data = {'home': 'A', 'away': 'B', 'odds_1': 1.5, 'odds_2': 3.0}
    live_data = {'score_home': 5, 'score_away': 0, 'minute': 80}
    opps = advisor.analyze_live_match('test1', match_data, live_data)
    result_markets = [opp for opp in opps if any(m in opp.market.lower() for m in ['win', 'match_winner', 'dnb', '1x', 'x2'])]
    if len(result_markets) == 0:
        print("   ✅ PASS: Nessun mercato risultato quando partita è 5-0 all'80'")
        results.append(True)
    else:
        print(f"   ❌ FAIL: Trovati {len(result_markets)} mercati risultato")
        results.append(False)
    
    # Test 2: 0-0 all'88' - nessun mercato risultato
    print("\n🧪 Test 2: 0-0 all'88'")
    live_data = {'score_home': 0, 'score_away': 0, 'minute': 88}
    opps = advisor.analyze_live_match('test2', match_data, live_data)
    result_markets = [opp for opp in opps if any(m in opp.market.lower() for m in ['win', 'match_winner'])]
    if len(result_markets) == 0:
        print("   ✅ PASS: Nessun mercato risultato quando è 0-0 all'88'")
        results.append(True)
    else:
        print(f"   ❌ FAIL: Trovati {len(result_markets)} mercati risultato")
        results.append(False)
    
    # Test 3: 1-0 all'89' - nessun mercato risultato
    print("\n🧪 Test 3: 1-0 all'89'")
    live_data = {'score_home': 1, 'score_away': 0, 'minute': 89}
    opps = advisor.analyze_live_match('test3', match_data, live_data)
    result_markets = [opp for opp in opps if any(m in opp.market.lower() for m in ['win', 'match_winner'])]
    if len(result_markets) == 0:
        print("   ✅ PASS: Nessun mercato risultato quando è 1-0 all'89'")
        results.append(True)
    else:
        print(f"   ❌ FAIL: Trovati {len(result_markets)} mercati risultato")
        results.append(False)
    
    # Test 4: 2-2 all'85' - nessun mercato risultato (troppo rischioso)
    print("\n🧪 Test 4: 2-2 all'85'")
    live_data = {'score_home': 2, 'score_away': 2, 'minute': 85}
    opps = advisor.analyze_live_match('test4', match_data, live_data)
    result_markets = [opp for opp in opps if any(m in opp.market.lower() for m in ['win', 'match_winner'])]
    if len(result_markets) == 0:
        print("   ✅ PASS: Nessun mercato risultato quando è 2-2 all'85'")
        results.append(True)
    else:
        print(f"   ❌ FAIL: Trovati {len(result_markets)} mercati risultato")
        results.append(False)
    
    # Test 5: Over 0.5 quando è 1-0 al 1' (già superato)
    print("\n🧪 Test 5: Over 0.5 quando è 1-0 al 1'")
    live_data = {'score_home': 1, 'score_away': 0, 'minute': 1}
    opps = advisor.analyze_live_match('test5', match_data, live_data)
    over_05 = [opp for opp in opps if 'over_0.5' in opp.market.lower() and 'ht' not in opp.market.lower()]
    if len(over_05) == 0:
        print("   ✅ PASS: Nessun Over 0.5 quando è già 1-0")
        results.append(True)
    else:
        print(f"   ❌ FAIL: Trovati {len(over_05)} Over 0.5")
        results.append(False)
    
    # Test 6: Under 1.5 quando è 1-0 al 50' (illogico)
    print("\n🧪 Test 6: Under 1.5 quando è 1-0 al 50'")
    live_data = {'score_home': 1, 'score_away': 0, 'minute': 50}
    opps = advisor.analyze_live_match('test6', match_data, live_data)
    under_15 = [opp for opp in opps if 'under_1.5' in opp.market.lower() and 'ht' not in opp.market.lower()]
    if len(under_15) == 0:
        print("   ✅ PASS: Nessun Under 1.5 quando è già 1-0 al 50'")
        results.append(True)
    else:
        print(f"   ❌ FAIL: Trovati {len(under_15)} Under 1.5")
        results.append(False)
    
    # Test 7: BTTS Yes quando è 1-1 al 1' (già successo)
    print("\n🧪 Test 7: BTTS Yes quando è 1-1 al 1'")
    live_data = {'score_home': 1, 'score_away': 1, 'minute': 1}
    opps = advisor.analyze_live_match('test7', match_data, live_data)
    btts_yes = [opp for opp in opps if 'btts_yes' in opp.market.lower()]
    if len(btts_yes) == 0:
        print("   ✅ PASS: Nessun BTTS Yes quando entrambe hanno già segnato")
        results.append(True)
    else:
        print(f"   ❌ FAIL: Trovati {len(btts_yes)} BTTS Yes")
        results.append(False)
    
    # Test 8: Team to Score First quando è 1-0 al 1'
    print("\n🧪 Test 8: Team to Score First quando è 1-0 al 1'")
    live_data = {'score_home': 1, 'score_away': 0, 'minute': 1}
    opps = advisor.analyze_live_match('test8', match_data, live_data)
    score_first = [opp for opp in opps if 'team_to_score_first' in opp.market.lower()]
    if len(score_first) == 0:
        print("   ✅ PASS: Nessun Team to Score First quando hanno già segnato")
        results.append(True)
    else:
        print(f"   ❌ FAIL: Trovati {len(score_first)} Team to Score First")
        results.append(False)
    
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO EDGE CASES")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"✅ Test passati: {passed}/{total}")
    print(f"❌ Test falliti: {total - passed}/{total}")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(test_edge_cases())

