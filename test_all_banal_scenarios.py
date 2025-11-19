"""
Test completo per verificare tutte le situazioni banali possibili
"""
import sys
from datetime import datetime
from live_betting_advisor import LiveBettingAdvisor

def test_scenario(description, match_data, live_data, expected_no_markets=None, expected_markets=None):
    """Test uno scenario specifico"""
    advisor = LiveBettingAdvisor()
    opportunities = advisor.analyze_live_match('test', match_data, live_data)
    
    markets = [opp.market.lower() for opp in opportunities]
    
    print(f"\n🧪 {description}")
    print(f"   Score: {live_data.get('score_home', 0)}-{live_data.get('score_away', 0)} al {live_data.get('minute', 0)}'")
    
    if expected_no_markets:
        found_banal = [m for m in markets if any(banal in m for banal in expected_no_markets)]
        if len(found_banal) == 0:
            print(f"   ✅ PASS: Nessun mercato banale trovato")
            return True
        else:
            print(f"   ❌ FAIL: Trovati mercati banali: {found_banal}")
            return False
    
    if expected_markets:
        found_expected = [m for m in markets if any(exp in m for exp in expected_markets)]
        if len(found_expected) > 0:
            print(f"   ✅ PASS: Mercati attesi trovati: {found_expected[:3]}")
            return True
        else:
            print(f"   ⚠️  WARNING: Nessun mercato atteso trovato (potrebbe essere OK)")
            return True
    
    return True

def run_comprehensive_tests():
    """Esegue test completi per tutte le situazioni banali"""
    print("=" * 70)
    print("🧪 TEST COMPLETO - TUTTE LE SITUAZIONI BANALI")
    print("=" * 70)
    
    results = []
    
    # Test 1: Over 0.5 quando c'è già 1 gol
    results.append(test_scenario(
        "Over 0.5 quando c'è già 1 gol",
        {'home': 'A', 'away': 'B', 'odds_1': 2.0, 'odds_2': 2.5},
        {'score_home': 1, 'score_away': 0, 'minute': 30},
        expected_no_markets=['over_0.5']
    ))
    
    # Test 2: Over 1.5 quando ci sono già 2 gol
    results.append(test_scenario(
        "Over 1.5 quando ci sono già 2 gol",
        {'home': 'A', 'away': 'B', 'odds_1': 2.0, 'odds_2': 2.5},
        {'score_home': 1, 'score_away': 1, 'minute': 40},
        expected_no_markets=['over_1.5']
    ))
    
    # Test 3: Over 2.5 quando ci sono già 3 gol
    results.append(test_scenario(
        "Over 2.5 quando ci sono già 3 gol",
        {'home': 'A', 'away': 'B', 'odds_1': 2.0, 'odds_2': 2.5},
        {'score_home': 2, 'score_away': 1, 'minute': 50},
        expected_no_markets=['over_2.5']
    ))
    
    # Test 4: 1X quando è già 1-0
    results.append(test_scenario(
        "1X quando è già 1-0",
        {'home': 'A', 'away': 'B', 'odds_1': 1.5, 'odds_2': 3.0},
        {'score_home': 1, 'score_away': 0, 'minute': 30},
        expected_no_markets=['1x']
    ))
    
    # Test 5: X2 quando è già 0-1
    results.append(test_scenario(
        "X2 quando è già 0-1",
        {'home': 'A', 'away': 'B', 'odds_1': 2.5, 'odds_2': 1.5},
        {'score_home': 0, 'score_away': 1, 'minute': 30},
        expected_no_markets=['x2']
    ))
    
    # Test 6: Segno 1 quando è già 2-0 avanzato
    results.append(test_scenario(
        "Segno 1 quando è già 2-0 avanzato",
        {'home': 'A', 'away': 'B', 'odds_1': 1.5, 'odds_2': 3.0},
        {'score_home': 2, 'score_away': 0, 'minute': 70},
        expected_no_markets=['home_win', '1x2_home']
    ))
    
    # Test 7: Clean Sheet quando è 3-0 all'85'
    results.append(test_scenario(
        "Clean Sheet quando è 3-0 all'85'",
        {'home': 'A', 'away': 'B', 'odds_1': 1.5, 'odds_2': 3.0},
        {'score_home': 3, 'score_away': 0, 'minute': 85, 'shots_on_target_away': 1},
        expected_no_markets=['clean_sheet']
    ))
    
    # Test 8: BTTS Yes quando entrambe hanno segnato
    results.append(test_scenario(
        "BTTS Yes quando entrambe hanno segnato",
        {'home': 'A', 'away': 'B', 'odds_1': 2.0, 'odds_2': 2.5},
        {'score_home': 1, 'score_away': 1, 'minute': 40},
        expected_no_markets=['btts_yes']
    ))
    
    # Test 9: Win to Nil quando è 2-0 avanzato
    results.append(test_scenario(
        "Win to Nil quando è 2-0 avanzato",
        {'home': 'A', 'away': 'B', 'odds_1': 1.5, 'odds_2': 3.0},
        {'score_home': 2, 'score_away': 0, 'minute': 75, 'shots_on_target_away': 0},
        expected_no_markets=['win_to_nil']
    ))
    
    # Test 10: Goal Range 0-1 quando c'è già 1 gol avanzato
    results.append(test_scenario(
        "Goal Range 0-1 quando c'è già 1 gol avanzato",
        {'home': 'A', 'away': 'B', 'odds_1': 2.0, 'odds_2': 2.5},
        {'score_home': 1, 'score_away': 0, 'minute': 65},
        expected_no_markets=['goal_range_0_1']
    ))
    
    # Test 11: Goal Range 2-3 quando ci sono già 4 gol
    results.append(test_scenario(
        "Goal Range 2-3 quando ci sono già 4 gol",
        {'home': 'A', 'away': 'B', 'odds_1': 2.0, 'odds_2': 2.5},
        {'score_home': 2, 'score_away': 2, 'minute': 60},
        expected_no_markets=['goal_range_2_3']
    ))
    
    # Test 12: Team to Score First quando hanno già segnato
    results.append(test_scenario(
        "Team to Score First quando hanno già segnato",
        {'home': 'A', 'away': 'B', 'odds_1': 2.0, 'odds_2': 2.5},
        {'score_home': 1, 'score_away': 0, 'minute': 30},
        expected_no_markets=['team_to_score_first']
    ))
    
    # Test 13: Prossimo gol quando è troppo tardi (87')
    results.append(test_scenario(
        "Prossimo gol quando è troppo tardi (87')",
        {'home': 'A', 'away': 'B', 'odds_1': 2.0, 'odds_2': 2.5},
        {'score_home': 1, 'score_away': 0, 'minute': 87},
        expected_no_markets=['next_goal', 'team_to_score_next']
    ))
    
    # Test 14: Over quando è troppo tardi (86')
    results.append(test_scenario(
        "Over quando è troppo tardi (86')",
        {'home': 'A', 'away': 'B', 'odds_1': 2.0, 'odds_2': 2.5},
        {'score_home': 1, 'score_away': 0, 'minute': 86, 'shots_home': 10, 'shots_away': 5},
        expected_no_markets=['over']
    ))
    
    # Test 15: Partita decisa (4-0) - nessun mercato risultato
    results.append(test_scenario(
        "Partita decisa (4-0) - nessun mercato risultato",
        {'home': 'A', 'away': 'B', 'odds_1': 1.5, 'odds_2': 3.0},
        {'score_home': 4, 'score_away': 0, 'minute': 60},
        expected_no_markets=['home_win', 'away_win', 'match_winner', 'dnb', '1x', 'x2']
    ))
    
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO TEST COMPLETI")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"✅ Test passati: {passed}/{total}")
    print(f"❌ Test falliti: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 TUTTI I TEST PASSATI!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} TEST FALLITI")
        return 1

if __name__ == '__main__':
    sys.exit(run_comprehensive_tests())

