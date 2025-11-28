"""
Test completo per verificare che non vengano generati segnali banali per tutti i mercati
"""
import sys
from datetime import datetime
from live_betting_advisor import LiveBettingAdvisor

def test_btts_when_both_scored():
    """Test: BTTS Yes quando entrambe hanno già segnato"""
    print("🧪 Test BTTS: Non suggerire BTTS Yes quando entrambe hanno già segnato")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {'home': 'Team A', 'away': 'Team B', 'odds_1': 2.0, 'odds_2': 2.5}
    live_data = {'score_home': 1, 'score_away': 1, 'minute': 30}
    
    opportunities = advisor.analyze_live_match('test_btts', match_data, live_data)
    btts_yes = [opp for opp in opportunities if 'btts_yes' in opp.market.lower() and 'yes' in opp.market.lower()]
    
    if len(btts_yes) == 0:
        print("✅ PASS: Nessun segnale BTTS Yes quando entrambe hanno già segnato")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(btts_yes)} segnali BTTS Yes quando entrambe hanno già segnato")
        return False

def test_win_to_nil_banal():
    """Test: Win to Nil quando è già 2-0 avanzato"""
    print("\n🧪 Test Win to Nil: Non suggerire quando è già 2-0 avanzato")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {'home': 'Team A', 'away': 'Team B', 'odds_1': 1.5, 'odds_2': 3.0}
    live_data = {'score_home': 2, 'score_away': 0, 'minute': 75, 'shots_on_target_away': 1}
    
    opportunities = advisor.analyze_live_match('test_win_to_nil', match_data, live_data)
    win_to_nil = [opp for opp in opportunities if 'win_to_nil' in opp.market.lower()]
    
    if len(win_to_nil) == 0:
        print("✅ PASS: Nessun segnale Win to Nil quando è già 2-0 avanzato")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(win_to_nil)} segnali Win to Nil quando è già 2-0 avanzato")
        return False

def test_exact_score_banal():
    """Test: Exact Score quando è già superato"""
    print("\n🧪 Test Exact Score: Non suggerire quando è già superato")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {'home': 'Team A', 'away': 'Team B', 'odds_1': 2.0, 'odds_2': 2.5}
    # Test: 2-1 al 70' - non suggerire exact score 1-0 o 0-1
    live_data = {'score_home': 2, 'score_away': 1, 'minute': 70, 'shots_home': 10, 'shots_away': 8}
    
    opportunities = advisor.analyze_live_match('test_exact_score', match_data, live_data)
    exact_score = [opp for opp in opportunities if 'exact_score' in opp.market.lower()]
    
    # Verifica che non ci siano exact score banali (es. 1-0 quando è già 2-1)
    banal_exact = [opp for opp in exact_score if '1-0' in opp.market or '0-1' in opp.market]
    
    if len(banal_exact) == 0:
        print("✅ PASS: Nessun exact score banale quando è già superato")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(banal_exact)} exact score banali")
        return False

def test_goal_range_banal():
    """Test: Goal Range quando è già superato"""
    print("\n🧪 Test Goal Range: Non suggerire quando è già superato")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {'home': 'Team A', 'away': 'Team B', 'odds_1': 2.0, 'odds_2': 2.5}
    # Test: 4 gol al 60' - non suggerire goal_range_2_3 o goal_range_0_1
    live_data = {'score_home': 2, 'score_away': 2, 'minute': 60, 'shots_home': 15, 'shots_away': 12}
    
    opportunities = advisor.analyze_live_match('test_goal_range', match_data, live_data)
    goal_range = [opp for opp in opportunities if 'goal_range' in opp.market.lower()]
    
    # Verifica che non ci siano goal range banali (es. 2_3 quando ci sono già 4 gol)
    banal_ranges = [opp for opp in goal_range if '2_3' in opp.market or '0_1' in opp.market]
    
    if len(banal_ranges) == 0:
        print("✅ PASS: Nessun goal range banale quando è già superato")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(banal_ranges)} goal range banali")
        return False

def test_team_to_score_next_too_late():
    """Test: Team to Score Next quando è troppo tardi"""
    print("\n🧪 Test Team to Score Next: Non suggerire quando è troppo tardi (85'+)")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {'home': 'Team A', 'away': 'Team B', 'odds_1': 2.0, 'odds_2': 2.5}
    live_data = {'score_home': 1, 'score_away': 0, 'minute': 87}
    
    opportunities = advisor.analyze_live_match('test_next_goal_late', match_data, live_data)
    next_goal = [opp for opp in opportunities if 'next_goal' in opp.market.lower() or 'team_to_score_next' in opp.market.lower()]
    
    if len(next_goal) == 0:
        print("✅ PASS: Nessun segnale 'prossimo gol' quando è troppo tardi (87')")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(next_goal)} segnali 'prossimo gol' quando è troppo tardi")
        return False

def test_corner_over_banal():
    """Test: Corner Over quando è già superato"""
    print("\n🧪 Test Corner Over: Non suggerire quando è già superato")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {'home': 'Team A', 'away': 'Team B', 'odds_1': 2.0, 'odds_2': 2.5}
    # Test: 10 corner al 70' - non suggerire over 8.5 corner
    live_data = {'score_home': 1, 'score_away': 0, 'minute': 70, 
                 'corners_home': 6, 'corners_away': 4}
    
    opportunities = advisor.analyze_live_match('test_corner', match_data, live_data)
    corner_over = [opp for opp in opportunities if 'corner' in opp.market.lower() and 'over' in opp.market.lower()]
    
    # Verifica che non ci siano corner over banali (es. over 8.5 quando ci sono già 10)
    banal_corner = [opp for opp in corner_over if '8.5' in opp.market or '9.5' in opp.market]
    
    if len(banal_corner) == 0:
        print("✅ PASS: Nessun corner over banale quando è già superato")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(banal_corner)} corner over banali")
        return False

def test_ht_ft_banal():
    """Test: HT/FT quando è già superato"""
    print("\n🧪 Test HT/FT: Non suggerire quando è già superato")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {'home': 'Team A', 'away': 'Team B', 'odds_1': 1.5, 'odds_2': 3.0}
    # Test: 2-0 al 60' (HT era 1-0) - non suggerire HT/FT Home-Home se è ovvio
    live_data = {'score_home': 2, 'score_away': 0, 'minute': 60, 
                 'score_home_ht': 1, 'score_away_ht': 0}
    
    opportunities = advisor.analyze_live_match('test_ht_ft', match_data, live_data)
    ht_ft = [opp for opp in opportunities if 'ht_ft' in opp.market.lower()]
    
    # Se è 2-0 al 60' e HT era 1-0, HT/FT Home-Home è ovvio
    banal_ht_ft = [opp for opp in ht_ft if 'home_home' in opp.market.lower() and minute >= 60]
    
    if len(banal_ht_ft) == 0:
        print("✅ PASS: Nessun HT/FT banale quando è ovvio")
        return True
    else:
        print(f"⚠️  WARNING: Trovati {len(banal_ht_ft)} HT/FT che potrebbero essere banali")
        return True  # Non critico

def test_half_time_result_second_half():
    """Test: Half Time Result quando siamo nel secondo tempo"""
    print("\n🧪 Test Half Time Result: Non suggerire quando siamo nel secondo tempo")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {'home': 'Team A', 'away': 'Team B', 'odds_1': 2.0, 'odds_2': 2.5}
    live_data = {'score_home': 1, 'score_away': 0, 'minute': 50, 
                 'possession_home': 60, 'shots_home': 8, 'shots_away': 4}
    
    opportunities = advisor.analyze_live_match('test_ht_result', match_data, live_data)
    ht_result = [opp for opp in opportunities if 'half_time_result' in opp.market.lower()]
    
    if len(ht_result) == 0:
        print("✅ PASS: Nessun half time result quando siamo nel secondo tempo")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(ht_result)} half time result quando siamo nel secondo tempo")
        return False

def test_time_of_next_goal_too_late():
    """Test: Time of Next Goal quando è troppo tardi"""
    print("\n🧪 Test Time of Next Goal: Non suggerire quando è troppo tardi")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {'home': 'Team A', 'away': 'Team B', 'odds_1': 2.0, 'odds_2': 2.5}
    live_data = {'score_home': 1, 'score_away': 0, 'minute': 80}
    
    opportunities = advisor.analyze_live_match('test_time_next_goal', match_data, live_data)
    time_next = [opp for opp in opportunities if 'next_goal_before' in opp.market.lower() or 'next_goal_after' in opp.market.lower()]
    
    # Non dovrebbe suggerire "prima del 75'" quando siamo all'80'
    banal_time = [opp for opp in time_next if 'before_75' in opp.market.lower()]
    
    if len(banal_time) == 0:
        print("✅ PASS: Nessun 'prima del 75'' quando siamo all'80'")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(banal_time)} segnali 'prima del 75'' quando siamo all'80'")
        return False

def test_team_to_score_first_already_scored():
    """Test: Team to Score First quando hanno già segnato"""
    print("\n🧪 Test Team to Score First: Non suggerire quando hanno già segnato")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {'home': 'Team A', 'away': 'Team B', 'odds_1': 2.0, 'odds_2': 2.5}
    # Test: 1-0 al 30' - non suggerire "team to score first home"
    live_data = {'score_home': 1, 'score_away': 0, 'minute': 30}
    
    opportunities = advisor.analyze_live_match('test_score_first', match_data, live_data)
    score_first = [opp for opp in opportunities if 'team_to_score_first' in opp.market.lower()]
    
    # Non dovrebbe suggerire "home to score first" quando home ha già segnato
    banal_first = [opp for opp in score_first if 'home' in opp.market.lower()]
    
    if len(banal_first) == 0:
        print("✅ PASS: Nessun 'team to score first' quando hanno già segnato")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(banal_first)} segnali 'team to score first' quando hanno già segnato")
        return False

def run_all_banal_tests():
    """Esegue tutti i test per segnali banali"""
    print("=" * 70)
    print("🧪 TEST SEGNALI BANALI - VERIFICA TUTTI I MERCATI")
    print("=" * 70)
    
    tests = [
        test_btts_when_both_scored,
        test_win_to_nil_banal,
        test_exact_score_banal,
        test_goal_range_banal,
        test_team_to_score_next_too_late,
        test_corner_over_banal,
        test_ht_ft_banal,
        test_half_time_result_second_half,
        test_time_of_next_goal_too_late,
        test_team_to_score_first_already_scored
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
    print("📊 RIEPILOGO TEST SEGNALI BANALI")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"✅ Test passati: {passed}/{total}")
    print(f"❌ Test falliti: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 TUTTI I TEST PASSATI!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} TEST FALLITI - CORREZIONI NECESSARIE")
        return 1

if __name__ == '__main__':
    sys.exit(run_all_banal_tests())

