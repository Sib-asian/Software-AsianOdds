"""
Test per verificare le correzioni ai segnali live betting
"""
import sys
from datetime import datetime
from live_betting_advisor import LiveBettingAdvisor, LiveBettingOpportunity

def test_primo_tempo_non_invia_se_secondo_tempo():
    """Test: Non inviare segnali primo tempo se siamo già nel secondo tempo"""
    print("🧪 Test 1: Primo tempo - non inviare se siamo nel secondo tempo")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {
        'home': 'Team A',
        'away': 'Team B',
        'odds_1': 2.0,
        'odds_2': 2.5
    }
    
    # Test: siamo al 50' (secondo tempo) - NON dovrebbe generare segnali primo tempo
    live_data = {
        'score_home': 0,
        'score_away': 0,
        'minute': 50,  # Secondo tempo
        'shots_home': 5,
        'shots_away': 3,
        'shots_on_target_home': 2,
        'shots_on_target_away': 1
    }
    
    opportunities = advisor.analyze_live_match('test_match_1', match_data, live_data)
    
    # Verifica che non ci siano segnali per mercati primo tempo
    ht_markets = [opp for opp in opportunities if 'ht' in opp.market.lower() or 'primo tempo' in opp.market.lower()]
    
    if len(ht_markets) == 0:
        print("✅ PASS: Nessun segnale primo tempo quando siamo nel secondo tempo")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(ht_markets)} segnali primo tempo quando siamo al 50'")
        for opp in ht_markets:
            print(f"   - {opp.market} al minuto {live_data['minute']}'")
        return False

def test_secondo_tempo_calcola_gol_correttamente():
    """Test: Calcolo corretto dei gol del secondo tempo"""
    print("\n🧪 Test 2: Secondo tempo - calcolo gol corretti")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {
        'home': 'Team A',
        'away': 'Team B',
        'odds_1': 2.0,
        'odds_2': 2.5
    }
    
    # Test: siamo al 60' con 2-0 totale, ma 1 gol nel secondo tempo
    # NON dovrebbe suggerire "over 0.5 secondo tempo" perché c'è già 1 gol
    live_data = {
        'score_home': 2,
        'score_away': 0,
        'score_home_ht': 1,  # 1 gol al primo tempo
        'score_away_ht': 0,
        'minute': 60,
        'shots_home': 10,
        'shots_away': 5
    }
    
    opportunities = advisor.analyze_live_match('test_match_2', match_data, live_data)
    
    # Verifica che non ci siano segnali "over 0.5 secondo tempo" se c'è già 1 gol nel secondo tempo
    second_half_over = [opp for opp in opportunities if 'over_0.5_second_half' in opp.market.lower() or 
                       ('over' in opp.market.lower() and 'second' in opp.market.lower() and '0.5' in opp.market.lower())]
    
    if len(second_half_over) == 0:
        print("✅ PASS: Nessun segnale 'over 0.5 secondo tempo' quando c'è già 1 gol nel secondo tempo")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(second_half_over)} segnali 'over 0.5 secondo tempo' quando c'è già 1 gol")
        return False

def test_clean_sheet_confidence_minima():
    """Test: Clean sheet richiede confidence minima 80%"""
    print("\n🧪 Test 3: Clean sheet - confidence minima 80%")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {
        'home': 'Team A',
        'away': 'Team B',
        'odds_1': 1.5,  # Favorita
        'odds_2': 3.0
    }
    
    # Test: 1-0 al 60' con avversaria senza tiri in porta
    live_data = {
        'score_home': 1,
        'score_away': 0,
        'minute': 60,
        'shots_on_target_away': 0,  # Nessun tiro in porta
        'shots_away': 2,
        'dangerous_attacks_away': 5,
        'xg_away': 0.1
    }
    
    opportunities = advisor.analyze_live_match('test_match_3', match_data, live_data)
    
    clean_sheet_opps = [opp for opp in opportunities if 'clean_sheet' in opp.market.lower()]
    
    if len(clean_sheet_opps) == 0:
        print("⚠️  WARNING: Nessun segnale clean sheet generato (potrebbe essere filtrato)")
        return True  # Non è un errore se filtrato correttamente
    
    for opp in clean_sheet_opps:
        if opp.confidence >= 80:
            print(f"✅ PASS: Clean sheet con confidence {opp.confidence:.0f}% >= 80%")
            return True
        else:
            print(f"❌ FAIL: Clean sheet con confidence {opp.confidence:.0f}% < 80%")
            return False
    
    return True

def test_prossimo_gol_favorita_sfavorita():
    """Test: Non suggerire '2 della sfavorita' se la favorita sta vincendo"""
    print("\n🧪 Test 4: Prossimo gol - verifica favorita/sfavorita")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {
        'home': 'Team A',
        'away': 'Team B',
        'odds_1': 1.5,  # Home è favorita
        'odds_2': 3.0   # Away è sfavorita
    }
    
    # Test: Home (favorita) sta vincendo 1-0 al 30'
    # NON dovrebbe suggerire "prossimo gol away" (sfavorita) se la favorita sta vincendo
    live_data = {
        'score_home': 1,
        'score_away': 0,
        'minute': 30
    }
    
    opportunities = advisor.analyze_live_match('test_match_4', match_data, live_data)
    
    next_goal_away = [opp for opp in opportunities if 'next_goal_away' in opp.market.lower() and 
                     'sfavorita' in opp.recommendation.lower()]
    
    if len(next_goal_away) == 0:
        print("✅ PASS: Nessun segnale 'prossimo gol sfavorita' quando la favorita sta vincendo")
        return True
    else:
        print(f"❌ FAIL: Trovati {len(next_goal_away)} segnali 'prossimo gol sfavorita' quando la favorita sta vincendo")
        return False

def test_statistiche_estratte():
    """Test: Verifica che le statistiche estratte siano presenti"""
    print("\n🧪 Test 5: Statistiche estratte presenti")
    
    advisor = LiveBettingAdvisor()
    
    match_data = {
        'home': 'Team A',
        'away': 'Team B',
        'odds_1': 2.0,
        'odds_2': 2.5
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
        'shots_on_target_home': 4
    }
    
    opportunities = advisor.analyze_live_match('test_match_5', match_data, live_data)
    
    if len(opportunities) == 0:
        print("⚠️  WARNING: Nessuna opportunità generata per test statistiche")
        return True
    
    # Verifica che almeno un'opportunità abbia statistiche estratte
    opp_with_stats = [opp for opp in opportunities if hasattr(opp, 'key_stats') and opp.key_stats]
    
    if len(opp_with_stats) > 0:
        print(f"✅ PASS: {len(opp_with_stats)} opportunità con statistiche estratte")
        # Mostra esempio
        if opp_with_stats[0].key_stats:
            print(f"   Esempio statistiche: {list(opp_with_stats[0].key_stats.keys())[:3]}")
        return True
    else:
        print("❌ FAIL: Nessuna opportunità con statistiche estratte")
        return False

def test_filtri_aumentati():
    """Test: Verifica che i filtri siano aumentati"""
    print("\n🧪 Test 6: Filtri aumentati (confidence 75%, EV 10%)")
    
    advisor = LiveBettingAdvisor()
    
    # Verifica valori di default
    if advisor.min_confidence == 75.0:
        print("✅ PASS: min_confidence = 75% (aumentato da 72%)")
    else:
        print(f"❌ FAIL: min_confidence = {advisor.min_confidence}% (atteso 75%)")
        return False
    
    if advisor.min_ev == 10.0:
        print("✅ PASS: min_ev = 10% (aumentato da 8%)")
    else:
        print(f"❌ FAIL: min_ev = {advisor.min_ev}% (atteso 10%)")
        return False
    
    # Verifica soglie mercati
    if advisor.market_min_confidence.get('clean_sheet', 0) >= 80:
        print("✅ PASS: clean_sheet threshold >= 80%")
    else:
        print(f"❌ FAIL: clean_sheet threshold = {advisor.market_min_confidence.get('clean_sheet', 0)}%")
        return False
    
    return True

def test_traduzioni_italiane():
    """Test: Verifica traduzioni italiane"""
    print("\n🧪 Test 7: Traduzioni italiane")
    
    advisor = LiveBettingAdvisor()
    
    # Test alcune traduzioni chiave
    test_markets = {
        'over_0.5_ht': 'Over 0.5 Primo Tempo',
        'over_0.5_second_half': 'Over 0.5 Secondo Tempo',
        'next_goal_home': 'Prossimo gol: Casa',
        'clean_sheet_home': 'Porta inviolata (Casa)'
    }
    
    all_ok = True
    for market, expected in test_markets.items():
        translated = advisor._translate_market_name(market)
        if expected.lower() in translated.lower() or translated.lower() in expected.lower():
            print(f"✅ PASS: {market} -> {translated}")
        else:
            print(f"❌ FAIL: {market} -> {translated} (atteso: {expected})")
            all_ok = False
    
    return all_ok

def run_all_tests():
    """Esegue tutti i test"""
    print("=" * 70)
    print("🧪 TEST SEGNALI LIVE BETTING - VERIFICA CORREZIONI")
    print("=" * 70)
    
    tests = [
        test_primo_tempo_non_invia_se_secondo_tempo,
        test_secondo_tempo_calcola_gol_correttamente,
        test_clean_sheet_confidence_minima,
        test_prossimo_gol_favorita_sfavorita,
        test_statistiche_estratte,
        test_filtri_aumentati,
        test_traduzioni_italiane
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ ERRORE in {test.__name__}: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO TEST")
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
    sys.exit(run_all_tests())

