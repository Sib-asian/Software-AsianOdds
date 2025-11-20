"""
Test approfondito per Opzione 4: Deduplicazione intelligente mercati
Verifica:
1. Blocco stesso mercato per 30 minuti
2. Penalizzazione -30% per mercati già usati
3. Bonus +20% per mercati alternativi
4. Tracking mercati nella history
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Simula le classi necessarie
class MockLiveOpportunity:
    def __init__(self, market: str, confidence: float, ev: float, has_live_stats: bool = True):
        self.market = market
        self.confidence = confidence
        self.ev = ev
        self.has_live_stats = has_live_stats
        self.match_stats = {'minute': 36, 'score_home': 0, 'score_away': 2}
        self.match_data = {'home': 'Team A', 'away': 'Team B', 'league': 'Test League'}

def test_opzione4_logic():
    """Test della logica dell'opzione 4"""
    print("=" * 80)
    print("🧪 TEST OPZIONE 4: DEDUPLICAZIONE INTELLIGENTE MERCATI")
    print("=" * 80)
    
    # Simula match_markets_history
    match_markets_history: Dict[str, List[Dict[str, Any]]] = {}
    match_id = "test_match_123"
    
    # Mercati alternativi (solo il primo tipo, es. "over" per "over_2.5", "btts" per "btts_yes")
    alternative_market_types = {'over', 'under', 'btts', 'clean', 'exact', 'goal', 'odd', 'ht'}
    
    def calculate_score_with_modifiers(market: str, base_score: float, match_id: str, now: datetime) -> tuple:
        """Calcola score con penalizzazioni e bonus"""
        # Estrai tipo di mercato come nel codice reale (automation_24h.py linea 1239-1240)
        # Es. "over_2.5" -> "over", "clean_sheet" -> "clean_sheet", "btts_yes" -> "btts"
        market_type = market.split('_')[0] if '_' in market else market
        
        score_modifier = 1.0
        modifier_reason = ""
        
        if match_id in match_markets_history:
            # Verifica se questo mercato è stato già suggerito
            market_already_used = False
            for market_entry in match_markets_history[match_id]:
                if market_entry['market'] == market:
                    market_already_used = True
                    break
            
            if market_already_used:
                score_modifier *= 0.7  # Penalizzazione -30%
                modifier_reason = " (penalizzato -30%: mercato già suggerito)"
            elif market_type in alternative_market_types:
                score_modifier *= 1.2  # Bonus +20%
                modifier_reason = " (bonus +20%: mercato alternativo)"
        
        final_score = base_score * score_modifier
        return final_score, modifier_reason
    
    def check_market_blocked(market: str, match_id: str, now: datetime, block_minutes: int = 30) -> bool:
        """Verifica se un mercato è bloccato"""
        if match_id not in match_markets_history:
            return False
        
        for market_entry in match_markets_history[match_id]:
            if market_entry['market'] == market:
                time_diff = (now - market_entry['timestamp']).total_seconds() / 60
                if time_diff < block_minutes:
                    return True
        return False
    
    def add_market_to_history(match_id: str, market: str, timestamp: datetime):
        """Aggiunge mercato alla history"""
        if match_id not in match_markets_history:
            match_markets_history[match_id] = []
        match_markets_history[match_id].append({
            'market': market,
            'timestamp': timestamp
        })
        # Pulisci entry vecchie
        match_markets_history[match_id] = [
            entry for entry in match_markets_history[match_id]
            if (datetime.now() - entry['timestamp']).total_seconds() / 60 < 60
        ]
    
    # TEST 1: Ciclo 1 - Primo mercato
    print("\n📋 TEST 1: Ciclo 1 - Primo mercato (next_goal_pressure_away)")
    print("-" * 80)
    now = datetime.now()
    market1 = "next_goal_pressure_away"
    base_score1 = 1.2
    
    # Verifica blocco
    is_blocked = check_market_blocked(market1, match_id, now)
    print(f"   Mercato bloccato? {is_blocked} (atteso: False)")
    assert not is_blocked, "❌ ERRORE: Mercato non dovrebbe essere bloccato al primo ciclo"
    
    # Calcola score
    score1, reason1 = calculate_score_with_modifiers(market1, base_score1, match_id, now)
    print(f"   Score base: {base_score1:.3f}")
    print(f"   Score finale: {score1:.3f}{reason1}")
    assert score1 == base_score1, f"❌ ERRORE: Score dovrebbe essere {base_score1}, ottenuto {score1}"
    
    # Aggiungi alla history
    add_market_to_history(match_id, market1, now)
    print(f"   ✅ Mercato aggiunto alla history")
    print(f"   ✅ TEST 1 PASSATO")
    
    # TEST 2: Ciclo 2 - Stesso mercato dopo 5 minuti (dovrebbe essere bloccato)
    print("\n📋 TEST 2: Ciclo 2 - Stesso mercato dopo 5 minuti (dovrebbe essere BLOCCATO)")
    print("-" * 80)
    now2 = now + timedelta(minutes=5)
    is_blocked2 = check_market_blocked(market1, match_id, now2)
    print(f"   Mercato bloccato? {is_blocked2} (atteso: True)")
    assert is_blocked2, "❌ ERRORE: Mercato dovrebbe essere bloccato dopo 5 minuti"
    print(f"   ✅ TEST 2 PASSATO")
    
    # TEST 3: Ciclo 2 - Mercato alternativo (dovrebbe avere bonus)
    print("\n📋 TEST 3: Ciclo 2 - Mercato alternativo (over_2.5) con bonus +20%")
    print("-" * 80)
    market2 = "over_2.5"
    base_score2 = 1.1
    score2, reason2 = calculate_score_with_modifiers(market2, base_score2, match_id, now2)
    print(f"   Score base: {base_score2:.3f}")
    print(f"   Score finale: {score2:.3f}{reason2}")
    expected_score2 = base_score2 * 1.2
    assert abs(score2 - expected_score2) < 0.001, f"❌ ERRORE: Score dovrebbe essere {expected_score2}, ottenuto {score2}"
    print(f"   ✅ TEST 3 PASSATO")
    
    # TEST 4: Ciclo 3 - Stesso mercato alternativo (dovrebbe essere penalizzato)
    print("\n📋 TEST 4: Ciclo 3 - Stesso mercato alternativo già usato (penalizzazione -30%)")
    print("-" * 80)
    # Aggiungi over_2.5 alla history
    add_market_to_history(match_id, market2, now2)
    now3 = now2 + timedelta(minutes=5)
    score3, reason3 = calculate_score_with_modifiers(market2, base_score2, match_id, now3)
    print(f"   Score base: {base_score2:.3f}")
    print(f"   Score finale: {score3:.3f}{reason3}")
    expected_score3 = base_score2 * 0.7
    assert abs(score3 - expected_score3) < 0.001, f"❌ ERRORE: Score dovrebbe essere {expected_score3}, ottenuto {score3}"
    print(f"   ✅ TEST 4 PASSATO")
    
    # TEST 5: Ciclo 4 - Mercato originale dopo 30 minuti (dovrebbe essere sbloccato ma penalizzato)
    print("\n📋 TEST 5: Ciclo 4 - Mercato originale dopo 30 minuti (sbloccato ma penalizzato)")
    print("-" * 80)
    now4 = now + timedelta(minutes=30)
    is_blocked4 = check_market_blocked(market1, match_id, now4)
    print(f"   Mercato bloccato? {is_blocked4} (atteso: False)")
    assert not is_blocked4, "❌ ERRORE: Mercato dovrebbe essere sbloccato dopo 30 minuti"
    
    score4, reason4 = calculate_score_with_modifiers(market1, base_score1, match_id, now4)
    print(f"   Score base: {base_score1:.3f}")
    print(f"   Score finale: {score4:.3f}{reason4}")
    expected_score4 = base_score1 * 0.7
    assert abs(score4 - expected_score4) < 0.001, f"❌ ERRORE: Score dovrebbe essere {expected_score4}, ottenuto {score4}"
    print(f"   ✅ TEST 5 PASSATO")
    
    # TEST 6: Verifica mercati alternativi (solo quelli NON ancora usati)
    print("\n📋 TEST 6: Verifica mercati alternativi NON usati ricevono bonus")
    print("-" * 80)
    # Pulisci history per questo test (tranne over_2.5 che è già stato usato)
    test_markets = ['under_1.5', 'btts_yes', 'clean_sheet_home', 'exact_score_2_0', 'goal_range_2_3', 'odd_even_odd', 'ht_ft_home_home']
    base_score = 1.0
    for test_market in test_markets:
        score, reason = calculate_score_with_modifiers(test_market, base_score, match_id, now4)
        # Estrai tipo mercato come nel codice reale
        market_type = test_market.split('_')[0] if '_' in test_market else test_market
        
        if market_type in alternative_market_types:
            expected = base_score * 1.2
            assert abs(score - expected) < 0.001, f"❌ ERRORE: {test_market} (tipo: {market_type}) dovrebbe avere bonus (atteso {expected}, ottenuto {score})"
            print(f"   ✅ {test_market} (tipo: {market_type}): {score:.3f} (bonus +20%)")
        else:
            print(f"   ⚠️  {test_market} (tipo: {market_type}): {score:.3f} (no bonus - tipo '{market_type}' non in {alternative_market_types})")
    
    # Verifica che over_2.5 (già usato) sia penalizzato
    score_over, reason_over = calculate_score_with_modifiers('over_2.5', base_score, match_id, now4)
    expected_penalized = base_score * 0.7
    assert abs(score_over - expected_penalized) < 0.001, f"❌ ERRORE: over_2.5 dovrebbe essere penalizzato"
    print(f"   ✅ over_2.5 (già usato): {score_over:.3f} (penalizzato -30%)")
    print(f"   ✅ TEST 6 PASSATO")
    
    # TEST 7: Verifica pulizia history
    print("\n📋 TEST 7: Verifica pulizia automatica history (> 60 minuti)")
    print("-" * 80)
    # Aggiungi un mercato con timestamp molto vecchio (70 minuti fa)
    old_timestamp = datetime.now() - timedelta(minutes=70)
    match_markets_history[match_id].append({
        'market': 'old_market',
        'timestamp': old_timestamp
    })
    initial_count = len(match_markets_history[match_id])
    print(f"   Entry nella history prima della pulizia: {initial_count}")
    print(f"   Timestamp più vecchio: {old_timestamp}")
    print(f"   Timestamp attuale: {datetime.now()}")
    print(f"   Differenza: {(datetime.now() - old_timestamp).total_seconds() / 60:.1f} minuti")
    
    # Simula pulizia (usa datetime.now() come nel codice reale)
    current_time = datetime.now()
    match_markets_history[match_id] = [
        entry for entry in match_markets_history[match_id]
        if (current_time - entry['timestamp']).total_seconds() / 60 < 60
    ]
    final_count = len(match_markets_history[match_id])
    print(f"   Entry nella history dopo la pulizia: {final_count}")
    assert final_count < initial_count, f"❌ ERRORE: La pulizia non ha funzionato (prima: {initial_count}, dopo: {final_count})"
    print(f"   ✅ TEST 7 PASSATO")
    
    print("\n" + "=" * 80)
    print("✅ TUTTI I TEST PASSATI CON SUCCESSO!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    try:
        test_opzione4_logic()
        print("\n🎉 Test completato con successo!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ ERRORE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRORE INATTESO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

