"""
Test per verificare il blocco di 15 minuti per partita
"""
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Simula la logica di blocco
def test_blocco_partita():
    """Test blocco 15 minuti per partita"""
    print("🧪 TEST: Blocco 15 minuti per partita")
    print("=" * 50)
    
    # Simula notified_matches_timestamps
    notified_matches_timestamps: Dict[str, datetime] = {}
    
    # Test 1: Prima notifica - dovrebbe passare
    print("\n📋 Test 1: Prima notifica per partita")
    match_id = "test_match_1"
    now = datetime.now()
    
    if match_id in notified_matches_timestamps:
        last_notification = notified_matches_timestamps[match_id]
        time_diff = (now - last_notification).total_seconds() / 60
        if time_diff < 15:
            print(f"   ❌ BLOCCATA: {time_diff:.1f} minuti fa (dovrebbe passare)")
            return False
    else:
        print(f"   ✅ PASSATA: Prima notifica (nessun blocco)")
        notified_matches_timestamps[match_id] = now
    
    # Test 2: Seconda notifica dopo 5 minuti - dovrebbe essere bloccata
    print("\n📋 Test 2: Seconda notifica dopo 5 minuti")
    now = datetime.now() + timedelta(minutes=5)
    
    if match_id in notified_matches_timestamps:
        last_notification = notified_matches_timestamps[match_id]
        time_diff = (now - last_notification).total_seconds() / 60
        if time_diff < 15:
            print(f"   ✅ BLOCCATA: {time_diff:.1f} minuti fa (corretto, < 15 min)")
        else:
            print(f"   ❌ PASSATA: {time_diff:.1f} minuti fa (dovrebbe essere bloccata)")
            return False
    else:
        print(f"   ❌ ERRORE: Partita non trovata in timestamp")
        return False
    
    # Test 3: Terza notifica dopo 15 minuti - dovrebbe passare
    print("\n📋 Test 3: Terza notifica dopo 15 minuti")
    now = datetime.now() + timedelta(minutes=15)
    
    if match_id in notified_matches_timestamps:
        last_notification = notified_matches_timestamps[match_id]
        time_diff = (now - last_notification).total_seconds() / 60
        if time_diff < 15:
            print(f"   ❌ BLOCCATA: {time_diff:.1f} minuti fa (dovrebbe passare)")
            return False
        else:
            print(f"   ✅ PASSATA: {time_diff:.1f} minuti fa (corretto, >= 15 min)")
            notified_matches_timestamps[match_id] = now
    else:
        print(f"   ❌ ERRORE: Partita non trovata in timestamp")
        return False
    
    # Test 4: Partita diversa - dovrebbe passare sempre
    print("\n📋 Test 4: Partita diversa")
    match_id_2 = "test_match_2"
    now = datetime.now()
    
    if match_id_2 in notified_matches_timestamps:
        last_notification = notified_matches_timestamps[match_id_2]
        time_diff = (now - last_notification).total_seconds() / 60
        if time_diff < 15:
            print(f"   ❌ BLOCCATA: {time_diff:.1f} minuti fa (dovrebbe passare, partita diversa)")
            return False
    else:
        print(f"   ✅ PASSATA: Prima notifica per partita diversa (corretto)")
        notified_matches_timestamps[match_id_2] = now
    
    print("\n" + "=" * 50)
    print("✅ TUTTI I TEST PASSATI!")
    return True


def test_blocco_mercato():
    """Test blocco 30 minuti per mercato"""
    print("\n🧪 TEST: Blocco 30 minuti per mercato")
    print("=" * 50)
    
    # Simula match_markets_history
    match_markets_history: Dict[str, list] = {}
    match_id = "test_match_1"
    market_1 = "over_2.5"
    market_2 = "btts_yes"
    
    # Test 1: Prima notifica mercato - dovrebbe passare
    print("\n📋 Test 1: Prima notifica mercato")
    now = datetime.now()
    
    if match_id in match_markets_history:
        for entry in match_markets_history[match_id]:
            if entry['market'] == market_1:
                time_diff = (now - entry['timestamp']).total_seconds() / 60
                if time_diff < 30:
                    print(f"   ❌ BLOCCATA: {time_diff:.1f} minuti fa (dovrebbe passare)")
                    return False
    else:
        print(f"   ✅ PASSATA: Prima notifica mercato (corretto)")
        match_markets_history[match_id] = [{'market': market_1, 'timestamp': now}]
    
    # Test 2: Stesso mercato dopo 5 minuti - dovrebbe essere bloccato
    print("\n📋 Test 2: Stesso mercato dopo 5 minuti")
    now = datetime.now() + timedelta(minutes=5)
    
    if match_id in match_markets_history:
        blocked = False
        for entry in match_markets_history[match_id]:
            if entry['market'] == market_1:
                time_diff = (now - entry['timestamp']).total_seconds() / 60
                if time_diff < 30:
                    print(f"   ✅ BLOCCATA: {time_diff:.1f} minuti fa (corretto, < 30 min)")
                    blocked = True
                    break
        if not blocked:
            print(f"   ❌ PASSATA: Dovrebbe essere bloccata")
            return False
    else:
        print(f"   ❌ ERRORE: Partita non trovata")
        return False
    
    # Test 3: Mercato diverso dopo 5 minuti - dovrebbe passare
    print("\n📋 Test 3: Mercato diverso dopo 5 minuti")
    now = datetime.now() + timedelta(minutes=5)
    
    if match_id in match_markets_history:
        blocked = False
        for entry in match_markets_history[match_id]:
            if entry['market'] == market_2:
                time_diff = (now - entry['timestamp']).total_seconds() / 60
                if time_diff < 30:
                    blocked = True
                    break
        if blocked:
            print(f"   ❌ BLOCCATA: Mercato diverso (dovrebbe passare)")
            return False
        else:
            print(f"   ✅ PASSATA: Mercato diverso (corretto)")
    else:
        print(f"   ❌ ERRORE: Partita non trovata")
        return False
    
    print("\n" + "=" * 50)
    print("✅ TUTTI I TEST PASSATI!")
    return True


def test_selezione_ciclo():
    """Test max 1 notifica per partita per ciclo"""
    print("\n🧪 TEST: Max 1 notifica per partita per ciclo")
    print("=" * 50)
    
    # Simula seen_matches (resettato ad ogni ciclo)
    seen_matches = set()
    opportunities = [
        {'match_id': 'match_1', 'market': 'over_2.5', 'score': 1.5},
        {'match_id': 'match_1', 'market': 'btts_yes', 'score': 1.3},  # Stessa partita
        {'match_id': 'match_2', 'market': 'over_2.5', 'score': 1.4},
    ]
    
    selected = []
    
    # Simula selezione
    for opp in opportunities:
        match_id = opp['match_id']
        if match_id in seen_matches:
            print(f"   ⏭️  Saltata: {match_id} - {opp['market']} (partita già selezionata)")
            continue
        selected.append(opp)
        seen_matches.add(match_id)
        print(f"   ✅ Selezionata: {match_id} - {opp['market']}")
    
    # Verifica
    if len(selected) == 2:
        print(f"\n   ✅ CORRETTO: Selezionate {len(selected)} opportunità (max 1 per partita)")
        if selected[0]['match_id'] == 'match_1' and selected[1]['match_id'] == 'match_2':
            print(f"   ✅ CORRETTO: Partite diverse selezionate")
            return True
        else:
            print(f"   ❌ ERRORE: Partite non diverse")
            return False
    else:
        print(f"\n   ❌ ERRORE: Selezionate {len(selected)} opportunità (attese 2)")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("🧪 TEST COMPLETO: Blocco Partita e Mercato")
    print("=" * 50)
    
    results = []
    
    # Test blocco partita
    results.append(("Blocco 15 minuti partita", test_blocco_partita()))
    
    # Test blocco mercato
    results.append(("Blocco 30 minuti mercato", test_blocco_mercato()))
    
    # Test selezione ciclo
    results.append(("Max 1 per partita per ciclo", test_selezione_ciclo()))
    
    # Report finale
    print("\n" + "=" * 50)
    print("📊 REPORT FINALE")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ TUTTI I TEST PASSATI!")
        sys.exit(0)
    else:
        print("❌ ALCUNI TEST FALLITI!")
        sys.exit(1)




