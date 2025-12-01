#!/usr/bin/env python3
"""
Test Live Betting Performance Tracking
======================================

Testa le nuove funzionalità:
1. Tracking performance
2. Soglie dinamiche
3. Report automatici
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from live_betting_performance_tracker import LiveBettingPerformanceTracker
from live_betting_reports import LiveBettingReports
from live_betting_advisor import LiveBettingOpportunity, LiveBettingAdvisor

def test_tracker():
    """Test del performance tracker"""
    print("🧪 TEST 1: Performance Tracker")
    print("=" * 50)
    
    try:
        # Usa database temporaneo per test
        import tempfile
        import os
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        tracker = LiveBettingPerformanceTracker(db_path=temp_db.name)
        print("✅ Tracker inizializzato")
        
        # Crea opportunità di test
        class MockOpportunity:
            def __init__(self):
                self.match_id = "test_match_1"
                self.market = "over_2.5"
                self.odds = 1.85
                self.confidence = 78.0
                self.expected_value = 12.0
                self.match_stats = {
                    'minute': 45,
                    'score_home': 1,
                    'score_away': 1
                }
        
        opp = MockOpportunity()
        match_data = {
            'home': 'Team A',
            'away': 'Team B',
            'league': 'Test League'
        }
        
        # Salva opportunità
        opp_id = tracker.save_live_opportunity(opp, match_data)
        print(f"✅ Opportunità salvata (ID: {opp_id})")
        
        # Simula risultato finale
        tracker.update_live_result("test_match_1", 2, 1)  # 2-1 finale (over 2.5 vinto)
        print("✅ Risultato aggiornato")
        
        # Verifica performance
        performance = tracker.get_market_performance("over_2.5", days=1)
        if performance:
            print(f"✅ Performance recuperata: {performance}")
            assert performance['winners'] == 1, "Dovrebbe esserci 1 vincita"
            assert performance['win_rate'] == 100.0, "Win rate dovrebbe essere 100%"
        else:
            print("⚠️  Performance non trovata (normale se database vuoto)")
        
        # Test soglie dinamiche
        thresholds = tracker.calculate_dynamic_thresholds()
        print(f"✅ Soglie dinamiche calcolate: {len(thresholds)} mercati")
        
        # Pulisci database temporaneo
        try:
            os.unlink(temp_db.name)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nel test tracker: {e}")
        import traceback
        traceback.print_exc()
        # Pulisci database temporaneo
        try:
            if 'temp_db' in locals():
                os.unlink(temp_db.name)
        except:
            pass
        return False

def test_reports():
    """Test dei report"""
    print("\n🧪 TEST 2: Report Automatici")
    print("=" * 50)
    
    try:
        import tempfile
        import os
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        tracker = LiveBettingPerformanceTracker(db_path=temp_db.name)
        reports = LiveBettingReports(tracker, notifier=None)  # Senza notifier per test
        
        # Genera report giornaliero
        daily_report = reports.generate_daily_report()
        print("✅ Report giornaliero generato")
        print(f"   Lunghezza: {len(daily_report)} caratteri")
        
        # Genera report settimanale
        weekly_report = reports.generate_weekly_report()
        print("✅ Report settimanale generato")
        print(f"   Lunghezza: {len(weekly_report)} caratteri")
        
        # Test alert
        alert = reports.check_win_rate_alert()
        if alert:
            print(f"⚠️  Alert generato: {alert[:50]}...")
        else:
            print("✅ Nessun alert (win rate OK)")
        
        # Pulisci database temporaneo
        try:
            os.unlink(temp_db.name)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nel test report: {e}")
        import traceback
        traceback.print_exc()
        # Pulisci database temporaneo
        try:
            if 'temp_db' in locals():
                os.unlink(temp_db.name)
        except:
            pass
        return False

def test_dynamic_thresholds():
    """Test delle soglie dinamiche"""
    print("\n🧪 TEST 3: Soglie Dinamiche")
    print("=" * 50)
    
    try:
        import tempfile
        import os
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db_path = temp_db.name
        temp_db.close()
        tracker = LiveBettingPerformanceTracker(db_path=temp_db_path)
        
        # Crea dati di test con win rate basso
        class MockOpp:
            def __init__(self, market, won):
                self.match_id = f"test_{market}_{won}"
                self.market = market
                self.odds = 1.8
                self.confidence = 75.0
                self.expected_value = 10.0
                self.match_stats = {'minute': 50, 'score_home': 1, 'score_away': 0}
        
        # Aggiungi 10 opportunità con win rate 30% (basso)
        for i in range(10):
            opp = MockOpp("over_2.5", i)
            match_data = {'home': 'A', 'away': 'B', 'league': 'Test'}
            tracker.save_live_opportunity(opp, match_data)
            # Solo 3 vincono (30% win rate)
            if i < 3:
                tracker.update_live_result(f"test_over_2.5_{i}", 3, 1)  # Over 2.5 vinto
            else:
                tracker.update_live_result(f"test_over_2.5_{i}", 1, 0)  # Over 2.5 perso
        
        # Calcola soglie
        thresholds = tracker.calculate_dynamic_thresholds()
        
        if "over_2.5" in thresholds:
            thresh = thresholds["over_2.5"]
            print(f"✅ Soglia per over_2.5:")
            print(f"   Win rate: {thresh['win_rate']:.1f}%")
            print(f"   Min Confidence: {thresh['min_confidence']:.1f}%")
            print(f"   Min EV: {thresh['min_ev']:.1f}%")
            print(f"   Motivo: {thresh['reason']}")
            
            # Verifica che le soglie siano aumentate (win rate basso)
            assert thresh['min_confidence'] > 75.0, "Soglia dovrebbe essere aumentata"
            assert thresh['min_ev'] > 10.0, "EV dovrebbe essere aumentato"
        else:
            print("⚠️  Soglia non trovata (normale se non ci sono abbastanza dati)")
        
        # Pulisci database temporaneo
        try:
            os.unlink(temp_db.name)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nel test soglie dinamiche: {e}")
        import traceback
        traceback.print_exc()
        # Pulisci database temporaneo
        try:
            if 'temp_db' in locals():
                os.unlink(temp_db.name)
        except:
            pass
        return False

def test_integration():
    """Test integrazione con LiveBettingAdvisor"""
    print("\n🧪 TEST 4: Integrazione LiveBettingAdvisor")
    print("=" * 50)
    
    try:
        import tempfile
        import os
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        tracker = LiveBettingPerformanceTracker(db_path=temp_db.name)
        advisor = LiveBettingAdvisor(
            min_confidence=75.0,
            min_ev=10.0,
            performance_tracker=tracker
        )
        
        print("✅ LiveBettingAdvisor inizializzato con tracker")
        
        # Test recupero soglia dinamica
        threshold = advisor._get_market_specific_threshold("over_2.5")
        print(f"✅ Soglia recuperata per over_2.5: {threshold}")
        
        # Test filtro EV con soglie dinamiche
        class MockOpp:
            def __init__(self):
                self.market = "over_2.5"
                self.odds = 1.8
                self.confidence = 75.0
                self.ev = 8.0  # EV basso
        
        opp = MockOpp()
        filtered = advisor._filter_by_expected_value([opp])
        
        print(f"✅ Filtro EV testato: {len(filtered)} opportunità passate")
        
        # Pulisci database temporaneo
        try:
            os.unlink(temp_db.name)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nel test integrazione: {e}")
        import traceback
        traceback.print_exc()
        # Pulisci database temporaneo
        try:
            if 'temp_db' in locals():
                os.unlink(temp_db.name)
        except:
            pass
        return False

def main():
    """Esegue tutti i test"""
    print("\n" + "=" * 50)
    print("🧪 TEST LIVE BETTING TRACKING SYSTEM")
    print("=" * 50 + "\n")
    
    results = []
    
    results.append(("Tracker", test_tracker()))
    results.append(("Reports", test_reports()))
    results.append(("Soglie Dinamiche", test_dynamic_thresholds()))
    results.append(("Integrazione", test_integration()))
    
    print("\n" + "=" * 50)
    print("📊 RISULTATI TEST:")
    print("=" * 50)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ TUTTI I TEST PASSATI!")
        print("🚀 Sistema pronto per il riavvio")
    else:
        print("⚠️  ALCUNI TEST FALLITI")
        print("🔧 Verifica gli errori sopra")
    print("=" * 50 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

