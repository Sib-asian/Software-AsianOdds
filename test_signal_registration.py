#!/usr/bin/env python3
"""
Test per verificare che i segnali vengano registrati correttamente nel database.
"""
import sys
import os
from datetime import datetime

# Aggiungi il percorso del progetto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_signal_registration():
    print("=" * 70)
    print("🧪 TEST REGISTRAZIONE SEGNALI")
    print("=" * 70)
    
    # 1. Test inizializzazione SignalQualityLearner
    print("\n1️⃣  Test inizializzazione SignalQualityLearner...")
    try:
        from ai_system.signal_quality_learner import SignalQualityLearner
        learner = SignalQualityLearner()
        print("   ✅ SignalQualityLearner inizializzato correttamente")
    except Exception as e:
        print(f"   ❌ Errore inizializzazione SignalQualityLearner: {e}")
        return False
    
    # 2. Test inizializzazione SignalQualityGate con learner
    print("\n2️⃣  Test inizializzazione SignalQualityGate con learner...")
    try:
        from ai_system.signal_quality_scorer import SignalQualityGate
        signal_gate = SignalQualityGate(
            ai_pipeline=None,  # Non necessario per il test
            min_quality_score=75.0,
            learner=learner
        )
        print("   ✅ SignalQualityGate inizializzato con learner")
        
        # Verifica che il learner sia stato passato
        if signal_gate.learner is None:
            print("   ❌ Learner NON è stato passato correttamente!")
            return False
        else:
            print("   ✅ Learner è presente in SignalQualityGate")
    except Exception as e:
        print(f"   ❌ Errore inizializzazione SignalQualityGate: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Test registrazione segnale diretto
    print("\n3️⃣  Test registrazione segnale diretto nel database...")
    try:
        record_id = learner.record_signal(
            match_id="test_match_123",
            market="over_2.5",
            minute=65,
            score_home=1,
            score_away=0,
            quality_score=85.5,
            context_score=90.0,
            data_quality_score=80.0,
            logic_score=85.0,
            timing_score=87.0,
            was_approved=True,
            block_reasons=[],
            confidence=75.0,
            ev=25.0
        )
        print(f"   ✅ Segnale registrato con ID: {record_id}")
    except Exception as e:
        print(f"   ❌ Errore registrazione segnale: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Verifica che il segnale sia nel database
    print("\n4️⃣  Verifica segnale nel database...")
    try:
        import sqlite3
        conn = sqlite3.connect(learner.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_records WHERE match_id = ?", ("test_match_123",))
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            print(f"   ✅ Segnale trovato nel database (count: {count})")
        else:
            print("   ❌ Segnale NON trovato nel database!")
            return False
    except Exception as e:
        print(f"   ❌ Errore verifica database: {e}")
        return False
    
    # 5. Test registrazione tramite SignalQualityGate (simulazione)
    print("\n5️⃣  Test registrazione tramite SignalQualityGate (simulazione)...")
    try:
        # Crea un'opportunità di test
        test_opportunity = {
            'match_id': 'test_match_456',
            'home': 'Test Home',
            'away': 'Test Away',
            'league': 'Test League',
            'live_opportunity': type('obj', (object,), {
                'market': 'over_1.5',
                'confidence': 80.0,
                'ev': 30.0,
                'match_stats': {
                    'minute': 70,
                    'score_home': 1,
                    'score_away': 1,
                    'shots_home': 10,
                    'shots_away': 8
                }
            })()
        }
        
        test_match_data = {
            'home': 'Test Home',
            'away': 'Test Away',
            'league': 'Test League'
        }
        
        test_live_data = {
            'minute': 70,
            'score_home': 1,
            'score_away': 1,
            'shots_home': 10,
            'shots_away': 8
        }
        
        # Chiama should_send_signal (dovrebbe registrare il segnale)
        should_send, quality_score = signal_gate.should_send_signal(
            opportunity=test_opportunity,
            match_data=test_match_data,
            live_data=test_live_data
        )
        
        print(f"   ✅ SignalQualityGate.should_send_signal eseguito")
        print(f"      - Should send: {should_send}")
        print(f"      - Quality Score: {quality_score.total_score:.1f}/100")
        print(f"      - Is Approved: {quality_score.is_approved}")
        
    except Exception as e:
        print(f"   ❌ Errore test SignalQualityGate: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Verifica che anche questo segnale sia nel database
    print("\n6️⃣  Verifica segnale test_match_456 nel database...")
    try:
        import sqlite3
        conn = sqlite3.connect(learner.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_records WHERE match_id = ?", ("test_match_456",))
        count = cursor.fetchone()[0]
        
        if count > 0:
            cursor.execute("""
                SELECT match_id, market, was_approved, quality_score 
                FROM signal_records 
                WHERE match_id = ?
            """, ("test_match_456",))
            signal = cursor.fetchone()
            print(f"   ✅ Segnale trovato nel database:")
            print(f"      - Match ID: {signal[0]}")
            print(f"      - Market: {signal[1]}")
            print(f"      - Approved: {signal[2]}")
            print(f"      - Quality Score: {signal[3]:.1f}")
        else:
            print("   ❌ Segnale test_match_456 NON trovato nel database!")
            return False
        conn.close()
    except Exception as e:
        print(f"   ❌ Errore verifica database: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. Verifica totale segnali nel database
    print("\n7️⃣  Verifica totale segnali nel database...")
    try:
        import sqlite3
        conn = sqlite3.connect(learner.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_records")
        total = cursor.fetchone()[0]
        conn.close()
        print(f"   ✅ Totale segnali nel database: {total}")
    except Exception as e:
        print(f"   ❌ Errore conteggio: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_signal_registration()
    sys.exit(0 if success else 1)

