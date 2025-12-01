#!/usr/bin/env python3
"""
Script diagnostico per capire perché su Render il DB rimane vuoto
"""
import os
import sys
import sqlite3
from pathlib import Path

print("=" * 60)
print("🔍 DIAGNOSI AUTOMATION SU RENDER")
print("=" * 60)

# 1. Verifica variabili ambiente
print("\n📊 VARIABILI AMBIENTE:")
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', 'NOT_SET')
telegram_chat = os.getenv('TELEGRAM_CHAT_ID', 'NOT_SET')
api_key = os.getenv('API_FOOTBALL_KEY', 'NOT_SET')

print(f"   TELEGRAM_BOT_TOKEN: {'✅ SET' if telegram_token != 'NOT_SET' else '❌ MISSING'}")
print(f"   TELEGRAM_CHAT_ID: {'✅ SET' if telegram_chat != 'NOT_SET' else '❌ MISSING'}")
print(f"   API_FOOTBALL_KEY: {'✅ SET' if api_key != 'NOT_SET' else '❌ MISSING'}")

# 2. Verifica file database
print("\n📁 DATABASE FILES:")
db_files = [
    'signal_quality_learning.db',
    'betting_results.db',
    'api_cache.db',
    'bankroll.db'
]

for db_file in db_files:
    db_path = Path(db_file)
    if db_path.exists():
        size = db_path.stat().st_size
        print(f"   ✅ {db_file}: {size} bytes")
        
        # Conta record in signal_quality_learning.db
        if db_file == 'signal_quality_learning.db':
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM signal_records')
                count = cursor.fetchone()[0]
                conn.close()
                print(f"      → {count} signal records")
                
                if count == 0:
                    print("      ⚠️  PROBLEMA: 0 record nonostante il DB esista!")
            except Exception as e:
                print(f"      ❌ Errore lettura: {e}")
    else:
        print(f"   ❌ {db_file}: NON ESISTE")

# 3. Test import moduli
print("\n📦 IMPORT MODULI:")
try:
    from ai_system.signal_quality_learner import SignalQualityLearner
    print("   ✅ SignalQualityLearner importato")
except Exception as e:
    print(f"   ❌ SignalQualityLearner: {e}")
    
try:
    from automation_24h import Automation24H
    print("   ✅ Automation24H importato")
except Exception as e:
    print(f"   ❌ Automation24H: {e}")

# 4. Test inizializzazione
print("\n🧪 TEST INIZIALIZZAZIONE:")
try:
    from ai_system.signal_quality_learner import SignalQualityLearner
    learner = SignalQualityLearner()
    print("   ✅ SignalQualityLearner si inizializza")
    print(f"   ✅ DB path: {learner.db_path}")
    
    # Test write
    test_id = learner.record_signal(
        match_id='diagnose_test',
        market='test',
        minute=50,
        score_home=0,
        score_away=0,
        quality_score=75.0,
        context_score=75.0,
        data_quality_score=75.0,
        logic_score=75.0,
        timing_score=75.0,
        was_approved=True,
        block_reasons=[],
        confidence=70.0,
        ev=8.0
    )
    print(f"   ✅ Test write OK (ID: {test_id})")
    
    # Cleanup
    conn = sqlite3.connect(learner.db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM signal_records WHERE match_id = ?', ('diagnose_test',))
    conn.commit()
    conn.close()
    print("   ✅ Test cleanup OK")
    
except Exception as e:
    print(f"   ❌ ERRORE: {e}")
    import traceback
    traceback.print_exc()

# 5. Possibili cause
print("\n" + "=" * 60)
print("💡 POSSIBILI CAUSE SE DB VUOTO:")
print("=" * 60)
print("""
1. ❌ Processo non in loop (crash dopo inizializzazione)
   → Controlla log Render per "Running analysis cycle..."
   
2. ❌ API_FOOTBALL_KEY mancante/invalida
   → Nessuna partita trovata = nessun segnale
   
3. ❌ Filtri troppo strict
   → Opportunità trovate ma tutte scartate
   
4. ❌ Database ephemeral (Render free tier)
   → DB si resetta ad ogni deploy
   
5. ❌ Permessi write su filesystem
   → DB readonly su Render

6. ❌ Working directory sbagliata
   → Scrive in path diverso da quello letto
""")

print("\n" + "=" * 60)
print("🔍 PROSSIMI PASSI:")
print("=" * 60)
print("""
1. Vai su Render Dashboard
2. Apri il worker "automation-24h"
3. Clicca "Logs"
4. Cerca queste stringhe:
   - "Running analysis cycle..." (deve esserci ogni 10 min)
   - "Segnale registrato nel database" (deve esserci quando trova opportunità)
   - "Found X total matches" (deve trovare partite)
   - Errori/Exception
   
5. Se NON vedi "Running analysis cycle...":
   → Il processo è crashato subito dopo l'init
   → Cerca l'errore nei log
   
6. Se vedi "Running analysis cycle..." ma non "Segnale registrato":
   → Non trova opportunità (normale se filtri strict)
   → O signal_quality_learner è None
""")

print("=" * 60)
