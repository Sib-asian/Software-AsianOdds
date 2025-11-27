#!/usr/bin/env python3
"""
Script per verificare database su Render
Controlla se ci sono segnali registrati e perché potrebbero mancare
"""

import os
import sys

print("🔍 Diagnostica Database Render")
print("=" * 60)

# 1. Verifica se il database esiste
db_path = "signal_quality_learning.db"
print(f"\n1. Verifica database: {db_path}")
if os.path.exists(db_path):
    print(f"   ✅ Database trovato: {os.path.abspath(db_path)}")
    size = os.path.getsize(db_path)
    print(f"   📊 Dimensione: {size} bytes")
else:
    print(f"   ❌ Database NON trovato in: {os.path.abspath(db_path)}")
    print(f"   📁 Directory corrente: {os.getcwd()}")
    print(f"   📁 File nella directory:")
    for f in os.listdir("."):
        if f.endswith(".db"):
            print(f"      - {f}")

# 2. Verifica database se esiste
if os.path.exists(db_path):
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Conta segnali
        cursor.execute("SELECT COUNT(*) FROM signal_records")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM signal_records WHERE was_correct IS NOT NULL")
        with_results = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM signal_records WHERE was_correct IS NULL")
        pending = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM signal_records WHERE was_approved = 1")
        approved = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM signal_records WHERE was_approved = 0")
        blocked = cursor.fetchone()[0]
        
        print(f"\n2. Statistiche segnali:")
        print(f"   Totale: {total}")
        print(f"   Con risultati: {with_results}")
        print(f"   In attesa: {pending}")
        print(f"   Approvati: {approved}")
        print(f"   Bloccati: {blocked}")
        
        # Ultimi 5 segnali
        if total > 0:
            cursor.execute("""
                SELECT match_id, market, minute, quality_score, was_approved, timestamp
                FROM signal_records
                ORDER BY timestamp DESC
                LIMIT 5
            """)
            rows = cursor.fetchall()
            print(f"\n3. Ultimi 5 segnali registrati:")
            for row in rows:
                match_id, market, minute, qs, approved, ts = row
                status = "✅ APPROVATO" if approved else "❌ BLOCCATO"
                print(f"   {ts} - {match_id}/{market} (min {minute}') - QS: {qs:.1f} - {status}")
        else:
            print(f"\n3. ⚠️  Nessun segnale registrato!")
            print(f"   Possibili cause:")
            print(f"   - Il sistema non sta trovando opportunità")
            print(f"   - La registrazione fallisce silenziosamente")
            print(f"   - Il signal_quality_learner non è inizializzato")
            print(f"   - I segnali vengono bloccati prima della registrazione")
        
        conn.close()
    except Exception as e:
        print(f"\n❌ Errore lettura database: {e}")

print("\n" + "=" * 60)
print("💡 Suggerimenti:")
print("   1. Controlla i log su Render per 'Segnale registrato'")
print("   2. Verifica che il sistema trovi opportunità (log 'Running analysis cycle')")
print("   3. Controlla se ci sono errori nella registrazione")
print("   4. Verifica che signal_quality_learner sia inizializzato")

