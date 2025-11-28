#!/usr/bin/env python3
import sqlite3
import os

db_path = "signal_quality_learning.db"

if not os.path.exists(db_path):
    print("❌ Database non trovato!")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Verifica tabelle
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"📊 Tabelle nel database: {tables}")

# Conta segnali
cursor.execute("SELECT COUNT(*) FROM signal_records")
total = cursor.fetchone()[0]
print(f"📈 Totale segnali: {total}")

# Ultimi 5 segnali
cursor.execute("""
    SELECT match_id, market, was_approved, quality_score, timestamp 
    FROM signal_records 
    ORDER BY timestamp DESC 
    LIMIT 5
""")
signals = cursor.fetchall()

if signals:
    print(f"\n📋 Ultimi {len(signals)} segnali:")
    for s in signals:
        match_id, market, was_approved, quality_score, timestamp = s
        status = "✅ APPROVATO" if was_approved else "❌ BLOCCATO"
        print(f"   - {match_id} | {market} | {status} | Score: {quality_score:.1f} | {timestamp}")
else:
    print("\n⚠️  Nessun segnale nel database")

conn.close()

