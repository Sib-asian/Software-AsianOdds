#!/usr/bin/env python3
"""Verifica segnali registrati nel database"""

import sqlite3
import os
from datetime import datetime

db_path = "signal_quality_learning.db"

if not os.path.exists(db_path):
    print(f"❌ Database non trovato: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Verifica tabelle
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("📊 Tabelle nel database:")
for t in tables:
    print(f"   - {t[0]}")

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

print(f"\n📈 Statistiche segnali:")
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
    print(f"\n📝 Ultimi 5 segnali registrati:")
    for row in rows:
        match_id, market, minute, qs, approved, ts = row
        status = "✅ APPROVATO" if approved else "❌ BLOCCATO"
        print(f"   {ts} - {match_id}/{market} (min {minute}') - QS: {qs:.1f} - {status}")
else:
    print("\n⚠️  Nessun segnale registrato nel database!")

conn.close()

