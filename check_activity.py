#!/usr/bin/env python3
"""Script per verificare attività IA"""
import sqlite3
import os
from datetime import datetime

print("=" * 60)
print("📊 VERIFICA ATTIVITÀ IA")
print("=" * 60)

# 1. Verifica database segnali
db_path = "signal_quality_learning.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM signal_records")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM signal_records WHERE was_correct IS NOT NULL")
    with_results = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM signal_records WHERE was_correct IS NULL")
    pending = cursor.fetchone()[0]
    
    print(f"\n📈 Database Segnali:")
    print(f"   Totale segnali: {total}")
    print(f"   Con risultati: {with_results}")
    print(f"   In attesa: {pending}")
    
    if total > 0:
        cursor.execute("""
            SELECT match_id, market, was_approved, was_correct, timestamp 
            FROM signal_records 
            ORDER BY timestamp DESC 
            LIMIT 5
        """)
        recent = cursor.fetchall()
        print(f"\n📝 Ultimi 5 segnali:")
        for r in recent:
            match_id, market, approved, correct, ts = r
            approved_str = "✅" if approved else "❌"
            correct_str = f"WIN" if correct == 1 else ("LOSS" if correct == 0 else "PENDING")
            print(f"   {approved_str} {match_id}/{market} - {correct_str} - {ts}")
    
    # Verifica partite con risultati aggiornati
    cursor.execute("""
        SELECT DISTINCT match_id, COUNT(*) as count 
        FROM signal_records 
        WHERE was_correct IS NOT NULL 
        GROUP BY match_id
    """)
    matches_with_results = cursor.fetchall()
    if matches_with_results:
        print(f"\n⚽ Partite con risultati aggiornati: {len(matches_with_results)}")
        for match_id, count in matches_with_results[:5]:
            print(f"   {match_id}: {count} segnali valutati")
    
    conn.close()
else:
    print("\n⚠️  Database non trovato (nessun segnale ancora registrato)")

# 2. Verifica log file
log_path = "automation_24h.log"
if os.path.exists(log_path):
    print(f"\n📋 Analisi log file...")
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        recent_lines = lines[-100:] if len(lines) > 100 else lines
        
        # Cerca pattern importanti
        patterns = {
            "Risultato recuperato": 0,
            "Recuperati.*eventi": 0,
            "Recuperate.*statistiche": 0,
            "Aggiornati.*segnali": 0,
            "update_results": 0,
            "FINISHED": 0,
            "Progresso Apprendimento": 0,
            "Apprendimento completato": 0
        }
        
        import re
        for line in recent_lines:
            for pattern in patterns.keys():
                if re.search(pattern, line, re.IGNORECASE):
                    patterns[pattern] += 1
        
        print(f"   Ultime 100 righe analizzate")
        for pattern, count in patterns.items():
            if count > 0:
                print(f"   ✅ {pattern}: {count} occorrenze")
        
        # Ultime righe rilevanti
        print(f"\n📝 Ultime righe rilevanti:")
        for line in reversed(recent_lines[-20:]):
            if any(p in line.lower() for p in ["risultato", "eventi", "statistiche", "aggiorn", "apprendimento", "progresso"]):
                print(f"   {line.strip()[:100]}")
else:
    print(f"\n⚠️  Log file non trovato")

print("\n" + "=" * 60)


