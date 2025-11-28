#!/usr/bin/env python3
"""Script per verificare attività IA e aggiornamento partite"""
import os
import sqlite3
from datetime import datetime
import re

print("=" * 70)
print("📊 VERIFICA ATTIVITÀ IA - AGGIORNAMENTO PARTITE")
print("=" * 70)

# 1. Verifica log file
log_path = "logs/automation_24h.log"
if os.path.exists(log_path):
    print(f"\n✅ Log file trovato: {log_path}")
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        recent_lines = lines[-500:] if len(lines) > 500 else lines
        
        # Cerca pattern importanti
        patterns = {
            "Tracciate.*partite": [],
            "Aggiornati.*risultati": [],
            "Partita finita": [],
            "Recuperati.*eventi": [],
            "Recuperate statistiche": [],
            "Aggiornati.*segnali": [],
            "update_results": [],
            "_fetch_match_result": []
        }
        
        for i, line in enumerate(recent_lines):
            for pattern in patterns.keys():
                if re.search(pattern, line, re.IGNORECASE):
                    patterns[pattern].append((i, line.strip()))
        
        print(f"\n📋 Analisi ultime {len(recent_lines)} righe:")
        for pattern, matches in patterns.items():
            if matches:
                print(f"   ✅ {pattern}: {len(matches)} occorrenze")
                # Mostra ultime 3 occorrenze
                for idx, match_line in matches[-3:]:
                    timestamp = match_line[:19] if len(match_line) > 19 else "N/A"
                    print(f"      • {timestamp}: {match_line[20:80] if len(match_line) > 20 else match_line}")
        
        # Ultime righe rilevanti
        print(f"\n📝 Ultime attività rilevanti:")
        relevant_lines = []
        for line in reversed(recent_lines[-50:]):
            if any(p in line.lower() for p in ["tracciate", "aggiornati", "risultato", "eventi", "statistiche", "partita finita", "recuperat"]):
                relevant_lines.append(line.strip())
                if len(relevant_lines) >= 10:
                    break
        
        for line in relevant_lines[:10]:
            print(f"   {line[:100]}")
else:
    print(f"\n⚠️  Log file non trovato: {log_path}")

# 2. Verifica database segnali
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
    print(f"   Totale: {total}")
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
            if correct == 1:
                correct_str = "WIN"
            elif correct == 0:
                correct_str = "LOSS"
            else:
                correct_str = "PENDING"
            print(f"   {approved_str} {match_id}/{market} - {correct_str} - {ts}")
    
    # Partite con risultati
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
    print(f"\n⚠️  Database non trovato (nessun segnale ancora registrato)")

print("\n" + "=" * 70)

