#!/usr/bin/env python3
"""Verifica log in tempo reale"""

from datetime import datetime
import os
import time

log_file = 'logs/automation_service_20251118.log'

if os.path.exists(log_file):
    # Verifica ultima modifica
    mtime = os.path.getmtime(log_file)
    now = time.time()
    age_seconds = now - mtime
    age_minutes = int(age_seconds // 60)
    age_secs = int(age_seconds % 60)
    
    # Leggi ultime righe
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        last_lines = lines[-10:] if len(lines) >= 10 else lines
    
    print("=" * 80)
    print("📋 VERIFICA LOG IN TEMPO REALE")
    print("=" * 80)
    print()
    print(f"📄 File log: automation_service_20251118.log")
    print(f"🕐 Ora attuale: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📏 Totale righe: {len(lines)}")
    print()
    print(f"⏰ Ultimo aggiornamento: {age_minutes} minuti e {age_secs} secondi fa")
    print()
    
    if age_seconds > 300:  # Più di 5 minuti
        print("⚠️  ATTENZIONE: Il sistema NON sta scrivendo log da più di 5 minuti!")
        print("   Il sistema potrebbe essere fermo o in errore.")
    else:
        print("✅ Sistema attivo (log aggiornato di recente)")
    
    print()
    print("=" * 80)
    print("📋 ULTIME 10 RIGHE DEL LOG:")
    print("=" * 80)
    print()
    for line in last_lines:
        print(line.rstrip())
    print()
    print("=" * 80)
else:
    print("❌ File log non trovato!")







