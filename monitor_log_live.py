#!/usr/bin/env python3
"""Monitor log in tempo reale"""

import time
from datetime import datetime
from pathlib import Path

log_file = Path('logs/automation_service_20251118.log')

print("=" * 80)
print("📊 MONITOR LOG IN TEMPO REALE")
print("=" * 80)
print()
print(f"🕐 Ora attuale: {datetime.now().strftime('%H:%M:%S')}")
print(f"📄 File: {log_file.name}")
print()
print("Monitoraggio attivo... (Ctrl+C per fermare)")
print()

if not log_file.exists():
    print("❌ File log non trovato!")
    exit(1)

# Leggi dimensione iniziale
size_before = log_file.stat().st_size
print(f"📏 Dimensione iniziale: {size_before} bytes")
print()

try:
    while True:
        time.sleep(5)
        
        if log_file.exists():
            size_after = log_file.stat().st_size
            mtime = log_file.stat().st_mtime
            age = time.time() - mtime
            
            if size_after > size_before:
                print(f"✅ {datetime.now().strftime('%H:%M:%S')} - Sistema sta scrivendo! (+{size_after - size_before} bytes)")
                size_before = size_after
                
                # Leggi ultime righe
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"   Ultima riga: {lines[-1].rstrip()[:100]}")
                        print()
            else:
                if age > 60:
                    print(f"⚠️  {datetime.now().strftime('%H:%M:%S')} - Nessun aggiornamento da {int(age)} secondi")
                else:
                    print(f"⏳ {datetime.now().strftime('%H:%M:%S')} - In attesa...")
        else:
            print("❌ File log non trovato!")
            break
            
except KeyboardInterrupt:
    print("\n\n🛑 Monitoraggio fermato")


