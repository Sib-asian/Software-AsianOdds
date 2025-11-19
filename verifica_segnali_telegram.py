#!/usr/bin/env python3
"""Verifica se il sistema sta inviando segnali su Telegram"""

import re
from datetime import datetime
from pathlib import Path

log_file = Path('logs/automation_service_20251118.log')

print("=" * 80)
print("📱 VERIFICA SEGNALI TELEGRAM")
print("=" * 80)
print()

if not log_file.exists():
    print("❌ File log non trovato!")
    exit(1)

# Cerca segnali inviati
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()
    
    # Pattern per trovare segnali inviati
    patterns = [
        r'✅ Notified opportunity:.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
        r'send_betting_opportunity',
        r'Notified opportunity',
        r'✅.*notified.*opportunity',
    ]
    
    notified_matches = []
    for pattern in patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            # Cerca timestamp nelle righe vicine
            lines = content[:match.end()].split('\n')
            for line in lines[-5:]:
                if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line):
                    timestamp = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp:
                        notified_matches.append(timestamp.group(1))
                    break
    
    # Rimuovi duplicati e ordina
    notified_matches = sorted(set(notified_matches), reverse=True)
    
    print(f"📊 Segnali Telegram trovati nel log: {len(notified_matches)}")
    print()
    
    if notified_matches:
        print("📋 Ultimi 10 segnali inviati:")
        print("-" * 80)
        for i, ts in enumerate(notified_matches[:10], 1):
            print(f"  {i}. {ts}")
        print()
        
        # Ultimo segnale
        last_signal = notified_matches[0]
        print(f"🕐 Ultimo segnale inviato: {last_signal}")
        
        # Calcola età
        try:
            last_time = datetime.strptime(last_signal, '%Y-%m-%d %H:%M:%S')
            now = datetime.now()
            age = (now - last_time).total_seconds() / 60  # minuti
            
            print(f"⏰ Età: {int(age)} minuti fa")
            print()
            
            if age > 60:
                print("⚠️  ATTENZIONE: Nessun segnale inviato da più di 1 ora!")
                print("   Il sistema potrebbe essere fermo o non trovare opportunità.")
            else:
                print("✅ Sistema sta inviando segnali recentemente")
        except:
            pass
    else:
        print("⚠️  NESSUN segnale Telegram trovato nel log!")
        print("   Questo significa che:")
        print("   1. Il sistema non sta trovando opportunità")
        print("   2. Il sistema è fermo")
        print("   3. I segnali non vengono loggati (improbabile)")
    
    print()
    print("=" * 80)
    print("💡 RISPOSTA ALLA TUA DOMANDA:")
    print("=" * 80)
    print()
    print("SÌ, se il sistema NON sta scrivendo log, probabilmente:")
    print("  ❌ NON sta eseguendo cicli di analisi")
    print("  ❌ NON sta trovando opportunità")
    print("  ❌ NON sta inviando segnali su Telegram")
    print()
    print("Il sistema deve essere ATTIVO e FUNZIONANTE per inviare segnali.")
    print("Se non scrive log, significa che è FERMO o BLOCCATO.")
    print()
    print("=" * 80)


