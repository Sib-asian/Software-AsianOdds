#!/usr/bin/env python3
"""Script per verificare lo stato del sistema"""
import subprocess
import sys
from pathlib import Path

print("=" * 80)
print("🔍 VERIFICA STATO SISTEMA")
print("=" * 80)
print()

# Verifica processi Python
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                          capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    python_processes = [l for l in lines if 'python.exe' in l.lower()]
    print(f"📊 Processi Python attivi: {len(python_processes) - 1}")  # -1 per l'intestazione
    if len(python_processes) > 1:
        print("   ✅ Ci sono processi Python attivi")
    else:
        print("   ❌ Nessun processo Python attivo")
except:
    print("   ⚠️  Impossibile verificare processi Python")

print()

# Verifica log
log_file = Path('logs/automation_service_20251119.log')
if log_file.exists():
    txt = log_file.read_text(encoding='utf-8', errors='ignore')
    lines = [l for l in txt.split('\n') if l.strip()]
    print(f"📄 File log: {len(lines)} righe")
    
    # Ultime righe
    recent = [l for l in lines if '01:3' in l or '01:4' in l or '01:5' in l]
    if recent:
        print(f"   Ultime righe recenti: {len(recent)}")
        print("   Ultime 3 righe:")
        for l in recent[-3:]:
            print(f"   {l[:100]}")
    else:
        print("   ⚠️  Nessuna riga recente (sistema potrebbe essere bloccato)")
else:
    print("   ❌ File log non trovato")

print()
print("=" * 80)
print("💡 Per avviare il sistema, esegui: python avvia_sistema_robusto.py")
print("=" * 80)


