#!/usr/bin/env python3
"""
Script per fermare TUTTE le automazioni attive
===============================================

Controlla e termina:
1. Processi Python automation_24h
2. Servizi Streamlit/Dashboard
3. Background services
"""

import os
import sys
import json
from pathlib import Path

print("=" * 70)
print("🛑 FERMA TUTTE LE AUTOMAZIONI")
print("=" * 70)
print()

# ============================================================
# 1. TROVA PROCESSI PYTHON ATTIVI
# ============================================================
print("1️⃣  VERIFICA PROCESSI PYTHON...")
print("-" * 70)

try:
    import psutil

    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                # Esclude questo script
                if 'ferma_tutto.py' not in cmdline:
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'cmd': cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if python_processes:
        print(f"🔴 TROVATI {len(python_processes)} PROCESSI PYTHON:")
        for p in python_processes:
            print(f"\n  PID {p['pid']}:")
            print(f"    {p['cmd'][:100]}")

            # Chiedi se terminare
            if any(keyword in p['cmd'].lower() for keyword in ['automation', 'streamlit', 'dashboard', 'frontendcloud']):
                print(f"    🔴 QUESTO PROCESSO PUÒ FARE CHIAMATE API!")

                # Su Windows usa taskkill, su Unix usa kill
                if sys.platform == 'win32':
                    print(f"    Comando per terminare: taskkill /F /PID {p['pid']}")
                else:
                    print(f"    Comando per terminare: kill -9 {p['pid']}")
    else:
        print("✅ Nessun processo Python attivo")

except ImportError:
    print("⚠️  psutil non installato")
    print("   Installa con: pip install psutil")
    print()
    print("   Oppure usa comandi manuali:")
    if sys.platform == 'win32':
        print("   tasklist | findstr python")
        print("   taskkill /F /PID <pid>")
    else:
        print("   ps aux | grep python")
        print("   kill -9 <pid>")

print()

# ============================================================
# 2. DISABILITA AUTOMAZIONE NEL CODICE
# ============================================================
print("2️⃣  DISABILITA AUTOMAZIONE NEL CODICE...")
print("-" * 70)

automation_file = Path("automation_24h.py")
if automation_file.exists():
    print(f"✅ Trovato: {automation_file}")
    print()
    print("📝 Per disabilitare permanentemente, aggiungi questa riga all'inizio di automation_24h.py:")
    print()
    print("    # AUTOMAZIONE DISABILITATA - Non eseguire")
    print("    import sys; sys.exit(0)")
    print()
else:
    print(f"⚠️  File non trovato: {automation_file}")

print()

# ============================================================
# 3. RINOMINA FILE SENSIBILI
# ============================================================
print("3️⃣  RINOMINA FILE CON CREDENZIALI...")
print("-" * 70)

sensitive_files = [
    '.env',
    'config.json',
    'automation_config.json'
]

for filename in sensitive_files:
    filepath = Path(filename)
    if filepath.exists():
        new_name = f"{filename}.DISABLED"
        print(f"🔴 Trovato: {filename}")
        print(f"   Comando per disabilitare:")
        if sys.platform == 'win32':
            print(f"   ren {filename} {new_name}")
        else:
            print(f"   mv {filename} {new_name}")
    else:
        print(f"✅ Non trovato: {filename}")

print()

# ============================================================
# 4. VERIFICA RENDER
# ============================================================
print("4️⃣  VERIFICA SERVIZI CLOUD...")
print("-" * 70)
print()
print("RENDER:")
print("  1. Vai su https://dashboard.render.com")
print("  2. Trova il tuo servizio")
print("  3. Verifica che sia SOSPESO")
print("  4. Se attivo, clicca 'Suspend'")
print()
print("ALTRI CLOUD (se usi):")
print("  - Heroku: heroku ps:scale web=0")
print("  - Railway: Sospendi dal dashboard")
print("  - Vercel: Cancella deployment")
print()

# ============================================================
# 5. RIEPILOGO
# ============================================================
print("=" * 70)
print("📊 RIEPILOGO AZIONI")
print("=" * 70)
print()
print("✅ COMPLETATI:")
print("  - Verifica processi Python")
print()
print("⏳ DA FARE MANUALMENTE:")
print("  1. Termina eventuali processi trovati")
print("  2. Rinomina file sensibili (.env, config.json)")
print("  3. Verifica Render sia sospeso")
print("  4. Opzionale: Disabilita automation_24h.py con sys.exit(0)")
print()
print("=" * 70)
print("✅ Script completato!")
print("=" * 70)
