"""
Script semplice per monitorare i log in tempo reale
"""
import time
import os
from pathlib import Path

def monitor_logs(log_file='automation_24h.log', lines_to_show=30):
    """Monitora i log in tempo reale"""
    log_path = Path(log_file)
    
    print("=" * 70)
    print("📊 MONITORAGGIO LOG LIVE - SISTEMA AUTOMATION 24H")
    print("=" * 70)
    print("\n✅ Nuove funzionalità attive:")
    print("   • 58 filtri anti-banali")
    print("   • Traduzioni italiane mercati")
    print("   • Statistiche live nei messaggi")
    print("   • Champions League femminile supportata")
    print("   • Europa Cup Women supportata")
    print("   • Soglie: 75% conf, 10% EV")
    print("\n" + "=" * 70)
    print("📋 Monitoraggio log in tempo reale (Ctrl+C per interrompere)")
    print("=" * 70 + "\n")
    
    if not log_path.exists():
        print(f"⚠️  File log non trovato: {log_file}")
        print("⏳ In attesa che il sistema generi il file log...\n")
        while not log_path.exists():
            time.sleep(2)
        print("✅ File log trovato! Inizio monitoraggio...\n")
    
    # Leggi le ultime righe
    last_position = 0
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if lines:
                print("📋 Ultime righe del log:\n")
                for line in lines[-lines_to_show:]:
                    print(line.rstrip())
                last_position = f.tell()
    
    print("\n" + "=" * 70)
    print("🔄 Monitoraggio continuo (nuove righe appaiono qui sotto)")
    print("=" * 70 + "\n")
    
    try:
        while True:
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    if new_lines:
                        for line in new_lines:
                            print(line.rstrip())
                        last_position = f.tell()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n✅ Monitoraggio interrotto dall'utente")
        print("📊 Il sistema continua a funzionare in background")

if __name__ == '__main__':
    monitor_logs()

