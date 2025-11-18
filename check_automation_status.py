#!/usr/bin/env python3
"""
Script per verificare se il sistema Automation 24/7 è attivo
"""

import os
import sys
import psutil
from datetime import datetime, timedelta
from pathlib import Path

def check_process():
    """Verifica se il processo automation_24h è in esecuzione"""
    print("\n" + "="*60)
    print("🔍 VERIFICA PROCESSO AUTOMATION")
    print("="*60)

    automation_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('automation_24h' in str(arg) for arg in cmdline):
                # Processo trovato
                create_time = datetime.fromtimestamp(proc.info['create_time'])
                uptime = datetime.now() - create_time

                print(f"✅ PROCESSO ATTIVO")
                print(f"   PID: {proc.info['pid']}")
                print(f"   Nome: {proc.info['name']}")
                print(f"   Avviato: {create_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Uptime: {uptime}")
                print(f"   Comando: {' '.join(cmdline)}")
                automation_running = True
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not automation_running:
        print("❌ PROCESSO NON ATTIVO")
        print("   Il sistema automation_24h.py non è in esecuzione")

    return automation_running

def check_logs():
    """Verifica i log recenti"""
    print("\n" + "="*60)
    print("📋 VERIFICA LOG RECENTI")
    print("="*60)

    log_file = Path(__file__).parent / "automation_24h.log"

    if not log_file.exists():
        print("❌ LOG FILE NON TROVATO")
        print(f"   File: {log_file}")
        return False

    # Leggi ultime 30 righe
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            recent_lines = lines[-30:] if len(lines) > 30 else lines

        # Trova ultima riga con timestamp
        last_activity = None
        for line in reversed(recent_lines):
            if line.strip():
                # Prova a estrarre timestamp (formato: YYYY-MM-DD HH:MM:SS)
                try:
                    timestamp_str = ' '.join(line.split()[:2])
                    last_activity = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    break
                except:
                    continue

        if last_activity:
            time_since = datetime.now() - last_activity
            minutes_since = time_since.total_seconds() / 60

            print(f"✅ LOG FILE TROVATO")
            print(f"   File: {log_file}")
            print(f"   Ultima attività: {last_activity.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Tempo trascorso: {int(minutes_since)} minuti fa")

            # Mostra ultime 10 righe
            print("\n   Ultime 10 righe:")
            for line in recent_lines[-10:]:
                print(f"   {line.rstrip()}")

            # Verifica se è attivo (ultima attività < 10 minuti)
            if minutes_since < 10:
                print(f"\n✅ SISTEMA ATTIVO (ultima attività {int(minutes_since)} min fa)")
                return True
            else:
                print(f"\n⚠️  SISTEMA INATTIVO (ultima attività {int(minutes_since)} min fa)")
                return False
        else:
            print(f"⚠️  LOG FILE TROVATO ma nessun timestamp valido")
            return False

    except Exception as e:
        print(f"❌ ERRORE LETTURA LOG: {e}")
        return False

def check_api_connections():
    """Verifica connessioni API"""
    print("\n" + "="*60)
    print("🌐 VERIFICA CONNESSIONI API")
    print("="*60)

    try:
        import requests

        # Test API-Football (solo status, non conta verso quota)
        api_key = os.getenv('API_FOOTBALL_KEY', '95c43f936816cd4389a747fd2cfe061a')
        url = "https://v3.football.api-sports.io/status"
        headers = {
            'x-rapidapi-key': api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }

        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data.get('response'):
                requests_info = data['response']['requests']
                print(f"✅ API-FOOTBALL CONNESSA")
                print(f"   Chiamate oggi: {requests_info.get('current', 0)}/{requests_info.get('limit_day', 100)}")
                print(f"   Quota rimanente: {requests_info.get('limit_day', 100) - requests_info.get('current', 0)}")
                return True

        print(f"⚠️  API-FOOTBALL: Status code {response.status_code}")
        return False

    except Exception as e:
        print(f"❌ ERRORE VERIFICA API: {e}")
        return False

def check_telegram():
    """Verifica configurazione Telegram"""
    print("\n" + "="*60)
    print("📱 VERIFICA TELEGRAM")
    print("="*60)

    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if token:
        print(f"✅ Token configurato: ...{token[-10:]}")
    else:
        print("⚠️  Token NON configurato (usando default)")

    if chat_id:
        print(f"✅ Chat ID configurato: {chat_id}")
    else:
        print("⚠️  Chat ID NON configurato (usando default)")

    return bool(token and chat_id)

def main():
    """Esegue tutte le verifiche"""
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*15 + "AUTOMATION 24/7 STATUS CHECK" + " "*15 + "║")
    print("╚" + "="*58 + "╝")
    print(f"\nData/Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        'processo': check_process(),
        'logs': check_logs(),
        'api': check_api_connections(),
        'telegram': check_telegram()
    }

    # Riepilogo
    print("\n" + "="*60)
    print("📊 RIEPILOGO STATO SISTEMA")
    print("="*60)

    for check, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"{emoji} {check.upper()}: {'OK' if status else 'NON OK'}")

    all_ok = all(results.values())

    if all_ok:
        print("\n🎉 SISTEMA COMPLETAMENTE OPERATIVO H24!")
        print("\n✅ Il sistema sta:")
        print("   - Monitorando partite in tempo reale")
        print("   - Analizzando value bet ogni 5 minuti")
        print("   - Inviando notifiche Telegram")
        print("   - Usando API con quota disponibile")
        return 0
    else:
        print("\n⚠️  SISTEMA NON COMPLETAMENTE OPERATIVO")

        if not results['processo'] and not results['logs']:
            print("\n❌ PROBLEMA: Sistema non in esecuzione")
            print("\n🔧 SOLUZIONE:")
            print("   Avvia il sistema con:")
            print("   python automation_24h.py")
        elif not results['api']:
            print("\n⚠️  ATTENZIONE: Problemi connessione API")
            print("   Il sistema potrebbe funzionare con limitazioni")
        elif not results['telegram']:
            print("\n⚠️  ATTENZIONE: Telegram non completamente configurato")
            print("   Potresti non ricevere tutte le notifiche")

        return 1

if __name__ == '__main__':
    try:
        # Verifica dipendenze
        import psutil
        import requests
    except ImportError as e:
        print(f"❌ Dipendenza mancante: {e}")
        print("\nInstalla con:")
        print("pip install psutil requests")
        sys.exit(1)

    sys.exit(main())
