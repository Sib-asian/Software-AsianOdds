#!/usr/bin/env python3
"""
Script Diagnostico - Trova Origine Chiamate API
================================================

Questo script aiuta a identificare cosa sta causando chiamate API
anche quando Render è sospeso.

Usage:
    python diagnosi_chiamate_api.py
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

print("🔍 DIAGNOSTICO CHIAMATE API")
print("=" * 60)
print()

# ============================================================
# 1. VERIFICA VARIABILI D'AMBIENTE
# ============================================================
print("📋 1. VERIFICA VARIABILI D'AMBIENTE")
print("-" * 60)

env_vars = {
    'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
    'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
    'API_FOOTBALL_KEY': os.getenv('API_FOOTBALL_KEY'),
    'THEODDS_API_KEY': os.getenv('THEODDS_API_KEY'),
}

for key, value in env_vars.items():
    if value:
        masked = value[:8] + '***' if len(value) > 8 else '***'
        print(f"  ✅ {key}: {masked}")
        if key == 'API_FOOTBALL_KEY':
            print(f"     ⚠️  CHIAVE API ATTIVA - Se vedi questa, c'è un file .env locale!")
    else:
        print(f"  ❌ {key}: NON CONFIGURATA")

print()

# ============================================================
# 2. VERIFICA FILE CONFIG.JSON
# ============================================================
print("📋 2. VERIFICA FILE CONFIG.JSON (INSICURO!)")
print("-" * 60)

config_files = ['config.json', 'automation_config.json', '.env', '.env.local']
for config_file in config_files:
    if os.path.exists(config_file):
        print(f"  🔴 TROVATO: {config_file}")
        if config_file == 'config.json':
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                if 'telegram_token' in config and config['telegram_token']:
                    print(f"     ⚠️  Contiene telegram_token!")
                if 'telegram_chat_id' in config and config['telegram_chat_id']:
                    print(f"     ⚠️  Contiene telegram_chat_id!")
                print(f"     ⚠️  Questo file può far partire script in background!")
            except:
                pass
    else:
        print(f"  ✅ Non trovato: {config_file}")

print()

# ============================================================
# 3. VERIFICA CACHE DATABASE
# ============================================================
print("📋 3. VERIFICA CACHE DATABASE")
print("-" * 60)

cache_files = ['api_cache.db', '/data/api_cache.db', 'betting_database.db']
for cache_file in cache_files:
    if os.path.exists(cache_file):
        print(f"  📁 TROVATO: {cache_file}")
        try:
            # Controlla ultima modifica
            mtime = os.path.getmtime(cache_file)
            last_modified = datetime.fromtimestamp(mtime)
            hours_ago = (datetime.now() - last_modified).total_seconds() / 3600

            print(f"     Ultima modifica: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"     Ore fa: {hours_ago:.1f} ore")

            if hours_ago < 2:
                print(f"     🔴 MODIFICATO RECENTEMENTE! Qualcosa sta ancora scrivendo!")

            # Controlla contenuto (se è SQLite)
            if cache_file.endswith('.db'):
                try:
                    conn = sqlite3.connect(cache_file)
                    cursor = conn.cursor()

                    # Verifica tabella api_usage
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]

                    if 'api_usage' in tables:
                        cursor.execute("SELECT date, provider, calls FROM api_usage ORDER BY date DESC LIMIT 5")
                        usage = cursor.fetchall()
                        if usage:
                            print(f"     Ultimi utilizzi API:")
                            for date, provider, calls in usage:
                                print(f"       {date}: {provider} = {calls} chiamate")

                    conn.close()
                except Exception as e:
                    print(f"     ⚠️  Errore lettura DB: {e}")
        except Exception as e:
            print(f"     ⚠️  Errore: {e}")
    else:
        print(f"  ❌ Non trovato: {cache_file}")

print()

# ============================================================
# 4. VERIFICA PROCESSI PYTHON IN ESECUZIONE
# ============================================================
print("📋 4. VERIFICA PROCESSI PYTHON ATTIVI")
print("-" * 60)

try:
    import psutil

    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                # Filtra questo stesso script
                if 'diagnosi_chiamate_api.py' not in cmdline:
                    create_time = datetime.fromtimestamp(proc.info['create_time'])
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'cmd': cmdline,
                        'started': create_time
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if python_processes:
        print(f"  🔴 TROVATI {len(python_processes)} PROCESSI PYTHON:")
        for proc in python_processes:
            hours_running = (datetime.now() - proc['started']).total_seconds() / 3600
            print(f"\n  PID {proc['pid']}:")
            print(f"    Avviato: {proc['started'].strftime('%Y-%m-%d %H:%M:%S')} ({hours_running:.1f} ore fa)")
            print(f"    Comando: {proc['cmd'][:100]}")

            # Verifica se è automation_24h
            if 'automation_24h' in proc['cmd'] or 'start_automation' in proc['cmd']:
                print(f"    🔴 QUESTO È IL TUO SCRIPT DI AUTOMAZIONE!")
                print(f"    🔴 STA FACENDO CHIAMATE API IN BACKGROUND!")
    else:
        print(f"  ✅ Nessun processo Python attivo (oltre a questo script)")

except ImportError:
    print(f"  ⚠️  psutil non installato - installa con: pip install psutil")
    print(f"  ℹ️  In alternativa usa questi comandi:")
    print()
    if sys.platform == 'win32':
        print(f"    tasklist | findstr python")
    else:
        print(f"    ps aux | grep python")

print()

# ============================================================
# 5. VERIFICA TASK SCHEDULER / CRON
# ============================================================
print("📋 5. VERIFICA TASK SCHEDULER / CRON")
print("-" * 60)

if sys.platform == 'win32':
    print(f"  ℹ️  Su Windows, controlla Task Scheduler:")
    print(f"    1. Apri Task Scheduler (taskschd.msc)")
    print(f"    2. Cerca task con nome 'automation' o 'betting'")
    print(f"    3. Verifica se ce n'è qualcuno attivo")
else:
    print(f"  ℹ️  Su Linux/Mac, controlla crontab:")
    print(f"    crontab -l")
    try:
        import subprocess
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if result.returncode == 0:
            cron_jobs = result.stdout.strip()
            if cron_jobs and not cron_jobs.startswith('no crontab'):
                print(f"\n  🔴 TROVATI CRON JOBS:")
                print(f"{cron_jobs}")
            else:
                print(f"  ✅ Nessun cron job attivo")
        else:
            print(f"  ✅ Nessun cron job attivo")
    except:
        print(f"  ⚠️  Non riesco a controllare crontab automaticamente")

print()

# ============================================================
# 6. RIEPILOGO E RACCOMANDAZIONI
# ============================================================
print("=" * 60)
print("📊 RIEPILOGO")
print("=" * 60)
print()

recommendations = []

# Check 1: Variabili ambiente
if env_vars['API_FOOTBALL_KEY']:
    recommendations.append(
        "🔴 CRITICO: Hai variabili d'ambiente configurate localmente!\n"
        "   Questo significa che se lanci script Python, possono fare chiamate API.\n"
        "   SOLUZIONE: Rinomina .env in .env.disabled"
    )

# Check 2: config.json
if os.path.exists('config.json'):
    recommendations.append(
        "🔴 CRITICO: Trovato config.json con credenziali!\n"
        "   Questo file può far partire script in background.\n"
        "   SOLUZIONE: Rinomina config.json in config.json.disabled"
    )

# Check 3: Cache recente
for cache_file in cache_files:
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        last_modified = datetime.fromtimestamp(mtime)
        hours_ago = (datetime.now() - last_modified).total_seconds() / 3600
        if hours_ago < 2:
            recommendations.append(
                f"🔴 CRITICO: {cache_file} modificato {hours_ago:.1f} ore fa!\n"
                f"   Qualcosa sta ancora scrivendo nel database.\n"
                f"   SOLUZIONE: Trova e ferma il processo"
            )

if recommendations:
    print("⚠️  PROBLEMI TROVATI:")
    print()
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
        print()
else:
    print("✅ Nessun problema evidente trovato!")
    print()
    print("🤔 Possibili cause rimanenti:")
    print("  1. Dashboard API che mostra statistiche aggregate del giorno")
    print("  2. Processo nascosto/servizio di sistema")
    print("  3. Docker container locale ancora in esecuzione")
    print()

print("=" * 60)
print("✅ Diagnostico completato!")
print("=" * 60)
