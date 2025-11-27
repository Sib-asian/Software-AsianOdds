#!/usr/bin/env python3
"""Monitora le chiamate API e le partite monitorate in tempo reale"""

import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

log_file = Path('logs/automation_service_20251118.log')

print("=" * 80)
print("📡 MONITORAGGIO CHIAMATE API E PARTITE")
print("=" * 80)
print()

if not log_file.exists():
    print("❌ File log non trovato!")
    exit(1)

# Leggi le ultime righe del log
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()
    
    # Cerca chiamate API recenti (ultime 200 righe)
    recent_lines = lines[-200:] if len(lines) > 200 else lines
    
    api_calls = []
    matches_found = []
    api_sports_calls = []
    theodds_calls = []
    football_data_calls = []
    
    for i, line in enumerate(recent_lines):
        # Chiamate API-SPORTS
        if 'api-sports' in line.lower() or 'v3.football' in line.lower() or 'API-SPORTS' in line:
            api_sports_calls.append((i, line.strip()))
        
        # Chiamate TheOddsAPI
        if 'theoddsapi' in line.lower() or 'theodds' in line.lower():
            theodds_calls.append((i, line.strip()))
        
        # Chiamate Football-Data.org
        if 'football-data' in line.lower() or 'footballdata' in line.lower():
            football_data_calls.append((i, line.strip()))
        
        # Partite trovate
        if 'trovate' in line.lower() and 'partite' in line.lower():
            matches_found.append((i, line.strip()))
        
        # Partite specifiche
        if 'vs' in line and ('INFO' in line or 'DEBUG' in line):
            # Cerca pattern tipo "Team1 vs Team2"
            match = re.search(r'([A-Za-z0-9\s]+)\s+vs\s+([A-Za-z0-9\s]+)', line)
            if match:
                matches_found.append((i, line.strip()))
        
        # Cicli di analisi
        if 'Running analysis cycle' in line or 'Ciclo' in line:
            api_calls.append((i, line.strip()))
    
    print("🔍 ULTIME CHIAMATE API (ultime 200 righe del log):")
    print("-" * 80)
    print()
    
    # API-SPORTS
    if api_sports_calls:
        print("📡 API-SPORTS (v3.football.api-sports.io):")
        for idx, call in api_sports_calls[-10:]:  # Ultime 10
            timestamp = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', call)
            if timestamp:
                print(f"  🕐 {timestamp.group(1)} - {call[:100]}")
            else:
                print(f"  📋 {call[:100]}")
        print()
    else:
        print("⚠️  Nessuna chiamata API-SPORTS trovata nelle ultime 200 righe")
        print()
    
    # TheOddsAPI
    if theodds_calls:
        print("📊 TheOddsAPI:")
        for idx, call in theodds_calls[-10:]:  # Ultime 10
            timestamp = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', call)
            if timestamp:
                print(f"  🕐 {timestamp.group(1)} - {call[:100]}")
            else:
                print(f"  📋 {call[:100]}")
        print()
    else:
        print("⚠️  Nessuna chiamata TheOddsAPI trovata nelle ultime 200 righe")
        print()
    
    # Football-Data.org
    if football_data_calls:
        print("🌐 Football-Data.org:")
        for idx, call in football_data_calls[-10:]:  # Ultime 10
            timestamp = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', call)
            if timestamp:
                print(f"  🕐 {timestamp.group(1)} - {call[:100]}")
            else:
                print(f"  📋 {call[:100]}")
        print()
    else:
        print("⚠️  Nessuna chiamata Football-Data.org trovata nelle ultime 200 righe")
        print()
    
    # Partite trovate
    if matches_found:
        print("⚽ PARTITE TROVATE:")
        print("-" * 80)
        for idx, match in matches_found[-15:]:  # Ultime 15
            timestamp = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', match)
            if timestamp:
                print(f"  🕐 {timestamp.group(1)}")
            print(f"  📋 {match[:120]}")
            print()
    else:
        print("⚠️  Nessuna partita trovata nelle ultime 200 righe")
        print()
    
    # Cicli di analisi
    if api_calls:
        print("🔄 CICLI DI ANALISI:")
        print("-" * 80)
        for idx, cycle in api_calls[-10:]:  # Ultimi 10
            timestamp = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', cycle)
            if timestamp:
                print(f"  🕐 {timestamp.group(1)} - {cycle[:100]}")
            else:
                print(f"  📋 {cycle[:100]}")
        print()
    
    # Cerca partite specifiche nelle ultime righe
    print("=" * 80)
    print("📋 ULTIME 20 RIGHE DEL LOG (per vedere attività recente):")
    print("=" * 80)
    print()
    for line in lines[-20:]:
        if line.strip():
            print(line.rstrip())
    
    print()
    print("=" * 80)
    print("💡 NOTA: Se non vedi chiamate recenti, il sistema potrebbe essere:")
    print("   1. In fase di inizializzazione")
    print("   2. In attesa tra un ciclo e l'altro (5 minuti)")
    print("   3. Bloccato durante l'import")
    print()
    print("   Controlla i log in tempo reale con: python verifica_log_tempo_reale.py")
    print("=" * 80)







