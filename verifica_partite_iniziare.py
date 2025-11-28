#!/usr/bin/env python3
"""
Verifica partite che potrebbero essere già iniziate
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

import requests

print("=" * 70)
print("VERIFICA PARTITE CHE POTREBBERO ESSERE GIÀ INIZIATE")
print("=" * 70)
print()

theodds_key = os.getenv("THEODDS_API_KEY", "")
if not theodds_key:
    print("❌ THEODDS_API_KEY non configurata")
    sys.exit(1)

try:
    url = "https://api.the-odds-api.com/v4/sports/soccer/odds"
    params = {
        "apiKey": theodds_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    events = response.json()
    
    now = datetime.now()
    print(f"Ora attuale: {now.strftime('%H:%M:%S')}")
    print()
    
    # Cerca partite che potrebbero essere già iniziate
    potentially_live = []
    
    for event in events:
        commence_time_str = event.get("commence_time")
        if not commence_time_str:
            continue
        
        commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        commence_time_local = commence_time.replace(tzinfo=None)
        
        # Partite di oggi che sono già iniziate
        if commence_time_local.date() == now.date() and commence_time_local < now:
            elapsed_minutes = int((now - commence_time_local).total_seconds() / 60)
            
            # Se è iniziata da meno di 2 ore, potrebbe essere ancora in corso
            if elapsed_minutes < 120:
                potentially_live.append({
                    'event': event,
                    'elapsed_minutes': elapsed_minutes,
                    'commence_time': commence_time_local
                })
    
    if potentially_live:
        print(f"Trovate {len(potentially_live)} partite che potrebbero essere LIVE:")
        print()
        
        for i, item in enumerate(potentially_live, 1):
            event = item['event']
            home = event.get("home_team", "?")
            away = event.get("away_team", "?")
            elapsed = item['elapsed_minutes']
            commence = item['commence_time']
            
            print(f"{i}. {home} vs {away}")
            print(f"   Iniziata alle: {commence.strftime('%H:%M')}")
            print(f"   Minuti trascorsi: ~{elapsed}'")
            print(f"   Status probabile: {'LIVE' if elapsed < 90 else 'FINITA'}")
            print()
        
        print("=" * 70)
        print("IMPORTANTE:")
        print("=" * 70)
        print()
        print("Queste partite POTREBBERO essere live, ma il sistema")
        print("NON può analizzarle senza API-Football per ottenere:")
        print("  - Score attuale")
        print("  - Minuto esatto")
        print("  - Statistiche (possesso, tiri, ecc.)")
        print()
        print("Per abilitare l'analisi live, configura API_FOOTBALL_KEY nel .env")
        print()
    else:
        print("Nessuna partita trovata che potrebbe essere già iniziata")
        print()
        print("Le partite trovate sono tutte pre-match (iniziano più tardi)")
        
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()








