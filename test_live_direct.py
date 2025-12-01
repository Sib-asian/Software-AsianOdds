#!/usr/bin/env python3
"""
Test diretto per verificare partite live
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta

# Carica .env
try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

print("=" * 70)
print("🔍 TEST DIRETTO PARTITE LIVE")
print("=" * 70)
print()

# Test 1: TheOddsAPI
print("📡 TEST 1: TheOddsAPI")
print("-" * 70)
theodds_key = os.getenv("THEODDS_API_KEY", "")
if theodds_key:
    print("✅ THEODDS_API_KEY trovata")
    try:
        # Cerca partite live
        now = datetime.now()
        url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={theodds_key}&regions=eu&markets=h2h&oddsFormat=decimal"
        
        print(f"🔍 Cercando partite live su TheOddsAPI...")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            
            if data:
                print(f"✅ Trovate {len(data)} partite totali")
                
                # Filtra partite live (iniziata ma non finita)
                live_matches = []
                for event in data:
                    commence_time_str = event.get("commence_time")
                    if commence_time_str:
                        commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
                        commence_time_local = commence_time.replace(tzinfo=None)
                        
                        # Partita iniziata nelle ultime 3 ore
                        if commence_time_local < now and (now - commence_time_local).total_seconds() < 10800:
                            live_matches.append(event)
                
                print(f"🎯 Partite potenzialmente LIVE: {len(live_matches)}")
                print()
                
                if live_matches:
                    print("📊 PARTITE LIVE TROVATE:")
                    print()
                    for i, match in enumerate(live_matches[:10], 1):  # Prime 10
                        home = match.get("home_team", "?")
                        away = match.get("away_team", "?")
                        commence = match.get("commence_time", "")
                        match_id = match.get("id", "?")
                        
                        # Calcola minuto approssimativo
                        if commence:
                            try:
                                commence_time = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                                commence_time_local = commence_time.replace(tzinfo=None)
                                elapsed_minutes = int((now - commence_time_local).total_seconds() / 60)
                                if elapsed_minutes > 90:
                                    elapsed_minutes = 90
                            except:
                                elapsed_minutes = 0
                        else:
                            elapsed_minutes = 0
                        
                        print(f"{i}. {home} vs {away}")
                        print(f"   🆔 ID: {match_id}")
                        print(f"   ⏰ Iniziata: {commence_time_local.strftime('%H:%M') if commence else '?'} (circa {elapsed_minutes}' fa)")
                        print()
            else:
                print("⚠️  Nessuna partita trovata")
    except Exception as e:
        print(f"❌ Errore TheOddsAPI: {e}")
else:
    print("⚠️  THEODDS_API_KEY non configurata")

print()
print("=" * 70)
print("📡 TEST 2: API-Football (con retry)")
print("-" * 70)

api_football_key = os.getenv("API_FOOTBALL_KEY", "")
if api_football_key:
    print("✅ API_FOOTBALL_KEY trovata")
    
    # Prova con retry
    for attempt in range(3):
        try:
            print(f"🔍 Tentativo {attempt + 1}/3...")
            url = "https://v3.football.api-sports.io/fixtures?live=all"
            headers = {
                "x-rapidapi-key": api_football_key,
                "x-rapidapi-host": "v3.football.api-sports.io"
            }
            
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
                
                if data and data.get("response"):
                    live_count = len(data["response"])
                    print(f"✅ TROVATE {live_count} PARTITE LIVE!")
                    print()
                    
                    for i, fixture in enumerate(data["response"][:10], 1):  # Prime 10
                        teams = fixture.get("teams", {})
                        home = teams.get("home", {}).get("name", "?")
                        away = teams.get("away", {}).get("name", "?")
                        
                        score = fixture.get("goals", {})
                        score_home = score.get("home", 0)
                        score_away = score.get("away", 0)
                        
                        fixture_info = fixture.get("fixture", {})
                        minute = fixture_info.get("status", {}).get("elapsed", 0)
                        status = fixture_info.get("status", {}).get("long", "Live")
                        
                        league = fixture.get("league", {})
                        league_name = league.get("name", "Unknown")
                        
                        print(f"{i}. {home} vs {away}")
                        print(f"   📍 {league_name}")
                        print(f"   ⚽ {score_home}-{score_away} al {minute}'")
                        print(f"   📊 {status}")
                        print()
                    
                    break
                else:
                    print("⚠️  Nessuna partita live trovata")
                    break
        except urllib.error.HTTPError as e:
            if e.code == 500:
                print(f"⚠️  Errore 500 (tentativo {attempt + 1}/3) - potrebbe essere temporaneo")
                if attempt < 2:
                    import time
                    time.sleep(2)
                    continue
            else:
                print(f"❌ Errore HTTP {e.code}: {e}")
                break
        except Exception as e:
            print(f"❌ Errore: {e}")
            break
else:
    print("⚠️  API_FOOTBALL_KEY non configurata")

print()
print("=" * 70)
print("✅ Test completato!")
print("=" * 70)








