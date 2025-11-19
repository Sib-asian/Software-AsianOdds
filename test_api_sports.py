#!/usr/bin/env python3
"""
Test API-SPORTS / API-Football
"""

import os
import sys
import urllib.request
import json
from pathlib import Path

# Carica .env
try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

print("=" * 70)
print("TEST API-SPORTS / API-Football")
print("=" * 70)
print()

api_key = os.getenv("API_FOOTBALL_KEY", "")
if not api_key:
    print("❌ API_FOOTBALL_KEY non trovata nel .env")
    print()
    print("Aggiungi al file .env:")
    print("API_FOOTBALL_KEY=94d5ec5f491217af0874f8a2874dfbd8")
    sys.exit(1)

print(f"✅ Chiave trovata: {api_key[:10]}...")
print()

# Test 1: Partite live
print("📡 TEST 1: Partite Live")
print("-" * 70)
try:
    url = "https://v3.football.api-sports.io/fixtures?live=all"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "v3.football.api-sports.io"
    }
    
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode())
        
        if data.get("response"):
            live_count = len(data["response"])
            print(f"✅ SUCCESSO! Trovate {live_count} partite LIVE!")
            print()
            
            if live_count > 0:
                print("📊 Prime partite live:")
                for i, fixture in enumerate(data["response"][:5], 1):
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
        else:
            print("⚠️  Nessuna partita live trovata in questo momento")
            
        # Verifica rate limit
        if "response" in data:
            print(f"📊 Rate limit info:")
            print(f"   Requests: {data.get('results', 'N/A')}")
        else:
            print(f"⚠️  Risposta: {data}")
            
except urllib.error.HTTPError as e:
    if e.code == 401:
        print("❌ ERRORE 401: Chiave API non valida o scaduta")
    elif e.code == 429:
        print("❌ ERRORE 429: Rate limit superato (100 chiamate/giorno)")
    elif e.code == 500:
        print("⚠️  ERRORE 500: Problema temporaneo del server")
    else:
        print(f"❌ ERRORE HTTP {e.code}: {e}")
except Exception as e:
    print(f"❌ Errore: {e}")

print()
print("=" * 70)
print("✅ Test completato!")
print("=" * 70)
print()
print("💡 Se il test è riuscito, il sistema ora può ottenere dati live reali!")
print("   Riavvia il servizio per applicare le modifiche.")



