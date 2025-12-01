#!/usr/bin/env python3
"""Test diretto delle chiamate API per vedere partite disponibili"""

import os
import requests
import json
from datetime import datetime

print("=" * 80)
print("🧪 TEST CHIAMATE API IN TEMPO REALE")
print("=" * 80)
print()

# API-SPORTS Key
api_key = os.getenv("API_FOOTBALL_KEY", "95c43f936816cd4389a747fd2cfe061a")
base_url = "https://v3.football.api-sports.io"

headers = {
    "x-rapidapi-key": api_key,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

print(f"🔑 API Key: {'***' + api_key[-10:] if len(api_key) > 10 else '***'}")
print()

# 1. Test chiamata fixtures (partite di oggi)
print("📡 1. Chiamata API-SPORTS: Fixtures di oggi...")
try:
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"{base_url}/fixtures"
    params = {
        "date": today,
        "timezone": "Europe/Rome"
    }
    
    response = requests.get(url, headers=headers, params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        fixtures = data.get("response", [])
        print(f"   ✅ Risposta OK: {len(fixtures)} partite trovate per oggi ({today})")
        print()
        
        if fixtures:
            print("   📋 PRIME 10 PARTITE DI OGGI:")
            print("   " + "-" * 76)
            for i, fixture in enumerate(fixtures[:10], 1):
                home = fixture.get("teams", {}).get("home", {}).get("name", "N/A")
                away = fixture.get("teams", {}).get("away", {}).get("name", "N/A")
                league = fixture.get("league", {}).get("name", "N/A")
                status = fixture.get("fixture", {}).get("status", {}).get("long", "N/A")
                minute = fixture.get("fixture", {}).get("status", {}).get("elapsed")
                score_home = fixture.get("goals", {}).get("home")
                score_away = fixture.get("goals", {}).get("away")
                
                score_str = f"{score_home}-{score_away}" if score_home is not None else "N/A"
                minute_str = f"{minute}'" if minute else ""
                
                print(f"   {i:2d}. {home} vs {away}")
                print(f"       📊 {league}")
                print(f"       ⏱️  {status} {minute_str}")
                if score_home is not None:
                    print(f"       ⚽ {score_str}")
                print()
        else:
            print("   ⚠️  Nessuna partita trovata per oggi")
    else:
        print(f"   ❌ Errore: {response.status_code}")
        print(f"   📄 Risposta: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Errore nella chiamata: {e}")

print()

# 2. Test chiamata fixtures LIVE
print("📡 2. Chiamata API-SPORTS: Partite LIVE...")
try:
    url = f"{base_url}/fixtures"
    params = {
        "live": "all",
        "timezone": "Europe/Rome"
    }
    
    response = requests.get(url, headers=headers, params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        fixtures = data.get("response", [])
        print(f"   ✅ Risposta OK: {len(fixtures)} partite LIVE trovate")
        print()
        
        if fixtures:
            print("   🔴 PARTITE LIVE IN CORSO:")
            print("   " + "-" * 76)
            for i, fixture in enumerate(fixtures[:15], 1):
                home = fixture.get("teams", {}).get("home", {}).get("name", "N/A")
                away = fixture.get("teams", {}).get("away", {}).get("name", "N/A")
                league = fixture.get("league", {}).get("name", "N/A")
                minute = fixture.get("fixture", {}).get("status", {}).get("elapsed")
                score_home = fixture.get("goals", {}).get("home", 0)
                score_away = fixture.get("goals", {}).get("away", 0)
                
                print(f"   {i:2d}. {home} vs {away}")
                print(f"       📊 {league}")
                print(f"       ⚽ {score_home}-{score_away} al {minute}'")
                print()
        else:
            print("   ⚠️  Nessuna partita LIVE al momento")
    else:
        print(f"   ❌ Errore: {response.status_code}")
        print(f"   📄 Risposta: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Errore nella chiamata: {e}")

print()

# 3. Verifica rate limit
print("📊 3. Verifica Rate Limit API-SPORTS...")
try:
    url = f"{base_url}/status"
    response = requests.get(url, headers=headers, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        if "response" in data:
            requests_limit = data["response"].get("requests", {}).get("limit", "N/A")
            requests_remaining = data["response"].get("requests", {}).get("remaining", "N/A")
            requests_reset = data["response"].get("requests", {}).get("reset", "N/A")
            
            print(f"   ✅ Limite giornaliero: {requests_limit}")
            print(f"   ✅ Chiamate rimanenti: {requests_remaining}")
            print(f"   ✅ Reset alle: {requests_reset}")
    else:
        print(f"   ⚠️  Impossibile verificare rate limit: {response.status_code}")
except Exception as e:
    print(f"   ⚠️  Errore nella verifica: {e}")

print()
print("=" * 80)
print("💡 CONCLUSIONE:")
print("=" * 80)
print()
print("Se vedi partite qui sopra, significa che:")
print("  ✅ L'API-SPORTS funziona correttamente")
print("  ✅ La chiave API è valida")
print("  ✅ Ci sono partite disponibili da monitorare")
print()
print("Se il sistema non sta chiamando l'API, potrebbe essere:")
print("  ⚠️  Bloccato durante l'import")
print("  ⚠️  In attesa tra un ciclo e l'altro (5 minuti)")
print("  ⚠️  In fase di inizializzazione")
print()
print("=" * 80)







