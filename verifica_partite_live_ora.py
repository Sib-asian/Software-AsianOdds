#!/usr/bin/env python3
"""
Verifica Partite Live in Questo Momento
"""

import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from multi_source_match_finder import MultiSourceMatchFinder
import urllib.request
import json
import urllib.parse

print("=" * 70)
print("VERIFICA PARTITE LIVE IN QUESTO MOMENTO")
print("=" * 70)
print()

now = datetime.now()
print(f"Ora attuale: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Test 1: Partite trovate dal sistema multi-fonte
print("TEST 1: Partite dal Sistema Multi-Fonte")
print("-" * 70)
finder = MultiSourceMatchFinder()
matches = finder.find_all_matches(days_ahead=1, include_minor_leagues=True, include_live=True)

print(f"Trovate {len(matches)} partite totali")
print()

# Conta partite live
live_matches = [m for m in matches if m.get('is_live')]
print(f"Partite LIVE trovate: {len(live_matches)}")
print()

# Test 2: Partite live reali da API-SPORTS
print("TEST 2: Partite Live Reali da API-SPORTS")
print("-" * 70)
api_key = os.getenv("API_FOOTBALL_KEY", "")
if api_key:
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
                print(f"Trovate {live_count} partite LIVE REALI da API-SPORTS!")
                print()
                
                if live_count > 0:
                    print("Partite Live Reali:")
                    for i, fixture in enumerate(data["response"][:20], 1):  # Prime 20
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
                        country = league.get("country", "")
                        
                        print(f"{i}. {home} vs {away}")
                        print(f"   {league_name} ({country})")
                        print(f"   Score: {score_home}-{score_away} al {minute}'")
                        print(f"   Status: {status}")
                        print()
            else:
                print("Nessuna partita live trovata da API-SPORTS")
    except Exception as e:
        print(f"Errore: {e}")
else:
    print("API-SPORTS non configurata")

print()
print("=" * 70)
print("RIEPILOGO")
print("=" * 70)
print(f"   Partite totali trovate: {len(matches)}")
print(f"   Partite LIVE trovate dal sistema: {len(live_matches)}")
if api_key:
    try:
        url = "https://v3.football.api-sports.io/fixtures?live=all"
        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "v3.football.api-sports.io"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
            live_real = len(data.get("response", []))
            print(f"   Partite LIVE REALI (API-SPORTS): {live_real}")
    except:
        print(f"   Partite LIVE REALI (API-SPORTS): N/A")
print()
print("=" * 70)

