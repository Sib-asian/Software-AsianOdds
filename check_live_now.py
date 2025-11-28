#!/usr/bin/env python3
"""
Script Rapido per Verificare Partite Live ORA
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Carica .env
try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from api_manager import APIManager

print("=" * 70)
print("🔍 VERIFICA PARTITE LIVE IN QUESTO MOMENTO")
print("=" * 70)
print()

api_manager = APIManager()
api_football_provider = api_manager.providers.get("api-football")

if not api_football_provider or not api_football_provider.api_key:
    print("❌ API-Football non disponibile (chiave mancante)")
    sys.exit(1)

print("📡 Chiamando API-Football per partite live...")
print()

try:
    live_fixtures = api_football_provider._request("fixtures", {"live": "all"})
    
    if not live_fixtures or not live_fixtures.get("response"):
        print("⚠️  Nessuna partita live trovata in questo momento")
        sys.exit(0)
    
    live_count = len(live_fixtures["response"])
    print(f"✅ TROVATE {live_count} PARTITE LIVE!")
    print()
    print("=" * 70)
    print("📊 ELENCO PARTITE LIVE:")
    print("=" * 70)
    print()
    
    for i, fixture in enumerate(live_fixtures["response"], 1):
        teams = fixture.get("teams", {})
        home = teams.get("home", {}).get("name", "?")
        away = teams.get("away", {}).get("name", "?")
        
        score = fixture.get("goals", {})
        score_home = score.get("home", 0)
        score_away = score.get("away", 0)
        
        fixture_info = fixture.get("fixture", {})
        status = fixture_info.get("status", {})
        minute = status.get("elapsed", 0)
        status_long = status.get("long", "Live")
        status_short = status.get("short", "LIVE")
        
        league = fixture.get("league", {})
        league_name = league.get("name", "Unknown")
        country = league.get("country", "Unknown")
        
        fixture_id = fixture_info.get("id", "?")
        
        print(f"{i}. {home} vs {away}")
        print(f"   📍 {league_name} ({country})")
        print(f"   ⚽ Score: {score_home}-{score_away} al {minute}'")
        print(f"   📊 Status: {status_long} ({status_short})")
        print(f"   🆔 ID: {fixture_id}")
        
        # Statistiche se disponibili
        stats = fixture.get("statistics", [])
        if stats:
            for stat_group in stats:
                team_stats = stat_group.get("statistics", [])
                team_id = stat_group.get("team", {}).get("id")
                is_home = team_id == teams.get("home", {}).get("id")
                
                for stat in team_stats:
                    stat_type = stat.get("type", "")
                    stat_value = stat.get("value")
                    
                    if stat_type == "Ball Possession" and is_home:
                        print(f"   📈 Possesso: {stat_value}")
                    elif stat_type == "Total Shots" and is_home:
                        shots_home = stat_value
                    elif stat_type == "Total Shots" and not is_home:
                        shots_away = stat_value
        
        print()
    
    print("=" * 70)
    print("✅ Verifica completata!")
    print()
    print("💡 Ora eseguiamo un test per vedere se il sistema le trova...")
    print()
    
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()








