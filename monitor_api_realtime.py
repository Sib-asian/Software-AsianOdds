#!/usr/bin/env python3
"""Monitora le chiamate API in tempo reale e mostra le partite disponibili"""

import os
import requests
import time
from datetime import datetime
from typing import List, Dict

# API-SPORTS Key
api_key = os.getenv("API_FOOTBALL_KEY", "95c43f936816cd4389a747fd2cfe061a")
base_url = "https://v3.football.api-sports.io"

headers = {
    "x-rapidapi-key": api_key,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

def get_live_matches() -> List[Dict]:
    """Ottiene partite LIVE da API-SPORTS"""
    try:
        url = f"{base_url}/fixtures"
        params = {"live": "all", "timezone": "Europe/Rome"}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", [])
        else:
            print(f"❌ Errore API: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Errore: {e}")
        return []

def get_today_matches() -> List[Dict]:
    """Ottiene partite di oggi da API-SPORTS"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"{base_url}/fixtures"
        params = {"date": today, "timezone": "Europe/Rome"}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", [])
        else:
            return []
    except:
        return []

def format_match(fixture: Dict) -> str:
    """Formatta una partita per la visualizzazione"""
    home = fixture.get("teams", {}).get("home", {}).get("name", "N/A")
    away = fixture.get("teams", {}).get("away", {}).get("name", "N/A")
    league = fixture.get("league", {}).get("name", "N/A")
    minute = fixture.get("fixture", {}).get("status", {}).get("elapsed")
    score_home = fixture.get("goals", {}).get("home")
    score_away = fixture.get("goals", {}).get("away")
    
    score_str = f"{score_home}-{score_away}" if score_home is not None else "N/A"
    minute_str = f"{minute}'" if minute else ""
    
    return f"  ⚽ {home} vs {away} | {score_str} {minute_str} | 📊 {league}"

def main():
    print("=" * 80)
    print("📡 MONITORAGGIO API-SPORTS IN TEMPO REALE")
    print("=" * 80)
    print()
    print("🔄 Aggiornamento ogni 30 secondi...")
    print("   Premi CTRL+C per fermare")
    print()
    
    try:
        while True:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"\n🕐 {now} - Controllo partite...")
            print("-" * 80)
            
            # Partite LIVE
            live_matches = get_live_matches()
            if live_matches:
                print(f"\n🔴 PARTITE LIVE ({len(live_matches)}):")
                for match in live_matches[:10]:  # Prime 10
                    print(format_match(match))
            else:
                print("\n⚠️  Nessuna partita LIVE al momento")
            
            # Partite di oggi (prossime)
            today_matches = get_today_matches()
            if today_matches:
                # Filtra solo partite non finite e non live
                upcoming = [
                    m for m in today_matches 
                    if m.get("fixture", {}).get("status", {}).get("short") not in ["FT", "AET", "PEN", "LIVE"]
                ]
                if upcoming:
                    print(f"\n📅 PROSSIME PARTITE DI OGGI ({len(upcoming)}):")
                    for match in upcoming[:5]:  # Prime 5
                        time_str = match.get("fixture", {}).get("date", "")
                        if time_str:
                            try:
                                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                                time_str = dt.strftime("%H:%M")
                            except:
                                pass
                        home = match.get("teams", {}).get("home", {}).get("name", "N/A")
                        away = match.get("teams", {}).get("away", {}).get("name", "N/A")
                        league = match.get("league", {}).get("name", "N/A")
                        print(f"  ⏰ {time_str} - {home} vs {away} | 📊 {league}")
            
            print("\n" + "=" * 80)
            print("⏳ Attendo 30 secondi...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoraggio fermato")

if __name__ == "__main__":
    main()







