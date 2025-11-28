"""
Test script per verificare l'estrazione del minuto dalle partite LIVE.
Questo script testa direttamente l'API per capire cosa restituisce.
"""
import os
import sys
import json
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Aggiungi la directory principale del progetto al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def test_api_football_fixtures():
    """Test diretto dell'API per vedere cosa restituisce per partite LIVE."""
    
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key:
        print("❌ API_FOOTBALL_KEY non configurata")
        return
    
    base_url = "https://v3.football.api-sports.io"
    
    # Data di oggi
    today = datetime.now(timezone.utc).date()
    date_str = today.strftime("%Y-%m-%d")
    
    print("=" * 80)
    print(f"TEST: Estrazione minuto da API-Football per partite LIVE del {date_str}")
    print("=" * 80)
    
    # 1. Fetch fixtures per oggi
    url = f"{base_url}/fixtures?date={date_str}"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "v3.football.api-sports.io"
    }
    
    print(f"\n📡 Chiamata API: {url}")
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        if data.get("errors"):
            print(f"❌ Errori API: {data['errors']}")
            return
        
        fixtures = data.get("response", [])
        print(f"✅ Trovate {len(fixtures)} partite per oggi")
        
        # Filtra solo partite LIVE
        live_fixtures = []
        for fixture in fixtures:
            fixture_data = fixture.get("fixture", {})
            status_short = fixture_data.get("status", {}).get("short", "")
            is_live = status_short in ["1H", "HT", "2H", "ET", "P", "LIVE"]
            
            if is_live:
                live_fixtures.append(fixture)
        
        print(f"✅ Trovate {len(live_fixtures)} partite LIVE")
        
        if not live_fixtures:
            print("⚠️ Nessuna partita LIVE trovata. Provo con ieri...")
            yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
            date_str_yesterday = yesterday.strftime("%Y-%m-%d")
            url_yesterday = f"{base_url}/fixtures?date={date_str_yesterday}"
            req_yesterday = urllib.request.Request(url_yesterday, headers=headers)
            with urllib.request.urlopen(req_yesterday, timeout=10) as response:
                data_yesterday = json.loads(response.read().decode())
            fixtures_yesterday = data_yesterday.get("response", [])
            for fixture in fixtures_yesterday:
                fixture_data = fixture.get("fixture", {})
                status_short = fixture_data.get("status", {}).get("short", "")
                is_live = status_short in ["1H", "HT", "2H", "ET", "P", "LIVE"]
                if is_live:
                    live_fixtures.append(fixture)
            print(f"✅ Trovate {len(live_fixtures)} partite LIVE (oggi + ieri)")
        
        if not live_fixtures:
            print("❌ Nessuna partita LIVE trovata. Impossibile testare.")
            return
        
        # Analizza le prime 3 partite LIVE
        for i, fixture in enumerate(live_fixtures[:3]):
            print(f"\n{'=' * 80}")
            print(f"PARTITA {i+1}:")
            print(f"{'=' * 80}")
            
            fixture_data = fixture.get("fixture", {})
            teams_data = fixture.get("teams", {})
            
            home_team = teams_data.get("home", {}).get("name", "?")
            away_team = teams_data.get("away", {}).get("name", "?")
            
            print(f"\n🏆 {home_team} vs {away_team}")
            
            # Analizza fixture_data
            print(f"\n📋 FIXTURE DATA:")
            print(f"   Keys disponibili: {list(fixture_data.keys())}")
            
            # Status
            status_data = fixture_data.get("status", {})
            print(f"\n📊 STATUS:")
            print(f"   Tipo: {type(status_data)}")
            if isinstance(status_data, dict):
                print(f"   Keys: {list(status_data.keys())}")
                print(f"   Contenuto completo: {json.dumps(status_data, indent=2, default=str)}")
            else:
                print(f"   Valore: {status_data}")
            
            # Estrai minuto
            minute = status_data.get("elapsed") if isinstance(status_data, dict) else None
            elapsed_time = status_data.get("elapsed_time") if isinstance(status_data, dict) else None
            status_short = status_data.get("short", "") if isinstance(status_data, dict) else ""
            status_long = status_data.get("long", "") if isinstance(status_data, dict) else ""
            
            print(f"\n⏰ MINUTO:")
            print(f"   elapsed: {minute}")
            print(f"   elapsed_time: {elapsed_time}")
            print(f"   status.short: {status_short}")
            print(f"   status.long: {status_long}")
            
            # Data partita
            date_str_fixture = fixture_data.get("date", "")
            print(f"\n📅 DATA PARTITA:")
            print(f"   date string: {date_str_fixture}")
            
            if date_str_fixture:
                try:
                    fixture_date = datetime.fromisoformat(date_str_fixture.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    time_diff = (now - fixture_date).total_seconds() / 60
                    print(f"   fixture_date (parsed): {fixture_date}")
                    print(f"   now (UTC): {now}")
                    print(f"   time_diff: {time_diff:.1f} minuti")
                    
                    if time_diff > 0 and time_diff < 120:
                        calculated_minute = int(time_diff)
                        print(f"   ⏰ Minuto calcolato dalla data: {calculated_minute}'")
                except Exception as e:
                    print(f"   ❌ Errore parsing data: {e}")
            
            # Score
            score_data = fixture_data.get("score", {})
            print(f"\n⚽ SCORE:")
            print(f"   score keys: {list(score_data.keys()) if isinstance(score_data, dict) else 'NOT_DICT'}")
            if isinstance(score_data, dict):
                print(f"   score completo: {json.dumps(score_data, indent=2, default=str)}")
            
            # Test statistiche
            fixture_id = fixture_data.get("id")
            if fixture_id:
                print(f"\n📊 TEST STATISTICHE (fixture_id={fixture_id}):")
                stats_url = f"{base_url}/fixtures/statistics?fixture={fixture_id}"
                try:
                    req_stats = urllib.request.Request(stats_url, headers=headers)
                    with urllib.request.urlopen(req_stats, timeout=10) as response_stats:
                        stats_data = json.loads(response_stats.read().decode())
                    
                    if stats_data.get("response"):
                        stats_response = stats_data["response"]
                        print(f"   ✅ Statistiche disponibili ({len(stats_response)} team)")
                        
                        # Cerca minuto nelle statistiche
                        for team_stat in stats_response:
                            print(f"   Team stat keys: {list(team_stat.keys()) if isinstance(team_stat, dict) else 'NOT_DICT'}")
                            if isinstance(team_stat, dict):
                                if "minute" in team_stat:
                                    print(f"   ⏰ Minuto trovato nelle statistiche: {team_stat.get('minute')}")
                                if "elapsed" in team_stat:
                                    print(f"   ⏰ Elapsed trovato nelle statistiche: {team_stat.get('elapsed')}")
                    else:
                        print(f"   ❌ Nessuna statistica disponibile")
                except Exception as e:
                    print(f"   ❌ Errore fetch statistiche: {e}")
            
            print(f"\n{'=' * 80}\n")
    
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from datetime import timedelta
    test_api_football_fixtures()

