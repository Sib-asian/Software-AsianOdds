#!/usr/bin/env python3
"""
Test delle connessioni API per verificare che l'AI possa chiamare servizi esterni
"""

import json
import urllib.request
import urllib.error
from datetime import datetime

# ============================================================
# API KEYS (da api_manager.py e Frontendcloud.py)
# ============================================================

API_FOOTBALL_KEY = "95c43f936816cd4389a747fd2cfe061a"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

OPENWEATHER_API_KEY = "01afa2183566fcf16d98b5a33c91eae1"
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"

THESPORTSDB_KEY = "3"  # Free key
THESPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json"


def test_api_football():
    """Test 1: API-Football - Informazioni su squadre, injuries, form"""
    print("=" * 60)
    print("TEST 1: API-Football")
    print("=" * 60)

    try:
        # Test endpoint: /status (verifica quota rimanente)
        url = f"{API_FOOTBALL_BASE}/status"

        req = urllib.request.Request(url)
        req.add_header("x-rapidapi-key", API_FOOTBALL_KEY)
        req.add_header("x-rapidapi-host", "v3.football.api-sports.io")

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

            if data.get('response'):
                account = data['response']['account']
                requests_info = data['response']['requests']

                print(f"✅ API-Football CONNESSA")
                print(f"   Account: {account.get('firstname', 'N/A')} {account.get('lastname', 'N/A')}")
                print(f"   Email: {account.get('email', 'N/A')}")
                print(f"   Chiamate oggi: {requests_info.get('current', 0)}/{requests_info.get('limit_day', 100)}")
                print(f"   Quota rimanente: {requests_info.get('limit_day', 100) - requests_info.get('current', 0)}")

                return True
            else:
                print(f"❌ API-Football: Risposta inattesa")
                print(f"   Response: {data}")
                return False

    except urllib.error.HTTPError as e:
        print(f"❌ API-Football: Errore HTTP {e.code}")
        print(f"   Messaggio: {e.read().decode()}")
        return False
    except Exception as e:
        print(f"❌ API-Football: Errore connessione - {e}")
        return False


def test_openweather():
    """Test 2: OpenWeather API - Dati meteo"""
    print("\n" + "=" * 60)
    print("TEST 2: OpenWeather API")
    print("=" * 60)

    try:
        # Test con città Milano
        city = "Milan"
        url = f"{OPENWEATHER_BASE}/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"

        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

            if data.get('weather'):
                weather = data['weather'][0]
                main = data['main']
                wind = data.get('wind', {})

                print(f"✅ OpenWeather API CONNESSA")
                print(f"   Città: {data.get('name')}")
                print(f"   Temperatura: {main.get('temp', 0):.1f}°C")
                print(f"   Condizioni: {weather.get('description', 'N/A')}")
                print(f"   Umidità: {main.get('humidity', 0)}%")
                print(f"   Vento: {wind.get('speed', 0):.1f} m/s")
                print(f"   Pressione: {main.get('pressure', 0)} hPa")

                # Test anche forecast (5 giorni)
                forecast_url = f"{OPENWEATHER_BASE}/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
                with urllib.request.urlopen(forecast_url, timeout=10) as forecast_response:
                    forecast_data = json.loads(forecast_response.read().decode())
                    forecast_count = len(forecast_data.get('list', []))
                    print(f"   Previsioni disponibili: {forecast_count} intervalli (5 giorni)")

                return True
            else:
                print(f"❌ OpenWeather: Risposta inattesa")
                print(f"   Response: {data}")
                return False

    except urllib.error.HTTPError as e:
        print(f"❌ OpenWeather: Errore HTTP {e.code}")
        print(f"   Messaggio: {e.read().decode()}")
        return False
    except Exception as e:
        print(f"❌ OpenWeather: Errore connessione - {e}")
        return False


def test_thesportsdb():
    """Test 3: TheSportsDB - Informazioni squadre (gratis, illimitato)"""
    print("\n" + "=" * 60)
    print("TEST 3: TheSportsDB (Free, Unlimited)")
    print("=" * 60)

    try:
        # Test: cerca squadra "Manchester United"
        team_name = "Manchester United"
        url = f"{THESPORTSDB_BASE}/{THESPORTSDB_KEY}/searchteams.php?t={urllib.parse.quote(team_name)}"

        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

            if data.get('teams') and len(data['teams']) > 0:
                team = data['teams'][0]

                print(f"✅ TheSportsDB CONNESSA")
                print(f"   Squadra trovata: {team.get('strTeam')}")
                print(f"   Lega: {team.get('strLeague')}")
                print(f"   Stadio: {team.get('strStadium')}")
                print(f"   Anno fondazione: {team.get('intFormedYear')}")
                print(f"   Descrizione: {team.get('strDescriptionEN', 'N/A')[:80]}...")

                return True
            else:
                print(f"❌ TheSportsDB: Nessuna squadra trovata")
                return False

    except Exception as e:
        print(f"❌ TheSportsDB: Errore connessione - {e}")
        return False


def test_api_football_injuries():
    """Test 4: API-Football - Endpoint injuries (esempio pratico)"""
    print("\n" + "=" * 60)
    print("TEST 4: API-Football - Injuries Endpoint")
    print("=" * 60)

    try:
        # Test con Premier League (season 2024)
        # League ID 39 = Premier League
        url = f"{API_FOOTBALL_BASE}/injuries?league=39&season=2024"

        req = urllib.request.Request(url)
        req.add_header("x-rapidapi-key", API_FOOTBALL_KEY)
        req.add_header("x-rapidapi-host", "v3.football.api-sports.io")

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

            if data.get('response'):
                injuries_count = len(data['response'])

                print(f"✅ Injuries Endpoint FUNZIONANTE")
                print(f"   Infortuni trovati (Premier League 2024): {injuries_count}")

                if injuries_count > 0:
                    # Mostra primi 3 infortuni
                    print(f"\n   Primi 3 infortuni:")
                    for i, injury in enumerate(data['response'][:3], 1):
                        player = injury.get('player', {})
                        team = injury.get('team', {})
                        print(f"   {i}. {player.get('name')} ({team.get('name')})")
                        print(f"      Tipo: {injury.get('type')} - {injury.get('reason')}")

                return True
            else:
                print(f"⚠️ Injuries Endpoint: Nessun infortunio trovato (potrebbe essere normale)")
                return True

    except urllib.error.HTTPError as e:
        error_msg = e.read().decode()
        print(f"❌ Injuries Endpoint: Errore HTTP {e.code}")
        print(f"   Messaggio: {error_msg}")
        # Nota: potrebbe fallire se quota esaurita
        return False
    except Exception as e:
        print(f"❌ Injuries Endpoint: Errore - {e}")
        return False


def main():
    """Esegue tutti i test delle API"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "API CONNECTIONS TEST SUITE" + " " * 17 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    print("Verifica che l'AI possa chiamare API esterne:")
    print("- API-Football (dati partite, injuries, form)")
    print("- OpenWeather (condizioni meteo)")
    print("- TheSportsDB (informazioni squadre)")
    print()

    results = {
        "API-Football": test_api_football(),
        "OpenWeather": test_openweather(),
        "TheSportsDB": test_thesportsdb(),
        "API-Football Injuries": test_api_football_injuries()
    }

    # Summary
    print("\n" + "=" * 60)
    print("RIEPILOGO TEST API")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for api_name, status in results.items():
        status_emoji = "✅" if status else "❌"
        print(f"{status_emoji} {api_name}: {'OK' if status else 'FAILED'}")

    print(f"\nTest passati: {passed}/{total}")

    if passed == total:
        print("\n🎉 TUTTE LE API SONO CONNESSE E FUNZIONANTI!")
        print("\n✅ L'AI System può chiamare:")
        print("   - API-Football per injuries, form, xG")
        print("   - OpenWeather per condizioni meteo")
        print("   - TheSportsDB per info squadre")
        print("\n💡 Nota: API-Football ha limite di 100 chiamate/giorno")
        print("   Controlla la quota sopra per vedere quante ne restano oggi.")
        return 0
    else:
        print("\n⚠️ ALCUNE API NON SONO RAGGIUNGIBILI")
        print("\n🔧 POSSIBILI CAUSE:")
        print("   - API keys non valide o scadute")
        print("   - Quota API esaurita (API-Football: 100/giorno)")
        print("   - Problemi di connessione internet")
        print("   - Rate limiting attivo")
        return 1


if __name__ == "__main__":
    exit(main())
