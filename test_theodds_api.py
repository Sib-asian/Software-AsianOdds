"""
Script per testare TheOddsAPI e verificare la qualità dei dati
"""
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Carica variabili ambiente
load_dotenv()

def test_theodds_api():
    """Testa TheOddsAPI e verifica la qualità dei dati"""
    
    print("=" * 70)
    print("🔍 TEST THEODDSAPI")
    print("=" * 70)
    
    # 1. Verifica API Key
    api_key = os.getenv("THEODDS_API_KEY", "")
    if not api_key:
        print("❌ ERRORE: THEODDS_API_KEY non trovata nel .env")
        return False
    
    print(f"✅ API Key trovata: {api_key[:10]}...{api_key[-5:]}")
    print()
    
    # 2. Test chiamata API
    print("📡 Test chiamata API...")
    url = "https://api.the-odds-api.com/v4/sports/soccer/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Chiamata API riuscita!")
            
            # Verifica header per quota rimanente
            remaining = response.headers.get('x-requests-remaining', 'N/A')
            used = response.headers.get('x-requests-used', 'N/A')
            print(f"   📊 Quota API: {used} usate, {remaining} rimanenti")
            print()
            
            # 3. Analizza dati ricevuti
            events = response.json()
            print(f"📊 Analisi dati ricevuti:")
            print(f"   Totale eventi: {len(events)}")
            print()
            
            if not events:
                print("⚠️  Nessun evento ricevuto")
                return False
            
            # 4. Analizza qualità dati
            now = datetime.now()
            max_future = now + timedelta(hours=24)
            min_past = now - timedelta(hours=2)
            
            valid_matches = []
            pre_match_count = 0
            live_count = 0
            matches_with_odds = 0
            
            for event in events:
                try:
                    commence_time_str = event.get("commence_time")
                    if not commence_time_str:
                        continue
                    
                    commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
                    commence_time_local = commence_time.replace(tzinfo=None)
                    
                    is_live = commence_time_local < now
                    is_prematch = commence_time_local >= now
                    
                    if is_prematch and commence_time_local <= max_future:
                        pre_match_count += 1
                        valid_matches.append({
                            'type': 'PRE-MATCH',
                            'home': event.get('home_team', 'N/A'),
                            'away': event.get('away_team', 'N/A'),
                            'time': commence_time_local,
                            'league': event.get('sport_title', 'N/A')
                        })
                    elif is_live and commence_time_local >= min_past:
                        live_count += 1
                        valid_matches.append({
                            'type': 'LIVE',
                            'home': event.get('home_team', 'N/A'),
                            'away': event.get('away_team', 'N/A'),
                            'time': commence_time_local,
                            'league': event.get('sport_title', 'N/A')
                        })
                    
                    # Verifica quote
                    bookmakers = event.get("bookmakers", [])
                    if bookmakers:
                        matches_with_odds += 1
                        
                except Exception as e:
                    print(f"   ⚠️  Errore processando evento: {e}")
                    continue
            
            print(f"   ✅ Partite PRE-MATCH (prossime 24h): {pre_match_count}")
            print(f"   ✅ Partite LIVE (ultime 2h): {live_count}")
            print(f"   ✅ Partite con quote disponibili: {matches_with_odds}")
            print()
            
            # 5. Mostra esempi
            if valid_matches:
                print("📋 Esempi partite trovate:")
                for i, match in enumerate(valid_matches[:5], 1):
                    time_str = match['time'].strftime("%Y-%m-%d %H:%M")
                    print(f"   {i}. [{match['type']}] {match['home']} vs {match['away']}")
                    print(f"      Lega: {match['league']}")
                    print(f"      Ora: {time_str}")
                    print()
            
            # 6. Verifica qualità quote
            print("💰 Verifica qualità quote:")
            matches_with_complete_odds = 0
            for event in events[:10]:  # Controlla prime 10
                bookmakers = event.get("bookmakers", [])
                if not bookmakers:
                    continue
                
                best_odds = {"home": None, "draw": None, "away": None}
                for bookmaker in bookmakers:
                    markets = bookmaker.get("markets", [])
                    h2h_market = next((m for m in markets if m.get("key") == "h2h"), None)
                    if not h2h_market:
                        continue
                    
                    outcomes = h2h_market.get("outcomes", [])
                    for outcome in outcomes:
                        name = outcome.get("name", "").lower()
                        price = outcome.get("price")
                        
                        if price is None:
                            continue
                        
                        home_team = event.get("home_team", "").lower()
                        away_team = event.get("away_team", "").lower()
                        
                        if name == home_team:
                            if best_odds["home"] is None or price > best_odds["home"]:
                                best_odds["home"] = price
                        elif name == away_team:
                            if best_odds["away"] is None or price > best_odds["away"]:
                                best_odds["away"] = price
                        elif name in ["draw", "pareggio", "x"]:
                            if best_odds["draw"] is None or price > best_odds["draw"]:
                                best_odds["draw"] = price
                
                if best_odds["home"] and best_odds["away"] and best_odds["draw"]:
                    matches_with_complete_odds += 1
            
            print(f"   ✅ Partite con quote complete (1X2): {matches_with_complete_odds}/10")
            print()
            
            # 7. Conclusione
            print("=" * 70)
            if pre_match_count > 0 or live_count > 0:
                print("✅ RISULTATO: TheOddsAPI funziona correttamente!")
                print(f"   Trovate {pre_match_count + live_count} partite valide")
                print(f"   {matches_with_odds} partite hanno quote disponibili")
            else:
                print("⚠️  RISULTATO: TheOddsAPI funziona ma nessuna partita valida trovata")
                print("   Potrebbe essere un momento con poche partite")
            print("=" * 70)
            
            return True
            
        elif response.status_code == 401:
            print("   ❌ ERRORE: API Key non valida o scaduta")
            print("   Verifica la tua API key su https://the-odds-api.com/")
            return False
        elif response.status_code == 429:
            print("   ⚠️  ERRORE: Quota API esaurita")
            print("   Attendi o aggiorna il tuo piano")
            return False
        else:
            print(f"   ❌ ERRORE: Status code {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ❌ ERRORE: Timeout nella chiamata API")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ ERRORE: {e}")
        return False
    except Exception as e:
        print(f"   ❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_theodds_api()

