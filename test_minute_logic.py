"""
Test della logica di estrazione del minuto.
Simula il comportamento del sistema senza chiamare l'API.
"""
from datetime import datetime, timezone, timedelta
import json

def test_minute_extraction_logic():
    """Test della logica di estrazione del minuto."""
    
    print("=" * 80)
    print("TEST: Logica estrazione minuto")
    print("=" * 80)
    
    # Simula diverse situazioni
    test_cases = [
        {
            "name": "Partita LIVE con elapsed presente",
            "status_data": {"short": "1H", "long": "First Half", "elapsed": 25},
            "fixture_date": datetime.now(timezone.utc) - timedelta(minutes=25),
            "expected_minute": 25
        },
        {
            "name": "Partita LIVE senza elapsed (calcolo da data)",
            "status_data": {"short": "1H", "long": "First Half", "elapsed": None},
            "fixture_date": datetime.now(timezone.utc) - timedelta(minutes=30),
            "expected_minute": 30
        },
        {
            "name": "Partita LIVE senza elapsed, time_diff negativo",
            "status_data": {"short": "1H", "long": "First Half", "elapsed": None},
            "fixture_date": datetime.now(timezone.utc) + timedelta(minutes=5),
            "expected_minute": 1  # Fallback a 1 per status 1H
        },
        {
            "name": "Partita HT senza elapsed",
            "status_data": {"short": "HT", "long": "Half Time", "elapsed": None},
            "fixture_date": datetime.now(timezone.utc) - timedelta(minutes=50),
            "expected_minute": 45  # Fallback a 45 per HT
        },
        {
            "name": "Partita 2H senza elapsed",
            "status_data": {"short": "2H", "long": "Second Half", "elapsed": None},
            "fixture_date": datetime.now(timezone.utc) - timedelta(minutes=60),
            "expected_minute": 60  # Calcolato da data (60 minuti fa)
        },
        {
            "name": "Partita LIVE senza elapsed, iniziata 5 minuti fa",
            "status_data": {"short": "LIVE", "long": "Live", "elapsed": None},
            "fixture_date": datetime.now(timezone.utc) - timedelta(minutes=5),
            "expected_minute": 5  # Calcolato da data
        },
        {
            "name": "Partita LIVE senza elapsed, iniziata più di 2 ore fa",
            "status_data": {"short": "LIVE", "long": "Live", "elapsed": None},
            "fixture_date": datetime.now(timezone.utc) - timedelta(hours=3),
            "expected_minute": 1  # Fallback a 1 per LIVE
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'=' * 80}")
        
        status_data = test_case["status_data"]
        fixture_date = test_case["fixture_date"]
        expected_minute = test_case["expected_minute"]
        
        # Simula la logica di estrazione
        minute = 0
        if isinstance(status_data, dict):
            minute = (status_data.get("elapsed") or 
                     status_data.get("elapsed_time") or 
                     status_data.get("elapsedTime") or
                     status_data.get("minute") or
                     status_data.get("time") or 0)
            
            if minute is None:
                minute = 0
            else:
                try:
                    minute = int(minute)
                except (ValueError, TypeError):
                    minute = 0
        
        status_short = status_data.get("short", "") if isinstance(status_data, dict) else ""
        
        print(f"   Minuto estratto da status: {minute}'")
        print(f"   Status: {status_short}")
        
        # Calcola dalla data se LIVE
        if status_short in ["1H", "HT", "2H", "ET", "P", "LIVE"]:
            try:
                now = datetime.now(timezone.utc)
                time_diff = (now - fixture_date).total_seconds() / 60
                
                print(f"   time_diff: {time_diff:.1f} minuti")
                
                if time_diff > 0 and time_diff < 120:
                    calculated_minute = int(time_diff)
                    if calculated_minute > minute or minute == 0:
                        minute = calculated_minute
                        print(f"   ⏰ Minuto calcolato dalla data: {minute}'")
            except Exception as e:
                print(f"   ⚠️ Errore: {e}")
        
        # Fallback
        if minute == 0:
            if status_short == "HT":
                minute = 45
                print(f"   ⏰ Minuto dedotto da status HT: 45'")
            elif status_short == "2H":
                minute = 46
                print(f"   ⏰ Minuto dedotto da status 2H: 46'")
            elif status_short == "1H":
                minute = 1
                print(f"   ⏰ Minuto dedotto da status 1H: 1'")
            elif status_short == "LIVE":
                minute = 1
                print(f"   ⏰ Minuto dedotto da status LIVE: 1'")
        
        print(f"   ✅ Minuto FINALE: {minute}'")
        print(f"   ✅ Minuto ATTESO: {expected_minute}'")
        
        if minute == expected_minute:
            print(f"   ✅ TEST PASSATO")
        else:
            print(f"   ❌ TEST FALLITO (atteso {expected_minute}, ottenuto {minute})")

if __name__ == "__main__":
    test_minute_extraction_logic()

