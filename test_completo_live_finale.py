#!/usr/bin/env python3
"""
Test Completo Sistema Live con API-SPORTS
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Carica .env
try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from automation_24h import Automation24H

print("=" * 70)
print("TEST COMPLETO SISTEMA LIVE - CON API-SPORTS")
print("=" * 70)
print()

# Verifica chiave
api_key = os.getenv("API_FOOTBALL_KEY", "")
if api_key:
    print(f"✅ API-SPORTS chiave configurata: {api_key[:10]}...")
else:
    print("❌ API-SPORTS chiave NON configurata")
    sys.exit(1)

print()

# Inizializza sistema
print("🚀 Inizializzando sistema...")
try:
    automation = Automation24H(
        telegram_token=os.getenv('TELEGRAM_BOT_TOKEN'),
        telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID'),
        min_ev=8.0,
        min_confidence=70.0,
        update_interval=300
    )
    print("✅ Sistema inizializzato")
    print()
except Exception as e:
    print(f"❌ Errore inizializzazione: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verifica API-Football provider
if automation.api_manager:
    api_football_provider = automation.api_manager.providers.get("api-football")
    if api_football_provider and api_football_provider.api_key:
        print("✅ API-Football provider disponibile e configurato")
    else:
        print("⚠️  API-Football provider non disponibile")
else:
    print("⚠️  API Manager non disponibile")

print()

# Test recupero partite
print("📡 Test recupero partite...")
try:
    matches = automation._get_matches_to_monitor()
    print(f"✅ Trovate {len(matches)} partite totali")
    
    live_matches = [m for m in matches if m.get('is_live')]
    print(f"🎯 Partite marcate come LIVE: {len(live_matches)}")
    print()
    
    if live_matches:
        print("📊 Partite LIVE trovate:")
        for i, match in enumerate(live_matches[:5], 1):
            home = match.get('home', '?')
            away = match.get('away', '?')
            print(f"   {i}. {home} vs {away}")
        print()
        
        # Test dati live
        print("🔍 Test recupero dati live da API-SPORTS...")
        live_data_cache = automation._get_all_live_data()
        
        if live_data_cache:
            print(f"✅ Dati live ottenuti per {len(live_data_cache)//2} partite")
            print()
            
            # Test analisi per prima partita live
            if live_matches:
                test_match = live_matches[0]
                home = test_match.get('home', '?')
                away = test_match.get('away', '?')
                
                print(f"🧪 Test analisi partita: {home} vs {away}")
                
                # Cerca dati live
                live_data = None
                match_id = test_match.get('id')
                if match_id and match_id in live_data_cache:
                    live_data = live_data_cache[match_id]
                else:
                    cache_key = f"{home.lower().strip()}_{away.lower().strip()}"
                    if cache_key in live_data_cache:
                        live_data = live_data_cache[cache_key]
                
                if live_data:
                    score_home = live_data.get('score_home', 0)
                    score_away = live_data.get('score_away', 0)
                    minute = live_data.get('minute', 0)
                    is_estimated = live_data.get('estimated', False)
                    
                    print(f"   ✅ Dati live ottenuti:")
                    print(f"      Score: {score_home}-{score_away} al {minute}'")
                    print(f"      Tipo: {'STIMATI' if is_estimated else 'REALI (API-SPORTS)'}")
                    print()
                    
                    # Test analisi opportunità
                    if automation.live_betting_advisor:
                        print(f"   🔍 Analizzando opportunità...")
                        try:
                            opportunities = automation.live_betting_advisor.analyze_live_match(
                                match_id=match_id,
                                match_data=test_match,
                                live_data=live_data
                            )
                            print(f"   ✅ Trovate {len(opportunities)} opportunità")
                            
                            for opp in opportunities[:3]:
                                print(f"      - {opp.situation}: {opp.confidence:.0f}% confidence")
                        except Exception as e:
                            print(f"   ⚠️  Errore analisi: {e}")
                else:
                    print(f"   ⚠️  Dati live non trovati per questa partita")
        else:
            print("⚠️  Nessun dato live ottenuto")
    else:
        print("ℹ️  Nessuna partita marcata come live in questo momento")
        print("   (Le partite potrebbero essere pre-match)")
    
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("✅ Test completato!")
print("=" * 70)
print()
print("💡 Il sistema è pronto. Riavvia il servizio per applicare tutto.")
print()



