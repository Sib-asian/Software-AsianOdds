#!/usr/bin/env python3
"""
Test completo del sistema per vedere se trova partite live
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

from automation_24h import Automation24H
import logging

# Setup logging dettagliato
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("TEST COMPLETO SISTEMA - PARTITE LIVE")
print("=" * 70)
print()

telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

print("📋 Configurazione:")
print(f"   Telegram Token: {'✅' if telegram_token else '❌'}")
print(f"   Telegram Chat ID: {'✅' if telegram_chat_id else '❌'}")
print(f"   THEODDS_API_KEY: {'✅' if os.getenv('THEODDS_API_KEY') else '❌'}")
print(f"   API_FOOTBALL_KEY: {'✅' if os.getenv('API_FOOTBALL_KEY') else '❌'}")
print()

if not os.getenv('API_FOOTBALL_KEY'):
    print("⚠️  ATTENZIONE: API_FOOTBALL_KEY non configurata!")
    print("   Senza questa chiave, il sistema NON può ottenere dati live reali")
    print("   (score, minuto, statistiche) dalle partite.")
    print("   Il sistema può solo identificare partite potenzialmente live")
    print("   basandosi sull'orario di inizio, ma non può analizzarle correttamente.")
    print()

print("🚀 Inizializzando sistema...")
print()

try:
    automation = Automation24H(
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        min_ev=8.0,
        min_confidence=70.0,
        update_interval=300
    )
    
    print("✅ Sistema inizializzato")
    print()
    
    # Ottieni partite
    print("📡 Recuperando partite...")
    matches = automation._get_matches_to_monitor()
    
    print(f"✅ Trovate {len(matches)} partite totali")
    print()
    
    # Filtra partite live
    live_matches = [m for m in matches if m.get('is_live')]
    print(f"🎯 Partite marcate come LIVE: {len(live_matches)}")
    print()
    
    if live_matches:
        print("📊 PARTITE LIVE TROVATE:")
        print("-" * 70)
        for i, match in enumerate(live_matches, 1):
            home = match.get('home', '?')
            away = match.get('away', '?')
            match_id = match.get('id', '?')
            date = match.get('date', '?')
            league = match.get('league', '?')
            
            if isinstance(date, datetime):
                date_str = date.strftime('%H:%M')
            else:
                date_str = str(date)
            
            print(f"{i}. {home} vs {away}")
            print(f"   📍 {league}")
            print(f"   🆔 ID: {match_id}")
            print(f"   ⏰ Inizio: {date_str}")
            print()
        
        print("=" * 70)
        print("🔍 Ora testiamo se il sistema può ottenere dati live...")
        print("=" * 70)
        print()
        
        # Test per ogni partita live
        for match in live_matches[:3]:  # Prime 3
            home = match.get('home', '?')
            away = match.get('away', '?')
            print(f"🔍 Testando: {home} vs {away}")
            
            # Prova a ottenere dati live
            live_data = automation._get_live_match_data(match)
            
            if live_data:
                score_home = live_data.get('score_home', 0)
                score_away = live_data.get('score_away', 0)
                minute = live_data.get('minute', 0)
                status = live_data.get('status', 'Live')
                
                print(f"   ✅ Dati live ottenuti!")
                print(f"   ⚽ Score: {score_home}-{score_away} al {minute}'")
                print(f"   📊 Status: {status}")
                
                # Test analisi
                print(f"   🔍 Analizzando opportunità...")
                try:
                    opportunities = automation.live_betting_advisor.analyze_live_match(
                        match_id=match.get('id'),
                        match_data=match,
                        live_data=live_data
                    )
                    print(f"   📊 Trovate {len(opportunities)} opportunità")
                    
                    for opp in opportunities[:3]:  # Prime 3
                        print(f"      - {opp.situation}: {opp.confidence:.0f}% confidence")
                except Exception as e:
                    print(f"   ❌ Errore analisi: {e}")
            else:
                print(f"   ⚠️  Nessun dato live disponibile")
                print(f"      (La partita potrebbe non essere realmente in corso")
                print(f"       o non essere trovata in API-Football)")
            print()
    else:
        print("ℹ️  Nessuna partita marcata come live in questo momento")
        print()
        print("💡 Possibili motivi:")
        print("   - Le partite non sono ancora iniziate (sono pre-match)")
        print("   - Le partite sono già finite")
        print("   - Il sistema non le ha ancora rilevate")
        print()
        print("📊 Partite pre-match trovate:")
        prematch = [m for m in matches if not m.get('is_live')]
        for i, match in enumerate(prematch[:5], 1):
            home = match.get('home', '?')
            away = match.get('away', '?')
            date = match.get('date', '?')
            if isinstance(date, datetime):
                date_str = date.strftime('%H:%M')
            else:
                date_str = str(date)
            print(f"   {i}. {home} vs {away} - {date_str}")
    
    print()
    print("=" * 70)
    print("✅ Test completato!")
    print("=" * 70)
    
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()

