#!/usr/bin/env python3
"""
Test Diagnostico Notifiche
===========================

Verifica perché non arrivano notifiche.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

print("=" * 70)
print("🔍 DIAGNOSTICA NOTIFICHE TELEGRAM")
print("=" * 70)

# 1. Verifica configurazione Telegram
print("\n1️⃣  VERIFICA CONFIGURAZIONE TELEGRAM")
print("-" * 70)

telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

if not telegram_token:
    print("❌ TELEGRAM_BOT_TOKEN non configurato!")
    sys.exit(1)
else:
    print(f"✅ TELEGRAM_BOT_TOKEN: {telegram_token[:10]}...{telegram_token[-5:]}")

if not telegram_chat_id:
    print("❌ TELEGRAM_CHAT_ID non configurato!")
    sys.exit(1)
else:
    print(f"✅ TELEGRAM_CHAT_ID: {telegram_chat_id}")

# 2. Test invio messaggio diretto
print("\n2️⃣  TEST INVIO MESSAGGIO TELEGRAM")
print("-" * 70)

try:
    from ai_system.telegram_notifier import TelegramNotifier
    
    notifier = TelegramNotifier(
        bot_token=telegram_token,
        chat_id=telegram_chat_id
    )
    
    test_message = f"🧪 Test notifica - {datetime.now().strftime('%H:%M:%S')}"
    result = notifier._send_message(test_message)
    
    if result:
        print("✅ Messaggio di test inviato con successo!")
        print("   Controlla Telegram per verificare")
    else:
        print("❌ Errore invio messaggio di test")
        
except Exception as e:
    print(f"❌ Errore test Telegram: {e}")
    import traceback
    traceback.print_exc()

# 3. Verifica partite disponibili
print("\n3️⃣  VERIFICA PARTITE DISPONIBILI")
print("-" * 70)

try:
    from api_manager import APIManager
    
    api_manager = APIManager()
    
    # Prova a ottenere partite
    print("   Recuperando partite da TheOddsAPI...")
    
    # Usa il metodo interno per ottenere partite
    from automation_24h import Automation24H
    
    auto = Automation24H(
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id
    )
    
    matches = auto._get_matches_to_monitor()
    print(f"   ✅ Trovate {len(matches)} partite")
    
    if matches:
        print("\n   📋 Prime 5 partite:")
        for i, match in enumerate(matches[:5], 1):
            home = match.get('home', 'N/A')
            away = match.get('away', 'N/A')
            match_date = match.get('date', 'N/A')
            is_live = match.get('is_live', False)
            status = "🔴 LIVE" if is_live else "⏰ PRE-MATCH"
            print(f"   {i}. {home} vs {away} - {status} - {match_date}")
    else:
        print("   ⚠️  Nessuna partita trovata")
        
except Exception as e:
    print(f"❌ Errore recupero partite: {e}")
    import traceback
    traceback.print_exc()

# 4. Test analisi partita
print("\n4️⃣  TEST ANALISI PARTITA")
print("-" * 70)

try:
    if matches and len(matches) > 0:
        test_match = matches[0]
        print(f"   Analizzando: {test_match.get('home')} vs {test_match.get('away')}")
        
        opportunity = auto._analyze_match(test_match)
        
        if opportunity:
            ai_result = opportunity.get('ai_result', {})
            summary = ai_result.get('summary', {})
            ev = summary.get('expected_value', 0)
            conf = summary.get('confidence', 0)
            
            print(f"   ✅ Opportunità trovata!")
            print(f"      EV: {ev:.2f}%")
            print(f"      Confidence: {conf:.2f}%")
            print(f"      Min EV richiesto: {auto.min_ev}%")
            print(f"      Min Confidence richiesta: {auto.min_confidence}%")
            
            if ev >= auto.min_ev and conf >= auto.min_confidence:
                print("   ✅ Soglie superate - DOVREBBE essere notificato!")
            else:
                print("   ⚠️  Soglie NON superate - per questo non viene notificato")
                print(f"      EV: {ev:.2f}% < {auto.min_ev}%")
                print(f"      Confidence: {conf:.2f}% < {auto.min_confidence}%")
        else:
            print("   ⚠️  Nessuna opportunità trovata per questa partita")
    else:
        print("   ⚠️  Nessuna partita disponibile per test")
        
except Exception as e:
    print(f"❌ Errore analisi: {e}")
    import traceback
    traceback.print_exc()

# 5. Verifica ciclo completo
print("\n5️⃣  TEST CICLO COMPLETO")
print("-" * 70)

try:
    print("   Eseguendo un ciclo completo...")
    auto._run_cycle()
    print("   ✅ Ciclo completato")
    
except Exception as e:
    print(f"❌ Errore ciclo: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ DIAGNOSTICA COMPLETATA")
print("=" * 70)

