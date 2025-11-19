#!/usr/bin/env python3
"""
Verifica Partite Live e Analisi
================================
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from automation_24h import Automation24H

print("=" * 70)
print("🔍 VERIFICA PARTITE LIVE E ANALISI")
print("=" * 70)

telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

auto = Automation24H(
    telegram_token=telegram_token,
    telegram_chat_id=telegram_chat_id
)

print(f"\n📊 SOGLIE ATTUALE:")
print(f"   Min EV: {auto.min_ev}%")
print(f"   Min Confidence: {auto.min_confidence}%")

print(f"\n🔍 Recuperando partite...")
matches = auto._get_matches_to_monitor()

print(f"\n✅ Trovate {len(matches)} partite")

if not matches:
    print("⚠️  Nessuna partita trovata!")
    sys.exit(0)

# Analizza prime 5 partite
print(f"\n📋 ANALISI PRIME 5 PARTITE:")
print("-" * 70)

for i, match in enumerate(matches[:5], 1):
    home = match.get('home', 'N/A')
    away = match.get('away', 'N/A')
    is_live = match.get('is_live', False)
    status = "🔴 LIVE" if is_live else "⏰ PRE-MATCH"
    
    print(f"\n{i}. {home} vs {away} - {status}")
    print(f"   Analizzando...")
    
    try:
        opportunity = auto._analyze_match(match)
        
        if opportunity:
            ai_result = opportunity.get('ai_result', {})
            summary = ai_result.get('summary', {})
            ev = summary.get('expected_value', 0)
            conf = summary.get('confidence', 0)
            
            print(f"   ✅ Opportunità trovata!")
            print(f"      EV: {ev:.2f}% (min richiesto: {auto.min_ev}%)")
            print(f"      Confidence: {conf:.2f}% (min richiesto: {auto.min_confidence}%)")
            
            if ev >= auto.min_ev and conf >= auto.min_confidence:
                print(f"   🎯 SOGLIE SUPERATE - DOVREBBE ESSERE NOTIFICATO!")
            else:
                print(f"   ⚠️  Soglie NON superate:")
                if ev < auto.min_ev:
                    print(f"      ❌ EV troppo basso: {ev:.2f}% < {auto.min_ev}%")
                if conf < auto.min_confidence:
                    print(f"      ❌ Confidence troppo bassa: {conf:.2f}% < {auto.min_confidence}%")
        else:
            print(f"   ⚠️  Nessuna opportunità trovata")
            
    except Exception as e:
        print(f"   ❌ Errore analisi: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("💡 SUGGERIMENTO:")
print("=" * 70)
print("Se le soglie sono troppo alte, prova ad abbassarle temporaneamente")
print("per vedere se il sistema trova opportunità.")
print("=" * 70)

