#!/usr/bin/env python3
"""
Verifica Date Partite
=====================

Verifica se il sistema sta recuperando partite già giocate.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from automation_24h import Automation24H

print("=" * 70)
print("🔍 VERIFICA DATE PARTITE")
print("=" * 70)

now = datetime.now()
print(f"\n📅 ORA ATTUALE: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📅 DATA OGGI: {now.date()}")

telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

auto = Automation24H(
    telegram_token=telegram_token,
    telegram_chat_id=telegram_chat_id
)

print(f"\n🔍 Recuperando partite...")
matches = auto._get_matches_to_monitor()

print(f"\n✅ Trovate {len(matches)} partite")
print(f"\n📋 ANALISI DATE:")
print("-" * 70)

partite_oggi = []
partite_ieri = []
partite_future = []
partite_passate = []

for match in matches:
    date = match.get('date')
    if isinstance(date, str):
        try:
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            date = date.replace(tzinfo=None)
        except:
            date = None
    
    if not date:
        continue
    
    home = match.get('home', 'N/A')
    away = match.get('away', 'N/A')
    is_live = match.get('is_live', False)
    
    # Calcola differenza
    diff = date - now
    hours_diff = diff.total_seconds() / 3600
    
    match_info = {
        'home': home,
        'away': away,
        'date': date,
        'hours_diff': hours_diff,
        'is_live': is_live
    }
    
    if date.date() == now.date():
        partite_oggi.append(match_info)
    elif date.date() == (now.date() - timedelta(days=1)):
        partite_ieri.append(match_info)
    elif date.date() > now.date():
        partite_future.append(match_info)
    else:
        partite_passate.append(match_info)

print(f"\n📅 PARTITE OGGI ({now.date()}): {len(partite_oggi)}")
for p in partite_oggi[:5]:
    status = "🔴 LIVE" if p['is_live'] else "⏰ PRE-MATCH"
    print(f"   • {p['home']} vs {p['away']} - {status} - {p['date'].strftime('%H:%M')} ({p['hours_diff']:+.1f}h)")

print(f"\n📅 PARTITE IERI ({now.date() - timedelta(days=1)}): {len(partite_ieri)}")
for p in partite_ieri[:5]:
    status = "🔴 LIVE" if p['is_live'] else "⏰ FINITA"
    print(f"   • {p['home']} vs {p['away']} - {status} - {p['date'].strftime('%Y-%m-%d %H:%M')} ({p['hours_diff']:+.1f}h)")

print(f"\n📅 PARTITE FUTURE: {len(partite_future)}")
for p in partite_future[:5]:
    print(f"   • {p['home']} vs {p['away']} - {p['date'].strftime('%Y-%m-%d %H:%M')} ({p['hours_diff']:+.1f}h)")

print(f"\n📅 PARTITE PASSATE (>1 giorno fa): {len(partite_passate)}")
for p in partite_passate[:5]:
    print(f"   • {p['home']} vs {p['away']} - {p['date'].strftime('%Y-%m-%d %H:%M')} ({p['hours_diff']:+.1f}h)")

print("\n" + "=" * 70)
print("🔍 PROBLEMA IDENTIFICATO:")
print("=" * 70)

if partite_ieri or partite_passate:
    print("❌ Il sistema sta recuperando partite già giocate!")
    print("   Questo è un problema - dovrebbe filtrare solo partite di oggi/future")
else:
    print("✅ Il sistema recupera solo partite di oggi/future")

if len(partite_oggi) == 0:
    print("⚠️  Nessuna partita di oggi trovata!")
    print("   Potrebbe essere un problema con TheOddsAPI o con il filtro date")

print("=" * 70)

