#!/usr/bin/env python3
"""
Test Sistema Multi-Fonte per Trovare Partite
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

from multi_source_match_finder import MultiSourceMatchFinder

print("=" * 70)
print("TEST SISTEMA MULTI-FONTE - TROVA PARTITE")
print("=" * 70)
print()

finder = MultiSourceMatchFinder()

# Verifica configurazione
print("📋 Configurazione Fonti:")
print(f"   TheOddsAPI: {'✅' if finder.theodds_key else '❌'}")
print(f"   API-SPORTS: {'✅' if finder.api_sports_key else '❌'}")
print(f"   Football-Data.org: {'✅' if finder.football_data_key else '❌'}")
print()

# Test 1: Trova tutte le partite
print("🔍 TEST 1: Trova tutte le partite (oggi)")
print("-" * 70)
try:
    matches = finder.find_all_matches(
        days_ahead=1,
        include_minor_leagues=True,
        countries=None
    )
    
    print(f"✅ Trovate {len(matches)} partite totali")
    print()
    
    if matches:
        # Raggruppa per fonte
        by_source = {}
        for match in matches:
            source = match.get('source', 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(match)
        
        print("📊 Partite per fonte:")
        for source, source_matches in by_source.items():
            print(f"   {source}: {len(source_matches)} partite")
        print()
        
        # Mostra prime 10 partite
        print("📋 Prime 10 partite trovate:")
        for i, match in enumerate(matches[:10], 1):
            home = match.get('home', '?')
            away = match.get('away', '?')
            league = match.get('league', '?')
            country = match.get('country', '')
            date = match.get('date', '?')
            source = match.get('source', '?')
            
            if isinstance(date, datetime):
                date_str = date.strftime('%H:%M')
            else:
                date_str = str(date)
            
            country_str = f" ({country})" if country else ""
            print(f"   {i}. {home} vs {away}")
            print(f"      📍 {league}{country_str}")
            print(f"      ⏰ {date_str} | Fonte: {source}")
            print()
        
        # Statistiche
        live_count = sum(1 for m in matches if m.get('is_live'))
        print(f"📊 Statistiche:")
        print(f"   Totale partite: {len(matches)}")
        print(f"   Partite live: {live_count}")
        print(f"   Partite pre-match: {len(matches) - live_count}")
    else:
        print("⚠️  Nessuna partita trovata")
        
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("🔍 TEST 2: Leghe disponibili")
print("-" * 70)
try:
    leagues = finder.get_leagues_available()
    
    for source, league_list in leagues.items():
        if league_list:
            print(f"\n📊 {source.upper()}: {len(league_list)} competizioni")
            # Mostra prime 10
            for league in league_list[:10]:
                country = league.get('country', '')
                country_str = f" ({country})" if country else ""
                print(f"   - {league.get('name', '?')}{country_str}")
        else:
            print(f"\n⚠️  {source.upper()}: Nessuna competizione trovata (chiave non configurata?)")
except Exception as e:
    print(f"❌ Errore: {e}")

print()
print("=" * 70)
print("✅ Test completato!")
print("=" * 70)



