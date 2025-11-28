#!/usr/bin/env python3
"""
Verifica Rapida Sistema Multi-Fonte
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from automation_24h import Automation24H
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

print("=" * 70)
print("VERIFICA SISTEMA MULTI-FONTE")
print("=" * 70)
print()

# Test 1: Verifica inizializzazione
print("TEST 1: Inizializzazione Sistema")
print("-" * 70)
try:
    automation = Automation24H(
        telegram_token=os.getenv('TELEGRAM_BOT_TOKEN'),
        telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID'),
        min_ev=8.0,
        min_confidence=70.0,
        update_interval=300
    )
    
    if hasattr(automation, 'multi_source_finder') and automation.multi_source_finder:
        print("✅ Sistema multi-fonte inizializzato correttamente")
    else:
        print("❌ Sistema multi-fonte NON inizializzato")
        sys.exit(1)
except Exception as e:
    print(f"❌ Errore inizializzazione: {e}")
    sys.exit(1)

print()

# Test 2: Verifica recupero partite
print("TEST 2: Recupero Partite")
print("-" * 70)
try:
    matches = automation._get_matches_to_monitor()
    
    print(f"✅ Trovate {len(matches)} partite totali")
    print()
    
    if matches:
        # Analizza per fonte
        by_source = {}
        for match in matches:
            source = match.get('source', 'unknown')
            if source not in by_source:
                by_source[source] = 0
            by_source[source] += 1
        
        print("📊 Partite per fonte:")
        for source, count in by_source.items():
            print(f"   {source}: {count} partite")
        print()
        
        # Verifica se ci sono partite da multiple fonti
        if len(by_source) > 1:
            print("✅ SUCCESSO: Sistema multi-fonte funziona!")
            print(f"   Trovate partite da {len(by_source)} fonti diverse")
        else:
            print("⚠️  Attenzione: Partite trovate solo da 1 fonte")
            print(f"   Fonte: {list(by_source.keys())[0] if by_source else 'nessuna'}")
        
        # Mostra esempi
        print()
        print("📋 Esempi partite trovate:")
        for i, match in enumerate(matches[:5], 1):
            home = match.get('home', '?')
            away = match.get('away', '?')
            league = match.get('league', '?')
            source = match.get('source', '?')
            print(f"   {i}. {home} vs {away} ({league}) - Fonte: {source}")
    else:
        print("⚠️  Nessuna partita trovata")
        
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("✅ Verifica completata!")
print("=" * 70)








