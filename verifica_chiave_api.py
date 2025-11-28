#!/usr/bin/env python3
"""Script per verificare la chiave API-SPORTS configurata"""

import os
import sys

# Carica la chiave come fa il sistema
api_key_env = os.getenv("API_FOOTBALL_KEY", "")
default_key = "95c43f936816cd4389a747fd2cfe061a"

# Determina quale chiave viene usata
if api_key_env:
    key_used = api_key_env
    source = "variabile d'ambiente API_FOOTBALL_KEY"
else:
    key_used = default_key
    source = "valore default hardcoded nel codice"

print("=" * 80)
print("🔑 VERIFICA CHIAVE API-SPORTS")
print("=" * 80)
print()
print(f"📋 Chiave attualmente configurata:")
print(f"   {key_used}")
print()
print(f"📂 Fonte: {source}")
print()
print(f"📏 Lunghezza: {len(key_used)} caratteri")
print()
print("=" * 80)
print("⚠️  NOTA")
print("=" * 80)
print()
print("La chiave viene caricata in questo ordine:")
print("1. Variabile d'ambiente API_FOOTBALL_KEY (se configurata)")
print("2. Valore default hardcoded nel codice (se variabile non configurata)")
print()
print("Per configurare la tua chiave Pro, crea/modifica il file .env:")
print("   API_FOOTBALL_KEY=la_tua_chiave_pro_qui")
print()
print("=" * 80)







