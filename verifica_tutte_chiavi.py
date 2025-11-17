#!/usr/bin/env python3
"""
Script per verificare tutte le chiavi API
"""

import os
from dotenv import load_dotenv

# Carica .env se esiste
load_dotenv()

print("=" * 70)
print("VERIFICA CHIAVI API - STATO ATTUALE")
print("=" * 70)
print()

# Lista tutte le chiavi
chiavi = {
    "THEODDS_API_KEY": {
        "priorita": "ALTA",
        "uso": "Partite reali e quote",
        "dove": "https://the-odds-api.com/",
        "costo": "GRATIS (500/mese)"
    },
    "TELEGRAM_BOT_TOKEN": {
        "priorita": "ALTA",
        "uso": "Notifiche Telegram",
        "dove": "https://t.me/BotFather",
        "costo": "GRATIS",
        "nota": "Attualmente usa default hardcoded"
    },
    "TELEGRAM_CHAT_ID": {
        "priorita": "ALTA",
        "uso": "ID chat Telegram",
        "dove": "https://t.me/userinfobot",
        "costo": "GRATIS",
        "nota": "Attualmente usa default hardcoded"
    },
    "API_FOOTBALL_KEY": {
        "priorita": "MEDIA",
        "uso": "Dati partite",
        "dove": "https://www.api-football.com/",
        "costo": "GRATIS (100/giorno)",
        "nota": "Già configurata (hardcoded)"
    },
    "FOOTBALL_DATA_KEY": {
        "priorita": "BASSA",
        "uso": "Dati aggiuntivi (opzionale)",
        "dove": "https://www.football-data.org/",
        "costo": "GRATIS (10/minuto)"
    },
    "HUGGINGFACE_API_KEY": {
        "priorita": "BASSA",
        "uso": "Sentiment analysis (opzionale)",
        "dove": "https://huggingface.co/settings/tokens",
        "costo": "GRATIS (con limiti)",
        "nota": "Funziona anche senza (free tier)"
    },
    "OPENWEATHER_API_KEY": {
        "priorita": "BASSA",
        "uso": "Dati meteo (opzionale)",
        "dove": "https://openweathermap.org/api",
        "costo": "GRATIS (limitato)",
        "nota": "Già configurata (hardcoded)"
    }
}

# Verifica stato
print("🔴 CHIAVI MANCANTI (DA CONFIGURARE):")
print("-" * 70)
mancanti = []
for chiave, info in chiavi.items():
    valore = os.getenv(chiave, "")
    if not valore:
        if "nota" in info and "hardcoded" in info["nota"]:
            print(f"⚠️  {chiave:25s} - {info['priorita']:6s} - {info['uso']}")
            print(f"    📝 {info['nota']}")
        else:
            print(f"❌ {chiave:25s} - {info['priorita']:6s} - {info['uso']}")
            print(f"    🔗 {info['dove']}")
            mancanti.append(chiave)
    else:
        masked = f"{valore[:10]}..." if len(valore) > 10 else valore
        print(f"✅ {chiave:25s} - Configurata: {masked}")

print()
print("=" * 70)
print("RIEPILOGO")
print("=" * 70)
print(f"✅ Chiavi configurate: {sum(1 for k in chiavi.keys() if os.getenv(k, ''))}/{len(chiavi)}")
print(f"❌ Chiavi mancanti: {len(mancanti)}/{len(chiavi)}")
print()

if mancanti:
    print("📋 CHIAVI DA CONFIGURARE:")
    for k in mancanti:
        info = chiavi[k]
        print(f"   • {k}")
        print(f"     Priorità: {info['priorita']}")
        print(f"     Uso: {info['uso']}")
        print(f"     Dove: {info['dove']}")
        print(f"     Costo: {info['costo']}")
        print()
    
    print("=" * 70)
    print("FORMATO PER FILE .env:")
    print("=" * 70)
    for k in mancanti:
        print(f"{k}=la_tua_chiave_qui")
else:
    print("🎉 Tutte le chiavi sono configurate!")

print("=" * 70)

