#!/usr/bin/env python3
"""Test invio messaggio Telegram per verificare che funzioni"""

import os
import requests

# Configurazione Telegram
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', "8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g")
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', "-1003278011521")

print("=" * 80)
print("🧪 TEST INVIO MESSAGGIO TELEGRAM")
print("=" * 80)
print()

print(f"📱 Chat ID: {telegram_chat_id}")
print(f"🔑 Token: {'***' + telegram_token[-10:] if len(telegram_token) > 10 else '***'}")
print()

# Test invio messaggio
url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
message = "🧪 Test messaggio dal sistema di betting - Se ricevi questo, Telegram funziona!"

data = {
    "chat_id": telegram_chat_id,
    "text": message,
    "parse_mode": "HTML"
}

print("📤 Invio messaggio di test...")
try:
    response = requests.post(url, json=data, timeout=10)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("ok"):
            print("✅ Messaggio inviato con successo!")
            print(f"   Message ID: {result.get('result', {}).get('message_id', 'N/A')}")
            print()
            print("💡 Controlla Telegram per vedere il messaggio di test")
        else:
            print(f"❌ Errore: {result.get('description', 'Errore sconosciuto')}")
    else:
        print(f"❌ Errore HTTP: {response.status_code}")
        print(f"   Risposta: {response.text[:200]}")
except Exception as e:
    print(f"❌ Errore: {e}")

print()
print("=" * 80)
print("💡 NOTA:")
print("=" * 80)
print()
print("Se il messaggio è stato inviato con successo, significa che:")
print("  ✅ La configurazione Telegram è corretta")
print("  ✅ Il bot può inviare messaggi")
print("  ✅ Il Chat ID è valido")
print()
print("Se il sistema principale non invia messaggi, potrebbe essere:")
print("  ⚠️  Il sistema è bloccato durante l'import")
print("  ⚠️  Non sta trovando opportunità (confidence < 72% o EV < 8%)")
print("  ⚠️  Non ci sono partite LIVE con opportunità valide")
print()
print("=" * 80)







