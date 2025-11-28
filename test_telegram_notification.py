#!/usr/bin/env python3
"""
Test Telegram Notification
==========================
Invia una notifica di prova per verificare che Telegram sia configurato correttamente.
"""

import json
import requests
import sys
from pathlib import Path

def test_telegram_notification():
    """Invia una notifica di prova su Telegram"""
    
    # Carica config.json
    config_path = Path("config.json")
    if not config_path.exists():
        print("❌ File config.json non trovato!")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    bot_token = config.get('telegram_token', '')
    chat_id = config.get('telegram_chat_id', '')
    
    if not bot_token or bot_token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("❌ Telegram Bot Token non configurato in config.json!")
        return False
    
    if not chat_id or chat_id == "YOUR_TELEGRAM_CHAT_ID_HERE":
        print("❌ Telegram Chat ID non configurato in config.json!")
        return False
    
    # Prepara messaggio di prova
    message = """✅ **NOTIFICA DI PROVA**

🔧 Sistema Live Betting configurato correttamente!

📊 Configurazione:
• Min EV: 8%
• Min Confidence: 70%
• Update Interval: 13 minuti
• Max Notifiche: 2 per ciclo

🎯 Il sistema è pronto per inviare notifiche quando trova opportunità valide!"""
    
    # Invia messaggio
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    # Prova prima con Chat ID normale, poi con negativo (per gruppi)
    chat_ids_to_try = [chat_id, f"-{chat_id}"]
    
    try:
        print(f"📱 Invio notifica di prova a Telegram...")
        print(f"   Bot Token: {bot_token[:20]}...")
        print(f"   Chat ID: {chat_id}")
        
        success = False
        for test_chat_id in chat_ids_to_try:
            data = {
                "chat_id": test_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            print(f"   Provo con Chat ID: {test_chat_id}...")
            response = requests.post(url, json=data, timeout=10)
        
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    print(f"✅ Notifica inviata con successo con Chat ID: {test_chat_id}!")
                    print(f"   Messaggio ID: {result.get('result', {}).get('message_id')}")
                    if test_chat_id != chat_id:
                        print(f"⚠️  NOTA: Il Chat ID corretto è {test_chat_id}, non {chat_id}")
                        print(f"   Aggiorna config.json con questo valore!")
                    success = True
                    break
                else:
                    error_desc = result.get('description', 'Unknown error')
                    if "chat not found" in error_desc.lower():
                        print(f"   ❌ Chat ID {test_chat_id} non trovato, provo alternativo...")
                        continue
                    else:
                        print(f"❌ Errore Telegram API: {error_desc}")
                        return False
            else:
                if response.status_code == 400:
                    print(f"   ❌ Chat ID {test_chat_id} non valido, provo alternativo...")
                    continue
                else:
                    print(f"❌ Errore HTTP {response.status_code}: {response.text}")
                    return False
        
        if not success:
            print("❌ Impossibile inviare notifica con nessun Chat ID provato")
            print("💡 Assicurati di:")
            print("   1. Aver inviato /start al bot su Telegram")
            print("   2. Che il Chat ID sia corretto")
            return False
        
        return True
            
    except requests.exceptions.Timeout:
        print("❌ Timeout: Impossibile connettersi a Telegram API")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore di connessione: {e}")
        return False
    except Exception as e:
        print(f"❌ Errore imprevisto: {e}")
        return False

if __name__ == "__main__":
    success = test_telegram_notification()
    sys.exit(0 if success else 1)
