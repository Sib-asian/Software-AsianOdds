#!/usr/bin/env python3
"""
Trova Chat ID di un Canale Telegram
====================================
Questo script ti aiuta a trovare il Chat ID corretto del tuo canale Telegram.
"""

import json
import requests
import sys
from pathlib import Path

def trova_chat_id_canale():
    """Trova il Chat ID del canale usando il bot token"""
    
    # Carica config.json
    config_path = Path("config.json")
    if not config_path.exists():
        print("❌ File config.json non trovato!")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    bot_token = config.get('telegram_token', '')
    
    if not bot_token or bot_token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("❌ Telegram Bot Token non configurato in config.json!")
        return None
    
    print("🔍 Cercando Chat ID del canale...")
    print(f"   Bot Token: {bot_token[:20]}...")
    print()
    print("📋 ISTRUZIONI:")
    print("   1. Aggiungi il bot al canale come amministratore")
    print("   2. Invia un messaggio qualsiasi nel canale")
    print("   3. Poi premi INVIO qui per continuare...")
    print()
    input("Premi INVIO quando hai inviato un messaggio nel canale...")
    
    # Recupera gli aggiornamenti
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    
    try:
        print("\n📡 Recupero aggiornamenti...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                updates = result.get('result', [])
                
                if not updates:
                    print("❌ Nessun aggiornamento trovato!")
                    print("💡 Assicurati di:")
                    print("   1. Aver aggiunto il bot al canale come amministratore")
                    print("   2. Aver inviato un messaggio nel canale")
                    return None
                
                print(f"✅ Trovati {len(updates)} aggiornamenti")
                print()
                
                # Cerca chat ID di canali (iniziano con -100)
                channel_ids = set()
                for update in updates:
                    message = update.get('message') or update.get('channel_post')
                    if message:
                        chat = message.get('chat', {})
                        chat_id = chat.get('id')
                        chat_type = chat.get('type')
                        chat_title = chat.get('title', 'N/A')
                        
                        if chat_type == 'channel':
                            channel_ids.add((chat_id, chat_title))
                
                if channel_ids:
                    print("📢 CANALI TROVATI:")
                    for chat_id, title in channel_ids:
                        print(f"   • {title}")
                        print(f"     Chat ID: {chat_id}")
                        print()
                    
                    # Se c'è un solo canale, suggerisci di usarlo
                    if len(channel_ids) == 1:
                        chat_id, title = list(channel_ids)[0]
                        print(f"✅ Chat ID del canale '{title}': {chat_id}")
                        return chat_id
                    else:
                        print("⚠️  Trovati più canali. Scegli quello corretto.")
                        return None
                else:
                    print("❌ Nessun canale trovato negli aggiornamenti")
                    print("💡 Assicurati che:")
                    print("   1. Il bot sia amministratore del canale")
                    print("   2. Tu abbia inviato un messaggio nel canale")
                    return None
            else:
                print(f"❌ Errore Telegram API: {result.get('description', 'Unknown error')}")
                return None
        else:
            print(f"❌ Errore HTTP {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Errore: {e}")
        return None

def aggiorna_config(chat_id):
    """Aggiorna config.json con il Chat ID corretto"""
    config_path = Path("config.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['telegram_chat_id'] = str(chat_id)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Config.json aggiornato con Chat ID: {chat_id}")

if __name__ == "__main__":
    chat_id = trova_chat_id_canale()
    
    if chat_id:
        print(f"\n💡 Vuoi aggiornare automaticamente config.json? (s/n): ", end='')
        risposta = input().strip().lower()
        
        if risposta == 's' or risposta == 'si' or risposta == 'y' or risposta == 'yes':
            aggiorna_config(chat_id)
            print("\n✅ Config.json aggiornato!")
            print("   Ora puoi testare la notifica con: python test_telegram_notification.py")
        else:
            print(f"\n📋 Chat ID del canale: {chat_id}")
            print("   Aggiorna manualmente config.json con questo valore")
    else:
        print("\n❌ Impossibile trovare il Chat ID del canale")
        sys.exit(1)

