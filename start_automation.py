#!/usr/bin/env python3
"""
Script per avviare sistema automazione 24/7

⚠️  AUTOMAZIONE DISABILITATA - Non eseguire questo script!
"""

# 🛑 AUTOMAZIONE DISABILITATA - Decommenta per riabilitare
import sys
print("🛑 AUTOMAZIONE DISABILITATA - start_automation.py terminato")
print("   Per riabilitare, rimuovi le righe 'sys.exit(0)' da start_automation.py")
sys.exit(0)

import os
from pathlib import Path
from dotenv import load_dotenv

# Carica variabili ambiente
load_dotenv()

# Import automation
sys.path.insert(0, str(Path(__file__).parent))
from automation_24h import Automation24H

def main():
    """Avvia sistema automazione"""
    
    # Config da variabili ambiente o valori hardcoded come fallback
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    # Fallback ai valori hardcoded se non configurati
    if not telegram_token:
        telegram_token = "8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g"
        print("⚠️  TELEGRAM_BOT_TOKEN non configurato, uso valore di default")
    
    if not telegram_chat_id:
        telegram_chat_id = "-1003278011521"
        print("⚠️  TELEGRAM_CHAT_ID non configurato, uso valore di default")
    
    min_ev = float(os.getenv('AUTOMATION_MIN_EV', '8.0'))
    min_confidence = float(os.getenv('AUTOMATION_MIN_CONFIDENCE', '70.0'))
    update_interval = int(os.getenv('AUTOMATION_UPDATE_INTERVAL', '1200'))
    
    print(f"✅ Telegram configurato:")
    print(f"   Token: {'***' + telegram_token[-10:] if len(telegram_token) > 10 else '***'}")
    print(f"   Chat ID: {telegram_chat_id}")
    
    # Crea e avvia sistema
    automation = Automation24H(
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        min_ev=min_ev,
        min_confidence=min_confidence,
        update_interval=update_interval
    )
    
    print("🚀 Avvio sistema automazione 24/7...")
    automation.start()

if __name__ == '__main__':
    main()

