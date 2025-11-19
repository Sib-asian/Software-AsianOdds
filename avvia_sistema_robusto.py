#!/usr/bin/env python3
"""
Script robusto per avviare il sistema 24/7 con gestione errori e logging
"""
import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"automation_service_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("🚀 AVVIO SISTEMA 24/7 - VERSIONE ROBUSTA")
logger.info("=" * 80)

# Import con gestione errori
try:
    logger.info("📦 Import automation_24h...")
    sys.path.insert(0, str(Path(__file__).parent))
    from automation_24h import Automation24H
    logger.info("✅ automation_24h importato con successo")
except Exception as e:
    logger.error(f"❌ Errore import automation_24h: {e}", exc_info=True)
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ dotenv caricato")
except Exception as e:
    logger.warning(f"⚠️  Errore caricamento dotenv: {e}")

# Configurazione
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', "8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g")
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', "-1003278011521")
update_interval = int(os.getenv('AUTOMATION_UPDATE_INTERVAL', '300'))

logger.info(f"📊 Configurazione:")
logger.info(f"   Update Interval: {update_interval}s (5 minuti)")
logger.info(f"   Telegram Token: {'***' + telegram_token[-10:] if len(telegram_token) > 10 else '***'}")
logger.info(f"   Telegram Chat ID: {telegram_chat_id}")

# Inizializza sistema
try:
    logger.info("🔧 Inizializzazione sistema...")
    automation = Automation24H(
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        update_interval=update_interval,
        min_ev=8.0,
        min_confidence=72.0
    )
    logger.info("✅ Sistema inizializzato con successo")
except Exception as e:
    logger.error(f"❌ Errore inizializzazione sistema: {e}", exc_info=True)
    sys.exit(1)

# Test Telegram
try:
    logger.info("🧪 Test invio messaggio Telegram...")
    import requests
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    data = {
        "chat_id": telegram_chat_id,
        "text": "🚀 Sistema 24/7 avviato con successo! Il sistema è ora attivo e monitorerà le partite live.",
        "parse_mode": "HTML"
    }
    response = requests.post(url, json=data, timeout=10)
    if response.status_code == 200 and response.json().get("ok"):
        logger.info("✅ Messaggio Telegram di test inviato con successo")
    else:
        logger.warning(f"⚠️  Errore invio messaggio Telegram: {response.text}")
except Exception as e:
    logger.warning(f"⚠️  Errore test Telegram: {e}")

# Avvia sistema
try:
    logger.info("▶️  Avvio ciclo automazione...")
    logger.info("=" * 80)
    logger.info("💡 Il sistema è ora ATTIVO e funzionante!")
    logger.info("📊 Monitoraggio partite live ogni 5 minuti")
    logger.info("📱 Messaggi Telegram verranno inviati immediatamente quando troverà opportunità")
    logger.info("   (Confidence >= 72%, EV >= 8%)")
    logger.info("=" * 80)
    automation.start()
except KeyboardInterrupt:
    logger.info("🛑 Interruzione manuale ricevuta")
except Exception as e:
    logger.error(f"❌ Errore durante esecuzione: {e}", exc_info=True)
    import traceback
    logger.error(traceback.format_exc())
finally:
    logger.info("⏸️  Sistema fermato")

