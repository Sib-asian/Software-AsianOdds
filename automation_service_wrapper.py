#!/usr/bin/env python3
"""
Wrapper per servizio Windows 24/7
Gestisce restart automatico e logging robusto
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime

# Setup logging robusto
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"automation_service_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import automation
sys.path.insert(0, str(Path(__file__).parent))
from automation_24h import Automation24H
from dotenv import load_dotenv

# Carica variabili ambiente
load_dotenv()

class AutomationService:
    """Wrapper servizio con auto-restart"""
    
    def __init__(self):
        self.max_restarts = 10  # Max restart per ora
        self.restart_delay = 60  # Secondi tra restart
        self.restart_count = 0
        self.last_restart_time = time.time()
        self.automation = None
        
    def run(self):
        """Esegue servizio con auto-restart"""
        logger.info("=" * 60)
        logger.info("🚀 AVVIO AUTOMAZIONE 24/7 - SERVIZIO WINDOWS")
        logger.info("=" * 60)
        
        while True:
            try:
                # Reset restart count se passata un'ora
                if time.time() - self.last_restart_time > 3600:
                    self.restart_count = 0
                
                # Check max restart
                if self.restart_count >= self.max_restarts:
                    logger.error(f"❌ TROPPI RESTART ({self.restart_count})! Attendo 1 ora...")
                    time.sleep(3600)
                    self.restart_count = 0
                    continue
                
                # Inizializza automation
                logger.info("📦 Inizializzazione sistema...")
                self._init_automation()
                
                # Avvia automation (blocca fino a crash/stop)
                logger.info("▶️  Avvio ciclo automazione...")
                self.automation.start()
                
                # Se arriva qui, automation si è fermato normalmente
                logger.info("⏸️  Automation fermato normalmente")
                break
                
            except KeyboardInterrupt:
                logger.info("🛑 Interruzione manuale ricevuta")
                break
                
            except Exception as e:
                self.restart_count += 1
                self.last_restart_time = time.time()
                
                logger.error("=" * 60)
                logger.error(f"❌ ERRORE CRITICO (Restart #{self.restart_count}/{self.max_restarts})")
                logger.error(f"   Errore: {str(e)}")
                logger.error(f"   Traceback:\n{traceback.format_exc()}")
                logger.error("=" * 60)
                
                # Cleanup
                if self.automation:
                    try:
                        self.automation.stop()
                    except:
                        pass
                    self.automation = None
                
                # Attendi prima di restart
                if self.restart_count < self.max_restarts:
                    logger.info(f"⏳ Attendo {self.restart_delay}s prima di riavviare...")
                    time.sleep(self.restart_delay)
                else:
                    logger.error("❌ Raggiunto limite restart, attendo 1 ora...")
                    time.sleep(3600)
                    self.restart_count = 0
    
    def _init_automation(self):
        """Inizializza sistema automation"""
        
        # Config da variabili ambiente
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Fallback ai valori hardcoded se non configurati
        if not telegram_token:
            telegram_token = "8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g"
            logger.warning("⚠️  TELEGRAM_BOT_TOKEN non configurato, uso valore di default")
        
        if not telegram_chat_id:
            telegram_chat_id = "-1003278011521"
            logger.warning("⚠️  TELEGRAM_CHAT_ID non configurato, uso valore di default")
        
        min_ev = float(os.getenv('AUTOMATION_MIN_EV', '8.0'))
        min_confidence = float(os.getenv('AUTOMATION_MIN_CONFIDENCE', '70.0'))
        update_interval = int(os.getenv('AUTOMATION_UPDATE_INTERVAL', '300'))
        
        logger.info(f"✅ Configurazione:")
        logger.info(f"   Telegram Token: {'***' + telegram_token[-10:] if len(telegram_token) > 10 else '***'}")
        logger.info(f"   Telegram Chat ID: {telegram_chat_id}")
        logger.info(f"   Min EV: {min_ev}%")
        logger.info(f"   Min Confidence: {min_confidence}%")
        logger.info(f"   Update Interval: {update_interval}s")
        
        # Crea automation
        self.automation = Automation24H(
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
            min_ev=min_ev,
            min_confidence=min_confidence,
            update_interval=update_interval
        )
        
        logger.info("✅ Automation inizializzato correttamente")


def main():
    """Entry point"""
    service = AutomationService()
    service.run()


if __name__ == '__main__':
    main()

