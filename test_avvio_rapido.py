#!/usr/bin/env python3
"""Test avvio rapido per vedere se il sistema si blocca"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup logging minimo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("🧪 TEST AVVIO RAPIDO")
logger.info("=" * 60)
logger.info(f"⏰ Ora: {datetime.now().strftime('%H:%M:%S')}")

try:
    logger.info("📦 Import automation_24h...")
    from automation_24h import Automation24H
    logger.info("✅ Import OK")
    
    logger.info("📦 Creazione istanza Automation24H...")
    automation = Automation24H(
        telegram_token="8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g",
        telegram_chat_id="-1003278011521",
        min_ev=8.0,
        min_confidence=72.0,
        update_interval=300
    )
    logger.info("✅ Automation24H creato")
    
    logger.info("▶️  Avvio primo ciclo (single_run=True)...")
    automation.start(single_run=True)
    logger.info("✅ Ciclo completato!")
    
except Exception as e:
    logger.error(f"❌ ERRORE: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)

logger.info("=" * 60)
logger.info("✅ TEST COMPLETATO")
logger.info("=" * 60)







