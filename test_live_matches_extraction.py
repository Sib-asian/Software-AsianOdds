"""
Test script per verificare estrazione partite LIVE con statistiche e quote
"""
import os
import sys
import logging
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import automation
try:
    from automation_24h import Automation24H
except ImportError as e:
    logger.error(f"Errore import: {e}")
    sys.exit(1)

def test_live_matches_extraction():
    """Test estrazione partite LIVE"""
    logger.info("=" * 80)
    logger.info("TEST: Estrazione partite LIVE con statistiche e quote")
    logger.info("=" * 80)
    
    # Inizializza automation
    try:
        automation = Automation24H(
            telegram_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID')
        )
        logger.info("✅ Automation24H inizializzato")
    except Exception as e:
        logger.error(f"❌ Errore inizializzazione: {e}")
        return False
    
    # Test estrazione partite
    try:
        logger.info("\n📡 Test: Recupero partite LIVE...")
        matches = automation._fetch_matches_with_odds_from_api_football()
        
        logger.info(f"\n📊 Risultati:")
        logger.info(f"   - Partite LIVE trovate: {len(matches)}")
        
        if len(matches) > 0:
            logger.info(f"\n✅ SUCCESS: Trovate {len(matches)} partite LIVE con statistiche e quote")
            
            # Mostra dettagli prime 3 partite
            for i, match in enumerate(matches[:3], 1):
                logger.info(f"\n   Partita {i}:")
                logger.info(f"      {match.get('home')} vs {match.get('away')}")
                logger.info(f"      League: {match.get('league')}")
                logger.info(f"      Status: {match.get('status')}")
                logger.info(f"      Score: {match.get('score_home')}-{match.get('score_away')} (min {match.get('minute')})")
                logger.info(f"      Quote 1X2: {match.get('odds_1')}/{match.get('odds_x')}/{match.get('odds_2')}")
                
                # Verifica statistiche
                has_stats = bool(match.get('home_total_shots') or match.get('home_shots_on_target'))
                logger.info(f"      Ha statistiche: {has_stats}")
                
                # Verifica quote
                has_odds = bool(match.get('odds_1') or match.get('odds_x') or match.get('odds_2'))
                logger.info(f"      Ha quote: {has_odds}")
            
            return True
        else:
            logger.warning(f"\n⚠️  Nessuna partita LIVE trovata con statistiche e quote")
            logger.info(f"   Questo può essere normale se non ci sono partite LIVE in questo momento")
            return False
            
    except Exception as e:
        logger.error(f"❌ Errore durante test: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_live_matches_extraction()
    sys.exit(0 if success else 1)

