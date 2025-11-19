#!/usr/bin/env python3
"""
Script di Test per Verificare Notifiche Live
============================================

Verifica se il sistema trova partite live e opportunità in questo momento.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import sistema
try:
    from automation_24h import Automation24H
    from api_manager import APIManager
    from live_betting_advisor import LiveBettingAdvisor
    from ai_system.telegram_notifier import TelegramNotifier
    from ai_system.pipeline import AIPipeline
    from ai_system.config import AIConfig
except ImportError as e:
    logger.error(f"❌ Errore import: {e}")
    sys.exit(1)


def test_live_notifications():
    """Test completo delle notifiche live"""
    print("=" * 70)
    print("🧪 TEST NOTIFICHE LIVE - Verifica Sistema")
    print("=" * 70)
    print()
    
    # 1. Verifica configurazione
    print("📋 STEP 1: Verifica Configurazione")
    print("-" * 70)
    
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    api_football_key = os.getenv('API_FOOTBALL_KEY')
    
    print(f"✅ Telegram Token: {'Configurato' if telegram_token else '❌ NON CONFIGURATO'}")
    print(f"✅ Telegram Chat ID: {'Configurato' if telegram_chat_id else '❌ NON CONFIGURATO'}")
    print(f"✅ API-Football Key: {'Configurato' if api_football_key else '❌ NON CONFIGURATO'}")
    print()
    
    if not api_football_key:
        print("⚠️  API-Football non configurata - le partite live non possono essere recuperate")
        print()
    
    # 2. Test API Manager
    print("📡 STEP 2: Test API Manager")
    print("-" * 70)
    try:
        api_manager = APIManager()
        print("✅ API Manager inizializzato")
        
        # Verifica provider API-Football
        api_football_provider = api_manager.providers.get("api-football")
        if api_football_provider and api_football_provider.api_key:
            print("✅ API-Football provider disponibile")
            
            # Test chiamata live
            print("🔍 Test chiamata API-Football per partite live...")
            try:
                live_fixtures = api_football_provider._request("fixtures", {"live": "all"})
                if live_fixtures and live_fixtures.get("response"):
                    live_count = len(live_fixtures["response"])
                    print(f"✅ Trovate {live_count} partite LIVE in questo momento!")
                    
                    # Mostra prime 5 partite
                    if live_count > 0:
                        print("\n📊 Prime partite live trovate:")
                        for i, fixture in enumerate(live_fixtures["response"][:5], 1):
                            teams = fixture.get("teams", {})
                            home = teams.get("home", {}).get("name", "?")
                            away = teams.get("away", {}).get("name", "?")
                            score = fixture.get("goals", {})
                            score_home = score.get("home", 0)
                            score_away = score.get("away", 0)
                            status = fixture.get("fixture", {}).get("status", {})
                            minute = status.get("elapsed", 0)
                            print(f"   {i}. {home} vs {away} - {score_home}-{score_away} al {minute}'")
                else:
                    print("⚠️  Nessuna partita live trovata in questo momento")
            except Exception as e:
                print(f"❌ Errore chiamata API-Football: {e}")
        else:
            print("⚠️  API-Football provider non disponibile (chiave mancante)")
    except Exception as e:
        print(f"❌ Errore inizializzazione API Manager: {e}")
    print()
    
    # 3. Test Live Betting Advisor
    print("🎯 STEP 3: Test Live Betting Advisor")
    print("-" * 70)
    try:
        # Inizializza AI Pipeline (opzionale)
        ai_pipeline = None
        try:
            ai_config = AIConfig()
            ai_config.use_ensemble = False  # Disabilita per test più veloce
            ai_pipeline = AIPipeline(ai_config)
            print("✅ AI Pipeline inizializzata (opzionale)")
        except Exception as e:
            print(f"⚠️  AI Pipeline non disponibile: {e}")
        
        # Inizializza notifier (opzionale per test)
        notifier = None
        if telegram_token and telegram_chat_id:
            try:
                notifier = TelegramNotifier(
                    bot_token=telegram_token,
                    chat_id=telegram_chat_id,
                    min_ev=8.0,
                    min_confidence=50.0,  # Stessa confidence del sistema
                    rate_limit_seconds=3
                )
                print("✅ Telegram Notifier inizializzato")
            except Exception as e:
                print(f"⚠️  Errore inizializzazione Telegram Notifier: {e}")
        else:
            print("⚠️  Telegram Notifier non disponibile (token/chat_id mancanti)")
        
        # Inizializza Live Betting Advisor
        live_advisor = LiveBettingAdvisor(
            notifier=notifier,
            min_confidence=50.0,  # Confidence abbassata
            ai_pipeline=ai_pipeline,
            min_ev=8.0,
            max_opportunities_per_match=3
        )
        print(f"✅ Live Betting Advisor inizializzato (min_confidence: {live_advisor.min_confidence}%)")
    except Exception as e:
        print(f"❌ Errore inizializzazione Live Betting Advisor: {e}")
    print()
    
    # 4. Test completo con Automation24H
    print("🚀 STEP 4: Test Completo Sistema")
    print("-" * 70)
    try:
        automation = Automation24H(
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
            min_ev=8.0,
            min_confidence=70.0,  # Per pre-match
            update_interval=300
        )
        print("✅ Automation24H inizializzato")
        
        # Esegui un singolo ciclo di analisi
        print("\n🔄 Esecuzione ciclo di analisi...")
        print("-" * 70)
        
        try:
            automation._run_cycle()
            print("\n✅ Ciclo di analisi completato!")
        except Exception as e:
            print(f"\n❌ Errore durante ciclo di analisi: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Errore inizializzazione Automation24H: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # 5. Riepilogo
    print("=" * 70)
    print("📊 RIEPILOGO TEST")
    print("=" * 70)
    print()
    print("✅ Test completato!")
    print()
    print("💡 Prossimi passi:")
    print("   1. Controlla i log sopra per vedere se sono state trovate partite live")
    print("   2. Se ci sono partite live, verifica se sono state trovate opportunità")
    print("   3. Controlla Telegram per vedere se sono arrivate notifiche")
    print("   4. Se non ci sono partite live ora, riprova durante un orario con partite")
    print()
    print("📝 Per monitorare in tempo reale:")
    print("   Get-Content automation_24h.log -Wait -Tail 50")
    print()


if __name__ == '__main__':
    # Carica variabili ambiente da .env se presente
    try:
        from dotenv import load_dotenv
        env_path = Path('.env')
        if env_path.exists():
            load_dotenv(env_path)
            print("✅ File .env caricato\n")
    except ImportError:
        pass  # dotenv non disponibile, usa variabili ambiente di sistema
    
    test_live_notifications()



