#!/usr/bin/env python3
"""
Test completo sistema:
- Deduplicazione segnali
- Live Match AI
- Integrazione
"""

import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_deduplicazione():
    """Test sistema deduplicazione"""
    print("\n" + "="*60)
    print("TEST 1: DEDUPLICAZIONE SEGNALI")
    print("="*60)
    
    try:
        from automation_24h import Automation24H
        
        # Crea istanza (senza avviare)
        automation = Automation24H(
            telegram_token="test_token",
            telegram_chat_id="test_chat",
            min_ev=8.0,
            min_confidence=70.0,
            update_interval=600
        )
        
        # Simula opportunità
        opp1 = {
            'match_id': 'match_123',
            'ai_result': {
                'market': '1X2_HOME',
                'final_decision': {'market': '1X2_HOME'}
            }
        }
        
        opp2 = {
            'match_id': 'match_123',
            'ai_result': {
                'market': '1X2_HOME',  # Stesso mercato
                'final_decision': {'market': '1X2_HOME'}
            }
        }
        
        opp3 = {
            'match_id': 'match_123',
            'ai_result': {
                'market': 'OVER_2_5',  # Mercato diverso
                'final_decision': {'market': 'OVER_2_5'}
            }
        }
        
        # Test deduplicazione
        key1 = f"{opp1['match_id']}|{opp1['ai_result']['market']}"
        key2 = f"{opp2['match_id']}|{opp2['ai_result']['market']}"
        key3 = f"{opp3['match_id']}|{opp3['ai_result']['market']}"
        
        notified = set()
        
        # Primo segnale
        if key1 not in notified:
            notified.add(key1)
            print(f"✅ Segnale 1 accettato: {key1}")
        else:
            print(f"❌ Segnale 1 bloccato (duplicato): {key1}")
        
        # Secondo segnale (duplicato)
        if key2 not in notified:
            notified.add(key2)
            print(f"✅ Segnale 2 accettato: {key2}")
        else:
            print(f"✅ Segnale 2 bloccato (duplicato): {key2}")
        
        # Terzo segnale (mercato diverso)
        if key3 not in notified:
            notified.add(key3)
            print(f"✅ Segnale 3 accettato: {key3}")
        else:
            print(f"❌ Segnale 3 bloccato (duplicato): {key3}")
        
        # Verifica risultati
        assert key2 in notified, "Errore: segnale duplicato non bloccato!"
        assert key3 in notified, "Errore: segnale con mercato diverso bloccato!"
        
        print("\n✅ TEST DEDUPLICAZIONE: PASSATO")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST DEDUPLICAZIONE: FALLITO - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_live_match_ai():
    """Test Live Match AI"""
    print("\n" + "="*60)
    print("TEST 2: LIVE MATCH AI")
    print("="*60)
    
    try:
        from ai_system.live_match_ai import LiveMatchAI
        
        # Crea istanza
        live_ai = LiveMatchAI(
            ai_pipeline=None,
            min_confidence=70.0,
            min_ev=8.0
        )
        
        # Dati test
        match_data = {
            'id': 'test_match_1',
            'home': 'Team A',
            'away': 'Team B',
            'league': 'Test League',
            'odds_1': 2.0,
            'odds_x': 3.0,
            'odds_2': 3.5
        }
        
        live_data = {
            'score_home': 1,
            'score_away': 0,
            'minute': 45,
            'possession_home': 60,
            'possession_away': 40,
            'shots_home': 8,
            'shots_away': 4,
            'shots_on_target_home': 4,
            'shots_on_target_away': 2
        }
        
        odds_data = {
            'home': 2.0,
            'draw': 3.0,
            'away': 3.5,
            'over_2_5': 2.2,
            'under_2_5': 1.8
        }
        
        # Analizza
        result = live_ai.analyze_live_match(match_data, live_data, odds_data)
        
        print(f"✅ Analisi completata")
        print(f"   Match ID: {result.get('match_id')}")
        print(f"   Opportunità trovate: {len(result.get('opportunities', []))}")
        
        # Verifica struttura
        assert 'match_id' in result, "Errore: match_id mancante"
        assert 'opportunities' in result, "Errore: opportunities mancante"
        assert 'situation_analysis' in result, "Errore: situation_analysis mancante"
        assert 'patterns' in result, "Errore: patterns mancante"
        
        # Verifica opportunità
        for opp in result.get('opportunities', []):
            assert 'market' in opp, "Errore: market mancante in opportunità"
            assert 'probability' in opp, "Errore: probability mancante"
            assert 'odds' in opp, "Errore: odds mancante"
            assert 'ev' in opp, "Errore: ev mancante"
            assert 'confidence' in opp, "Errore: confidence mancante"
            print(f"   - {opp['market']}: EV={opp['ev']:.1f}%, Conf={opp['confidence']:.0f}%")
        
        print("\n✅ TEST LIVE MATCH AI: PASSATO")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST LIVE MATCH AI: FALLITO - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrazione():
    """Test integrazione completa"""
    print("\n" + "="*60)
    print("TEST 3: INTEGRAZIONE COMPLETA")
    print("="*60)
    
    try:
        from automation_24h import Automation24H
        
        # Crea istanza
        automation = Automation24H(
            telegram_token="test_token",
            telegram_chat_id="test_chat",
            min_ev=8.0,
            min_confidence=70.0,
            update_interval=600
        )
        
        # Verifica componenti
        print("Verifica componenti...")
        
        # Live Match AI
        if hasattr(automation, 'live_match_ai'):
            if automation.live_match_ai:
                print("✅ Live Match AI inizializzata")
            else:
                print("⚠️  Live Match AI non disponibile (opzionale)")
        else:
            print("❌ Live Match AI non trovata")
            return False
        
        # Notified opportunities (deduplicazione)
        if hasattr(automation, 'notified_opportunities'):
            print("✅ Sistema deduplicazione presente")
            print(f"   Tipo: {type(automation.notified_opportunities)}")
        else:
            print("❌ Sistema deduplicazione non trovato")
            return False
        
        # Verifica metodo _handle_opportunity
        if hasattr(automation, '_handle_opportunity'):
            print("✅ Metodo _handle_opportunity presente")
        else:
            print("❌ Metodo _handle_opportunity non trovato")
            return False
        
        # Verifica metodo _format_live_ai_opportunity_message
        if hasattr(automation, '_format_live_ai_opportunity_message'):
            print("✅ Metodo _format_live_ai_opportunity_message presente")
        else:
            print("❌ Metodo _format_live_ai_opportunity_message non trovato")
            return False
        
        print("\n✅ TEST INTEGRAZIONE: PASSATO")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST INTEGRAZIONE: FALLITO - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Esegue tutti i test"""
    print("\n" + "="*60)
    print("🧪 TEST COMPLETO SISTEMA")
    print("="*60)
    
    results = []
    
    # Test 1: Deduplicazione
    results.append(("Deduplicazione", test_deduplicazione()))
    
    # Test 2: Live Match AI
    results.append(("Live Match AI", test_live_match_ai()))
    
    # Test 3: Integrazione
    results.append(("Integrazione", test_integrazione()))
    
    # Riepilogo
    print("\n" + "="*60)
    print("📊 RIEPILOGO TEST")
    print("="*60)
    
    for name, result in results:
        status = "✅ PASSATO" if result else "❌ FALLITO"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 TUTTI I TEST SONO PASSATI!")
        print("✅ Il sistema è pronto per essere avviato")
    else:
        print("\n⚠️  ALCUNI TEST SONO FALLITI")
        print("❌ Controlla gli errori sopra prima di avviare")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)



