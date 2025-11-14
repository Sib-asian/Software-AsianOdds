#!/usr/bin/env python3
"""
Script di verifica rapida per controllare la visibilità dell'AI System
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_ai_imports():
    """Test 1: Verifica che l'AI system sia importabile"""
    print("=" * 60)
    print("TEST 1: Verifica Import AI System")
    print("=" * 60)

    try:
        from ai_system.pipeline import quick_analyze, AIPipeline
        from ai_system.config import AIConfig
        print("✅ AI Pipeline importato correttamente")
        print(f"   - quick_analyze: {quick_analyze}")
        print(f"   - AIPipeline: {AIPipeline}")
        print(f"   - AIConfig: {AIConfig}")
        return True
    except ImportError as e:
        print(f"❌ Errore import: {e}")
        return False

def test_ai_config():
    """Test 2: Verifica configurazione AI"""
    print("\n" + "=" * 60)
    print("TEST 2: Verifica Configurazione AI")
    print("=" * 60)

    try:
        from ai_system.config import AIConfig, get_conservative_config

        config = AIConfig()
        print(f"✅ Configurazione AI creata")
        print(f"   - Verbose: {config.verbose}")
        print(f"   - Min confidence: {config.min_confidence_to_bet}")
        print(f"   - Kelly fraction: {config.kelly_fraction}")
        print(f"   - Ensemble enabled: {config.use_ensemble}")

        # Test preset
        conservative = get_conservative_config()
        print(f"\n✅ Preset Conservative caricato")
        print(f"   - Min confidence: {conservative.min_confidence_to_bet}")
        print(f"   - Kelly fraction: {conservative.kelly_fraction}")

        return True
    except Exception as e:
        print(f"❌ Errore configurazione: {e}")
        return False

def test_ai_blocchi():
    """Test 3: Verifica che tutti i 7 blocchi siano presenti"""
    print("\n" + "=" * 60)
    print("TEST 3: Verifica Blocchi AI")
    print("=" * 60)

    blocchi = [
        ("blocco_0_api_engine", "API Data Engine"),
        ("blocco_1_calibrator", "Probability Calibrator"),
        ("blocco_2_confidence", "Confidence Scorer"),
        ("blocco_3_value_detector", "Value Detector"),
        ("blocco_4_kelly", "Smart Kelly Optimizer"),
        ("blocco_5_risk_manager", "Risk Manager"),
        ("blocco_6_odds_tracker", "Odds Movement Tracker"),
    ]

    all_ok = True
    for module_name, display_name in blocchi:
        try:
            module = __import__(f"ai_system.{module_name}", fromlist=[""])
            print(f"✅ [{module_name}] {display_name} - OK")
        except ImportError as e:
            print(f"❌ [{module_name}] {display_name} - ERRORE: {e}")
            all_ok = False

    return all_ok

def test_quick_analyze():
    """Test 4: Verifica analisi rapida"""
    print("\n" + "=" * 60)
    print("TEST 4: Test Analisi Rapida (Mock Data)")
    print("=" * 60)

    try:
        from ai_system.pipeline import quick_analyze
        from ai_system.config import AIConfig

        # Dati di test
        result = quick_analyze(
            home_team="Test Home",
            away_team="Test Away",
            league="Premier League",
            prob_dixon_coles=0.55,
            odds=1.80,
            bankroll=1000.0,
            config=AIConfig(verbose=False)
        )

        print("✅ Analisi completata")
        print("\nSezioni presenti nel risultato:")
        for key in result.keys():
            print(f"   - {key}")

        # Verifica sezioni chiave
        required_sections = ['final_decision', 'summary', 'calibrated',
                           'confidence', 'value', 'kelly', 'risk_decision', 'timing']
        missing = [s for s in required_sections if s not in result]

        if missing:
            print(f"\n⚠️ Sezioni mancanti: {missing}")
            return False
        else:
            print("\n✅ Tutte le sezioni richieste sono presenti!")

            # Mostra decisione finale
            decision = result['final_decision']
            print(f"\n📊 RISULTATO ANALISI:")
            print(f"   - Azione: {decision['action']}")
            print(f"   - Stake: €{decision['stake']:.2f}")
            print(f"   - Timing: {decision['timing']}")
            print(f"   - Priority: {decision['priority']}")

            return True

    except Exception as e:
        print(f"❌ Errore analisi: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_integration():
    """Test 5: Verifica integrazione Streamlit"""
    print("\n" + "=" * 60)
    print("TEST 5: Verifica Integrazione Streamlit")
    print("=" * 60)

    try:
        with open("Frontendcloud.py", "r") as f:
            content = f.read()

        checks = [
            ("from ai_system.pipeline import", "Import AI pipeline"),
            ("AI_SYSTEM_AVAILABLE", "Flag disponibilità AI"),
            ("ai_enabled", "Checkbox abilitazione AI"),
            ("quick_analyze", "Chiamata analisi AI"),
            ("AI System - Betting Recommendation", "Sezione risultati AI"),
            ("Dettagli Analisi AI Completa (7 Blocchi)", "Expander dettagli"),
        ]

        all_ok = True
        for pattern, description in checks:
            if pattern in content:
                print(f"✅ {description}")
            else:
                print(f"❌ {description} - NON TROVATO")
                all_ok = False

        return all_ok

    except Exception as e:
        print(f"❌ Errore lettura Streamlit: {e}")
        return False

def main():
    """Esegue tutti i test"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "AI SYSTEM VISIBILITY TEST SUITE" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    tests = [
        test_ai_imports,
        test_ai_config,
        test_ai_blocchi,
        test_quick_analyze,
        test_streamlit_integration,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test fallito con eccezione: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("RIEPILOGO TEST")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"\nTest passati: {passed}/{total}")

    if passed == total:
        print("\n🎉 TUTTI I TEST PASSATI! Il sistema AI è completamente funzionante.")
        print("\n✅ PROSSIMI PASSI:")
        print("   1. Avvia Streamlit: streamlit run Frontendcloud.py")
        print("   2. Cerca la sezione '🤖 AI System - Enhanced Predictions'")
        print("   3. Abilita il checkbox '✅ Abilita AI Analysis'")
        print("   4. Inserisci i dati di una partita e clicca 'Analizza'")
        print("   5. Verifica i risultati AI sotto 'Betting Recommendation'")
        return 0
    else:
        print("\n⚠️ ALCUNI TEST FALLITI")
        print("\n🔧 AZIONI CORRETTIVE:")
        print("   1. Installa dipendenze: pip install -r requirements.txt")
        print("   2. Verifica che la cartella ai_system/ esista")
        print("   3. Controlla i log per errori specifici")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
