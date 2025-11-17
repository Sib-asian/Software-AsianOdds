#!/usr/bin/env python3
"""
AI System - Esempio Completo di Utilizzo
==========================================

Questo script dimostra come usare il sistema AI completo per analizzare
match e ottenere raccomandazioni di betting.

Usage:
    python ai_system_example.py
"""

import logging
import sys
from pathlib import Path

# Add ai_system to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_system.pipeline import AIPipeline, quick_analyze
from ai_system.config import AIConfig
from ai_system.utils.data_preparation import create_synthetic_training_data


def setup_logging():
    """Setup logging con colori e formato"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def example_1_basic_analysis():
    """Esempio 1: Analisi basica di un match"""
    print("\n" + "="*70)
    print("ESEMPIO 1: Analisi Basica")
    print("="*70)

    # Usa la funzione quick_analyze per analisi veloce
    result = quick_analyze(
        home_team="Inter",
        away_team="Napoli",
        league="Serie A",
        prob_dixon_coles=0.65,  # Probabilità dal tuo modello Dixon-Coles
        odds=1.85,               # Quote attuali
        bankroll=1000.0
    )

    # Accedi ai risultati
    print(f"\n📊 RISULTATI:")
    print(f"Decisione: {result['final_decision']['action']}")
    print(f"Stake Raccomandato: €{result['final_decision']['stake']:.2f}")
    print(f"Timing: {result['final_decision']['timing']}")
    print(f"Priorità: {result['final_decision']['priority']}")

    print(f"\n📈 METRICHE:")
    print(f"Probabilità Calibrata: {result['summary']['probability']:.1%}")
    print(f"Confidence Score: {result['summary']['confidence']:.0f}/100")
    print(f"Value Score: {result['summary']['value_score']:.0f}/100")
    print(f"Expected Value: {result['summary']['expected_value']:+.1%}")

    return result


def example_2_advanced_analysis():
    """Esempio 2: Analisi avanzata con tutti i parametri"""
    print("\n" + "="*70)
    print("ESEMPIO 2: Analisi Avanzata con Odds History")
    print("="*70)

    # Inizializza pipeline con config custom
    config = AIConfig()
    config.min_confidence_to_bet = 60.0  # Più conservativo
    config.kelly_default_fraction = 0.20  # Meno aggressivo

    pipeline = AIPipeline(config)

    # Match data completo
    match = {
        "home": "Milan",
        "away": "Juventus",
        "league": "Serie A",
        "date": "2025-11-20",
        "importance": 0.90  # Derby importante
    }

    # Odds data con storico
    odds_data = {
        "odds_current": 2.10,
        "market": "1x2",
        "odds_history": [
            {"odds": 2.20, "time": "08:00", "volume": 5000},
            {"odds": 2.15, "time": "10:00", "volume": 8000},
            {"odds": 2.10, "time": "12:00", "volume": 12000}  # Sharp money!
        ],
        "time_to_kickoff_hours": 3.0,
        "historical_accuracy": 0.75,
        "similar_bets_roi": 0.12,
        "similar_bets_count": 45,
        "similar_bets_winrate": 0.58
    }

    # Portfolio state
    portfolio = {
        "bankroll": 2000.0,
        "active_bets": [
            {"home": "Inter", "away": "Roma", "stake": 50, "league": "Serie A"},
            {"home": "Napoli", "away": "Lazio", "stake": 40, "league": "Serie A"}
        ]
    }

    # Analisi completa
    result = pipeline.analyze(
        match=match,
        prob_dixon_coles=0.48,
        odds_data=odds_data,
        bankroll=portfolio["bankroll"],
        portfolio_state=portfolio
    )

    # Accedi a risultati dettagliati di ogni blocco
    print(f"\n🔍 DETTAGLI PER BLOCCO:")

    # Blocco 1: Calibration
    print(f"\n[BLOCCO 1] Calibration:")
    print(f"  Raw probability: {result['calibrated']['prob_raw']:.1%}")
    print(f"  Calibrated: {result['calibrated']['prob_calibrated']:.1%}")
    print(f"  Shift: {result['calibrated']['calibration_shift']:+.1%}")

    # Blocco 2: Confidence
    print(f"\n[BLOCCO 2] Confidence:")
    print(f"  Score: {result['confidence']['confidence_score']:.0f}/100")
    print(f"  Level: {result['confidence']['confidence_level']}")
    if result['confidence']['risk_factors']:
        print(f"  Red flags: {result['confidence']['risk_factors'][:2]}")

    # Blocco 3: Value
    print(f"\n[BLOCCO 3] Value:")
    print(f"  Value type: {result['value']['value_type']}")
    print(f"  Value score: {result['value']['value_score']:.0f}/100")
    print(f"  Expected Value: {result['value']['expected_value']:+.1%}")
    print(f"  Sharp money: {result['value']['sharp_money_detected']}")

    # Blocco 4: Kelly
    print(f"\n[BLOCCO 4] Kelly Optimization:")
    print(f"  Optimal stake: €{result['kelly']['optimal_stake']:.2f}")
    print(f"  Kelly fraction: {result['kelly']['kelly_fraction']:.2f}")
    print(f"  Stake %: {result['kelly']['stake_percentage']:.1f}%")

    # Blocco 5: Risk
    print(f"\n[BLOCCO 5] Risk Management:")
    print(f"  Decision: {result['risk_decision']['decision']}")
    print(f"  Final stake: €{result['risk_decision']['final_stake']:.2f}")
    print(f"  Priority: {result['risk_decision']['priority']}")
    print(f"  Risk score: {result['risk_decision']['risk_score']:.0f}/100")

    # Blocco 6: Timing
    print(f"\n[BLOCCO 6] Timing:")
    print(f"  Recommendation: {result['timing']['timing_recommendation']}")
    print(f"  Urgency: {result['timing']['urgency']}")
    print(f"  Current odds: {result['timing']['current_odds']:.2f}")
    print(f"  Predicted 1h: {result['timing']['predicted_odds_1h']:.2f}")

    return result


def example_3_batch_analysis():
    """Esempio 3: Analisi batch di multiple partite"""
    print("\n" + "="*70)
    print("ESEMPIO 3: Analisi Batch (Multiple Matches)")
    print("="*70)

    pipeline = AIPipeline()

    # Lista di match da analizzare
    matches = [
        {
            "home": "Inter", "away": "Genoa", "league": "Serie A",
            "prob": 0.72, "odds": 1.65
        },
        {
            "home": "Roma", "away": "Lecce", "league": "Serie A",
            "prob": 0.58, "odds": 1.90
        },
        {
            "home": "Napoli", "away": "Empoli", "league": "Serie A",
            "prob": 0.68, "odds": 1.70
        },
        {
            "home": "Milan", "away": "Juventus", "league": "Serie A",
            "prob": 0.48, "odds": 2.20
        },
    ]

    bankroll = 1000.0
    results = []

    print(f"\nAnalizzando {len(matches)} partite...\n")

    for match_data in matches:
        match = {
            "home": match_data["home"],
            "away": match_data["away"],
            "league": match_data["league"],
            "date": "2025-11-20"
        }

        odds_data = {
            "odds_current": match_data["odds"],
            "odds_history": [],
            "time_to_kickoff_hours": 24.0
        }

        result = pipeline.analyze(
            match=match,
            prob_dixon_coles=match_data["prob"],
            odds_data=odds_data,
            bankroll=bankroll
        )

        results.append(result)

    # Riassunto
    print("\n" + "="*70)
    print("📊 RIASSUNTO BATCH")
    print("="*70)

    print(f"\n{'Match':<30} {'Decision':<8} {'Stake':<10} {'Priority':<8} {'Conf':<6} {'Value':<6}")
    print("-" * 70)

    for r in results:
        match_name = f"{r['match']['home']} vs {r['match']['away']}"
        decision = r['final_decision']['action']
        stake = f"€{r['final_decision']['stake']:.0f}"
        priority = r['final_decision']['priority']
        confidence = f"{r['summary']['confidence']:.0f}/100"
        value = f"{r['summary']['value_score']:.0f}/100"

        print(f"{match_name:<30} {decision:<8} {stake:<10} {priority:<8} {confidence:<6} {value:<6}")

    # Statistiche aggregate
    total_stake = sum(r['final_decision']['stake'] for r in results)
    bets_approved = sum(1 for r in results if r['final_decision']['action'] == 'BET')

    print("-" * 70)
    print(f"Totale stake raccomandato: €{total_stake:.2f}")
    print(f"Bets approvati: {bets_approved}/{len(results)}")
    print(f"% Bankroll utilizzato: {(total_stake/bankroll)*100:.1f}%")

    return results


def example_4_training():
    """Esempio 4: Training del calibratore (opzionale)"""
    print("\n" + "="*70)
    print("ESEMPIO 4: Training del Calibratore (Opzionale)")
    print("="*70)

    print("\n⚠️  NOTA: Il sistema funziona anche senza training!")
    print("I blocchi usano rule-based logic finché non vengono trainati.\n")

    # Crea dati sintetici per demo
    print("Creando dati sintetici per training...")
    df = create_synthetic_training_data(n_samples=2000, bias=0.15)

    print(f"\nDati creati: {len(df)} samples")
    print(f"Actual win rate: {df['outcome'].mean():.1%}")
    print(f"Predicted avg: {df['prob_raw'].mean():.1%}")
    print(f"Bias: +{((df['prob_raw'].mean() / df['outcome'].mean()) - 1)*100:.1f}%")

    # Train calibrator
    print("\nTraining calibrator...")
    from ai_system.blocco_1_calibrator import ProbabilityCalibrator

    calibrator = ProbabilityCalibrator()

    try:
        metrics = calibrator.train(df, validation_split=0.2)

        print(f"\n✅ Training completato!")
        print(f"   Validation Brier Score: {metrics['brier_score']:.4f}")
        print(f"   Validation Log Loss: {metrics['log_loss']:.4f}")
        print(f"   Epochs trained: {metrics['epochs_trained']}")

        # Save model
        model_path = calibrator.save()
        print(f"   Model salvato: {model_path}")

    except Exception as e:
        print(f"⚠️  Training error: {e}")
        print("   Il sistema continuerà a usare rule-based logic")

    return df


def example_5_integration_frontendcloud():
    """Esempio 5: Come integrare con Frontendcloud.py"""
    print("\n" + "="*70)
    print("ESEMPIO 5: Integrazione con Frontendcloud.py")
    print("="*70)

    print("""
INTEGRAZIONE CON IL TUO SISTEMA ESISTENTE:
==========================================

1. Nel tuo Frontendcloud.py, dopo aver calcolato le probabilità Dixon-Coles:

    from ai_system.pipeline import quick_analyze

    # Dopo calc_all_probabilities()
    prob_1 = risultato_1x2["prob_1"]  # Probabilità casa
    quota_1 = odds_1  # Quote casa

    # Analisi AI
    ai_result = quick_analyze(
        home_team=team_home,
        away_team=team_away,
        league=league,
        prob_dixon_coles=prob_1,
        odds=quota_1,
        bankroll=st.session_state.get("bankroll", 1000)
    )

    # Usa i risultati
    if ai_result['final_decision']['action'] == 'BET':
        stake = ai_result['final_decision']['stake']
        st.success(f"✅ AI Raccomandazione: Scommetti €{stake:.2f}")
        st.write(f"Confidence: {ai_result['summary']['confidence']:.0f}/100")
        st.write(f"Value Score: {ai_result['summary']['value_score']:.0f}/100")
        st.write(f"Expected Value: {ai_result['summary']['expected_value']:+.1%}")
    else:
        st.warning(f"⚠️ AI Raccomandazione: {ai_result['final_decision']['action']}")


2. Aggiungi sezione AI nel sidebar:

    with st.sidebar:
        st.subheader("🤖 AI System")

        # Config AI
        use_ai = st.checkbox("Usa AI System", value=True)

        if use_ai:
            min_confidence = st.slider(
                "Min Confidence",
                min_value=30, max_value=90, value=60
            )

            kelly_fraction = st.slider(
                "Kelly Fraction",
                min_value=0.1, max_value=0.5, value=0.25, step=0.05
            )

            # Applica config custom
            from ai_system.config import AIConfig
            config = AIConfig()
            config.min_confidence_to_bet = min_confidence
            config.kelly_default_fraction = kelly_fraction


3. Visualizza risultati AI in una expander:

    with st.expander("🔍 Dettagli AI Analysis"):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Probability (Calibrated)",
                     f"{ai_result['summary']['probability']:.1%}")
            st.metric("Confidence",
                     f"{ai_result['summary']['confidence']:.0f}/100")

        with col2:
            st.metric("Value Score",
                     f"{ai_result['summary']['value_score']:.0f}/100")
            st.metric("Expected Value",
                     f"{ai_result['summary']['expected_value']:+.1%}")

        # Reasoning
        st.write("**Reasoning:**")
        st.write(ai_result['risk_decision']['reasoning'])

        # Red/Green flags
        if ai_result['risk_decision']['red_flags']:
            st.warning("⚠️ Red Flags:")
            for flag in ai_result['risk_decision']['red_flags']:
                st.write(f"  - {flag}")

        if ai_result['risk_decision']['green_flags']:
            st.success("✅ Green Flags:")
            for flag in ai_result['risk_decision']['green_flags']:
                st.write(f"  - {flag}")


4. Tracking risultati AI:

    # Dopo il match
    if ai_result['final_decision']['action'] == 'BET':
        # Record bet
        bet_record = {
            "match": f"{team_home} vs {team_away}",
            "stake": ai_result['final_decision']['stake'],
            "odds": quota_1,
            "ai_confidence": ai_result['summary']['confidence'],
            "ai_value_score": ai_result['summary']['value_score'],
            "outcome": None  # Da aggiornare dopo match
        }

        # Salva nel database
        save_bet_to_db(bet_record)
    """)

    return None


def main():
    """Main function"""
    setup_logging()

    print("\n" + "="*70)
    print("🤖 AI SYSTEM - ESEMPI DI UTILIZZO")
    print("="*70)

    try:
        # Esegui esempi
        example_1_basic_analysis()
        input("\nPremere ENTER per continuare all'esempio 2...")

        example_2_advanced_analysis()
        input("\nPremere ENTER per continuare all'esempio 3...")

        example_3_batch_analysis()
        input("\nPremere ENTER per continuare all'esempio 4...")

        example_4_training()
        input("\nPremere ENTER per vedere esempio integrazione...")

        example_5_integration_frontendcloud()

        print("\n" + "="*70)
        print("✅ TUTTI GLI ESEMPI COMPLETATI!")
        print("="*70)
        print("\nProssimi passi:")
        print("1. Integra il sistema nel tuo Frontendcloud.py")
        print("2. Testa con dati reali")
        print("3. (Opzionale) Trainal calibratore con dati storici")
        print("4. Monitora performance e adatta configurazione")
        print("\nDocumentazione completa: ai_system/README.md")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Esempi interrotti dall'utente")
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
