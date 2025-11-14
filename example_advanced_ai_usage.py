#!/usr/bin/env python3
"""
Esempio Pratico - Come Usare i Nuovi Sistemi IA Avanzati

Questo script mostra esempi concreti di utilizzo dei nuovi blocchi IA
per migliorare precisione e affidabilità delle previsioni betting.
"""

import sys
from datetime import datetime

print("=" * 80)
print("🎯 ESEMPIO PRATICO - ADVANCED AI SYSTEMS")
print("=" * 80)
print()

# ============================================================================
# ESEMPIO 1: USO DELLA PIPELINE COMPLETA (RACCOMANDATO)
# ============================================================================

print("📊 ESEMPIO 1: Pipeline Completa (Tutti gli 8 Sistemi IA)")
print("-" * 80)

try:
    from ai_system.advanced_precision_pipeline import AdvancedPrecisionPipeline

    # Inizializza la pipeline (usa tutti gli 8 blocchi automaticamente)
    pipeline = AdvancedPrecisionPipeline(strict_mode=False)

    # Dati di una predizione reale
    match_data = {
        # Dati base (richiesti)
        "probability": 0.65,          # Probabilità predetta dal tuo modello
        "lambda_home": 1.8,           # Expected goals squadra casa
        "lambda_away": 1.2,           # Expected goals squadra trasferta
        "market_type": "home_win",    # Tipo di mercato

        # Dati mercato (opzionali ma consigliati)
        "market_odds": 1.75,          # Quote offerte dal bookmaker
        "historical_margins": [0.05, 0.06, 0.04, 0.05],

        # Metadata (opzionali)
        "league": "Premier League",

        # Ensemble predictions (se hai più modelli)
        "ensemble_predictions": [0.63, 0.65, 0.67, 0.64],
        "model_confidences": [0.85, 0.90, 0.82, 0.88],

        # Info modelli (per consensus validation)
        "model_predictions": [
            {"name": "Dixon-Coles", "probability": 0.63, "confidence": 0.85, "type": "statistical"},
            {"name": "XGBoost", "probability": 0.67, "confidence": 0.90, "type": "ml"},
            {"name": "LSTM", "probability": 0.64, "confidence": 0.82, "type": "dl"},
            {"name": "Meta-Learner", "probability": 0.65, "confidence": 0.88, "type": "ensemble"}
        ]
    }

    # Processa la predizione attraverso TUTTI gli 8 sistemi IA
    result = pipeline.process_prediction(match_data)

    # Mostra risultati completi
    print()
    print("=" * 80)
    print("📊 RISULTATI ANALISI AVANZATA")
    print("=" * 80)

    print(f"\n🎲 PROBABILITÀ:")
    print(f"  • Originale:         {result.original_probability:.4f}")
    print(f"  • Calibrata:         {result.calibrated_probability:.4f}")
    print(f"  • Consensus:         {result.consensus_probability:.4f}")
    print(f"  • Raccomandata:      {result.recommended_probability:.4f}")
    print(f"  • Intervallo 95%:    [{result.credible_interval_95[0]:.4f}, {result.credible_interval_95[1]:.4f}]")

    print(f"\n🎯 AFFIDABILITÀ:")
    print(f"  • Confidence Score:  {result.confidence_score:.1f}/100")
    print(f"  • Reliability Index: {result.reliability_index:.2f}")
    print(f"  • Robustness Score:  {result.robustness_score:.1f}/100")
    print(f"  • Uncertainty:       {result.uncertainty_level}")

    print(f"\n✅ VALIDAZIONE:")
    print(f"  • Validation OK:     {result.validation_passed}")
    print(f"  • Validation Score:  {result.validation_score:.1f}/100")
    print(f"  • Consistency Score: {result.consistency_score:.1f}/100")

    print(f"\n⚠️  RISCHI:")
    print(f"  • Risk Level:        {result.risk_level}")
    print(f"  • Anomaly Detected:  {result.anomaly_detected}")
    if result.anomaly_detected:
        print(f"  • Anomaly Severity:  {result.anomaly_severity}")

    print(f"\n🤝 CONSENSO:")
    print(f"  • Consensus Reached: {result.consensus_reached}")
    print(f"  • Agreement Score:   {result.agreement_score:.1f}/100")
    if result.outlier_models:
        print(f"  • Outlier Models:    {', '.join(result.outlier_models)}")

    print(f"\n💰 OPPORTUNITÀ:")
    print(f"  • Expected Value:    {result.expected_value_pct:.2f}%")
    if result.arbitrage_opportunities:
        for opp in result.arbitrage_opportunities:
            print(f"  • {opp['type']}: EV {opp['ev_pct']:.2f}% (Risk: {opp['risk']})")

    print(f"\n🎯 RACCOMANDAZIONE FINALE: {result.recommendation}")

    print(f"\n📝 MOTIVAZIONI:")
    for i, reason in enumerate(result.reasoning, 1):
        print(f"  {i}. {reason}")

    # Decisione basata sulla raccomandazione
    print()
    print("=" * 80)
    print("💡 ESEMPIO DECISIONE AUTOMATICA")
    print("=" * 80)

    if result.recommendation == "BET":
        print(f"\n✅ PROCEDI CON LA BET!")
        print(f"  • Probabilità da usare: {result.recommended_probability:.4f}")
        print(f"  • Expected Value: {result.expected_value_pct:.2f}%")
        print(f"  • Confidence: {result.confidence_score:.1f}/100")
        print(f"  • Nota: applica le tue regole di bankroll esterne (nessun stake automatico)")

    elif result.recommendation == "SKIP":
        print(f"\n❌ SKIP QUESTA OPPORTUNITÀ")
        print(f"  Motivo: {result.reasoning[0]}")

    elif result.recommendation == "WATCH":
        print(f"\n⏸️  MONITORA MA NON BETTARE ANCORA")
        print(f"  • Confidence: {result.confidence_score:.1f}/100 (troppo bassa)")
        print(f"  • Attendi migliori condizioni")

    else:  # INVESTIGATE
        print(f"\n🔍 RICHIEDE ULTERIORE ANALISI")
        print(f"  • Controlla manualmente i dati")
        print(f"  • Verifica assunzioni del modello")

    print()

except ImportError as e:
    print(f"⚠️  Dipendenze mancanti: {e}")
    print("   Esegui: pip install numpy scipy scikit-learn")
    print()

# ============================================================================
# ESEMPIO 2: USO SINGOLI BLOCCHI (AVANZATO)
# ============================================================================

print()
print("=" * 80)
print("🔧 ESEMPIO 2: Uso Singoli Blocchi IA (Personalizzato)")
print("=" * 80)
print()

# Esempio 2A: Bayesian Uncertainty
print("📊 2A. Bayesian Uncertainty Quantification")
print("-" * 80)

try:
    from ai_system.blocco_7_bayesian_uncertainty import BayesianUncertaintyQuantifier

    quantifier = BayesianUncertaintyQuantifier(n_samples=5000)

    # Calcola posterior per una singola predizione
    bayesian_result = quantifier.calculate_posterior(
        predicted_prob=0.65,
        historical_success_rate=0.62,  # Da dati storici
        n_historical=100
    )

    print(f"  • Mean Posterior:     {bayesian_result.mean:.4f}")
    print(f"  • 95% CI:             [{bayesian_result.credible_interval_95[0]:.4f}, {bayesian_result.credible_interval_95[1]:.4f}]")
    print(f"  • Uncertainty:        {bayesian_result.uncertainty_level}")
    print(f"  • Confidence:         {bayesian_result.confidence_score:.1f}/100")
    print()

except ImportError as e:
    print(f"  ⚠️  Richiede: {e}")
    print()

# Esempio 2B: Monte Carlo Simulation
print("🎲 2B. Monte Carlo Robustness Testing")
print("-" * 80)

try:
    from ai_system.blocco_8_monte_carlo import MonteCarloSimulator

    simulator = MonteCarloSimulator(n_simulations=5000)

    # Simula ROI su 100 bet
    mc_result = simulator.simulate_betting_roi(
        probability=0.55,    # Tua probabilità stimata
        odds=2.0,            # Quote del bookmaker
        stake=10.0,          # Stake per bet
        probability_uncertainty=0.05,  # Incertezza sulla probabilità
        n_bets=100           # Numero bet da simulare
    )

    print(f"  • Expected ROI:       {mc_result.mean_outcome:.2f}%")
    print(f"  • VaR 95%:            {mc_result.var_95:.2f}%")
    print(f"  • CVaR 95%:           {mc_result.cvar_95:.2f}%")
    print(f"  • Best Case:          {mc_result.best_case:.2f}%")
    print(f"  • Worst Case:         {mc_result.worst_case:.2f}%")
    print(f"  • Prob Positive:      {mc_result.probability_positive_ev:.2%}")
    print(f"  • Robustness:         {mc_result.robustness_score:.1f}/100")
    print()

except ImportError as e:
    print(f"  ⚠️  Richiede: {e}")
    print()

# Esempio 2C: Anomaly Detection
print("🚨 2C. Anomaly Detection")
print("-" * 80)

try:
    from ai_system.blocco_9_anomaly_detection import AnomalyDetector

    detector = AnomalyDetector(sensitivity="high")

    # Rileva anomalie nel mercato
    anomaly = detector.detect_market_anomalies(
        predicted_prob=0.65,
        market_odds=3.5,  # Implied = 0.286 (grande discrepanza!)
        historical_margins=[0.05, 0.06, 0.04, 0.05]
    )

    print(f"  • Anomaly Detected:   {anomaly.is_anomaly}")
    print(f"  • Anomaly Score:      {anomaly.anomaly_score:.1f}/100")
    print(f"  • Severity:           {anomaly.severity}")
    print(f"  • Confidence:         {anomaly.confidence:.2f}")
    if anomaly.recommendations:
        print(f"  • Recommendation:     {anomaly.recommendations[0]}")
    print()

except ImportError as e:
    print(f"  ⚠️  Richiede: {e}")
    print()

# Esempio 2D: Arbitrage Detection
print("💰 2D. Statistical Arbitrage Detection")
print("-" * 80)

try:
    from ai_system.blocco_13_arbitrage_detector import StatisticalArbitrageDetector

    arb_detector = StatisticalArbitrageDetector()

    # Rileva arbitraggio statistico
    arb = arb_detector.detect_statistical_arbitrage(
        model_probability=0.58,
        market_odds=2.0,  # Implied = 0.50
        model_confidence=0.85,
        historical_edge=7.5
    )

    if arb:
        print(f"  • Opportunity:        {arb.opportunity_type}")
        print(f"  • Expected Value:     {arb.expected_value_pct:.2f}%")
        print(f"  • Risk Level:         {arb.risk_level}")
        print(f"  • Recommended Stake:  {arb.stakes_allocation['bet']:.1f}%")
        print(f"  • Time Sensitivity:   {arb.time_sensitivity}")
    else:
        print(f"  • No arbitrage opportunity found")
    print()

except ImportError as e:
    print(f"  ⚠️  Richiede: {e}")
    print()

# ============================================================================
# STATISTICHE PIPELINE
# ============================================================================

print("=" * 80)
print("📈 STATISTICHE DELLA PIPELINE")
print("=" * 80)

try:
    stats = pipeline.get_pipeline_statistics()

    print(f"\n  • Predictions Processed: {stats['predictions_processed']}")
    print(f"  • Anomalies Detected:    {stats['anomalies_detected']}")
    print(f"  • Validations Failed:    {stats['validations_failed']}")
    print(f"  • Anomaly Rate:          {stats['anomaly_rate']:.2%}")
    print(f"  • Failure Rate:          {stats['validation_failure_rate']:.2%}")

    if 'calibration_status' in stats:
        cal_status = stats['calibration_status']
        print(f"\n  📊 Calibration System:")
        print(f"  • Total Observations:    {cal_status['total_observations']}")
        print(f"  • Calibration Error:     {cal_status['global']['calibration_error']:.4f}")
        print(f"  • Leagues Calibrated:    {cal_status['n_leagues_calibrated']}")

    print()

except:
    pass

# ============================================================================
# ESEMPIO 3: INTEGRAZIONE CON SISTEMA ESISTENTE
# ============================================================================

print("=" * 80)
print("🔌 ESEMPIO 3: Integrazione con Sistema Esistente")
print("=" * 80)
print()

print("""
# Come integrare nella tua pipeline esistente:

# Nel tuo Frontendcloud.py o script principale:

from ai_system.advanced_precision_pipeline import AdvancedPrecisionPipeline

# Inizializza (una volta all'inizio)
advanced_pipeline = AdvancedPrecisionPipeline(strict_mode=False)

# Per ogni match che analizzi:
def analyze_match_with_advanced_ai(match_data, existing_prediction):
    '''
    Wrapper che arricchisce la tua predizione esistente
    con i nuovi sistemi IA avanzati
    '''

    # Prepara dati per pipeline avanzata
    advanced_data = {
        "probability": existing_prediction["prob"],
        "lambda_home": match_data["lambda_home"],
        "lambda_away": match_data["lambda_away"],
        "market_type": "home_win",

        # Aggiungi le tue predizioni da modelli diversi
        "ensemble_predictions": [
            existing_prediction["dixon_coles"],
            existing_prediction["xgboost"],
            existing_prediction["lstm"],
            existing_prediction["meta_learner"]
        ],
        "model_confidences": [0.85, 0.90, 0.82, 0.88],

        # Dati dal mercato
        "market_odds": match_data.get("market_odds", 2.0),
        "league": match_data.get("league", "Unknown")
    }

    # Passa attraverso i nuovi sistemi IA
    advanced_result = advanced_pipeline.process_prediction(advanced_data)

    # Decisione finale basata sui nuovi sistemi
    if advanced_result.recommendation == "BET":
        return {
            "action": "BET",
            "probability": advanced_result.recommended_probability,
            "confidence": advanced_result.confidence_score,
            "expected_value": advanced_result.expected_value_pct,
            "reasoning": advanced_result.reasoning
        }
    elif advanced_result.recommendation == "SKIP":
        return {
            "action": "SKIP",
            "reason": advanced_result.reasoning[0],
            "details": advanced_result
        }
    else:
        return {
            "action": advanced_result.recommendation,
            "details": advanced_result
        }

""")

print()
print("=" * 80)
print("✅ ESEMPI COMPLETATI!")
print("=" * 80)
print()
print("📚 Per maggiori dettagli, consulta:")
print("  • QUICK_START.md")
print("  • ADVANCED_AI_SYSTEMS_README.md")
print("  • INSTALLATION_GUIDE.md")
print()
print("🚀 Buon betting con IA avanzata!")
print("=" * 80)
