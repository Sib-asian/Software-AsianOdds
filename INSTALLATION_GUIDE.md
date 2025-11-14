# 🚀 Guida all'Installazione - Advanced AI Systems

## ✅ Dipendenze Necessarie

**OTTIMA NOTIZIA**: I nuovi sistemi IA (Blocchi 7-14) **NON richiedono nuove dipendenze**!

Usano le stesse librerie già presenti nel sistema:
- ✅ numpy ≥1.23.0
- ✅ scipy ≥1.9.0
- ✅ scikit-learn ≥1.2.0
- ✅ Built-in Python (dataclasses, typing, datetime)

---

## 📦 Installazione

### Se è la Prima Installazione:

```bash
cd /home/user/Software-AsianOdds

# Installa tutte le dipendenze
pip install -r requirements.txt
```

### Se il Sistema è Già Funzionante:

**Non serve fare nulla!** I nuovi blocchi 7-14 sono già pronti all'uso. ✅

---

## 🧪 Verifica Installazione

Esegui questo comando per verificare che tutto funzioni:

```bash
python3 -c "
from ai_system.advanced_precision_pipeline import AdvancedPrecisionPipeline

pipeline = AdvancedPrecisionPipeline()
print('✓ Advanced AI Systems caricati con successo!')
print(f'✓ Pipeline pronta con {8} nuovi blocchi IA')
"
```

Se vedi i messaggi di successo, sei pronto! 🎉

---

## 🎮 Test Rapido

### Test dei Singoli Blocchi:

```bash
# Test Bayesian Uncertainty
python3 ai_system/blocco_7_bayesian_uncertainty.py

# Test Monte Carlo
python3 ai_system/blocco_8_monte_carlo.py

# Test Anomaly Detection
python3 ai_system/blocco_9_anomaly_detection.py

# Test Market Consistency
python3 ai_system/blocco_10_market_consistency.py

# Test Adaptive Calibration
python3 ai_system/blocco_11_adaptive_calibration.py

# Test Consensus Validator
python3 ai_system/blocco_12_consensus_validator.py

# Test Arbitrage Detector
python3 ai_system/blocco_13_arbitrage_detector.py

# Test Realtime Validation
python3 ai_system/blocco_14_realtime_validation.py
```

### Test Pipeline Completa:

```bash
python3 ai_system/advanced_precision_pipeline.py
```

Output atteso:
```
=== TEST: Advanced Precision Pipeline ===

============================================================
ADVANCED PREDICTION ANALYSIS REPORT
============================================================

📊 PROBABILITIES:
  Original:     0.6500
  Calibrated:   0.6523
  Consensus:    0.6485
  Recommended:  0.6450
  95% CI:       [0.6289, 0.6687]

🎯 CONFIDENCE & RELIABILITY:
  Confidence Score:   78.5/100
  Reliability Index:  0.85
  Robustness Score:   72.3/100
  Uncertainty Level:  MEDIUM

✅ VALIDATION:
  Validation Passed:  True
  Validation Score:   95.0/100
  Consistency Score:  98.5/100

🎯 FINAL RECOMMENDATION: BET

✓ Advanced Precision Pipeline Test Completed!
```

---

## 🔧 Uso Base

### Esempio Minimo:

```python
from ai_system.advanced_precision_pipeline import AdvancedPrecisionPipeline

# Inizializza
pipeline = AdvancedPrecisionPipeline()

# Prepara dati
prediction_data = {
    "probability": 0.65,
    "lambda_home": 1.8,
    "lambda_away": 1.2,
    "market_type": "home_win",
    "market_odds": 1.75
}

# Processa
result = pipeline.process_prediction(prediction_data)

# Usa il risultato
print(f"Recommendation: {result.recommendation}")
print(f"Confidence: {result.confidence_score:.1f}/100")

if result.recommendation == "BET":
    print(f"Suggested probability: {result.recommended_probability:.4f}")
    print(f"Expected Value: {result.expected_value_pct:.2f}%")
```

### Esempio Completo:

```python
from ai_system.advanced_precision_pipeline import AdvancedPrecisionPipeline

pipeline = AdvancedPrecisionPipeline(strict_mode=False)

# Dati completi con ensemble
prediction_data = {
    "probability": 0.65,
    "lambda_home": 1.8,
    "lambda_away": 1.2,
    "market_type": "home_win",

    # Ensemble predictions
    "ensemble_predictions": [0.63, 0.65, 0.67, 0.64],
    "model_confidences": [0.85, 0.90, 0.82, 0.88],

    # Market data
    "market_odds": 1.75,
    "historical_margins": [0.05, 0.06, 0.04, 0.05],

    # Metadata
    "league": "Premier League",

    # Multiple models
    "model_predictions": [
        {"name": "Dixon-Coles", "probability": 0.63, "confidence": 0.85, "type": "statistical"},
        {"name": "XGBoost", "probability": 0.67, "confidence": 0.90, "type": "ml"},
        {"name": "LSTM", "probability": 0.64, "confidence": 0.82, "type": "dl"}
    ]
}

result = pipeline.process_prediction(prediction_data)

# Analisi completa disponibile
print(f"📊 Probabilities:")
print(f"  Original:     {result.original_probability:.4f}")
print(f"  Calibrated:   {result.calibrated_probability:.4f}")
print(f"  Consensus:    {result.consensus_probability:.4f}")
print(f"  Recommended:  {result.recommended_probability:.4f}")
print(f"  95% CI:       [{result.credible_interval_95[0]:.4f}, {result.credible_interval_95[1]:.4f}]")

print(f"\n🎯 Assessment:")
print(f"  Confidence:   {result.confidence_score:.1f}/100")
print(f"  Reliability:  {result.reliability_index:.2f}")
print(f"  Risk Level:   {result.risk_level}")

print(f"\n✅ Validation:")
print(f"  Passed:       {result.validation_passed}")
print(f"  Consistent:   {result.consistency_score:.1f}/100")

print(f"\n💰 Opportunities:")
print(f"  Expected Value: {result.expected_value_pct:.2f}%")
if result.arbitrage_opportunities:
    for opp in result.arbitrage_opportunities:
        print(f"  {opp['type']}: {opp['ev_pct']:.2f}%")

print(f"\n🎯 RECOMMENDATION: {result.recommendation}")
print(f"\n📝 Reasoning:")
for reason in result.reasoning:
    print(f"  • {reason}")
```

---

## 🔍 Troubleshooting

### Problema: ImportError numpy/scipy

**Soluzione**:
```bash
pip install numpy scipy scikit-learn
```

### Problema: Module not found 'ai_system'

**Soluzione**:
Assicurati di essere nella directory corretta:
```bash
cd /home/user/Software-AsianOdds
python3 -m ai_system.advanced_precision_pipeline
```

### Problema: Performance lenta

**Soluzione**:
Riduci numero di simulazioni Monte Carlo:
```python
from ai_system.blocco_8_monte_carlo import MonteCarloSimulator

# Default: 10,000 simulazioni
simulator = MonteCarloSimulator(n_simulations=10000)

# Più veloce: 5,000 simulazioni
simulator = MonteCarloSimulator(n_simulations=5000)

# Molto veloce: 1,000 simulazioni (per testing)
simulator = MonteCarloSimulator(n_simulations=1000)
```

---

## 📊 Statistiche Sistema

Dopo aver usato la pipeline, puoi controllare le statistiche:

```python
stats = pipeline.get_pipeline_statistics()

print(f"Predictions Processed: {stats['predictions_processed']}")
print(f"Anomalies Detected: {stats['anomalies_detected']}")
print(f"Anomaly Rate: {stats['anomaly_rate']:.2%}")
print(f"Validation Failures: {stats['validations_failed']}")
```

---

## 📚 Documentazione Completa

Per la documentazione completa, consulta:
- **`ADVANCED_AI_SYSTEMS_README.md`** - Guida completa a tutti i sistemi
- **`CODEBASE_EXPLORATION_REPORT.md`** - Analisi del codebase esistente
- Docstrings nei singoli file per API reference

---

## ✅ Checklist Post-Installazione

- [ ] `pip install -r requirements.txt` completato
- [ ] Test pipeline: `python3 ai_system/advanced_precision_pipeline.py`
- [ ] Import test superato senza errori
- [ ] Primo prediction processato con successo
- [ ] Statistiche pipeline verificate

---

## 🆘 Supporto

In caso di problemi:
1. Verifica versione Python: `python3 --version` (richiesto ≥3.7)
2. Verifica dipendenze: `pip list | grep -E "numpy|scipy|scikit"`
3. Controlla i log di errore completi
4. Consulta la documentazione nei file sorgente

---

## 🎉 Pronto!

Se tutti i test passano, sei pronto per usare gli 8 nuovi sistemi IA avanzati!

**Buon betting con precisione aumentata! 🚀📈**
