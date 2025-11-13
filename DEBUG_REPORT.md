# 🔍 Debug & Test Report

Sistema di debug completo per verificare che tutto funzioni perfettamente.

---

## 🚀 Quick Debug

```bash
# Debug completo (raccomandato)
python debug_system.py

# Test specifici
python debug_system.py --test imports      # Solo import moduli
python debug_system.py --test ai_blocks    # Solo AI blocks
python debug_system.py --test pipeline     # Solo pipeline
python debug_system.py --test models       # Solo modelli ML
python debug_system.py --test features     # Solo features avanzate
```

---

## 📦 Installazione Dipendenze

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

Questo installa:
- **Required:** numpy, pandas, scipy, scikit-learn, streamlit
- **AI Models:** torch, xgboost
- **Utils:** matplotlib, seaborn, plotly, beautifulsoup4

### Step 2: Verifica Installazione

```bash
python debug_system.py
```

---

## ✅ Cosa Viene Testato

### Test 1: Python Version
- Verifica Python >= 3.8

### Test 2: Dependencies
- Verifica pacchetti required (numpy, pandas, ecc.)
- Verifica pacchetti optional (torch, xgboost, openai)

### Test 3: File Structure
- Verifica presenza file essenziali
- Verifica directory ai_system/

### Test 4: Module Imports
- Importa tutti i moduli ai_system
- Verifica 7 blocchi AI (0-6)
- Verifica modelli ML (ensemble, xgboost, lstm)
- Verifica features avanzate

### Test 5: Configuration
- Carica e verifica ai_system/config.py
- Mostra configurazione attuale

### Test 6: AI Blocks (0-6)
- Inizializza tutti e 7 i blocchi
- Test funzionamento base

### Test 7: ML Models
- Inizializza Ensemble, XGBoost, LSTM
- Verifica compatibilità

### Test 8: Complete Pipeline
- Test end-to-end con dati di esempio
- Test con e senza Ensemble

### Test 9: Advanced Features
- LLM Analyst (senza chiamata API)
- Sentiment Analyzer
- Live Betting Adjuster
- Backtester
- Telegram Notifier

---

## 📊 Output Atteso

### ✅ Success (Tutto OK)

```
======================================================================
                 🔍 ASIAN ODDS SYSTEM - COMPLETE DEBUG
======================================================================

TEST 1: Python Version
ℹ️  Testing Python version...
✅ Python 3.11.14 (OK)

TEST 2: Dependencies
ℹ️  Testing required dependencies...
✅ numpy
✅ pandas
✅ scipy
✅ sklearn
✅ requests
✅ streamlit
✅ torch (optional)
✅ xgboost (optional)
...

📊 DEBUG SUMMARY
  Python Version: ✅ PASS
  Dependencies: ✅ PASS
  Imports: ✅ PASS
  Ai Blocks: ✅ PASS
  Ml Models: ✅ PASS
  Pipeline: ✅ PASS
  Advanced Features: ✅ PASS
  Config: ✅ PASS
  File Structure: ✅ PASS

Results: 9/9 tests passed

🎉 ALL TESTS PASSED! System is ready to use! 🎉
```

### ⚠️ Warning (Pacchetti Opzionali Mancanti)

Se vedi warnings per pacchetti optional:

```
⚠️  torch (optional - some features disabled)
⚠️  xgboost (optional - some features disabled)
⚠️  openai (optional - some features disabled)

Optional packages missing:
  - torch
  - xgboost
  - openai

Some features will be disabled without these packages.
```

**È OK!** Il sistema funziona, solo alcune features avanzate saranno disabilitate.

### ❌ Error (Problemi Critici)

Se vedi errori:

```
❌ numpy (REQUIRED)
❌ pandas (REQUIRED)

❌ Critical dependencies missing. Install them first!
ℹ️  Install with: pip install numpy pandas scipy sklearn streamlit
```

**Soluzione:** Installa dipendenze mancanti.

---

## 🐛 Troubleshooting

### Errore: ModuleNotFoundError

**Problema:**
```
ModuleNotFoundError: No module named 'numpy'
```

**Soluzione:**
```bash
pip install numpy pandas scipy scikit-learn streamlit
```

### Errore: Import ai_system failed

**Problema:**
```
❌ ai_system.pipeline.quick_analyze: ...
```

**Soluzione:**
1. Verifica di essere nella directory root del progetto
2. Verifica che `ai_system/__init__.py` esista
3. Run: `python debug_system.py --test imports` per dettagli

### Warning: torch not found

**Problema:**
```
⚠️  torch (optional - some features disabled)
```

**Soluzione (opzionale):**
```bash
# Se vuoi LSTM e features avanzate:
pip install torch

# Oppure ignora: il sistema funziona senza
```

### Pipeline test fails

**Problema:**
```
❌ Pipeline test failed: ...
```

**Soluzione:**
1. Verifica che tutti i blocchi AI (0-6) passino
2. Run: `python debug_system.py --test ai_blocks`
3. Check error details

---

## 📝 Report Dettagliato

Dopo aver eseguito `python debug_system.py`, il sistema mostra:

1. **Versione Python** - OK se >= 3.8
2. **Dipendenze** - Lista complete con status
3. **File Structure** - Verifica file essenziali
4. **Module Imports** - 20+ moduli importati
5. **AI Blocks** - 7 blocchi inizializzati
6. **ML Models** - Ensemble + modelli individuali
7. **Pipeline** - Test con dati reali
8. **Advanced** - LLM, Sentiment, Backtesting
9. **Config** - Configurazione attuale

### Summary

```
Results: X/9 tests passed

✅ 9/9 = Perfetto! Sistema pronto
⚠️ 7-8/9 = OK, alcune features opzionali mancanti
❌ <7/9 = Problemi, fix richiesto
```

---

## 🎯 Cosa Fare Dopo Debug OK

### 1. Run Sistema

```bash
streamlit run Frontendcloud.py
```

### 2. Test Manuale

1. Inserisci una partita di esempio
2. Seleziona mercato (1X2, Asian Handicap, ecc.)
3. Click "Analizza Match"
4. Verifica risultati AI

### 3. Test Features Avanzate

- Prova modalità "Auto + API" per auto-detection squadre
- Abilita "Usa AI Ensemble" per 3 modelli
- Test multiple bets contemporaneamente

### 4. (Opzionale) Setup Telegram

Se vuoi notifiche:
1. Leggi `README_TELEGRAM.md`
2. Configura bot Telegram (2 min)
3. Modifica `ai_system/config.py`

---

## 🔧 Debug Avanzato

### Log File

Il sistema salva logs in `app.log`:

```bash
# View logs
tail -f app.log

# Search errors
grep ERROR app.log

# Last 100 lines
tail -n 100 app.log
```

### Python Interactive Debug

```python
# In Python REPL
from ai_system.pipeline import quick_analyze

result = quick_analyze(
    home_team="Test Home",
    away_team="Test Away",
    league="Test League",
    prob_dixon_coles=0.65,
    odds=1.90,
    bankroll=1000.0
)

print(result)
```

### Check Specific Module

```python
# Test specific AI block
from ai_system.blocco_1_calibrator import ProbabilityCalibrator

calibrator = ProbabilityCalibrator()
calibrated = calibrator.calibrate(0.65)
print(f"Input: 0.65 → Output: {calibrated:.3f}")
```

---

## 📚 Riferimenti

- **debug_system.py** - Script debug completo
- **ai_system/README.md** - Guida sistema AI
- **AI_SYSTEM_COMPLETE_GUIDE.md** - Guida implementazione
- **TELEGRAM_SETUP.md** - Setup Telegram (opzionale)

---

## ✅ Checklist Debug

Dopo aver eseguito debug, verifica:

- [ ] Python >= 3.8
- [ ] Tutte dipendenze required installate
- [ ] ai_system/ directory presente
- [ ] Tutti i moduli si importano correttamente
- [ ] AI blocks (0-6) funzionanti
- [ ] Pipeline end-to-end OK
- [ ] Config caricata correttamente
- [ ] Frontendcloud.py runs (`streamlit run Frontendcloud.py`)

**Se tutti ✅ → Sistema 100% funzionante! 🎉**

---

## 🆘 Supporto

**Problemi dopo debug?**

1. Check `DEBUG_REPORT.md` (questo file)
2. Run `python debug_system.py` per dettagli
3. Check logs: `tail -f app.log`
4. Test modulo specifico: `python debug_system.py --test <nome>`

**Errori persistenti:** Copia output di `python debug_system.py` e chiedi aiuto.
