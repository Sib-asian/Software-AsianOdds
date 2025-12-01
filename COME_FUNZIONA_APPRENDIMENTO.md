# Sistema di Apprendimento Signal Quality Gate

## 📚 Come Funziona

### Meccanismo di Apprendimento

Il sistema apprende automaticamente dai risultati delle partite per migliorare la qualità dei segnali.

#### 1. **Tracciamento Segnali** (Automatico H24)
- Ogni segnale valutato (inviato o bloccato) viene registrato nel database
- Vengono salvati: Quality Score, Context Score, Data Quality, Logic, Timing, Confidence, EV
- Viene tracciato se il segnale è stato approvato o bloccato

#### 2. **Aggiornamento Risultati** (Automatico H24)
- Quando una partita finisce, il sistema confronta il risultato finale con la previsione
- Calcola se il segnale era corretto o sbagliato
- Aggiorna automaticamente i record nel database

#### 3. **Apprendimento** (Periodico o Manuale)
- Analizza tutti i segnali con risultati noti
- Calcola metriche: Precision, Recall, Accuracy
- Identifica pattern (es. "il contesto è più importante del timing")
- Aggiorna automaticamente:
  - **Pesi del Quality Score** (35% contesto, 25% dati, ecc.)
  - **Min Quality Score threshold** (75.0)

---

## 🔄 Modalità di Funzionamento

### Opzione 1: H24 (Consigliato)
**Il software gira sempre H24**

✅ **Vantaggi:**
- Tracciamento automatico continuo
- Aggiornamento risultati in tempo reale
- Apprendimento periodico automatico (es. ogni 6 ore)
- Miglioramento continuo

**Come funziona:**
1. Il software traccia ogni segnale automaticamente
2. Quando una partita finisce, aggiorna i risultati automaticamente
3. Ogni 6 ore (configurabile), esegue apprendimento automatico
4. I parametri vengono aggiornati e applicati immediatamente

### Opzione 2: Batch (Alternativa)
**Il software gira solo quando necessario**

✅ **Vantaggi:**
- Non serve tenere il software sempre acceso
- Puoi eseguire apprendimento manualmente quando vuoi

**Come funziona:**
1. Il software traccia i segnali quando gira
2. Quando una partita finisce, aggiorna i risultati (se il software è acceso)
3. Esegui apprendimento manualmente quando hai abbastanza dati:
   ```python
   from ai_system.signal_quality_learner import SignalQualityLearner
   
   learner = SignalQualityLearner()
   results = learner.learn_from_results(min_samples=50)
   print(f"Precision: {results['precision']:.2%}")
   print(f"Recall: {results['recall']:.2%}")
   ```

---

## 📊 Cosa Apprende

### 1. **Pesi del Quality Score**
Il sistema identifica quali componenti sono più importanti:

- **Context Score** (contesto partita): Se i segnali corretti hanno context_score alto, aumenta il peso
- **Data Quality Score** (qualità dati): Se i dati statistici sono importanti, aumenta il peso
- **Logic Score** (logica segnale): Se la logica è importante, aumenta il peso
- **Timing Score** (timing): Se il timing è importante, aumenta il peso

**Esempio:**
- Se i segnali corretti hanno context_score medio 90 e timing_score medio 60
- Il sistema aumenta il peso di context e diminuisce il peso di timing
- Nuovi pesi: Context 40%, Timing 10% (invece di 35% e 15%)

### 2. **Min Quality Score Threshold**
Il sistema ottimizza la soglia minima:

- **Se Precision è alta (>80%) ma Recall è bassa (<50%)**: Abbassa soglia (più segnali)
- **Se Precision è bassa (<60%)**: Alza soglia (meno segnali ma più precisi)

**Esempio:**
- Precision 85%, Recall 40% → Troppi falsi negativi → Soglia: 75 → 74
- Precision 55%, Recall 70% → Troppi falsi positivi → Soglia: 75 → 76

---

## 🎯 Metriche Calcolate

### Precision
**Percentuale di segnali approvati che erano corretti**
- Precision alta = Pochi falsi positivi
- Precision bassa = Molti falsi positivi

### Recall
**Percentuale di segnali corretti che sono stati approvati**
- Recall alta = Pochi falsi negativi
- Recall bassa = Molti falsi negativi

### Accuracy
**Percentuale di decisioni corrette (approvati corretti + bloccati sbagliati)**
- Accuracy alta = Sistema funziona bene
- Accuracy bassa = Sistema ha bisogno di miglioramenti

---

## ⚙️ Configurazione

### Apprendimento Automatico H24

Il sistema esegue apprendimento automatico ogni 6 ore (configurabile).

Per cambiare frequenza, modifica in `automation_24h.py`:
```python
# Apprendimento ogni 6 ore
if time_since_learning > 21600:  # 6 ore in secondi
    learner.learn_from_results(min_samples=50)
```

### Apprendimento Manuale

Esegui quando vuoi:
```python
from ai_system.signal_quality_learner import SignalQualityLearner

learner = SignalQualityLearner()
results = learner.learn_from_results(min_samples=50)

print(f"Precision: {results['precision']:.2%}")
print(f"Recall: {results['recall']:.2%}")
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Nuovi pesi: {results['weights']}")
print(f"Nuova soglia: {results['min_quality_score']}")
```

---

## 📈 Esempio di Apprendimento

### Situazione Iniziale
- Pesi: Context 35%, Data 25%, Logic 25%, Timing 15%
- Min Score: 75.0
- Precision: 70%, Recall: 60%

### Dopo 100 Segnali
Il sistema analizza:
- Segnali corretti: Context medio 88, Timing medio 65
- Segnali sbagliati: Context medio 72, Timing medio 70

**Apprendimento:**
- Context è più importante → Aumenta peso a 38%
- Timing è meno importante → Diminuisce peso a 12%
- Precision 75%, Recall 55% → Abbassa soglia a 74.0

### Dopo 500 Segnali
- Pesi ottimizzati: Context 40%, Data 25%, Logic 25%, Timing 10%
- Min Score: 73.5
- Precision: 82%, Recall: 68%

---

## ❓ FAQ

### Serve tenere il software sempre acceso?
**No, ma è consigliato per:**
- Tracciamento automatico continuo
- Aggiornamento risultati in tempo reale
- Apprendimento periodico automatico

**Alternativa:**
- Gira il software quando vuoi
- Esegui apprendimento manuale quando hai abbastanza dati (min 50 segnali)

### Quanti segnali servono per apprendere?
**Minimo 50 segnali con risultati noti**

Con più segnali, l'apprendimento è più accurato:
- 50-100 segnali: Apprendimento base
- 100-500 segnali: Apprendimento buono
- 500+ segnali: Apprendimento ottimale

### I parametri vengono salvati?
**Sì**, i parametri appresi vengono salvati nel database `signal_quality_learning.db` e vengono caricati automaticamente all'avvio.

### Posso resettare l'apprendimento?
**Sì**, elimina il database:
```bash
rm signal_quality_learning.db
```

Il sistema ricomincerà con parametri default.

---

## 🚀 Inizio Rapido

1. **Avvia il software H24** (consigliato)
2. **Lascia tracciare segnali** per qualche giorno
3. **L'apprendimento avviene automaticamente** ogni 6 ore
4. **Monitora i log** per vedere i miglioramenti:
   ```
   ✅ Apprendimento completato: Precision=82.5%, Recall=68.3%, Accuracy=78.1%
   Pesi: Context=40.2%, Data=24.8%, Logic=25.1%, Timing=9.9%
   Min Score=73.5
   ```

Il sistema migliorerà automaticamente nel tempo! 🎯


