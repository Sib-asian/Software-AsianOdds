# 📊 ANALISI: Intervallo Chiamate API

## ⏱️ SITUAZIONE ATTUALE

### Intervallo Attuale
- **Update Interval**: 300 secondi = **5 minuti**
- **Configurazione**: `AUTOMATION_UPDATE_INTERVAL=300` nel `.env`

### Calcolo Chiamate
- **24 ore** = 1440 minuti
- **Chiamate possibili**: 1440 / 5 = **288 chiamate/giorno**
- **Budget configurato**: 100 chiamate/giorno
- **Budget TheOddsAPI**: Dipende dal piano (verificare quota)

---

## 📈 ANALISI

### Scenario Attuale (5 minuti)
- **Chiamate/giorno**: ~288 possibili, limitate a 100 dal budget
- **Frequenza**: Ogni 5 minuti
- **Vantaggi**:
  - ✅ Aggiornamenti frequenti
  - ✅ Rileva cambiamenti quote rapidamente
  - ✅ Buono per partite live

### Scenario Aumentato (10 minuti)
- **Chiamate/giorno**: ~144 possibili
- **Frequenza**: Ogni 10 minuti
- **Vantaggi**:
  - ✅ Risparmia quota API
  - ✅ Meno rischio di esaurire quota
  - ✅ Sufficiente per pre-match

### Scenario Aumentato (15 minuti)
- **Chiamate/giorno**: ~96 possibili
- **Frequenza**: Ogni 15 minuti
- **Vantaggi**:
  - ✅ Risparmia molto quota API
  - ✅ Sicuro per non esaurire quota
  - ⚠️ Meno frequente per live

---

## 🎯 RACCOMANDAZIONE

### ✅ SÌ, conviene aumentare l'intervallo a 10 minuti

**Motivi:**

1. **Risparmio Quota**
   - Con 5 minuti: 288 chiamate/giorno possibili
   - Con 10 minuti: 144 chiamate/giorno
   - Risparmio: 50% delle chiamate

2. **Budget Attuale**
   - Budget configurato: 100 chiamate/giorno
   - Con 10 minuti: 144 chiamate possibili (sopra budget, ma sicuro)
   - Con 5 minuti: 288 chiamate possibili (troppo alto)

3. **Sufficiente per Pre-Match**
   - Partite pre-match non cambiano così rapidamente
   - 10 minuti è sufficiente per rilevare opportunità

4. **Live Betting**
   - Per partite live, 10 minuti è ancora accettabile
   - Le quote cambiano ma non così rapidamente

---

## ⚙️ COME MODIFICARE

### Opzione 1: Modifica `.env`
```env
AUTOMATION_UPDATE_INTERVAL=600  # 10 minuti (600 secondi)
```

### Opzione 2: Valori Consigliati
- **10 minuti (600s)**: Bilanciato - consigliato
- **15 minuti (900s)**: Più conservativo
- **20 minuti (1200s)**: Molto conservativo

---

## 📊 CONFRONTO

| Intervallo | Chiamate/Giorno | Risparmio | Adatto per |
|------------|-----------------|-----------|------------|
| 5 min (attuale) | 288 | - | Live + Pre-match |
| 10 min | 144 | 50% | Pre-match + Live |
| 15 min | 96 | 67% | Pre-match |
| 20 min | 72 | 75% | Solo pre-match |

---

## ✅ CONCLUSIONE

**Raccomandazione: Aumenta a 10 minuti (600 secondi)**

- ✅ Risparmia quota API
- ✅ Sufficiente per trovare opportunità
- ✅ Sicuro per non esaurire quota
- ✅ Bilanciato tra frequenza e risparmio

