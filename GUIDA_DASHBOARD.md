# 📊 GUIDA COMPLETA DASHBOARD STREAMLIT

## 🎯 **COSA È LA DASHBOARD**

La dashboard è un'interfaccia web interattiva per monitorare e controllare il sistema di automazione 24/7 in tempo reale.

**Accesso:** http://localhost:8501

---

## 🏠 **LAYOUT PRINCIPALE**

### **Header**
- **Titolo:** "🤖 Automazione 24/7 - Monitor"
- **Stato Servizio:** Mostra se il sistema è ATTIVO (🟢) o FERMO (🔴)
- **PID:** Process ID del servizio in esecuzione

### **Sidebar (Sinistra)**
- ⚙️ **Configurazione:**
  - **Auto-refresh:** Checkbox per aggiornamento automatico
  - **Intervallo refresh:** Slider per scegliere ogni quanti secondi aggiornare (10-60s)

---

## 📊 **SEZIONI PRINCIPALI**

### **1. 📊 STATISTICHE OGGI**

Mostra 4 metriche chiave per la giornata corrente:

- **Opportunità Trovate:** Numero totale di opportunità rilevate oggi
- **Vincite:** Numero di scommesse vinte (con delta % win rate)
- **Perdite:** Numero di scommesse perse
- **Profit/Loss:** Profitto/Perdita totale in € (con delta ROI %)

**Colori:**
- Verde se in profitto
- Rosso se in perdita

---

### **2. 📈 STATISTICHE PERIODO**

Analisi performance su un periodo più lungo.

**Selettore Periodo:**
- 7 giorni
- 14 giorni
- 30 giorni (default)
- 60 giorni
- 90 giorni

**Due Colonne:**

#### **Colonna 1: Performance Generale**
- Totale Opportunità
- Win Rate (%)
- ROI (%)
- Profit/Loss Totale (€)

#### **Colonna 2: Distribuzione Risultati**
- **Grafico a torta (Pie Chart):**
  - Fetta verde: Vincite
  - Fetta rossa: Perdite
  - Fetta grigia: Pending (in attesa di risultato)

---

### **3. 📊 PERFORMANCE NEL TEMPO**

Due grafici interattivi che mostrano l'andamento nel tempo:

#### **Grafico 1: Profit/Loss Cumulativo**
- **Linea verde/rossa** che mostra l'andamento del profitto cumulativo
- **X-axis:** Data
- **Y-axis:** € (euro)
- **Hover:** Mostra dettagli al passaggio del mouse

#### **Grafico 2: ROI Giornaliero**
- **Barre verdi/rosse** per ogni giorno
- **X-axis:** Data
- **Y-axis:** ROI %
- Verde se ROI positivo, rosso se negativo

---

### **4. 🎯 PERFORMANCE PER MARKET**

Tabella e grafico che mostrano come performano i diversi mercati.

**Tabella:**
- Market (es: 1X2, Over 2.5, BTTS, ecc.)
- Count (numero scommesse)
- Winners (vincite)
- Win Rate % (percentuale vincite)
- Profit/Loss (€)

**Grafico a Barre:**
- Barre verdi/rosse per ogni market
- Mostra Profit/Loss per market
- Colore basato su profitto/perdita

---

### **5. 🏆 PERFORMANCE PER LEGA**

Tabella che mostra performance per campionato/lega:

- Lega (es: Serie A, Premier League, ecc.)
- Count (numero scommesse)
- Winners (vincite)
- Win Rate % (percentuale vincite)
- Profit/Loss (€)

**Utile per capire:**
- Quali leghe performano meglio
- Dove concentrare l'attenzione
- Diversificazione geografica

---

### **6. 📋 OPPORTUNITÀ RECENTI**

Tabella con le ultime 20 opportunità trovate dal sistema:

**Colonne:**
- **home_team:** Squadra casa
- **away_team:** Squadra ospite
- **league:** Campionato
- **market:** Tipo mercato (1X2, Over/Under, ecc.)
- **odds:** Quota
- **expected_value:** EV (%)
- **confidence:** Confidence (%)
- **result:** Risultato (W=Win, L=Loss, P=Pending)
- **profit_loss:** Profitto/Perdita (€)
- **notified_at:** Data/ora notifica

**Ordinamento:** Più recenti in alto

---

### **7. 📝 LOG RECENTI**

Mostra gli ultimi 50 log del sistema:

- Timestamp
- Livello (INFO, WARNING, ERROR)
- Messaggio
- Utile per debugging e monitoraggio

**Aggiornamento:** Automatico con auto-refresh

---

### **8. 🔄 BACKTESTING AUTOMATICO** 🆕

Nuova sezione con 3 tab:

#### **Tab 1: Esegui Backtest**

Interfaccia per testare strategie su dati storici:

**Parametri:**
- **Data Inizio:** Selettore data
- **Data Fine:** Selettore data
- **Min EV %:** Slider (0-20%, default 8%)
- **Min Confidence %:** Slider (0-100%, default 70%)
- **Bankroll Iniziale:** Input numerico (€, default 1000)
- **Kelly Fraction:** Slider (0.1-1.0, default 0.25)
- **Nome Strategia:** Input testo

**Pulsante:** "🚀 Esegui Backtest"

**Risultati:**
- 4 metriche principali:
  - Totale Scommesse
  - Win Rate (%)
  - ROI (%)
  - Profit Totale (€)
- **Grafico Cumulative Profit:**
  - Linea che mostra profitto cumulativo nel tempo
  - Verde se positivo, rosso se negativo

#### **Tab 2: Risultati Storici**

Tabella con tutti i backtest eseguiti:

- Strategy Name
- Start Date
- End Date
- Total Bets
- Win Rate
- ROI
- Total Profit

**Utile per:**
- Confrontare strategie
- Vedere storico performance
- Analizzare trend

#### **Tab 3: Ottimizzazione Parametri**

🚧 **In sviluppo** - Ottimizza automaticamente parametri strategia

---

## ⚙️ **FUNZIONALITÀ AVANZATE**

### **Auto-Refresh**
- ✅ Attiva/Disattiva con checkbox
- ⏱️ Intervallo configurabile (10-60 secondi)
- 🔄 Aggiorna automaticamente tutti i dati

### **Interattività**
- 🖱️ **Hover:** Passa il mouse sui grafici per dettagli
- 🔍 **Zoom:** Zoom su grafici Plotly
- 📥 **Download:** Esporta dati (funzionalità base)

### **Responsive Design**
- 📱 Funziona su desktop
- 💻 Layout wide per schermi grandi
- 📊 Grafici adattivi

---

## 🎯 **COME USARE**

### **1. Monitoraggio Real-Time**
1. Apri http://localhost:8501
2. Attiva auto-refresh
3. Monitora statistiche in tempo reale

### **2. Analisi Performance**
1. Seleziona periodo (7/14/30/60/90 giorni)
2. Analizza grafici performance
3. Identifica trend e pattern

### **3. Analisi Market/Lega**
1. Vai a "Performance per Market"
2. Identifica mercati più profittevoli
3. Analizza performance per lega

### **4. Backtesting**
1. Vai a tab "Backtesting Automatico"
2. Configura parametri
3. Clicca "Esegui Backtest"
4. Analizza risultati

### **5. Debug**
1. Vai a "Log Recenti"
2. Cerca errori o warning
3. Monitora attività sistema

---

## 💡 **TIP E TRUCCHI**

### **Performance:**
- ⚡ Usa auto-refresh con intervallo 30s per bilanciare aggiornamento/performance
- 📊 Periodo 30 giorni è un buon compromesso per analisi

### **Analisi:**
- 🎯 Focus su mercati con ROI positivo
- 🏆 Identifica leghe più profittevoli
- 📈 Monitora trend nel tempo

### **Backtesting:**
- 🔄 Testa strategie su periodi diversi
- 📊 Confronta risultati
- ⚙️ Ottimizza parametri gradualmente

---

## 🚨 **TROUBLESHOOTING**

### **Dashboard non si carica:**
- Verifica che Streamlit sia avviato
- Controlla porta 8501 libera
- Riavvia: `streamlit run automation_dashboard.py`

### **Dati non aggiornano:**
- Attiva auto-refresh
- Verifica che il servizio sia attivo
- Controlla log per errori

### **Grafici vuoti:**
- Verifica che ci siano dati nel database
- Controlla periodo selezionato
- Aspetta che il sistema raccolga dati

---

## 📱 **ACCESSO REMOTO**

Per accedere da altri dispositivi:

1. **Trova IP PC:**
   ```powershell
   ipconfig
   ```

2. **Apri Streamlit con IP:**
   ```bash
   streamlit run automation_dashboard.py --server.address 0.0.0.0
   ```

3. **Accedi da altro dispositivo:**
   ```
   http://TUO_IP:8501
   ```

---

## ✅ **RIEPILOGO**

La dashboard ti permette di:
- ✅ Monitorare sistema 24/7 in tempo reale
- ✅ Analizzare performance dettagliate
- ✅ Identificare pattern e trend
- ✅ Testare strategie con backtesting
- ✅ Debug e troubleshooting
- ✅ Accesso remoto da qualsiasi dispositivo

**Tutto in un'unica interfaccia web interattiva!** 🚀

