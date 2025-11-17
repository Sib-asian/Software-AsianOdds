# 📱 Come Accedere al Dashboard da Cellulare

## ✅ Risposta Rapida

**SÌ, puoi accedere da cellulare!** E il PC continua a far girare l'automazione 24/7 in background.

---

## 🎯 Come Funziona

### 1. **Automazione 24/7** (PC)
- ✅ Gira in background indipendentemente
- ✅ Non si ferma quando apri Streamlit
- ✅ Continua a monitorare partite e inviare notifiche

### 2. **Dashboard Streamlit** (Accesso Remoto)
- ✅ Puoi aprire da cellulare
- ✅ Analizza partite manualmente
- ✅ Non interferisce con l'automazione

---

## 🚀 Setup Accesso Remoto

### Opzione 1: Stessa Rete WiFi (CONSIGLIATA)

**Sul PC:**

1. **Trova l'IP del PC:**
   ```powershell
   ipconfig
   ```
   Cerca "IPv4 Address" (es: 192.168.1.100)

2. **Avvia Streamlit in modalità remota:**
   ```powershell
   cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"
   streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
   ```

3. **Sul cellulare (stessa WiFi):**
   - Apri browser
   - Vai a: `http://192.168.1.100:8501`
   - Sostituisci `192.168.1.100` con l'IP del tuo PC

---

### Opzione 2: Accesso da Internet (Avanzato)

**Usa un servizio di tunneling:**

1. **Ngrok (Gratis):**
   ```powershell
   # Installa ngrok da https://ngrok.com/
   ngrok http 8501
   ```
   - Ti dà un URL pubblico (es: `https://abc123.ngrok.io`)
   - Accedi da qualsiasi luogo

2. **Cloudflare Tunnel (Gratis):**
   ```powershell
   # Installa cloudflared
   cloudflared tunnel --url http://localhost:8501
   ```

---

## 📋 Script per Avvio Facile

Ho creato uno script che avvia Streamlit in modalità remota:

**`start_dashboard_remote.bat`** - Doppio-click e parte!

---

## 🔒 Sicurezza

### Per Accesso Locale (WiFi):
- ✅ Sicuro se sei sulla stessa rete
- ⚠️  Non accessibile da Internet

### Per Accesso Internet:
- ⚠️  Configura password/autenticazione
- ⚠️  Usa HTTPS (ngrok lo fa automaticamente)
- ⚠️  Non condividere l'URL pubblicamente

---

## 📱 Cosa Puoi Fare da Cellulare

✅ **Analizzare partite manualmente:**
- Inserisci partita
- Vedi analisi AI completa
- Calcola value bet
- Vedi raccomandazioni

✅ **Visualizzare dashboard:**
- Performance overview
- Grafici profit/loss
- Statistiche avanzate

✅ **Monitorare automazione:**
- Vedi log attività
- Controlla stato sistema

---

## ⚙️ Configurazione Avanzata

### Avvio Automatico Streamlit (Opzionale)

Se vuoi che Streamlit si avvii automaticamente:

1. **Crea task schedulato** (come per automazione)
2. **Oppure aggiungi a startup** (come per automazione)

---

## 🎯 Riepilogo

✅ **Automazione 24/7:**
- Gira sempre in background
- Non si ferma quando apri Streamlit
- Continua a monitorare e notificare

✅ **Dashboard Streamlit:**
- Accessibile da cellulare
- Analisi manuale partite
- Non interferisce con automazione

✅ **Setup:**
- Stessa WiFi: `http://IP_PC:8501`
- Da Internet: Usa ngrok/cloudflare

---

## 🚀 Quick Start

1. **Sul PC, avvia:**
   ```powershell
   streamlit run dashboard.py --server.address 0.0.0.0
   ```

2. **Sul cellulare (stessa WiFi):**
   - Apri browser
   - Vai a: `http://IP_DEL_TUO_PC:8501`

3. **Fatto!** 🎉

---

**Il PC continua a far girare l'automazione 24/7 mentre tu analizzi partite da cellulare!** 📱⚽

