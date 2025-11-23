# 🛑 PROCEDURA SOSPENSIONE IMMEDIATA

## 🎯 OBIETTIVO
Fermare il servizio `automation-24h` su Render che sta facendo chiamate API dall'IP 208.77.244.67

---

## 📋 PASSO 1: Sospendi Servizio

### Via Dashboard (Più facile)

1. Vai su: https://dashboard.render.com
2. Click su **`automation-24h`**
3. Scroll in alto a destra
4. Click menu **"⋮"** (tre puntini verticali)
5. Nel menu a tendina, click **"Suspend Service"**
6. Conferma nel popup che appare
7. ✅ Verifica che lo status cambi in **"Suspended"**

### Via Settings (Alternativo)

1. Vai su: https://dashboard.render.com
2. Click su **`automation-24h`**
3. Nel menu laterale sinistro, click **"Settings"**
4. Scroll in basso fino alla sezione **"Danger Zone"**
5. Click **"Suspend Service"**
6. Conferma
7. ✅ Verifica che lo status cambi in **"Suspended"**

---

## ⏱️ PASSO 2: Aspetta il Shutdown Completo

Il container Docker impiega tempo per fermarsi:

```
T+0min:  Click "Suspend"
         → Render invia SIGTERM al container

T+1min:  Container riceve segnale
         → Inizia shutdown graceful
         → Completa eventuali chiamate API in corso

T+2min:  Container chiude connessioni
         → Salva database su disco
         → Termina processi Python

T+5min:  Container completamente fermo
         → Nessuna nuova chiamata API
         → IP 208.77.244.67 inattivo
```

**IMPORTANTE**: Anche dopo "Suspend", potrebbero esserci chiamate API per 1-2 minuti!

---

## 🔍 PASSO 3: Verifica Sospensione

### 3A. Controlla Dashboard Render (Immediato)

1. Vai su: https://dashboard.render.com
2. Verifica che `automation-24h` mostri:
   ```
   ✅ Status: Suspended
   ✅ Last deployed: [data ultima]
   ✅ No active deployments
   ```

### 3B. Controlla Log Render (Dopo 2 minuti)

1. Vai su `automation-24h`
2. Click **"Logs"** nel menu laterale
3. Dovresti vedere l'ultima riga tipo:
   ```
   🛑 Shutdown requested
   ✅ Sistema automazione fermato
   ```

### 3C. Controlla API-SPORTS (Dopo 5 minuti)

1. Vai su: https://dashboard.api-football.com/
2. Sezione "Requests"
3. Verifica:
   ```
   ✅ Nessuna nuova chiamata dall'IP 208.77.244.67
   ✅ Contatore chiamate fermo
   ✅ Ultimo timestamp: circa 5 minuti fa
   ```

---

## 🚨 TROUBLESHOOTING

### Problema: "Status torna su Running dopo qualche minuto"

**Causa**: Auto-deploy attivo o health check che riavvia

**Soluzione**:

```bash
# Verifica render.yaml
cat render.yaml | grep autoDeploy

# Deve essere:
autoDeploy: false
```

Se è `true`, cambio io il file e faccio commit.

---

### Problema: "Non trovo il bottone Suspend"

**Passo alternativo**:

1. Vai su `automation-24h` → **Settings**
2. Scroll fino a **"Danger Zone"** (in rosso)
3. Click **"Suspend Service"**

---

### Problema: "Chiamate API continuano dopo 10 minuti"

**Opzione 1**: Elimina servizio (temporaneo)

```
Dashboard → automation-24h → Settings → Danger Zone
→ "Delete Service"
```

**ATTENZIONE**: Elimina anche il disco `/data`! Salva database prima.

---

**Opzione 2**: Cambia chiave API (definitivo - RACCOMANDATO)

```
1. https://dashboard.api-football.com/
2. Account → API Keys
3. "Regenerate Key"
4. ✅ Vecchia chiave IMMEDIATAMENTE invalidata
5. ✅ IP 208.77.244.67 non può più fare chiamate
```

Poi quando riavvii Render, aggiorno io la chiave nel dashboard.

---

## ✅ CHECKLIST COMPLETA

- [ ] Dashboard Render aperto
- [ ] `automation-24h` trovato
- [ ] Status PRIMA della sospensione: ________
- [ ] Click "Suspend Service"
- [ ] Conferma popup
- [ ] Status DOPO: "Suspended" ✅
- [ ] Aspettato 5 minuti
- [ ] Log Render: ultimo messaggio shutdown
- [ ] API-SPORTS: nessuna nuova chiamata
- [ ] IP 208.77.244.67: inattivo
- [ ] ✅ PROBLEMA RISOLTO!

---

## 📊 TIMELINE ATTESA

```
Ora attuale: 11:12 (ultimo log API)
├─ 11:15: Sospensione effettuata
├─ 11:16: Container riceve SIGTERM
├─ 11:17: Shutdown graceful iniziato
├─ 11:18: Ultime chiamate API completate ←
├─ 11:20: Container completamente fermo
└─ 11:25: ✅ Verifica nessuna nuova chiamata
```

---

## 🎯 DOPO LA SOSPENSIONE

Quando vorrai riavviare Render:

1. Click **"Resume Service"** su dashboard
2. Render si riavvia automaticamente
3. Tutte le funzionalità tornano attive:
   - ✅ Quote live
   - ✅ Statistiche partite
   - ✅ Notifiche Telegram
   - ✅ Cache database (su disco `/data`)

**Nessuna configurazione necessaria** - riprende da dove si era fermato!

---

**Creato**: 2025-11-23 11:20
**Versione**: 1.0
