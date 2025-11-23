# 🛑 GUIDA: Sospendere Correttamente Render

## 🚨 PROBLEMA IDENTIFICATO

I log API-SPORTS mostrano:
```
IP: 208.77.244.67 (IP di Render!)
Timestamp: 2025-11-23 11:12:48-49
Chiamate: v3/odds/live, v3/fixtures/statistics, v3/odds
```

**CONCLUSIONE**: Render è ANCORA ATTIVO, non sospeso!

---

## ✅ COME SOSPENDERE CORRETTAMENTE

### METODO A: Via Dashboard Render (Raccomandato)

#### 1. Sospendi Worker `automation-24h`

1. Vai su: https://dashboard.render.com
2. Click su **`automation-24h`** (il worker)
3. In alto a destra, cerca il menu **"⋮"** (tre puntini)
4. Click **"Suspend Service"**
5. Conferma
6. ✅ Verifica che lo stato diventi **"Suspended"**

#### 2. Sospendi Web Service `streamlit-dashboard`

1. Torna alla dashboard: https://dashboard.render.com
2. Click su **`streamlit-dashboard`** (il web service)
3. In alto a destra, menu **"⋮"** (tre puntini)
4. Click **"Suspend Service"**
5. Conferma
6. ✅ Verifica che lo stato diventi **"Suspended"**

---

### METODO B: Via CLI Render

```bash
# Installa CLI Render (se non l'hai)
npm install -g @render-com/cli

# Login
render login

# Sospendi worker
render services suspend automation-24h

# Sospendi web service
render services suspend streamlit-dashboard

# Verifica stato
render services list
```

**Output atteso**:
```
automation-24h          worker    suspended
streamlit-dashboard     web       suspended
```

---

## 🔍 VERIFICA SOSPENSIONE

Dopo aver sospeso **ENTRAMBI** i servizi:

### 1. Controlla Dashboard Render

Vai su: https://dashboard.render.com

Dovresti vedere:
```
✅ automation-24h
   Status: Suspended

✅ streamlit-dashboard
   Status: Suspended
```

### 2. Aspetta 5 Minuti

I container Docker impiegano qualche minuto per fermarsi completamente.

### 3. Controlla Log API-SPORTS

Vai su: https://dashboard.api-football.com/

Controlla la sezione "Requests":
- ✅ **Nessuna nuova chiamata** dall'IP 208.77.244.67
- ✅ **Contatore fermo**

---

## 🎯 TIMELINE ATTESA

```
T+0min:  Click "Suspend" su Render
T+1min:  Container inizia shutdown graceful
T+2min:  Container completamente fermo
T+5min:  Nessuna nuova chiamata API visibile
```

---

## 🚨 SE LE CHIAMATE CONTINUANO

Se dopo 10 minuti vedi ancora chiamate dall'IP 208.77.244.67:

### Opzione 1: Elimina i Servizi (Temporaneo)

```bash
# Via Dashboard
# automation-24h → ⋮ → Delete Service
# streamlit-dashboard → ⋮ → Delete Service

# Via CLI
render services delete automation-24h
render services delete streamlit-dashboard
```

**ATTENZIONE**: Elimina anche il disco persistente! Salva i database prima.

### Opzione 2: Cambia Chiave API (Definitivo)

1. Vai su: https://dashboard.api-football.com/
2. Account → API Keys
3. Click **"Regenerate Key"**
4. ✅ Vecchia chiave IMMEDIATAMENTE invalidata
5. ✅ Chiamate si fermano ISTANTANEAMENTE

---

## 📊 MONITORAGGIO

Dopo la sospensione, monitora per 1 ora:

```bash
# Ogni 10 minuti, controlla:
# 1. Dashboard Render → Entrambi "Suspended"
# 2. Dashboard API-SPORTS → Nessuna nuova chiamata
# 3. IP 208.77.244.67 → Nessuna attività
```

---

## ✅ CHECKLIST SOSPENSIONE

- [ ] Dashboard Render aperto
- [ ] `automation-24h` → Status = "Suspended"
- [ ] `streamlit-dashboard` → Status = "Suspended"
- [ ] Aspettato 5 minuti
- [ ] Controllato dashboard API-SPORTS
- [ ] IP 208.77.244.67 → Nessuna attività
- [ ] Contatore chiamate API fermo
- [ ] ✅ Problema risolto!

---

## 🆘 TROUBLESHOOTING

### Problema: "Non trovo il bottone Suspend"

**Soluzione**:
- Vai al servizio
- Scroll in basso
- Cerca "Settings" → "Suspend Service"

### Problema: "Servizio si riavvia automaticamente"

**Causa**: Render ha auto-deploy attivo

**Soluzione**:
```yaml
# Nel file render.yaml
autoDeploy: false  # ← Verifica che sia false!
```

### Problema: "IP 208.77.244.67 continua dopo 10 minuti"

**Causa**: Render non ha rilasciato l'IP o c'è un altro servizio

**Soluzione**:
1. Elimina servizi (non solo sospendere)
2. Oppure rigenera chiave API

---

## 📞 SUPPORTO

Se dopo tutto questo le chiamate continuano:

1. Screenshot dello stato servizi Render (entrambi "Suspended")
2. Screenshot log API-SPORTS con timestamp recenti
3. Output di: `curl -s https://api.ipify.org` (il tuo IP pubblico)

---

**Ultima modifica**: 2025-11-23 11:15
**Versione**: 1.0
