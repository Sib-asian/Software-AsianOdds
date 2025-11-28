# 💤 Configurazione PC per Funzionamento H24

## ✅ Il Sistema Funziona Anche Con:
- ✅ Schermo spento
- ✅ Coperchio chiuso (se configurato correttamente)
- ✅ PC in modalità "Non sospendere"

## ⚙️ Impostazioni Consigliate

### 1. **Collega il PC alla Corrente (AC)**
   - Il sistema funziona meglio quando collegato alla corrente
   - Evita problemi di risparmio energia

### 2. **Configura Cosa Succede Quando Chiudi il Coperchio**

**Opzione A: PowerShell (Rapido)**
```powershell
# Imposta "Non fare nulla" quando chiudi coperchio (collegato alla corrente)
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0

# Imposta "Sospendi" quando chiudi coperchio (a batteria) - opzionale
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1

# Applica le modifiche
powercfg /setactive SCHEME_CURRENT
```

**Opzione B: Pannello di Controllo**
1. Vai a **Pannello di Controllo** → **Hardware e suoni** → **Opzioni risparmio energia**
2. Clicca su **Specifica comportamento pulsanti di alimentazione**
3. Imposta **Quando chiudo il coperchio**:
   - **Collegato alla corrente**: **Non fare nulla**
   - **A batteria**: **Sospendi** (o **Non fare nulla** se vuoi)

### 3. **Disattiva Sospensione Quando Collegato alla Corrente**

**PowerShell:**
```powershell
# Disattiva sospensione (collegato alla corrente)
powercfg /change standby-timeout-ac 0

# Disattiva ibernazione (collegato alla corrente)
powercfg /change hibernate-timeout-ac 0
```

**Pannello di Controllo:**
1. Vai a **Opzioni risparmio energia**
2. Clicca su **Modifica impostazioni combinazione**
3. Imposta:
   - **Metti il computer in sospensione**: **Mai** (collegato alla corrente)
   - **Disattiva schermo**: **15 minuti** (va bene, risparmia energia)

### 4. **Verifica che il Servizio Funzioni**

Il servizio automation è configurato come **Scheduled Task** di Windows, quindi:
- ✅ Funziona anche con schermo spento
- ✅ Funziona anche con coperchio chiuso (se configurato)
- ✅ Continua a funzionare in background

## 🔍 Verifica Stato

### Controlla se il Servizio è Attivo:
```powershell
Get-Process python* | Where-Object {$_.Path -like "*python*"}
```

### Controlla Log:
```powershell
Get-Content logs\automation_service_*.log -Tail 20
```

## ⚠️ Cosa NON Fare

❌ **NON mettere il PC in sospensione** (sleep mode)
❌ **NON ibernare il PC** (hibernate)
❌ **NON spegnere il PC**

## ✅ Cosa Puoi Fare

✅ **Chiudere il coperchio** (se configurato "Non fare nulla")
✅ **Spegnere lo schermo** (va bene, risparmia energia)
✅ **Lasciare il PC acceso** 24/7
✅ **Collegare alla corrente** (consigliato)

## 🎯 Configurazione Rapida (Tutto in Uno)

Esegui questo comando PowerShell per configurare tutto automaticamente:

```powershell
# Imposta "Non fare nulla" quando chiudi coperchio (AC)
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0

# Disattiva sospensione (AC)
powercfg /change standby-timeout-ac 0

# Disattiva ibernazione (AC)
powercfg /change hibernate-timeout-ac 0

# Applica modifiche
powercfg /setactive SCHEME_CURRENT

Write-Host "✅ Configurazione completata! Puoi chiudere il coperchio."
```

## 📱 Monitoraggio Remoto

Puoi monitorare il sistema anche da remoto:
- **Dashboard Streamlit**: Accessibile da qualsiasi dispositivo
- **Telegram**: Ricevi notifiche in tempo reale
- **Log**: Controlla i log per verificare attività

## 💡 Consigli Finali

1. **Collega sempre alla corrente** quando lasci il PC acceso H24
2. **Chiudi il coperchio** tranquillamente (se configurato)
3. **Spegni lo schermo** per risparmiare energia
4. **Verifica periodicamente** i log per assicurarti che tutto funzioni
5. **Ricevi notifiche Telegram** per monitorare il sistema

Il sistema continuerà a funzionare perfettamente anche con coperchio chiuso e schermo spento! 🚀

