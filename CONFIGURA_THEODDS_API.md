# 🔑 Configurazione TheOddsAPI Key

## 📋 Situazione Attuale

La chiave **THEODDS_API_KEY** non è attualmente configurata nel sistema.

## 🎯 Come Configurarla

### Opzione 1: File .env (CONSIGLIATA)

1. **Crea file `.env`** nella cartella principale:
   ```
   Software-AsianOdds-main\.env
   ```

2. **Aggiungi la tua chiave:**
   ```env
   THEODDS_API_KEY=your_api_key_here
   ```

3. **Riavvia il servizio:**
   - Il sistema leggerà automaticamente la chiave dal file `.env`

### Opzione 2: Variabile d'Ambiente Windows

1. **Apri PowerShell come Amministratore**

2. **Imposta variabile d'ambiente:**
   ```powershell
   [System.Environment]::SetEnvironmentVariable("THEODDS_API_KEY", "your_api_key_here", "User")
   ```

3. **Riavvia il servizio**

## 🔑 Come Ottenere la Chiave

1. Vai su: https://the-odds-api.com/
2. Registrati (gratis)
3. Ottieni la tua API key dal dashboard
4. Free tier: 500 richieste/mese

## ✅ Verifica Configurazione

Dopo aver configurato, verifica con:
```powershell
python verify_api_keys.py
```

Dovresti vedere:
```
✅ THEODDS_API_KEY: your_key...
```

## 🚨 Se Non Configuri

Se non configuri la chiave:
- ✅ Il sistema funziona comunque
- ⚠️  Usa dati mock per testing
- ℹ️  I log mostrano: "THEODDS_API_KEY non configurata, usando mock data"

## 📝 Note

- La chiave viene letta automaticamente all'avvio
- Non serve riavviare il PC, solo il servizio
- La chiave è sicura nel file `.env` (non viene committata su git)

