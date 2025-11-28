# 🚀 Come Mandare le Modifiche su GitHub

## ✅ Setup Completato!

Ora che sei autenticato, è semplicissimo mandare le modifiche su GitHub.

---

## 📝 Processo Semplice

### Dopo ogni modifica che faccio al codice:

**1. Apri PowerShell nella cartella del progetto:**
```powershell
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"
```

**2. Esegui questo comando:**
```powershell
.\quick_pr.ps1 -Message "Descrizione delle modifiche"
```

**3. Fatto!** 🎉
- Lo script crea automaticamente un branch
- Fa commit delle modifiche
- Pusha su GitHub
- Crea il Pull Request

---

## 📋 Esempi Pratici

### Esempio 1: Fix Bug
```powershell
.\quick_pr.ps1 -Message "Fix: Corretto calcolo probabilità Dixon-Coles"
```

### Esempio 2: Nuova Feature
```powershell
.\quick_pr.ps1 -Message "Feature: Aggiunto supporto sentiment analysis"
```

### Esempio 3: Refactoring
```powershell
.\quick_pr.ps1 -Message "Refactor: Separato logica AI in moduli"
```

### Esempio 4: Documentazione
```powershell
.\quick_pr.ps1 -Message "Docs: Aggiornata documentazione API"
```

---

## 🔍 Verifica Modifiche Prima di Creare PR

Se vuoi vedere cosa è stato modificato prima di creare il PR:

```powershell
# Vedi file modificati
git status

# Vedi differenze
git diff

# Poi crea il PR
.\quick_pr.ps1 -Message "Descrizione modifiche"
```

---

## 🎯 Workflow Completo

1. **Io faccio modifiche al codice** (tramite questa chat)
2. **Tu esegui:** `.\quick_pr.ps1 -Message "Descrizione"`
3. **Lo script fa tutto automaticamente:**
   - ✅ Crea branch (es: `update-20250115-143022`)
   - ✅ Aggiunge file modificati
   - ✅ Crea commit
   - ✅ Pusha su GitHub
   - ✅ Crea Pull Request
4. **Vai su GitHub** e verifica/merge il PR

---

## 📚 Script Disponibili

### `quick_pr.ps1` (Consigliato - Veloce)
```powershell
.\quick_pr.ps1 -Message "Descrizione"
```

### `create_pr.ps1` (Completo - Più opzioni)
```powershell
.\create_pr.ps1 -CommitMessage "Commit" -PRDescription "Descrizione dettagliata"
```

---

## ⚡ Comando Rapido

Puoi anche creare un alias per essere ancora più veloce:

Aggiungi al tuo `$PROFILE` (PowerShell):
```powershell
function pr { & "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main\quick_pr.ps1" -Message $args[0] }
```

Poi usa semplicemente:
```powershell
pr "Descrizione modifiche"
```

---

## ❓ Domande Frequenti

### "Cosa succede se eseguo lo script due volte?"
- Lo script crea sempre un nuovo branch, quindi nessun problema!

### "Posso modificare il messaggio del PR dopo?"
- Sì, puoi modificare il PR direttamente su GitHub

### "Come vedo i PR creati?"
- Vai su: https://github.com/Sib-asian/Software-AsianOdds/pulls

### "Come merge il PR?"
- Vai sul PR su GitHub e clicca "Merge pull request"

---

## ✅ Checklist

- [x] Git installato
- [x] Repository inizializzato
- [x] GitHub CLI installato
- [x] Autenticazione completata
- [x] Script PR pronti
- [ ] **Pronto a creare PR!** 🎉

---

**Ora ogni volta che faccio modifiche, esegui semplicemente:**
```powershell
.\quick_pr.ps1 -Message "Descrizione modifiche"
```

**E il PR verrà creato automaticamente su GitHub!** 🚀






