# 🔄 Sistema Automatico Creazione PR

## ✅ Cosa è Stato Creato

Ho preparato un sistema completo per creare automaticamente Pull Request su GitHub dopo ogni modifica al codice.

### File Creati:

1. **`setup_git_repo.ps1`** - Setup iniziale repository Git
2. **`create_pr.ps1`** - Script completo per creare PR
3. **`quick_pr.ps1`** - Versione semplificata per uso veloce
4. **`.github/workflows/auto-pr.yml`** - GitHub Actions workflow
5. **`PR_WORKFLOW.md`** - Documentazione completa workflow
6. **`ISTRUZIONI_SETUP.md`** - Guida setup passo-passo
7. **`README_PR.md`** - Documentazione originale

---

## 🚀 Quick Start

### 1. Setup Iniziale (Una Volta)

```powershell
# Installa Git (se non già installato)
# https://git-scm.com/download/win

# Configura Git
git config --global user.name "Tuo Nome"
git config --global user.email "tua.email@example.com"

# Setup repository
.\setup_git_repo.ps1
```

### 2. Configura Autenticazione GitHub

**Opzione A: Personal Access Token**
1. Vai su: https://github.com/settings/tokens
2. Crea token con permessi `repo`
3. Usa token come password quando richiesto

**Opzione B: GitHub CLI**
```powershell
gh auth login
```

### 3. Usa Dopo Ogni Modifica

```powershell
.\quick_pr.ps1 -Message "Descrizione modifiche"
```

---

## 📋 Esempi Pratici

```powershell
# Fix bug
.\quick_pr.ps1 -Message "Fix: Corretto calcolo probabilità Dixon-Coles"

# Nuova feature
.\quick_pr.ps1 -Message "Feature: Aggiunto supporto sentiment analysis" `
  -Description "Integrato analizzatore sentiment per migliorare predizioni"

# Refactoring
.\quick_pr.ps1 -Message "Refactor: Separato logica AI in moduli"

# Documentazione
.\quick_pr.ps1 -Message "Docs: Aggiornata documentazione API"
```

---

## 🔧 Come Funziona

Quando esegui `.\quick_pr.ps1 -Message "..."`:

1. ✅ Verifica che Git sia installato
2. ✅ Crea un nuovo branch (es. `update-20250115-143022`)
3. ✅ Aggiunge tutti i file modificati
4. ✅ Crea un commit con il messaggio specificato
5. ✅ Pusha il branch su GitHub
6. ✅ Crea automaticamente il PR (se GitHub CLI disponibile)

---

## 📚 Documentazione Completa

- **`ISTRUZIONI_SETUP.md`** - Guida setup dettagliata
- **`PR_WORKFLOW.md`** - Workflow completo con best practices
- **`README_PR.md`** - Documentazione originale

---

## ⚠️ Note Importanti

1. **Git deve essere installato** - https://git-scm.com/download/win
2. **Autenticazione GitHub richiesta** - Configura token o GitHub CLI
3. **Repository deve essere inizializzato** - Esegui `setup_git_repo.ps1` prima volta

---

## 🎯 Prossimi Passi

1. ✅ Installa Git (se non già fatto)
2. ✅ Esegui `.\setup_git_repo.ps1`
3. ✅ Configura autenticazione GitHub
4. ✅ Testa con una modifica di esempio
5. ✅ Usa `.\quick_pr.ps1` per ogni modifica futura

---

## ❓ Supporto

Se hai problemi:
1. Verifica che Git sia installato: `git --version`
2. Verifica repository: `git status`
3. Controlla autenticazione: `gh auth status` (se usi GitHub CLI)
4. Leggi `ISTRUZIONI_SETUP.md` per troubleshooting

---

**Sistema pronto! Ora puoi creare PR con un solo comando! 🎉**

