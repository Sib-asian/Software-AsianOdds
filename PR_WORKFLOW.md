# 🔄 Workflow Creazione PR Automatica

## 🚀 Setup Iniziale (Una Volta)

### 1. Installa Git
Scarica e installa Git da: https://git-scm.com/download/win

### 2. Configura Git (Prima Volta)
```powershell
git config --global user.name "Tuo Nome"
git config --global user.email "tua.email@example.com"
```

### 3. Setup Repository
```powershell
.\setup_git_repo.ps1
```

Questo script:
- ✅ Verifica che Git sia installato
- ✅ Inizializza il repository Git (se necessario)
- ✅ Configura il remote GitHub
- ✅ Crea .gitignore
- ✅ Crea il branch main

### 4. Configura Autenticazione GitHub

**Opzione A: Personal Access Token (Raccomandato)**
1. Vai su: https://github.com/settings/tokens
2. Crea un nuovo token con permessi `repo`
3. Usa il token quando richiesto durante il push

**Opzione B: GitHub CLI**
```powershell
# Installa GitHub CLI: https://cli.github.com/
gh auth login
```

**Opzione C: SSH Key**
```powershell
# Genera chiave SSH
ssh-keygen -t ed25519 -C "tua.email@example.com"

# Aggiungi chiave a GitHub: https://github.com/settings/keys
git remote set-url origin git@github.com:Sib-asian/Software-AsianOdds.git
```

---

## 📝 Workflow Quotidiano

### Dopo Ogni Modifica al Codice:

#### Metodo Veloce (Raccomandato)
```powershell
.\quick_pr.ps1 -Message "Descrizione breve delle modifiche"
```

#### Metodo Completo
```powershell
.\create_pr.ps1 -CommitMessage "Descrizione commit" -PRDescription "Descrizione dettagliata PR"
```

### Esempi Pratici:

```powershell
# Fix bug
.\quick_pr.ps1 -Message "Fix: Corretto calcolo probabilità Dixon-Coles"

# Nuova feature
.\quick_pr.ps1 -Message "Feature: Aggiunto supporto sentiment analysis" -Description "Integrato analizzatore sentiment per migliorare predizioni"

# Refactoring
.\quick_pr.ps1 -Message "Refactor: Separato logica AI in moduli"

# Documentazione
.\quick_pr.ps1 -Message "Docs: Aggiornata documentazione API"
```

---

## 🔧 Script Disponibili

### `setup_git_repo.ps1`
Setup iniziale repository. Esegui **una sola volta**.

### `create_pr.ps1`
Script completo per creare PR con tutti i parametri.

**Parametri:**
- `-BranchName`: Nome branch (default: auto-generato)
- `-CommitMessage`: Messaggio commit (obbligatorio)
- `-PRTitle`: Titolo PR (default: stesso di commit)
- `-PRDescription`: Descrizione PR
- `-BaseBranch`: Branch base (default: "main")

### `quick_pr.ps1`
Wrapper semplificato per uso veloce.

**Parametri:**
- `-Message`: Messaggio commit/PR (obbligatorio)
- `-Description`: Descrizione PR (opzionale)

---

## 📋 Cosa Fa lo Script

1. ✅ **Verifica Git** - Controlla che Git sia installato
2. ✅ **Crea Branch** - Genera branch con timestamp (es. `update-20250115-143022`)
3. ✅ **Aggiunge File** - Aggiunge tutte le modifiche
4. ✅ **Crea Commit** - Committa con il messaggio specificato
5. ✅ **Push GitHub** - Pusha il branch su GitHub
6. ✅ **Crea PR** - Crea automaticamente il PR (se GitHub CLI disponibile)

---

## 🎯 Best Practices

### Messaggi Commit
Usa prefissi chiari:
- `Fix:` per bug fix
- `Feature:` per nuove funzionalità
- `Refactor:` per refactoring
- `Docs:` per documentazione
- `Test:` per test
- `Perf:` per ottimizzazioni

### Branch Naming
Lo script genera automaticamente nomi come:
- `update-20250115-143022`
- `fix-ai-calibration-20250115`
- `feature-sentiment-20250115`

### PR Description
Includi sempre:
- **Cosa** è stato modificato
- **Perché** è stato modificato
- **Come** testare le modifiche
- **Screenshot** (se modifiche UI)

---

## ❓ Troubleshooting

### "Git non trovato"
```powershell
# Verifica installazione
where.exe git

# Se non trovato, installa: https://git-scm.com/download/win
# Riavvia PowerShell dopo installazione
```

### "Repository non inizializzato"
```powershell
.\setup_git_repo.ps1
```

### "Errore autenticazione"
```powershell
# Usa GitHub CLI
gh auth login

# Oppure configura token
git config --global credential.helper wincred
```

### "Branch già esiste"
Lo script gestisce automaticamente, ma se vuoi forzare:
```powershell
git branch -D nome-branch
```

### "Nessuna modifica da committare"
Verifica le modifiche:
```powershell
git status
git diff
```

---

## 🚀 Automazione Avanzata

### Alias PowerShell
Aggiungi al tuo `$PROFILE`:
```powershell
function New-PR {
    param([string]$Message, [string]$Desc = "")
    & "C:\path\to\Software-AsianOdds-main\quick_pr.ps1" -Message $Message -Description $Desc
}

Set-Alias -Name pr -Value New-PR
```

Poi usa semplicemente:
```powershell
pr "Fix bug"
```

### Git Hooks
Crea `.git/hooks/post-commit` per creare PR automaticamente dopo ogni commit:
```bash
#!/bin/sh
powershell.exe -File ".\create_pr.ps1" -CommitMessage "$(git log -1 --pretty=%B)"
```

---

## 📚 Risorse

- [Git Documentation](https://git-scm.com/doc)
- [GitHub CLI Docs](https://cli.github.com/manual/)
- [GitHub PR Guide](https://docs.github.com/en/pull-requests)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## ✅ Checklist Setup

- [ ] Git installato
- [ ] Git configurato (nome e email)
- [ ] Repository inizializzato (`setup_git_repo.ps1`)
- [ ] Autenticazione GitHub configurata
- [ ] Script testato con modifica di esempio

---

**Pronto! Ora puoi creare PR con un solo comando! 🎉**






