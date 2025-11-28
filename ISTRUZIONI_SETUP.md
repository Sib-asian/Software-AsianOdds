# 🚀 Istruzioni Setup Completo

## Setup Iniziale (Una Volta)

### Passo 1: Installa Git
1. Scarica Git da: https://git-scm.com/download/win
2. Installa con le impostazioni di default
3. **Riavvia PowerShell** dopo l'installazione

### Passo 2: Configura Git
Apri PowerShell e esegui:
```powershell
git config --global user.name "Tuo Nome"
git config --global user.email "tua.email@example.com"
```

### Passo 3: Setup Repository
Nella cartella del progetto, esegui:
```powershell
.\setup_git_repo.ps1
```

Questo script:
- ✅ Verifica che Git sia installato
- ✅ Inizializza il repository Git
- ✅ Configura il remote GitHub
- ✅ Crea .gitignore
- ✅ Crea il branch main

### Passo 4: Configura Autenticazione GitHub

**Opzione A: Personal Access Token (Consigliato)**
1. Vai su: https://github.com/settings/tokens
2. Clicca "Generate new token (classic)"
3. Seleziona permessi `repo` (tutti)
4. Copia il token
5. Quando fai push, usa il token come password

**Opzione B: GitHub CLI**
```powershell
# Installa: https://cli.github.com/
gh auth login
```

**Opzione C: SSH Key**
```powershell
# Genera chiave
ssh-keygen -t ed25519 -C "tua.email@example.com"

# Aggiungi a GitHub: https://github.com/settings/keys
# Poi cambia remote:
git remote set-url origin git@github.com:Sib-asian/Software-AsianOdds.git
```

---

## Uso Quotidiano

### Dopo Ogni Modifica al Codice:

```powershell
.\quick_pr.ps1 -Message "Descrizione delle modifiche"
```

**Esempi:**
```powershell
# Fix bug
.\quick_pr.ps1 -Message "Fix: Corretto calcolo probabilità"

# Nuova feature  
.\quick_pr.ps1 -Message "Feature: Aggiunto sentiment analysis"

# Refactoring
.\quick_pr.ps1 -Message "Refactor: Separato logica AI"
```

---

## Script Disponibili

### `setup_git_repo.ps1`
Setup iniziale - esegui **una sola volta**

### `create_pr.ps1`
Script completo con tutti i parametri

### `quick_pr.ps1`
Versione semplificata per uso veloce

---

## Troubleshooting

### "Git non trovato"
- Installa Git: https://git-scm.com/download/win
- Riavvia PowerShell dopo installazione

### "Errore autenticazione"
- Configura Personal Access Token
- Oppure usa: `gh auth login`

### "Repository non inizializzato"
```powershell
.\setup_git_repo.ps1
```

---

## ✅ Checklist

- [ ] Git installato
- [ ] Git configurato (nome e email)
- [ ] Repository inizializzato
- [ ] Autenticazione GitHub configurata
- [ ] Script testato

---

**Pronto! Ora puoi creare PR con un comando! 🎉**






