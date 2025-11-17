# 🔄 Guida Creazione Pull Request Automatica

Questo documento spiega come creare automaticamente Pull Request su GitHub dopo ogni modifica al codice.

## 🚀 Metodo 1: Script PowerShell (Raccomandato)

### Prerequisiti
1. **Git installato**: https://git-scm.com/download/win
2. **GitHub CLI (opzionale)**: https://cli.github.com/

### Utilizzo

```powershell
# Modifica base - PR automatica
.\create_pr.ps1

# Con messaggio personalizzato
.\create_pr.ps1 -CommitMessage "Fix bug in AI pipeline" -PRDescription "Risolto problema nel calcolo delle probabilità"

# Con branch personalizzato
.\create_pr.ps1 -BranchName "fix/ai-calibration" -CommitMessage "Fix calibration bug"
```

### Cosa fa lo script:
1. ✅ Verifica che Git sia installato
2. ✅ Crea un nuovo branch (o usa quello specificato)
3. ✅ Aggiunge tutti i file modificati
4. ✅ Crea un commit con il messaggio specificato
5. ✅ Pusha il branch su GitHub
6. ✅ Crea automaticamente il PR (se GitHub CLI è installato)

---

## 🛠️ Metodo 2: Comandi Git Manuali

Se preferisci fare tutto manualmente:

```bash
# 1. Crea branch
git checkout -b update-$(date +%Y%m%d-%H%M%S)

# 2. Aggiungi modifiche
git add .

# 3. Commit
git commit -m "Descrizione delle modifiche"

# 4. Push
git push -u origin <nome-branch>

# 5. Crea PR (con GitHub CLI)
gh pr create --title "Titolo PR" --body "Descrizione PR"
```

Oppure crea il PR manualmente su GitHub:
https://github.com/Sib-asian/Software-AsianOdds/compare

---

## 🤖 Metodo 3: GitHub Actions (Automatico)

Se configuri GitHub Actions, ogni push su branch `update-*` creerà automaticamente un PR.

Il workflow è già configurato in `.github/workflows/auto-pr.yml`.

---

## 📝 Template PR

Quando crei un PR, usa questo template:

```markdown
## 📋 Descrizione
Breve descrizione delle modifiche

## 🔧 Modifiche
- [ ] Modifica 1
- [ ] Modifica 2
- [ ] Modifica 3

## 🧪 Testing
- [ ] Test eseguiti localmente
- [ ] Nessun errore di linting
- [ ] Documentazione aggiornata

## 📸 Screenshot (se applicabile)
[Aggiungi screenshot se modifiche UI]

## ✅ Checklist
- [ ] Codice testato
- [ ] Commenti aggiunti dove necessario
- [ ] Nessun warning/errore
- [ ] Compatibilità backward mantenuta
```

---

## ⚙️ Configurazione Iniziale

### Prima volta - Setup Repository

```powershell
# Se hai solo lo ZIP, clona il repository:
cd ..
git clone https://github.com/Sib-asian/Software-AsianOdds.git
cd Software-AsianOdds

# Oppure inizializza la cartella corrente:
git init
git remote add origin https://github.com/Sib-asian/Software-AsianOdds.git
git fetch
git checkout -b main
git branch --set-upstream-to=origin/main main
```

### Configurazione Git (prima volta)

```bash
git config --global user.name "Tuo Nome"
git config --global user.email "tua.email@example.com"
```

### Autenticazione GitHub

**Opzione 1: Personal Access Token**
```bash
git remote set-url origin https://<TOKEN>@github.com/Sib-asian/Software-AsianOdds.git
```

**Opzione 2: SSH**
```bash
git remote set-url origin git@github.com:Sib-asian/Software-AsianOdds.git
```

**Opzione 3: GitHub CLI**
```bash
gh auth login
```

---

## 🎯 Workflow Consigliato

1. **Fai le modifiche al codice**
2. **Esegui lo script**: `.\create_pr.ps1 -CommitMessage "Descrizione"`
3. **Verifica il PR** su GitHub
4. **Merge** quando approvato

---

## ❓ Troubleshooting

### Git non trovato
```powershell
# Verifica installazione
where.exe git

# Se non trovato, installa da: https://git-scm.com/download/win
```

### Errore autenticazione
```bash
# Usa GitHub CLI
gh auth login

# Oppure configura token
git config --global credential.helper wincred
```

### Branch già esiste
Lo script gestisce automaticamente, ma se vuoi forzare:
```bash
git branch -D nome-branch
git checkout -b nome-branch
```

---

## 📚 Risorse

- [Documentazione Git](https://git-scm.com/doc)
- [GitHub CLI Docs](https://cli.github.com/manual/)
- [GitHub PR Guide](https://docs.github.com/en/pull-requests)

