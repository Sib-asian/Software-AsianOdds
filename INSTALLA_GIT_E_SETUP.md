# 🔧 Installa Git e Completa Step 1

## ❌ Situazione Attuale

Git **NON è installato** sul tuo sistema. Per completare Step 1 e fare push su GitHub, devi installare Git.

---

## 🚀 OPZIONE 1: Installa Git (CONSIGLIATO)

### Passo 1: Scarica Git

1. Vai su: **https://git-scm.com/download/win**
2. Clicca su **"Download for Windows"**
3. Il download inizierà automaticamente (circa 50 MB)

### Passo 2: Installa Git

1. **Apri il file scaricato** (es: `Git-2.43.0-64-bit.exe`)
2. **Clicca "Next"** su tutte le schermate
3. **IMPORTANTE**: Lascia tutte le opzioni di default (sono già ottimali)
4. **Clicca "Install"**
5. Attendi il completamento (1-2 minuti)
6. **Clicca "Finish"**

### Passo 3: Riavvia PowerShell

1. **Chiudi** tutte le finestre PowerShell/Terminal
2. **Riapri** PowerShell
3. **Verifica installazione**:
   ```powershell
   git --version
   ```
   Dovresti vedere: `git version 2.xx.x`

### Passo 4: Configura Git (Prima Volta)

```powershell
git config --global user.name "Tuo Nome"
git config --global user.email "tua.email@example.com"
```

**Sostituisci**:
- `"Tuo Nome"` con il tuo nome (es: "Alessandro")
- `"tua.email@example.com"` con la tua email GitHub

### Passo 5: Esegui Setup Repository

Ora che Git è installato, esegui:

```powershell
cd Software-AsianOdds-main
.\setup_git_repo.ps1
```

### Passo 6: Crea Repository su GitHub

1. Vai su: **https://github.com/new**
2. **Repository name**: `Software-AsianOdds` (o nome che preferisci)
3. Scegli: **Public** (gratis) o **Private**
4. **NON** selezionare "Add a README file" (già lo hai)
5. **Clicca "Create repository"**

### Passo 7: Push su GitHub

```powershell
# Se lo script ha già configurato il remote, esegui:
git push -u origin main

# Se devi ancora configurare il remote:
git remote add origin https://github.com/TUO-USERNAME/Software-AsianOdds.git
git branch -M main
git push -u origin main
```

**Sostituisci** `TUO-USERNAME` con il tuo username GitHub.

---

## 🎨 OPZIONE 2: GitHub Desktop (PIÙ FACILE)

Se preferisci un'interfaccia grafica invece della riga di comando:

### Passo 1: Scarica GitHub Desktop

1. Vai su: **https://desktop.github.com/**
2. Clicca **"Download for Windows"**
3. Installa il file scaricato

### Passo 2: Login con GitHub

1. Apri GitHub Desktop
2. Clicca **"Sign in to GitHub.com"**
3. Login con le tue credenziali GitHub

### Passo 3: Aggiungi Repository Locale

1. **File** → **Add Local Repository**
2. Clicca **"Choose..."**
3. Seleziona la cartella: `C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main`
4. Clicca **"Add repository"**

### Passo 4: Pubblica su GitHub

1. Clicca **"Publish repository"** (in alto)
2. **Nome**: `Software-AsianOdds`
3. Scegli: **Public** o **Private**
4. **NON** selezionare "Keep this code private" se vuoi pubblico
5. Clicca **"Publish Repository"**

**✅ FATTO!** Il codice è ora su GitHub!

---

## 📤 OPZIONE 3: Upload Manuale (Senza Git)

Se non vuoi installare nulla, puoi caricare manualmente:

### Passo 1: Crea Repository su GitHub

1. Vai su: **https://github.com/new**
2. **Repository name**: `Software-AsianOdds`
3. Scegli: **Public** o **Private**
4. **NON** selezionare "Add a README file"
5. **Clicca "Create repository"**

### Passo 2: Upload File

1. Nella pagina del repository, clicca **"uploading an existing file"**
2. **Trascina** tutta la cartella `Software-AsianOdds-main` nella pagina
3. Oppure clicca **"choose your files"** e seleziona tutti i file
4. Scorri in basso
5. **Commit message**: `Initial commit - Ready for Render deployment`
6. Clicca **"Commit changes"**

**⚠️ NOTA**: Questo metodo è più lento e non mantiene la cronologia Git, ma funziona!

---

## ✅ Verifica Step 1 Completato

Dopo aver completato una delle opzioni sopra, verifica:

1. Vai su: **https://github.com/TUO-USERNAME/Software-AsianOdds**
2. Dovresti vedere tutti i file:
   - ✅ `Dockerfile.automation`
   - ✅ `start_automation.py`
   - ✅ `automation_24h.py`
   - ✅ `requirements.automation.txt`
   - ✅ Cartella `ai_system/`

**Se vedi tutto, Step 1 è completato!** ✅

---

## 🎯 Quale Opzione Scegliere?

| Opzione | Difficoltà | Tempo | Consigliato per |
|---------|------------|-------|-----------------|
| **Git (CLI)** | Media | 10 min | Chi vuole imparare Git |
| **GitHub Desktop** | Facile | 5 min | Chi preferisce interfaccia grafica ⭐ |
| **Upload Manuale** | Facilissima | 15 min | Chi non vuole installare nulla |

**Raccomandazione**: **GitHub Desktop** - È il più facile e veloce!

---

## 🆘 Problemi Comuni

### "Git non è riconosciuto dopo installazione"
→ **Riavvia** PowerShell/Terminal completamente

### "Permission denied" durante push
→ Configura autenticazione GitHub:
   - Vai su: GitHub → Settings → Developer settings → Personal access tokens
   - Crea nuovo token con permessi `repo`
   - Usa token come password durante push

### "Repository already exists"
→ Usa l'URL del repository esistente invece di crearne uno nuovo

---

## 📝 Prossimi Passi

Una volta completato Step 1:

1. ✅ **Step 1**: Codice su GitHub ← **COMPLETATO**
2. ⏭️ **Step 2**: Crea Account Render.com
3. ⏭️ **Step 3**: Crea Background Worker
4. ⏭️ **Step 4**: Configura variabili ambiente
5. ⏭️ **Step 5**: Deploy!

**Vedi**: `GUIDA_RENDER_PASSO_PASSO.md` per Step 2-5

---

## 💡 Tips

- **GitHub Desktop** è la scelta più semplice per iniziare
- Puoi sempre passare a Git CLI dopo se vuoi
- Il repository può essere **pubblico** o **privato** (Render supporta entrambi)
- Se il repository è privato, Render richiede di connettere l'account GitHub

**Buon setup! 🚀**

