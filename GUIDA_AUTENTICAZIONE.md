# 🔐 Guida Autenticazione GitHub CLI

## Metodo 1: Autenticazione Interattiva (Consigliato)

### Passo 1: Apri PowerShell nella cartella del progetto
```powershell
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"
```

### Passo 2: Esegui il comando di autenticazione
```powershell
gh auth login
```

### Passo 3: Segui le istruzioni a schermo

1. **"Where do you use GitHub?"**
   - Seleziona: **GitHub.com** (premi INVIO)

2. **"What is your preferred protocol for Git operations?"**
   - Seleziona: **HTTPS** (premi INVIO)

3. **"Authenticate Git with your GitHub credentials?"**
   - Seleziona: **Yes** (premi INVIO)

4. **"How would you like to authenticate GitHub CLI?"**
   - Seleziona: **Login with a web browser** (premi INVIO)

5. **Ti verrà mostrato un codice** (es: `ABCD-1234`)
   - Premi **INVIO** per aprire il browser automaticamente
   - Oppure copia il codice e vai su: https://github.com/login/device

6. **Nel browser:**
   - Inserisci il codice mostrato
   - Autorizza GitHub CLI
   - Completa l'autenticazione

7. **Torna a PowerShell** - Dovresti vedere "✓ Authentication complete"

---

## Metodo 2: Autenticazione con Token (Alternativo)

### Passo 1: Crea un Personal Access Token
1. Vai su: https://github.com/settings/tokens
2. Clicca "Generate new token (classic)"
3. Dai un nome al token (es: "GitHub CLI")
4. Seleziona permessi:
   - ✅ `repo` (tutti)
   - ✅ `workflow`
5. Clicca "Generate token"
6. **COPIA IL TOKEN** (lo vedrai solo una volta!)

### Passo 2: Autentica con il token
```powershell
gh auth login --with-token
```

### Passo 3: Incolla il token
- Incolla il token copiato
- Premi INVIO
- Premi Ctrl+Z e INVIO per terminare

---

## Verifica Autenticazione

Dopo l'autenticazione, verifica che funzioni:

```powershell
gh auth status
```

Dovresti vedere:
```
✓ Logged in to github.com as [tuo-username]
✓ Git operations for github.com configured to use HTTPS
✓ Token: gho_xxxxxxxxxxxxx
```

---

## Test Completo

Prova a creare un PR di test:

```powershell
# Fai una piccola modifica a un file (es: aggiungi un commento)
# Poi:
.\quick_pr.ps1 -Message "Test: Verifica autenticazione GitHub"
```

---

## Troubleshooting

### "gh: command not found"
- Riavvia PowerShell dopo l'installazione
- Verifica: `gh --version`

### "Authentication failed"
- Riprova: `gh auth login`
- Oppure usa il metodo con token

### "Permission denied"
- Verifica che il token abbia permessi `repo`
- Rigenera il token se necessario

### "Repository not found"
- Verifica di avere accesso al repository
- Controlla: `git remote -v`

---

## Comandi Utili

```powershell
# Verifica stato autenticazione
gh auth status

# Logout (se necessario)
gh auth logout

# Login di nuovo
gh auth login

# Verifica repository
gh repo view Sib-asian/Software-AsianOdds
```

---

**Una volta autenticato, puoi creare PR automaticamente con `.\quick_pr.ps1`! 🎉**

