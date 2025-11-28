# 🚀 Market Movement Analyzer - Deploy su Streamlit Cloud

Guida completa per pubblicare l'app **Market Movement Analyzer** su Streamlit Cloud.

## 📋 Prerequisiti

1. ✅ Account GitHub (già configurato)
2. ✅ Repository: `https://github.com/Sib-asian/Software-AsianOdds.git`
3. ✅ Account Streamlit Cloud (gratuito): https://share.streamlit.io/

## 🔧 Step 1: Commit e Push del nuovo file

L'app è già nel repository locale. Per pubblicarla:

```bash
# Aggiungi il nuovo file
git add market_movement_analyzer_app.py
git add .streamlit/config.toml
git add MARKET_ANALYZER_DEPLOY.md

# Commit
git commit -m "Add Market Movement Analyzer Streamlit app"

# Push su GitHub
git push origin main
```

## 🌐 Step 2: Deploy su Streamlit Cloud

### 2.1 Accedi a Streamlit Cloud

1. Vai su: https://share.streamlit.io/
2. Fai login con il tuo account GitHub
3. Clicca su **"New app"**

### 2.2 Configura l'app

Compila il form:

- **Repository**: `Sib-asian/Software-AsianOdds`
- **Branch**: `main` (o il branch che preferisci)
- **Main file path**: `market_movement_analyzer_app.py`
- **App URL** (opzionale): scegli un nome personalizzato, es. `market-movement-analyzer`

### 2.3 Deploy

1. Clicca **"Deploy!"**
2. Streamlit Cloud installerà automaticamente le dipendenze da `requirements.txt`
3. L'app sarà disponibile in pochi minuti all'URL:
   ```
   https://market-movement-analyzer.streamlit.app
   ```
   (o il nome che hai scelto)

## ✅ Verifica

Una volta deployata, l'app sarà accessibile:
- ✅ Da qualsiasi dispositivo con browser
- ✅ Condivisibile con un semplice link
- ✅ Aggiornata automaticamente ad ogni push su GitHub

## 🔄 Aggiornamenti futuri

Per aggiornare l'app dopo modifiche:

```bash
git add market_movement_analyzer_app.py
git commit -m "Update Market Analyzer"
git push origin main
```

Streamlit Cloud aggiornerà automaticamente l'app in 1-2 minuti.

## 📝 Note

- L'app è **standalone** e non richiede configurazioni speciali
- Non servono variabili d'ambiente o secrets per questa app
- Tutte le dipendenze sono già in `requirements.txt`
- L'app funziona completamente offline (calcoli lato client)

## 🆘 Troubleshooting

### L'app non si avvia
- Verifica che `streamlit>=1.28.0` sia in `requirements.txt` ✅ (già presente)
- Controlla i log su Streamlit Cloud nella sezione "Manage app" → "Logs"

### Errori di import
- Verifica che tutte le dipendenze siano in `requirements.txt`
- L'app usa solo librerie standard Python + Streamlit (nessuna dipendenza esterna)

### Problemi con il deploy
- Assicurati che il file `market_movement_analyzer_app.py` sia nella root del repository
- Verifica che il branch su GitHub sia aggiornato con `git push`

## 🎉 Fatto!

Una volta deployata, condividi il link con chi vuoi! L'app è completamente funzionale e disponibile 24/7.

