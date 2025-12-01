# 📊 STATO SISTEMA 24/7 - LIVE BETTING ADVISOR

## ✅ CONFIGURAZIONE ATTUALE

- **Modalità**: 24/7 (continuo)
- **Update Interval**: 300 secondi (5 minuti) - configurabile
- **Min Confidence**: 72%
- **Min EV**: 8%
- **API-SPORTS**: Piano Pro (7500 chiamate/giorno)

## 🎯 COSA FA IL SISTEMA

1. **Cerca partite** ogni 5 minuti da:
   - API-SPORTS (primario)
   - TheOddsAPI (quote)
   - Football-Data.org (backup)

2. **Analizza partite live** con:
   - AI Pipeline avanzata
   - Live Betting Advisor
   - Filtri anti-ovvietà
   - Validazione segnali

3. **Invia segnali Telegram** quando trova:
   - Opportunità con confidence ≥ 72%
   - EV ≥ 8%
   - Segnali validati e non ovvi

## 🌙 PARTITE NOTTURNE

**SÌ, il sistema è già configurato per le partite notturne!**

Il sistema gira 24/7 e monitora:
- Partite europee (sera)
- Partite americane (notte)
- Partite asiatiche (mattina)
- Tutte le partite live disponibili

## 📱 COME VERIFICARE CHE FUNZIONA

1. **Controlla i log**: `logs/automation_service_*.log`
2. **Controlla Telegram**: Dovresti ricevere segnali quando trova opportunità
3. **Verifica processi**: `tasklist /FI "IMAGENAME eq python.exe"`

## 🚀 COME AVVIARE

```bash
# Metodo 1: Script Python robusto
python avvia_background_robusto.py

# Metodo 2: Batch file
AVVIA_24H.bat

# Metodo 3: Manuale (foreground per debug)
python automation_service_wrapper.py
```

## 🛑 COME FERMARE

```bash
# Batch file
FERMA_24H.bat

# Manuale
taskkill /F /FI "IMAGENAME eq python.exe"
```

## ⚙️ CONFIGURAZIONE

Le impostazioni sono in `automation_service_wrapper.py`:
- `AUTOMATION_UPDATE_INTERVAL`: Intervallo tra cicli (default: 300s)
- `AUTOMATION_MIN_CONFIDENCE`: Confidence minima (default: 72%)
- `AUTOMATION_MIN_EV`: EV minimo (default: 8%)

## 📊 MONITORAGGIO

- **Log in tempo reale**: `python verifica_log_tempo_reale.py`
- **Verifica segnali Telegram**: `python verifica_segnali_telegram.py`
- **Processi attivi**: `tasklist /FI "IMAGENAME eq python.exe"`

## ⚠️ NOTE IMPORTANTI

1. Il sistema è progettato per girare 24/7
2. Si riavvia automaticamente in caso di crash
3. I log vengono salvati giornalmente
4. Il sistema monitora automaticamente tutte le partite disponibili







