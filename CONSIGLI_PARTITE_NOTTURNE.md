# 🌙 CONSIGLI PER PARTITE NOTTURNE

## ✅ SÌ, PUOI ATTIVARE PYTHON H24 PER LE PARTITE NOTTURNE!

Il sistema è **già configurato** per funzionare 24/7, incluse le partite notturne.

## 🎯 COSA CONSIGLIO DI FARE ORA

### 1. **Verifica che il sistema sia attivo**
```bash
# Controlla se Python è in esecuzione
tasklist /FI "IMAGENAME eq python.exe"

# Controlla i log più recenti
python verifica_log_tempo_reale.py
```

### 2. **Se il sistema NON è attivo, riavvialo**
```bash
# Metodo più robusto
python avvia_background_robusto.py
```

### 3. **Lascialo attivo per la notte**
- Il sistema monitora automaticamente tutte le partite
- Invia segnali su Telegram quando trova opportunità
- Si riavvia automaticamente in caso di crash

## 🌍 COPERTURA PARTITE NOTTURNE

Il sistema monitora:
- **Europa (sera)**: 19:00 - 23:00
- **Americhe (notte)**: 00:00 - 06:00
- **Asia (mattina)**: 06:00 - 12:00
- **Tutte le partite live** disponibili su API-SPORTS

## 📊 INTERVALLI DI CONTROLLO

- **Default**: Ogni 5 minuti (300 secondi)
- **Configurabile**: Modifica `AUTOMATION_UPDATE_INTERVAL` in `.env` o `automation_service_wrapper.py`
- **Consigliato per notte**: 5-10 minuti (abbastanza frequente senza sprecare API calls)

## ⚡ OTTIMIZZAZIONI PER NOTTE

1. **Intervallo più lungo** (se vuoi risparmiare API calls):
   - Imposta `AUTOMATION_UPDATE_INTERVAL=600` (10 minuti)
   - Adatto se ci sono poche partite notturne

2. **Intervallo standard** (consigliato):
   - Mantieni `AUTOMATION_UPDATE_INTERVAL=300` (5 minuti)
   - Buon equilibrio tra copertura e uso API

3. **Intervallo più corto** (se ci sono molte partite):
   - Imposta `AUTOMATION_UPDATE_INTERVAL=180` (3 minuti)
   - Solo se hai molte partite notturne importanti

## 🔔 NOTIFICHE TELEGRAM

Il sistema invia automaticamente:
- ✅ Segnali quando trova opportunità
- ✅ Alert per partite importanti
- ✅ Report giornalieri (se configurato)

**Assicurati che Telegram sia attivo** per ricevere i segnali notturni!

## 🛡️ SICUREZZA

- ✅ Auto-restart in caso di crash
- ✅ Log dettagliati per debugging
- ✅ Gestione errori robusta
- ✅ Limiti API rispettati (7500/giorno)

## 📱 MONITORAGGIO NOTTURNO

Puoi monitorare il sistema anche di notte:
1. **Telegram**: Ricevi i segnali direttamente
2. **Log**: Controlla `logs/automation_service_*.log` al mattino
3. **Processi**: Verifica che Python sia attivo

## 🎯 CONCLUSIONE

**SÌ, lascia il sistema attivo 24/7!**

Il sistema è progettato per:
- ✅ Funzionare continuamente
- ✅ Monitorare tutte le partite (incluse notturne)
- ✅ Inviare segnali automaticamente
- ✅ Riavviarsi in caso di problemi

**Non serve fare nulla di speciale per le partite notturne** - il sistema le gestisce automaticamente!


